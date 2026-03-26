from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_lewm import LeWMConfig
from .lewm_core import ARPredictor, Embedder, LeWMVisionEncoder, MLP, SIGReg


class LeWMPolicy(PreTrainedPolicy):
    """LeWM BC policy scaffold: image history -> latent -> one-step action."""

    config_class = LeWMConfig
    name = "lewm"

    def __init__(self, config: LeWMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        action_dim = self.config.action_feature.shape[0]
        self.encoder = LeWMVisionEncoder(
            encoder_scale=self.config.encoder_scale,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
        )
        self.action_head = MLP(
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.action_head_hidden_dim,
            output_dim=action_dim,
        )
        self.action_encoder = Embedder(input_dim=action_dim, emb_dim=self.config.embed_dim)
        self.predictor = ARPredictor(
            num_frames=self.config.n_obs_steps,
            depth=4,
            heads=4,
            mlp_dim=4 * self.config.embed_dim,
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.embed_dim,
            output_dim=self.config.embed_dim,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        )
        self.sigreg = SIGReg(
            knots=self.config.sigreg_knots,
            num_proj=self.config.sigreg_num_proj,
        )

        self._image_key = self.config.selected_image_key
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=1)
        self._obs_queue = deque([], maxlen=self.config.n_obs_steps)

    def _prepare_pixels(self, image: Tensor) -> Tensor:
        """Convert image tensor to float (B, T, C, H, W), resized to config image_size."""

        if image.ndim == 4:
            image = image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(f"Expected image tensor with 4 or 5 dims, got shape={tuple(image.shape)}")

        # If channels-last, convert to channels-first.
        if image.shape[-1] in (1, 3) and image.shape[-2] > 4:
            image = image.permute(0, 1, 4, 2, 3)

        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0

        bsz, t, ch, h, w = image.shape
        if t not in {1, self.config.n_obs_steps}:
            raise ValueError(
                f"Unexpected temporal dimension T={t}. Expected T=1 or T={self.config.n_obs_steps}."
            )
        if (h, w) != (self.config.image_size, self.config.image_size):
            flat = image.reshape(bsz * t, ch, h, w)
            flat = F.interpolate(
                flat,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
            image = flat.reshape(bsz, t, ch, self.config.image_size, self.config.image_size)

        return image

    def _stack_temporal_from_queue(self) -> Tensor:
        if len(self._obs_queue) == 0:
            raise RuntimeError("Observation queue is empty in select_action().")
        if len(self._obs_queue) < self.config.n_obs_steps:
            first = self._obs_queue[0]
            while len(self._obs_queue) < self.config.n_obs_steps:
                self._obs_queue.appendleft(first)
        return torch.stack(list(self._obs_queue), dim=1)

    def _predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self._image_key not in batch:
            raise KeyError(f"Expected visual key '{self._image_key}' in batch")

        pixels = self._prepare_pixels(batch[self._image_key])
        emb = self.encoder(pixels)
        last_emb = emb[:, -1]
        action = self.action_head(last_emb)
        return action.unsqueeze(1)  # (B, 1, action_dim)

    def _get_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise KeyError("Expected 'action' key in training batch")
        action_tgt = batch[ACTION]
        if action_tgt.ndim == 2:
            return action_tgt.unsqueeze(1)
        if action_tgt.ndim == 3:
            return action_tgt
        raise ValueError(f"Expected action target with shape (B,A) or (B,T,A), got {tuple(action_tgt.shape)}")

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self._image_key not in batch:
            raise KeyError(f"Expected visual key '{self._image_key}' in batch")
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction='{reduction}'. Use 'mean' or 'none'.")

        pixels = self._prepare_pixels(batch[self._image_key])
        z_seq = self.encoder(pixels)
        action_seq = self._get_action_targets(batch)
        last_tgt = action_seq[:, -1]

        action_pred = self.action_head(z_seq[:, -1])
        action_mse_per_sample = (action_pred - last_tgt).pow(2).mean(dim=-1)
        action_mae_per_sample = (action_pred - last_tgt).abs().mean(dim=-1)

        if self.config.use_world_model_loss:
            if action_seq.shape[1] != z_seq.shape[1]:
                raise ValueError(
                    "World-model training expects action sequence aligned with image history: "
                    f"got action T={action_seq.shape[1]} and image T={z_seq.shape[1]}"
                )
            if z_seq.shape[1] < 2:
                raise ValueError("World-model training requires temporal horizon T >= 2.")
            act_emb = self.action_encoder(action_seq)
            pred_seq = self.predictor(z_seq, act_emb)
            pred_next = pred_seq[:, :-1]
            tgt_next = z_seq[:, 1:].detach()
            dyn_mse_per_sample = (pred_next - tgt_next).pow(2).mean(dim=(1, 2))
            sigreg_loss = self.sigreg(z_seq.transpose(0, 1))
        else:
            dyn_mse_per_sample = torch.zeros_like(action_mse_per_sample)
            sigreg_loss = torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)

        total_per_sample = (
            self.config.loss_weight_action * action_mse_per_sample
            + self.config.loss_weight_dynamics * dyn_mse_per_sample
            + self.config.loss_weight_sigreg * sigreg_loss
        )
        loss = total_per_sample if reduction == "none" else total_per_sample.mean()

        log_dict: dict[str, float] = {
            "loss_total": total_per_sample.mean().item(),
            "loss_action": action_mse_per_sample.mean().item(),
            "loss_dynamics": dyn_mse_per_sample.mean().item(),
            "loss_sigreg": sigreg_loss.item(),
            "mse_loss": action_mse_per_sample.mean().item(),
            "mae_loss": action_mae_per_sample.mean().item(),
        }
        if self.config.debug_metrics:
            pred_flat = action_pred.detach().reshape(-1)
            tgt_flat = last_tgt.detach().reshape(-1)
            l2_per_sample = (action_pred - last_tgt).pow(2).sum(dim=-1).sqrt()
            log_dict.update(
                {
                    "pred_mean": pred_flat.mean().item(),
                    "pred_std": pred_flat.std().item(),
                    "target_mean": tgt_flat.mean().item(),
                    "target_std": tgt_flat.std().item(),
                    "pred_target_l2": l2_per_sample.mean().item(),
                }
            )

        return loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self._predict_action(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        if self._image_key not in batch:
            raise KeyError(f"Expected visual key '{self._image_key}' in batch")

        image = batch[self._image_key]
        if image.ndim == 5 and image.shape[1] > 1:
            infer_batch = batch
        else:
            if image.ndim == 5:
                if image.shape[1] != 1:
                    raise ValueError(f"Expected image with T=1 in inference queue mode, got shape={tuple(image.shape)}")
                image = image[:, 0]
            if image.ndim != 4:
                raise ValueError(
                    "Expected image tensor with shape (B,C,H,W), (B,H,W,C), or (B,1,C,H,W) in select_action; "
                    f"got {tuple(image.shape)}"
                )
            self._obs_queue.append(image)
            stacked = self._stack_temporal_from_queue()
            infer_batch = dict(batch)
            infer_batch[self._image_key] = stacked

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(infer_batch)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()
