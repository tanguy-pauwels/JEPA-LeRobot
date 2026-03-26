from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("lewm")
@dataclass
class LeWMConfig(PreTrainedConfig):
    """Configuration for LeWM BC policy scaffold."""

    n_obs_steps: int = 3
    n_action_steps: int = 1

    image_size: int = 224
    patch_size: int = 14
    encoder_scale: str = "tiny"
    embed_dim: int = 192
    action_head_hidden_dim: int = 512
    debug_metrics: bool = False
    preferred_image_key: str | None = "observation.images.laptop"
    use_world_model_loss: bool = True
    world_model_predictor: str = "ar"
    loss_weight_action: float = 1.0
    loss_weight_dynamics: float = 1.0
    loss_weight_sigreg: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}")
        if self.n_action_steps != 1:
            raise ValueError(f"n_action_steps must be 1 for this scaffold, got {self.n_action_steps}")
        if self.encoder_scale not in {"tiny", "small", "base"}:
            raise ValueError(
                f"Unsupported encoder_scale '{self.encoder_scale}'. Supported: tiny, small, base"
            )
        if self.use_world_model_loss and self.n_obs_steps < 2:
            raise ValueError("n_obs_steps must be >= 2 when use_world_model_loss=True")
        if self.world_model_predictor != "ar":
            raise ValueError(
                f"Unsupported world_model_predictor '{self.world_model_predictor}'. Supported: ar"
            )
        if self.loss_weight_action < 0 or self.loss_weight_dynamics < 0 or self.loss_weight_sigreg < 0:
            raise ValueError("Loss weights must be >= 0")
        if self.sigreg_knots < 2:
            raise ValueError(f"sigreg_knots must be >= 2, got {self.sigreg_knots}")
        if self.sigreg_num_proj < 1:
            raise ValueError(f"sigreg_num_proj must be >= 1, got {self.sigreg_num_proj}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        image_keys = list(self.image_features.keys())
        if len(image_keys) == 0:
            raise ValueError("LeWM policy requires at least one visual feature.")
        if len(image_keys) > 1:
            if self.preferred_image_key is None:
                raise ValueError(
                    "LeWM policy received multiple visual features. "
                    "Set policy.preferred_image_key to one of: "
                    f"{image_keys}"
                )
            if self.preferred_image_key not in image_keys:
                raise ValueError(
                    "LeWM policy preferred_image_key was not found in visual features. "
                    f"preferred_image_key='{self.preferred_image_key}', available={image_keys}"
                )
        if self.action_feature is None:
            raise ValueError("LeWM policy requires one ACTION output feature named 'action'.")

    @property
    def selected_image_key(self) -> str:
        image_keys = list(self.image_features.keys())
        if len(image_keys) == 1:
            return image_keys[0]
        if self.preferred_image_key is None:
            raise ValueError(
                "Multiple visual features available but preferred_image_key is None. "
                f"Available keys: {image_keys}"
            )
        if self.preferred_image_key not in image_keys:
            raise ValueError(
                f"preferred_image_key='{self.preferred_image_key}' not found. "
                f"Available keys: {image_keys}"
            )
        return self.preferred_image_key

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        if self.use_world_model_loss:
            return list(range(1 - self.n_obs_steps, 1))
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
