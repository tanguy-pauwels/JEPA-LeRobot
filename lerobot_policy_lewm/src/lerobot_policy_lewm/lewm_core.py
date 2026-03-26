from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import ViTConfig, ViTModel


class MLP(nn.Module):
    """Simple 2-layer MLP used for latent projection and action head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """proj: (T, B, D)."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv]
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.ada[-1].weight, 0)
        nn.init.constant_(self.ada[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    """Action embedder: (B, T, A) -> (B, T, D)."""

    def __init__(self, input_dim: int, emb_dim: int, smoothed_dim: int | None = None, mlp_scale: int = 4):
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x: (B,T,D), c: (B,T,D)."""
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)


def make_vit_config(encoder_scale: str, image_size: int, patch_size: int) -> ViTConfig:
    """Create a ViT config compatible with LeWM-style tiny/small/base settings."""

    scale_map = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    }
    if encoder_scale not in scale_map:
        raise ValueError(
            f"Unsupported encoder_scale '{encoder_scale}'. Supported values: {list(scale_map)}"
        )

    cfg = scale_map[encoder_scale]
    return ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=4 * cfg["hidden_size"],
    )


class LeWMVisionEncoder(nn.Module):
    """ViT encoder producing sequence embeddings shaped (B, T, D)."""

    def __init__(self, encoder_scale: str, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        vit_cfg = make_vit_config(
            encoder_scale=encoder_scale,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.vit = ViTModel(vit_cfg)
        self.projector = MLP(vit_cfg.hidden_size, hidden_dim=1024, output_dim=embed_dim)

    def forward(self, pixels: Tensor) -> Tensor:
        """Encode pixels (B, T, C, H, W) into embeddings (B, T, D)."""

        bsz, time_steps = pixels.shape[:2]
        flat = pixels.reshape(bsz * time_steps, *pixels.shape[2:])
        output = self.vit(pixel_values=flat)
        cls = output.last_hidden_state[:, 0]
        emb = self.projector(cls)
        return emb.reshape(bsz, time_steps, -1)
