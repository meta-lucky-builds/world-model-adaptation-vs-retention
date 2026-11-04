"""LeWorldModel architecture and checkpoint loading.

The classes from ``modulate`` through ``ARPredictor`` are copied unchanged from
lucas-maes/le-wm (MIT License), commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac,
file ``module.py``. ``LeWorldModel.predict`` is copied from ``jepa.py`` in the
same commit. They are vendored so the published ``weights.pt`` stays loadable
regardless of later upstream refactors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import ViTConfig, ViTModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class LeWorldModel(nn.Module):
    """The modules and state-dict layout of upstream ``JEPA``."""

    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map preprocessed images (N, 3, H, W) to latents (N, D)."""
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        return self.projector(output.last_hidden_state[:, 0])

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Map uint8 RGB frames (N, H, W, 3) to latents (N, D)."""
        device = next(self.parameters()).device
        return self.encode_pixels(preprocess_frames(frames, device))


def preprocess_frames(frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """Apply upstream's image transform to uint8 RGB frames (N, H, W, 3).

    Upstream scales to [0, 1], applies ImageNet normalization, then resizes to
    224. Every Two-Room frame here is already 224x224, so the resize is an
    identity and any other size is rejected rather than silently resampled.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 frames (N, H, W, 3), got {frames.dtype} {frames.shape}"
        )
    if frames.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected {IMAGE_SIZE}x{IMAGE_SIZE} frames, got {frames.shape}")
    pixels = torch.from_numpy(np.array(frames)).to(device)
    pixels = pixels.permute(0, 3, 1, 2).float() / 255.0
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _without_target(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def build_model(cfg: dict) -> LeWorldModel:
    """Build the architecture described by a Hugging Face ``config.json``.

    The published config carries Hydra ``_target_`` keys, so they are stripped
    before the arguments reach the constructors.
    """
    enc = cfg["encoder"]
    if enc["size"] != "tiny":
        raise ValueError(f"unsupported encoder size {enc['size']!r}")
    # Same arguments as stable_pretraining.backbone.utils.vit_hf("tiny", ...).
    encoder = ViTModel(
        ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=enc["image_size"],
            patch_size=enc["patch_size"],
        ),
        add_pooling_layer=False,
        use_mask_token=enc.get("use_mask_token", False),
    )

    def projection(key: str) -> MLP:
        sub = cfg[key]
        return MLP(
            input_dim=sub["input_dim"],
            hidden_dim=sub["hidden_dim"],
            output_dim=sub["output_dim"],
            norm_fn=nn.BatchNorm1d,
        )

    return LeWorldModel(
        encoder=encoder,
        predictor=ARPredictor(**_without_target(cfg["predictor"])),
        action_encoder=Embedder(**_without_target(cfg["action_encoder"])),
        projector=projection("projector"),
        pred_proj=projection("pred_proj"),
    )


def load_pretrained(
    weights_path: str | Path, config_path: str | Path, device: torch.device | str
) -> LeWorldModel:
    """Load a published checkpoint strictly and return it frozen in eval mode."""
    cfg = json.loads(Path(config_path).read_text())
    model = build_model(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model
