"""Torch mirror of laya_mlx.model.py, shaped for the Apple Neural Engine.

The exported graph contains only ANE-eligible ops (linear, matmul, layer_norm,
softmax, gelu, add, mul, slice, reshape, transpose, concat). Everything the ANE
compiler rejects is moved to the host side and fed in as data:

* token embeddings      -> looked up on the host (no ``gather``)
* RoPE cos/sin tables   -> precomputed on the host (no ``cumsum``/``cos``/``sin``)
* attention masks       -> additive fp16 biases (no ``select``/``fill_like``)
* type embedding        -> one-hot @ table (no ``gather``)
* marker extraction     -> one-hot @ hidden (no ``gather_along_axis``)
* entropy/top-2/act_head-> host post-processing (no ``log``/``argsort``)

The weights are the reference checkpoint's, unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

MASK_BIAS = -1e4


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    local_attention: int = 128
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0

    @classmethod
    def load(cls, path: Path) -> "Config":
        value = json.loads(path.read_text())
        names = {field for field in cls.__dataclass_fields__}
        cfg = cls(**{key: value[key] for key in value.keys() & names})

        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]

        return cfg

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind: str) -> float:
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


# ---------------------------------------------------------------------------
# Host-side helpers (mirrored in Swift). Kept here so verify() and the Swift
# runtime derive their inputs from one definition.
# ---------------------------------------------------------------------------

def rope_table(length: int, base: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [length, dim // 2] for non-traditional RoPE."""
    half = dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.arange(length, dtype=torch.float32)[:, None] * inv[None, :]

    return torch.cos(angles), torch.sin(angles)


def attention_biases(valid: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive biases: global [B,1,1,L] over keys, sliding [B,1,L,L].

    Padded queries may attend to all valid keys so their softmax rows are never
    empty; they are never read as keys or outputs, matching the reference.
    """
    keys = valid[:, None, None, :]
    pos = torch.arange(valid.shape[1])
    local = (pos[:, None] - pos[None, :]).abs() <= window // 2
    sliding = torch.logical_and(torch.logical_or(local[None, None], torch.logical_not(valid[:, None, :, None])), keys)

    to_bias = lambda mask: torch.where(mask, torch.zeros((), dtype=torch.float32), torch.full((), MASK_BIAS))
    return to_bias(keys), to_bias(sliding)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, half: int) -> torch.Tensor:
    """Non-traditional RoPE (pairs x[i] with x[i+half]); ``half`` must be a static int for the tracer."""
    first, second = x[..., :half], x[..., half:]

    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x, bias, cos, sin):
        qkv = self.qkv(x).reshape(1, -1, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        q, k = apply_rope(q, cos, sin, self.dim // 2), apply_rope(k, cos, sin, self.dim // 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.dim ** -0.5) + bias
        weights = torch.softmax(scores, dim=-1)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(1, -1, self.heads * self.dim)

        return self.out(result)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config, index: int):
        super().__init__()
        self.kind = cfg.layer_types[index]
        self.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x, bias, cos, sin):
        x = x + self.attn(self.attn_norm(x), bias, cos, sin)
        value, gate = self.wi(self.mlp_norm(x)).chunk(2, dim=-1)

        return x + self.wo(torch.nn.functional.gelu(value) * gate)


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden_size)  # host-side lookup; not traced
        self.emb_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.layers = nn.ModuleList([EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.half = cfg.head_dim // 2

    def forward(self, embeds, global_bias, sliding_bias, rope):
        """rope: [1, 2, L, head_dim] — index 0 global base, 1 local base; (cos | sin) along the last axis."""
        x = self.emb_norm(embeds)
        h = self.half
        tables = {
            "full_attention": (global_bias, rope[:, 0:1, :, :h], rope[:, 0:1, :, h:2 * h]),
            "sliding_attention": (sliding_bias, rope[:, 1:2, :, :h], rope[:, 1:2, :, h:2 * h]),
        }

        for layer in self.layers:
            x = layer(x, *tables[layer.kind])

        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.heads = max(1, dims // 64)
        self.dim = dims // self.heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x, bias):
        qkv = self.in_proj(x).reshape(1, -1, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.dim ** -0.5) + bias
        out = torch.matmul(torch.softmax(scores, -1), v)

        return self.out_proj(out.transpose(1, 2).reshape(1, -1, self.heads * self.dim))


class HeadLayer(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.self_attn = HeadAttention(dims)
        self.norm1, self.norm2 = nn.LayerNorm(dims), nn.LayerNorm(dims)
        self.linear1, self.linear2 = nn.Linear(dims, 4 * dims), nn.Linear(4 * dims, dims)

    def forward(self, x, bias):
        x = x + self.self_attn(self.norm1(x), bias)

        return x + self.linear2(torch.relu(self.linear1(self.norm2(x))))


class LayaModel(nn.Module):
    def __init__(self, cfg: Config, agent: dict):
        super().__init__()
        dims = cfg.hidden_size
        self.encoder = Encoder(cfg)
        self.head = nn.Module()
        self.head.layers = nn.ModuleList([HeadLayer(dims) for _ in range(agent.get("head_layers", 2))])
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1))
        # act_head runs on the host from exported weights; kept so load_state_dict is strict.
        self.act_head = nn.Sequential(nn.Linear(dims + 4, 256), nn.GELU(), nn.Linear(256, len(agent.get("act_costs", {})) + 1))

    def forward(self, embeds, global_bias, sliding_bias, rope, type_onehot, marker_onehot, marker_mask):
        h = self.encoder(embeds, global_bias, sliding_bias, rope)
        h = h + torch.matmul(type_onehot, self.type_emb.weight)[:, None, :]

        for layer in self.head.layers:
            h = layer(h, global_bias)

        markers = torch.matmul(marker_onehot, h)
        logits = self.scorer(markers).squeeze(-1)
        logits = logits * marker_mask + (marker_mask - 1.0) * (-MASK_BIAS)

        return logits, h[:, 0]


def load_model(root: Path) -> LayaModel:
    cfg = Config.load(root / "encoder/config.json")
    model = LayaModel(cfg, json.loads((root / "rl_agent_config.json").read_text())).eval()

    source = load_file(str(root / "model.safetensors"), device="cpu")
    weights = {}
    for name, value in source.items():
        if name == "temperature":
            continue

        name = name.replace(".Wqkv.", ".qkv.")
        name = name.replace(".mlp.Wi.", ".wi.").replace(".mlp.Wo.", ".wo.")
        name = name.replace(".Wo.", ".out.").replace(".mlp.out.", ".wo.")
        name = name.replace("encoder.embeddings.tok_embeddings", "encoder.tok")
        name = name.replace("encoder.embeddings.norm", "encoder.emb_norm")
        name = name.replace("scorer.layers.", "scorer.").replace("act_head.layers.", "act_head.")
        weights[name] = value

    model.load_state_dict(weights, strict=True)

    return model
