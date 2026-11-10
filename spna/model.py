"""Shared nested EEG/text representations for SPNA-CARG."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    channels: int = 104
    max_samples: int = 1280
    patch_size: int = 16
    hidden_dim: int = 256
    encoder_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.1
    text_dim: int = 1024
    projection_hidden_dim: int = 512
    embedding_dim: int = 256


class EEGEncoder(nn.Module):
    """Patch-transformer EEG encoder; inputs use [batch, time, channels]."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.config = config
        self.patch = nn.Conv1d(
            config.channels,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        patch_count = (config.max_samples + config.patch_size - 1) // config.patch_size
        self.position = nn.Parameter(torch.empty(1, patch_count, config.hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.encoder_layers, norm=nn.LayerNorm(config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool = nn.MultiheadAttention(
            config.hidden_dim, config.attention_heads, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, eeg: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if eeg.ndim != 3 or eeg.shape[-1] != cfg.channels:
            raise ValueError("EEG must have shape [batch, time, channels]")
        if valid_mask.shape != eeg.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, time]")
        if eeg.shape[1] > cfg.max_samples:
            raise ValueError("EEG time length exceeds configured max_samples")
        valid_mask = valid_mask.bool()
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every EEG trial must contain at least one valid sample")

        eeg = eeg.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        pad = (-eeg.shape[1]) % cfg.patch_size
        channels_first = F.pad(eeg.transpose(1, 2), (0, pad))
        padded_mask = F.pad(valid_mask, (0, pad), value=False)
        patch_mask = padded_mask.unfold(1, cfg.patch_size, cfg.patch_size).any(-1)
        tokens = self.patch(channels_first).transpose(1, 2)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.transformer(tokens, src_key_padding_mask=~patch_mask)
        query = self.pool_query.expand(eeg.shape[0], -1, -1)
        pooled, _ = self.pool(
            query, tokens, tokens, key_padding_mask=~patch_mask, need_weights=False
        )
        return self.pool_norm(pooled[:, 0])


def projection(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SPNA(nn.Module):
    """One shared D-dimensional EEG/text space with normalized prefixes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.eeg_encoder = EEGEncoder(config)
        self.eeg_projection = projection(
            config.hidden_dim, config.projection_hidden_dim, config.embedding_dim
        )
        self.text_projection = projection(
            config.text_dim, config.projection_hidden_dim, config.embedding_dim
        )

    def encode_eeg(self, eeg: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.eeg_projection(self.eeg_encoder(eeg, valid_mask)).float()

    def encode_text(self, frozen_text_features: torch.Tensor) -> torch.Tensor:
        if frozen_text_features.ndim != 2:
            raise ValueError("Frozen text features must have shape [batch, text_dim]")
        if frozen_text_features.shape[-1] != self.config.text_dim:
            raise ValueError("Text feature dimension does not match ModelConfig.text_dim")
        return self.text_projection(frozen_text_features.detach().float()).float()

    @staticmethod
    def prefix(vectors: torch.Tensor, dimension: int) -> torch.Tensor:
        if dimension < 1 or dimension > vectors.shape[-1]:
            raise ValueError(f"Invalid prefix dimension {dimension}")
        return F.normalize(vectors[..., :dimension].float(), dim=-1)

    def score(
        self,
        eeg_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        dimension: int,
        temperature: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        eeg = self.prefix(eeg_vectors, dimension)
        text = self.prefix(text_vectors, dimension)
        return (eeg @ text.transpose(0, 1)) / temperature

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

