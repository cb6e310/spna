"""SPNA-CARG objectives and relation-control targets."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def text_relation_target(
    frozen_text: torch.Tensor,
    temperature: float,
    mode: str = "true",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return row-stochastic off-diagonal teacher relations for unique stimuli."""
    if frozen_text.ndim != 2:
        raise ValueError("frozen_text must have shape [batch, feature_dim]")
    batch = frozen_text.shape[0]
    if batch < 3:
        raise ValueError("CARG requires at least three unique stimuli per batch")
    if temperature <= 0:
        raise ValueError("relation temperature must be positive")
    if mode not in {"true", "uniform", "shuffled"}:
        raise ValueError(f"Unknown relation mode: {mode}")

    mask = ~torch.eye(batch, dtype=torch.bool, device=frozen_text.device)
    if mode == "uniform":
        relations = mask.to(frozen_text.dtype)
    else:
        teacher = F.normalize(frozen_text.detach().float(), dim=-1)
        logits = (teacher @ teacher.T) / temperature
        if mode == "shuffled":
            relations = torch.zeros_like(logits)
            for row in range(batch):
                candidates = torch.arange(batch, device=frozen_text.device)
                candidates = candidates[candidates != row]
                permutation = torch.randperm(len(candidates), generator=generator)
                relations[row, candidates] = logits[row, candidates[permutation]]
            relations = torch.softmax(relations.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
        else:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            relations = torch.softmax(logits, dim=-1)
    return relations.detach()


def mixed_target(relations: torch.Tensor, eta: float) -> torch.Tensor:
    if not 0 <= eta < 1:
        raise ValueError("eta must satisfy 0 <= eta < 1")
    batch = relations.shape[0]
    identity = torch.eye(batch, dtype=relations.dtype, device=relations.device)
    return ((1.0 - eta) * identity + eta * relations).detach()


def distribution_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("logits and target must have matching [batch, candidates] shapes")
    return F.kl_div(F.log_softmax(logits.float(), dim=-1), target.float(), reduction="batchmean")


def hard_pair_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Pair logits must be a square [batch, batch] matrix")
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits.float(), labels)


def spna_objective(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    relations: torch.Tensor,
    eta: float,
    alpha_small: float = 0.5,
    alpha_large: float = 0.5,
    small_mode: str = "carg",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if alpha_small < 0 or alpha_large < 0 or alpha_small + alpha_large <= 0:
        raise ValueError("Capacity weights must be nonnegative and not both zero")
    if small_mode == "carg":
        small_target = mixed_target(relations, eta)
        small_loss = distribution_loss(small_logits, small_target)
    elif small_mode == "hard":
        small_loss = hard_pair_loss(small_logits)
    elif small_mode == "disabled":
        small_loss = small_logits.sum() * 0.0
    else:
        raise ValueError(f"Unknown small_mode: {small_mode}")
    large_loss = hard_pair_loss(large_logits)
    total = alpha_small * small_loss + alpha_large * large_loss
    return total, {"small": small_loss.detach(), "large": large_loss.detach()}
