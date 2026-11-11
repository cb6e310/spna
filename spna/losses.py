"""SPNA-CARG objectives and relation-control targets."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

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
        relations = mask.to(frozen_text.dtype) / (batch - 1)
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


def distribution_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("logits and target must have matching [batch, candidates] shapes")
    return F.kl_div(F.log_softmax(logits.float(), dim=-1), target.float(), reduction="batchmean")


def negative_relation_loss(
    small_logits: torch.Tensor,
    relations: torch.Tensor,
) -> torch.Tensor:
    """KL(Q_neg || P_neg) with each row normalized over non-paired candidates only."""
    if small_logits.shape != relations.shape or small_logits.ndim != 2:
        raise ValueError("small_logits and relations must have matching [batch, batch] shapes")
    if small_logits.shape[0] != small_logits.shape[1] or small_logits.shape[0] < 2:
        raise ValueError("Relation logits must be square with at least two candidates")
    mask = ~torch.eye(small_logits.shape[0], dtype=torch.bool, device=small_logits.device)
    target = relations.detach().float().masked_fill(~mask, 0.0)
    row_sums = target.sum(dim=-1, keepdim=True)
    if not torch.isfinite(target).all() or (target < 0).any() or (row_sums <= 0).any():
        raise ValueError("Teacher relation rows must have positive finite non-paired mass")
    target = target / row_sums
    log_probs = F.log_softmax(
        small_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min), dim=-1
    )
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return F.kl_div(log_probs, target, reduction="batchmean")


def hard_pair_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Pair logits must be a square [batch, batch] matrix")
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits.float(), labels)


def weighted_capacity_loss(
    logits_by_capacity: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine hard-pair losses for nested embedding prefixes."""
    if not logits_by_capacity or len(logits_by_capacity) != len(weights):
        raise ValueError("Each capacity logit matrix must have one weight")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Capacity weights must be nonnegative and not all zero")
    losses = [hard_pair_loss(logits) for logits in logits_by_capacity]
    total = sum(float(weight) * loss for weight, loss in zip(weights, losses))
    return total, {str(index): loss for index, loss in enumerate(losses)}


def gradient_conflict_diagnostics(
    losses_by_capacity: dict[str, torch.Tensor],
    shared_parameters: Sequence[torch.nn.Parameter],
) -> dict[str, float]:
    """Measure pairwise gradient cosine and norms on shared model parameters."""
    names = list(losses_by_capacity)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for name, loss in losses_by_capacity.items():
        gradients[name] = torch.autograd.grad(
            loss, shared_parameters, retain_graph=True, allow_unused=True
        )

    result: dict[str, float] = {}
    flattened: dict[str, torch.Tensor] = {}
    for name, grads in gradients.items():
        pieces = [grad.detach().float().reshape(-1) for grad in grads if grad is not None]
        flattened[name] = torch.cat(pieces) if pieces else torch.zeros(1)
        result[f"norm_{name}"] = float(flattened[name].norm().cpu())
    for left, right in combinations(names, 2):
        a, b = flattened[left], flattened[right]
        denominator = a.norm() * b.norm()
        cosine = torch.dot(a, b) / denominator.clamp_min(1e-12)
        result[f"cos_{left}_{right}"] = float(cosine.cpu())
    return result


def spna_objective(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    relations: torch.Tensor | None,
    lambda_relation: float = 0.0,
    alpha_small: float = 0.5,
    alpha_large: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if alpha_small < 0 or alpha_large < 0 or alpha_small + alpha_large <= 0:
        raise ValueError("Capacity weights must be nonnegative and not both zero")
    if lambda_relation < 0:
        raise ValueError("lambda_relation must be nonnegative")
    small_loss = hard_pair_loss(small_logits)
    large_loss = hard_pair_loss(large_logits)
    if lambda_relation > 0 and relations is None:
        raise ValueError("relations are required when lambda_relation is positive")
    relation_loss = (negative_relation_loss(small_logits, relations)
                     if lambda_relation > 0 else small_logits.sum() * 0.0)
    total = (alpha_small * small_loss + alpha_large * large_loss
             + lambda_relation * relation_loss)
    return total, {"small": small_loss.detach(), "large": large_loss.detach(),
                   "relation": relation_loss.detach()}
