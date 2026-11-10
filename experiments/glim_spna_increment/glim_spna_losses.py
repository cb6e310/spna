"""Experimental nested CARG objective for the GLIM-SPNA increment."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from spna.losses import hard_pair_loss, negative_relation_loss, text_relation_target

def nested_carg_objective(
    eeg_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    frozen_teacher: torch.Tensor,
    small_dim: int = 64,
    large_dim: int = 256,
    lambda_relation: float = 0.1,
    relation_temperature: float = 0.1,
    logit_temperature: float = 0.07,
    alpha_small: float = 0.5,
    alpha_large: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply SPNA's nested CARG objective to GLIM's aligned embeddings.

    GLIM can keep its native encoder, full-width contrastive loss and generation
    loss while this auxiliary objective supervises nested low/high-capacity
    prefixes of its shared EEG/text embedding space.
    """
    if eeg_vectors.ndim != 2 or text_vectors.ndim != 2 or frozen_teacher.ndim != 2:
        raise ValueError("All embedding inputs must have shape [batch, dimension]")
    if eeg_vectors.shape[0] != text_vectors.shape[0] or eeg_vectors.shape[0] != frozen_teacher.shape[0]:
        raise ValueError("EEG, text, and teacher batch sizes must match")
    if eeg_vectors.shape[0] < 3:
        raise ValueError("Nested CARG requires at least three unique stimuli")
    if not 0 < small_dim <= large_dim <= min(eeg_vectors.shape[1], text_vectors.shape[1]):
        raise ValueError("Require 0 < small_dim <= large_dim <= student embedding dimensions")
    if logit_temperature <= 0:
        raise ValueError("logit_temperature must be positive")
    if lambda_relation < 0:
        raise ValueError("lambda_relation must be nonnegative")
    if alpha_small < 0 or alpha_large < 0 or alpha_small + alpha_large <= 0:
        raise ValueError("Capacity weights must be nonnegative and not both zero")

    relations = text_relation_target(frozen_teacher, relation_temperature)
    small_logits = (
        F.normalize(eeg_vectors[:, :small_dim].float(), dim=-1)
        @ F.normalize(text_vectors[:, :small_dim].float(), dim=-1).T
    ) / logit_temperature
    large_logits = (
        F.normalize(eeg_vectors[:, :large_dim].float(), dim=-1)
        @ F.normalize(text_vectors[:, :large_dim].float(), dim=-1).T
    ) / logit_temperature
    small_loss = hard_pair_loss(small_logits)
    large_loss = hard_pair_loss(large_logits)
    relation_loss = negative_relation_loss(small_logits, relations)
    total = alpha_small * small_loss + alpha_large * large_loss + lambda_relation * relation_loss
    return total, {"small": small_loss.detach(), "large": large_loss.detach(),
                   "relation": relation_loss.detach()}
