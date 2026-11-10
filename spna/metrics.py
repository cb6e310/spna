"""Full-bank EEG-to-text retrieval metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def retrieval_metrics(
    eeg_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    target_indices: torch.Tensor,
    dimension: int,
    temperature: float = 1.0,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    eeg = F.normalize(eeg_vectors[:, :dimension].float(), dim=-1)
    texts = F.normalize(text_vectors[:, :dimension].float(), dim=-1)
    scores = (eeg @ texts.T) / temperature
    order = scores.argsort(dim=-1, descending=True)
    ranks = (order == target_indices[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    result = {f"r@{k}": (ranks <= k).float().mean().item() for k in ks}
    result["mrr"] = (1.0 / ranks.float()).mean().item()
    result["n_queries"] = float(eeg.shape[0])
    result["n_candidates"] = float(texts.shape[0])
    return result

