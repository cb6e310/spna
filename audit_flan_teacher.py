"""Audit FLAN teacher relations on the exact ZuCo2 SPNA training anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "they", "this", "to",
    "was", "were", "which", "who", "with",
}
TEMPERATURES = (0.05, 0.1, 0.2, 0.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p10": float(np.quantile(arr, 0.1)), "p90": float(np.quantile(arr, 0.9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    run = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(run["seed"])
    batch_size = int(run["config"]["training"]["batch_size"])
    base_temperature = float(run["config"]["teacher"]["relation_temperature"])
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    train_rows = [row for row in rows if row["split"] == "train"]
    stimulus_rows: dict[str, list[int]] = defaultdict(list)
    texts: dict[str, str] = {}
    for index, row in enumerate(train_rows):
        stimulus = row["stimulus_id"]
        stimulus_rows[stimulus].append(index)
        texts.setdefault(stimulus, row["text"])
    stimulus_ids = list(stimulus_rows)

    with np.load(args.cache, allow_pickle=False) as cache:
        cache_ids = cache["stimulus_ids"].astype(str).tolist()
        cache_vectors = np.asarray(cache["embeddings"], dtype=np.float32)
    cache_index = {key: i for i, key in enumerate(cache_ids)}
    missing = sorted(set(stimulus_ids) - set(cache_index))
    if missing:
        raise ValueError(f"Missing {len(missing)} training stimuli from text cache")
    vectors = cache_vectors[[cache_index[key] for key in stimulus_ids]].astype(np.float64)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    n = len(stimulus_ids)
    if n != int(run["train_stimuli"]):
        raise ValueError(f"Expected {run['train_stimuli']} training stimuli, found {n}")

    # Replay the same sampler order used by the seed-42, fixed-step training run.
    per_anchor: dict[float, dict[str, list[list[float]]]] = {
        t: {key: [[] for _ in range(4)] for key in stimulus_ids} for t in TEMPERATURES
    }
    step = 0
    epoch = 0
    while step < args.steps:
        rng = random.Random(seed + epoch)
        shuffled = stimulus_ids.copy()
        rng.shuffle(shuffled)
        batches: list[list[str]] = []
        current: list[str] = []
        for stimulus in shuffled:
            current.append(stimulus)
            if len(current) == batch_size:
                batches.append(current)
                current = []
        if len(current) >= 3:
            batches.append(current)
        elif current and batches:
            batches[-1].extend(current)
        for batch in batches:
            if step >= args.steps:
                break
            batch_vec = vectors[[stimulus_ids.index(s) for s in batch]]
            logits = batch_vec @ batch_vec.T
            diagonal = np.eye(len(batch), dtype=bool)
            logits[diagonal] = -np.inf
            for temperature in TEMPERATURES:
                scaled = logits / temperature
                scaled -= np.max(scaled, axis=1, keepdims=True)
                probabilities = np.exp(scaled)
                probabilities[diagonal] = 0.0
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                sorted_p = np.sort(probabilities, axis=1)[:, ::-1]
                entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum(axis=1)
                normalized_entropy = entropy / math.log(len(batch) - 1)
                for row_index, stimulus in enumerate(batch):
                    slots = per_anchor[temperature][stimulus]
                    slots[0].append(float(normalized_entropy[row_index]))
                    slots[1].append(float(sorted_p[row_index, 0]))
                    slots[2].append(float(sorted_p[row_index, :min(5, len(batch)-1)].sum()))
                    slots[3].append(float(len(batch) - 1))
            step += 1
        epoch += 1

    batch_metrics = {}
    for temperature, anchors in per_anchor.items():
        anchor_stats = np.asarray([
            [float(np.mean(metric_values)) for metric_values in anchors[s]]
            for s in stimulus_ids
        ])
        batch_metrics[str(temperature)] = {
            "normalized_entropy": summarize(anchor_stats[:, 0].tolist()),
            "top1_mass": summarize(anchor_stats[:, 1].tolist()),
            "top5_mass": summarize(anchor_stats[:, 2].tolist()),
            "anchors_seen": int(sum(bool(anchors[s][0]) for s in stimulus_ids)),
            "batch_count": int(step),
            "mean_candidates_per_anchor": float(anchor_stats[:, 3].mean()),
        }

    # Full training-bank neighborhood audit at the configured teacher temperature.
    cosine = vectors @ vectors.T
    np.fill_diagonal(cosine, -np.inf)
    probs = np.exp((cosine - np.max(cosine, axis=1, keepdims=True)) / base_temperature)
    np.fill_diagonal(probs, 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    neighbors = np.argsort(cosine, axis=1)[:, ::-1]
    token_sets = [tokenize(texts[s]) for s in stimulus_ids]
    word_counts = [len(re.findall(r"\b\w+\b", texts[s])) for s in stimulus_ids]
    top1_jaccard, top5_jaccard, top1_length_diff, weighted_jaccard = [], [], [], []
    for i in range(n):
        top = neighbors[i, :5]
        jaccards = []
        for j in top:
            union = token_sets[i] | token_sets[j]
            jaccards.append(len(token_sets[i] & token_sets[j]) / len(union) if union else 0.0)
        top1_jaccard.append(jaccards[0])
        top5_jaccard.append(float(np.mean(jaccards)))
        top1_length_diff.append(abs(word_counts[i] - word_counts[top[0]]))
        weighted_jaccard.append(float(np.dot(probs[i], np.asarray([
            len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j])
            if token_sets[i] | token_sets[j] else 0.0 for j in range(n)
        ]))))

    # Row-wise permutation reproduces the shuffled-relation control's assignment.
    rng = np.random.default_rng(20261110)
    true_rank_corr, shuffled_rank_corr = [], []
    flat_true, flat_shuffled = [], []
    base_q = probs
    shuffled_q = np.zeros_like(base_q)
    for i in range(n):
        candidates = np.flatnonzero(np.arange(n) != i)
        permuted = rng.permutation(candidates)
        shuffled_q[i, candidates] = base_q[i, permuted]
        true_rank_corr.append(correlation(rankdata(base_q[i, candidates]),
                                          rankdata(cosine[i, candidates])))
        shuffled_rank_corr.append(correlation(rankdata(base_q[i, candidates]),
                                              rankdata(shuffled_q[i, candidates])))
        flat_true.extend(base_q[i, candidates].tolist())
        flat_shuffled.extend(shuffled_q[i, candidates].tolist())

    # Positive temperature scaling preserves all raw-cosine candidate ranks.
    temp_rank_stability = {str(t): 1.0 for t in TEMPERATURES}
    temp_pairwise_stability = {}
    for left in TEMPERATURES:
        for right in TEMPERATURES:
            temp_pairwise_stability[f"{left:g}_vs_{right:g}"] = 1.0

    examples = []
    for i in np.argsort(np.max(cosine, axis=1))[::-1][:12]:
        j = int(neighbors[i, 0])
        examples.append({
            "anchor_id": stimulus_ids[i], "anchor": texts[stimulus_ids[i]],
            "nearest_id": stimulus_ids[j], "nearest": texts[stimulus_ids[j]],
            "cosine": float(cosine[i, j]), "anchor_word_count": word_counts[i],
            "nearest_word_count": word_counts[j], "content_lexical_jaccard": top1_jaccard[i],
            "teacher_probability_full_train_bank": float(probs[i, j]),
        })

    output = {
        "protocol": "FLAN-T5-large attention-mask mean pooled vectors; cosine similarities; off-diagonal row softmax",
        "source": {"manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
                   "text_cache": str(args.cache), "text_cache_sha256": sha256(args.cache),
                   "run_config": str(args.config)},
        "training": {"seed": seed, "optimizer_steps_replayed": step, "epochs_replayed": epoch,
                     "train_stimuli": n, "batch_size": batch_size,
                     "batch_sizes_from_sampler": sorted(set(len(b) for b in batches)),
                     "configured_relation_temperature": base_temperature},
        "temperatures": {"values": list(TEMPERATURES), "training_batch_metrics": batch_metrics,
                         "candidate_rank_stability": temp_rank_stability,
                         "pairwise_rank_stability": temp_pairwise_stability,
                         "note": "Scaling cosine logits by any positive temperature cannot change candidate order; probabilities and entropy do change."},
        "full_training_bank_at_configured_temperature": {
            "candidate_count_per_anchor": n - 1,
            "normalized_entropy": summarize((-(probs * np.log(np.maximum(probs, 1e-300))).sum(axis=1)
                                               / math.log(n - 1)).tolist()),
            "top1_mass": summarize(np.max(probs, axis=1).tolist()),
            "top5_mass": summarize(np.sort(probs, axis=1)[:, -5:].sum(axis=1).tolist()),
            "expected_content_lexical_jaccard": summarize(weighted_jaccard),
            "top1_content_lexical_jaccard": summarize(top1_jaccard),
            "mean_top5_content_lexical_jaccard": summarize(top5_jaccard),
            "top1_absolute_word_count_difference": summarize(top1_length_diff),
            "true_vs_shuffled_relation": {
                "true_probability_vs_cosine_mean_row_spearman": float(np.mean(true_rank_corr)),
                "true_vs_shuffled_mean_row_spearman": float(np.mean(shuffled_rank_corr)),
                "true_vs_shuffled_flat_pearson": correlation(np.asarray(flat_true), np.asarray(flat_shuffled)),
                "shuffle_seed": 20261110,
            },
        },
        "qualitative_nearest_neighbor_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training": output["training"],
                      "batch_metrics": batch_metrics,
                      "full_bank": output["full_training_bank_at_configured_temperature"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
