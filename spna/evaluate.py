"""Evaluate a selected checkpoint on a full split stimulus bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ZuCoManifestDataset, load_text_cache, read_manifest
from .metrics import retrieval_metrics
from .model import ModelConfig, SPNA
from .train import collate_trials


@torch.inference_mode()
def evaluate_glim_batches(model: SPNA, dataset: ZuCoManifestDataset, cfg: dict,
                          device: torch.device) -> dict:
    """ZuCo2-only GLIM-style fixed 24-candidate normal/noise/mismatch test."""
    model.eval()
    # Match GLIMSampler.sample_batches for a non-shuffled test phase: scan the
    # remaining rows, emit a 24-row batch with unique stimulus IDs, and carry
    # duplicate-ID rows into the next pass. Drop only the final incomplete
    # batch, preserving all rows that fit a full candidate set.
    remaining = list(range(len(dataset)))
    batches: list[list[int]] = []
    while remaining:
        used: set[str] = set()
        current: list[int] = []
        unused: list[int] = []
        for row_index in remaining:
            stimulus_id = dataset.rows[row_index]["stimulus_id"]
            if stimulus_id not in used and len(current) < 24:
                used.add(stimulus_id)
                current.append(row_index)
            else:
                unused.append(row_index)
        if len(current) < 24:
            break
        batches.append(current)
        remaining = unused
    if not batches:
        raise ValueError("ZuCo2 test split has fewer than 24 unique stimuli")

    def score_condition(condition: str) -> tuple[dict, list[dict]]:
        rng = np.random.default_rng(42)
        noise_array = None
        if condition == "noise":
            noise_array = rng.standard_normal(
                (len(dataset), cfg["model"]["max_samples"], cfg["model"]["channels"]),
                dtype=np.float32,
            )
        hits = {k: 0 for k in (1, 5, 10)}
        mismatch_hits = {k: 0 for k in (1, 5, 10)}
        domain_stats: dict[str, dict[str, int]] = {}
        predictions: list[dict] = []
        started = time.monotonic()
        for batch_index, row_indices in enumerate(batches):
            samples = [dataset[row_index] for row_index in row_indices]
            batch = collate_trials(samples)
            eeg = batch["eeg"]
            mask = batch["mask"]
            if condition == "noise":
                # Draw consecutive rows from the same seeded float32 Gaussian
                # stream as GLIM; noise has a full-valid mask.
                eeg = torch.from_numpy(noise_array[row_indices])
                mask = torch.ones_like(mask)
            eeg_vectors = model.encode_eeg(eeg.to(device), mask.to(device)).float().cpu()
            ids = batch["stimulus_ids"]
            text = torch.stack([
                dataset.text_embeddings[dataset.text_index[stimulus]] for stimulus in ids
            ]).to(device)
            text_vectors = model.encode_text(text).float().cpu()
            eeg_norm = torch.nn.functional.normalize(eeg_vectors, dim=-1)
            text_norm = torch.nn.functional.normalize(text_vectors, dim=-1)
            scores = eeg_norm @ text_norm.T
            shifted = scores.roll(1, dims=0)
            order = scores.argsort(dim=-1, descending=True)
            mismatch_order = shifted.argsort(dim=-1, descending=True)
            for position, row_index in enumerate(row_indices):
                row = dataset.rows[row_index]
                domain = f"{row.get('dataset', 'ZuCo2')}_{row.get('task', 'unknown')}"
                stats = domain_stats.setdefault(domain, {"n": 0, **{f"top{k}": 0 for k in (1, 5, 10)}})
                stats["n"] += 1
                record = {"batch": batch_index, "position": position,
                          "manifest_row": row_index, "stimulus_id": ids[position],
                          "domain": domain, "subject": row.get("subject_id", "")}
                for k in (1, 5, 10):
                    hit = bool((order[position, :k] == position).any())
                    miss_hit = bool((mismatch_order[position, :k] == position).any())
                    hits[k] += int(hit)
                    mismatch_hits[k] += int(miss_hit)
                    stats[f"top{k}"] += int(hit)
                    record[f"hit_top{k}"] = hit
                record["top5_stimulus_ids"] = [ids[int(i)] for i in order[position, :5]]
                predictions.append(record)
        n = sum(len(batch) for batch in batches)
        metrics = {
            "condition": condition,
            "n_queries": n,
            "n_batches": len(batches),
            "candidate_count": 24,
            **{f"top{k}": hits[k] / n for k in hits},
            "cyclic_mismatched_eeg": {f"top{k}": mismatch_hits[k] / n for k in hits},
            "domains": {name: {key: value / stats["n"] for key, value in stats.items()
                               if key != "n"} | {"n_queries": stats["n"]}
                        for name, stats in domain_stats.items()},
            "omitted_test_records": len(dataset) - n,
            "batch_manifest_sha256": hashlib.sha256(
                json.dumps(batches, separators=(",", ":")).encode()
            ).hexdigest(),
            "seconds": time.monotonic() - started,
        }
        return metrics, predictions

    normal, normal_predictions = score_condition("normal")
    noise, noise_predictions = score_condition("noise")
    return {"protocol": "GLIM-style ZuCo2-only; fixed ordered batches of 24 unique test stimuli",
            "normal": normal, "noise": noise,
            "normal_predictions": normal_predictions,
            "noise_predictions": noise_predictions}


@torch.inference_mode()
def evaluate_full_bank(model: SPNA, dataset: ZuCoManifestDataset, cfg: dict,
                       device: torch.device, batch_size: int) -> dict:
    """Evaluate every trial against every unique stimulus in the split."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_trials)
    eeg_vectors, targets = [], []
    for batch in loader:
        eeg_vectors.append(model.encode_eeg(batch["eeg"].to(device), batch["mask"].to(device)).cpu())
        targets.extend(batch["stimulus_ids"])
    candidate_ids = sorted(dataset.stimulus_ids)
    candidate_text = torch.stack([
        dataset.text_embeddings[dataset.text_index[stimulus]] for stimulus in candidate_ids
    ]).to(device)
    candidate_vectors = model.encode_text(candidate_text).cpu()
    candidate_index = {stimulus: index for index, stimulus in enumerate(candidate_ids)}
    target_indices = torch.tensor([candidate_index[stimulus] for stimulus in targets])
    eeg_vectors = torch.cat(eeg_vectors)
    full_dim = cfg["model"]["embedding_dim"]
    return {
        "candidate_stimuli": len(candidate_ids),
        "small_capacity": retrieval_metrics(
            eeg_vectors, candidate_vectors, target_indices, cfg["capacities"]["small"]
        ),
        "full_capacity": retrieval_metrics(eeg_vectors, candidate_vectors, target_indices, full_dim),
        "primary_metric": "full-bank EEG-to-text R@1 at small capacity",
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--protocol", choices=("benchmark", "full_bank", "both"),
                        default="benchmark")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    rows = read_manifest(args.manifest)
    text_index, text_embeddings = load_text_cache(args.text_cache)
    dataset = ZuCoManifestDataset(
        rows, args.split, text_index, text_embeddings,
        cfg["model"]["channels"], cfg["model"]["max_samples"],
    )
    model = SPNA(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if args.split == "test":
        metrics = {"split": args.split, "variant": checkpoint["variant"],
                   "seed": checkpoint["seed"], "checkpoint_epoch": checkpoint["epoch"]}
        if args.protocol in {"benchmark", "both"}:
            benchmark = evaluate_glim_batches(model, dataset, cfg, device)
            normal_predictions = benchmark.pop("normal_predictions")
            noise_predictions = benchmark.pop("noise_predictions")
            metrics["glim_benchmark"] = benchmark
        if args.protocol in {"full_bank", "both"}:
            metrics["full_bank"] = evaluate_full_bank(
                model, dataset, cfg, device, args.batch_size
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if args.protocol in {"benchmark", "both"}:
            args.output.with_name(args.output.stem + "_predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in normal_predictions) + "\n", encoding="utf-8")
            args.output.with_name(args.output.stem + "_noise_predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in noise_predictions) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    metrics = evaluate_full_bank(model, dataset, cfg, device, args.batch_size)
    metrics.update({"split": args.split, "variant": checkpoint["variant"],
                    "seed": checkpoint["seed"], "checkpoint_epoch": checkpoint["epoch"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

