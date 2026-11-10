"""Train SPNA-CARG and the specified capacity/smoothing controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import UniqueStimulusBatchSampler, ZuCoManifestDataset, load_text_cache, read_manifest
from .losses import spna_objective, text_relation_target
from .metrics import retrieval_metrics
from .model import ModelConfig, SPNA


VARIANTS = {
    "d_only": {"alpha_small": 0.0, "alpha_large": 1.0, "relation_mode": "none"},
    "small_only": {"alpha_small": 1.0, "alpha_large": 0.0, "relation_mode": "none"},
    "nested_hard": {"alpha_small": 0.5, "alpha_large": 0.5, "relation_mode": "none"},
    "spna_carg": {"alpha_small": 0.5, "alpha_large": 0.5,
                  "relation_mode": "true"},
    "uniform_smoothing": {"alpha_small": 0.5, "alpha_large": 0.5,
                          "relation_mode": "uniform"},
    "shuffled_relations": {"alpha_small": 0.5, "alpha_large": 0.5,
                           "relation_mode": "shuffled"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_trials(batch: list[dict]) -> dict:
    return {
        "eeg": torch.stack([item["eeg"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "text": torch.stack([item["text"] for item in batch]),
        "stimulus_ids": [item["stimulus_id"] for item in batch],
    }


@torch.no_grad()
def evaluate_validation(model: SPNA, dataset: ZuCoManifestDataset, cfg: dict,
                        device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg["training"]["eval_batch_size"],
                        shuffle=False, num_workers=0, collate_fn=collate_trials)
    eeg_vectors = []
    target_ids = []
    for batch in loader:
        eeg_vectors.append(model.encode_eeg(batch["eeg"].to(device), batch["mask"].to(device)).cpu())
        target_ids.extend(batch["stimulus_ids"])
    candidate_ids = sorted(dataset.stimulus_ids)
    candidate_text = torch.stack([
        dataset.text_embeddings[dataset.text_index[stimulus]] for stimulus in candidate_ids
    ]).to(device)
    candidate_vectors = model.encode_text(candidate_text).cpu()
    target_map = {stimulus: i for i, stimulus in enumerate(candidate_ids)}
    target_indices = torch.tensor([target_map[stimulus] for stimulus in target_ids])
    eeg_vectors = torch.cat(eeg_vectors)
    small_dim, full_dim = cfg["capacities"]["small"], cfg["model"]["embedding_dim"]
    return {
        "small": retrieval_metrics(eeg_vectors, candidate_vectors, target_indices, small_dim),
        "full": retrieval_metrics(eeg_vectors, candidate_vectors, target_indices, full_dim),
    }


def run(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if float(cfg["teacher"]["relation_temperature"]) <= 0:
        raise ValueError("teacher relation_temperature must be positive")
    if float(cfg["teacher"]["lambda_relation"]) < 0:
        raise ValueError("teacher lambda_relation must be nonnegative")
    variant = VARIANTS[args.variant]
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    manifest_path = Path(cfg["data"]["manifest"])
    text_cache_path = Path(cfg["data"]["text_cache"])
    rows = read_manifest(manifest_path)
    text_index, text_embeddings = load_text_cache(text_cache_path)
    train_stimuli = sorted({row["stimulus_id"] for row in rows if row["split"] == "train"})
    if not 0 < args.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    fraction_rng = np.random.default_rng(args.seed)
    train_order = fraction_rng.permutation(train_stimuli).tolist()
    train_count = max(3, int(round(len(train_stimuli) * args.train_fraction)))
    selected_train = set(train_order[:train_count])
    selected_rows = [
        row for row in rows
        if row["split"] != "train" or row["stimulus_id"] in selected_train
    ]
    train_data = ZuCoManifestDataset(
        selected_rows, "train", text_index, text_embeddings,
        cfg["model"]["channels"], cfg["model"]["max_samples"],
    )
    validation_data = ZuCoManifestDataset(
        selected_rows, "validation", text_index, text_embeddings,
        cfg["model"]["channels"], cfg["model"]["max_samples"],
    )
    if len(train_data.stimulus_ids) < 3:
        raise ValueError("Training split needs at least three unique stimuli")
    if cfg["capacities"]["small"] >= cfg["model"]["embedding_dim"]:
        raise ValueError("small capacity must be less than embedding_dim")

    output_dir = Path(args.output) / args.variant / f"fraction_{args.train_fraction:g}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "variant": args.variant,
        "variant_objective": variant,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_stimuli": len(train_data.stimulus_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "text_cache_sha256": sha256_file(text_cache_path),
        "config": cfg,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    model = SPNA(ModelConfig(**cfg["model"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sampler = UniqueStimulusBatchSampler(train_data, cfg["training"]["batch_size"], args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=0,
                        collate_fn=collate_trials, pin_memory=(device.type == "cuda"))
    trace_path = output_dir / "train.jsonl"
    best_small_r1 = float("-inf")
    best_epoch = 0
    best_step = 0
    best_validation = None
    start = time.monotonic()
    optimizer_step = 0

    with trace_path.open("w", encoding="utf-8", buffering=1) as trace:
        epoch = 0
        while optimizer_step < args.steps:
            sampler.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            steps = 0
            for batch in loader:
                if optimizer_step >= args.steps:
                    break
                if len(set(batch["stimulus_ids"])) != len(batch["stimulus_ids"]):
                    raise RuntimeError("Duplicate stimulus in training batch")
                eeg = batch["eeg"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                frozen_text = batch["text"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                eeg_vectors = model.encode_eeg(eeg, mask)
                text_vectors = model.encode_text(frozen_text)
                small_dim = cfg["capacities"]["small"]
                small_logits = model.score(
                    eeg_vectors, text_vectors, small_dim, cfg["capacities"]["small_temperature"]
                )
                large_logits = model.score(
                    eeg_vectors, text_vectors, cfg["model"]["embedding_dim"],
                    cfg["capacities"]["large_temperature"],
                )
                lambda_relation = (
                    cfg["teacher"]["lambda_relation"]
                    if variant["relation_mode"] != "none" else 0.0
                )
                relations = (
                    text_relation_target(
                        frozen_text, cfg["teacher"]["relation_temperature"],
                        mode=variant["relation_mode"],
                    ) if lambda_relation > 0 else None
                )
                loss, components = spna_objective(
                    small_logits, large_logits, relations,
                    lambda_relation=lambda_relation,
                    alpha_small=variant["alpha_small"],
                    alpha_large=variant["alpha_large"],
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                optimizer.step()
                optimizer_step += 1
                loss_sum += float(loss.detach())
                steps += 1
                record = {
                    "step": optimizer_step,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.detach()),
                    "hard_small_loss": float(components["small"]),
                    "hard_large_loss": float(components["large"]),
                    "relation_loss": float(components["relation"]),
                    "lambda_relation": lambda_relation,
                    "seconds": time.monotonic() - start,
                    "max_cuda_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
                }
                trace.write(json.dumps(record) + "\n")
                if optimizer_step == 1 or optimizer_step % 25 == 0:
                    print(json.dumps(record), flush=True)
            epoch += 1
            validation = evaluate_validation(model, validation_data, cfg, device)
            current_small_r1 = validation["small"]["r@1"]
            trace.write(json.dumps({
                "epoch": epoch,
                "step": optimizer_step,
                "validation": validation,
                "validation_small_r1": current_small_r1,
                "seconds": time.monotonic() - start,
            }) + "\n")
            if current_small_r1 > best_small_r1:
                best_small_r1 = current_small_r1
                best_epoch = epoch
                best_step = optimizer_step
                best_validation = validation
                torch.save(
                    {
                        "config": cfg,
                        "variant": args.variant,
                        "seed": args.seed,
                        "train_fraction": args.train_fraction,
                        "train_stimuli": len(train_data.stimulus_ids),
                        "epoch": best_epoch,
                        "step": best_step,
                        "validation": best_validation,
                        "selected_by": "best validation small-capacity full-bank R@1",
                        "model_state": model.state_dict(),
                    },
                    output_dir / "best.pt",
                )
    (output_dir / "run.json").write_text(json.dumps({
        "variant": args.variant,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_stimuli": len(train_data.stimulus_ids),
        "epochs": epoch,
        "optimizer_steps": optimizer_step,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_validation_small_r1": best_small_r1,
        "best_validation": best_validation,
        "elapsed_seconds": time.monotonic() - start,
        "selected_by": "best validation small-capacity full-bank R@1",
        "test_evaluated": False,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="spna_carg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, choices=(0.25, 0.5, 1.0), default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

