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
from .losses import (gradient_conflict_diagnostics, hard_pair_loss,
                     negative_relation_loss, spna_objective, text_relation_target,
                     weighted_capacity_loss)
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
    "nested_three": {"relation_mode": "none", "capacity_mode": "three_layer"},
    "staged_nested_three": {"relation_mode": "none", "capacity_mode": "three_layer_staged"},
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
                        device: torch.device, dimensions: list[int]) -> dict:
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
    return {
        str(dimension): retrieval_metrics(
            eeg_vectors, candidate_vectors, target_indices, dimension
        ) for dimension in dimensions
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

    multi_capacity = variant.get("capacity_mode", "") in {
        "three_layer", "three_layer_staged"
    }
    dimensions = ([int(value) for value in cfg["capacities"].get(
        "layers", [cfg["capacities"]["small"], 128, cfg["model"]["embedding_dim"]]
    )] if multi_capacity else [
        int(cfg["capacities"]["small"]), int(cfg["model"]["embedding_dim"])
    ])
    if sorted(set(dimensions)) != dimensions or dimensions[-1] != int(cfg["model"]["embedding_dim"]):
        raise ValueError("Capacity layers must be increasing and end at embedding_dim")
    if dimensions[0] != int(cfg["capacities"]["small"]):
        raise ValueError("First capacity layer must match capacities.small")
    if any(value < 1 or value > int(cfg["model"]["embedding_dim"]) for value in dimensions):
        raise ValueError("Capacity layers must lie within the model embedding dimension")
    weights = ([float(value) for value in cfg["capacities"].get("weights", [0.35, 0.15, 0.50])]
               if multi_capacity else [float(variant["alpha_small"]),
                                      float(variant["alpha_large"])])
    if (len(weights) != len(dimensions) or any(value < 0 for value in weights)
            or abs(sum(weights) - 1.0) > 1e-6):
        raise ValueError("Capacity weights must match layers and sum to one")
    evaluation_dimensions = [int(value) for value in cfg["capacities"].get(
        "evaluation_layers", dimensions
    )]
    if (not evaluation_dimensions
            or sorted(set(evaluation_dimensions)) != evaluation_dimensions
            or evaluation_dimensions[0] < 1
            or evaluation_dimensions[-1] > int(cfg["model"]["embedding_dim"])):
        raise ValueError("Evaluation capacities must be increasing and within embedding_dim")
    selection_weights = [float(value) for value in cfg["capacities"].get(
        "selection_weights", [1.0] + [0.0] * (len(evaluation_dimensions) - 1)
    )]
    if len(selection_weights) != len(evaluation_dimensions) or abs(sum(selection_weights) - 1.0) > 1e-6:
        raise ValueError("Selection weights must match evaluation_layers and sum to one")
    selection_description = (
        "best weighted validation full-bank R@1 across capacities"
        if "selection_weights" in cfg["capacities"]
        else "best validation small-capacity full-bank R@1"
    )

    if variant.get("capacity_mode") == "three_layer_staged":
        warmup_steps = int(round(args.steps * float(
            cfg["training"].get("high_capacity_warmup_fraction", 0.25)
        )))
        if not 0 < warmup_steps < args.steps:
            raise ValueError("High-capacity warmup must leave steps for nested fine-tuning")
        phases = [
            {"name": "high_capacity_warmup", "steps": warmup_steps,
             "weights": [0.0] * (len(dimensions) - 1) + [1.0], "select": False},
            {"name": "nested_finetune", "steps": args.steps - warmup_steps,
             "weights": weights, "select": True},
        ]
    else:
        phases = [{"name": "joint", "steps": args.steps, "weights": weights, "select": True}]

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
    best_selection_score = float("-inf")
    start = time.monotonic()
    optimizer_step = 0

    with trace_path.open("w", encoding="utf-8", buffering=1) as trace:
        epoch = 0
        for phase in phases:
            phase_step_target = optimizer_step + int(phase["steps"])
            phase_epoch = 0
            while optimizer_step < phase_step_target:
                sampler.set_epoch(epoch)
                model.train()
                for batch in loader:
                    if optimizer_step >= phase_step_target:
                        break
                    if len(set(batch["stimulus_ids"])) != len(batch["stimulus_ids"]):
                        raise RuntimeError("Duplicate stimulus in training batch")
                    eeg = batch["eeg"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    frozen_text = batch["text"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    eeg_vectors = model.encode_eeg(eeg, mask)
                    text_vectors = model.encode_text(frozen_text)
                    logits_by_dimension = [
                        model.score(
                            eeg_vectors, text_vectors, dimension,
                            cfg["capacities"].get("temperature", cfg["capacities"].get(
                                "small_temperature" if index == 0 else "large_temperature", 0.07
                            )),
                        ) for index, dimension in enumerate(dimensions)
                    ]
                    small_logits, large_logits = logits_by_dimension[0], logits_by_dimension[-1]
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
                    if multi_capacity:
                        loss, capacity_loss_map = weighted_capacity_loss(
                            logits_by_dimension, phase["weights"]
                        )
                        capacity_losses = [capacity_loss_map[str(index)]
                                           for index in range(len(dimensions))]
                        relation_loss = (negative_relation_loss(small_logits, relations)
                                         if lambda_relation > 0 else small_logits.sum() * 0.0)
                        loss = loss + lambda_relation * relation_loss
                        components = {
                            **{f"hard_{dimension}": part.detach()
                               for dimension, part in zip(dimensions, capacity_losses)},
                            "relation": relation_loss.detach(),
                        }
                    else:
                        capacity_losses = ([hard_pair_loss(logits) for logits in logits_by_dimension]
                                           if args.variant == "nested_hard" else None)
                        loss, components = spna_objective(
                            small_logits, large_logits, relations,
                            lambda_relation=lambda_relation,
                            alpha_small=variant["alpha_small"],
                            alpha_large=variant["alpha_large"],
                        )
                    if (capacity_losses is not None and (optimizer_step == 0
                            or (optimizer_step + 1) % 25 == 0) and phase["select"]):
                        shared_parameters = [parameter for parameter in model.parameters()
                                             if parameter.requires_grad]
                        grad_diagnostics = gradient_conflict_diagnostics(
                            {str(dimension): part for dimension, part
                             in zip(dimensions, capacity_losses)}, shared_parameters
                        )
                    else:
                        grad_diagnostics = None
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch={epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                    optimizer.step()
                    optimizer_step += 1
                    record = {
                        "step": optimizer_step,
                        "epoch": epoch + 1,
                        "phase": phase["name"],
                        "train_loss": float(loss.detach()),
                        "capacity_losses": {
                            key: float(value) for key, value in components.items()
                            if key.startswith("hard_")
                        } if multi_capacity else {
                            "small": float(components["small"]),
                            "large": float(components["large"]),
                        },
                        "relation_loss": float(components["relation"]),
                        "lambda_relation": lambda_relation,
                        "capacity_weights": phase["weights"],
                        "gradient_conflicts": grad_diagnostics,
                        "seconds": time.monotonic() - start,
                        "max_cuda_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
                    }
                    trace.write(json.dumps(record) + "\n")
                    if optimizer_step == 1 or optimizer_step % 25 == 0:
                        print(json.dumps(record), flush=True)
                epoch += 1
                phase_epoch += 1
                validation = evaluate_validation(
                    model, validation_data, cfg, device, evaluation_dimensions
                )
                current_small_r1 = validation[str(evaluation_dimensions[0])]["r@1"]
                selection_score = sum(
                    weight * validation[str(dimension)]["r@1"]
                    for dimension, weight in zip(evaluation_dimensions, selection_weights)
                )
                trace.write(json.dumps({
                    "epoch": epoch,
                    "phase": phase["name"],
                    "phase_epoch": phase_epoch,
                    "step": optimizer_step,
                    "validation": validation,
                    "validation_selection_score": selection_score,
                    "seconds": time.monotonic() - start,
                }) + "\n")
                if phase["select"] and selection_score > best_selection_score:
                    best_selection_score = selection_score
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
                            "capacity_dimensions": dimensions,
                            "capacity_weights": weights,
                            "evaluation_dimensions": evaluation_dimensions,
                            "selection_weights": selection_weights,
                            "validation_selection_score": selection_score,
                            "selected_by": selection_description,
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
        "capacity_dimensions": dimensions,
        "capacity_weights": weights,
        "evaluation_dimensions": evaluation_dimensions,
        "selection_weights": selection_weights,
        "validation_selection_score": best_selection_score,
        "phases": [{"name": phase["name"], "steps": phase["steps"],
                    "weights": phase["weights"]} for phase in phases],
        "elapsed_seconds": time.monotonic() - start,
        "selected_by": selection_description,
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

