"""Add SPNA nested-CARG supervision to GLIM and run the matched ZuCo2 test."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
from glim_spna_losses import nested_carg_objective  # noqa: E402

# The experiment runs beside the official GLIM checkout on the 4090 server.
sys.path.insert(0, "/home/song/projects/glim_zuco")
import run_glim_fair as fair  # noqa: E402

runner = fair.runner
PROJECTS = fair.PROJECTS
GLIM = fair.GLIM
OUT_NAME = "glim_spna_carg_seed42_500"
OUT = GLIM / "outputs" / OUT_NAME
HP_CHECKPOINT = GLIM / "outputs/rapid_full/checkpoint.pt"
SEED = 42
STEPS = 500
MICRO_BATCH = 12
BATCH_SIZE = 72
SMALL_DIM = 64
LARGE_DIM = 256
LAMBDA_RELATION = 0.1
RELATION_TEMPERATURE = 0.05
LOGIT_TEMPERATURE = 0.07
EXPECTED_BATCH_HASH = fair.EXPECTED_BATCH_HASH


class SPNAPrefixHeads(nn.Module):
    """Modality-specific projections into SPNA's shared nested space."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = LARGE_DIM):
        super().__init__()
        self.eeg = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))
        self.text = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, output_dim))

    def forward(self, eeg: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.eeg(eeg), self.text(text)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hp() -> dict:
    checkpoint = torch.load(HP_CHECKPOINT, map_location="cpu", weights_only=False)
    return checkpoint["config"]["hyper_parameters"]


def build_model(hp: dict, device: str) -> nn.Module:
    model = runner.NativeGLIM(hp)
    model.spna_prefix = SPNAPrefixHeads(hp["embed_dim"])
    return model.to(device)


def masked_mean_teacher(cache: dict, uids: torch.Tensor, device: str) -> torch.Tensor:
    ids = uids.cpu()
    hidden = cache["hidden"][ids].to(device=device, dtype=torch.float32)
    mask = cache["masks"][ids].to(device=device, dtype=torch.float32)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)


def evaluate(model: nn.Module, frame, cache: dict, device: str, condition: str) -> dict:
    model.eval()
    dataset = runner.DiskDataset(frame, "test", cache)
    batches, batch_hash = fair.fixed_batches(frame)
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)
    spaces = {
        "glim_1024": {k: 0 for k in (1, 5, 10)},
        "spna_256": {k: 0 for k in (1, 5, 10)},
        "spna_64": {k: 0 for k in (1, 5, 10)},
    }
    predictions = {name: [] for name in spaces}
    noise = None
    if condition == "noise":
        noise = np.random.default_rng(42).standard_normal(
            (len(dataset), 1280, 128), dtype=np.float32)
    actual_batches = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            eeg, mask = batch["eeg"], batch["mask"]
            if noise is not None:
                eeg = torch.from_numpy(noise[batch["row"].numpy()])
                mask = torch.ones_like(mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, eeg_full = model.encode(eeg.to(device), mask.to(device), batch["pids"].to(device))
                text_full = model.aligner.embed_text(
                    cache["hidden"][batch["uid"]].to(device),
                    cache["masks"][batch["uid"]].to(device),
                )
                eeg_nested, text_nested = model.spna_prefix(eeg_full, text_full)
            scores = {
                "glim_1024": F.normalize(eeg_full.float(), dim=-1)
                @ F.normalize(text_full.float(), dim=-1).T,
                "spna_256": F.normalize(eeg_nested.float(), dim=-1)
                @ F.normalize(text_nested.float(), dim=-1).T,
                "spna_64": F.normalize(eeg_nested[:, :SMALL_DIM].float(), dim=-1)
                @ F.normalize(text_nested[:, :SMALL_DIM].float(), dim=-1).T,
            }
            ids = batch["uid"].tolist()
            if len(set(ids)) != 24:
                raise ValueError("A test batch contains duplicate GLIM text uids")
            actual_batches.append(batch["row"].tolist())
            for name, matrix in scores.items():
                order = matrix.argsort(dim=1, descending=True)
                for pos in range(24):
                    row = dataset.df.iloc[int(batch["row"][pos])]
                    item = {"batch": batch_index, "position": pos,
                            "trial_id": row.trial_id, "stimulus_id": row.stimulus_id,
                            "text_uid": int(ids[pos]), "task": row.task,
                            "subject": row.subject, "rank": int((order[pos] == pos).nonzero()[0]) + 1,
                            "noise": condition == "noise"}
                    for k in (1, 5, 10):
                        hit = bool((order[pos, :k] == pos).any())
                        spaces[name][k] += int(hit)
                        item[f"hit_top{k}"] = hit
                    predictions[name].append(item)
    if actual_batches != batches:
        raise ValueError("Hybrid evaluation batch order differs from SPNA")
    n = sum(map(len, batches))
    result = {
        name: {"split": "test", "condition": condition, "n_queries": n,
               "n_batches": len(batches), "candidate_count": 24,
               **{f"top{k}": spaces[name][k] / n for k in (1, 5, 10)},
               "omitted_test_records": len(dataset) - n,
               "batch_manifest_sha256": batch_hash}
        for name in spaces
    }
    target = OUT / condition
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, rows in predictions.items():
        (target / f"{name}_predictions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (target / "batches.json").write_text(json.dumps(actual_batches), encoding="utf-8")
    print("HYBRID_EVAL", json.dumps(result), flush=True)
    return result


def train(frame, cache: dict, hp: dict, device: str) -> nn.Module:
    seed_all(SEED)
    model = build_model(hp, device)
    initial_state_sha = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values()
    )).hexdigest()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_path = PROJECTS / "llm/flan-t5-large"
    runner.T5 = str(model_path)
    tokenizer, lm = runner.load_lm(device)
    lm.eval()
    lm.decoder.gradient_checkpointing_enable()
    lm.decoder.train()

    train_set = runner.DiskDataset(frame, "train", cache)
    loader, sampler = runner.make_loader(train_set, "train", BATCH_SIZE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    config = {
        "seed": SEED, "steps": STEPS, "batch_size": BATCH_SIZE,
        "micro_batch": MICRO_BATCH, "learning_rate": 1e-4, "weight_decay": 0,
        "glim_initialization": "random model; old checkpoint used for hyperparameters only",
        "initial_state_sha256": initial_state_sha,
        "trainable_parameters": trainable_parameters,
        "base": "GLIM native EEG encoder, aligner, frozen FLAN-T5 generation objective",
        "spna_increment": {
            "method": "nested CARG on projected GLIM embeddings",
            "large_dimension": LARGE_DIM, "small_dimension": SMALL_DIM,
            "lambda_relation": LAMBDA_RELATION,
            "relation_temperature": RELATION_TEMPERATURE,
            "logit_temperature": LOGIT_TEMPERATURE,
            "teacher": "frozen FLAN token hidden states with attention-mask mean pooling",
            "capacity_weights": {"small_hard_pair": 0.5, "large_hard_pair": 0.5},
        },
        "combined_weights": {"glim_cmc": 0.25, "spna_nested_carg": 0.25,
                             "glim_generation": 0.5},
        "hyper_parameters": hp,
        "text_feature_cache_sha256": sha256_file(GLIM / "data/text_cache.pt"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trace = (OUT / "train.jsonl").open("w", buffering=1, encoding="utf-8")
    model.train()
    step, epoch, started = 0, 0, time.monotonic()
    while step < STEPS:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= STEPS:
                break
            if len(set(batch["uid"].tolist())) != BATCH_SIZE:
                raise RuntimeError("GLIM training sampler returned duplicate text uids")
            optimizer.zero_grad(set_to_none=True)
            teacher = masked_mean_teacher(cache, batch["uid"], device)
            base_eeg_parts, base_text_parts, nested_eeg_parts, nested_text_parts = [], [], [], []
            rng_states = []
            with torch.no_grad():
                for start in range(0, BATCH_SIZE, MICRO_BATCH):
                    end = start + MICRO_BATCH
                    rng_states.append(torch.cuda.get_rng_state(device))
                    mini = {key: value[start:end].to(device) for key, value in batch.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, eeg_vec = model.encode(mini["eeg"], mini["mask"], mini["pids"])
                        text_vec = model.aligner.embed_text(
                            cache["hidden"][mini["uid"].cpu()].to(device),
                            cache["masks"][mini["uid"].cpu()].to(device),
                        )
                        eeg_nested, text_nested = model.spna_prefix(eeg_vec, text_vec)
                    base_eeg_parts.append(eeg_vec.float())
                    base_text_parts.append(text_vec.float())
                    nested_eeg_parts.append(eeg_nested.float())
                    nested_text_parts.append(text_nested.float())
            eeg_leaf = torch.cat(base_eeg_parts).detach().requires_grad_()
            text_leaf = torch.cat(base_text_parts).detach().requires_grad_()
            eeg_nested_leaf = torch.cat(nested_eeg_parts).detach().requires_grad_()
            text_nested_leaf = torch.cat(nested_text_parts).detach().requires_grad_()
            glim_cmc, _ = model.aligner.align_emb_vector(eeg_leaf, text_leaf)
            spna_carg, carg_parts = nested_carg_objective(
                eeg_nested_leaf, text_nested_leaf, teacher,
                small_dim=SMALL_DIM, large_dim=LARGE_DIM,
                lambda_relation=LAMBDA_RELATION,
                relation_temperature=RELATION_TEMPERATURE,
                logit_temperature=LOGIT_TEMPERATURE,
            )
            (0.25 * glim_cmc + 0.25 * spna_carg).backward()
            grad_eeg, grad_text = eeg_leaf.grad.detach(), text_leaf.grad.detach()
            grad_eeg_nested = eeg_nested_leaf.grad.detach()
            grad_text_nested = text_nested_leaf.grad.detach()
            del eeg_leaf, text_leaf, eeg_nested_leaf, text_nested_leaf

            loss_lm_total = 0.0
            for mini_index, start in enumerate(range(0, BATCH_SIZE, MICRO_BATCH)):
                end = start + MICRO_BATCH
                mini = {key: value[start:end].to(device) for key, value in batch.items()}
                torch.cuda.set_rng_state(rng_states[mini_index], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eeg_sequence, eeg_vec = model.encode(mini["eeg"], mini["mask"], mini["pids"])
                    text_vec = model.aligner.embed_text(
                        cache["hidden"][mini["uid"].cpu()].to(device),
                        cache["masks"][mini["uid"].cpu()].to(device),
                    )
                    eeg_nested, text_nested = model.spna_prefix(eeg_vec, text_vec)
                    surrogate = (eeg_vec.float() * grad_eeg[start:end]).sum()
                    surrogate = surrogate + (text_vec.float() * grad_text[start:end]).sum()
                    surrogate = surrogate + (eeg_nested.float() * grad_eeg_nested[start:end]).sum()
                    surrogate = surrogate + (text_nested.float() * grad_text_nested[start:end]).sum()
                    labels = cache["targets"][mini["uid"].cpu(), mini["variant"].cpu()].to(device).clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    lm_loss = lm(encoder_outputs=runner.BaseModelOutput(eeg_sequence),
                                 labels=labels, use_cache=False).loss
                    scaled = surrogate + 0.5 * lm_loss * (len(range(start, end)) / BATCH_SIZE)
                if not torch.isfinite(scaled):
                    raise RuntimeError(f"Nonfinite hybrid loss at step {step}")
                scaled.backward()
                loss_lm_total += float(lm_loss.detach()) / (BATCH_SIZE // MICRO_BATCH)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            row = {"step": step, "epoch": epoch, "seconds": time.monotonic() - started,
                   "loss": 0.25 * float(glim_cmc.detach()) + 0.25 * float(spna_carg.detach())
                   + 0.5 * loss_lm_total,
                   "glim_cmc": float(glim_cmc.detach()),
                   "spna_carg": float(spna_carg.detach()),
                   "spna_small": float(carg_parts["small"]),
                   "spna_large": float(carg_parts["large"]),
                   "spna_relation": float(carg_parts["relation"]),
                   "lm": loss_lm_total,
                   "max_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
            trace.write(json.dumps(row) + "\n")
            if step == 1 or step % 10 == 0:
                print("TRAIN", OUT_NAME, json.dumps(row), flush=True)
            if step % 250 == 0 or step == STEPS:
                torch.save({"state_dict": model.state_dict(), "step": step,
                            "config": config}, OUT / "checkpoint.pt")
        epoch += 1
    trace.close()
    (OUT / "COMPLETE").write_text(json.dumps({"steps": step, "seconds": time.monotonic()-started}))
    del optimizer, lm
    torch.cuda.empty_cache()
    return model


def preflight(frame, cache: dict, hp: dict, device: str) -> None:
    """Exercise one real GLIM batch, both losses, and all gradient paths."""
    seed_all(SEED)
    model = build_model(hp, device)
    model.train()
    tokenizer, lm = runner.load_lm(str(PROJECTS / "llm/flan-t5-large"))
    lm.decoder.gradient_checkpointing_enable()
    lm.decoder.train()
    dataset = runner.DiskDataset(frame, "train", cache)
    loader, _sampler = runner.make_loader(dataset, "train", BATCH_SIZE)
    batch = next(iter(loader))
    if len(set(batch["uid"].tolist())) != BATCH_SIZE:
        raise RuntimeError("Preflight sampler returned duplicate text uids")
    micro = {key: value[:MICRO_BATCH].to(device) for key, value in batch.items()}
    teacher = masked_mean_teacher(cache, batch["uid"], device)[:MICRO_BATCH]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sequence, eeg_vec = model.encode(micro["eeg"], micro["mask"], micro["pids"])
        text_vec = model.aligner.embed_text(
            cache["hidden"][micro["uid"].cpu()].to(device),
            cache["masks"][micro["uid"].cpu()].to(device),
        )
        eeg_nested, text_nested = model.spna_prefix(eeg_vec, text_vec)
        cmc, _ = model.aligner.align_emb_vector(eeg_vec.float(), text_vec.float())
        carg, _parts = nested_carg_objective(
            eeg_nested.float(), text_nested.float(), teacher,
            small_dim=SMALL_DIM, large_dim=LARGE_DIM,
            lambda_relation=LAMBDA_RELATION,
            relation_temperature=RELATION_TEMPERATURE,
            logit_temperature=LOGIT_TEMPERATURE)
        labels = cache["targets"][micro["uid"].cpu(), micro["variant"].cpu()].to(device).clone()
        labels[labels == tokenizer.pad_token_id] = -100
        lm_loss = lm(encoder_outputs=runner.BaseModelOutput(sequence), labels=labels,
                     use_cache=False).loss
        total = 0.25 * cmc + 0.25 * carg + 0.5 * lm_loss
    total.backward()
    if model.spna_prefix.eeg[-1].weight.grad is None or model.aligner.q_x.grad is None:
        raise RuntimeError("Preflight failed to reach GLIM and SPNA parameters")
    if not torch.isfinite(total):
        raise RuntimeError("Preflight loss is nonfinite")
    print("PREFLIGHT_OK", json.dumps({"batch_size": BATCH_SIZE, "micro_batch": MICRO_BATCH,
                                    "glim_cmc": float(cmc.detach()), "spna_carg": float(carg.detach()),
                                    "lm": float(lm_loss.detach()), "total": float(total.detach()),
                                    "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                                 if p.requires_grad)}), flush=True)
    del lm, model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train", "eval"), default="train")
    args = parser.parse_args()
    device = "cuda:0"
    frame, mapping = fair.load_aligned_frame()
    batches, digest = fair.fixed_batches(frame)
    if digest != EXPECTED_BATCH_HASH:
        raise RuntimeError("Shared evaluation batch hash changed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mapping_audit.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {"manifest_sha256": sha256_file(fair.MANIFEST),
                "rows": len(frame), "stimuli": int(frame.stimulus_id.nunique()),
                "split_rows": frame.manifest_split.value_counts().to_dict(),
                "split_stimuli": frame.groupby("manifest_split").stimulus_id.nunique().to_dict(),
                "test_batches": len(batches), "test_queries": sum(map(len, batches)),
                "omitted_test_records": int(frame.phase.eq("test").sum()) - sum(map(len, batches)),
                "batch_manifest_sha256": digest,
                "training": {"seed": SEED, "steps": STEPS, "batch_size": BATCH_SIZE,
                             "micro_batch": MICRO_BATCH, "lr": 1e-4},
                "text_mapping_note": "Reuses audited GLIM ZuCo2 label mapping, including seven unique near-matches and three normalized uid merges."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = torch.load(GLIM / "data/text_cache.pt", map_location="cpu", weights_only=False)
    hp = load_hp()
    seed_all(SEED)
    if args.mode == "preflight":
        preflight(frame, cache, hp, device)
        return
    if args.mode == "train":
        model = train(frame, cache, hp, device)
    else:
        checkpoint = torch.load(OUT / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = build_model(hp, device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    evaluate(model, frame, cache, device, "normal")
    evaluate(model, frame, cache, device, "noise")


if __name__ == "__main__":
    main()
