"""Train GLIM on the SPNA ZuCo2 split and evaluate on identical 24-way batches."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.machinery
import json
import re
import sys
import types
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECTS = Path("/home/song/projects")
SPNA = PROJECTS / "spna"
GLIM = PROJECTS / "glim_zuco"
MANIFEST = SPNA / "data/zuco2/zuco2_manifest.csv"
LABELS = GLIM / "upstream/data/tmp/zuco_label_8variants.df"
OUT_NAME = "fair_zuco2_seed42_500"
EXPECTED_BATCH_HASH = "3a50fca068d73d1711d948066e94eb4689ee0a426208a7bd92351f68cad5186d"

sys.path.insert(0, str(SPNA / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(GLIM / "upstream"))
sys.path.insert(0, str(GLIM / ".deps"))
sys.path.insert(0, str(GLIM))

# The GLIM blocks use only timm's default two-layer MLP configuration.
if "timm" not in sys.modules:
    class TimmCompatibleMlp(torch.nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None,
                     act_layer=None, drop=0.0, **_kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = torch.nn.Linear(in_features, hidden_features)
            self.act = (act_layer or torch.nn.GELU)()
            self.drop1 = torch.nn.Dropout(drop)
            self.norm = torch.nn.Identity()
            self.fc2 = torch.nn.Linear(hidden_features, out_features)
            self.drop2 = torch.nn.Dropout(drop)

        def forward(self, value):
            return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(value))))))

    timm_module = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_vit = types.ModuleType("timm.models.vision_transformer")
    timm_module.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None, is_package=True)
    timm_models.__spec__ = importlib.machinery.ModuleSpec("timm.models", loader=None, is_package=True)
    timm_vit.__spec__ = importlib.machinery.ModuleSpec("timm.models.vision_transformer", loader=None)
    timm_vit.Mlp = TimmCompatibleMlp
    timm_module.models = timm_models
    timm_models.vision_transformer = timm_vit
    sys.modules.update({"timm": timm_module, "timm.models": timm_models,
                       "timm.models.vision_transformer": timm_vit})

import run_experiment as runner  # noqa: E402


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(char for char in value if not unicodedata.combining(char))
    for codepoint in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212):
        value = value.replace(chr(codepoint), "-")
    return re.sub(r"\s+", " ", value.replace("?", "-").casefold()).strip()


def load_aligned_frame() -> tuple[pd.DataFrame, list[dict]]:
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    labels = pd.read_pickle(LABELS)
    zuco2 = labels.loc[labels.dataset.eq("ZuCo2")].drop_duplicates("text uid")
    by_key: dict[str, list[dict]] = {}
    for row in zuco2.to_dict("records"):
        by_key.setdefault(normalize_text(row["input text"]), []).append(row)

    map_rows = []
    aligned = []
    for row in manifest.to_dict("records"):
        key = normalize_text(row["text"])
        matches = by_key.get(key, [])
        score = 1.0
        if not matches:
            candidates = sorted(
                ((difflib.SequenceMatcher(None, key, candidate).ratio(), candidate, values)
                 for candidate, values in by_key.items()), reverse=True
            )
            score, _, matches = candidates[0]
            next_score = candidates[1][0] if len(candidates) > 1 else 0.0
            if score < 0.97 or score - next_score < 0.05:
                raise ValueError(f"Ambiguous GLIM label mapping for {row['trial_id']}: {score:.4f}")
        uids = {int(match["text uid"]) for match in matches}
        if len(uids) != 1:
            raise ValueError(f"Colliding GLIM text uids for {row['trial_id']}: {uids}")
        label = matches[0]
        split = {"validation": "val"}.get(row["split"], row["split"])
        aligned.append({
            **label,
            "dataset": "ZuCo2",
            "task": row["task"],
            "subject": row["subject_id"],
            "phase": split,
            "mask": int(row["valid_samples"]),
            "eeg": (row["eeg_file"], int(row["eeg_index"])),
            "stimulus_id": row["stimulus_id"],
            "trial_id": row["trial_id"],
            "source_text": row["text"],
            "manifest_split": row["split"],
        })
        map_rows.append({"trial_id": row["trial_id"], "stimulus_id": row["stimulus_id"],
                         "text_uid": next(iter(uids)), "similarity": score,
                         "manifest_text": row["text"], "glim_text": label["input text"]})
    frame = pd.DataFrame(aligned)
    if len(frame) != len(manifest) or frame["stimulus_id"].nunique() != manifest["stimulus_id"].nunique():
        raise ValueError("Manifest alignment changed row or stimulus counts")
    if frame.groupby("stimulus_id").manifest_split.nunique().max() != 1:
        raise ValueError("A stimulus appears in multiple splits")
    return frame, map_rows


def fixed_batches(frame: pd.DataFrame) -> tuple[list[list[int]], str]:
    test = frame.loc[frame.phase.eq("test")].reset_index(drop=True)
    remaining = list(range(len(test)))
    batches = []
    while remaining:
        used, current, unused = set(), [], []
        for index in remaining:
            stimulus = test.iloc[index].stimulus_id
            if stimulus not in used and len(current) < 24:
                used.add(stimulus)
                current.append(index)
            else:
                unused.append(index)
        if len(current) < 24:
            break
        batches.append(current)
        remaining = unused
    digest = hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()
    if digest != EXPECTED_BATCH_HASH:
        raise ValueError(f"SPNA test batch mismatch: {digest}")
    return batches, digest


def evaluate_common(model, frame, cache, device, out, noise=False, **_kwargs):
    model.eval()
    ds = runner.DiskDataset(frame, "test", cache)
    batches, digest = fixed_batches(frame)
    loader = DataLoader(ds, batch_sampler=batches, num_workers=0, pin_memory=True)
    hits = {1: 0, 5: 0, 10: 0}
    predictions, actual_batches = [], []
    rng = np.random.default_rng(seed=42)
    noise_values = rng.standard_normal((len(ds), 1280, 128), dtype=np.float32) if noise else None
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            eeg, mask = batch["eeg"], batch["mask"]
            if noise:
                eeg = torch.from_numpy(noise_values[batch["row"].numpy()])
                mask = torch.ones_like(mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xe = model.encode(eeg.to(device), mask.to(device), batch["pids"].to(device))[1]
                ye = model.aligner.embed_text(
                    cache["hidden"][batch["uid"]].to(device),
                    cache["masks"][batch["uid"]].to(device),
                )
            scores = F.normalize(xe.float(), dim=-1) @ F.normalize(ye.float(), dim=-1).T
            order = scores.argsort(dim=1, descending=True)
            ids = batch["uid"].tolist()
            if len(set(ids)) != 24:
                raise ValueError("Duplicate candidate text uid in a test batch")
            actual_batches.append(batch["row"].tolist())
            for position in range(24):
                row_index = int(batch["row"][position])
                row = ds.df.iloc[row_index]
                item = {"batch": batch_index, "position": position,
                        "trial_id": row.trial_id, "stimulus_id": row.stimulus_id,
                        "text_uid": int(ids[position]), "subject": row.subject,
                        "task": row.task, "noise": noise}
                for k in hits:
                    hit = bool((order[position, :k] == position).any())
                    hits[k] += int(hit)
                    item[f"hit_top{k}"] = hit
                item["rank"] = int((order[position] == position).nonzero()[0]) + 1
                item["top5_text_uids"] = [int(ids[j]) for j in order[position, :5]]
                predictions.append(item)
    if actual_batches != batches:
        raise ValueError("DataLoader changed the fixed candidate batch order")
    result = {"split": "test", "noise": noise, "n_queries": len(predictions),
              "n_batches": len(batches), "candidate_count": 24,
              "top1": hits[1] / len(predictions), "top5": hits[5] / len(predictions),
              "top10": hits[10] / len(predictions), "omitted_test_records": len(ds)-len(predictions),
              "batch_manifest_sha256": digest,
              "scoring": "cosine similarity of L2-normalized GLIM EEG/text embeddings"}
    target = out
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (target / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")
    (target / "batches.json").write_text(json.dumps(actual_batches), encoding="utf-8")
    print("FAIR_EVAL", json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cache", "train", "eval"), default="train")
    args = parser.parse_args()
    frame, mapping = load_aligned_frame()
    batches, digest = fixed_batches(frame)
    out = GLIM / "outputs" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "mapping_audit.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps({
        "manifest": str(MANIFEST), "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "rows": len(frame), "stimuli": int(frame.stimulus_id.nunique()),
        "split_rows": frame.manifest_split.value_counts().to_dict(),
        "split_stimuli": frame.groupby("manifest_split").stimulus_id.nunique().to_dict(),
        "batch_count": len(batches), "test_queries": sum(map(len, batches)),
        "test_omitted": int(frame.phase.eq("test").sum()) - sum(map(len, batches)),
        "batch_manifest_sha256": digest,
        "text_mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
        "train": {"seed": 42, "steps": 500, "batch_size": 72, "learning_rate": 1e-4,
                  "objective": "GLIM full objective: 0.5 CMC + 0.5 frozen-FLAN generation",
                  "initialization": "random GLIM modules; prior checkpoint used for hyperparameters only"},
        "comparability": "same ZuCo2 trial rows, splits, ordered test batches and cosine metric; GLIM label text mapping includes seven audited near-matches and three normalized UID merges; objectives and architectures differ"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    runner.T5 = str(PROJECTS / "llm/flan-t5-large")
    runner.CKPT = str(GLIM / "outputs/rapid_full/checkpoint.pt")
    runner.load_df = lambda: frame
    runner.evaluate = lambda model, df, cache, device, path, noise=False, **kw: evaluate_common(
        model, frame, cache, device, path, noise=noise, **kw)
    if args.mode == "cache":
        runner.build_cache("cuda:0")
        protocol_path = out / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        cache_path = GLIM / "data/text_cache.pt"
        protocol["flan_text_cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        checkpoint_path = runner.CKPT
        source_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hyper_parameters = source_checkpoint["config"]["hyper_parameters"]
        original_torch_load = torch.load

        def load_hyperparameters(path, *load_args, **load_kwargs):
            if str(path) == str(checkpoint_path):
                return {"hyper_parameters": hyper_parameters}
            return original_torch_load(path, *load_args, **load_kwargs)

        torch.load = load_hyperparameters
        try:
            runner.run(argparse.Namespace(mode="full" if args.mode == "train" else "eval",
                                          name=OUT_NAME, seed=42, steps=500,
                                          micro_batch=12, generate=0, gradient_checkpointing=True))
        finally:
            torch.load = original_torch_load


if __name__ == "__main__":
    main()
