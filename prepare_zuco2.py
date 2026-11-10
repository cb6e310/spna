"""Convert raw ZuCo 2.0 MATLAB v7.3 trials into fixed-length EEG shards.

This script is intended to run on the machine that holds the raw dataset.
It does not download data, models, or dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample


def stimulus_key(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def read_ref(handle: h5py.File, dataset: h5py.Dataset, index: int):
    if dataset.ndim == 1:
        value = dataset[index]
    elif dataset.shape[0] == 1:
        value = dataset[0, index]
    else:
        value = dataset[index, 0]
    refs = np.asarray(value, dtype=object).reshape(-1)
    for ref in refs:
        if isinstance(ref, h5py.Reference) and ref:
            return handle[ref]
    raise ValueError(f"Missing HDF5 reference at {dataset.name}[{index}]")


def decode_content(handle: h5py.File, content: h5py.Dataset, index: int) -> str:
    value = read_ref(handle, content, index)
    codes = np.asarray(value).reshape(-1)
    return "".join(chr(int(code)) for code in codes if int(code) != 0).strip()


def load_trial(
    eeg_leaf: h5py.Dataset,
    channels: int,
    source_rate: int,
    target_rate: int,
    max_samples: int,
) -> tuple[np.ndarray, int] | None:
    raw = np.asarray(eeg_leaf, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        return None
    if raw.shape[1] in (channels, channels + 1):
        samples_by_channel = raw
    elif raw.shape[0] in (channels, channels + 1):
        samples_by_channel = raw.T
    else:
        return None
    # Match the GLIM preprocessing: source MAT files have 105 channels, of
    # which the final channel is empty; retain the 104 EEG channels and pad
    # them to the model's 128-channel input.
    if samples_by_channel.shape[1] == channels + 1:
        if np.any(samples_by_channel[:, -1]):
            return None
        samples_by_channel = samples_by_channel[:, :channels]
    if samples_by_channel.shape[1] != channels:
        return None
    source_samples = samples_by_channel.shape[0]
    if source_samples < round(0.5 * source_rate) or source_samples > round(10.0 * source_rate):
        return None
    target_samples = int(source_samples * target_rate / source_rate)
    if target_samples < 1 or target_samples > max_samples:
        return None
    # scipy.signal.resample is the FFT resampling operation used by GLIM.
    signal = resample(samples_by_channel, target_samples, axis=0)
    valid_samples = int(signal.shape[0])
    if valid_samples < 1 or valid_samples > max_samples:
        return None
    padded = np.zeros((max_samples, 128), dtype=np.float32)
    padded[:valid_samples, :channels] = signal
    return padded, valid_samples


def collect_records(root: Path, shard_dir: Path, channels: int, source_rate: int,
                    target_rate: int, max_samples: int) -> list[dict]:
    domains = ["task1 - NR/Matlab files", "task2 - TSR/Matlab files"]
    files = sorted(path for domain in domains for path in (root / domain).glob("results*.mat"))
    if not files:
        raise FileNotFoundError(f"No ZuCo 2.0 results*.mat files under {root}")
    records = []
    for source in files:
        task = "task2" if "task1" in source.as_posix() else "task3"
        subject = re.sub(r"^results|_(NR|TSR)$", "", source.stem)
        with h5py.File(source, "r") as handle:
            sentence_data = handle["sentenceData"]
            raw_data = sentence_data["rawData"]
            content = sentence_data["content"]
            count = max(raw_data.shape)
            signals, local_rows = [], []
            for index in range(count):
                text = decode_content(handle, content, index)
                eeg_leaf = read_ref(handle, raw_data, index)
                prepared = load_trial(eeg_leaf, channels, source_rate, target_rate, max_samples)
                if not text or prepared is None:
                    continue
                signal, valid_samples = prepared
                local_index = len(signals)
                signals.append(signal)
                local_rows.append({
                    "trial_id": f"ZuCo2_{task}:{subject}:{index + 1}",
                    "stimulus_id": stimulus_key(text),
                    "text": text,
                    "dataset": "ZuCo2",
                    "subject_id": subject,
                    "task": task,
                    "source_index": index,
                    "valid_samples": valid_samples,
                    "eeg_index": local_index,
                })
            if signals:
                shard = shard_dir / f"{source.stem}.eeg.npy"
                np.save(shard, np.stack(signals), allow_pickle=False)
                for row in local_rows:
                    row["eeg_file"] = str(shard.resolve())
                    records.append(row)
        print(f"PREP {source.name} accepted={len(local_rows)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zuco-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channels", type=int, default=104)
    parser.add_argument("--source-rate", type=int, default=500)
    parser.add_argument("--target-rate", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("Train and validation ratios must be positive and sum to less than 1")
    args.output.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output / "shards"
    shard_dir.mkdir(exist_ok=True)
    max_samples = round(args.target_rate * args.max_seconds)
    records = collect_records(
        args.zuco_root, shard_dir, args.channels, args.source_rate, args.target_rate, max_samples
    )

    stimuli = sorted({row["stimulus_id"] for row in records})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(stimuli)
    train_end = round(len(stimuli) * args.train_ratio)
    validation_end = round(len(stimuli) * (args.train_ratio + args.validation_ratio))
    split_by_stimulus = {
        stimulus: "train" if index < train_end else
        "validation" if index < validation_end else "test"
        for index, stimulus in enumerate(stimuli)
    }
    for row in records:
        row["split"] = split_by_stimulus[row["stimulus_id"]]

    manifest = args.output / "zuco2_manifest.csv"
    fields = ["trial_id", "stimulus_id", "text", "split", "eeg_file", "eeg_index",
              "valid_samples", "dataset", "subject_id", "task", "source_index"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    audit = {
        "records": len(records),
        "stimuli": len(stimuli),
        "split_stimuli": {split: sum(value == split for value in split_by_stimulus.values())
                           for split in ("train", "validation", "test")},
        "split_records": {split: sum(row["split"] == split for row in records)
                          for split in ("train", "validation", "test")},
        "stimulus_overlap": 0,
        "eeg_preprocessing": {
            "source_rate_hz": args.source_rate,
            "target_rate_hz": args.target_rate,
            "channels": args.channels,
            "max_seconds": args.max_seconds,
            "model_channels": 128,
            "resampling": "GLIM-matched scipy.signal.resample; remove empty channel 105; right-pad to 1280x128; no amplitude normalization",
        },
        "seed": args.seed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

