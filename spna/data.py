"""Manifest-backed ZuCo EEG dataset and unique-stimulus batch sampler."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_COLUMNS = {
    "trial_id", "stimulus_id", "text", "split", "eeg_file", "eeg_index", "valid_samples"
}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    seen_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen_splits[row["stimulus_id"]].add(row["split"])
    overlap = {stimulus: splits for stimulus, splits in seen_splits.items() if len(splits) > 1}
    if overlap:
        raise ValueError(f"Stimulus leakage across splits; examples: {list(overlap.items())[:5]}")
    return rows


def load_text_cache(path: str | Path) -> tuple[dict[str, int], torch.Tensor]:
    with np.load(path, allow_pickle=False) as cache:
        if not {"stimulus_ids", "embeddings"}.issubset(cache.files):
            raise ValueError("Text cache requires stimulus_ids and embeddings arrays")
        stimulus_ids = cache["stimulus_ids"].astype(str).tolist()
        embeddings = torch.from_numpy(np.asarray(cache["embeddings"], dtype=np.float32))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(stimulus_ids):
        raise ValueError("Text cache embeddings must have shape [stimuli, feature_dim]")
    if len(set(stimulus_ids)) != len(stimulus_ids):
        raise ValueError("Text cache contains duplicate stimulus_ids")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Text cache contains non-finite values")
    return {stimulus: i for i, stimulus in enumerate(stimulus_ids)}, embeddings


class ZuCoManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        split: str,
        text_index: dict[str, int],
        text_embeddings: torch.Tensor,
        channels: int,
        max_samples: int,
    ):
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split!r} rows in manifest")
        self.split = split
        self.text_index = text_index
        self.text_embeddings = text_embeddings
        self.channels = channels
        self.max_samples = max_samples
        self._arrays: dict[str, np.ndarray] = {}
        for row in self.rows:
            if row["stimulus_id"] not in text_index:
                raise ValueError(f"No frozen text feature for {row['stimulus_id']}")
        self.stimulus_to_rows: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.stimulus_to_rows[row["stimulus_id"]].append(index)
        self.stimulus_ids = list(self.stimulus_to_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        eeg_file = str(Path(row["eeg_file"]))
        if eeg_file not in self._arrays:
            self._arrays[eeg_file] = np.load(eeg_file, mmap_mode="r")
        eeg = np.asarray(self._arrays[eeg_file][int(row["eeg_index"])], dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[1] != self.channels or eeg.shape[0] > self.max_samples:
            raise ValueError(f"Invalid EEG shape in {eeg_file}: {eeg.shape}")
        valid_samples = int(row["valid_samples"])
        if not 0 < valid_samples <= eeg.shape[0]:
            raise ValueError(f"Invalid valid_samples for trial {row['trial_id']}")
        mask = torch.arange(eeg.shape[0]) < valid_samples
        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "mask": mask,
            "text": self.text_embeddings[self.text_index[row["stimulus_id"]]],
            "stimulus_id": row["stimulus_id"],
            "row_index": index,
        }


class UniqueStimulusBatchSampler(Sampler[list[int]]):
    """One EEG trial per stimulus per epoch, rotating repeated records."""

    def __init__(self, dataset: ZuCoManifestDataset, batch_size: int, seed: int):
        if batch_size < 3:
            raise ValueError("batch_size must be at least 3 for CARG relation targets")
        if len(dataset.stimulus_ids) < 3:
            raise ValueError("Training split must contain at least three unique stimuli")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        stimuli = self.dataset.stimulus_ids.copy()
        rng.shuffle(stimuli)
        batches: list[list[int]] = []
        current: list[int] = []
        for stimulus in stimuli:
            current.append(rng.choice(self.dataset.stimulus_to_rows[stimulus]))
            if len(current) == self.batch_size:
                batches.append(current)
                current = []
        if len(current) >= 3:
            batches.append(current)
        elif current and batches:
            batches[-1].extend(current)
        elif current:
            raise ValueError("Cannot form a non-degenerate CARG batch")
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        count = len(self.dataset.stimulus_ids)
        return count // self.batch_size + int(count % self.batch_size >= 3)

