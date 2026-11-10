# SPNA-CARG

This folder contains a runnable PyTorch implementation of the SPNA-CARG method in `../spec/SPNA_Method_Specification_v2.1.md`. The data and training commands are intended to run on the 4090 server; no data or model files are copied into this source folder.

## Implementation scope

- Shared maximum-width EEG and text projections with independently normalized prefixes.
- Low-capacity CARG target `q_tilde = (1 - eta) * one_hot + eta * q_text` and high-capacity hard pair CE.
- True, uniform and shuffled relation targets; `d_only`, `small_only`, `nested_hard` and `spna_carg` comparisons.
- Unique-stimulus batches with one rotating EEG trial per stimulus, and stimulus-disjoint manifest validation.
- Full-bank R@1, R@5 and MRR on validation and test splits.
- Deterministic 25%, 50% and 100% training-stimulus subsets for the limited-pair study.
- Raw ZuCo 2.0 MATLAB v7.3 conversion to fixed EEG shards and an auditable manifest.

The spec does not define an EEG backbone or signal preprocessing. The engineering implementation uses a patch Transformer and matches the prior GLIM ZuCo preprocessing: reject non-finite trials and durations outside 0.5–10 seconds, remove the empty 105th channel, FFT-resample from 500 to 128 Hz, and right-pad to 1280×128 with no amplitude normalization.

The spec also names Original SPNA as a baseline but does not define its EEG-to-EEG relation target, temperature or loss weight. That baseline is not included until those details are fixed. The model comparison commands below cover the fully specified controls and proposed method.

## Data format

`prepare_zuco2.py` reads the `task1 - NR/Matlab files` and `task2 - TSR/Matlab files` directories under a raw ZuCo 2.0 root. It writes one `[trials, time, channels]` NumPy shard per source file, `zuco2_manifest.csv`, and `audit.json`. Exact repeated text is assigned a deterministic SHA-256 stimulus ID; all trials with that ID share one split.

The frozen text cache is an `.npz` file with:

- `stimulus_ids`: one string ID per unique stimulus, matching the manifest.
- `embeddings`: a finite float array of shape `[unique_stimuli, text_dim]`.

The cache includes all split stimuli for retrieval, while training relation targets are constructed only from the unique training stimuli in each batch.

## Prepare and cache

Run these commands from this directory on the 4090 server inside the project-level `.venv`; `requirements.txt` records the imports the code expects. The raw data root is the relocated `/home/song/projects/datasets/zuco_2.0`.

```bash
python prepare_zuco2.py \
  --zuco-root /home/song/projects/trust_align/01_data_protocol/datasets/zuco_2.0 \
  --output ./data/zuco2 --seed 42
```

The text embedding command requires an already-present local Qwen3-Embedding-8B directory. It sets `local_files_only=True` and stops if the supplied path is missing; it never downloads model files.

```bash
python -m spna.embed_qwen3 \
  --manifest ./data/zuco2/zuco2_manifest.csv \
  --model-path /path/to/local/Qwen3-Embedding-8B \
  --output ./data/zuco2/qwen3_embedding_8b.npz
```

The Qwen implementation uses left padding, last-token pooling, and L2 normalization. Texts are embedded without a query-only instruction because each stimulus is an item in a symmetric text-neighborhood graph.

## Train and evaluate

Copy `config.example.json` to `config.json`, set the data paths, then run from this directory:

```bash
python -m spna.train --config config.json --variant spna_carg --seed 42 --steps 500
python -m spna.evaluate \
  --checkpoint ./runs/spna_carg/fraction_1/seed_42/best.pt \
  --manifest ./data/zuco2/zuco2_manifest.csv \
  --text-cache ./data/zuco2/qwen3_embedding_8b.npz \
  --split test --protocol both \
  --output ./runs/spna_carg/fraction_1/seed_42/test.json
```

The default `benchmark` protocol preserves the fixed 24-candidate GLIM-style evaluation. Use `--protocol full_bank` for all unique test stimuli, or `--protocol both` to report both protocols in one JSON file. Full-bank results include small- and full-capacity R@1/R@5/MRR; the 24-candidate benchmark remains a separate secondary comparison.

Supported variants are `d_only`, `small_only`, `nested_hard`, `spna_carg`, `uniform_smoothing` and `shuffled_relations`. Use the same seed and training fraction for each comparison. The 25%, 50% and 100% subsets are nested and selected deterministically by seed:

```bash
python -m spna.train --config config.json --variant spna_carg --seed 42 --train-fraction 0.25
python -m spna.train --config config.json --variant spna_carg --seed 42 --train-fraction 0.5
```

Run the complete key ablation suite (five seeds, 500 steps) with:

```bash
python run_key_ablations.py \
  --config /home/song/projects/spna/configs/zuco2_flan_t5_large.json \
  --output /home/song/projects/spna/runs/ablations_flan \
  --python /home/song/projects/spna/.venv/bin/python \
  --seeds 42 43 44 45 46 --devices 0 1 2 --steps 500
```

The suite evaluates all six full-data variants and D-only/SPNA-CARG at 25% and 50% training stimuli. Each run stores its config, fixed-step checkpoint, run metadata and compact test metrics; `ablation_summary.json` reports seed mean and sample standard deviation.

Training selects the checkpoint on validation small-capacity full-bank R@1 and does not read test metrics. Run test evaluation only after model and hyperparameters have been selected. Repeat with five seeds for the paper protocol.

## Configuration choices still requiring validation

The example config uses `d_s=64`, `D=256`, `eta=0.2`, batch size 72, Adam with learning rate 1e-4 and no weight decay, matching the prior GLIM rapid-training budget. These are controlled starting points, not validated optima. Search `eta` over 0.1/0.2/0.3 and tune temperatures and capacity weights on validation only. The training code's initial variant weights mirror the spec's suggested 0.5/0.5; adjust them only through validation runs.

## ZuCo2 GLIM-style test benchmark

`spna.evaluate --split test` evaluates both ZuCo2 tasks only. It creates ordered batches of 24 unique held-out stimuli, dropping the incomplete final batch, then reports Top-1/5/10 for normal EEG, seed-42 float32 Gaussian-noise EEG, and a cyclically mismatched EEG control. It writes a JSON summary and per-query prediction JSONL files. This matches the prior GLIM evaluation structure and preprocessing; because the earlier GLIM test batches mixed ZuCo1 and ZuCo2, the ZuCo2-only batches and split are rebuilt from the ZuCo2 manifest rather than reusing mixed-dataset candidates.

The text teacher is Qwen3-Embedding-8B, frozen locally, while GLIM used cached FLAN-T5-large token states. SPNA's method and teacher are intentionally different, so the benchmark supports a controlled retrieval comparison but does not make the models' training objectives or text spaces identical.

