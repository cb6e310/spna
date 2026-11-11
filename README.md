# SPNA-CARG

This folder contains a runnable PyTorch implementation of the SPNA-CARG method in `../spec/SPNA_Method_Specification_v2.1.md`. The data and training commands are intended to run on the 4090 server; no data or model files are copied into this source folder.

## Implementation scope

- Shared maximum-width EEG and text projections with independently normalized prefixes.
- Hard pair cross-entropy at both 64- and 256-dimensional capacities, plus an independently weighted 64-dimensional relation KL over non-paired candidates.
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

The text embedding command requires an already-present local model directory. It sets `local_files_only=True` and stops if the supplied path is missing; it never downloads model files.

```bash
python -m spna.embed_qwen3 \
  --manifest ./data/zuco2/zuco2_manifest.csv \
  --model-path /path/to/local/Qwen3-Embedding-8B \
  --output ./data/zuco2/qwen3_embedding_8b.npz
```

The Qwen implementation uses left padding, last-token pooling, and L2 normalization. Texts are embedded without a query-only instruction because each stimulus is an item in a symmetric text-neighborhood graph.

For the `sentence-transformers/all-mpnet-base-v2` teacher, use the locally cached ModelScope snapshot and masked mean pooling. MPNet vectors are stored unnormalized, matching the FLAN teacher convention; the learned projection and relation-target softmax handle scale. The frozen ZuCo2 cache used by the current ablation has 662 stimuli and 768 dimensions.

```bash
python -m spna.embed_qwen3 \
  --encoder mpnet \
  --manifest ./data/zuco2/zuco2_manifest.csv \
  --model-path /path/to/all-mpnet-base-v2 \
  --output ./data/zuco2/all_mpnet_base_v2_mean.npz
```

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

For the cached FLAN-T5-large teacher on the 4090 server, start from
`config.flan_t5_large.example.json`. Its `relation_temperature` and
`lambda_relation` are independent settings for the teacher distribution and
the relation-loss weight.

For all-mpnet-base-v2, start from `config.all_mpnet_base_v2.example.json` and
set the manifest and cache paths for the server. The five-seed, 50-run ablation
command used for this teacher is:

```bash
python run_key_ablations.py \
  --config /home/song/projects/spna/code/config.all_mpnet_base_v2.example.json \
  --output /home/song/projects/spna/runs/ablations_all_mpnet_20261111 \
  --python /home/song/projects/spna/.venv/bin/python \
  --seeds 42 43 44 45 46 --devices 1 2 3 --steps 500
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

The suite evaluates all six full-data variants and D-only/SPNA-CARG at 25% and 50% training stimuli. Each run stores its config, best-validation checkpoint, run metadata and compact test metrics; `ablation_summary.json` reports seed mean and sample standard deviation.

Training evaluates validation small-capacity full-bank R@1 after each epoch and saves the best epoch as `best.pt`; it does not read test metrics. Run test evaluation only after model and hyperparameters have been selected. Repeat with five seeds for the paper protocol.

## Focused three-capacity training experiment

`config.mpnet_three_capacity.example.json` configures 64D/128D/256D nested prefixes with hard-pair CE weights `[0.35, 0.15, 0.50]`. `nested_three` trains all three capacities jointly; `staged_nested_three` first trains the 256D hard-pair objective for 25% of the step budget, then fine-tunes all three capacities. Both select checkpoints by mean validation full-bank R@1 across 64D/128D/256D, and training traces record pairwise gradient cosine diagnostics every 25 optimizer steps. The targeted runner compares those two schedules with the current two-capacity `nested_hard` baseline and does not run unrelated ablations:

```bash
python run_capacity_training.py \
  --config /home/song/projects/spna/code/config.mpnet_three_capacity.example.json \
  --output /home/song/projects/spna/runs/three_capacity_20261111 \
  --python /home/song/projects/spna/.venv/bin/python \
  --seeds 42 43 44 45 46 --devices 1 2 3 --steps 1000
```

The test evaluator reports retrieval metrics separately for each saved capacity. The runner saves the aggregate in `capacity_summary.json` and keeps run-level checkpoints and traces under the output directory.

## Configuration choices still requiring validation

The example configs use `d_s=64`, `D=256`, `lambda_relation=0.1`, batch size 72, Adam with learning rate 1e-4 and no weight decay; the generic Qwen config uses `T_T=0.1`, and the FLAN-specific config uses `T_T=0.05`. The relation target `Q_neg^T` and student distribution `P_neg^64` are each normalized over non-paired candidates only; the relation KL is added separately to the two hard-pair CE terms. These values are starting points, not validated optima. Tune `lambda_relation`, `relation_temperature`, and capacity weights using validation data only. The training code selects the best validation small-capacity full-bank R@1 checkpoint after each epoch.

## ZuCo2 GLIM-style test benchmark

`spna.evaluate --split test` evaluates both ZuCo2 tasks only. It creates ordered batches of 24 unique held-out stimuli, dropping the incomplete final batch, then reports Top-1/5/10 for normal EEG, seed-42 float32 Gaussian-noise EEG, and a cyclically mismatched EEG control. It writes a JSON summary and per-query prediction JSONL files. This matches the prior GLIM evaluation structure and preprocessing; because the earlier GLIM test batches mixed ZuCo1 and ZuCo2, the ZuCo2-only batches and split are rebuilt from the ZuCo2 manifest rather than reusing mixed-dataset candidates.

The text teacher is Qwen3-Embedding-8B, frozen locally, while GLIM used cached FLAN-T5-large token states. SPNA's method and teacher are intentionally different, so the benchmark supports a controlled retrieval comparison but does not make the models' training objectives or text spaces identical.

