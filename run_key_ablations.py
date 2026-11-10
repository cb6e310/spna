"""Run the specified SPNA capacity, relation, and low-data ablations."""

from __future__ import annotations

import argparse
import json
import tempfile
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FULL_DATA_VARIANTS = (
    "d_only",
    "small_only",
    "nested_hard",
    "spna_carg",
    "uniform_smoothing",
    "shuffled_relations",
)


def run_one(
    python: str,
    code_dir: Path,
    config: Path,
    output: Path,
    variant: str,
    fraction: float,
    seed: int,
    steps: int,
    device: str,
) -> dict:
    run_dir = output / variant / f"fraction_{fraction:g}" / f"seed_{seed}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["PYTHONUNBUFFERED"] = "1"
    run_metadata = run_dir / "run.json"
    checkpoint = run_dir / "best.pt"

    if not (run_metadata.exists() and checkpoint.exists()):
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Incomplete run directory exists: {run_dir}")
        command = [
            python, "-m", "spna.train", "--config", str(config), "--variant", variant,
            "--seed", str(seed), "--train-fraction", str(fraction), "--steps", str(steps),
            "--output", str(output),
        ]
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
            try:
                subprocess.run(command, cwd=code_dir, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                log.seek(max(0, log.tell() - 6000))
                raise RuntimeError(f"Training failed for {variant}/{fraction}/{seed}:\n{log.read()}") from error
    else:
        if not run_dir.is_dir():
            raise RuntimeError(f"Run files exist without a run directory: {run_dir}")
        metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
        if (metadata["variant"], metadata["seed"], metadata["train_fraction"],
                metadata["optimizer_steps"]) != (variant, seed, fraction, steps):
            raise RuntimeError(f"Existing run metadata does not match request: {run_dir}")

    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        command = [
            python, "-m", "spna.evaluate", "--checkpoint", str(checkpoint),
            "--manifest", "/unused", "--text-cache", "/unused", "--split", "test",
            "--protocol", "both", "--output", str(metrics_path),
        ]
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        command[command.index("/unused")] = run_config["config"]["data"]["manifest"]
        command[command.index("/unused")] = run_config["config"]["data"]["text_cache"]
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
            try:
                subprocess.run(command, cwd=code_dir, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                log.seek(max(0, log.tell() - 6000))
                raise RuntimeError(f"Evaluation failed for {variant}/{fraction}/{seed}:\n{log.read()}") from error
        for suffix in ("_predictions.jsonl", "_noise_predictions.jsonl"):
            metrics_path.with_name(metrics_path.stem + suffix).unlink(missing_ok=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--devices", type=str, nargs="+", default=["1", "2", "3"])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    config = args.config.resolve()
    output = args.output.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if args.steps < 1 or not args.devices or not args.seeds:
        raise ValueError("steps, devices, and seeds must be non-empty and positive")

    jobs = [
        (variant, 1.0, seed)
        for variant in FULL_DATA_VARIANTS
        for seed in args.seeds
    ]
    jobs.extend(
        (variant, fraction, seed)
        for variant in ("d_only", "spna_carg")
        for fraction in (0.25, 0.5)
        for seed in args.seeds
    )
    failures = []
    futures = {}
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        for index, (variant, fraction, seed) in enumerate(jobs):
            device = args.devices[index % len(args.devices)]
            future = pool.submit(
                run_one, args.python, code_dir, config, output,
                variant, fraction, seed, args.steps, device,
            )
            futures[future] = (variant, fraction, seed)
        for future in as_completed(futures):
            identity = futures[future]
            try:
                result = future.result()
                print(json.dumps({"completed": identity, "full_bank": result.get("full_bank", {})}),
                      flush=True)
            except Exception as error:
                failures.append({"run": identity, "error": repr(error)})
                print(json.dumps({"failed": identity, "error": repr(error)}), flush=True)

    summaries = {}
    for variant in FULL_DATA_VARIANTS:
        metrics_by_fraction = {}
        fractions = (1.0, 0.25, 0.5) if variant in {"d_only", "spna_carg"} else (1.0,)
        for fraction in fractions:
            runs = []
            for seed in args.seeds:
                path = output / variant / f"fraction_{fraction:g}" / f"seed_{seed}" / "test_metrics.json"
                if path.exists():
                    runs.append(json.loads(path.read_text(encoding="utf-8")))
            if runs:
                full = [run["full_bank"] for run in runs]
                benchmark = [run["glim_benchmark"]["normal"] for run in runs]
                metrics_by_fraction[str(fraction)] = {
                    "n_seeds": len(runs),
                    "seeds": [run["seed"] for run in runs],
                    "full_bank_small_capacity": {
                        key: mean_std([item["small_capacity"][key] for item in full])
                        for key in ("r@1", "r@5", "mrr")
                    },
                    "full_bank_full_capacity": {
                        key: mean_std([item["full_capacity"][key] for item in full])
                        for key in ("r@1", "r@5", "mrr")
                    },
                    "glim_24_candidate_normal": {
                        key: mean_std([item[key] for item in benchmark])
                        for key in ("top1", "top5", "top10")
                    },
                }
        summaries[variant] = metrics_by_fraction

    summary = {
        "config": str(config),
        "steps": args.steps,
        "seeds": args.seeds,
        "primary_metric": "full-bank EEG-to-text small-capacity R@1",
        "results": summaries,
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
