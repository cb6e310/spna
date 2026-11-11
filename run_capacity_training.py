"""Run only the focused two-layer versus three-layer capacity experiment."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VARIANTS = ("nested_hard", "nested_three", "staged_nested_three")


def run_one(python: str, code_dir: Path, config: Path, output: Path,
            variant: str, seed: int, steps: int, device: str) -> dict:
    run_dir = output / variant / "fraction_1" / f"seed_{seed}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["PYTHONUNBUFFERED"] = "1"
    checkpoint = run_dir / "best.pt"
    metadata = run_dir / "run.json"
    if not (checkpoint.exists() and metadata.exists()):
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Incomplete run directory exists: {run_dir}")
        subprocess.run([
            python, "-m", "spna.train", "--config", str(config),
            "--variant", variant, "--seed", str(seed), "--steps", str(steps),
            "--output", str(output),
        ], cwd=code_dir, env=env, check=True)
    run = json.loads(metadata.read_text(encoding="utf-8"))
    if (run["variant"], run["seed"], run["optimizer_steps"]) != (variant, seed, steps):
        raise RuntimeError(f"Run metadata mismatch: {metadata}")
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg = run_config["config"]["data"]
        subprocess.run([
            python, "-m", "spna.evaluate", "--checkpoint", str(checkpoint),
            "--manifest", cfg["manifest"], "--text-cache", cfg["text_cache"],
            "--split", "test", "--protocol", "both", "--output", str(metrics_path),
        ], cwd=code_dir, env=env, check=True)
        for suffix in ("_predictions.jsonl", "_noise_predictions.jsonl"):
            metrics_path.with_name(metrics_path.stem + suffix).unlink(missing_ok=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--devices", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    jobs = [(variant, seed) for variant in VARIANTS for seed in args.seeds]
    results: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = {
            pool.submit(
                run_one, args.python, code_dir, args.config.resolve(), args.output.resolve(),
                variant, seed, args.steps, args.devices[index % len(args.devices)]
            ): (variant, seed)
            for index, (variant, seed) in enumerate(jobs)
        }
        for future in as_completed(futures):
            identity = futures[future]
            try:
                result = future.result()
                results[identity[0]].append(result)
                print(json.dumps({"completed": identity, "full_bank": result.get("full_bank", {})}), flush=True)
            except Exception as error:
                failures.append({"run": identity, "error": repr(error)})
                print(json.dumps({"failed": identity, "error": repr(error)}), flush=True)

    summary = {"steps": args.steps, "seeds": args.seeds, "variants": VARIANTS, "failures": failures}
    for variant, runs in results.items():
        if not runs:
            continue
        dimension_keys = sorted({
            dimension for run in runs
            for dimension in run["full_bank"]["capacity_metrics"]
        }, key=int)
        summary[variant] = {
            "n_seeds": len(runs),
            "capacity_metrics": {
                dimension: {
                    metric: {
                        "mean": statistics.mean(
                            run["full_bank"]["capacity_metrics"][dimension][metric]
                            for run in runs
                        ),
                        "std": statistics.stdev(
                            run["full_bank"]["capacity_metrics"][dimension][metric]
                            for run in runs
                        ) if len(runs) > 1 else 0.0,
                    } for metric in ("r@1", "r@5", "mrr")
                } for dimension in dimension_keys
            },
            "checkpoint_epochs": [
                {"seed": run["seed"], "checkpoint_epoch": run["checkpoint_epoch"]}
                for run in runs
            ],
        }
        cosines: dict[str, list[float]] = defaultdict(list)
        elapsed = []
        for seed in args.seeds:
            run_dir = args.output / variant / "fraction_1" / f"seed_{seed}"
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            elapsed.append(float(metadata["elapsed_seconds"]))
            for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                for key, value in (event.get("gradient_conflicts") or {}).items():
                    if key.startswith("cos_"):
                        cosines[key].append(float(value))
        summary[variant]["elapsed_seconds"] = {
            "mean": statistics.mean(elapsed),
            "std": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
        }
        summary[variant]["gradient_cosines"] = {
            key: {"mean": statistics.mean(values),
                  "fraction_negative": sum(value < 0 for value in values) / len(values),
                  "n": len(values)}
            for key, values in cosines.items()
        }
    (args.output / "capacity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
