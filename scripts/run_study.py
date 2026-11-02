"""Execute the preregistered two-run local study and a single final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pcb-ind-v4.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/pcb-ind"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--amp", action="store_true", help="explicit mixed-precision experiment")
    parser.add_argument("--study-id", default="rtx4060-study-v1")
    parser.add_argument("--release", default="pcb-ind-ssdlite-v1")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    # IDs name local outputs only; never permit path traversal through an ID.
    for name in (args.study_id, args.release):
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name):
            parser.error(
                "study and release IDs require lowercase letters, digits, hyphens or underscores"
            )
    reports = Path("reports") / args.study_id
    if reports.exists():
        raise FileExistsError("Use a fresh study ID; recorded studies are never overwritten")
    common = ["--manifest", str(args.manifest), "--data-root", str(args.data_root)]

    def run(module: str, *arguments: str):
        command = [sys.executable, "-m", module, *arguments]
        print("RUN", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)

    preregistration = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "study_id": args.study_id,
        "manifest_sha256": digest(args.manifest),
        "epochs_per_run": args.epochs,
        "learning_rate": args.learning_rate,
        "amp": args.amp,
        "batch_size": args.batch_size,
        "seed": 42,
        "architecture": "ssdlite320_mobilenet_v3_large",
        "device": args.device,
        "comparison": "identical initialization/seed/budget; no flip vs horizontal flip p=.5",
        "selection": "highest best-validation AP50:.95; no-flip wins a tie",
        "threshold": "highest micro F1 at IoU .5 on validation; highest threshold wins a tie",
        "holdout_policy": "selected checkpoint and threshold frozen before single test evaluation",
        "optimization": "batch size 1 vs 4 on same validation subset, CPU and GPU; quality measured for both",
        "budget": "two bounded runs; no test-guided retraining or unbounded search",
    }
    write_new(reports / "preregistration.json", preregistration)
    candidates = []
    for variant in ("no-flip", "flip"):
        output = Path("runs") / f"{args.study_id}-{variant}"
        options = [
            "--output",
            str(output),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            "42",
            "--device",
            args.device,
            "--lr",
            str(args.learning_rate),
        ]
        if args.amp:
            options.append("--amp")
        if variant == "flip":
            options.append("--horizontal-flip")
        run("inspection_dl.train", *common, *options)
        record = read(output / "run.json")
        candidates.append(
            {
                "variant": variant,
                "run_id": output.name,
                "validation_map50_95": record["best_validation_map50_95"],
                "checkpoint": str(output / "best.pt"),
                "checkpoint_sha256": digest(output / "best.pt"),
            }
        )
        destination = reports / "experiments" / variant
        destination.mkdir(parents=True)
        for filename in (
            "run.json",
            "history.json",
            "prior.json",
            "baseline_validation.json",
            "best_validation.json",
            "best_validation_predictions.json",
        ):
            shutil.copyfile(output / filename, destination / filename)
    selected = max(candidates, key=lambda item: item["validation_map50_95"])
    run(
        "inspection_dl.evaluate",
        *common,
        "--checkpoint",
        selected["checkpoint"],
        "--output",
        str(reports / "validation"),
        "--split",
        "val",
        "--select-threshold",
        "--device",
        args.device,
        "--robustness",
    )
    threshold = read(reports / "validation" / "report.json")["threshold"]
    write_new(
        reports / "selection.json",
        {
            "candidates": candidates,
            "selected": selected,
            "threshold": threshold,
            "manifest_sha256": digest(args.manifest),
            "decision_basis": "validation only; this file written before final holdout access",
        },
    )
    run(
        "inspection_dl.evaluate",
        *common,
        "--checkpoint",
        selected["checkpoint"],
        "--output",
        str(reports / "holdout"),
        "--split",
        "test",
        "--final-holdout",
        "--threshold",
        str(threshold),
        "--device",
        args.device,
    )
    licensing = read(Path("configs/licensing.json"))
    licensing["pretrained_initialization"] = read(Path(selected["checkpoint"]).parent / "run.json")[
        "pretrained"
    ]
    write_new(reports / "licensing.json", licensing)
    bundle = Path("artifacts") / args.release
    run(
        "inspection_dl.bundle",
        selected["checkpoint"],
        str(bundle),
        "--manifest",
        str(args.manifest),
        "--report",
        str(reports / "holdout" / "report.json"),
        "--model-version",
        args.release,
        "--licensing",
        str(reports / "licensing.json"),
        "--threshold",
        str(threshold),
    )
    release_record = Path("releases") / args.release
    release_record.mkdir(parents=True, exist_ok=False)
    for filename in ("metadata.json", "evaluation.json"):
        shutil.copyfile(bundle / filename, release_record / filename)
    run(
        "inspection_dl.benchmark",
        "--bundle",
        str(bundle),
        *common,
        "--output",
        str(reports / "benchmark.json"),
        "--samples",
        "100",
        "--devices",
        "cpu",
        "cuda",
        "--batch-sizes",
        "1",
        "4",
    )
    print(
        json.dumps(
            {
                "study": str(reports),
                "bundle": str(bundle),
                "selected": selected,
                "threshold": threshold,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
