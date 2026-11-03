"""Execute the preregistered two-run local study and a single final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def write_or_verify(path: Path, value: dict) -> None:
    """Reuse an identical frozen decision without replacing its original bytes."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing study evidence differs: {path}")
    else:
        write_new(path, value)


def verify_evaluation(
    path: Path,
    selected: dict,
    manifest: dict,
    manifest_hash: str,
    device: str,
    split: str,
    threshold: float | None = None,
) -> dict:
    report = read(path / "report.json")
    expected = {
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "training_run_id": selected["run_id"],
        "manifest_sha256": manifest_hash,
        "dataset_id": manifest["dataset_id"],
        "split": split,
        "device": device,
        "final_holdout": split == "test",
        "threshold_selected_here_on_validation": split == "val",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing {split} evaluation identity or protocol differs")
    if report.get("images") != sum(row["split"] == split for row in manifest["records"]):
        raise ValueError(f"Existing {split} evaluation split size differs")
    if [{"id": row["id"], "name": row["name"]} for row in report["per_class"]] != manifest[
        "classes"
    ]:
        raise ValueError(f"Existing {split} evaluation taxonomy differs")
    if split == "val":
        threshold = max(report["threshold_sweep"], key=lambda row: (row["f1"], row["threshold"]))[
            "threshold"
        ]
        if not (path / "robustness.json").is_file():
            raise ValueError("Completed validation lacks its robustness evidence")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid frozen operating threshold")
    if report.get("threshold") != threshold:
        raise ValueError(f"Existing {split} evaluation threshold differs")
    return report


def verify_export(
    path: Path,
    selected: dict,
    manifest: dict,
    manifest_hash: str,
    release: str,
    threshold: float,
    holdout: dict,
    licensing: dict,
):
    import torch
    from inspection_dl.bundle import validate_bundle

    metadata = validate_bundle(path)
    expected = {
        "dataset_manifest_sha256": manifest_hash,
        "training_run_id": selected["run_id"],
        "dataset_id": manifest["dataset_id"],
        "taxonomy_version": manifest["taxonomy_version"],
        "model_version": release,
        "confidence_threshold": threshold,
        "licensing": licensing,
    }
    if (
        any(getattr(metadata, key) != value for key, value in expected.items())
        or [item.model_dump() for item in metadata.classes] != manifest["classes"]
        or read(path / "evaluation.json") != holdout
    ):
        raise ValueError("Existing bundle differs from the frozen study release")
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    if weights.keys() != checkpoint["model"].keys() or any(
        not torch.equal(weights[key], checkpoint["model"][key]) for key in weights
    ):
        raise ValueError("Existing bundle tensors differ from the selected checkpoint")
    return metadata


def verify_benchmark(path: Path, metadata, manifest: dict, manifest_hash: str) -> None:
    report = read(path)
    expected = {
        key: getattr(metadata, key)
        for key in (
            "artifact_digest",
            "weights_sha256",
            "model_version",
            "taxonomy_version",
            "dataset_id",
        )
    }
    sample_ids = sorted(row["id"] for row in manifest["records"] if row["split"] == "val")[:100]
    if (
        any(report.get(key) != value for key, value in expected.items())
        or report.get("manifest_sha256") != manifest_hash
        or report.get("classes") != manifest["classes"]
        or report.get("split") != "val"
        or report.get("sample_ids") != sample_ids
        or not report.get("benchmark_source", {}).get("package_source_sha256")
    ):
        raise ValueError("Existing benchmark release, subset or source identity differs")
    configurations = set()
    unavailable = set()
    for result in report["results"]:
        device = result.get("device")
        if result.get("unavailable"):
            if device != "cuda" or device in unavailable:
                raise ValueError("Invalid benchmark unavailable-device record")
            unavailable.add(device)
            continue
        key = (device, result.get("batch_size"))
        if key in configurations or device not in {"cpu", "cuda"} or key[1] not in {1, 4}:
            raise ValueError("Unexpected benchmark configuration")
        configurations.add(key)
        quality = result["quality_on_identical_validation_subset"]
        if (
            any(result.get(name) != value for name, value in expected.items())
            or quality.get("images") != len(sample_ids)
            or result.get("images") != len(sample_ids)
            or quality.get("threshold") != metadata.confidence_threshold
        ):
            raise ValueError(
                "Existing benchmark configuration identity or quality protocol differs"
            )
    expected_configs = {
        (device, batch) for device in {"cpu", "cuda"} - unavailable for batch in (1, 4)
    }
    if configurations != expected_configs or not configurations:
        raise ValueError("Benchmark is incomplete")


def verify_completed_run(output: Path, expected: dict) -> dict:
    record = read(output / "run.json")
    if not record.get("completed") or any(
        record["config"].get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Candidate is incomplete or differs from the preregistered configuration")
    if digest(output / "best.pt") != record["best_checkpoint_sha256"]:
        raise ValueError("Candidate checkpoint integrity mismatch")
    return record


def verify_initialization(paths: list[Path]) -> dict:
    import torch

    initialized = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    reference = initialized[0]
    for candidate in initialized[1:]:
        if (
            candidate["classes"] != reference["classes"]
            or candidate["model"].keys() != reference["model"].keys()
        ):
            raise ValueError("Candidate initial taxonomy or parameter names differ")
        if any(
            not torch.equal(reference["model"][key], candidate["model"][key])
            for key in reference["model"]
        ):
            raise ValueError("Candidate initialization differs")
    checksum = hashlib.sha256()
    for key, tensor in sorted(reference["model"].items()):
        checksum.update(key.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        checksum.update(str(tuple(tensor.shape)).encode() + b"\0")
        checksum.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "identical_initial_model_tensors": True,
        "tensor_count": len(reference["model"]),
        "initial_tensor_digest": checksum.hexdigest(),
        "candidates": [path.parent.name for path in paths],
        "definition": "Exact CPU equality of every initialized parameter and buffer; digest includes sorted name, dtype, shape and bytes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pcb-ind-v4.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/pcb-ind"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--amp", action="store_true", help="explicit mixed-precision experiment")
    parser.add_argument("--study-id", default="local-study")
    parser.add_argument("--release", default="pcb-ind-ssdlite-v1")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="verify and finish an existing study whose two training runs completed",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("epochs, batch size and learning rate must be positive")
    # IDs name local outputs only; never permit path traversal through an ID.
    for name in (args.study_id, args.release):
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name):
            parser.error(
                "study and release IDs require lowercase letters, digits, hyphens or underscores"
            )
    reports = Path("reports") / args.study_id
    bundle = Path("artifacts") / args.release
    release_record = Path("releases") / args.release
    if not args.finalize_only:
        outputs = [reports, bundle, release_record] + [
            Path("runs") / f"{args.study_id}-{variant}" for variant in ("no-flip", "flip")
        ]
        if any(path.exists() for path in outputs):
            raise FileExistsError("Use fresh study and release IDs; an output already exists")
    release_binding = reports / "release-binding.json"
    if release_binding.exists() and read(release_binding).get("release") != args.release:
        raise ValueError("Finalization differs from the frozen release binding")
    manifest = read(args.manifest)
    manifest_hash = digest(args.manifest)
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
        "manifest_sha256": manifest_hash,
        "release": args.release,
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
    if args.finalize_only:
        previous = read(reports / "preregistration.json")
        for key in (
            "study_id",
            "manifest_sha256",
            "epochs_per_run",
            "batch_size",
            "seed",
            "architecture",
            "device",
            "learning_rate",
            "amp",
        ):
            if previous.get(key) != preregistration[key]:
                raise ValueError(f"Finalization differs from preregistered {key}")
        if "release" in previous and previous["release"] != args.release:
            raise ValueError("Finalization differs from preregistered release")
    else:
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
        if not args.finalize_only:
            run("inspection_dl.train", *common, *options)
        expected = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": 42,
            "learning_rate": args.learning_rate,
            "amp": args.amp,
            "manifest_sha256": manifest_hash,
            "horizontal_flip_probability": 0.5 if variant == "flip" else 0.0,
            "freeze_backbone": False,
            "pretrained_initialization": True,
            "architecture": "ssdlite320_mobilenet_v3_large",
            "preprocessing": "ssdlite320_rgb_unit_v1",
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "scheduler": "CosineAnnealingLR",
        }
        record = verify_completed_run(output, expected)
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
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (
            "run.json",
            "history.json",
            "prior.json",
            "baseline_validation.json",
            "best_validation.json",
            "best_validation_predictions.json",
        ):
            if (destination / filename).exists():
                if digest(output / filename) != digest(destination / filename):
                    raise ValueError("Existing experiment evidence differs; refusing overwrite")
            else:
                shutil.copyfile(output / filename, destination / filename)
    parity = verify_initialization(
        [Path(item["checkpoint"]).parent / "initialized.pt" for item in candidates]
    )
    write_or_verify(reports / "initialization-parity.json", parity)
    selected = max(candidates, key=lambda item: item["validation_map50_95"])
    if not (reports / "validation/report.json").exists():
        if (reports / "selection.json").exists() or (reports / "holdout").exists():
            raise ValueError("Frozen selection exists but completed validation is missing")
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
    validation = verify_evaluation(
        reports / "validation", selected, manifest, manifest_hash, args.device, "val"
    )
    threshold = validation["threshold"]
    if not (reports / "selection.json").exists() and (
        (reports / "holdout-started.json").exists() or (reports / "holdout").exists()
    ):
        raise ValueError(
            "Holdout evidence exists without its frozen selection; refusing reselection"
        )
    write_or_verify(
        reports / "selection.json",
        {
            "candidates": candidates,
            "selected": selected,
            "threshold": threshold,
            "manifest_sha256": manifest_hash,
            "decision_basis": "validation only; this file written before final holdout access",
        },
    )
    holdout_started = reports / "holdout-started.json"
    holdout_identity = {"selection_sha256": digest(reports / "selection.json")}
    if not (reports / "holdout/report.json").exists():
        if holdout_started.exists() or (reports / "holdout").exists():
            raise ValueError(
                "Holdout evaluation started without a completed report; automatic repetition is forbidden"
            )
        write_new(holdout_started, holdout_identity)
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
    if holdout_started.exists():
        write_or_verify(holdout_started, holdout_identity)
    holdout = verify_evaluation(
        reports / "holdout", selected, manifest, manifest_hash, args.device, "test", threshold
    )
    licensing = read(Path("configs/licensing.json"))
    licensing["pretrained_initialization"] = read(Path(selected["checkpoint"]).parent / "run.json")[
        "pretrained"
    ]
    write_or_verify(reports / "licensing.json", licensing)
    if not bundle.exists():
        run(
            "inspection_dl.bundle",
            selected["checkpoint"],
            str(bundle),
            "--manifest",
            str(args.manifest),
            "--report",
            str(reports / "holdout/report.json"),
            "--model-version",
            args.release,
            "--licensing",
            str(reports / "licensing.json"),
            "--threshold",
            str(threshold),
        )
    metadata = verify_export(
        bundle, selected, manifest, manifest_hash, args.release, threshold, holdout, licensing
    )
    write_or_verify(
        release_binding,
        {
            "release": args.release,
            "artifact_digest": metadata.artifact_digest,
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "manifest_sha256": manifest_hash,
            "preregistration_sha256": digest(reports / "preregistration.json"),
            "legacy_preregistration": "release" not in read(reports / "preregistration.json"),
        },
    )
    release_record.mkdir(parents=True, exist_ok=True)
    for filename in ("metadata.json", "evaluation.json", "weights.pt"):
        target = release_record / filename
        if target.exists():
            if digest(target) != digest(bundle / filename):
                raise ValueError("Existing release file differs from the verified study bundle")
        else:
            with target.open("xb") as stream:
                stream.write((bundle / filename).read_bytes())
    benchmark_path = reports / "benchmark-release.json"
    if not benchmark_path.exists():
        run(
            "inspection_dl.benchmark",
            "--bundle",
            str(bundle),
            *common,
            "--output",
            str(benchmark_path),
            "--samples",
            "100",
            "--devices",
            "cpu",
            "cuda",
            "--batch-sizes",
            "1",
            "4",
        )
    verify_benchmark(benchmark_path, metadata, manifest, manifest_hash)
    write_or_verify(
        reports / "completed.json",
        {
            "selection_sha256": digest(reports / "selection.json"),
            "holdout_report_sha256": digest(reports / "holdout/report.json"),
            "artifact_digest": metadata.artifact_digest,
            "benchmark_sha256": digest(benchmark_path),
        },
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
