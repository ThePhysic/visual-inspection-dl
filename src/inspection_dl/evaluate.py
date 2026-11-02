"""Auditable evaluation, baseline, failure examples and robustness reports."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .bundle import load_bundle
from .metrics import (
    box_iou,
    confusion_matrix,
    evaluate_detections,
    grouped_bootstrap,
    threshold_sweep,
)
from .model import image_tensor, load_checkpoint_model
from .train import (
    atomic_json,
    file_digest,
    load_manifest,
    predict_records,
    prior_predictions,
    safe_image_path,
    source_provenance,
    train_prior,
)


def per_image_errors(records: list[dict], predictions: dict, threshold: float) -> list[dict]:
    results = []
    for record in records:
        prediction = predictions[record["id"]]
        remaining = set(range(len(record["boxes"])))
        tp, fp, poor, overlap_count = 0, 0, 0, 0
        selected = [i for i, score in enumerate(prediction["scores"]) if score >= threshold]
        selected.sort(key=lambda index: -prediction["scores"][index])
        for offset, index in enumerate(selected):
            box = prediction["boxes"][index]
            label = prediction["labels"][index]
            overlaps = box_iou(box, record["boxes"])
            candidates = [i for i in remaining if record["labels"][i] == label]
            best = max(candidates, key=lambda i: overlaps[i], default=None)
            if best is not None and overlaps[best] >= 0.5:
                tp += 1
                remaining.remove(best)
            else:
                fp += 1
                if best is not None and 0.1 <= overlaps[best] < 0.5:
                    poor += 1
            other = [prediction["boxes"][i] for i in selected[offset + 1 :]]
            overlap_count += int((box_iou(box, other) >= 0.5).sum())
        results.append(
            {
                "id": record["id"],
                "tp": tp,
                "fp": fp,
                "fn": len(remaining),
                "poor_localization_candidates": poor,
                "overlapping_prediction_pairs": overlap_count,
                "ground_truth": len(record["boxes"]),
                "predictions": len(selected),
                "invalid_predictions_rejected": prediction.get("invalid_predictions_rejected", 0),
            }
        )
    return results


def annotated_examples(records, predictions, errors, classes, root, output, threshold):
    """Deterministic coverage plus maximal observed error categories, with index."""
    sample_ids = [
        records[index]["id"]
        for index in np.linspace(0, len(records) - 1, min(12, len(records)), dtype=int)
    ]
    reasons = {sample_id: ["evenly spaced sorted split sample"] for sample_id in sample_ids}
    for category in (
        "tp",
        "fp",
        "fn",
        "poor_localization_candidates",
        "overlapping_prediction_pairs",
    ):
        chosen = max(errors, key=lambda item: (item[category], str(item["id"])))
        if chosen[category] > 0:
            reasons.setdefault(chosen["id"], []).append(
                f"maximum {category} in split ({chosen[category]})"
            )
    lookup = {record["id"]: record for record in records}
    names = {item["id"]: item["name"] for item in classes}
    directory = output / "examples"
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (image_id, reason) in enumerate(reasons.items()):
        record = lookup[image_id]
        with Image.open(safe_image_path(Path(root), record["image"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        # Left truth, right actual model; equal pixels and a legend.
        canvas = Image.new("RGB", (image.width * 2, image.height + 32), "white")
        canvas.paste(image, (0, 32))
        canvas.paste(image, (image.width, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 6), "Ground truth (green)", fill="black")
        draw.text((image.width + 4, 6), f"Model (red), score >= {threshold:g}", fill="black")
        for box, label in zip(record["boxes"], record["labels"], strict=True):
            draw.rectangle((box[0], box[1] + 32, box[2], box[3] + 32), outline="lime", width=2)
            draw.text(
                (box[0], box[1] + 32),
                names[label],
                fill="lime",
                stroke_width=1,
                stroke_fill="black",
            )
        prediction = predictions[image_id]
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"], strict=True
        ):
            if score < threshold:
                continue
            draw.rectangle(
                (box[0] + image.width, box[1] + 32, box[2] + image.width, box[3] + 32),
                outline="red",
                width=2,
            )
            draw.text(
                (box[0] + image.width, box[1] + 32),
                f"{names[label]} {score:.2f}",
                fill="red",
                stroke_width=1,
                stroke_fill="black",
            )
        name = f"sample_{number:02d}.jpg"
        canvas.save(directory / name, quality=90)
        index.append({"image_id": image_id, "file": f"examples/{name}", "selection": reason})
    atomic_json(
        output / "examples.json",
        {
            "convention": "truth left; actual model predictions right; deterministic sample coverage and maximum-error examples",
            "examples": index,
        },
    )


def plot_report(report: dict, history_path: Path | None, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for item in report["per_class"]:
        curve = item["pr_curve_iou50"]
        axes[0].plot(curve["recall"], curve["precision"], label=item["name"])
        centers = np.linspace(0.05, 0.95, 10)
        axes[1].plot(centers, item["score_histogram"], label=item["name"])
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="IoU .50, class-aware greedy matching",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(
        xlabel="Confidence score",
        ylabel="Prediction count",
        title="Confidence histogram (all retained predictions)",
    )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "precision_recall_and_scores.png", dpi=160)
    plt.close(fig)
    if history_path and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in history[0]["loss"]:
            axes[0].plot(
                [row["epoch"] + 1 for row in history],
                [row["loss"][key] for row in history],
                label=key,
            )
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training losses")
        axes[0].legend()
        axes[1].plot(
            [row["epoch"] + 1 for row in history],
            [row["validation_map50"] for row in history],
            label="AP50",
        )
        axes[1].plot(
            [row["epoch"] + 1 for row in history],
            [row["validation_map50_95"] for row in history],
            label="AP50:.95",
        )
        axes[1].set(xlabel="Epoch", ylabel="Validation AP", title="Validation checkpoint selection")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "training_history.png", dpi=160)
        plt.close(fig)


@torch.inference_mode()
def robustness_report(model, records, root, classes, device, threshold, limit=100):
    model.eval()
    chosen = records[: min(limit, len(records))]
    reports = {}
    variants = {
        "unperturbed": lambda image: image,
        "brightness_0.7": lambda image: ImageEnhance.Brightness(image).enhance(0.7),
        "gaussian_blur_radius_1": lambda image: image.filter(ImageFilter.GaussianBlur(1.0)),
        "jpeg_quality_50": None,
    }
    for name, transform in variants.items():
        predictions = {}
        for record in chosen:
            with Image.open(safe_image_path(Path(root), record["image"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            if transform:
                image = transform(image)
            else:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=50)
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    image = compressed.convert("RGB")
            result = model([image_tensor(image).to(device)])[0]
            boxes, labels, scores = (
                result[key].detach().cpu() for key in ("boxes", "labels", "scores")
            )
            valid = (
                torch.isfinite(boxes).all(1)
                & torch.isfinite(scores)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            predictions[record["id"]] = {
                "boxes": boxes[valid].tolist(),
                "labels": labels[valid].tolist(),
                "scores": scores[valid].tolist(),
                "invalid_predictions_rejected": int((~valid).sum()),
            }
        reports[name] = evaluate_detections(chosen, predictions, classes, threshold)
    return {
        "definition": "synthetic perturbations, not naturally observed acquisition metadata; first N sorted image IDs, same labels/geometry",
        "batch_size": 1,
        "device": device,
        "threshold": threshold,
        "images": [record["id"] for record in chosen],
        "unperturbed_same_subset": reports.pop("unperturbed"),
        "variants": reports,
    }


def evaluate(args) -> dict:
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("score threshold must be between zero and one")
    if args.batch_size < 1 or args.threads < 1 or args.robustness_limit < 1:
        raise ValueError("batch size, threads and robustness limit must be positive")
    if args.bootstrap < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    if args.split == "test" and not args.final_holdout:
        raise ValueError(
            "test split requires explicit --final-holdout; select using validation first"
        )
    if args.split == "test" and args.select_threshold:
        raise ValueError("threshold selection on final holdout is forbidden")
    output = Path(args.output)
    if (output / "report.json").exists():
        raise FileExistsError("evaluation exists; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest)
    records = sorted(
        [record for record in manifest["records"] if record["split"] == args.split],
        key=lambda record: str(record["id"]),
    )
    if not records:
        raise ValueError("requested split is empty")
    for record in records:
        if (
            record.get("sha256")
            and file_digest(safe_image_path(Path(args.data_root), record["image"]))
            != record["sha256"]
        ):
            raise ValueError(f"image checksum mismatch: {record['id']}")
    bundle_path = getattr(args, "bundle", None)
    checkpoint_path = getattr(args, "checkpoint", None)
    if bool(bundle_path) == bool(checkpoint_path):
        raise ValueError("provide exactly one --bundle or --checkpoint")
    manifest_digest = file_digest(args.manifest)
    if bundle_path:
        loaded = load_bundle(bundle_path, args.device)
        metadata = loaded.metadata
        if (
            [item.model_dump() for item in metadata.classes] != manifest["classes"]
            or metadata.dataset_manifest_sha256 != manifest_digest
            or metadata.dataset_id != manifest["dataset_id"]
            or metadata.taxonomy_version != manifest["taxonomy_version"]
        ):
            raise ValueError(
                "bundle taxonomy, dataset identity or data manifest differs from evaluation"
            )
        embedded_evaluation = json.loads(
            (Path(bundle_path) / "evaluation.json").read_text(encoding="utf-8")
        )
        if not isinstance(embedded_evaluation, dict):
            raise ValueError("bundle evaluation must be a report object")
        model = loaded.model
        if args.threshold is None:
            args.threshold = metadata.confidence_threshold
        model_source = {
            "source_type": "bundle",
            "artifact_digest": metadata.artifact_digest,
            "weights_sha256": metadata.weights_sha256,
            "model_version": metadata.model_version,
            "taxonomy_version": metadata.taxonomy_version,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "training_checkpoint_sha256": embedded_evaluation.get("checkpoint_sha256"),
            "training_run_id": metadata.training_run_id,
            "epoch": embedded_evaluation.get("epoch"),
        }
        history_path = None
    else:
        model, checkpoint = load_checkpoint_model(checkpoint_path, args.device)
        if (
            checkpoint["classes"] != manifest["classes"]
            or checkpoint["config"]["manifest_sha256"] != manifest_digest
        ):
            raise ValueError("checkpoint taxonomy or data manifest differs from evaluation")
        if args.threshold is None:
            args.threshold = 0.25
        model_source = {
            "source_type": "checkpoint",
            "checkpoint": str(Path(checkpoint_path).name),
            "checkpoint_sha256": file_digest(checkpoint_path),
            "training_run_id": checkpoint.get("training_run_id", Path(checkpoint_path).parent.name),
            "epoch": checkpoint["epoch"],
            "taxonomy_version": manifest["taxonomy_version"],
        }
        history_path = Path(checkpoint_path).parent / "history.json"
    evaluation_source = source_provenance()
    start = time.perf_counter()
    predictions = predict_records(model, records, args.data_root, args.device, args.batch_size)
    report = evaluate_detections(records, predictions, manifest["classes"], args.threshold)
    sweep = threshold_sweep(report)
    if args.select_threshold:
        # Most conservative threshold wins F1 ties, selected on validation alone.
        args.threshold = max(sweep, key=lambda row: (row["f1"], row["threshold"]))["threshold"]
        report = evaluate_detections(records, predictions, manifest["classes"], args.threshold)
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    prior = train_prior(train_records, manifest["classes"])
    baseline = evaluate_detections(
        records, prior_predictions(prior, records), manifest["classes"], args.threshold
    )
    errors = per_image_errors(records, predictions, args.threshold)
    report.update(
        {
            "split": args.split,
            "final_holdout": args.final_holdout,
            **model_source,
            "evaluation_source": evaluation_source,
            "manifest_sha256": manifest_digest,
            "dataset_id": manifest["dataset_id"],
            "device": args.device,
            "threshold_selected_here_on_validation": args.select_threshold,
            "threshold_sweep": sweep,
            "baseline_definition": "TRAIN-only class prevalence and median normalized box; one proposal per class per image",
            "baseline": baseline,
            "confusion": confusion_matrix(
                records, predictions, manifest["classes"], args.threshold
            ),
            "uncertainty": grouped_bootstrap(
                records, predictions, manifest["classes"], args.threshold, args.bootstrap
            ),
            "elapsed_seconds": time.perf_counter() - start,
            "missing_slice_metadata": [
                "lighting",
                "factory",
                "acquisition timestamp",
                "natural blur severity",
            ],
        }
    )
    slices = {}
    for name, predicate in {
        "images_with_small_box_lt32px_squared": lambda area: area < 32**2,
        "images_with_medium_box_32to96px_squared": lambda area: 32**2 <= area < 96**2,
        "images_with_large_box_ge96px_squared": lambda area: area >= 96**2,
    }.items():
        subset = [
            record
            for record in records
            if any(predicate((box[2] - box[0]) * (box[3] - box[1])) for box in record["boxes"])
        ]
        if subset:
            result = evaluate_detections(subset, predictions, manifest["classes"], args.threshold)
            slices[name] = {
                key: result[key] for key in ("images", "map50", "map50_95", "operating")
            }
    report["image_slices"] = {
        "definition": "overlapping IMAGE subsets containing at least one box of named area; all labels retained, not COCO object-area AP",
        "slices": slices,
    }
    atomic_json(output / "predictions.json", predictions)
    atomic_json(output / "per_image_errors.json", {"images": errors})
    annotated_examples(
        records, predictions, errors, manifest["classes"], args.data_root, output, args.threshold
    )
    plot_report(report, history_path, output)
    if args.robustness:
        robustness = robustness_report(
            model,
            records,
            args.data_root,
            manifest["classes"],
            args.device,
            args.threshold,
            args.robustness_limit,
        )
        atomic_json(output / "robustness.json", robustness)
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "split": args.split,
                "images": report["images"],
                "threshold": args.threshold,
                "map50": report["map50"],
                "map50_95": report["map50_95"],
                "operating": report["operating"],
                "baseline_map50": baseline["map50"],
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", help="training checkpoint including its model and provenance"
    )
    source.add_argument(
        "--bundle", help="self-contained released model bundle; no training checkpoint required"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-holdout", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        help="override score threshold; defaults to bundle operating threshold or .25 for checkpoint",
    )
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--robustness-limit", type=int, default=100)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
