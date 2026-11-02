"""Detection metrics: class-aware greedy matching and 101-point interpolated AP.

No crowd/ignore annotations or area/maxDet COCO averaging are implemented; this
is explicitly a COCO-style AP definition, not an assertion of COCOeval parity.
Every prediction is capped upstream to 100 detections per image. Classes with
no ground truth have AP=null and are excluded from the macro AP denominator.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

IOU_THRESHOLDS = [round(0.5 + i * 0.05, 2) for i in range(10)]


def box_iou(box: list[float], boxes: list[list[float]]) -> np.ndarray:
    candidates = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(candidates):
        return np.zeros(0, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    intersection = np.maximum(
        0, np.minimum(box[2:], candidates[:, 2:]) - np.maximum(box[:2], candidates[:, :2])
    ).prod(axis=1)
    areas = (candidates[:, 2:] - candidates[:, :2]).prod(axis=1)
    area = (box[2:] - box[:2]).prod()
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def _class_curve(records: list[dict], predictions: dict, class_id: int, iou: float):
    truth = {
        record["id"]: [
            box for box, label in zip(record["boxes"], record["labels"], strict=True)
            if label == class_id
        ]
        for record in records
    }
    n_truth = sum(map(len, truth.values()))
    ranked = []
    for record in records:
        prediction = predictions.get(record["id"], {"boxes": [], "labels": [], "scores": []})
        for index, (box, label, score) in enumerate(
            zip(prediction["boxes"], prediction["labels"], prediction["scores"], strict=True)
        ):
            if label == class_id:
                ranked.append((float(score), str(record["id"]), index, box, record["id"]))
    # Stable, fully specified tie break: score descending, image id, source index.
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in truth.items()}
    true_positive, scores = [], []
    for score, _, _, box, image_id in ranked:
        overlaps = box_iou(box, truth[image_id])
        overlaps[matched[image_id]] = -1
        hit = bool(len(overlaps) and overlaps.max() >= iou)
        if hit:
            matched[image_id][int(overlaps.argmax())] = True
        true_positive.append(int(hit))
        scores.append(score)
    tp = np.asarray(true_positive, dtype=np.int64)
    cumulative_tp = np.cumsum(tp)
    precision = cumulative_tp / np.maximum(np.arange(len(tp)) + 1, 1)
    recall = cumulative_tp / max(n_truth, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = [
        float(envelope[recall >= target].max()) if np.any(recall >= target) else 0.0
        for target in np.linspace(0, 1, 101)
    ]
    return {
        "ap": float(np.mean(sampled)) if n_truth else None,
        "ground_truth": n_truth,
        "scores": scores,
        "true_positive": true_positive,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }


def operating_counts(curve: dict, threshold: float) -> dict:
    selected = np.asarray(curve["scores"]) >= threshold
    tp = int(np.asarray(curve["true_positive"])[selected].sum())
    fp = int(selected.sum()) - tp
    fn = curve["ground_truth"] - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def confidence_reliability(curve: dict) -> list[dict]:
    """Empirical detection precision by score bin, not a probability guarantee."""
    scores = np.asarray(curve["scores"], dtype=float)
    hits = np.asarray(curve["true_positive"], dtype=float)
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected = (scores >= lower) & (scores <= upper if index == 9 else scores < upper)
        bins.append({"lower_inclusive": lower, "upper": upper, "upper_inclusive": index == 9, "detections": int(selected.sum()), "mean_score": float(scores[selected].mean()) if selected.any() else None, "observed_precision_iou50": float(hits[selected].mean()) if selected.any() else None})
    return bins


def evaluate_detections(
    records: list[dict], predictions: dict, classes: list[dict], threshold: float = 0.25
) -> dict:
    if not records:
        raise ValueError("evaluation split is empty")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("evaluation image ids must be unique")
    allowed = {item["id"] for item in classes}
    for record in records:
        if set(record["labels"]) - allowed:
            raise ValueError("unknown ground-truth class")
        prediction = predictions.get(record["id"], {"boxes": [], "labels": [], "scores": []})
        if set(prediction["labels"]) - allowed:
            raise ValueError("unknown prediction class")
        for box, _, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"], strict=True
        ):
            if len(box) != 4 or not np.isfinite(box).all() or not 0 <= score <= 1:
                raise ValueError("invalid prediction geometry or score")
            if not (0 <= box[0] < box[2] <= record["width"] and 0 <= box[1] < box[3] <= record["height"]):
                raise ValueError("prediction outside original frame")
    per_class = []
    all_ap = []
    aggregate_counts = {"tp": 0, "fp": 0, "fn": 0}
    for item in classes:
        curves = [_class_curve(records, predictions, item["id"], iou) for iou in IOU_THRESHOLDS]
        counts = operating_counts(curves[0], threshold)
        aps = [curve["ap"] for curve in curves if curve["ap"] is not None]
        result = {
            **item,
            "images_with_class": sum(item["id"] in record["labels"] for record in records),
            "ground_truth": curves[0]["ground_truth"],
            "ap50": curves[0]["ap"],
            "ap50_95": float(np.mean(aps)) if aps else None,
            "ap_by_iou": {str(iou): curve["ap"] for iou, curve in zip(IOU_THRESHOLDS, curves, strict=True)},
            "operating": counts,
            "pr_curve_iou50": curves[0],
            "score_histogram": np.histogram(curves[0]["scores"], bins=np.linspace(0, 1, 11))[0].tolist(),
            "confidence_reliability": confidence_reliability(curves[0]),
        }
        per_class.append(result)
        all_ap.extend(aps)
        for key in aggregate_counts:
            aggregate_counts[key] += counts[key]
    tp, fp, fn = (aggregate_counts[key] for key in ("tp", "fp", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "definition": "101-point interpolated class-aware AP; IoU .50:.05:.95; no crowd/ignore; max100 detections/image; absent GT classes excluded from macro AP",
        "images": len(records),
        "iou_thresholds": IOU_THRESHOLDS,
        "threshold": threshold,
        "map50": float(np.mean([item["ap50"] for item in per_class if item["ap50"] is not None])) if all_ap else None,
        "map50_95": float(np.mean(all_ap)) if all_ap else None,
        "operating": {**aggregate_counts, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
        "per_class": per_class,
    }


def threshold_sweep(report: dict) -> list[dict]:
    results = []
    for threshold in [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for item in report["per_class"]:
            current = operating_counts(item["pr_curve_iou50"], threshold)
            for key in counts:
                counts[key] += current[key]
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        results.append({"threshold": threshold, **counts, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return results


def confusion_matrix(records: list[dict], predictions: dict, classes: list[dict], threshold: float) -> dict:
    """Greedy class-agnostic IoU .5 matches; rows truth, columns prediction."""
    ids = [item["id"] for item in classes]
    lookup = {class_id: index for index, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids) + 1, len(ids) + 1), dtype=np.int64)
    for record in records:
        prediction = predictions[record["id"]]
        remaining = set(range(len(record["boxes"])))
        order = sorted(range(len(prediction["scores"])), key=lambda i: -prediction["scores"][i])
        for index in order:
            if prediction["scores"][index] < threshold:
                continue
            overlaps = box_iou(prediction["boxes"][index], record["boxes"])
            for candidate in range(len(overlaps)):
                if candidate not in remaining:
                    overlaps[candidate] = -1
            col = lookup[prediction["labels"][index]]
            if len(overlaps) and overlaps.max() >= 0.5:
                best = int(overlaps.argmax())
                matrix[lookup[record["labels"][best]], col] += 1
                remaining.remove(best)
            else:
                matrix[-1, col] += 1
        for index in remaining:
            matrix[lookup[record["labels"][index]], -1] += 1
    return {"labels": [item["name"] for item in classes] + ["background/missed"], "rows": "ground_truth", "columns": "prediction", "iou": 0.5, "threshold": threshold, "counts": matrix.tolist()}


def grouped_bootstrap(
    records: list[dict], predictions: dict, classes: list[dict], threshold: float,
    repetitions: int = 200, seed: int = 2026,
) -> dict:
    """Group-resampled micro precision/recall, preserving correlated acquisitions."""
    groups = defaultdict(list)
    for record in records:
        groups[record.get("group", record["id"])].append(record)
    if len(groups) < 5 or repetitions < 1:
        return {"method": "group bootstrap", "status": "insufficient groups", "groups": len(groups)}
    group_counts = []
    for subset in groups.values():
        report = evaluate_detections(subset, predictions, classes, threshold)
        group_counts.append([report["operating"][key] for key in ("tp", "fp", "fn")])
    array = np.asarray(group_counts)
    rng = np.random.default_rng(seed)
    precision, recall = [], []
    for _ in range(repetitions):
        tp, fp, fn = array[rng.integers(0, len(array), len(array))].sum(axis=0)
        precision.append(float(tp / max(tp + fp, 1)))
        recall.append(float(tp / max(tp + fn, 1)))
    return {"method": "percentile group bootstrap of micro precision/recall at IoU .5; no training-seed variability estimate", "groups": len(groups), "repetitions": repetitions, "seed": seed, "precision_95_interval": np.quantile(precision, [0.025, 0.975]).tolist(), "recall_95_interval": np.quantile(recall, [0.025, 0.975]).tolist()}
