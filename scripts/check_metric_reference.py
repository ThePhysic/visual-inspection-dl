"""One-off scientific cross-check; synthetic fixtures, no model-quality claim."""

import importlib.metadata
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from inspection_dl.metrics import IOU_THRESHOLDS, evaluate_detections


def xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def fixture(seed):
    rng = np.random.default_rng(seed)
    classes = [{"id": i, "name": f"synthetic_{i}"} for i in (1, 2, 3)]
    records, predictions, annotations, results = [], {}, [], []
    for image_id in range(1, 31):
        boxes, labels = [], []
        prediction = {"boxes": [], "labels": [], "scores": []}
        for _ in range(int(rng.integers(1, 5))):
            start = rng.uniform(0, 55, 2)
            box = np.r_[start, start + rng.uniform(10, 40, 2)].tolist()
            label = int(rng.integers(1, 4))
            boxes.append(box)
            labels.append(label)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": xywh(box),
                    "area": (box[2] - box[0]) * (box[3] - box[1]),
                    "iscrowd": 0,
                }
            )
            if rng.random() < 0.8:
                noisy = np.clip(np.asarray(box) + rng.normal(0, 3, 4), 0, 100).tolist()
                if noisy[0] >= noisy[2] or noisy[1] >= noisy[3]:
                    noisy = box
                prediction["boxes"].append(noisy)
                prediction["labels"].append(
                    label if rng.random() > 0.15 else int(rng.integers(1, 4))
                )
                prediction["scores"].append(float(rng.uniform(0.15, 0.99)))
                if rng.random() < 0.3:
                    prediction["boxes"].append(noisy)
                    prediction["labels"].append(label)
                    prediction["scores"].append(float(rng.uniform(0.01, 0.6)))
        for _ in range(int(rng.integers(0, 4))):
            start = rng.uniform(0, 60, 2)
            prediction["boxes"].append(np.r_[start, start + rng.uniform(5, 35, 2)].tolist())
            prediction["labels"].append(int(rng.integers(1, 4)))
            prediction["scores"].append(float(rng.uniform(0.01, 0.95)))
        records.append(
            {"id": image_id, "width": 100, "height": 100, "boxes": boxes, "labels": labels}
        )
        predictions[image_id] = prediction
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"], strict=True
        ):
            results.append(
                {"image_id": image_id, "category_id": label, "bbox": xywh(box), "score": score}
            )
    dataset = COCO()
    dataset.dataset = {
        "info": {"description": "generated synthetic metric reference only"},
        "images": [{"id": r["id"], "width": 100, "height": 100} for r in records],
        "categories": classes,
        "annotations": annotations,
    }
    dataset.createIndex()
    reference = COCOeval(dataset, dataset.loadRes(results), "bbox")
    reference.params.imgIds = [record["id"] for record in records]
    reference.params.catIds = [1, 2, 3]
    reference.evaluate()
    reference.accumulate()
    # precision shape is IoU, recall, class, area, maxDet; all area, maxDet100.
    precision = reference.eval["precision"][:, :, :, 0, 2]
    report = evaluate_detections(records, predictions, classes)
    comparisons = []
    for class_index, item in enumerate(report["per_class"]):
        for iou_index, iou in enumerate(IOU_THRESHOLDS):
            values = precision[iou_index, :, class_index]
            expected = float(values[values >= 0].mean())
            actual = item["ap_by_iou"][str(iou)]
            comparisons.append(
                {
                    "class_id": item["id"],
                    "iou": iou,
                    "implementation": actual,
                    "pycocotools": expected,
                    "absolute_difference": abs(actual - expected),
                }
            )
    return {
        "seed": seed,
        "images": len(records),
        "ground_truth": len(annotations),
        "predictions": len(results),
        "comparisons": comparisons,
        "max_absolute_difference": max(row["absolute_difference"] for row in comparisons),
    }


cases = [fixture(seed) for seed in (17, 43, 71)]
tolerance = 1e-12
evidence = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "kind": "synthetic independent metric verification, not trained-model evaluation",
    "reference": "pycocotools.COCOeval bbox",
    "pycocotools_version": importlib.metadata.version("pycocotools"),
    "numpy_version": np.__version__,
    "source": "https://github.com/cocodataset/cocoapi",
    "fixture_definition": "three seeded 30-image multiclass fixtures, random finite boxes, partial recall, noisy localization, wrong classes, duplicate detections, random false positives; no crowd/ignore; all area and maxDet100",
    "tolerance": tolerance,
    "cases": cases,
    "max_absolute_difference": max(case["max_absolute_difference"] for case in cases),
}
evidence["passed"] = evidence["max_absolute_difference"] <= tolerance
Path("reports").mkdir(exist_ok=True)
Path("reports/metric-reference.json").write_text(
    json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps({key: evidence[key] for key in ("passed", "max_absolute_difference", "tolerance")})
)
assert evidence["passed"]
