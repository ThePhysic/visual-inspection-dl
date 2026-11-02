"""Hand-computable detection fixtures; synthetic, never performance evidence."""

import numpy as np
import pytest

from inspection_dl.metrics import (
    box_iou,
    confusion_matrix,
    evaluate_detections,
    grouped_bootstrap,
    threshold_sweep,
)

CLASSES = [{"id": 1, "name": "scratch"}, {"id": 2, "name": "missing"}]


def record(image_id="one", boxes=None, labels=None):
    return {
        "id": image_id,
        "width": 100,
        "height": 100,
        "group": image_id,
        "boxes": [[10, 10, 30, 30]] if boxes is None else boxes,
        "labels": [1] if labels is None else labels,
    }


def test_iou_boundary_and_exclusive_geometry():
    assert box_iou(
        [0, 0, 10, 10], [[0, 0, 10, 10], [10, 0, 20, 10], [5, 0, 15, 10]]
    ).tolist() == pytest.approx([1, 0, 1 / 3])
    assert box_iou([0, 0, 1, 1], []).shape == (0,)


def test_perfect_detection_and_absent_class_not_in_macro():
    prediction = {"one": {"boxes": [[10, 10, 30, 30]], "labels": [1], "scores": [0.9]}}
    report = evaluate_detections([record()], prediction, CLASSES)
    assert report["map50"] == 1
    assert report["map50_95"] == 1
    assert report["per_class"][1]["ap50"] is None
    assert report["operating"] == {"tp": 1, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1}


def test_high_scoring_false_positive_reduces_ap():
    prediction = {
        "one": {
            "boxes": [[60, 60, 80, 80], [10, 10, 30, 30]],
            "labels": [1, 1],
            "scores": [0.95, 0.9],
        }
    }
    report = evaluate_detections([record()], prediction, CLASSES)
    assert report["map50"] == pytest.approx(0.5)
    assert report["operating"]["fp"] == 1


def test_duplicate_detection_only_matches_once_and_threshold_counts():
    prediction = {
        "one": {
            "boxes": [[10, 10, 30, 30], [10, 10, 30, 30]],
            "labels": [1, 1],
            "scores": [0.9, 0.3],
        }
    }
    report = evaluate_detections([record()], prediction, CLASSES, threshold=0.5)
    assert report["map50"] == 1
    assert report["operating"]["tp"] == 1
    assert report["operating"]["fp"] == 0
    assert next(item for item in threshold_sweep(report) if item["threshold"] == 0.25)["fp"] == 1


def test_classes_matched_independently_and_confusion_reports_wrong_class():
    prediction = {"one": {"boxes": [[10, 10, 30, 30]], "labels": [2], "scores": [0.9]}}
    report = evaluate_detections([record()], prediction, CLASSES)
    assert report["map50"] == 0
    assert report["operating"]["fp"] == 1
    assert report["operating"]["fn"] == 1
    confusion = confusion_matrix([record()], prediction, CLASSES, 0.25)
    assert confusion["counts"][0][1] == 1


def test_empty_predictions_and_empty_truth_are_explicit():
    empty = {"one": {"boxes": [], "labels": [], "scores": []}}
    assert evaluate_detections([record()], empty, CLASSES)["map50"] == 0
    result = evaluate_detections([record(boxes=[], labels=[])], empty, CLASSES)
    assert result["map50"] is None
    assert result["operating"]["tp"] == 0


def test_one_of_two_recalled_matches_101_point_interpolation():
    records = [record("one"), record("two")]
    prediction = {"one": {"boxes": [[10, 10, 30, 30]], "labels": [1], "scores": [0.9]}}
    assert evaluate_detections(records, prediction, CLASSES)["map50"] == pytest.approx(51 / 101)


@pytest.mark.parametrize(
    "box,score",
    [
        ([0, 0, float("nan"), 5], 0.5),
        ([0, 0, 101, 20], 0.5),
        ([3, 3, 3, 8], 0.5),
        ([0, 0, 3, 5], float("nan")),
        ([0, 0, 3, 5], 1.1),
    ],
)
def test_invalid_prediction_rejected(box, score):
    with pytest.raises(ValueError):
        evaluate_detections(
            [record()], {"one": {"boxes": [box], "labels": [1], "scores": [score]}}, CLASSES
        )


def test_bootstrap_resamples_groups_reproducibly():
    records = [record(str(i)) for i in range(8)]
    predictions = {
        item["id"]: {"boxes": item["boxes"], "labels": item["labels"], "scores": [0.9]}
        for item in records
    }
    report = grouped_bootstrap(records, predictions, CLASSES, 0.25, repetitions=20)
    assert report["precision_95_interval"] == [1, 1]
    assert report["recall_95_interval"] == [1, 1]
    assert report == grouped_bootstrap(records, predictions, CLASSES, 0.25, repetitions=20)
    assert np.isfinite(report["precision_95_interval"]).all()


def test_filtered_prediction_exclusions_remain_visible_in_aggregate():
    records = [record("one"), record("two")]
    predictions = {
        item["id"]: {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "scores": [0.9],
            "invalid_predictions_rejected": index + 1,
        }
        for index, item in enumerate(records)
    }
    report = evaluate_detections(records, predictions, CLASSES)
    assert report["map50"] == 1
    assert report["invalid_predictions_rejected"] == 3
