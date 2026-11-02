"""Released-model evaluation on generated fixtures, without training artifacts."""

import hashlib
import json
from argparse import Namespace

import pytest
import torch
from PIL import Image

from inspection_dl.bundle import canonical_json, export_bundle
from inspection_dl.evaluate import evaluate
from inspection_dl.model import ARCHITECTURE, build_detector
from inspection_dl.train import file_digest


@pytest.fixture
def released_fixture(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(92)
    classes = [{"id": 1, "name": "synthetic-fixture"}]
    records = []
    for index, split in enumerate(("train", "val")):
        image = tmp_path / f"fixture_{index}.png"
        Image.new("RGB", (87, 53), color=(45 + index, 85, 130)).save(image)
        records.append(
            {
                "id": str(index),
                "image": image.name,
                "width": 87,
                "height": 53,
                "sha256": file_digest(image),
                "group": str(index),
                "split": split,
                "boxes": [[5, 8, 37, 42]],
                "labels": [1],
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "generated-fixture-only",
                "taxonomy_version": "fixture-1",
                "classes": classes,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    model = build_detector(2, pretrained=False)
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "model": model.state_dict(),
            "classes": classes,
            "config": {"manifest_sha256": file_digest(manifest)},
            "training_run_id": "synthetic-no-training-claim",
            "epoch": 7,
        },
        checkpoint,
    )
    bundle = tmp_path / "release"
    metadata = export_bundle(
        model.state_dict(),
        bundle,
        classes=classes,
        model_version="synthetic-fixture-v1",
        taxonomy_version="fixture-1",
        dataset_id="generated-fixture-only",
        dataset_manifest_sha256=file_digest(manifest),
        training_run_id="synthetic-no-training-claim",
        report={
            "epoch": 7,
            "checkpoint_sha256": file_digest(checkpoint),
            "kind": "synthetic fixture only",
        },
        licensing={"fixture": "generated in test, no external images or weights"},
        threshold=0.41,
    )
    config = {
        "manifest": str(manifest),
        "data_root": str(tmp_path),
        "checkpoint": None,
        "bundle": str(bundle),
        "threshold": None,
        "batch_size": 1,
        "threads": 2,
        "robustness_limit": 1,
        "bootstrap": 0,
        "split": "val",
        "final_holdout": False,
        "select_threshold": False,
        "device": "cpu",
        "robustness": False,
    }
    return {
        "config": config,
        "checkpoint": checkpoint,
        "bundle": bundle,
        "metadata": metadata,
        "root": tmp_path,
    }


def test_bundle_evaluation_matches_checkpoint_without_training_files(released_fixture):
    fixture = released_fixture
    root = fixture["root"]
    checkpoint_report = evaluate(
        Namespace(
            **{
                **fixture["config"],
                "checkpoint": str(fixture["checkpoint"]),
                "bundle": None,
                "threshold": 0.41,
                "output": str(root / "from-checkpoint"),
            }
        )
    )
    checkpoint_digest = file_digest(fixture["checkpoint"])
    # The release path must work after the original training checkpoint is absent.
    fixture["checkpoint"].unlink()
    bundle_report = evaluate(Namespace(**fixture["config"], output=str(root / "from-bundle")))
    assert checkpoint_report["checkpoint_sha256"] == checkpoint_digest
    assert checkpoint_report["source_type"] == "checkpoint"
    assert bundle_report["source_type"] == "bundle"
    assert bundle_report["threshold"] == fixture["metadata"].confidence_threshold == 0.41
    assert bundle_report["checkpoint"] is bundle_report["checkpoint_sha256"] is None
    assert bundle_report["training_checkpoint_sha256"] == checkpoint_digest
    assert bundle_report["artifact_digest"] == fixture["metadata"].artifact_digest
    assert bundle_report["weights_sha256"] == fixture["metadata"].weights_sha256
    assert bundle_report["training_run_id"] == fixture["metadata"].training_run_id
    assert bundle_report["epoch"] == 7
    for key in (
        "map50",
        "map50_95",
        "operating",
        "per_class",
        "baseline",
        "confusion",
        "image_slices",
    ):
        assert bundle_report[key] == checkpoint_report[key]
    assert (root / "from-checkpoint" / "predictions.json").read_bytes() == (
        root / "from-bundle" / "predictions.json"
    ).read_bytes()
    assert not (root / "from-bundle" / "training_history.png").exists()
    assert len(bundle_report["evaluation_source"]["package_source_sha256"]) == 64
    assert bundle_report["evaluation_source"]["revision"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", "different-dataset"),
        ("taxonomy_version", "different-taxonomy"),
        ("dataset_manifest_sha256", "0" * 64),
        ("classes", [{"id": 1, "name": "different-class"}]),
    ],
)
def test_bundle_evaluation_rejects_signed_semantic_mismatch(released_fixture, field, value):
    fixture = released_fixture
    metadata_path = fixture["bundle"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    del metadata["artifact_digest"]
    metadata["artifact_digest"] = hashlib.sha256(canonical_json(metadata)).hexdigest()
    metadata_path.write_bytes(canonical_json(metadata))
    with pytest.raises(ValueError, match="bundle taxonomy, dataset identity or data manifest"):
        evaluate(Namespace(**fixture["config"], output=str(fixture["root"] / "mismatch")))
    assert not (fixture["root"] / "mismatch" / "report.json").exists()


def test_bundle_evaluation_rejects_ambiguous_source(released_fixture):
    fixture = released_fixture
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(
            Namespace(
                **{
                    **fixture["config"],
                    "checkpoint": str(fixture["checkpoint"]),
                    "output": str(fixture["root"] / "ambiguous"),
                }
            )
        )
