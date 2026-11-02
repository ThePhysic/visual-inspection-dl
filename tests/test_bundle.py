import hashlib
import json
import sys

import pytest
import torch

from inspection_dl.bundle import (
    BundleError,
    canonical_json,
    export_bundle,
    load_bundle,
    sha256_file,
    validate_bundle,
    main,
)
from inspection_dl.model import build_detector


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(91)
    torch.set_num_threads(2)
    model = build_detector(3, pretrained=False).eval()
    path = tmp_path / "release"
    metadata = export_bundle(
        model.state_dict(),
        path,
        classes=[{"id": 1, "name": "fixture-a"}, {"id": 2, "name": "fixture-b"}],
        model_version="untrained-test-fixture",
        taxonomy_version="fixture-v1",
        dataset_id="synthetic-test-fixture",
        dataset_manifest_sha256="a" * 64,
        training_run_id="fixture-run",
        report={"simulated": True, "note": "No trained-model quality claim"},
        licensing={"source": "synthetic-test-fixture", "license": "project test code"},
    )
    return path, metadata, model


def rewrite_metadata(path, change):
    metadata = json.loads((path / "metadata.json").read_text())
    change(metadata)
    del metadata["artifact_digest"]
    metadata["artifact_digest"] = hashlib.sha256(canonical_json(metadata)).hexdigest()
    (path / "metadata.json").write_bytes(canonical_json(metadata))


def test_real_architecture_round_trip_predictions(exported):
    path, metadata, original = exported
    loaded = load_bundle(path)
    tensor = torch.rand(3, 64, 80)
    with torch.inference_mode():
        before = original([tensor])[0]
        after = loaded.model([tensor])[0]
    assert metadata.artifact_digest == loaded.metadata.artifact_digest
    assert loaded.device == "cpu"
    for key in ("boxes", "scores", "labels"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_immutable_export_refuses_existing_destination(exported):
    path, _, model = exported
    with pytest.raises(FileExistsError):
        export_bundle(
            model.state_dict(),
            path,
            classes=[{"id": 1, "name": "a"}],
            model_version="v",
            taxonomy_version="v",
            dataset_id="d",
            dataset_manifest_sha256="a" * 64,
            training_run_id="r",
            report={},
            licensing={"source": "fixture"},
        )


@pytest.mark.parametrize("filename", ["weights.pt", "evaluation.json", "metadata.json"])
def test_missing_files_rejected(exported, filename):
    path, _, _ = exported
    (path / filename).unlink()
    with pytest.raises(BundleError):
        load_bundle(path)


@pytest.mark.parametrize("filename", ["weights.pt", "evaluation.json"])
def test_corrupt_artifacts_rejected_before_load(exported, filename):
    path, _, _ = exported
    with (path / filename).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(BundleError):
        validate_bundle(path)


def test_metadata_integrity_covers_class_mapping(exported):
    path, _, _ = exported
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["classes"][0]["name"] = "silently-substituted-taxonomy"
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(BundleError, match="metadata digest"):
        validate_bundle(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "2"),
        ("preprocessing", "unknown"),
        ("architecture", "other-detector"),
        ("coordinate_system", "normalized_xywh"),
        ("input_size", [640, 640]),
    ],
)
def test_incompatible_semantics_rejected_even_with_recomputed_digest(exported, key, value):
    path, _, _ = exported
    rewrite_metadata(path, lambda metadata: metadata.update({key: value}))
    with pytest.raises(BundleError):
        validate_bundle(path)


def test_incompatible_weight_shapes_rejected(exported):
    path, _, _ = exported
    torch.save({"wrong.weight": torch.ones(1)}, path / "weights.pt")
    rewrite_metadata(
        path, lambda metadata: metadata.update(weights_sha256=sha256_file(path / "weights.pt"))
    )
    with pytest.raises(BundleError, match="incompatible"):
        load_bundle(path)


def test_runtime_incompatibility_rejected(exported):
    path, _, _ = exported
    rewrite_metadata(path, lambda metadata: metadata["runtime"].update(torch="99.0.0"))
    with pytest.raises(BundleError, match="runtime"):
        load_bundle(path)


def test_taxonomy_ids_are_explicit_and_consecutive(exported):
    path, _, _ = exported
    rewrite_metadata(path, lambda metadata: metadata["classes"][0].update(id=42))
    with pytest.raises(BundleError):
        validate_bundle(path)


@pytest.mark.parametrize(
    "mismatch",
    [
        "classes",
        "dataset_id",
        "taxonomy_version",
        "manifest",
        "checkpoint_report",
        "report_manifest",
        "threshold",
    ],
)
def test_export_cli_rejects_mismatched_provenance(tmp_path, monkeypatch, mismatch):
    manifest = {
        "classes": [{"id": 1, "name": "fixture"}],
        "dataset_id": "fixture-dataset",
        "taxonomy_version": "fixture-taxonomy",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    checkpoint = {
        **manifest,
        "architecture": "ssdlite320_mobilenet_v3_large",
        "model": {"dummy": torch.ones(1)},
        "config": {
            "manifest_sha256": sha256_file(manifest_path),
            "preprocessing": "ssdlite320_rgb_unit_v1",
        },
    }
    if mismatch in {"classes", "dataset_id", "taxonomy_version"}:
        checkpoint[mismatch] = [{"id": 1, "name": "wrong"}] if mismatch == "classes" else "wrong"
    if mismatch == "manifest":
        checkpoint["config"]["manifest_sha256"] = "b" * 64
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": sha256_file(manifest_path),
        "threshold": 0.25,
    }
    if mismatch == "checkpoint_report":
        report["checkpoint_sha256"] = "b" * 64
    if mismatch == "report_manifest":
        report["manifest_sha256"] = "b" * 64
    if mismatch == "threshold":
        report["threshold"] = 0.5
    report_path = tmp_path / "report.json"
    report_path.write_bytes(canonical_json(report))
    license_path = tmp_path / "license.json"
    license_path.write_bytes(canonical_json({"source": "fixture"}))
    destination = tmp_path / "release"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bundle",
            str(checkpoint_path),
            str(destination),
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--licensing",
            str(license_path),
            "--model-version",
            "fixture",
        ],
    )
    with pytest.raises(BundleError):
        main()
    assert not destination.exists()
