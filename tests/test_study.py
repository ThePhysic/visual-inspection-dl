"""A finalization must reject substituted candidates before accessing holdout."""

import json
import runpy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

STUDY = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_study.py"))


def test_finalization_rejects_incomplete_changed_config_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    expected = {"seed": 42, "horizontal_flip_probability": 0.5, "manifest_sha256": "original"}
    record = {
        "completed": True,
        "config": expected.copy(),
        "best_checkpoint_sha256": STUDY["digest"](checkpoint),
    }

    def save():
        (tmp_path / "run.json").write_text(json.dumps(record))

    save()
    assert STUDY["verify_completed_run"](tmp_path, expected) == record
    record["completed"] = False
    save()
    with pytest.raises(ValueError, match="incomplete"):
        STUDY["verify_completed_run"](tmp_path, expected)
    record["completed"] = True
    record["config"]["manifest_sha256"] = "substituted"
    save()
    with pytest.raises(ValueError, match="configuration"):
        STUDY["verify_completed_run"](tmp_path, expected)
    record["config"] = expected.copy()
    save()
    checkpoint.write_bytes(b"substituted weights")
    with pytest.raises(ValueError, match="integrity"):
        STUDY["verify_completed_run"](tmp_path, expected)


def test_controlled_comparison_requires_identical_initial_tensors(tmp_path):
    paths = [tmp_path / "first.pt", tmp_path / "second.pt"]
    state = {
        "classes": [{"id": 1, "name": "defect"}],
        "model": {"weight": torch.tensor([[1.0, 2.0]]), "counter": torch.tensor(0)},
    }
    for path in paths:
        torch.save(state, path)
    report = STUDY["verify_initialization"](paths)
    assert report["identical_initial_model_tensors"]
    assert len(report["initial_tensor_digest"]) == 64
    state["model"]["weight"][0, 0] += 1
    torch.save(state, paths[1])
    with pytest.raises(ValueError, match="initialization differs"):
        STUDY["verify_initialization"](paths)


@pytest.fixture
def study_workflow(tmp_path, monkeypatch):
    """Exercise the actual runner with tiny persisted artifacts and no model execution."""
    from inspection_dl.bundle import export_bundle, validate_bundle

    monkeypatch.chdir(tmp_path)
    classes = [{"id": 1, "name": "fixture"}]
    manifest = {
        "dataset_id": "audit-fixture",
        "taxonomy_version": "fixture-v1",
        "classes": classes,
        "records": [{"id": "v", "split": "val"}, {"id": "t", "split": "test"}],
    }

    def save(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    save("data/manifests/pcb-ind-v4.json", manifest)
    save("configs/licensing.json", {"license": "fixture"})
    calls = Counter()
    control = {"failure": None}
    argv = [
        "run_study.py",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--study-id",
        "audit",
        "--release",
        "audit-model",
    ]
    globals_ = STUDY["main"].__globals__
    original_write = globals_["write_or_verify"]

    def write_or_fail(path, value):
        if path.name == "selection.json" and control["failure"] == "after-validation":
            control["failure"] = None
            raise RuntimeError("injected after-validation")
        original_write(path, value)

    def execute(command, check):
        module = command[2].split(".")[-1]
        stage = module
        if module == "evaluate":
            stage = "validation" if command[command.index("--split") + 1] == "val" else "holdout"
        calls[stage] += 1
        if control["failure"] == stage:
            control["failure"] = None
            raise RuntimeError(f"injected {stage}")
        manifest_hash = STUDY["digest"](Path("data/manifests/pcb-ind-v4.json"))
        if module == "train":
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            flip = "--horizontal-flip" in command
            config = {
                "epochs": 1,
                "batch_size": 2,
                "seed": 42,
                "learning_rate": 0.0003,
                "amp": False,
                "manifest_sha256": manifest_hash,
                "horizontal_flip_probability": 0.5 if flip else 0.0,
                "freeze_backbone": False,
                "pretrained_initialization": True,
                "architecture": "ssdlite320_mobilenet_v3_large",
                "preprocessing": "ssdlite320_rgb_unit_v1",
                "optimizer": "AdamW",
                "weight_decay": 0.0005,
                "scheduler": "CosineAnnealingLR",
            }
            state = {"model": {"weight": torch.tensor([1.0])}, "classes": classes}
            torch.save(state, output / "initialized.pt")
            torch.save(state, output / "best.pt")
            save(
                output / "run.json",
                {
                    "completed": True,
                    "config": config,
                    "pretrained": {},
                    "best_checkpoint_sha256": STUDY["digest"](output / "best.pt"),
                    "best_validation_map50_95": 0.2 if flip else 0.1,
                },
            )
            for filename in (
                "history.json",
                "prior.json",
                "baseline_validation.json",
                "best_validation.json",
                "best_validation_predictions.json",
            ):
                save(output / filename, {})
        elif module == "evaluate":
            output = Path(command[command.index("--output") + 1])
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            split = "val" if stage == "validation" else "test"
            report = {
                "checkpoint_sha256": STUDY["digest"](checkpoint),
                "training_run_id": checkpoint.parent.name,
                "manifest_sha256": manifest_hash,
                "dataset_id": manifest["dataset_id"],
                "split": split,
                "device": "cpu",
                "final_holdout": split == "test",
                "threshold_selected_here_on_validation": split == "val",
                "images": 1,
                "per_class": classes,
                "threshold": 0.2,
                "threshold_sweep": [{"threshold": 0.2, "f1": 0.5}],
            }
            save(output / "robustness.json", {})
            save(output / "report.json", report)
        elif module == "bundle":
            report = STUDY["read"](Path(command[command.index("--report") + 1]))
            export_bundle(
                {"weight": torch.tensor([1.0])},
                command[4],
                classes=classes,
                model_version="audit-model",
                taxonomy_version=manifest["taxonomy_version"],
                dataset_id=manifest["dataset_id"],
                dataset_manifest_sha256=manifest_hash,
                training_run_id="audit-flip",
                report=report,
                licensing=STUDY["read"](Path(command[command.index("--licensing") + 1])),
                threshold=0.2,
            )
        elif module == "benchmark":
            metadata = validate_bundle(command[command.index("--bundle") + 1])
            identity = {
                key: getattr(metadata, key)
                for key in (
                    "artifact_digest",
                    "weights_sha256",
                    "model_version",
                    "taxonomy_version",
                    "dataset_id",
                )
            }
            save(
                command[command.index("--output") + 1],
                {
                    **identity,
                    "manifest_sha256": manifest_hash,
                    "classes": classes,
                    "split": "val",
                    "sample_ids": ["v"],
                    "benchmark_source": {"package_source_sha256": "a" * 64},
                    "results": [
                        {
                            **identity,
                            "device": "cpu",
                            "batch_size": batch,
                            "images": 1,
                            "quality_on_identical_validation_subset": {
                                "images": 1,
                                "threshold": 0.2,
                            },
                        }
                        for batch in (1, 4)
                    ]
                    + [{"device": "cuda", "unavailable": True}],
                },
            )

    monkeypatch.setitem(globals_, "write_or_verify", write_or_fail)
    monkeypatch.setitem(
        globals_,
        "subprocess",
        SimpleNamespace(
            run=execute,
            check_output=lambda command, **kwargs: "a" * 40 if command[1] == "rev-parse" else "",
            list2cmdline=lambda command: " ".join(command),
        ),
    )

    def invoke(finalize=False):
        monkeypatch.setattr("sys.argv", argv + (["--finalize-only"] if finalize else []))
        STUDY["main"]()

    return SimpleNamespace(root=tmp_path, invoke=invoke, calls=calls, control=control, save=save)


@pytest.mark.parametrize("failure", ["validation", "after-validation", "bundle", "benchmark"])
def test_finalization_recovers_completed_stages_without_repeating_holdout(study_workflow, failure):
    flow = study_workflow
    flow.control["failure"] = failure
    with pytest.raises(RuntimeError, match="injected"):
        flow.invoke()
    flow.invoke(finalize=True)
    assert flow.calls["holdout"] == 1
    assert flow.calls["train"] == 2
    assert (flow.root / "reports/audit/completed.json").is_file()
    original_calls = flow.calls.copy()
    original_files = {
        path: path.read_bytes() for path in (flow.root / "reports/audit").rglob("*.json")
    }
    flow.invoke(finalize=True)
    assert flow.calls == original_calls
    assert all(path.read_bytes() == content for path, content in original_files.items())


def test_finalization_refuses_incomplete_holdout_and_missing_frozen_selection(study_workflow):
    flow = study_workflow
    flow.control["failure"] = "holdout"
    with pytest.raises(RuntimeError, match="injected holdout"):
        flow.invoke()
    with pytest.raises(ValueError, match="automatic repetition is forbidden"):
        flow.invoke(finalize=True)
    assert flow.calls["holdout"] == 1
    (flow.root / "reports/audit/selection.json").unlink()
    with pytest.raises(ValueError, match="refusing reselection"):
        flow.invoke(finalize=True)


@pytest.mark.parametrize(
    "relative,field",
    [
        ("reports/audit/validation/report.json", "checkpoint_sha256"),
        ("reports/audit/holdout/report.json", "threshold"),
        ("reports/audit/benchmark-release.json", "weights_sha256"),
    ],
)
def test_completed_finalization_rejects_changed_evidence(study_workflow, relative, field):
    flow = study_workflow
    flow.invoke()
    path = flow.root / relative
    value = STUDY["read"](path)
    value[field] = 0.9 if field == "threshold" else "0" * 64
    flow.save(path, value)
    calls = flow.calls.copy()
    with pytest.raises(ValueError, match="differs"):
        flow.invoke(finalize=True)
    assert flow.calls == calls


@pytest.mark.parametrize(
    "path", ["releases/audit-model", "artifacts/audit-model", "runs/audit-flip"]
)
def test_output_collisions_fail_before_training(study_workflow, path):
    flow = study_workflow
    (flow.root / path).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fresh study and release IDs"):
        flow.invoke()
    assert not flow.calls


def test_legacy_completed_study_gets_verified_binding_without_repeating_results(study_workflow):
    flow = study_workflow
    flow.invoke()
    preregistration_path = flow.root / "reports/audit/preregistration.json"
    preregistration = STUDY["read"](preregistration_path)
    del preregistration["release"]
    flow.save(preregistration_path, preregistration)
    for filename in ("release-binding.json", "holdout-started.json", "completed.json"):
        (flow.root / "reports/audit" / filename).unlink()
    frozen_bytes = preregistration_path.read_bytes()
    calls = flow.calls.copy()
    flow.invoke(finalize=True)
    assert flow.calls == calls
    assert preregistration_path.read_bytes() == frozen_bytes
    binding = STUDY["read"](flow.root / "reports/audit/release-binding.json")
    assert binding["legacy_preregistration"] is True
    assert binding["release"] == "audit-model"
    flow.invoke(finalize=True)
    assert flow.calls == calls


def test_validly_rehashed_bundle_with_different_tensors_is_rejected(study_workflow):
    from inspection_dl.bundle import canonical_json
    import hashlib

    flow = study_workflow
    flow.invoke()
    bundle = flow.root / "artifacts/audit-model"
    torch.save({"weight": torch.tensor([2.0])}, bundle / "weights.pt")
    metadata = STUDY["read"](bundle / "metadata.json")
    metadata["weights_sha256"] = STUDY["digest"](bundle / "weights.pt")
    metadata["artifact_digest"] = hashlib.sha256(
        canonical_json({key: value for key, value in metadata.items() if key != "artifact_digest"})
    ).hexdigest()
    flow.save(bundle / "metadata.json", metadata)
    calls = flow.calls.copy()
    with pytest.raises(ValueError, match="tensors differ"):
        flow.invoke(finalize=True)
    assert flow.calls == calls
