"""Crash-after-publication tests using tiny CPU training and real detection metrics.

The injected model is a deterministic test fixture, never an evaluated/released
detector. The real training loop, optimizer, scheduler, checkpoint writer, metrics,
and resume publication reconciliation run unchanged.
"""

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import inspection_dl.train as training


class PublicationInterrupted(RuntimeError):
    pass


class TinyDetector(torch.nn.Module):
    def __init__(self, *, early_best=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.early_best = early_best
        self.training_forwards = 0

    def forward(self, images, targets=None):
        if targets is not None:
            self.training_forwards += 1
            return {"classification": (self.weight - 1).square()}
        weight = self.weight.detach().clamp(0, 1)
        # AdamW makes the parameter rise monotonically. Real IoU/AP therefore
        # improves each epoch, or peaks in epoch zero for the older-best cases.
        end = 16 + (weight - 0.5).clamp(min=0) * 20 if self.early_best else 24 - 8 * weight
        box = torch.stack([weight * 0 + 4, weight * 0 + 4, end, end]).reshape(1, 4)
        return [
            {
                "boxes": box.clone(),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.85]),
            }
            for _ in images
        ]


@pytest.fixture
def tiny_study(tmp_path, monkeypatch):
    records = []
    for index, split in enumerate(("train", "train", "val")):
        image = tmp_path / f"synthetic_{index}.png"
        Image.new("RGB", (24, 24), color=(index * 50, 7, 11)).save(image)
        records.append(
            {
                "id": f"fixture-{index}",
                "image": image.name,
                "width": 24,
                "height": 24,
                "sha256": training.file_digest(image),
                "group": f"independent-fixture-{index}",
                "split": split,
                "boxes": [[4, 4, 16, 16]],
                "labels": [1],
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "synthetic-publication-test-only",
                "taxonomy_version": "synthetic-fixture-v1",
                "classes": [{"id": 1, "name": "synthetic-fixture"}],
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "manifest": str(manifest),
        "data_root": str(tmp_path),
        "epochs": 3,
        "batch_size": 1,
        "lr": 0.2,
        "seed": 97,
        "threads": 1,
        "device": "cpu",
        "horizontal_flip": False,
        "freeze_backbone": False,
        "amp": False,
        "no_pretrained": True,
        "resume": None,
    }
    constructed = []

    def install_model(*, early_best=False):
        def factory(*args, **kwargs):
            model = TinyDetector(early_best=early_best)
            constructed.append(model)
            return model

        monkeypatch.setattr(training, "build_detector", factory)

    install_model()
    return tmp_path, settings, constructed, install_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def args_for(settings, directory, *, resume=False):
    return Namespace(
        **{
            **settings,
            "output": str(directory),
            "resume": str(directory / "last.pt") if resume else None,
        }
    )


def scientific_report(report):
    return {
        key: value
        for key, value in report.items()
        if key not in {"checkpoint_sha256", "training_run_id"}
    }


def assert_consistent_publication(directory):
    latest = load_checkpoint(directory / "last.pt")
    best = load_checkpoint(directory / "best.pt")
    chosen = max(latest["history"], key=lambda row: row["validation_map50_95"])
    assert best["epoch"] == chosen["epoch"]
    assert best["best_ap"] == chosen["validation_map50_95"]
    assert training.model_state_digest(best["model"]) == chosen["model_state_sha256"]
    report = read_json(directory / "best_validation.json")
    predictions = read_json(directory / "best_validation_predictions.json")
    assert report["epoch"] == best["epoch"]
    assert report["map50_95"] == chosen["validation_map50_95"]
    assert report["map50"] == chosen["validation_map50"]
    assert report["model_state_sha256"] == training.model_state_digest(best["model"])
    assert report["checkpoint_sha256"] == training.file_digest(directory / "best.pt")
    assert report["predictions_sha256"] == training.json_digest(predictions)
    assert predictions == best["validation_predictions"]
    assert scientific_report(report) == scientific_report(best["validation_report"])
    assert read_json(directory / "history.json")["epochs"] == latest["history"]
    provenance = read_json(directory / "run.json")
    assert provenance["completed"] is True
    assert provenance["best_checkpoint_sha256"] == training.file_digest(directory / "best.pt")
    return latest, best, report, predictions, provenance


@pytest.mark.parametrize(
    "boundary",
    [
        "last.pt",
        "best.pt",
        "best_validation.json",
        "best_validation_predictions.json",
        "history.json",
    ],
)
@pytest.mark.parametrize("interrupted_epoch", [0, 1, 2])
def test_resume_reconciles_crash_after_each_publication(
    tiny_study, monkeypatch, boundary, interrupted_epoch
):
    root, settings, constructed, _ = tiny_study
    reference = root / "uninterrupted"
    training.train(args_for(settings, reference))
    expected_last, expected_best, expected_report, expected_predictions, _ = (
        assert_consistent_publication(reference)
    )
    assert [row["selected"] for row in expected_last["history"]] == [True, True, True]
    interrupted = root / "interrupted"
    original_checkpoint = training.save_checkpoint
    original_json = training.atomic_json
    fault_triggered = False

    def fail_after_write(path, payload, original):
        nonlocal fault_triggered
        original(path, payload)
        if Path(path).name != boundary:
            return
        epoch = load_checkpoint(interrupted / "last.pt")["epoch"]
        if epoch == interrupted_epoch and not fault_triggered:
            fault_triggered = True
            raise PublicationInterrupted(f"crash after {boundary} epoch {epoch}")

    with monkeypatch.context() as injection:
        injection.setattr(
            training,
            "save_checkpoint",
            lambda path, payload: fail_after_write(path, payload, original_checkpoint),
        )
        injection.setattr(
            training,
            "atomic_json",
            lambda path, payload: fail_after_write(path, payload, original_json),
        )
        with pytest.raises(PublicationInterrupted, match=f"crash after {boundary}"):
            training.train(args_for(settings, interrupted))
    assert fault_triggered
    committed = load_checkpoint(interrupted / "last.pt")
    assert committed["epoch"] == interrupted_epoch
    committed_digest = training.file_digest(interrupted / "last.pt")
    training.train(args_for(settings, interrupted, resume=True))
    actual_last, actual_best, actual_report, actual_predictions, provenance = (
        assert_consistent_publication(interrupted)
    )
    assert constructed[-1].training_forwards == (2 - interrupted_epoch) * 2
    assert provenance["resumed_from"]["epoch"] == interrupted_epoch
    assert provenance["resumed_from"]["checkpoint_sha256"] == committed_digest
    for expected, actual in ((expected_last, actual_last), (expected_best, actual_best)):
        for name, tensor in expected["model"].items():
            torch.testing.assert_close(actual["model"][name], tensor, rtol=0, atol=0)
        assert expected["scheduler"] == actual["scheduler"]
        assert expected["optimizer"]["param_groups"] == actual["optimizer"]["param_groups"]
        for parameter, state in expected["optimizer"]["state"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(
                    actual["optimizer"]["state"][parameter][name], tensor, rtol=0, atol=0
                )
    assert scientific_report(actual_report) == scientific_report(expected_report)
    assert actual_predictions == expected_predictions
    if boundary != "history.json":
        recovery = read_json(interrupted / "recovery.json")
        assert recovery["recoveries"]
        assert provenance["recovery"] == recovery
        assert provenance["recovery_sha256"] == training.file_digest(interrupted / "recovery.json")


@pytest.mark.parametrize("damage", ["missing", "stale"])
def test_resume_refuses_irrecoverable_older_best_weights(tiny_study, damage):
    root, settings, _, install_model = tiny_study
    install_model(early_best=True)
    run = root / "older-best"
    training.train(args_for(settings, run))
    latest, best, _, _, _ = assert_consistent_publication(run)
    assert latest["epoch"] == 2 and best["epoch"] == 0
    if damage == "missing":
        (run / "best.pt").unlink()
    else:
        # last.pt has the right architecture/config but cannot replace the
        # distinct model that earned the earlier canonical best metric.
        training.save_checkpoint(run / "best.pt", latest)
    with pytest.raises(ValueError, match="canonical older best checkpoint.*cannot be recovered"):
        training.train(args_for(settings, run, resume=True))


def test_resume_refuses_conflicting_published_scientific_metrics(tiny_study):
    root, settings, _, _ = tiny_study
    run = root / "conflicting-report"
    training.train(args_for(settings, run))
    original = read_json(run / "best_validation.json")
    conflicting = {**original, "map50_95": 0.123456789}
    training.atomic_json(run / "best_validation.json", conflicting)
    with pytest.raises(ValueError, match="published validation report conflicts"):
        training.train(args_for(settings, run, resume=True))
    assert read_json(run / "best_validation.json") == conflicting


def test_resume_refuses_best_metric_that_disagrees_with_committed_history(tiny_study):
    root, settings, _, _ = tiny_study
    run = root / "conflicting-history"
    training.train(args_for(settings, run))
    latest = load_checkpoint(run / "last.pt")
    latest["best_ap"] = 0.123456789
    training.save_checkpoint(run / "last.pt", latest)
    with pytest.raises(ValueError, match="best_ap disagrees with canonical validation history"):
        training.train(args_for(settings, run, resume=True))


def test_resume_repairs_sidecars_for_an_older_verified_best(tiny_study):
    root, settings, _, install_model = tiny_study
    install_model(early_best=True)
    run = root / "older-best-sidecars"
    training.train(args_for(settings, run))
    _, original_best, original_report, original_predictions, _ = assert_consistent_publication(run)
    for name in ("best_validation.json", "best_validation_predictions.json", "history.json"):
        (run / name).unlink()
    training.train(args_for(settings, run, resume=True))
    _, repaired_best, report, predictions, _ = assert_consistent_publication(run)
    assert repaired_best["epoch"] == original_best["epoch"] == 0
    assert report == original_report and predictions == original_predictions
    repaired_names = {
        item["file"] for item in read_json(run / "recovery.json")["recoveries"][-1]["repairs"]
    }
    assert repaired_names == {
        "best_validation.json",
        "best_validation_predictions.json",
        "history.json",
    }
