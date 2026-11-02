"""A finalization must reject substituted candidates before accessing holdout."""

import json
import runpy
from pathlib import Path

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
