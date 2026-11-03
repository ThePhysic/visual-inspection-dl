"""Resume must retain the provenance of weights learned before the resume."""

import copy
import json
import runpy
from pathlib import Path

import pytest

import inspection_dl.train as training

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_checkpoint_interruptions.py")))


@pytest.fixture
def provenance_study(tmp_path, monkeypatch):
    root, settings, _, _ = HELPERS["tiny_study"].__wrapped__(tmp_path, monkeypatch)
    settings["no_pretrained"] = False
    cache = root / "cache"
    weight = cache / "checkpoints" / training.PRETRAINED_URL.rsplit("/", 1)[-1]
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"synthetic initialization metadata fixture")
    monkeypatch.setattr(training.torch.hub, "get_dir", lambda: str(cache))
    monkeypatch.setattr(training, "source_provenance", lambda: {"revision": "initial-source"})
    output = root / "experiment"
    training.train(HELPERS["args_for"](settings, output))
    initial = json.loads((output / "run.json").read_text())
    return settings, output, weight, initial


def test_zero_step_resumes_preserve_origin_and_record_each_execution(provenance_study, monkeypatch):
    settings, output, cache, initial = provenance_study
    cache.write_bytes(b"different cache contents; not used by resume")
    monkeypatch.setattr(training, "source_provenance", lambda: {"revision": "resume-one"})
    training.train(HELPERS["args_for"](settings, output, resume=True))
    cache.unlink()
    monkeypatch.setattr(training, "source_provenance", lambda: {"revision": "resume-two"})
    training.train(HELPERS["args_for"](settings, output, resume=True))
    resumed = json.loads((output / "run.json").read_text())
    for field in ("source", "environment", "pretrained", "initial_provenance"):
        assert resumed[field] == initial[field]
    assert [event["source"]["revision"] for event in resumed["resume_history"]] == [
        "resume-one",
        "resume-two",
    ]
    assert resumed["execution"]["source"]["revision"] == "resume-two"


def test_checkpoint_recovers_origin_when_run_json_is_missing(provenance_study, monkeypatch):
    settings, output, cache, initial = provenance_study
    (output / "run.json").unlink()
    cache.unlink()
    monkeypatch.setattr(training, "source_provenance", lambda: {"revision": "recovery-source"})
    training.train(HELPERS["args_for"](settings, output, resume=True))
    resumed = json.loads((output / "run.json").read_text())
    assert resumed["initial_provenance"] == initial["initial_provenance"]
    assert resumed["source"]["revision"] == "initial-source"
    assert resumed["execution"]["source"]["revision"] == "recovery-source"


def test_conflicting_saved_origins_are_rejected(provenance_study):
    settings, output, _, initial = provenance_study
    altered = copy.deepcopy(initial)
    altered["initial_provenance"]["source"] = {"revision": "unrelated-experiment"}
    training.atomic_json(output / "run.json", altered)
    with pytest.raises(ValueError, match="disagree on initialization provenance"):
        training.train(HELPERS["args_for"](settings, output, resume=True))
    assert json.loads((output / "run.json").read_text()) == altered


def test_missing_legacy_origin_is_explicit_instead_of_fabricated(provenance_study, monkeypatch):
    settings, output, _, _ = provenance_study
    checkpoint = HELPERS["load_checkpoint"](output / "last.pt")
    checkpoint.pop("run_provenance")
    training.save_checkpoint(output / "last.pt", checkpoint)
    (output / "run.json").unlink()
    monkeypatch.setattr(training, "source_provenance", lambda: {"revision": "legacy-resume"})
    training.train(HELPERS["args_for"](settings, output, resume=True))
    resumed = json.loads((output / "run.json").read_text())
    assert resumed["source"]["revision"] is None
    assert resumed["pretrained"]["sha256"] is None
    assert resumed["pretrained"]["initialization"].startswith("unknown:")
    assert resumed["execution"]["source"]["revision"] == "legacy-resume"
