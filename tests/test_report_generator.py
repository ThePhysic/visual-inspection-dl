"""Documentation must reject benchmark evidence from a substituted release."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

REPORT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "summarize_study.py"))
validate_benchmark_identity = REPORT["validate_benchmark_identity"]


def identity_fixture():
    metadata = {
        "artifact_digest": "a" * 64,
        "weights_sha256": "b" * 64,
        "model_version": "released-detector",
        "taxonomy_version": "defects-v1",
        "dataset_id": "audited-dataset",
        "dataset_manifest_sha256": "c" * 64,
        "classes": [{"id": 1, "name": "defect"}],
    }
    identity = {
        key: metadata[key]
        for key in (
            "artifact_digest",
            "weights_sha256",
            "model_version",
            "taxonomy_version",
            "dataset_id",
        )
    }
    benchmark = {
        **identity,
        "manifest_sha256": metadata["dataset_manifest_sha256"],
        "classes": deepcopy(metadata["classes"]),
        "benchmark_source": {
            "revision": "d" * 40,
            "package_source_sha256": "e" * 64,
            "dirty": False,
            "git_available": True,
        },
        "results": [
            {"device": "cpu", "batch_size": 1, **identity},
            {"device": "cuda", "unavailable": True},
        ],
    }
    return metadata, benchmark


def test_release_bound_benchmark_accepts_matching_identity():
    metadata, benchmark = identity_fixture()
    validate_benchmark_identity(metadata, benchmark)


@pytest.mark.parametrize("field", ["artifact_digest", "weights_sha256", "model_version"])
def test_substituted_release_with_identical_dataset_is_rejected(field):
    metadata, benchmark = identity_fixture()
    benchmark[field] = "substituted"
    with pytest.raises(ValueError, match=field):
        validate_benchmark_identity(metadata, benchmark)


@pytest.mark.parametrize(
    "field",
    ["artifact_digest", "weights_sha256", "model_version", "taxonomy_version", "dataset_id"],
)
@pytest.mark.parametrize("change", ["substitute", "omit"])
def test_each_available_configuration_must_identify_the_release(field, change):
    metadata, benchmark = identity_fixture()
    if change == "omit":
        benchmark["results"][0].pop(field)
    else:
        benchmark["results"][0][field] = "different loaded model"
    with pytest.raises(ValueError, match=f"configuration 0.*{field}"):
        validate_benchmark_identity(metadata, benchmark)


def test_historical_dataset_only_benchmark_cannot_be_release_evidence():
    metadata, benchmark = identity_fixture()
    historical = {key: benchmark[key] for key in ("manifest_sha256", "results")}
    with pytest.raises(ValueError, match="artifact_digest"):
        validate_benchmark_identity(metadata, historical)


@pytest.mark.parametrize(
    "field,substitute",
    [("manifest_sha256", "changed manifest"), ("classes", [{"id": 1, "name": "wrong defect"}])],
)
def test_manifest_and_class_mapping_must_match_release(field, substitute):
    metadata, benchmark = identity_fixture()
    benchmark[field] = substitute
    with pytest.raises(ValueError, match=field):
        validate_benchmark_identity(metadata, benchmark)


def test_benchmark_requires_actual_source_identity():
    metadata, benchmark = identity_fixture()
    benchmark.pop("benchmark_source")
    with pytest.raises(ValueError, match="source identity"):
        validate_benchmark_identity(metadata, benchmark)


@pytest.fixture
def completed_evidence(monkeypatch):
    """Use frozen reports; stub Git reconstruction so shallow CPU CI can run this check."""
    root = Path(__file__).parents[1]
    args = SimpleNamespace(
        repo_root=root,
        study=Path("reports/rtx4060-study-v2"),
        release=Path("releases/pcb-ind-ssdlite-v1"),
        manifest=Path("data/manifests/pcb-ind-v4.json"),
    )
    benchmark = json.loads((root / args.study / "benchmark-release.json").read_bytes())
    globals_ = REPORT["load_study"].__globals__
    monkeypatch.setitem(
        globals_,
        "git_snapshot",
        lambda *args: (benchmark["benchmark_source"]["package_source_sha256"], {}, {}),
    )
    monkeypatch.setitem(globals_, "reconstruct_sources", lambda *args: [])
    return args


def test_fresh_study_does_not_require_or_claim_a_historical_benchmark(
    completed_evidence, monkeypatch, tmp_path
):
    original_path = REPORT["Evidence"].path

    def hide_historical(self, relative):
        if Path(relative) == completed_evidence.study / "benchmark.json":
            return tmp_path / "absent-historical-benchmark.json"
        return original_path(self, relative)

    monkeypatch.setattr(REPORT["Evidence"], "path", hide_historical)
    context = REPORT["load_study"](completed_evidence)
    assert context["historical_benchmark"] is None
    assert "earlier [benchmark.json]" not in REPORT["model_card"](context)
    assert "original `benchmark.json` remains historical" not in REPORT["reproducibility"](context)
    assert "benchmark-release.json" in REPORT["model_card"](context)


@pytest.mark.parametrize("field", ["best_checkpoint_sha256", "run_id"])
def test_candidate_identity_must_match_its_completed_run(completed_evidence, monkeypatch, field):
    original_read = REPORT["Evidence"].read

    def substitute_run(self, relative):
        result = original_read(self, relative)
        if Path(relative) == completed_evidence.study / "experiments/flip/run.json":
            result[field] = "substituted"
        return result

    monkeypatch.setattr(REPORT["Evidence"], "read", substitute_run)
    with pytest.raises(ValueError, match="Candidate checkpoint/run identity"):
        REPORT["load_study"](completed_evidence)
