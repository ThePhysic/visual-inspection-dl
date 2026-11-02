"""Measurements cannot silently combine checkpoints under one release identity."""

import copy
from types import SimpleNamespace

import pytest

from inspection_dl.benchmark import bind_model_identity
from inspection_dl.bundle import Taxon


def metadata():
    return SimpleNamespace(
        artifact_digest="a" * 64,
        weights_sha256="b" * 64,
        model_version="release-1",
        taxonomy_version="tax-1",
        dataset_id="data-1",
        classes=[Taxon(id=1, name="defect")],
    )


@pytest.mark.parametrize(
    "field",
    [
        "artifact_digest",
        "weights_sha256",
        "model_version",
        "taxonomy_version",
        "dataset_id",
        "classes",
    ],
)
def test_benchmark_refuses_to_mix_release_identities(field):
    original = metadata()
    output = {}
    identity = bind_model_identity(output, original)
    assert identity["artifact_digest"] == output["artifact_digest"]
    assert output["classes"] == [{"id": 1, "name": "defect"}]
    assert bind_model_identity(output, copy.deepcopy(original)) == identity
    changed = copy.deepcopy(original)
    setattr(changed, field, [Taxon(id=1, name="other")] if field == "classes" else "different")
    with pytest.raises(ValueError, match="identity changed"):
        bind_model_identity(output, changed)
