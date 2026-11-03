"""CPU API smoke of the shipped model on an attributed PCB-IND training fixture.

This checks serving and release integrity, not held-out detection quality. It never
scores annotations, evaluates the test split, selects thresholds, or downloads weights.
"""

import asyncio
import json
import math
import socket
import urllib.request
from pathlib import Path

import httpx
import pytest
import torch
import torchvision.models._api

from inspection_dl.api import TrainedBackend, create_app
from inspection_dl.bundle import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE = REPOSITORY / "releases/pcb-ind-ssdlite-v1"
FIXTURE = REPOSITORY / "tests/fixtures/pcb-ind"


def test_checked_in_release_cpu_training_fixture_smoke(monkeypatch, tmp_path):
    metadata = json.loads((RELEASE / "metadata.json").read_text(encoding="utf-8"))
    fixture = json.loads((FIXTURE / "fixture.json").read_text(encoding="utf-8"))
    image_path = FIXTURE / fixture["image"]
    assert sha256_file(image_path) == fixture["sha256"]
    assert fixture["taxonomy_version"] == metadata["taxonomy_version"]
    assert fixture["classes"] == metadata["classes"]

    def forbidden_network_or_download(*args, **kwargs):
        pytest.fail("Released-model CPU smoke attempted network access or a model download")

    def forbidden_cuda(*args, **kwargs):
        pytest.fail("Released-model CPU smoke attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda)

    original_load = torch.load
    loaded_paths = []

    def release_weights_only(path, *args, **kwargs):
        # No training checkpoint, external weights, or GPU deserialization is allowed.
        assert Path(path).resolve() == RELEASE / "weights.pt"
        assert kwargs.get("map_location") == "cpu"
        assert kwargs.get("weights_only") is True
        loaded_paths.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", release_weights_only)
    # Resolve checked-in inputs independently of cwd; CI needs no raw dataset or runs.
    monkeypatch.chdir(tmp_path)

    async def scenario():
        # Initialize the event loop before these guards: Windows uses a loopback
        # wakeup socket internally. ASGI itself needs no TCP connection. All guards
        # apply before app startup, including warm-cache model downloader calls.
        for target, names in (
            (socket, ("create_connection", "getaddrinfo")),
            (socket.socket, ("connect", "connect_ex")),
            (urllib.request, ("urlopen", "urlretrieve")),
            (torch.hub, ("download_url_to_file", "load_state_dict_from_url")),
            (torchvision.models._api, ("load_state_dict_from_url",)),
        ):
            for name in names:
                monkeypatch.setattr(target, name, forbidden_network_or_download)

        app = create_app(RELEASE, backend="trained", device="cpu")
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            ready = await client.get("/ready")
            assert ready.status_code == 200
            assert ready.json() == {
                "ready": True,
                "backend": "trained",
                "simulated": False,
                "model_version": metadata["model_version"],
            }
            backend = app.state.backend
            assert isinstance(backend, TrainedBackend)
            assert backend.bundle.device == "cpu"
            assert not backend.bundle.model.training
            assert all(
                tensor.device.type == "cpu"
                for tensor in (
                    *backend.bundle.model.parameters(),
                    *backend.bundle.model.buffers(),
                )
            )
            model = await client.get("/v1/model")
            assert model.status_code == 200
            assert model.json() == {**metadata, "backend": "trained", "simulated": False}

            response = await client.post(
                "/v1/predict",
                content=image_path.read_bytes(),
                headers={"Content-Type": "image/jpeg"},
            )
            assert response.status_code == 200
            return response.json()

    original_threads = torch.get_num_threads()
    try:
        body = asyncio.run(scenario())
    finally:
        torch.set_num_threads(original_threads)

    assert loaded_paths == [RELEASE / "weights.pt"]
    assert body["backend"] == "trained" and body["simulated"] is False
    for key in (
        "model_version",
        "taxonomy_version",
        "artifact_digest",
        "coordinate_system",
        "confidence_threshold",
        "nms_threshold",
    ):
        assert body[key] == metadata[key]
    assert body["image"] == {"width": fixture["width"], "height": fixture["height"]}
    assert isinstance(body["invalid_predictions_rejected"], int)
    assert body["invalid_predictions_rejected"] >= 0

    # Require an exercised detection contract, without fixing proposal counts or
    # floating-point coordinates/scores across Windows and Ubuntu CPU kernels.
    assert body["detections"]
    assert len(body["detections"]) <= metadata["max_detections"]
    taxonomy = {entry["id"]: entry["name"] for entry in metadata["classes"]}
    for detection in body["detections"]:
        assert detection["label"] == taxonomy[detection["class_id"]]
        confidence = detection["confidence"]
        assert math.isfinite(confidence)
        assert metadata["confidence_threshold"] <= confidence <= 1
        box = detection["box"]
        assert all(math.isfinite(value) for value in box.values())
        assert 0 <= box["x_min"] < box["x_max"] <= fixture["width"]
        assert 0 <= box["y_min"] < box["y_max"] <= fixture["height"]
    assert set(body["timing_ms"]) == {
        "decode_preprocess",
        "model_inference",
        "response_postprocess",
        "request_total",
    }
    assert all(math.isfinite(value) and value >= 0 for value in body["timing_ms"].values())
