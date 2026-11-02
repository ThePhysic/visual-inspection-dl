import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inspection_dl.api import FixtureBackend, MAX_UPLOAD_BYTES, create_app


def image_bytes(size=(80, 40), image_format="PNG", exif=None):
    stream = io.BytesIO()
    image = Image.new("RGB", size, "white")
    image.save(stream, format=image_format, **({"exif": exif} if exif is not None else {}))
    return stream.getvalue()


@pytest.fixture
def client():
    with TestClient(create_app(backend="fixture")) as test_client:
        yield test_client


def test_fixture_success_and_provenance(client):
    response = client.post(
        "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "fixture" and body["simulated"] is True
    assert body["model_version"] == "simulated-fixture-v1"
    assert body["invalid_predictions_rejected"] == 0
    assert body["coordinate_system"] == "xyxy_pixels_original_image"
    assert body["image"] == {"width": 80, "height": 40}
    assert body["detections"][0]["box"] == {"x_min": 20, "y_min": 10, "x_max": 60, "y_max": 30}
    assert body["detections"][0]["label"] == "simulated-defect"
    assert set(body["timing_ms"]) == {
        "decode_preprocess",
        "model_inference",
        "response_postprocess",
        "request_total",
    }
    assert all(value >= 0 for value in body["timing_ms"].values())


def test_empty_detections_are_success():
    with TestClient(create_app(backend=FixtureBackend(empty=True))) as client:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 200 and response.json()["detections"] == []


def test_threshold_is_applied_and_recorded(client):
    response = client.post(
        "/v1/predict?threshold=0.8", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert response.json()["confidence_threshold"] == 0.8


@pytest.mark.parametrize("threshold", ["-0.1", "1.1", "nan", "inf", "bad"])
def test_invalid_threshold_rejected(client, threshold):
    response = client.post(
        f"/v1/predict?threshold={threshold}",
        content=image_bytes(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 422


def test_health_readiness_metadata(client):
    assert client.get("/health").json()["status"] == "alive"
    assert client.get("/ready").json()["ready"] is True
    assert client.get("/v1/model").json()["simulated"] is True


def test_missing_bundle_is_unready_and_never_falls_back(tmp_path):
    with TestClient(create_app(tmp_path / "private-missing-checkpoint")) as client:
        assert client.get("/health").status_code == 200
        readiness = client.get("/ready")
        assert readiness.status_code == 503 and readiness.json()["ready"] is False
        assert client.get("/v1/model").status_code == 503
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert "private-missing-checkpoint" not in response.text


@pytest.mark.parametrize("media_type", ["application/json", "image/gif", "", "multipart/form-data"])
def test_unsupported_media_type(client, media_type):
    response = client.post(
        "/v1/predict", content=image_bytes(), headers={"Content-Type": media_type}
    )
    assert response.status_code == 415


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_corrupt_image_predictable_error(client, payload):
    response = client.post("/v1/predict", content=payload, headers={"Content-Type": "image/png"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_actual_format_must_match_declared_type(client):
    response = client.post(
        "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/jpeg"}
    )
    assert response.status_code == 415


def test_upload_limit_with_and_without_content_length(client):
    response = client.post(
        "/v1/predict",
        content=b"x",
        headers={"Content-Type": "image/png", "Content-Length": str(MAX_UPLOAD_BYTES + 1)},
    )
    assert response.status_code == 413
    response = client.post(
        "/v1/predict",
        content=iter([b"x" * (MAX_UPLOAD_BYTES // 2), b"x" * (MAX_UPLOAD_BYTES // 2 + 1)]),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize("length", ["-1", "garbage"])
def test_bad_content_length(client, length):
    response = client.post(
        "/v1/predict", content=b"x", headers={"Content-Type": "image/png", "Content-Length": length}
    )
    assert response.status_code == 400


def test_declared_length_must_match_actual_bytes(client):
    response = client.post(
        "/v1/predict",
        content=image_bytes(),
        headers={"Content-Type": "image/png", "Content-Length": "1"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_length_mismatch"


def test_pixel_limit_checked_before_decode(client):
    response = client.post(
        "/v1/predict", content=image_bytes((4001, 4000)), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 413


def test_exif_orientation_uses_oriented_original_frame(client):
    exif = Image.Exif()
    exif[274] = 6
    response = client.post(
        "/v1/predict",
        content=image_bytes((80, 40), "JPEG", exif),
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 200
    assert response.json()["image"] == {"width": 40, "height": 80}
    assert response.json()["detections"][0]["box"]["x_max"] == 30


def test_busy_request_is_bounded(client):
    lock = client.app.state.model_lock
    lock.acquire()
    try:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 503 and response.json()["error"]["code"] == "inference_busy"
    finally:
        lock.release()


class BadBackend(FixtureBackend):
    def __init__(self, output=None, error=None):
        super().__init__()
        self.output, self.error = output, error

    def infer(self, image):
        if self.error is not None:
            raise self.error
        return self.output, 0.0, 0.1


@pytest.mark.parametrize(
    "output",
    [
        {"boxes": [[1, 1, 10, 10]], "labels": [99], "scores": [0.8]},
        {"boxes": [[1, 1, 10, 10]], "labels": [1], "scores": [float("nan")]},
        {"boxes": [[1, 1, 10, 10]], "labels": [1], "scores": [-0.1]},
        {"boxes": [[1, 1, 10, 10]], "labels": [1], "scores": [1.1]},
        {"boxes": [[1, 1, 10]], "labels": [1], "scores": [0.8]},
        {"boxes": [], "labels": [1], "scores": [0.8]},
        {"boxes": [[80, 1, 80, 10]], "labels": [99], "scores": [0.8]},
        {"boxes": [[80, 1, 80, 10]], "labels": [1], "scores": [1.1]},
    ],
)
def test_invalid_backend_output_is_error(output):
    with TestClient(create_app(backend=BadBackend(output))) as client:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_failed"
        assert "detections" not in response.json()


@pytest.mark.parametrize(
    "box",
    [
        [80, 5, 80, 30],
        [10, 5, 5, 30],
        [-1, 5, 30, 30],
        [10, 5, 81, 30],
        [10, 5, 30, 41],
        [float("nan"), 5, 30, 30],
        [10, 5, float("inf"), 30],
    ],
)
def test_invalid_geometry_is_explicitly_counted_and_discarded(box):
    selected = BadBackend({"boxes": [box], "labels": [1], "scores": [0.8]})
    with TestClient(create_app(backend=selected)) as client:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 200
        assert response.json()["detections"] == []
        assert response.json()["invalid_predictions_rejected"] == 1


class ClippedBoundaryBackend(FixtureBackend):
    def __init__(self, all_invalid):
        super().__init__()
        self.all_invalid = all_invalid

    def infer(self, image):
        import torch
        from torchvision.ops import clip_boxes_to_image

        boxes = torch.tensor(
            [[10, 5, 40, 30], [image.width + 1, 5, image.width + 5, 30]], dtype=torch.float32
        )
        if self.all_invalid:
            boxes = boxes[1:]
        clipped = clip_boxes_to_image(boxes, (image.height, image.width))
        return (
            {
                "boxes": clipped,
                "labels": torch.ones(len(clipped), dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.0005] if not self.all_invalid else [0.0005]),
            },
            0.0,
            0.1,
        )


@pytest.mark.parametrize("all_invalid", [False, True])
def test_torchvision_boundary_clipping_preserves_valid_predictions(all_invalid):
    with TestClient(create_app(backend=ClippedBoundaryBackend(all_invalid))) as client:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 200
        assert response.json()["invalid_predictions_rejected"] == 1
        assert len(response.json()["detections"]) == (0 if all_invalid else 1)
        if not all_invalid:
            assert response.json()["detections"][0]["box"] == {
                "x_min": 10,
                "y_min": 5,
                "x_max": 40,
                "y_max": 30,
            }


def test_resource_failure_is_explicit_and_lock_recovers():
    selected = BadBackend(error=MemoryError("sensitive-local-details"))
    with TestClient(create_app(backend=selected)) as client:
        response = client.post(
            "/v1/predict", content=image_bytes(), headers={"Content-Type": "image/png"}
        )
        assert (
            response.status_code == 503 and response.json()["error"]["code"] == "resource_exhausted"
        )
        assert "sensitive-local-details" not in response.text
        assert client.app.state.model_lock.acquire(blocking=False)
        client.app.state.model_lock.release()
