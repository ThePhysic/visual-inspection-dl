import asyncio
import io
import threading
from pathlib import Path

import anyio
import httpx
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


@pytest.mark.parametrize(
    "corruption", ["missing_iend", "missing_iend_crc", "bad_idat_crc", "bad_iend_crc"]
)
def test_png_requires_complete_chunks_and_valid_crcs(client, corruption):
    payload = bytearray(image_bytes())
    if corruption == "missing_iend":
        del payload[-12:]
    elif corruption == "missing_iend_crc":
        del payload[-4:]
    elif corruption == "bad_idat_crc":
        payload[-13] ^= 1
    else:
        payload[-1] ^= 1
    response = client.post(
        "/v1/predict", content=bytes(payload), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


@pytest.mark.parametrize("source", ["generated", "pcb_fixture", "pcb_fixture_embedded_eoi"])
def test_jpeg_requires_complete_pixel_stream(client, source):
    payload = image_bytes(image_format="JPEG")
    if source != "generated":
        payload = (Path(__file__).parent / "fixtures/pcb-ind/5810_t_007.jpg").read_bytes()
    if source == "pcb_fixture_embedded_eoi":
        # APP15 metadata contains marker-looking bytes, not the image's actual EOI.
        payload = payload[:2] + b"\xff\xef\x00\x04\xff\xd9" + payload[2:]
    response = client.post(
        "/v1/predict", content=payload[:-2],
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


@pytest.mark.parametrize("progressive", [False, True])
def test_valid_jpeg_with_trailing_bytes_remains_accepted(client, progressive):
    stream = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(stream, format="JPEG", progressive=progressive)
    response = client.post(
        "/v1/predict", content=stream.getvalue() + b"trailing bytes",
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 200
    assert response.json()["image"] == {"width": 80, "height": 40}


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


def test_cancelled_request_keeps_worker_exclusive_until_completion():
    class BlockingBackend(FixtureBackend):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.active = 0
            self.max_active = 0

        def infer(self, image):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.set()
            try:
                if not self.release.wait(5):
                    raise TimeoutError("test did not release worker")
                return super().infer(image)
            finally:
                self.active -= 1

    async def scenario():
        selected = BlockingBackend()
        app = create_app(backend=selected)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                request_args = {
                    "content": image_bytes(), "headers": {"Content-Type": "image/png"}
                }
                pending = asyncio.create_task(client.post("/v1/predict", **request_args))
                try:
                    assert await asyncio.to_thread(selected.entered.wait, 5)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert selected.active == 1
                    response = await client.post("/v1/predict", **request_args)
                    assert response.status_code == 503
                    assert response.json()["error"]["code"] == "inference_busy"
                finally:
                    selected.release.set()
                    if not pending.done():
                        pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await pending
                    # Wait for the actual worker's finally, including postprocessing.
                    acquired = await asyncio.to_thread(app.state.execution_lock.acquire, timeout=5)
                    assert acquired
                    app.state.execution_lock.release()
                response = await client.post("/v1/predict", **request_args)
                assert response.status_code == 200
                assert selected.max_active == 1

    asyncio.run(scenario())


def test_cancellation_before_worker_start_does_not_strand_lock(monkeypatch):
    from inspection_dl import api

    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        old_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        occupied, release = threading.Event(), threading.Event()
        scheduled = asyncio.Event()
        original_runner = api.run_in_threadpool

        def occupy_pool():
            occupied.set()
            if not release.wait(5):
                raise TimeoutError("test did not release thread pool")

        async def announce_worker(*args, **kwargs):
            scheduled.set()
            return await original_runner(*args, **kwargs)

        monkeypatch.setattr(api, "run_in_threadpool", announce_worker)
        blocker = asyncio.create_task(anyio.to_thread.run_sync(occupy_pool))
        pending = None
        try:
            assert await asyncio.to_thread(occupied.wait, 5)
            app = create_app(backend="fixture")
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    request_args = {
                        "content": image_bytes(), "headers": {"Content-Type": "image/png"}
                    }
                    pending = asyncio.create_task(client.post("/v1/predict", **request_args))
                    async with asyncio.timeout(5):
                        await scheduled.wait()
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    release.set()
                    await blocker
                    response = await client.post("/v1/predict", **request_args)
                    assert response.status_code == 200
        finally:
            release.set()
            if pending is not None and not pending.done():
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            await blocker
            limiter.total_tokens = old_tokens

    asyncio.run(scenario())


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
