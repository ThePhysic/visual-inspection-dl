"""Local standalone inference API; start with `python -m inspection_dl.api`."""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import math
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from inspection_dl.bundle import LoadedBundle, load_bundle
from inspection_dl.schemas import Box, Detection, ImageShape, Prediction, Timing

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
logger = logging.getLogger(__name__)


class InferenceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _milliseconds(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _complete_png_end(payload: bytes) -> bool:
    # Pillow.verify() checks chunk CRCs but stops before checking IEND's data/CRC.
    offset = 8
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        end = offset + length + 12
        if end > len(payload):
            return False
        if payload[offset + 4 : offset + 8] == b"IEND":
            return payload[offset:end] == b"\x00\x00\x00\x00IEND\xaeB`\x82"
        offset = end
    return False


def _complete_jpeg_end(payload: bytes) -> bool:
    # Skip metadata rather than mistaking an embedded thumbnail's EOI for the
    # enclosing image's end. Entropy data escapes FF bytes and has restart markers.
    offset, in_scan = 2, False
    while offset < len(payload):
        if in_scan:
            offset = payload.find(b"\xff", offset)
            if offset < 0:
                return False
        elif payload[offset] != 0xFF:
            return False
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset == len(payload):
            return False
        marker = payload[offset]
        offset += 1
        if marker == 0x00:
            if not in_scan:
                return False
            continue
        if marker == 0xD9:
            return True
        if marker == 0xD8:
            return False
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            return False
        length = int.from_bytes(payload[offset : offset + 2], "big")
        if length < 2 or offset + length > len(payload):
            return False
        offset += length
        # DNL may supply the height within an entropy-coded scan.
        in_scan = marker == 0xDA or (in_scan and marker == 0xDC)
    return False


def decode_image(payload: bytes, media_type: str) -> Image.Image:
    """Decode into the EXIF-oriented RGB original-image coordinate frame."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}[media_type]
                if image.format != expected_format:
                    raise InferenceError(
                        415, "media_type_mismatch", "Image bytes must match Content-Type."
                    )
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InferenceError(413, "image_too_large", "Image exceeds the pixel limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise InferenceError(
                        415, "unsupported_image", "Animated images are unsupported."
                    )
                image.verify()
                if expected_format == "PNG" and not _complete_png_end(payload):
                    raise ValueError("PNG has no complete, valid terminating chunk")
                if expected_format == "JPEG" and not _complete_jpeg_end(payload):
                    raise ValueError("JPEG has no complete terminating marker")
            # Verification invalidates the decoder; use a fresh one for pixel loading.
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except InferenceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InferenceError(413, "image_too_large", "Image exceeds the pixel limit.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InferenceError(400, "invalid_image", "Image could not be decoded.") from exc
    except MemoryError as exc:
        raise InferenceError(
            503, "resource_exhausted", "Image decoding resources are unavailable."
        ) from exc


class TrainedBackend:
    simulated = False
    name = "trained"

    def __init__(self, bundle: LoadedBundle):
        self.bundle = bundle
        self.metadata = bundle.metadata.model_dump(mode="json")

    def infer(self, image: Image.Image):
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        prep_start = time.perf_counter()
        tensor = pil_to_tensor(image).to(device=self.bundle.device, dtype=torch.float32).div_(255)
        if self.bundle.device.startswith("cuda"):
            torch.cuda.synchronize(self.bundle.device)
        preprocessing_ms = _milliseconds(prep_start)
        model_start = time.perf_counter()
        with torch.inference_mode():
            output = self.bundle.model([tensor])[0]
        if self.bundle.device.startswith("cuda"):
            torch.cuda.synchronize(self.bundle.device)
        return output, preprocessing_ms, _milliseconds(model_start)


class FixtureBackend:
    """Explicitly simulated backend; never used as an error fallback."""

    simulated = True
    name = "fixture"

    def __init__(self, empty: bool = False):
        self.empty = empty
        self.metadata = {
            "schema_version": "1",
            "model_version": "simulated-fixture-v1",
            "taxonomy_version": "simulated-fixture-v1",
            "artifact_digest": "simulated-no-model-artifact",
            "classes": [{"id": 1, "name": "simulated-defect"}],
            "confidence_threshold": 0.25,
            "nms_threshold": 0.55,
            "coordinate_system": "xyxy_pixels_original_image",
            "preprocessing": "exif-oriented-rgb-fixture",
            "max_detections": 1,
        }

    def infer(self, image: Image.Image):
        start = time.perf_counter()
        output = {
            "boxes": []
            if self.empty
            else [[image.width / 4, image.height / 4, image.width * 3 / 4, image.height * 3 / 4]],
            "labels": [] if self.empty else [1],
            "scores": [] if self.empty else [0.75],
        }
        return output, 0.0, _milliseconds(start)


def _detections(
    output: dict, classes: list[dict], threshold: float, width: int, height: int
) -> tuple[list[Detection], int]:
    """Discard/count invalid geometry; malformed taxonomy/scores remain backend bugs.

    Torchvision clips proposals to the original image. An otherwise legitimate
    proposal can become zero-area at a boundary, including below the operating
    threshold. Preserve valid detections while making every geometry discard visible.
    """
    taxonomy = {entry["id"]: entry["name"] for entry in classes}
    converted = {}
    for key in ("boxes", "labels", "scores"):
        value = output[key]
        converted[key] = value.tolist() if hasattr(value, "tolist") else value
    if not len(converted["boxes"]) == len(converted["labels"]) == len(converted["scores"]):
        raise ValueError("backend output lengths disagree")
    result, rejected = [], 0
    for box, label, score in zip(
        converted["boxes"], converted["labels"], converted["scores"], strict=True
    ):
        if label not in taxonomy or isinstance(label, bool) or int(label) != label:
            raise ValueError("backend returned an unknown class")
        if len(box) != 4:
            raise ValueError("backend box must have four coordinates")
        # Validate scores even when the candidate's geometry will be discarded.
        confidence = float(score)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("backend score must be finite and within [0,1]")
        if not all(math.isfinite(coordinate) for coordinate in box) or not (
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
        ):
            rejected += 1
            continue
        detection = Detection(
            class_id=label,
            label=taxonomy[label],
            confidence=confidence,
            box=Box(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        )
        if detection.confidence >= threshold:
            result.append(detection)
    return result, rejected


def create_app(
    bundle_path: str | Path | None = None,
    *,
    backend: str | FixtureBackend = "trained",
    device: str = "cpu",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = None
        app.state.model_lock = threading.Lock()
        app.state.execution_lock = threading.Lock()
        try:
            if isinstance(backend, FixtureBackend):
                app.state.backend = backend
            elif backend == "fixture":
                app.state.backend = FixtureBackend()
            elif backend == "trained":
                import torch

                threads = int(os.environ.get("INSPECTION_DL_THREADS", "4"))
                if not 1 <= threads <= 32:
                    raise ValueError("INSPECTION_DL_THREADS must be between 1 and 32")
                torch.set_num_threads(threads)
                selected = bundle_path or os.environ.get(
                    "INSPECTION_DL_BUNDLE", "artifacts/release"
                )
                app.state.backend = TrainedBackend(load_bundle(selected, device=device))
            else:
                raise ValueError("unknown backend")
        except Exception:
            # Details stay in local logs; API errors never expose filesystem paths.
            logger.exception("Inference model initialization failed")
        yield
        app.state.backend = None

    app = FastAPI(title="Visual Inspection DL", version="1", lifespan=lifespan)

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, exc: InferenceError):
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    def ready_backend():
        selected = app.state.backend
        if selected is None:
            raise InferenceError(503, "model_unavailable", "A valid model bundle is not ready.")
        return selected

    @app.get("/health")
    def health():
        return {"status": "alive", "schema_version": "1"}

    @app.get("/ready")
    def ready():
        if app.state.backend is None:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "error": {
                        "code": "model_unavailable",
                        "message": "A valid model bundle is not ready.",
                    },
                },
            )
        selected = ready_backend()
        return {
            "ready": True,
            "backend": selected.name,
            "simulated": selected.simulated,
            "model_version": selected.metadata["model_version"],
        }

    @app.get("/v1/model")
    def model_metadata():
        selected = ready_backend()
        return {**selected.metadata, "backend": selected.name, "simulated": selected.simulated}

    def execute_prediction(
        payload: bytes, media_type: str, threshold: float | None, started: float
    ) -> Prediction:
        selected = ready_backend()
        decode_start = time.perf_counter()
        image = decode_image(payload, media_type)
        decode_ms = _milliseconds(decode_start)
        try:
            output, preprocessing_ms, model_ms = selected.infer(image)
            post_start = time.perf_counter()
            applied_threshold = (
                selected.metadata["confidence_threshold"] if threshold is None else threshold
            )
            detections, rejected = _detections(
                output, selected.metadata["classes"], applied_threshold, image.width, image.height
            )
            prediction = Prediction(
                backend=selected.name,
                simulated=selected.simulated,
                model_version=selected.metadata["model_version"],
                taxonomy_version=selected.metadata["taxonomy_version"],
                artifact_digest=selected.metadata["artifact_digest"],
                image=ImageShape(width=image.width, height=image.height),
                confidence_threshold=applied_threshold,
                nms_threshold=selected.metadata["nms_threshold"],
                invalid_predictions_rejected=rejected,
                detections=detections,
                timing_ms=Timing(
                    decode_preprocess=decode_ms + preprocessing_ms,
                    model_inference=model_ms,
                    response_postprocess=_milliseconds(post_start),
                    request_total=_milliseconds(started),
                ),
            )
            return prediction
        except MemoryError as exc:
            raise InferenceError(
                503, "resource_exhausted", "Inference resources are unavailable."
            ) from exc
        except Exception as exc:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise InferenceError(
                    503, "resource_exhausted", "Inference resources are unavailable."
                ) from exc
            logger.exception("Inference failed")
            raise InferenceError(
                500, "inference_failed", "The model could not produce a valid prediction."
            ) from exc
        finally:
            image.close()

    def execute(
        payload: bytes, media_type: str, threshold: float | None, started: float
    ) -> Prediction:
        # Request cancellation cannot stop a running thread. Its own lock must cover
        # decoding/inference through completion, even after admission is released.
        # Acquire here so cancellation before thread startup cannot strand the lock.
        if not app.state.execution_lock.acquire(blocking=False):
            raise InferenceError(
                503, "inference_busy", "Another inference request is active; retry later."
            )
        try:
            return execute_prediction(payload, media_type, threshold, started)
        finally:
            app.state.execution_lock.release()

    @app.post("/v1/predict", response_model=Prediction)
    async def predict(request: Request, threshold: float | None = Query(default=None, ge=0, le=1)):
        started = time.perf_counter()
        ready_backend()
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in {"image/png", "image/jpeg"}:
            raise InferenceError(
                415, "unsupported_media_type", "Use image/png or image/jpeg with raw image bytes."
            )
        length = request.headers.get("content-length")
        if length is not None:
            try:
                parsed_length = int(length)
                if parsed_length < 0:
                    raise ValueError
            except ValueError as exc:
                raise InferenceError(
                    400, "invalid_content_length", "Content-Length is invalid."
                ) from exc
            if parsed_length > MAX_UPLOAD_BYTES:
                raise InferenceError(413, "payload_too_large", "Image exceeds the upload limit.")
        if not app.state.model_lock.acquire(blocking=False):
            raise InferenceError(
                503, "inference_busy", "Another inference request is active; retry later."
            )
        try:
            if app.state.execution_lock.locked():
                raise InferenceError(
                    503, "inference_busy", "Another inference request is active; retry later."
                )
            payload = bytearray()
            try:
                async with asyncio.timeout(30):
                    async for chunk in request.stream():
                        if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
                            raise InferenceError(
                                413, "payload_too_large", "Image exceeds the upload limit."
                            )
                        payload.extend(chunk)
            except ClientDisconnect as exc:
                raise InferenceError(
                    400, "upload_interrupted", "Image upload was interrupted."
                ) from exc
            except TimeoutError as exc:
                raise InferenceError(
                    408, "upload_timeout", "Image upload exceeded 30 seconds."
                ) from exc
            if length is not None and len(payload) != parsed_length:
                raise InferenceError(
                    400, "content_length_mismatch", "Content-Length does not match uploaded bytes."
                )
            return await run_in_threadpool(execute, bytes(payload), media_type, threshold, started)
        finally:
            app.state.model_lock.release()

    return app


app = create_app(
    backend=os.environ.get("INSPECTION_DL_BACKEND", "trained"),
    device=os.environ.get("INSPECTION_DL_DEVICE", "cpu"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/release"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fixture", action="store_true", help="explicit simulated backend")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(
            args.bundle, backend="fixture" if args.fixture else "trained", device=args.device
        ),
        host="127.0.0.1",
        port=args.port,
        limit_concurrency=8,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
