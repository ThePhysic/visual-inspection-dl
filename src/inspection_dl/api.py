"""Local standalone inference API; start with `python -m inspection_dl.api`."""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
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


def _detections(output: dict, classes: list[dict], threshold: float) -> list[Detection]:
    """Validate backend output. Invalid predictions fail rather than being fabricated."""
    taxonomy = {entry["id"]: entry["name"] for entry in classes}
    converted = {}
    for key in ("boxes", "labels", "scores"):
        value = output[key]
        converted[key] = value.tolist() if hasattr(value, "tolist") else value
    if not len(converted["boxes"]) == len(converted["labels"]) == len(converted["scores"]):
        raise ValueError("backend output lengths disagree")
    result = []
    for box, label, score in zip(
        converted["boxes"], converted["labels"], converted["scores"], strict=True
    ):
        if label not in taxonomy or isinstance(label, bool) or int(label) != label:
            raise ValueError("backend returned an unknown class")
        if len(box) != 4:
            raise ValueError("backend box must have four coordinates")
        detection = Detection(
            class_id=label,
            label=taxonomy[label],
            confidence=score,
            box=Box(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        )
        if detection.confidence >= threshold:
            result.append(detection)
    return result


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

    def execute(
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
            detections = _detections(output, selected.metadata["classes"], applied_threshold)
            prediction = Prediction(
                backend=selected.name,
                simulated=selected.simulated,
                model_version=selected.metadata["model_version"],
                taxonomy_version=selected.metadata["taxonomy_version"],
                artifact_digest=selected.metadata["artifact_digest"],
                image=ImageShape(width=image.width, height=image.height),
                confidence_threshold=applied_threshold,
                nms_threshold=selected.metadata["nms_threshold"],
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
