# Standalone inference contract v1

This service is independently runnable. It owns model loading and image-to-detections
inference. It has no inspection IDs, database writes, upload storage, queues, review
history, or training triggered by an inference request. The related SWE application's
actual schema must be inspected during a later integration milestone.

Start a real model with:

```powershell
python -m inspection_dl.api --bundle artifacts/release --device cpu --port 8000
```

`--device cuda` selects an available CUDA device; failure to initialize leaves readiness
false. The CLI binds `127.0.0.1`, uses one process, permits at most eight concurrent
connections, and permits one active prediction request. A second prediction receives
`503 inference_busy`; the client can retry with a bounded backoff. The service does
not queue work. Torch CPU threads default to four and can be configured with
`INSPECTION_DL_THREADS` (1–32). Larger concurrency and external hosting are outside
the demonstrated deployment.

For an explicitly simulated contract demonstration, use `--fixture`. Fixture responses
always report `backend: "fixture"`, `simulated: true`, and a fixture model version.
No model initialization or inference failure falls back to that backend.

For an ASGI server, `inspection_dl.api:app` reads `INSPECTION_DL_BUNDLE`,
`INSPECTION_DL_DEVICE` (default `cpu`), and `INSPECTION_DL_BACKEND` (default `trained`).
Use the CLI defaults or equivalently bind the ASGI server to loopback with bounded
concurrency. The service includes no authentication for external deployment.

## Endpoints

| Method/path | Behavior |
| --- | --- |
| `GET /health` | Process liveness, including when model loading failed. |
| `GET /ready` | `200` with `ready: true` and backend identity; `503` with `ready: false` when model loading failed. |
| `GET /v1/model` | Bundle metadata, taxonomy, operating thresholds, artifact digest, and simulated flag; `503` when unready. |
| `POST /v1/predict` | Raw `image/png` or `image/jpeg` request body; typed JSON prediction. |

Example request (Windows curl executable):

```powershell
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@example.jpg" http://127.0.0.1:8000/v1/predict
```

The optional `?threshold=0.4` overrides the release's confidence threshold for that
request and is recorded in its response. Values must be finite and between zero and
one. Class-aware NMS uses the fixed release threshold (`nms_threshold`, normally 0.55).
The model retains at most `max_detections` predictions per image, using an internal
score floor of 0.001; lowering the request threshold below that floor cannot recover
discarded candidates. Empty detections are a successful result.

## Image and geometry rules

- The body is limited to **8 MiB**, enforced using both declared length and streamed
  bytes, with a 30-second upload deadline. Declared lengths must match actual bytes.
  The decoded image is limited to **16,000,000 pixels**, checked from its header
  before decompression. Only single-frame PNG and JPEG are accepted; declared and
  actual formats must match. Corrupt/truncated images are rejected.
- Pillow applies EXIF orientation and converts to RGB. Grayscale is replicated into
  RGB; alpha is discarded without compositing. This matches the training pipeline's
  channel convention. Uploaded bytes are not persisted.
- `image.width`/`image.height` describe the **EXIF-oriented original raster**. A JPEG
  whose EXIF rotates an 80×40 stored raster can therefore report width 40, height 80.
  Clients should display that same oriented raster.
- `coordinate_system: "xyxy_pixels_original_image"` means `x_min,y_min,x_max,y_max`,
  in floating-point pixels, with exclusive upper edges; `0 <= x_min < x_max <= width`
  and `0 <= y_min < y_max <= height`. Coordinates are never normalized.
- The SSDLite model resizes to 320×320 internally, normalizes RGB `[0,1]` using mean
  and standard deviation 0.5, and converts detections back to the original dimensions.
  No crop or letterbox is applied. A proposal clipped to an image boundary can become
  zero-area. The service discards zero-area, inverted, non-finite, or out-of-bounds
  geometry and explicitly reports the count in `invalid_predictions_rejected`.
  This count covers all model candidates before the request's confidence threshold,
  including discarded low-score proposals. Valid detections are retained; all-invalid
  geometry yields a successful empty detection list with a nonzero rejection count.
  Unknown classes, invalid/non-finite confidence scores, malformed boxes, or mismatched
  output lengths remain model errors and return HTTP 500.

Success contains `schema_version`, `provider`, `backend`, `simulated`, `model_version`,
`taxonomy_version`, `artifact_digest`, `image`, `coordinate_system`,
`confidence_threshold`, `nms_threshold`, `invalid_predictions_rejected`, `detections`,
and `timing_ms`. Each detection
has `class_id`, `label`, `confidence`, and a `box` object. Class IDs start at one;
zero is background and is never returned. Confidence is a model score, not a calibrated
probability of a manufacturing defect.

Timing is local server wall time in milliseconds, synchronized around CUDA work:

| Field | Includes |
| --- | --- |
| `decode_preprocess` | Pillow decode/orientation/RGB conversion plus tensor conversion and host-to-device transfer. |
| `model_inference` | Complete torchvision model call, including its internal resize/normalization, detector execution, NMS, and original-frame restoration. |
| `response_postprocess` | Transfer/conversion of predictions to typed response detections. |
| `request_total` | Handler entry through prediction assembly, including reading the body; excludes ASGI serialization, networking, and client latency. |

These measurements are not a separately instrumented pure neural-network kernel time.
Fixture timing measures simulated code and is unsuitable for model benchmarks.

## Errors

Handled errors use `{"error":{"code":"...","message":"..."}}` with no local paths
or exception details. Query validation uses FastAPI's standard `422 detail` response.

| Status | Example codes |
| --- | --- |
| `400` | `invalid_image`, `invalid_content_length`, `content_length_mismatch`, `upload_interrupted` |
| `408` | `upload_timeout` |
| `413` | `payload_too_large`, `image_too_large` |
| `415` | `unsupported_media_type`, `media_type_mismatch`, `unsupported_image` |
| `422` | Invalid query threshold |
| `503` | `model_unavailable`, `inference_busy`, `resource_exhausted` |
| `500` | `inference_failed` (model failure, invalid score/taxonomy, or malformed output) |

Local logs retain initialization/inference exception details for diagnosis. They should
remain private. No error response contains invented detections.

## Model bundles

Each immutable release directory contains `weights.pt` (tensor-only state dict),
`metadata.json`, and `evaluation.json`. Metadata records architecture/preprocessing,
taxonomy and versions, operating thresholds, dataset manifest hash, run identity,
runtime requirements, and source/license attribution. SHA-256 covers weights and
evaluation bytes; `artifact_digest` covers canonical metadata excluding itself.
These hashes detect inconsistency and corruption, not authenticity against an attacker
who can replace both metadata and weights.

The loader checks required files and digests, supported schema/architecture/preprocessing,
consecutive taxonomy IDs, and torch/torchvision major.minor compatibility. It uses
`torch.load(..., weights_only=True)` and strict state-dict loading, with no download
or fallback checkpoint. Patch-version updates are allowed. Missing, mismatched, or
corrupted releases leave readiness false. Releases are selected explicitly by path;
the HTTP API accepts no filesystem paths.

Export a trained checkpoint (its `model` key must hold the state dict):

```powershell
python -m inspection_dl.bundle runs/selected/best.pt artifacts/release --manifest data/manifest.json --report reports/final-evaluation.json --model-version pcb-ind-v1 --licensing configs/licensing.json --threshold 0.25
```

Choose the actual threshold on validation data before final reporting. The export CLI
checks checkpoint taxonomy/dataset/preprocessing and manifest identity against the
release inputs, and checks the evaluation's checkpoint/manifest hashes and operating
threshold. Export refuses
to overwrite an existing destination. Interrupted exports remain incomplete and fail
validation; use a new destination for another attempt. The tensor round-trip test
checks a real SSDLite architecture with **untrained test weights** and makes no quality
claim about those fixture weights.
