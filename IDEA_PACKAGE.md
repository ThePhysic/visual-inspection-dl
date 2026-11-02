# Visual Inspection Deep Learning & Model Serving — Project Idea Package

## 1. Project identity and purpose

**Working name:** Visual Inspection Deep Learning & Model Serving (the **Matroid DL component** of the portfolio).

Build a self-contained, reproducible computer-vision system that takes a real, appropriately licensed visual-inspection dataset from raw images and annotations through model training, rigorous evaluation, model versioning, and an inference service. The principal task is **object/defect detection with bounding boxes**, because this output naturally fits the existing human-review application. The project should expose genuine deep-learning work, not merely call a hosted vision API or relabel the current simulated inference as a trained model.

The eventual outcome is a versioned vision-inference provider that the separate Visual Inspection Platform can consume, alongside an independently demonstrable ML repository with transparent metrics, artifacts, limitations, and reproducible experiments.

This is a personal portfolio project inspired by industrial visual inspection and the technical work of a deep-learning/field engineer. It is not a Matroid product, proprietary integration, or claim of association with Matroid.

## 2. Relationship to the existing SWE component

**Existing public reference:** https://github.com/ThePhysic/visual-inspection-platform

The existing SWE component is a React/Vite/TypeScript and Fastify application with MongoDB, Redis/BullMQ, Docker, CI, and browser/integration testing. Its workflows encompass image upload, validation, durable asynchronous inspection jobs, simulated inference, bounding-box review/editing, separately persisted human corrections and history, retries, recovery, and reconciliation. Its primary responsibility is application and workflow reliability; it is not yet evidence of training a real vision model.

**Boundary:** The DL component is a separate Python/PyTorch project and independently runnable service. Its responsibility is preparing data, training and qualifying models, producing immutable model artifacts, and serving predictions. It does not become the owner of the SWE application's uploads, queues, inspection records, MongoDB data, human-review revisions, or frontend.

**Scope decision:** Complete and demonstrate the DL component independently first. Within this project, define and test a clean, versioned inference contract suitable for future integration. The work of replacing the SWE mock provider, deploying the two services together, mapping their exact schemas, and testing end-to-end review workflows belongs to a **later, separately scoped integration milestone**, not the core DL build.

The existing GitHub repository is useful architectural context and should be referenced, but its current concrete types, routes, and provider contracts must be inspected rather than presumed when integration begins. Both projects should retain independent repositories, tests, and documentation.

## 3. Real-world problem and user-facing outcome

Human inspection of manufactured items, equipment, or visual assets is expensive and inconsistent. A model can propose localized findings—such as suspected surface defects or missing/incorrect components—but its predictions must be quantitatively validated, inspectable, and correctable by a person.

The DL project should demonstrate the full loop:

1. Establish exactly what visual defect/object classes are being detected and where the training labels came from.
2. Audit, split, transform, and version the visual data without leaking related samples across training and evaluation.
3. Train or fine-tune a suitable pretrained computer-vision detector and compare it with a meaningful baseline.
4. Measure detection quality, calibration/confidence behavior, robustness, and failure patterns on held-out data.
5. Publish an immutable, traceable model bundle and serve image-to-detections requests.
6. Provide enough metadata and stable geometry for a later human-review platform to display, accept, reject, or correct predictions.

The result should be credible even when the model's accuracy is modest. Honest negative findings, failed experiments, and dataset limitations are first-class outputs.

## 4. Primary ML task and dataset

### Primary task

**Supervised, multi-class object/defect detection with rectangular bounding boxes**. Prefer a practical industrial/quality-inspection domain, for example surface defects, component defects, or visually identifiable industrial anomalies, where ground-truth bounding boxes and class labels exist.

Choose one coherent dataset/problem and finish it well. Broadly supporting multiple unrelated datasets, video tracking, segmentation, or classification is not necessary for the initial component.

### Dataset requirements

- Select a real, obtainable, legally usable dataset with documented provenance and enough labeled examples for detection training and held-out evaluation. Review dataset, image, annotation, and pretrained-weight licenses; do not silently redistribute restricted images or weights.
- Build a validated dataset manifest containing source identifiers, image dimensions, labels, bounding boxes, split membership, and relevant checksums/versions.
- Inspect invalid/corrupt images, missing or malformed labels, coordinate conventions, duplicate images, class imbalance, rare classes, and label inconsistencies. Report exclusions rather than silently discarding data.
- Ensure no target/label leakage into model inputs. Prevent exact or near-duplicate images, adjacent video frames, or related acquisition groups from straddling train/test boundaries when relevant.
- Use a reproducible train/validation/test split, ideally grouped by part, source, sequence, or acquisition session when metadata permits. Freeze the final holdout for evaluation, not iterative model selection.
- Define the class taxonomy and an explicit mapping from original annotations to the model's class IDs and human-readable labels.
- Keep expensive/raw licensed data out of the Git repository when redistribution is not permitted. Provide a reproducible acquisition/preparation path and a tiny legally shareable fixture for CI.

### Geometry and preprocessing

Document image orientation (including EXIF), resizing/letterboxing/cropping, channel handling, normalization, and any augmentation applied only to training data. Preserve the transforms needed to convert predictions back to the **original image coordinate frame**. Reject impossible, non-finite, inverted, or out-of-bounds boxes; do not accidentally swap x/y or normalized/pixel coordinates.

## 5. Model development and experiments

Use **PyTorch** as the primary deep-learning framework. A lightweight pretrained detector from an established open-source ecosystem (for example a torchvision detection architecture) is an appropriate starting point; select based on task, licensing, maintainability, and the available local GPU rather than model-fashion claims.

The training system should include:

- A simple reference/baseline approach and at least one substantive trained or fine-tuned model. Comparisons must use compatible held-out data and definitions.
- Explicit dataset/config/model versions, random seeds, environment/dependency versions, checkpoint metadata, training hyperparameters, augmentation choices, and data manifests.
- Training and validation loops with loss logging, learning-rate behavior, checkpoint selection on validation data, reasonable early-stopping or fixed-budget policy, and a documented resource budget.
- Robust checkpoint save/load behavior; resume training where practical, with correct optimizer/scheduler state if claiming resumability.
- Experiment tracking through a local, portable mechanism (structured run manifests, metrics files, plots, and optionally MLflow), without requiring a hosted paid service.
- A small set of meaningful, controlled comparisons such as model architecture/size, fine-tuning depth, augmentation, input resolution, or confidence threshold; avoid an unbounded hyperparameter search.
- Explicit separation of exploratory results, validation-based selection, and one-time final holdout reporting.

The user's primary development machine is an AMD Ryzen 9 laptop with an NVIDIA RTX 4060 and approximately 8 GB of GPU VRAM, 32 GB RAM, and Windows 11. The design should accommodate bounded local experiments and offer a CPU inference path. Large clusters, multi-GPU training, and expensive cloud GPU use are not core dependencies.

## 6. Model evaluation and failure analysis

Evaluation is a core product of this repository, not an afterthought. Include:

- Detection metrics with clearly defined matching/IoU conventions: per-class and aggregate precision/recall, AP/mAP (including AP@0.50 and, where supported, AP across IoU thresholds), false positives/false negatives, and relevant confusion/threshold summaries.
- Side-by-side baseline-versus-trained-model results on the same holdout or controlled validation protocol.
- Per-class sample counts, class imbalance, confidence-score distributions, precision–recall curves, and threshold selection tied to the intended use case.
- Qualitative visualizations of correct detections, misses, false positives, overlapping boxes, poor localization, and hard/ambiguous examples, with no cherry-picked-only evidence.
- Evaluation slices where the dataset supports them, such as defect size, image quality, lighting, source group, class frequency, or backgrounds; document missing slice metadata instead of inventing it.
- Robustness checks relevant to the problem (for example sensible brightness, blur, resolution, or compression variations) while distinguishing naturally occurring conditions from synthetic perturbations.
- Where sample sizes permit, intervals or variability estimates across seeds/splits; do not imply statistical certainty from a small or correlated holdout.
- A concise model card and error-analysis report covering supported data, failure modes, known limitations, and deployment cautions.

The project should never report high accuracy without its dataset, split, metric definition, and evaluation limitations. Avoid claims of manufacturing deployment, customer benefit, or safety-critical fitness without evidence.

## 7. Versioned model artifacts and reproducibility

Each candidate release should have a traceable bundle containing at least:

- Model weights or a reproducible, documented way to obtain them;
- Architecture/preprocessing configuration and input-size assumptions;
- Taxonomy and class-index mapping;
- Dataset and training-run identifiers, with appropriate hashes/manifests;
- Evaluation report and selected operating thresholds;
- Model version and artifact digest;
- Runtime/dependency requirements and licensing/source attribution.

Model loading must not silently substitute a different checkpoint, taxonomy, or preprocessing pipeline. Fail clearly if a bundle is missing, incompatible, corrupted, or mismatched. A release should support repeatable evaluation and standalone inference without access to the original training process.

## 8. Standalone inference service

Provide a small **Python/FastAPI** service that loads a selected versioned model bundle and serves inference. It should be runnable without the SWE platform.

Core capabilities:

- Accept an image with well-defined media type, size limits, validation, and documented preprocessing.
- Return zero or more typed detections with class identifier/name, confidence score, and bounding-box geometry expressed unambiguously in the original-image frame.
- Report the model version, taxonomy version, input dimensions, coordinate system, and relevant inference metadata (including distinguishable model compute time vs request overhead when measured).
- Apply documented confidence and NMS/postprocessing behavior; keep score thresholds configurable and recorded.
- Offer health/readiness and model metadata endpoints, with readiness false when weights/model initialization fail.
- Support predictable error responses for corrupt images, invalid payloads, unsupported formats, model failures, and resource constraints. Never return fabricated predictions to disguise an error.
- Have a deterministic mock/fixture backend for service/contract tests, explicitly identified as simulated, while the main demonstrated backend uses real trained weights.
- Avoid external exposure by default. No hardcoded secrets, unbounded uploads, debug data leaks, or filesystem path traversal.

### Illustrative integration-facing response (not a claim about existing SWE types)

```json
{
  "schema_version": "1",
  "provider": "visual-inspection-dl",
  "model_version": "example-release-id",
  "taxonomy_version": "example-taxonomy-id",
  "image": { "width": 1280, "height": 720 },
  "coordinate_system": "xyxy_pixels_original_image",
  "detections": [
    {
      "class_id": 2,
      "label": "example-defect",
      "confidence": 0.87,
      "box": { "x_min": 120, "y_min": 80, "x_max": 260, "y_max": 205 }
    }
  ],
  "timing_ms": { "model_inference": 42.0 }
}
```

This example defines desired semantics only. The real contract should be reconciled with the existing application's versioned provider/data model during the later integration project. The DL service should not require or mutate an `inspection_id`, MongoDB document, BullMQ job, or human-review revision to perform inference.

## 9. Performance and deployment evidence

Demonstrate measured quality/latency tradeoffs instead of relying on model size or architecture reputation.

- Benchmark load time, warm inference latency (at least p50/p95, p99 when sample volume supports it), throughput, batch-size effects where relevant, and peak memory/VRAM on explicitly identified hardware.
- Separate data decoding/preprocessing, model execution, and postprocessing costs where practical.
- Compare CPU fallback with GPU execution when both are supported. Keep results labeled by hardware, precision mode, input resolution, concurrency, and sample selection.
- Investigate a bounded, justified optimization (for example input resolution, model variant, batch policy, quantization only if validated, or preprocessing/postprocessing changes) and show quality as well as performance effects.
- Avoid implying observed local benchmark numbers are production SLA guarantees or equivalent to external network round trips.
- Supply a reproducible container or environment setup for serving, plus smoke tests without requiring the full training dataset or a GPU in CI.

Cloud deployment, Kubernetes, auto-scaling, multi-node inference, and large observability stacks are optional follow-ons, not core scope.

## 10. Validation and software quality

The repository should combine ML scientific validation and software engineering discipline:

- Unit tests for annotation parsing, class mapping, split membership, leakage/duplicate checks, geometry transforms, and postprocessing.
- Model/metric tests on small deterministic fixtures, including correct empty detections, multi-class cases, boundary boxes, and score behavior.
- Inference API/contract tests covering success, malformed uploads, excessive payloads, missing/incompatible checkpoints, readiness, and error semantics.
- A lightweight real-model smoke evaluation and load/save parity check; expensive/full training should not be required to run ordinary CI.
- Consistent run configs, pinned/reproducible dependencies, seed documentation, clear commands, and machine-readable results.
- CI that runs practical tests and formatting/static checks on a supported environment; honest disclosure of GPU-only experiments and non-CI-reproduced training runs.

Scientific results and model artifacts must be tied to evidence; synthetic fixtures and mocked inference must be labeled as such.

## 11. Future integration with the Visual Inspection Platform

**This section is a compatibility destination, not part of the initial standalone implementation.**

Desired eventual workflow:

1. The existing Fastify API validates an image and persists an inspection request.
2. The existing BullMQ worker selects a configured inference provider. The current mock remains available for deterministic tests and fallback demonstrations.
3. A new provider adapter calls the Python inference service with validated image bytes and bounded timeouts, mapping labels, confidence, geometry, model identity, and errors to the SWE application's *actual* versioned schema.
4. The application stores predictions as an immutable inspection result and preserves its existing retry/idempotency/recovery semantics. The Python service does **not** write to MongoDB or own BullMQ lifecycle state.
5. The React/Konva review UI displays model detections and permits human corrections. Review revisions and annotation history remain independent of the original predictions.
6. End-to-end tests verify successful inference, no detections, model unavailability, timeouts/retries, incompatible taxonomy/geometry, worker crashes, and preservation of review history.

Compatibility questions to resolve only when both codebases can be inspected: existing `InferenceProvider` boundary, upload/image-storage access, exact normalized-box convention, class taxonomy, schema/version fields, timeouts, worker retry behavior, inspection identity, immutable prediction storage, error taxonomy, and Docker network/configuration.

The integration should preserve and regress-test the existing SWE system; training should remain offline and should not be triggered by ordinary inspection requests. Corrected human annotations may later become a *reviewed, consented, versioned* training dataset, but no automatic self-training or feedback loop is claimed without actually building and evaluating one.

## 12. Component outcomes and demonstration

A convincing standalone demo should be able to show:

- An auditable dataset manifest and defensible split;
- A baseline and a trained PyTorch detector with repeatable evaluation;
- Explicit quantitative results, error categories, confidence/threshold behavior, and limitations;
- A selectable, versioned model bundle with working local inference;
- Annotated sample images generated by the real model;
- A live local API request returning typed boxes, scores, and provenance;
- Latency/resource measurements with exact hardware/environment context;
- Passing practical tests and a coherent README/model card/reproducibility report.

The most valuable project story is **training → scientific validation → packaged inference**, with later **inference → asynchronous app workflow → human review** integration. Report both successful and unsuccessful experiments faithfully.

## 13. Scope exclusions and optional expansions

The initial project does **not** require:

- Changes to the existing SWE repository, frontend, queues, or databases;
- Raspberry Pi camera ingestion or edge deployment;
- Reinforcement learning, LLM fine-tuning, or agent orchestration;
- Live video, tracking, 3D reconstruction, or a generalized AutoML platform;
- Many model architectures/datasets, expensive distributed training, or cloud GPU spend;
- Kubernetes, large-scale online inference, or customer/production deployment;
- An automated retraining loop that consumes every human correction.

Potential follow-ons, only if justified by measurable value: supervised segmentation or anomaly detection on a second dataset; edge inference on Raspberry Pi or another constrained device; model monitoring/drift checks; approved review-data export and retraining studies; export/quantization acceleration; deployment into the SWE platform; and domain-specific active learning.

## 14. Repository references and access guidance

- **SWE public reference:** https://github.com/ThePhysic/visual-inspection-platform
- **DL repository:** independent new repository for this project's training, evaluation, artifacts, and FastAPI inference service; do not copy the SWE repository into it.
- **Idea stage:** The URL is sufficient to convey the related project and intended boundary. The DL component can be designed and built without a local copy of the SWE repository.
- **Integration stage:** Prefer **two local sibling checkouts** of the exact revisions being integrated, each with its own Git history. This permits Codex to inspect real types/providers, run tests, and validate joint behavior. The GitHub URL remains the canonical reference, but a local checkout is the better working input for code-level integration.
- Do not assume the current hosted repository's interface from this conceptual document; pin the revisions and inspect the actual code before authoring integration changes.

## 15. Guiding principles

Make the deep-learning component scientifically honest, reproducible, and useful in isolation; make its inference boundary narrow and stable; preserve the application platform's existing reliability and review semantics; avoid unnecessary cloud complexity and unsupported production claims; and value a well-evaluated real model over many unvalidated techniques.
