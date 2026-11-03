# Visual Inspection Deep Learning & Model Serving

[![validate](https://github.com/ThePhysic/visual-inspection-dl/actions/workflows/ci.yml/badge.svg)](https://github.com/ThePhysic/visual-inspection-dl/actions/workflows/ci.yml)

A reproducible PyTorch detector for eight circuit-board defect classes, with audited
data, controlled experiments, held-out evaluation, a versioned trained model, and a
standalone local FastAPI service. Training and evaluation use explicit commands;
users submit image bytes to the API and receive typed boxes, scores, and provenance.

## At a glance

- **Task:** detect eight PCB defects in **300×300 AOI candidate ROI patches**.
  Full-board search and production-domain performance are unproven.
- **Model:** fine-tuned SSDLite320 MobileNetV3 Large, shipped as the versioned
  `pcb-ind-ssdlite-v1` bundle with a typed FastAPI inference API.
- **Data:** audited [PCB-IND v4](https://zenodo.org/records/19723114), with
  **4,788 images: 3,352 train / 714 validation / 722 test**. See the
  [dataset audit](docs/DATASET.md) for exclusions, duplicate checks and split limits.
- **Frozen result:** **44.17% AP50 / 27.19% AP50:.95** on the final 722-image
  test holdout, after validation-only checkpoint and score-threshold selection.
  The operating threshold is **0.20**; missed defects and false alarms remain.

[Results index](docs/RESULTS_INDEX.md) · [Model card](docs/MODEL_CARD.md) ·
[Error analysis](docs/ERROR_ANALYSIS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Run the released model locally

After the [Python 3.12 CPU setup](#serve-the-trained-model), run these Windows
commands from the repository root. Start the server, then send an image from a
second terminal:

```powershell
.venv/Scripts/python.exe -m inspection_dl.api --bundle releases/pcb-ind-ssdlite-v1 --device cpu
# In a second terminal:
curl.exe --fail-with-body http://127.0.0.1:8000/ready
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@tests/fixtures/pcb-ind/5810_t_007.jpg" http://127.0.0.1:8000/v1/predict
```

On Linux, substitute `.venv/bin/python` and `curl`.
The included `5810_t_007.jpg` is an attributed **training-set API smoke fixture**;
its response demonstrates serving, not holdout generalization. Released inference
needs no dataset download, training run or GPU. Interactive API docs: `/docs`.

## Visual results

These existing comparisons come from the frozen **722-image test holdout**, with
the validation-selected score threshold **0.20**. **Green = ground truth (left);
red = model predictions (right).** TP/FP/FN counts use same-class IoU 0.50 matching.
The original 600×332 images are shown without modifying predictions. The first two
are from the evaluator's evenly spaced sorted split samples; the third is its
maximum-missed-defect case. The [selection record](reports/rtx4060-study-v2/holdout/examples.json)
preserves image IDs and selection reasons.

![Holdout sample 02: green scratch annotation and red scratch prediction cover the same broad defect.](reports/rtx4060-study-v2/holdout/examples/sample_02.jpg)

**True detection (`sample_02`):** the model detects the annotated `scratch`
(score 0.95): **1 TP, 0 FP, 0 FN**. This is one successful case, not an estimate
of overall performance.

![Holdout sample 04: green stain annotation is predicted as red spurious_copper.](reports/rtx4060-study-v2/holdout/examples/sample_04.jpg)

**Confident wrong class (`sample_04`):** the annotated `stain` is predicted as
`spurious_copper` (score 0.97): **0 TP, 1 FP, 1 FN**. A high score does not imply
a correct label or a calibrated probability.

![Holdout sample 14: green missing_copper, mouse_bite and stain boxes; red mouse_bite and scratch predictions.](reports/rtx4060-study-v2/holdout/examples/sample_14.jpg)

**Challenging patch (`sample_14`, maximum FN in the split):** `mouse_bite` is
detected, all three `missing_copper` and all three `stain` annotations are missed,
and an extra `scratch` prediction is false: **1 TP, 1 FP, 6 FN**.

Images: **PCB-IND v4, Hao Yan, Fujian Polytechnic Normal University**,
[Zenodo record 19723114](https://zenodo.org/records/19723114), **CC BY 4.0**;
the committed comparisons add annotation and prediction overlays. See
[all examples and plots](docs/RESULTS_INDEX.md) and the
[full error analysis](docs/ERROR_ANALYSIS.md) for broader evidence.

## Measured results

The frozen **722-image holdout** was evaluated after validation-only checkpoint
and threshold selection. Both methods below use the same test images and metric
definition; the operating threshold is **0.20**.

| Metric | Geometric sanity baseline | Fine-tuned SSDLite |
| --- | ---: | ---: |
| AP at IoU 0.50 | 2.09% | 44.17% |
| Mean AP across IoU 0.50–0.95 | 1.03% | 27.19% |
| Micro precision at IoU 0.50 | 4.43% | 53.31% |
| Micro recall at IoU 0.50 | 7.40% | 56.76% |

The train-derived median-box class-prior reference is a **geometric sanity
baseline**, not a competitive detector. It proposes the same normalized box per
class for every image, with training-image class prevalence as confidence.

The model misses defects and produces false alarms. The rare `open` class is
particularly weak: one of 18 test instances was detected at this operating point.
There is only **one training seed**; filename/source groups are heuristic proxies
whose physical-board/session meaning is unconfirmed. Scores are uncalibrated.
The measured task is 300×300 candidate ROI detection, with full-board search and
production-domain performance unproven. These results support a reproducible
research demonstration with human review, not manufacturing deployment fitness.

The [model card](docs/MODEL_CARD.md), [error analysis](docs/ERROR_ANALYSIS.md),
and [reproducibility report](docs/REPRODUCIBILITY.md) contain per-class results,
group-bootstrap intervals, representative and difficult images, robustness checks,
source identity, and full limitations. The [results index](docs/RESULTS_INDEX.md)
provides direct links to plots and raw evidence.

On the recorded Ryzen 9 7940HS / RTX 4060 laptop, warmed forward-pipeline p50
latency on 100 validation images was **54.38 ms on CPU and 36.79 ms on GPU** at batch
size one. GPU batching increased measured forward throughput from 28.16 to 56.93
images/s with small numerical/postprocessing differences. These measurements use
preloaded inputs and exclude HTTP, transfer, and CPU output materialization;
[the release-bound benchmark record](reports/rtx4060-study-v2/benchmark-release.json) defines timing,
sample counts, quality differences, and memory boundaries. These are **warmed
forward-pipeline timings, not end-to-end HTTP response latency**.

## Serve the trained model

Python **3.12**, Windows or Linux. From the repository root, create a CPU environment:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-cpu.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
.venv/Scripts/python.exe -m inspection_dl.api --bundle releases/pcb-ind-ssdlite-v1 --device cpu
```

On Linux, create the environment with `python3.12 -m venv .venv` and substitute
`.venv/bin/python`. No dataset, training run, GPU, cloud account, or hosted vision
API is needed for inference with the released bundle. The default server binds to
`127.0.0.1:8000`; interactive endpoint documentation is at `/docs`.

From a second terminal on Windows:

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/ready
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@tests/fixtures/pcb-ind/5810_t_007.jpg" http://127.0.0.1:8000/v1/predict
```

This attributed training-set fixture exercises the API and released model; it is
separate from the held-out examples and scientific evaluation above.

The response identifies the real backend, model/taxonomy versions, artifact digest,
EXIF-oriented image dimensions, applied confidence/NMS thresholds, and detections.
Boxes are exclusive-upper-edge `xyxy` pixels in that original image frame. JPEG
and PNG uploads are bounded to 8 MiB and 16 million pixels. Errors never fall back
to fabricated predictions. A missing or incompatible bundle keeps readiness false.

See [API semantics](docs/API.md) for errors and timing definitions and
[deployment](docs/DEPLOYMENT.md) for uv, containers, and read-only model mounts.
An explicit `--fixture` flag selects simulated contract-test behavior and is
identified as such in responses.

## Reproduce the study

Use a separate Python 3.12 environment with the Windows CUDA 12.6 dependency
snapshot in `requirements-gpu.txt`; the experiment machine is an RTX 4060 Laptop
GPU with approximately 8 GB VRAM. Install dependencies and then the source package:

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-gpu.txt
.venv-gpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-gpu/Scripts/python.exe scripts/acquire_data.py
.venv-gpu/Scripts/python.exe -m inspection_dl.data --root data/raw/pcb-ind --output data/manifests/reproduced.json
.venv-gpu/Scripts/python.exe scripts/run_study.py --manifest data/manifests/pcb-ind-v4.json --epochs 12 --batch-size 8 --device cuda --learning-rate 0.0003 --study-id reproduced-study --release reproduced-model
```

The acquisition command checks the pinned author archive's size, MD5, and SHA256.
Manifest generation is deterministic and refuses to overwrite existing output;
compare the reproduced canonical digest with the committed manifest. The study
runner similarly requires fresh study/release IDs and never overwrites evidence.
The GPU snapshot targets the demonstrated Windows environment; CPU dependencies
are fully hash-pinned. With uv, include `--index-strategy unsafe-best-match` when
installing either requirements file so both official indexes are considered.

The bounded comparison changes horizontal-flip augmentation only, keeping seed,
initialization, architecture, learning rate, and training budget fixed. The best
checkpoint is selected by validation AP50:.95, then the confidence threshold by
validation micro F1. A recorded selection decision precedes a single final holdout
evaluation. The baseline proposes train-derived class-prior median boxes. An early
failed mixed-precision attempt and its bounded diagnosis remain in `reports/`.

[Training and evaluation details](docs/TRAINING.md) cover geometry, checkpoint
resume, supported metric conventions, synthetic robustness checks, and resource
limits. The complete commands retain experiments, plots, representative failure
images, predictions, uncertainty estimates, and machine-readable reports.

## Validation and scope

```powershell
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m pytest
```

Ordinary tests exercise annotation parsing and splits, metrics, original-frame
geometry, real-detector gradient/update and save/load parity, interrupted training
resume, model corruption/provenance, and HTTP contracts. They use small, labeled
fixtures and require neither full data nor CUDA. CI runs this practical suite on
Windows and Linux CPU environments, including a real released-bundle ASGI smoke
test using the training-set fixture. It does not repeat full GPU training or
holdout evaluation.

The custom 101-point AP implementation was independently compared with COCOeval
on supported no-crowd fixtures; [the evidence](reports/metric-reference.json)
does not claim support for all COCO crowd/ignore/area conventions. Original COCO
and YOLO annotations are independently cross-checked against the frozen manifest.

The code is MIT licensed. PCB-IND images/labels remain CC BY 4.0; pretrained-model,
fine-tuned weight and underlying COCO terms are separate, with no claim of
unrestricted commercial weight rights. See
[third-party attribution](docs/THIRD_PARTY.md). Measurements describe this local
study, with limited rare-class support and inferred source groups. This is a
personal portfolio/research component, with no Matroid affiliation or claim of
manufacturing deployment or safety-critical fitness.

The existing [Visual Inspection Platform](https://github.com/ThePhysic/visual-inspection-platform)
remains an independent application. Its frontend, queues, database, and review
history are outside this build. [The architecture boundary](docs/ARCHITECTURE.md)
defines later integration; the superseded
[historical concept brief](docs/HISTORICAL_IDEA_PACKAGE.md) records the initial scope.
