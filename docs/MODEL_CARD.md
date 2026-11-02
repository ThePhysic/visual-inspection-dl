# Model card: pcb-ind-ssdlite-v1

This released PyTorch detector proposes eight classes of PCB surface defects with
bounding boxes. It is independently runnable through a local FastAPI service.
The measured task is **300×300 PCB-IND AOI candidate ROI detection**. Other image
dimensions may be accepted by the API, but their detection quality is unvalidated.
Full-board search, other factories/cameras/materials and production false-alarm
rates are outside the demonstrated evidence. Human review remains necessary.

## Model and image processing

Architecture: `ssdlite320_mobilenet_v3_large`; preprocessing version:
`ssdlite320_rgb_unit_v1`. The COCO_V1 pretrained backbone and box regressor
initialize the model; its classification head is replaced for eight defects plus
background. PyTorch **2.7.1+cu126** and torchvision
**0.22.1+cu126** execute the recorded study.

Input decoding applies EXIF orientation, converts to RGB and creates floating
CHW pixels in [0,1]. Torchvision stretches to 320×320 without letterboxing and
normalizes each channel with mean/std 0.5/0.5. Returned boxes use exclusive-upper-edge
`xyxy` pixels in the original EXIF-oriented image frame. Training augmentation in
the selected run reflects horizontally with probability
**0.5** and transforms target boxes accordingly.

The model retains candidates above score 0.001, applies class-aware NMS at
IoU **0.55**, and caps outputs at
**100** detections/image. Serving's default confidence
threshold is **0.2**, selected solely by validation micro
F1 at IoU 0.50; higher thresholds win ties. This is a demonstration operating point,
not an industrial acceptance criterion or a calibrated probability cutoff.

## Data and model selection

The author release [PCB-IND v4](https://zenodo.org/records/19723114) supplies real
AOI production-line image patches under CC BY 4.0. The frozen audit accepts
4788 images: 3352 train,
714 validation and 722 test.
1 malformed image/annotation record is explicitly excluded.
All shared numeric filename-prefix proxies and detected visual duplicates stay
within one split. These inferred groups reduce obvious correlation leakage;
their physical-board/session semantics are unconfirmed. See [dataset details](DATASET.md).

Two bounded runs share seed 42, batch 8,
AdamW at learning rate 0.0003, weight decay
0.0005, cosine scheduling, pretrained initialization and epoch
budget. Only horizontal reflection probability differs. BatchNorm running
statistics are frozen while trainable affine parameters are fine-tuned.
Training uses FP32; the earlier failed AMP experiment remains separately reported.
There is one training seed, so the comparison does not establish seed-robust gains.
The [initialization check](../reports/rtx4060-study-v2/initialization-parity.json) confirms exact
CPU equality of every initial model parameter and buffer across both runs
(476 tensors); matching seeds alone are not used
as evidence that initialization was identical.

| Variant | Completed epochs | Best epoch (1-based) | Validation AP50:.95 | Epoch time total, min | Skipped AMP steps |
| --- | --- | --- | --- | --- | --- |
| no-flip | 12 | 11 | 0.2256 | 21.00 | 0 |
| flip | 12 | 11 | 0.2513 | 15.97 | 0 |

Selected variant: **flip**. Checkpoint and threshold selection
use validation only, followed by a recorded decision and final holdout evaluation.
The simple baseline proposes one train-derived median normalized box per class,
with its training-image prevalence as confidence. It returns the same normalized
geometry for every image and is a deliberately limited reference.

## Quantitative results

All AP, precision, recall and F1 values below use a 0–1 scale. AP uses class-aware
greedy IoU matching, a 101-point precision envelope, and IoUs 0.50:0.05:0.95.
No crowd/ignore handling or COCO object-area/maxDet averaging is claimed.
Operating precision/recall/F1 use same-class IoU 0.50 matching at the shared frozen
threshold for both baseline and trained model.

| Evaluation | Images | AP50 | AP50:.95 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Validation: train-derived prior baseline | 714 | 0.0134 | 0.0057 | 0.0413 | 0.0719 | 0.0525 |
| Validation: trained detector | 714 | 0.4386 | 0.2513 | 0.4754 | 0.5530 | 0.5113 |
| Final holdout: train-derived prior baseline | 722 | 0.0209 | 0.0103 | 0.0443 | 0.0740 | 0.0554 |
| Final holdout: trained detector | 722 | 0.4417 | 0.2719 | 0.5331 | 0.5676 | 0.5498 |

| Class | Train images | Val images | Test images | Test GT boxes | Test AP50 | Test AP50:.95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mouse_bite | 570 | 123 | 122 | 129 | 0.3246 | 0.1396 | 0.4634 | 0.2946 |
| missing_copper | 723 | 156 | 155 | 229 | 0.6507 | 0.4345 | 0.6853 | 0.6943 |
| scratch | 295 | 64 | 90 | 90 | 0.8306 | 0.7023 | 0.5685 | 0.9222 |
| spurious_copper | 430 | 93 | 91 | 104 | 0.5334 | 0.3209 | 0.5816 | 0.5481 |
| copper_burr | 740 | 159 | 158 | 160 | 0.5158 | 0.2447 | 0.4622 | 0.6500 |
| stain | 364 | 79 | 78 | 83 | 0.3104 | 0.1767 | 0.3409 | 0.3614 |
| short | 225 | 48 | 48 | 52 | 0.3435 | 0.1444 | 0.4419 | 0.3654 |
| open | 78 | 17 | 17 | 18 | 0.0248 | 0.0121 | 0.1429 | 0.0556 |

A percentile bootstrap over 97 held-out source/duplicate groups (200 resamples, seed 2026) gives 95% intervals for micro precision **[0.4630, 0.6167]** and recall **[0.5036, 0.6235]** at IoU 0.50 and the frozen threshold. These intervals reflect observed group sampling; they do not estimate training-seed variation, unseen-board independence or production-domain shift.

The `open` class has only 17 test images;
per-class results on rare defects have limited precision. Score histograms,
precision–recall curves and binned observed detection precision are descriptive;
scores are not asserted to be calibrated probabilities. Full failure evidence is
in [error analysis](ERROR_ANALYSIS.md), with [machine-readable holdout results](../reports/rtx4060-study-v2/holdout/report.json)
and [PR/score plots](../reports/rtx4060-study-v2/holdout/precision_recall_and_scores.png).

## Local performance and bounded optimization

All performance values below come from the newly measured,
[release-bound benchmark](../reports/rtx4060-study-v2/benchmark-release.json), whose artifact,
weights, model/taxonomy version, dataset and class mapping exactly match this
release. Each available configuration records the identity actually loaded.
The earlier [benchmark.json](../reports/rtx4060-study-v2/benchmark.json) is preserved as historical
measurement only: its missing model identity fields prevent it from establishing
release-specific performance provenance. It supplies none of the values below.

Hardware: **NVIDIA GeForce RTX 4060 Laptop GPU**; CPU identification:
**AMD Ryzen 9 7940HS w/ Radeon 780M Graphics**; platform `Windows-11-10.0.26200-SP0`.
The [hardware record](../reports/environment.json) reports
31.21GiB physical RAM
and GPU model/VRAM/driver `NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 566.36`.
The CPU path uses 4 threads. The benchmark
uses the first 100 sorted validation IDs, FP32,
concurrency 1, and three untimed warm-up batches for each configuration.
This measures local warmed inference without HTTP/network overhead.

| Device | Batch | Timed batches | Bundle load, ms | Batch p50, ms | Batch p95, ms | Images/s | Peak allocated VRAM, MiB | Lifetime peak RSS, MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 1 | 100 | 586.13 | 54.38 | 67.05 | 18.05 | not measured | 715.1 |
| cpu | 4 | 25 | 586.13 | 137.81 | 195.06 | 26.59 | not measured | 800.9 |
| cuda | 1 | 100 | 457.23 | 36.79 | 41.86 | 28.16 | 130.6 | 1413.8 |
| cuda | 4 | 25 | 457.23 | 70.92 | 81.66 | 56.93 | 177.1 | 1443.2 |

| Device | Batch | Subset AP50 | Subset AP50:.95 | Precision | Recall | Counts equal vs batch1 | Labels equal | Max box difference, px | Max score difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 1 | 0.5766 | 0.3401 | 0.5734 | 0.6777 | True | True | 0.000000 | 0.00000000 |
| cpu | 4 | 0.5766 | 0.3401 | 0.5734 | 0.6777 | True | True | 0.000122 | 0.00000077 |
| cuda | 1 | 0.5767 | 0.3403 | 0.5734 | 0.6777 | True | True | 0.000000 | 0.00000000 |
| cuda | 4 | 0.5767 | 0.3403 | 0.5775 | 0.6777 | True | False | 296.653351 | 0.00346604 |

cpu: batch 4/batch 1 throughput ratio 1.473; AP50:.95 change +0.000000; operating-count changes TP +0, FP +0, FN +0 on the identical subset. cuda: batch 4/batch 1 throughput ratio 2.022; AP50:.95 change -0.000025; operating-count changes TP +0, FP -1, FN +0 on the identical subset. The optimization changes batch policy only; these
quality comparisons use identical images and the same released weights.
Exact output parity is not established for cuda batch 4. Raw model output counts and label order are compared before validation and operating-threshold filtering. Coordinate and score differences compare rows by score rank; when labels/order differ, the coordinate maximum is not displacement of matched detections. The AP and operating-count comparisons measure the observed quality effect independently of this rank comparison.
Batch latency is not per-image request latency. Forward timing includes resize,
normalization, model execution, NMS and coordinate restoration, using preloaded
device inputs. Its throughput excludes input transfer and CPU output materialization.
Decoding/tensor
conversion p50/p95 are 1.743/
2.555ms; transfer times are
reported separately in [benchmark JSON](../reports/rtx4060-study-v2/benchmark-release.json). Decode/tensor
timing includes image-file I/O and excludes preceding checksum verification.
Bundle load timing includes integrity validation and model construction, but
excludes Python imports and initial CUDA context creation; cache state is uncontrolled.
It does not establish full process startup. p99 is omitted below 1,000 timed batches.
GPU memory is PyTorch peak allocated/reserved, excluding driver/other processes;
process RSS is a lifetime high-water mark including preloaded data and previous
configurations. These measurements are not production SLA guarantees.

## Release identity and use limits

| Field | Recorded value |
| --- | --- |
| Model version | pcb-ind-ssdlite-v1 |
| Artifact digest | 9037234654bcc190f6f5dd8cda07e70a119ba19b139b2b99118f4d9f27d7e05e |
| Weights SHA256 | 28271a688aa7335521bb7725423fd6ea6622f3d1964bfaed83dd97b4f0ff34d9 |
| Selected checkpoint SHA256 | c6d6532da7201fd8f8ba182117e4e92eca5562578b9592bf50b9a44c8670227b |
| Dataset ID | pcb-ind-zenodo-19723114-v4 |
| Manifest file SHA256 | 0b320a9502f0a785f35c7d8496eb3c7a1d8cf5a83804a452a41e271958bb6bbd |
| Canonical manifest digest | e33d69b335bf830eec61cfd7cec4348f4f57d87291941f8679b9f57b1fc37d31 |
| Taxonomy version | pcb-ind-v4-1 |
| Training run | rtx4060-study-v2-flip |

Load the checked bundle from `releases/pcb-ind-ssdlite-v1`; it includes weights, metadata and the
exact evaluation report. Integrity and compatibility failures prevent readiness;
inference does not silently substitute checkpoints or simulated predictions.
See [reproducibility](REPRODUCIBILITY.md) and [API semantics](API.md).

The source code is MIT; dataset images/labels remain CC BY 4.0 with author
attribution. Pretrained-model and underlying COCO image terms remain applicable;
no unrestricted commercial weight rights are asserted. See
[third-party terms](THIRD_PARTY.md) and the release metadata's licensing record.
This portfolio/research project has no Matroid affiliation, manufacturing deployment
claim or evidence of safety-critical fitness.

Generated deterministically by `scripts/summarize_study.py` from completed,
identity-checked JSON evidence. Input digests are recorded with the study.
