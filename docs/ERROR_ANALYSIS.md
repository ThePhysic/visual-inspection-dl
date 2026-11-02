# Error analysis

The final holdout contains **722** images. At the validation-selected
score threshold **0.2**, same-class IoU 0.50 matching produces
**491 TP, 430 FP and 374 FN**;
micro precision/recall/F1 are **0.5331/
0.5676/0.5498**. These measured errors
describe candidate ROI patches from this dataset, not a production defect rate.
Among 13 negative holdout patches,
0 have at least one
false-positive prediction at this threshold. This dataset-specific count does not
estimate an unfiltered production-line false-alarm rate.

## Full-split error accounting

| Category | Total | Images with category | Maximum per image | Definition |
| --- | --- | --- | --- | --- |
| tp | 491 | 432 | 13 | Matched correct-class boxes |
| fp | 430 | 291 | 5 | Unmatched/wrong-class predictions |
| fn | 374 | 323 | 6 | Unmatched ground-truth boxes |
| poor_localization_candidates | 43 | 35 | 3 | FP with same-class unmatched IoU in [0.1,0.5) |
| overlapping_prediction_pairs | 95 | 82 | 3 | Prediction pairs with IoU≥0.5, regardless of class |

Categories overlap: poor-localization candidates are false positives, and duplicate
or overlapping predictions may also be false positives. A wrong-class prediction
can generate both an FP and an FN. Counts of visually overlapping predictions are
diagnostic; overlaps alone do not establish an error. Invalid predictions rejected
before metric computation: **59**.
[Every image's counts](../reports/rtx4060-study-v2/holdout/per_image_errors.json) are retained.

| Class, weakest AP first | Images | GT boxes | AP50:.95 | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| open | 17 | 18 | 0.0121 | 1 | 6 | 17 | 0.1429 | 0.0556 |
| mouse_bite | 122 | 129 | 0.1396 | 38 | 44 | 91 | 0.4634 | 0.2946 |
| short | 48 | 52 | 0.1444 | 19 | 24 | 33 | 0.4419 | 0.3654 |
| stain | 78 | 83 | 0.1767 | 30 | 58 | 53 | 0.3409 | 0.3614 |
| copper_burr | 158 | 160 | 0.2447 | 104 | 121 | 56 | 0.4622 | 0.6500 |
| spurious_copper | 91 | 104 | 0.3209 | 57 | 41 | 47 | 0.5816 | 0.5481 |
| missing_copper | 155 | 229 | 0.4345 | 159 | 73 | 70 | 0.6853 | 0.6943 |
| scratch | 90 | 90 | 0.7023 | 83 | 63 | 7 | 0.5685 | 0.9222 |

Rare-class estimates depend on few samples, including only
17 test images for `open`.
A percentile bootstrap over 97 held-out source/duplicate groups (200 resamples, seed 2026) gives 95% intervals for micro precision **[0.4630, 0.6167]** and recall **[0.5036, 0.6235]** at IoU 0.50 and the frozen threshold. These intervals reflect observed group sampling; they do not estimate training-seed variation, unseen-board independence or production-domain shift.

## Confusions and available slices

The separate confusion matrix matches class-agnostically at IoU 0.50, then records
truth rows and predicted columns, including a background/missed category. Its
counts have different semantics from same-class TP/FP/FN. Largest observed
class-to-class confusions are:

| Truth | Predicted | Class-agnostic matched boxes |
| --- | --- | --- |
| stain | scratch | 11 |
| mouse_bite | copper_burr | 6 |
| spurious_copper | scratch | 5 |
| copper_burr | spurious_copper | 4 |
| open | short | 4 |
| spurious_copper | copper_burr | 4 |
| missing_copper | stain | 3 |
| mouse_bite | spurious_copper | 3 |
| spurious_copper | stain | 3 |
| stain | missing_copper | 3 |

| Image subset | Images | AP50 | AP50:.95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| images_with_small_box_lt32px_squared | 384 | 0.3648 | 0.1914 | 0.4959 | 0.4919 |
| images_with_medium_box_32to96px_squared | 253 | 0.4542 | 0.3042 | 0.5536 | 0.5736 |
| images_with_large_box_ge96px_squared | 127 | 0.3428 | 0.2416 | 0.6582 | 0.6082 |

These are overlapping image subsets containing at least one box of the named
area; all labels in each image remain in the metric. They are not COCO object-area
AP and their results cannot be added together. Natural slice metadata is missing
for lighting, factory, acquisition timestamp, natural blur severity. Filename prefixes are unverified
source proxies, not confirmed physical board, factory or acquisition-session IDs.

## Confidence behavior and threshold choice

The score threshold maximizes recorded validation micro F1; it is frozen before
the final test evaluation. The validation tradeoff is:

| Score threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.0121 | 0.8940 | 0.0239 | 734 | 59980 | 87 |
| 0.025 | 0.0152 | 0.8551 | 0.0298 | 702 | 45599 | 119 |
| 0.05 | 0.0267 | 0.8002 | 0.0516 | 657 | 23977 | 164 |
| 0.1 | 0.1612 | 0.7077 | 0.2625 | 581 | 3024 | 240 |
| 0.15 | 0.3576 | 0.6224 | 0.4542 | 511 | 918 | 310 |
| 0.2 | 0.4754 | 0.5530 | 0.5113 | 454 | 501 | 367 |
| 0.25 | 0.5453 | 0.4762 | 0.5085 | 391 | 326 | 430 |
| 0.3 | 0.5663 | 0.4007 | 0.4693 | 329 | 252 | 492 |
| 0.4 | 0.6196 | 0.2996 | 0.4039 | 246 | 151 | 575 |
| 0.5 | 0.6395 | 0.2290 | 0.3372 | 188 | 106 | 633 |
| 0.6 | 0.6550 | 0.1827 | 0.2857 | 150 | 79 | 671 |
| 0.7 | 0.6722 | 0.1474 | 0.2418 | 121 | 59 | 700 |
| 0.8 | 0.7328 | 0.1169 | 0.2017 | 96 | 35 | 725 |
| 0.9 | 0.8242 | 0.0914 | 0.1645 | 75 | 16 | 746 |

Held-out detection scores across classes are binned below. Observed precision uses
the ranked same-class IoU 0.50 matching outcome and weights each retained detection
equally. Bins include low-score candidates below the operating threshold.

| Score bin | Detections | Mean score | Observed precision |
| --- | --- | --- | --- |
| [0.0,0.1) | 68420 | 0.0368 | 0.0020 |
| [0.1,0.2) | 2748 | 0.1329 | 0.0539 |
| [0.2,0.3) | 373 | 0.2415 | 0.3566 |
| [0.3,0.4) | 166 | 0.3444 | 0.5422 |
| [0.4,0.5) | 98 | 0.4473 | 0.5918 |
| [0.5,0.6) | 50 | 0.5407 | 0.6200 |
| [0.6,0.7) | 38 | 0.6480 | 0.5263 |
| [0.7,0.8) | 36 | 0.7574 | 0.7500 |
| [0.8,0.9) | 52 | 0.8528 | 0.7500 |
| [0.9,1.0] | 108 | 0.9520 | 0.8611 |

This is a descriptive confidence analysis. No calibration fit or probability
guarantee is claimed; class imbalance, matching and retained-candidate selection
affect these observations. Per-class curves and histograms remain in
[the complete report](../reports/rtx4060-study-v2/holdout/report.json) and its
[plots](../reports/rtx4060-study-v2/holdout/precision_recall_and_scores.png).

## Synthetic robustness

The following comparisons all use the same **100 validation images**:
the first sorted IDs, with unchanged labels and geometry. The reference is their
own clean-subset result, not full-validation performance. Perturbations are
brightness×0.7, Gaussian blur radius 1 and JPEG quality 50. They are artificial
stress tests rather than measured naturally occurring production conditions.
The clean reference is a supplementary batch-1 evaluation of the frozen released weights on the identical IDs. This matches the one-image-at-a-time perturbation pipeline; its artifact, weight, manifest and threshold identities are checked. The original batched clean reference remains unchanged in robustness.json as historical evidence. The companion robustness-control.json supplies the controlled baseline used for the deltas below. No test images or model/threshold changes are involved in this additional validation-only control.

| Variant | Images | AP50 | AP50:.95 | ΔAP50:.95 vs clean | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Unperturbed clean control, batch 1 | 100 | 0.5767 | 0.3403 | 0.000000 | 0.5734 | 0.6777 |
| brightness_0.7 | 100 | 0.5841 | 0.3404 | 0.000049 | 0.5786 | 0.6694 |
| gaussian_blur_radius_1 | 100 | 0.5771 | 0.3471 | 0.006792 | 0.6043 | 0.6942 |
| jpeg_quality_50 | 100 | 0.5559 | 0.3099 | -0.030406 | 0.5766 | 0.6529 |

## Qualitative sample selection

The evaluator takes up to 12 evenly spaced images across the entire sorted holdout,
then adds maximal observed TP, FP, FN, poor-localization-candidate and overlapping
prediction cases when the category is nonzero. Ties use a deterministic image-ID
rule. Reasons are preserved in [examples.json](../reports/rtx4060-study-v2/holdout/examples.json).
This combines broad coverage with explicit hard cases; it does not present only
successful predictions. Truth is green on the left, model predictions red on the
right, using the frozen operating threshold.

| Image ID | Rendered comparison | Selection reason |
| --- | --- | --- |
| 001603633495e00bdde3 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_00.jpg) | evenly spaced sorted split sample |
| 18cd75c3fd8fef7f6b55 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_01.jpg) | evenly spaced sorted split sample |
| 312b35936bf7fe6043d3 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_02.jpg) | evenly spaced sorted split sample |
| 4c35813950b33bd73524 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_03.jpg) | evenly spaced sorted split sample |
| 641634b13a789fa4ade8 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_04.jpg) | evenly spaced sorted split sample |
| 7c0898c8cc8e90ea54b3 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_05.jpg) | evenly spaced sorted split sample |
| 90366f22d8a360053624 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_06.jpg) | evenly spaced sorted split sample |
| a40383be69e00b5cbcba | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_07.jpg) | evenly spaced sorted split sample |
| be009a51141157be197c | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_08.jpg) | evenly spaced sorted split sample |
| d2f1e136b4635eb1a9cf | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_09.jpg) | evenly spaced sorted split sample |
| eb14bcfa42658c98dc39 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_10.jpg) | evenly spaced sorted split sample |
| ffe13ebcc5de28620ac4 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_11.jpg) | evenly spaced sorted split sample |
| 16101cfb88bb7f77909c | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_12.jpg) | maximum tp in split (13) |
| 4700eb7cdd4b83400c18 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_13.jpg) | maximum fp in split (5) |
| cf18f29e21f446ea28fc | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_14.jpg) | maximum fn in split (6) |
| dff47a68647dc1a34630 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_15.jpg) | maximum poor_localization_candidates in split (3) |
| deb2a80f42394090aaf0 | [Image](../reports/rtx4060-study-v2/holdout/examples/sample_16.jpg) | maximum overlapping_prediction_pairs in split (3) |

## Failed experiment and diagnostic limits

The initial `rtx4060-study-v1` attempt used AMP and learning rate
0.001 and stopped before its first completed
epoch with a nonfinite forward loss. It produced no accuracy result and did not
access final holdout images. A bounded diagnostic replay used the same initialized
weights and training-only samples:

| Precision | Learning rate | Batches | Nonfinite forward batches | Nonfinite-gradient batches | First total loss | Last total loss |
| --- | --- | --- | --- | --- | --- | --- |
| AMP | 0.001 | 50 | 0 | 8 | 9.4486 | 13.3387 |
| FP32 | 0.0003 | 50 | 0 | 0 | 9.6986 | 9.4218 |

The replay did not reproduce the original nonfinite forward failure. It changed
precision and learning rate together, so it cannot isolate which caused the
stability difference. Fifty batches per recipe do not establish full-run stability.
The separately named FP32 study has its own completed histories and held-out
results; the failed attempt is retained as a negative finding. See
[diagnostic evidence](../reports/training-diagnosis.json) and
[initial failure record](../reports/rtx4060-study-v1/failure.json).

Generated deterministically from recorded evidence by `scripts/summarize_study.py`.
