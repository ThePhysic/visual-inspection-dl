# Results and evidence index

Start with the images and summaries below; large raw prediction files are listed
last. The released detector was evaluated once on the frozen **722-image test
holdout**, after validation-only checkpoint and threshold selection. Operating
counts use score **0.20** and same-class IoU **0.50** matching. AP50 is **44.17%**;
AP50:.95 is **27.19%**, with micro precision **53.31%** and recall **56.76%** at
that operating point. These are results on 300×300 AOI candidate ROI patches;
full-board search and production performance are unproven.

## Images, plots and readable summaries

In the original comparison images, **green is ground truth on the left; red is
model predictions on the right**, at the frozen score threshold 0.20. The two
ordinary samples below were selected by the evaluator's evenly spaced sorted
split sampling; the hard case maximizes false negatives. The README displays
these three comparisons at their original 600×332 size.

| Start here | What it shows |
| --- | --- |
| [True detection: sample 02](../reports/rtx4060-study-v2/holdout/examples/sample_02.jpg) | Annotated `scratch` detected at score 0.95; 1 TP, 0 FP, 0 FN. |
| [Wrong-class failure: sample 04](../reports/rtx4060-study-v2/holdout/examples/sample_04.jpg) | `stain` predicted as `spurious_copper` at score 0.97; 0 TP, 1 FP, 1 FN. |
| [Challenging patch: sample 14](../reports/rtx4060-study-v2/holdout/examples/sample_14.jpg) | Split's maximum-FN example: `mouse_bite` detected; three `missing_copper` and three `stain` missed; extra `scratch` FP. 1 TP, 1 FP, 6 FN. |
| [All 17 holdout comparisons](ERROR_ANALYSIS.md#qualitative-sample-selection) | Deterministic coverage plus maximum-error cases, with image IDs and selection reasons. |
| [Holdout PR curves and score plots](../reports/rtx4060-study-v2/holdout/precision_recall_and_scores.png) | Per-class precision–recall behavior and descriptive confidence distributions; scores are uncalibrated. |
| [Per-class metrics](MODEL_CARD.md#quantitative-results) | AP, precision, recall, support counts and uncertainty; `open` has only 1 of 18 test instances detected. |
| [Training history plot](../reports/rtx4060-study-v2/validation/training_history.png) | Recorded training/validation history, separate from the final test evidence. |
| [Model card](MODEL_CARD.md) | Architecture, data, model selection, frozen results, release identity, performance and use limits. |
| [Error analysis](ERROR_ANALYSIS.md) | False alarms, misses, class confusions, confidence behavior and validation-only synthetic robustness checks. |
| [Reproducibility report](REPRODUCIBILITY.md) | Commands, environment, split and release provenance, recorded hashes and limitations. |
| [Release-bound benchmark JSON](../reports/rtx4060-study-v2/benchmark-release.json) | Identity-checked CPU/GPU timings and batch comparison on 100 validation images. CPU/GPU batch-1 p50 54.38/36.79 ms measures warmed forward processing, not HTTP response latency. |

Images are **PCB-IND v4 by Hao Yan, Fujian Polytechnic Normal University**,
[Zenodo record 19723114](https://zenodo.org/records/19723114), **CC BY 4.0**, with
annotation/prediction overlays. [Dataset attribution](DATASET.md) and
[third-party terms](THIRD_PARTY.md) distinguish dataset and pretrained/COCO terms
from the MIT source-code license.

## Raw evidence and provenance

The files below preserve the underlying observations. JSON files marked large
can be awkward to preview on GitHub; use the readable sources above first.

| Evidence | Purpose |
| --- | --- |
| [Holdout example selection](../reports/rtx4060-study-v2/holdout/examples.json) | Original image IDs, filenames and deterministic selection labels for the rendered comparisons. |
| [Holdout error counts by image](../reports/rtx4060-study-v2/holdout/per_image_errors.json) | TP/FP/FN and diagnostic counts used to cross-check the gallery captions. |
| [Holdout report JSON](../reports/rtx4060-study-v2/holdout/report.json) **(large)** | Full metrics, baseline results, per-class curves, score bins, confusion and uncertainty. |
| [Holdout predictions JSON](../reports/rtx4060-study-v2/holdout/predictions.json) **(large)** | Original boxes, labels and scores, including candidates below the operating threshold. |
| [Validation report JSON](../reports/rtx4060-study-v2/validation/report.json) **(large)** | Validation metrics and threshold tradeoffs used before the holdout evaluation. |
| [Selection decision](../reports/rtx4060-study-v2/selection.json) | Recorded checkpoint/threshold choice before the final holdout. |
| [Release binding](../reports/rtx4060-study-v2/release-binding.json) | Connects the released bundle to recorded study evidence. |
| [Release metadata](../releases/pcb-ind-ssdlite-v1/metadata.json) | Model/taxonomy versions, processing conventions, artifact identity and licensing record. |
| [Released evaluation JSON](../releases/pcb-ind-ssdlite-v1/evaluation.json) **(large)** | The exact evaluation report packaged with the immutable local release. |
| [Documentation input hashes](../reports/rtx4060-study-v2/documentation-inputs.json) | Input identities for the generated model card, error analysis and reproducibility report. |

The train-derived median-box class-prior reference (AP50 **2.09%**, AP50:.95
**1.03%**) is a geometric sanity baseline, not a competitive detector. There is
one training seed; filename/source groups are heuristic proxies; confidence is
uncalibrated; rare-class evidence is limited. See the model card before drawing
conclusions about new factories, cameras or manufacturing deployment.

[Back to the demonstration and local serving commands](../README.md)
