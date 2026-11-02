# Training and evaluation protocol

The detector is torchvision 0.22.1 `ssdlite320_mobilenet_v3_large` under PyTorch
2.7.1. Its COCO_V1 checkpoint supplies the backbone and box regressor. The
classification head is replaced with a seeded, normally initialized head for
the dataset classes plus background. Class zero is background and is never a
reported defect. This is transfer learning on real inspection images; the
training input is image pixels alone, never masks or target annotations.

`python -m inspection_dl.train --help` and
`python -m inspection_dl.evaluate --help` list every supported argument.
Use the preparation commands in DATASET.md first. Set `--manifest` to the
prepared manifest JSON and `--data-root` to the directory against which its
`image` paths are relative. The commands below use those placeholders so they
remain independent of a particular checkout location.

```powershell
python -m inspection_dl.train --manifest MANIFEST --data-root DATA_ROOT --output runs/no_flip --epochs 12 --batch-size 8 --device cuda --amp
python -m inspection_dl.train --manifest MANIFEST --data-root DATA_ROOT --output runs/flip --epochs 12 --batch-size 8 --device cuda --amp --horizontal-flip
```

The controlled comparison changes horizontal reflection probability from zero
to 0.5, with the same seed, split, optimizer, model initialization, batch size,
precision and epoch budget. Reflection also transforms target boxes. Both
variants see all training images each epoch. `--freeze-backbone` is available
for future explicitly documented studies, but is not part of this comparison.
No architecture or hyperparameter search is implied.

## Bounded resources and optimization

Defaults are 12 epochs, batch size 8, seed 42, four CPU threads, AdamW learning
rate 0.001 and weight decay 0.0005, with cosine learning-rate decay over the
fixed epoch budget. AMP is explicit and CUDA-only; omit `--amp` for FP32. The
intended machine is the documented RTX 4060 laptop GPU with approximately
8 GB VRAM. There is no cloud or multi-GPU requirement. `--device cpu` supports
CPU training/inference but full training is expected to be slower.

BatchNorm uses frozen running statistics during fine-tuning, while affine
parameters remain trainable unless their backbone is frozen. This makes
small and final singleton batches valid. Gradients are clipped to norm 10.
GradScaler skips transient FP16 overflow updates and records their count in
each epoch; a nonfinite forward loss fails clearly. CUDNN benchmarking is off
and its deterministic mode is on. Seeds reduce variability but do not promise
bitwise agreement between CUDA versions, devices or operating systems.

## Image and geometry conventions

Pillow applies EXIF orientation, converts grayscale/RGBA to RGB, and creates
a floating CHW tensor in [0,1]. Prepared annotation dimensions must match the
EXIF-oriented image. Torchvision stretches each image to 320 by 320 without
letterboxing, normalizes with mean/std [0.5,0.5,0.5], and rescales predicted
boxes to the original oriented dimensions during evaluation. Boxes are
`[x_min,y_min,x_max,y_max]` pixels with exclusive upper bounds. Invalid target
geometry, changed dimensions and checksum mismatches fail. Model outputs with
nonfinite or zero-area boxes are omitted rather than included as detections.

## Portable run evidence and resume

Each run writes `run.json`, `history.json`, `prior.json`,
`baseline_validation.json`, `best_validation.json`, and validation predictions.
`initialized.pt` preserves the actual pretrained-backbone/new-head state before
training for an optional untrained-head comparison. `last.pt` saves each epoch;
`best.pt` retains the earliest checkpoint with the highest validation AP50:.95.
The primary simple baseline uses training annotations only: one median
normalized box per class and its training-image prevalence as confidence.
It proposes the same normalized geometry for every image. Its limitations
are explicit; it does not use held-out labels or claim to be a trained network.

The checkpoint contains model, optimizer, cosine scheduler, AMP scaler, epoch,
history, and Python/Torch/loader/CUDA RNG states, using `weights_only=True`
compatible types. Resume the same run directory and identical configuration,
including the original total epoch count, seed, data-manifest digest, precision,
and augmentation:

```powershell
python -m inspection_dl.train --manifest MANIFEST --data-root DATA_ROOT --output runs/no_flip --epochs 12 --batch-size 8 --device cuda --amp --resume runs/no_flip/last.pt
```

Resume occurs at epoch boundaries. An interrupted partial epoch is rerun from
the last completed checkpoint. Exact replay across different runtime/device
versions is not guaranteed. Existing run outputs are not overwritten by a new
training invocation without explicit resume. Keep a run's `best.pt` alongside
`last.pt` when resuming, so its prior best release candidate is preserved.

## Validation selection and final holdout

Training accesses only train and validation images. Choose the augmentation
variant by its best validation AP50:.95. Select an operating score threshold
from that model's validation micro F1 at IoU 0.5; the higher threshold wins F1
ties. This balances review workload and missed detections for the demonstration
and is not evidence of a manufacturing acceptance criterion.

```powershell
python -m inspection_dl.evaluate --manifest MANIFEST --data-root DATA_ROOT --checkpoint runs/SELECTED/best.pt --output reports/validation --split val --select-threshold --device cuda --robustness
python -m inspection_dl.evaluate --manifest MANIFEST --data-root DATA_ROOT --checkpoint runs/SELECTED/best.pt --output reports/holdout --split test --final-holdout --threshold SELECTED_THRESHOLD --device cuda --robustness
```

The test command requires explicit `--final-holdout` and forbids threshold
selection on test. Report the final model once after validation choices are
frozen. Changing parsing or metric correctness may require a documented
recomputation; do not describe such a recomputation as an unseen holdout.
Evaluation refuses to overwrite an existing report directory containing
`report.json`. The report records checkpoint and manifest digests, split,
threshold, checkpoint epoch and training-run identity.

## Metric definitions and limits

Predictions are class-aware post-NMS, score >=0.001, at most 100 per image.
AP ranks scores with a deterministic tie break, greedily matches each prediction
to its best unmatched same-class ground truth at each IoU threshold, builds a
monotonic precision envelope, and averages it at 101 recall levels. AP50 is at
IoU 0.5. AP50:.95 averages ten thresholds 0.50,0.55,...,0.95 and classes with
ground truth. Classes absent from the evaluation split have null AP and are
excluded from the macro denominator; their false-positive counts remain visible.
There is no crowd/ignore handling or COCO object-area/maxDet averaging. Results
are labeled COCO-style, without claiming general COCOeval equivalence.

At the selected score threshold, per-class and micro TP/FP/FN, precision, recall
and F1 use same-class IoU0.5 matching. The separate confusion matrix uses
class-agnostic IoU0.5 matching, with truth rows, prediction columns, and an
extra missed/background category; therefore its errors need not equal
same-class TP/FP/FN. PR curves, score histograms and binned observed detection
precision versus mean confidence expose confidence behavior. Scores are not
assumed to be calibrated probabilities.

Group bootstrap percentile intervals resample acquisition/duplicate groups
and estimate micro precision/recall variability at the operating threshold.
They do not measure training-seed variability or domain shift. Intervals are
omitted for fewer than five groups. Size slices are overlapping image subsets
containing at least one small (<32 squared pixels), medium (32 squared to
96 squared) or large (>=96 squared) box; all labels in selected images remain
in the metric. These are not COCO object-area AP measures.

Qualitative examples include evenly spaced sorted split samples plus maximal
TP, FP, FN, poor-localization and overlapping-box cases. Each image's selection
reason is recorded in `examples.json`; truth appears left and actual model
predictions right. `per_image_errors.json` covers the entire split. Synthetic
robustness checks use the first 100 sorted split images (configurable), compare
the identical clean subset, and apply brightness 0.7, Gaussian blur radius 1,
and JPEG quality 50. They do not establish naturally occurring robustness.
Unavailable acquisition/lighting metadata is disclosed, never fabricated.

## CI scope

Tests use generated, explicitly synthetic fixtures for hand-computable metric
cases, NumPy-versus-torchvision IoU comparison, horizontal-flip geometry,
EXIF handling, path/dimension validation, and real SSDLite forward/backward,
optimizer update, original-coordinate inference and exact CPU load/save parity.
They need neither the full dataset nor downloaded pretrained weights. Full GPU
experiments and final reports are separate evidence, with recorded environments.
