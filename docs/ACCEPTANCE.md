# Scope and evidence map

The accepted goal is the standalone ML component in [the idea package](../IDEA_PACKAGE.md),
with parallel implementation, coherent git commits, independent Astra medium
adversarial review until clean, and verified publication to a private GitHub
repository. The existing application is outside this milestone.

| Deliverable | Implementation and evidence |
| --- | --- |
| Original-source licensed dataset and taxonomy | `docs/DATASET.md`, `data/manifests/pcb-ind-v4.source.json`, `scripts/acquire_data.py` |
| Validated manifest, exclusions, duplicates and source groups | `src/inspection_dl/data.py`, committed manifest/audit, `tests/test_data.py` |
| Independent source-format verification | `scripts/crosscheck_annotations.py`, `reports/validation/annotation-crosscheck.json` |
| Real PyTorch transfer learning and controlled comparison | `src/inspection_dl/model.py`, `train.py`, `scripts/run_study.py`, per-run configs/histories under `reports/rtx4060-study-v2/experiments/` |
| Baseline and rigorous held-out evaluation | `metrics.py`, `evaluate.py`, validation/holdout reports, predictions, curves, confusion and slices |
| Metric implementation checked independently | `scripts/check_metric_reference.py`, `reports/metric-reference.json`, hand-computable tests |
| Confidence, robustness, variability and failures | Threshold sweep, confidence reliability bins, same-subset synthetic robustness, group-bootstrap operating-point intervals and representative/worst-case images |
| Honest unsuccessful experiment evidence | `reports/rtx4060-study-v1/`, `reports/training-diagnosis.json`, bounded replay script |
| Reconstructable implementation provenance | Training source/run hashes and `reports/validation/source-reconstruction.json` |
| Immutable trained release | `releases/pcb-ind-ssdlite-v1/`, digest-validated metadata, weights, evaluation and source licensing |
| Standalone local HTTP serving | `api.py`, `schemas.py`, `docs/API.md`, real-model HTTP demo evidence |
| CPU fallback and quality/performance tradeoff | `benchmark.py`, validation-subset CPU/GPU and batch-size measurements with quality, timing and memory |
| Practical software checks | Data/metric/model/contract/bundle tests, real gradient/update/parity/resume checks, Ruff, CPU-only environment evidence |
| Reproducible environments and CI | Hash-pinned CPU lock, exact Windows CUDA snapshot, Docker recipe, Windows/Linux CPU workflow |
| Reviewer-visible conclusions and limitations | `MODEL_CARD.md`, `ERROR_ANALYSIS.md`, `REPRODUCIBILITY.md`, README and raw reports |

Confidence reliability is descriptive and does not claim calibrated probabilities.
Bootstrap intervals cover sampled source groups at a fixed threshold, not training-seed
variance or certified population coverage. Synthetic fixtures and perturbations remain
identified separately from natural data and measured trained-model predictions.

Dataset, experiment, release and service evidence are required independently of a
review verdict. A clean review does not substitute for actual training or evaluation.
GitHub privacy and commit identity are verified after publication; local CPU/GPU
measurements are not production guarantees.
