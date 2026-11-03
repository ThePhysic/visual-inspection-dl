# Scope and evidence map

This repository covers offline dataset preparation, PyTorch defect-detector
training and evaluation, an immutable model bundle, and standalone local
inference. Start with the [visual results](../README.md#visual-results) and
[results index](RESULTS_INDEX.md), then follow the evidence below.

| Deliverable | Implementation and evidence |
| --- | --- |
| Licensed dataset and explicit taxonomy | [Dataset audit](DATASET.md), [acquisition record](../data/manifests/pcb-ind-v4.acquisition.json), [acquisition script](../scripts/acquire_data.py) |
| Frozen manifest, exclusions, duplicate and source groups | [Data preparation](../src/inspection_dl/data.py), [manifest](../data/manifests/pcb-ind-v4.json), [audit](../data/manifests/pcb-ind-v4.audit.json), [data tests](../tests/test_data.py) |
| Independent annotation verification | [COCO/YOLO cross-check](../scripts/crosscheck_annotations.py), [verification record](../reports/validation/annotation-crosscheck.json) |
| Real transfer learning and controlled comparison | [Detector](../src/inspection_dl/model.py), [training loop](../src/inspection_dl/train.py), [study runner](../scripts/run_study.py), [experiment records](../reports/rtx4060-study-v2/experiments/) |
| Geometric sanity baseline and held-out evaluation | [Metric definitions](TRAINING.md#metric-definitions-and-limits), [evaluator](../src/inspection_dl/evaluate.py), [model card](MODEL_CARD.md), [holdout report](../reports/rtx4060-study-v2/holdout/report.json) |
| Independently checked AP calculation | [Reference check](../scripts/check_metric_reference.py), [COCOeval comparison](../reports/metric-reference.json), [metric tests](../tests/test_metrics.py) |
| Confidence, robustness, uncertainty and failures | [Error analysis](ERROR_ANALYSIS.md), [matched-batch robustness control](../reports/rtx4060-study-v2/validation/robustness-control.json), [held-out examples](../reports/rtx4060-study-v2/holdout/examples.json) |
| Unsuccessful experiments retained | [Initial failed study](../reports/rtx4060-study-v1/), [bounded numerical diagnosis](../reports/training-diagnosis.json) |
| Reconstructable source and experiment identity | [Reproducibility report](REPRODUCIBILITY.md), [source reconstruction](../reports/validation/source-reconstruction.json) |
| Immutable model release | [Bundle files](../releases/pcb-ind-ssdlite-v1/), [metadata and digests](../releases/pcb-ind-ssdlite-v1/metadata.json), [bundle validation](../src/inspection_dl/bundle.py) |
| Standalone typed HTTP inference | [API contract](API.md), [service](../src/inspection_dl/api.py), [native CPU demo](../reports/validation/bug-audit-api.json), [container demo](../reports/validation/bug-audit-container.json) |
| CPU/GPU quality and performance measurements | [Release-bound benchmark](../reports/rtx4060-study-v2/benchmark-release.json), [benchmark implementation](../src/inspection_dl/benchmark.py) |
| Practical tests and CPU-only release smoke | [Test suite](../tests/), [checked-in release smoke](../tests/test_released_model_smoke.py), [Windows/Linux CI](../.github/workflows/ci.yml) |
| Reproducible serving environment | [Deployment instructions](DEPLOYMENT.md), [CPU requirements](../requirements-cpu.txt), [Docker recipe](../Dockerfile) |

The model operates on 300×300 candidate ROI patches, not full-board search. The
recorded comparison uses one training seed and heuristic source-group proxies.
Confidence reliability is descriptive, not proof of calibrated probabilities.
Group-bootstrap intervals describe sampled groups at a fixed threshold; they do
not measure training-seed variance or certified population coverage. Synthetic
fixtures and perturbations are identified separately from held-out evidence.

The geometric class-prior baseline is a sanity reference, not a competitive
detector. The [model card](MODEL_CARD.md) preserves its exact metrics alongside
the trained model's measured limitations. Local forward-pipeline timings are
distinct from end-to-end HTTP latency and are not production guarantees.

The separate application owns uploads, queues, storage, and human review; joint
integration remains [outside this component](ARCHITECTURE.md#later-integration).
The [historical concept brief](HISTORICAL_IDEA_PACKAGE.md) records the initial
rationale. Code, dataset, and pretrained-weight terms remain separate in
[third-party attribution](THIRD_PARTY.md).
