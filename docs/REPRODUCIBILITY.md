# Reproducibility report

The released bundle `pcb-ind-ssdlite-v1` is tied to completed local GPU
experiments, validation-based selection, one final holdout evaluation, CPU/GPU
benchmarks and recorded provenance. This report is generated from actual JSON
evidence and rejects incomplete runs or mismatched dataset, checkpoint, threshold,
taxonomy, artifact and report identities.

## Recreate the environment and acquire data

Python 3.12 is supported. The demonstrated Windows training environment was:

| Recorded field | Value |
| --- | --- |
| cuda | 12.6 |
| device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.12.15 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |

From the repository root on Windows:

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-gpu.txt
.venv-gpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-gpu/Scripts/python.exe scripts/acquire_data.py
.venv-gpu/Scripts/python.exe -m inspection_dl.data --root data/raw/pcb-ind --output data/manifests/reproduced.json --seed 42
.venv-gpu/Scripts/python.exe scripts/crosscheck_annotations.py
```

`requirements-gpu.txt` snapshots the demonstrated Windows CUDA environment.
`requirements-cpu.txt` provides fully hash-pinned serving/CI dependencies.
Linux CPU setup uses `python3.12 -m venv` and `.venv/bin/python`;
see [deployment setup](DEPLOYMENT.md) for platform-specific instructions.
With uv, both official indexes must be considered using
`--index-strategy unsafe-best-match` when installing the requirements snapshots.

Acquisition downloads only the pinned author archive from
https://zenodo.org/records/19723114, checks size 101336845 bytes,
MD5 `1325f8dffeb73bef4db34fbd22b31ec5` and
SHA256 `29cf919c64cab7e60e90e16bf3f42cce674688f99295d66a8e10d0066836e3c9`, and extracts the YOLO copy and metadata.
Raw data remains outside Git. Verify the reproduced canonical manifest digest
against **`e33d69b335bf830eec61cfd7cec4348f4f57d87291941f8679b9f57b1fc37d31`**. JSON file SHA256 is a separate
byte-level identity, while canonical content hashing ignores formatting.
Frozen outputs refuse conflicting overwrites.
Git attributes preserve checksummed JSON bytes across Windows/Linux checkouts.
A discovered CRLF-to-LF Git normalization mismatch was corrected without changing
the frozen dataset content or bytes used by training; the retained
[portability check](../reports/validation/manifest-portability.json) confirms that
the committed blob and working manifest have the same training-bound file SHA256.

The independent source-format audit checks original COCO/YOLO class and coordinate
semantics, all accepted image and annotation hashes, decoded dimensions and EXIF.
It reports **4788** accepted
images with status `pass`. The metric-reference report
cross-checks supported no-crowd AP behavior against pycocotools using explicitly
synthetic fixtures; it is not evidence of trained accuracy.

## Reproduce bounded training and selection

```powershell
.venv-gpu/Scripts/python.exe scripts/run_study.py --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --epochs 12 --batch-size 8 --device cuda --learning-rate 0.0003 --study-id reproduced-study --release reproduced-model
```

Fresh study/release IDs preserve existing evidence. The runner performs both
augmentation variants, selects the best validation AP50:.95 checkpoint, selects a
validation micro-F1 threshold, writes the decision, evaluates the final holdout,
packages a bundle, and benchmarks CPU/GPU batch 1 versus 4 on an identical subset.
Before selection, it verifies exact initial parameter/buffer equality and records
`initialization-parity.json`, including tensor count and an initial-tensor digest.
Individual commands and epoch-boundary resume are documented in [TRAINING.md](TRAINING.md).

| Run | Seed | Epochs | Batch | LR | Horizontal flip p | AMP | Elapsed min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rtx4060-study-v2-no-flip | 42 | 12 | 8 | 0.0003 | 0.0 | False | 21.35 |
| rtx4060-study-v2-flip | 42 | 12 | 8 | 0.0003 | 0.5 | False | 16.21 |

AdamW, weight decay 0.0005 and cosine scheduling are recorded in each run's config.
BatchNorm running statistics remain frozen; gradients are clipped to norm 10.
Checkpoints retain model, optimizer, scheduler, scaler and RNG states. Resume
requires the original configuration, epoch budget, run directory and best/last
checkpoints. Training was local and single-GPU; there is no cloud dependency.
Seeds and deterministic CUDNN settings reduce variability but do not promise
bitwise agreement across CUDA versions, operating systems, devices or libraries.
One training seed does not establish seed variability or generalization certainty.

## Source identity and immutable evidence

The package hash uses sorted direct `src/inspection_dl/*.py` files, appending
each UTF8 basename, NUL, exact file bytes, and NUL into SHA256. Some runs began
with a dirty working tree, so their launch revision alone does not identify the
actual source. The generator independently reconstructs package bytes from
retained Git commits and requires exact recorded-hash matches:

| Run | Recorded launch revision | Dirty at launch | Actual package hash | Retained exact matching commit |
| --- | --- | --- | --- | --- |
| rtx4060-study-v2-no-flip | 33a3fb4d9e28c772a5c3a24b83d67668f2233f3d | True | fbe981ab9ef719e3d3692df9c83e178619e24eaa665a25d759837bbaa9b89277 | 678e693d65d334871e7f5980caab85dc2670c15f |
| rtx4060-study-v2-flip | 4a1bef9e968828ffee43470abbccd9b2a3a04d02 | True | f2c97b02b6b4ff3e9b29438595c0611be8a19e5a65ed3e7975123862d24bd5a4 | 567c8133dc8c9b12b3a43519e60c83512648a200 |

Training/model/metric modules are compared by Python AST across the retained
snapshots and must agree, allowing formatting-only changes. Other modules may
change between runs without changing training computation; the complete source
identities remain explicit. Source reconstruction and every documentation input
digest are retained in `reports/rtx4060-study-v2/documentation-inputs.json`.

Selected checkpoint SHA256: `c6d6532da7201fd8f8ba182117e4e92eca5562578b9592bf50b9a44c8670227b`.
Released weights SHA256: `28271a688aa7335521bb7725423fd6ea6622f3d1964bfaed83dd97b4f0ff34d9`.
Artifact digest: `9037234654bcc190f6f5dd8cda07e70a119ba19b139b2b99118f4d9f27d7e05e`.
The checkpoint contains training state; the released weights are a separate
tensor-only state dictionary, so their distinct hashes are expected.
Bundle metadata also pins preprocessing, class mapping, thresholds, runtime,
dataset manifest and evaluation digest. Loading validates identity and rejects
missing, incompatible or corrupted files before serving.

## Standalone inference and evaluation commands

```powershell
py -3.12 -m venv .venv-cpu
.venv-cpu/Scripts/python.exe -m pip install -r requirements-cpu.txt
.venv-cpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-cpu/Scripts/python.exe -m inspection_dl.api --bundle releases/pcb-ind-ssdlite-v1 --device cpu
```

The released bundle runs without raw data, training directories or downloaded
pretrained initialization weights. It serves localhost by default. Submit the
attributed sample image from a second terminal:

```powershell
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@tests/fixtures/pcb-ind/5810_t_007.jpg" http://127.0.0.1:8000/v1/predict
```

To repeat evaluation from a fresh clone, use the released bundle and acquired
dataset directly, with a fresh output directory. This CPU validation-split recheck
does not require any ignored training checkpoint or optimizer state:

```powershell
.venv-cpu/Scripts/python.exe -m inspection_dl.evaluate --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --bundle releases/pcb-ind-ssdlite-v1 --output reports/release-validation-recheck --split val --device cpu
.venv-gpu/Scripts/python.exe -m inspection_dl.benchmark --bundle releases/pcb-ind-ssdlite-v1 --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --output reports/recomputed-benchmark.json --samples 100 --devices cpu cuda --batch-sizes 1 4
```

Bundle evaluation uses the release's recorded confidence threshold by default
and reports its artifact digest, weights hash and model version as the evaluated
identity. The original training-checkpoint digest remains historical provenance;
it is not presented as an input file when evaluating a bundle. Evaluation reports
also record the evaluator's actual Git revision and source-package hash separately
from the training source. The original study's final holdout remains its frozen
report; this repeatability command evaluates validation images.
Full training is reproducible through the fresh-study command. Raw data and
optimizer checkpoints are ignored by Git; the trained release weights are retained
for direct inference. GPU training, full-data evaluation and local benchmarks
were measured separately from ordinary CPU CI.

## Validation and disclosure

```powershell
.venv-cpu/Scripts/python.exe -m pytest
.venv-cpu/Scripts/python.exe -m ruff check .
.venv-cpu/Scripts/python.exe -m ruff format --check .
python scripts/summarize_study.py --study reports/rtx4060-study-v2 --release releases/pcb-ind-ssdlite-v1
```

Tests use generated synthetic fixtures and one attributed real image/label pair;
they do not repeat full GPU training. See the committed CI workflow and validation
records for actual checks. The failed initial AMP/0.001 study and confounded
FP32/0.0003 diagnostic replay remain disclosed in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).
Natural acquisition metadata is absent, source grouping is a heuristic, and rare
classes limit inference strength. No production SLA, calibrated probability,
manufacturing deployment or safety-critical claim follows from these results.

The source code is MIT, PCB-IND data is CC BY 4.0, and pretrained/COCO terms remain
separate. Attribution and model-use limitations are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md), release metadata and licensing JSON.

## Documentation input checksums

| Evidence path | SHA256 |
| --- | --- |
| .gitattributes | aead7ef56095f276d9f6a4edd98086fa6d2fed55f584cb80b7e5436b35659067 |
| data/manifests/pcb-ind-v4.acquisition.json | b562c101379bcce36266acaab95dd3619c29edd0ad9c8cbeef6fde61428f7322 |
| data/manifests/pcb-ind-v4.audit.json | 3967fdb1a565b3494dc6178415bbe62534edb9da8354696f45775d3662132eee |
| data/manifests/pcb-ind-v4.json | 0b320a9502f0a785f35c7d8496eb3c7a1d8cf5a83804a452a41e271958bb6bbd |
| pyproject.toml | 4a8f9ac24c775df6f34ade26c95d1ebc7fe0d21216d637f23ceb38e71bd2af32 |
| releases/pcb-ind-ssdlite-v1/evaluation.json | acc89d13a50caf389f74bc2cb200b0558fcbe6f6ead6e7f94f6e5ab5817b2dfd |
| releases/pcb-ind-ssdlite-v1/metadata.json | d0c4ed35b18363a05e1cf01452365e3d1ad85b65fc967266f17de6563d0d3452 |
| releases/pcb-ind-ssdlite-v1/weights.pt | 28271a688aa7335521bb7725423fd6ea6622f3d1964bfaed83dd97b4f0ff34d9 |
| reports/environment.json | b01dee16de9fad14af1c89eda6d7adbf0be58b4f13873b7c5a1139666364a74a |
| reports/metric-reference.json | 6c770ceed9267eef0ea384327fce942f709f5c9567c5726baba0300a343f9d7d |
| reports/rtx4060-study-v1/failed-run.json | 2cab934fd5652614ace35a832ea321c219d255c877fc297529508327218db857 |
| reports/rtx4060-study-v1/failure.json | 89b9bf1507a232811379158dad8cdd85c08043f16ab44794b606a8b2cf3eb6b0 |
| reports/rtx4060-study-v2/benchmark.json | 6c44044acc5ab797859222b98adcca6ff5730e64b858e6d60a6635b290c4cfe0 |
| reports/rtx4060-study-v2/experiments/flip/history.json | 9eaf5c528d1be56e0e2ce3cfe93d1cd7ea7c0e5ec2b9ad948a3de802023b845c |
| reports/rtx4060-study-v2/experiments/flip/run.json | 575f819990693743a084dd7ce3f8ba6cf5dd34aa5486ac17242c0da9c4706baf |
| reports/rtx4060-study-v2/experiments/no-flip/history.json | 97915b9c5a8a423f414f72b7a8b7b61ce41557d8c907ed231ac3c22bc9b8ec47 |
| reports/rtx4060-study-v2/experiments/no-flip/run.json | b2db441d53d0fbab08466b5676b72fa1770a09597c1b0830fef53d8f17db19dc |
| reports/rtx4060-study-v2/holdout/examples.json | 68c185c27a79942f2df81a95274e9e2ef91f787601510479ee03e4a094246a82 |
| reports/rtx4060-study-v2/holdout/examples/sample_00.jpg | bd7e4c8b2dd633f3c87a1c01b3ae7f0d5726e558c77db34ec74c110c2b5e2eb2 |
| reports/rtx4060-study-v2/holdout/examples/sample_01.jpg | b5da2c5118fb9805b06e7da5ad9a58bd4ebdf71fc5d8c66ed28a5dd0a77d5830 |
| reports/rtx4060-study-v2/holdout/examples/sample_02.jpg | 363dd6c10db4b651fa738135e95e58e4b0c4c3e019dad37428da3c65b4af71f1 |
| reports/rtx4060-study-v2/holdout/examples/sample_03.jpg | 5f638efd560cefbb71cbfc6401b5efda8147d06673844327167206fe299e6e8b |
| reports/rtx4060-study-v2/holdout/examples/sample_04.jpg | 272a9effd01d35f4eefce4c425fa0e143abdf72264cc8ad81bb707708eca5aaa |
| reports/rtx4060-study-v2/holdout/examples/sample_05.jpg | 08a8f4688f15489b9361ac43e5a98fabea5901362510b7993b8904e43b20bdba |
| reports/rtx4060-study-v2/holdout/examples/sample_06.jpg | 2b0d8104d7b200b993169755dae67bc7a8294869be2241309e0341761ccafe56 |
| reports/rtx4060-study-v2/holdout/examples/sample_07.jpg | fc3a2e99e4eb69c70a841e4c9adc35c72dac214a3db6e716422b2f0e7f4c90f6 |
| reports/rtx4060-study-v2/holdout/examples/sample_08.jpg | 8c4ad4f0537a1cfe3e0c2fb19fecc032fa59a101af32f8bb9c9b5e99a8d35f75 |
| reports/rtx4060-study-v2/holdout/examples/sample_09.jpg | c5138e696da38830d7c53707764d03f1d648c58b369d864e98366c518063aab6 |
| reports/rtx4060-study-v2/holdout/examples/sample_10.jpg | 394f4c103a85169544be7f65efb996e4489e1c875669bd50593c83cf40a28a2b |
| reports/rtx4060-study-v2/holdout/examples/sample_11.jpg | cc3753cceb16dfc2ea5fec36bc3c3a28e13728c60ea1806850360795fb7b6f0e |
| reports/rtx4060-study-v2/holdout/examples/sample_12.jpg | 5122c85db6b2756dbef6de9918b4f818573d3aebe2b5aaa8ea3a1de46bc70f46 |
| reports/rtx4060-study-v2/holdout/examples/sample_13.jpg | 97b8c94fe0a50cb20e1efd88ed45dbeae1b12a9b7e901592b4bc5f97cbcbbcc9 |
| reports/rtx4060-study-v2/holdout/examples/sample_14.jpg | c84f016d2b6998f2e3f8b86a616bdf1def21198bc4eb9e4f8dbeb17852d22b06 |
| reports/rtx4060-study-v2/holdout/examples/sample_15.jpg | 1870440d55645e2437db1d6da9b1d99de090f529750891c815ed9a66c9d20f5a |
| reports/rtx4060-study-v2/holdout/examples/sample_16.jpg | 9a7b89f903cc55f90c609f47817be988e876df606e4699976dec4bf9f809e4b9 |
| reports/rtx4060-study-v2/holdout/per_image_errors.json | 05a5b5c22cbe0084050ab2cb5de4244850543f2b1b939ab387e9b9299936b8a7 |
| reports/rtx4060-study-v2/holdout/precision_recall_and_scores.png | 1f25149c95c76f6c9be16a3844a95a38bf4cb40488d54efeea88ec893542c2b5 |
| reports/rtx4060-study-v2/holdout/report.json | c4e2d5c30f18dca31e89cf4ea0f8ee26a28afa551b19e79e4815c894c01efe6c |
| reports/rtx4060-study-v2/initialization-parity.json | 6bbc07b25585b1856033ce86d4de86003f822c61f447296ad1dcf6d6bd5c057b |
| reports/rtx4060-study-v2/preregistration.json | 9ca3f3a7693d339decd0020c35c470b1cfea53b959af3682a6954d5eda057949 |
| reports/rtx4060-study-v2/selection.json | 336f3145c115071871acf610b83394c1cbdb167205ac0f3c6c649d5c982070ce |
| reports/rtx4060-study-v2/validation/precision_recall_and_scores.png | 49e7a087ebc699dce6e7ad7cfcfdb742d7be920fcf8caef374540dd9a7ad78fd |
| reports/rtx4060-study-v2/validation/report.json | b9dbab5c5789510ac9d1484e0e164c09f4d362967cfb4c456d5a39748072a95a |
| reports/rtx4060-study-v2/validation/robustness-control.json | b090209e34b9e4317e315a72083d6f46735e6aa1918498fa0f1f7511d9c8945e |
| reports/rtx4060-study-v2/validation/robustness.json | 8b5ead388ececdfbf0f632dc08c5003e913abbb8ad9a26c69b6d5bf6d44c2864 |
| reports/training-diagnosis.json | 949d689d0838c20c8a033d076fdf729b0e69160c89325b972423f60083226fff |
| reports/validation/annotation-crosscheck.json | 9927a470c58c1929f3e989ae7bed70657c79629133f79c37e5893908f7ac30d7 |
| reports/validation/manifest-portability.json | e7832f289624b0b40e4d0553e67d151b2057198e86657161a3c71a8d5fde55f9 |
| requirements-cpu.lock | 815b43d532424cbbdcf2960102bd3e3353a1963577d277e8767bc3ddc8ba716a |
| requirements-cpu.txt | fb79b27549b42529a58d6d1e55ab1b6aef65c5dcd730b0bcd14bd065bec3e59d |
| requirements-gpu.lock | dc39816f43e18bb918395c2e86ce2e0170b60e2e30d8f284b35c1c630e96df27 |
| requirements-gpu.txt | e5ea6fa26f0deaa45abd9378ef9e9ee4f5da4ddb15f635762542111c9912d7ae |
| scripts/summarize_study.py | 093ffc2a2213bd2146bed1e1cad04a9b447c1e408077915183d5dcc6c37b2f8d |

Generated deterministically without current timestamps or external requests by
`scripts/summarize_study.py`. Regeneration requires completed matching evidence;
missing results never become placeholder success claims.
