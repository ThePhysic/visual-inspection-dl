# Portfolio polish delivery

This presentation and reliability pass starts at
`a34facf859f0b8ff16b832a9c6f349269b94bcc7`. It adds no training, model selection,
holdout evaluation, architecture changes or platform integration.

## Changes and commits

| Commit | Files and result |
| --- | --- |
| `5d9d018e9c57fba34a606ce5fb9817a450d41bb6` | [Release smoke test](../tests/test_released_model_smoke.py) and [CI workflow](../.github/workflows/ci.yml): run the checked-in trained detector through CPU ASGI inference on both operating systems. |
| `9642a317afa9b11c3e31400b098978980c0e73d5` | [README](../README.md): task/result summary, local serving commands, three real holdout comparisons, limitations, geometric sanity baseline and CI badge. [Results index](RESULTS_INDEX.md): readable evidence before large JSON. [Acceptance](ACCEPTANCE.md): external scope/evidence map. Move root `IDEA_PACKAGE.md` to the clearly superseded [historical brief](HISTORICAL_IDEA_PACKAGE.md). Prepare [remote follow-up](REMOTE_FOLLOWUP.md) and [Release notes](RELEASE_NOTES_DRAFT.md). |

This report is the only additional documentation file. The changed-file list
above is complete; no image assets were added or altered.

## Validation

- Pinned Python 3.12 CPU environment: **214 tests passed** in 52.70 seconds,
  with 15 dependency deprecation warnings. Torch 2.7.1+cpu and torchvision
  0.22.1+cpu; no CUDA runtime. Ruff check, Ruff format check and Git whitespace
  checks passed.
- The new actual-release smoke also passed independently. It checks readiness,
  model identity and prediction metadata, nonempty valid detections, in-frame
  boxes, scores and timings without fixing floating-point outputs or counts.
  CUDA, network and model-download guards are active during model startup and
  serving; the only permitted `torch.load` input is the committed CPU weights.
  It runs from an empty working directory using the attributed training fixture.
  This checks serving, not holdout quality.
- The Windows/Linux [CI workflow](../.github/workflows/ci.yml) runs the smoke
  explicitly and then the remaining suite. See [Actions](https://github.com/ThePhysic/visual-inspection-dl/actions/workflows/ci.yml)
  for commit-specific hosted results; the local test count above is not a claim
  about a hosted run.
- README rendered through GitHub's Markdown API and inspected at a normal
  1024-pixel browser width with GitHub-style CSS. All three original 600×332
  comparisons were inspected visually. Captions were checked against selection,
  prediction and per-image-error records. The gallery identifies the test split,
  threshold 0.20, green ground truth/left and red predictions/right, as well as
  actual successes and failures. All 161 relative document/image links and
  heading anchors across 16 Markdown documents passed validation.

## Frozen evidence and remaining limits

The targeted diff from the baseline is empty for `src/`, `releases/`,
`data/manifests/` and `reports/`. The three release files retain these SHA256s:

| File in `releases/pcb-ind-ssdlite-v1/` | SHA256 |
| --- | --- |
| `weights.pt` | `28271a688aa7335521bb7725423fd6ea6622f3d1964bfaed83dd97b4f0ff34d9` |
| `metadata.json` | `d0c4ed35b18363a05e1cf01452365e3d1ad85b65fc967266f17de6563d0d3452` |
| `evaluation.json` | `acc89d13a50caf389f74bc2cb200b0558fcbe6f6ead6e7f94f6e5ab5817b2dfd` |

Artifact digest remains
`9037234654bcc190f6f5dd8cda07e70a119ba19b139b2b99118f4d9f27d7e05e`.
All 54 recorded input hashes and three generated-document hashes match the
unchanged checksum inventory. The model card, error analysis and reproducibility
report are unchanged. No new data or weights were downloaded for this pass.
Existing dataset attribution and separate code/pretrained terms remain.

The [model card](MODEL_CARD.md) still describes one seed, heuristic source
groups, uncalibrated scores, weak rare-class evidence and 300×300 candidate ROIs.
Forward-pipeline timings do not measure HTTP latency. No full-board search or
manufacturing-readiness claim follows from the smoke test or gallery.

## Separate remote-only suggestions

Repository visibility, About text and topics were preserved. No tag or GitHub
Release was created. Suggested topics, the `pcb-ind-ssdlite-v1` annotated tag,
Release notes and command/UI instructions are prepared in
[REMOTE_FOLLOWUP.md](REMOTE_FOLLOWUP.md). Those actions require separate approval;
the normal source commits and CI push do not apply them.
