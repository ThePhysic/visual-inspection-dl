# PCB-IND SSDLite v1

Standalone PyTorch detection of eight PCB defect classes, with a versioned model
bundle, CPU inference through FastAPI, audited data, and reproducible evaluation.

The existing in-repository bundle is `releases/pcb-ind-ssdlite-v1`.
Artifact digest: `9037234654bcc190f6f5dd8cda07e70a119ba19b139b2b99118f4d9f27d7e05e`.

On the frozen 722-image holdout, the model achieved AP50 **44.17%** and mean AP
across IoU 0.50–0.95 **27.19%**. At the validation-selected score threshold
**0.20**, micro precision was **53.31%** and recall **56.76%** at IoU 0.50.
The train-derived median-box class-prior reference is a geometric sanity
baseline, not a competitive detection method.

The model sees 300×300 candidate ROI patches, not full-board search. Evidence
uses a single training seed and heuristic source-group proxies; confidence is
not calibrated. The rare `open` class detected only 1 of 18 test instances.
There is no manufacturing-readiness or production-domain claim.

Warmed forward-pipeline p50 latency on the recorded laptop was 54.38 ms on CPU
and 36.79 ms on GPU. Those timings exclude end-to-end HTTP request overhead.

The README provides real held-out successes and failures, local CPU serving
instructions, and links to the model card, error analysis and raw evidence.
The attributed training image in the API smoke test proves serving behavior;
it is not evidence of held-out generalization. No duplicate binary asset is
needed because the exact immutable bundle is already committed.

Project code is MIT licensed; PCB-IND images and annotations are CC BY 4.0.
Pretrained-model and COCO terms remain separate. This research/portfolio
snapshot does not assert unrestricted commercial rights over weights or
safety-critical fitness. See `docs/THIRD_PARTY.md` before reuse.
