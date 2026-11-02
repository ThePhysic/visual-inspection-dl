# Third-party sources and scope of permissions

Project-authored code is MIT licensed. Dataset images/annotations and pretrained
or fine-tuned model weights are separate assets and are not relicensed by the
project MIT license. Raw images, original pretrained weights, and training
checkpoints are excluded from ordinary git history.

## Dataset

PCB-IND is obtained from the authors' immutable
[Zenodo record 19723114](https://zenodo.org/records/19723114), with CC BY 4.0
metadata. See [DATASET.md](DATASET.md) and the saved source metadata for author
attribution, archive checksums, exact class mapping, and preprocessing details.
Retain attribution when sharing source images or annotated derivatives. CI uses
project-generated synthetic fixtures, explicitly separate from scientific data.

## Framework and initialization

- [PyTorch](https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE): BSD-style
  framework license and bundled dependency notices.
- [TorchVision 0.22.1](https://github.com/pytorch/vision/blob/v0.22.1/LICENSE):
  BSD 3-Clause code license, Copyright Soumith Chintala 2016.
- Initialization:
  [SSDLite320 MobileNetV3 Large COCO_V1](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html),
  original URL and full downloaded SHA256 recorded with each training run.
  The defect classifier is newly initialized; the backbone and box regressor
  start from official COCO-pretrained parameters.

TorchVision explicitly states that
[pretrained models may have separate terms derived from training data](https://github.com/pytorch/vision/blob/v0.22.1/README.md#pre-trained-model-license).
The code BSD license alone is not a commercial-use clearance for weights.
COCO's [terms](https://cocodataset.org/#termsofuse) distinguish annotation terms
from the source images' individual licenses. This project uses the model for
noncommercial research and personal portfolio evaluation, retains the original
source attribution, and makes no claim of unrestricted commercial rights or
safety-critical fitness. The source model is downloaded from its official
provider on explicit training invocation, not mirrored in this repository.

The saved bundle licensing metadata must preserve these distinctions and the
dataset attribution. Reproducing experiments acquires the original assets from
their sources; it does not imply additional rights over those assets.
