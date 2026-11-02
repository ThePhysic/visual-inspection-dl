# PCB-IND dataset and split protocol

The project uses **PCB-IND**, collected from a real automated optical inspection (AOI)
production line by **Hao Yan, Fujian Polytechnic Normal University**. The pinned author
release is [Zenodo record 19723114](https://zenodo.org/records/19723114), archive
`PCB-IND_v4.zip` (101,336,845 bytes; published MD5
`1325f8dffeb73bef4db34fbd22b31ec5`). The author's record explicitly licenses the images
and annotations under **CC BY 4.0**. Attribute the author, dataset title, release and
record when reusing sample images or annotations. This repository's source-code
license does not replace the dataset license.

The release describes 4,789 RGB ROI patches, each 300×300 pixels, and eight defect
classes. These are AOI candidate patches, not representative full-board images or
unfiltered production streams. Detection precision here cannot establish production
false-alarm rates or justify autonomous manufacturing decisions.

The frozen audit accepts **4,788** images: **3,352 train, 714 validation, 722 test**.
It excludes `YOLO/images/val/5799_t_038.jpg` because its fifth annotation has a
non-positive box extent. There are 127 supplied empty-label negative patches,
no exact decoded-image duplicates, 63 near-duplicate pairs, and 685 filename-prefix
proxies merged into 664 final components (largest: 267 images). All eight classes
appear in all splits. The rarest class, `open`, has only 78/17/17 train/validation/test
images (84/30/18 instances), which limits per-class statistical precision.

An independent [source-format cross-check](../reports/validation/annotation-crosscheck.json)
compares all 4,788 accepted records against both original COCO and YOLO annotations,
including 127 negatives and 906 images with boxes touching an image boundary.
All class mappings, image/label hashes, decoded dimensions and EXIF orientations
agree. The largest coordinate difference is 0.00015 pixels due to decimal boundary
serialization and clipping; the independent comparison tolerance is 0.001 pixels.
The excluded image has four valid COCO boxes and five YOLO rows: COCO already drops
the degenerate fifth YOLO box. We conservatively exclude the entire image rather
than silently correcting this source-format discrepancy. Reproduce the audit with:

```text
python scripts/crosscheck_annotations.py
```

The release's documented zero-based YOLO taxonomy maps to detector labels as follows:

| Original ID | Model ID | Name |
| --- | --- | --- |
| 0 | 1 | mouse_bite |
| 1 | 2 | missing_copper |
| 2 | 3 | scratch |
| 3 | 4 | spurious_copper |
| 4 | 5 | copper_burr |
| 5 | 6 | stain |
| 6 | 7 | short |
| 7 | 8 | open |

Model ID 0 is reserved for background. The taxonomy version is `pcb-ind-v4-1`.
The companion [GitHub README](https://github.com/gnmtdt/PCB-IND) has a different,
older class listing; its class names must not be substituted for the pinned archive.

## Acquisition and preparation

From the repository root, after installing the environment:

```text
python scripts/acquire_data.py --dest data/raw/pcb-ind
python -m inspection_dl.data --root data/raw/pcb-ind --output data/manifests/pcb-ind-v4.json --seed 42
```

Acquisition verifies the published archive size and MD5 before extraction, rejects
path traversal and symlinks, caps uncompressed size, and records a SHA256 digest.
Raw archives and images are ignored by Git. Preparation writes a deterministic
manifest and adjacent `.audit.json`; conflicting contents cannot overwrite frozen
outputs. To explore another split, select another output filename and treat it as
a new experiment rather than silently changing the reported holdout.

Every accepted image records its source path, dimensions, image and annotation
SHA256, class IDs, boxes, source split, filename-prefix proxy, merged group, and split membership.
The manifest digest covers the canonical sorted JSON before adding the digest field.
Pillow fully decodes images. Non-identity EXIF orientation is excluded until its
annotations can be reconciled; image bytes are not silently rotated away from labels.
Images may be converted to RGB, but label files are never model inputs.

YOLO normalized `cx, cy, width, height` coordinates become floating-point
`x_min,y_min,x_max,y_max` pixels in the original image frame. Upper edges are
exclusive; a full-image box is `[0,0,width,height]`. The parser rejects malformed,
non-finite, inverted, zero-area, duplicate and out-of-bounds annotations. A tolerance
of 1e-6 normalized units permits only serialization noise at image boundaries;
these near-boundary values are clamped and pixel values rounded to six decimals.
Missing annotations and decoding failures are explicitly excluded. Supplied empty
annotation files represent negative images; missing files do not.

## Leakage controls and split interpretation

We recombine the author's supplied splits, then assign complete groups to a
new deterministic 70/15/15 train/validation/test split using seed 42 and a greedy
multilabel balance objective. All released filenames match `<numeric>_[t|b]_<index>`;
we conservatively group each shared numeric prefix as a possible common source.
The release does not define the identifier's semantics, so this is an unverified
acquisition proxy, not a claim that a prefix is a confirmed physical-board ID.
Exact decoded-RGB duplicates are grouped even when
their file encodings differ. Near-duplicate candidates must satisfy both a 64-bit
dHash Hamming distance ≤4 and 32×32 grayscale thumbnail mean absolute error ≤6.
Connected components merge prefix and visual-duplicate links, keeping transitive
chains in one split. All classes
contribute to the balance objective, including rare classes. The audit gives actual
split sizes, class image/instance counts, exclusions, duplicate pairs and group sizes.

This heuristic can miss crops, rotations, substantial lighting changes, and different
views of the same board. The release does not supply explicit board, parent-frame,
camera or acquisition-session metadata. Prefix and visual grouping reduce obvious
leakage but cannot certify independence by physical board or acquisition session.
Report this limitation with every generalization claim. No source/session/lighting
slice is invented from absent metadata. Sample-level uncertainty intervals reflect
the observed dataset, not unseen production conditions.

Training and model selection use train/validation only. Freeze the final test split
before experiments, select operating thresholds on validation, and use test only for
the final report. A single attributed, unchanged real training image and YOLO label
are included in `tests/fixtures/pcb-ind/` under CC BY 4.0 to check real-data ingestion
in CI. Synthetic CI fixtures are generated by the other tests. Both are explicitly
separate from quantitative held-out real-data evidence.
