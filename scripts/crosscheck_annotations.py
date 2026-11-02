"""Independently compare the pinned archive's COCO/YOLO labels with the frozen manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

from PIL import Image

ARCHIVE_SHA256 = "29cf919c64cab7e60e90e16bf3f42cce674688f99295d66a8e10d0066836e3c9"
TOLERANCE_PIXELS = 0.001


def box_error(left: list, right: list) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def match_boxes(left: list[tuple], right: list[tuple]) -> float:
    """Match by category and geometry, independently of source annotation ordering."""
    if len(left) != len(right):
        raise ValueError(f"annotation counts differ: {len(left)} versus {len(right)}")
    remaining = right.copy()
    maximum = 0.0
    for label, box in sorted(left):
        options = [(box_error(box, other_box), index)
                   for index, (other_label, other_box) in enumerate(remaining)
                   if label == other_label]
        if not options:
            raise ValueError(f"no matching category for class {label}")
        distance, index = min(options)
        if distance > TOLERANCE_PIXELS:
            raise ValueError(f"box geometry differs by {distance:.9f} pixels")
        maximum = max(maximum, distance)
        remaining.pop(index)
    return maximum


def crosscheck(archive: Path, manifest_path: Path, output: Path) -> dict:
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive is not the pinned author release")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    content = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest_content_digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_content_digest != manifest["manifest_digest"]:
        raise ValueError("Manifest canonical content digest mismatch")
    errors, checked, excluded = [], [], []
    max_yolo_coco_error = max_yolo_manifest_error = 0.0
    source_class_counts, manifest_class_counts = Counter(), Counter()
    dimension_checks = hash_checks = annotation_hash_checks = negatives = 0
    boundary_records = []
    manifest_index = {row["image"]: row for row in manifest["records"]}
    with zipfile.ZipFile(BytesIO(payload)) as source:
        taxonomy = json.loads(source.read("classes.json"))["classes"]
        expected = [{"id": row["id"] + 1, "name": row["name"]} for row in taxonomy]
        if manifest["classes"] != expected:
            raise ValueError("Archive and manifest taxonomy mappings differ")
        source_images = set()
        for split in ("train", "val", "test"):
            coco = json.loads(source.read(f"COCO/annotations/{split}.json"))
            if coco["categories"] != taxonomy:
                raise ValueError(f"COCO {split} categories differ from classes.json")
            annotations = defaultdict(list)
            for item in coco["annotations"]:
                annotations[item["image_id"]].append(item)
                source_class_counts[item["category_id"] + 1] += 1
            for image in sorted(coco["images"], key=lambda item: item["file_name"]):
                path = "YOLO/" + image["file_name"]
                if path in source_images:
                    raise ValueError("Repeated source image path")
                source_images.add(path)
                row = manifest_index.get(path)
                raw_items = annotations[image["id"]]
                if row is None:
                    label_path = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                    excluded_lines = [line for line in source.read(label_path).decode("utf-8-sig").splitlines()
                                      if line.strip()]
                    invalid_rows = []
                    for index, line in enumerate(excluded_lines, 1):
                        fields = line.split()
                        if len(fields) != 5:
                            invalid_rows.append({"line": index, "reason": "incorrect field count"})
                            continue
                        values = list(map(float, fields))
                        if not all(math.isfinite(value) for value in values):
                            invalid_rows.append({"line": index, "reason": "non-finite value"})
                        elif values[3] <= 0 or values[4] <= 0:
                            invalid_rows.append({"line": index, "reason": "non-positive box extent"})
                    excluded.append({"source_path": path, "source_annotation_count": len(raw_items),
                                     "original_yolo_annotation_count": len(excluded_lines),
                                     "invalid_yolo_rows": invalid_rows,
                                     "nonpositive_coco_boxes": sum(item["bbox"][2] <= 0 or item["bbox"][3] <= 0
                                                                   for item in raw_items)})
                    continue
                try:
                    width, height = image["width"], image["height"]
                    if (width, height) != (row["width"], row["height"]):
                        raise ValueError("COCO and manifest image dimensions differ")
                    annotation_path = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                    raw_yolo = source.read(annotation_path)
                    if hashlib.sha256(raw_yolo).hexdigest() != row["annotation_sha256"]:
                        raise ValueError("Source annotation bytes differ from manifest hash")
                    annotation_hash_checks += 1
                    # Independent conversion: scale center and extent separately, then form corners.
                    yolo_boxes = []
                    for line in raw_yolo.decode("utf-8-sig").splitlines():
                        if not line.strip():
                            continue
                        fields = line.split()
                        if len(fields) != 5:
                            raise ValueError("Malformed original YOLO annotation")
                        category = int(fields[0]) + 1
                        cx, cy, bw, bh = map(float, fields[1:])
                        center_x, center_y, extent_x, extent_y = cx * width, cy * height, bw * width, bh * height
                        box = [center_x - extent_x / 2, center_y - extent_y / 2,
                               center_x + extent_x / 2, center_y + extent_y / 2]
                        if not all(math.isfinite(value) for value in box) or extent_x <= 0 or extent_y <= 0:
                            raise ValueError("Invalid original YOLO box")
                        yolo_boxes.append((category, box))
                    coco_boxes = []
                    for item in raw_items:
                        x, y, bw, bh = item["bbox"]
                        if item.get("iscrowd", 0):
                            raise ValueError("Crowd annotations require separate matching semantics")
                        coco_boxes.append((item["category_id"] + 1, [x, y, x + bw, y + bh]))
                    manifest_boxes = list(zip(row["labels"], row["boxes"]))
                    max_yolo_coco_error = max(max_yolo_coco_error, match_boxes(yolo_boxes, coco_boxes))
                    max_yolo_manifest_error = max(max_yolo_manifest_error, match_boxes(yolo_boxes, manifest_boxes))
                    raw_image = source.read(path)
                    if hashlib.sha256(raw_image).hexdigest() != row["sha256"]:
                        raise ValueError("Source image bytes differ from manifest hash")
                    hash_checks += 1
                    with Image.open(BytesIO(raw_image)) as decoded:
                        decoded.load()
                        if decoded.size != (width, height) or decoded.getexif().get(274, 1) != 1:
                            raise ValueError("Decoded image dimensions or EXIF orientation disagree")
                    dimension_checks += 1
                    manifest_class_counts.update(row["labels"])
                    negatives += not row["labels"]
                    margins = [min(box[0], box[1], width - box[2], height - box[3]) for _, box in manifest_boxes]
                    summary = {"source_path": path, "model_labels": row["labels"], "boxes": row["boxes"],
                               "new_split": row["split"], "original_split": split,
                               "minimum_boundary_margin_pixels": min(margins) if margins else None}
                    checked.append(summary)
                    if any(margin <= TOLERANCE_PIXELS for margin in margins):
                        boundary_records.append(summary)
                except (ValueError, KeyError, OSError) as error:
                    errors.append({"source_path": path, "error": str(error)})
        unknown = sorted(set(manifest_index) - source_images)
        if unknown:
            errors.append({"unknown_manifest_paths": unknown})
    sample_by_class = {}
    for entry in manifest["classes"]:
        sample_by_class[entry["name"]] = sorted(
            [row for row in checked if entry["id"] in row["model_labels"]],
            key=lambda row: row["source_path"],
        )[:3]
    report = {
        "status": "pass" if not errors else "fail",
        "archive_sha256": digest,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_content_digest": manifest_content_digest,
        "source": "https://zenodo.org/records/19723114",
        "scope": "All accepted manifest records, independently compared against the author's COCO and YOLO copies",
        "coordinate_semantics": "YOLO normalized cxcywh; COCO pixel xywh; manifest pixel xyxy with exclusive upper edges; source IDs0..7 map to model IDs1..8",
        "taxonomy": manifest["classes"],
        "absolute_tolerance_pixels": TOLERANCE_PIXELS,
        "source_images": len(source_images),
        "accepted_manifest_images": len(manifest_index),
        "successfully_compared_images": len(checked),
        "source_image_hash_checks": hash_checks,
        "source_annotation_hash_checks": annotation_hash_checks,
        "decoded_dimensions_and_identity_exif_checks": dimension_checks,
        "negative_images_compared": negatives,
        "boundary_images_compared": len(boundary_records),
        "maximum_yolo_coco_coordinate_difference_pixels": max_yolo_coco_error,
        "maximum_yolo_manifest_coordinate_difference_pixels": max_yolo_manifest_error,
        "source_class_instance_counts_including_excluded_image": dict(source_class_counts),
        "accepted_class_instance_counts": dict(manifest_class_counts),
        "excluded_source_images": excluded,
        "deterministic_samples_per_class": sample_by_class,
        "deterministic_negative_samples": sorted([row for row in checked if not row["model_labels"]],
                                                  key=lambda row: row["source_path"])[:5],
        "deterministic_boundary_samples": sorted(boundary_records, key=lambda row: row["source_path"])[:10],
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/PCB-IND_v4.zip"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pcb-ind-v4.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/validation/annotation-crosscheck.json"))
    args = parser.parse_args()
    report = crosscheck(args.archive, args.manifest, args.output)
    print(json.dumps({key: report[key] for key in ("status", "successfully_compared_images", "negative_images_compared",
                                                  "boundary_images_compared", "maximum_yolo_coco_coordinate_difference_pixels",
                                                  "maximum_yolo_manifest_coordinate_difference_pixels", "errors")}, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)
