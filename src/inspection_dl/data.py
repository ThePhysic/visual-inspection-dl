"""Audited PCB-IND ingestion and deterministic duplicate-group-aware splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

CLASS_NAMES = (
    "mouse_bite",
    "missing_copper",
    "scratch",
    "spurious_copper",
    "copper_burr",
    "stain",
    "short",
    "open",
)
DATASET_ID = "pcb-ind-zenodo-19723114-v4"


def parse_yolo(text: str, width: int, height: int, num_classes: int = 8) -> tuple[list, list]:
    """Convert zero-based YOLO normalized cxcywh into 1-based, exclusive-edge xyxy pixels."""
    boxes, labels = [], []
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected five YOLO fields")
        try:
            category_float, cx, cy, bw, bh = map(float, parts)
        except ValueError as error:
            raise ValueError(f"line {line_number}: non-numeric label") from error
        if not all(math.isfinite(value) for value in (category_float, cx, cy, bw, bh)):
            raise ValueError(f"line {line_number}: non-finite label")
        if not category_float.is_integer() or not 0 <= category_float < num_classes:
            raise ValueError(f"line {line_number}: unknown class")
        if bw <= 0 or bh <= 0:
            raise ValueError(f"line {line_number}: non-positive box extent")
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        # Decimal YOLO serialization can put an intended boundary ~1e-6 outside the image.
        if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
            raise ValueError(f"line {line_number}: out-of-bounds box")
        box = [
            round(max(0.0, x1) * width, 6),
            round(max(0.0, y1) * height, 6),
            round(min(1.0, x2) * width, 6),
            round(min(1.0, y2) * height, 6),
        ]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f"line {line_number}: invalid box extent after clipping and rounding")
        boxes.append(box)
        labels.append(int(category_float) + 1)
    if len(set(zip(labels, map(tuple, boxes)))) != len(boxes):
        raise ValueError("Duplicate label rows")
    return boxes, labels


def _fingerprint(image: Image.Image) -> tuple[str, int, np.ndarray]:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    gray = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.int16)
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return digest, value, gray


def duplicate_groups(records: list[dict], fingerprints: list[tuple]) -> tuple[list[str], dict]:
    """Connected components prevent transitive duplicate chains crossing split boundaries."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(left)] = find(right)

    exact, near = [], []
    seen = {}
    source_groups = {}
    for index, record in enumerate(records):
        source = record.get("source_group")
        if source is not None:
            if source in source_groups:
                union(index, source_groups[source])
            else:
                source_groups[source] = index
    for index, (digest, dhash, gray) in enumerate(fingerprints):
        if digest in seen:
            previous = seen[digest]
            union(index, previous)
            exact.append([records[previous]["id"], records[index]["id"]])
        else:
            seen[digest] = index
        for previous in range(index):
            other_digest, other_hash, other_gray = fingerprints[previous]
            if digest == other_digest:
                continue
            if (dhash ^ other_hash).bit_count() <= 4:
                difference = float(np.abs(gray - other_gray).mean())
                if difference <= 6.0:
                    union(index, previous)
                    near.append(
                        {
                            "left": records[previous]["id"],
                            "right": records[index]["id"],
                            "dhash_distance": (dhash ^ other_hash).bit_count(),
                            "thumbnail_gray_mae": round(difference, 6),
                        }
                    )
    members = defaultdict(list)
    for index, record in enumerate(records):
        members[find(index)].append(record["id"])
    names = {key: min(values) for key, values in members.items()}
    groups = ["visual-" + names[find(index)] for index in range(len(records))]
    return groups, {
        "exact_duplicate_pairs": exact,
        "near_duplicate_pairs": near,
        "group_count": len(members),
        "largest_group": max(map(len, members.values()), default=0),
        "source_prefix_group_count": len(source_groups),
        "algorithm": "shared numeric filename prefix OR decoded-RGB SHA256 OR (64-bit dHash Hamming<=4 AND 32x32 grayscale MAE<=6), connected components",
    }


def split_groups(records: list[dict], seed: int = 42) -> dict[str, str]:
    """Greedy multilabel stratification, assigning whole groups toward 70/15/15 targets."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    classes = sorted({label for record in records for label in record["labels"]})
    totals = np.array(
        [len(records)] + [sum(label in row["labels"] for row in records) for label in classes],
        dtype=float,
    )
    fractions = np.array([0.70, 0.15, 0.15])
    targets = fractions[:, None] * totals[None, :]
    current = np.zeros_like(targets)
    names = ["train", "val", "test"]
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda key: len(grouped[key]), reverse=True)
    result = {}
    for key in groups:
        rows = grouped[key]
        addition = np.array(
            [len(rows)] + [sum(label in row["labels"] for row in rows) for label in classes],
            dtype=float,
        )
        # Normalized incremental squared target error gives rare classes equal influence.
        scale = np.maximum(targets, 1)
        costs = (((current + addition - targets) ** 2 - (current - targets) ** 2) / scale).sum(
            axis=1
        )
        chosen = int(np.argmin(costs))
        current[chosen] += addition
        result[key] = names[chosen]
    return result


def validate_manifest(manifest: dict) -> None:
    """Validate the training contract and fail on any split, taxonomy or geometry corruption."""
    if manifest.get("schema_version") != "1":
        raise ValueError("Unsupported manifest schema")
    classes = manifest.get("classes", [])
    class_ids = [entry["id"] for entry in classes]
    if (
        not classes
        or class_ids != list(range(1, len(classes) + 1))
        or len({c["name"] for c in classes}) != len(classes)
    ):
        raise ValueError("Taxonomy must have unique names and contiguous 1-based IDs")
    if "manifest_digest" in manifest:
        content = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        digest = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != manifest["manifest_digest"]:
            raise ValueError("Manifest content digest mismatch")
    seen, group_splits, digest_splits = set(), {}, {}
    for row in manifest.get("records", []):
        if row["id"] in seen:
            raise ValueError("Duplicate record ID")
        seen.add(row["id"])
        path = Path(row["image"])
        if (
            path.is_absolute()
            or row["image"].startswith("/")
            or ".." in path.parts
            or "\\" in row["image"]
            or ":" in row["image"]
            or not path.name
        ):
            raise ValueError("Unsafe image path")
        width, height = row["width"], row["height"]
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid split")
        for mapping, key in ((group_splits, row["group"]), (digest_splits, row["sha256"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Cross-split duplicate/group leakage")
            mapping[key] = row["split"]
        if len(row["boxes"]) != len(row["labels"]):
            raise ValueError("Box/label count mismatch")
        for box, label in zip(row["boxes"], row["labels"]):
            if label not in class_ids or len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError("Invalid label or box")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Invalid box geometry")


def _write_immutable(path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"Refusing to overwrite frozen output: {path}; choose a new output name")
    path.write_text(payload, encoding="utf-8")


def prepare_dataset(root: Path, output: Path, seed: int = 42) -> dict:
    """Build a frozen manifest from the author's v4 YOLO image/label release."""
    root = root.resolve()
    taxonomy_path = root / "classes.json"
    if taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
        expected = [{"id": index, "name": name} for index, name in enumerate(CLASS_NAMES)]
        if taxonomy.get("version") != "v4" or taxonomy.get("classes") != expected:
            raise ValueError("Source taxonomy differs from the pinned PCB-IND v4 release")
    image_root = root / "YOLO" if (root / "YOLO").is_dir() else root
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and "images" in path.relative_to(root).parts
    )
    if not image_paths:
        raise ValueError("No images found under an images directory; acquire PCB-IND_v4 first")
    records, fingerprints, exclusions = [], [], []
    all_labels = {path.resolve() for path in image_root.rglob("*.txt") if "labels" in path.parts}
    used_labels = set()
    for path in image_paths:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        parts[parts.index("images")] = "labels"
        label_path = root.joinpath(*parts).with_suffix(".txt")
        try:
            if not label_path.is_file():
                raise ValueError("Missing YOLO annotation")
            used_labels.add(label_path.resolve())
            with Image.open(path) as image:
                image.load()
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(
                        "Non-identity EXIF orientation requires annotation reconciliation"
                    )
                width, height = image.size
                fingerprint = _fingerprint(image)
            boxes, labels = parse_yolo(label_path.read_text(encoding="utf-8-sig"), width, height)
            relative_str = relative.as_posix()
            prefix_match = re.fullmatch(r"([0-9]+)_[tb]_[0-9]+", path.stem)
            records.append(
                {
                    "id": hashlib.sha256(relative_str.encode()).hexdigest()[:20],
                    "source_id": relative_str,
                    "image": relative_str,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "annotation_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                    "source_split": relative.parts[relative.parts.index("images") + 1],
                    "source_group": "filename-prefix-" + prefix_match[1] if prefix_match else None,
                    "boxes": boxes,
                    "labels": labels,
                }
            )
            fingerprints.append(fingerprint)
        except (ValueError, OSError) as error:
            exclusions.append({"image": relative.as_posix(), "reason": str(error)})
    if not records:
        raise ValueError("No valid records after audit")
    groups, duplicate_audit = duplicate_groups(records, fingerprints)
    for row, group in zip(records, groups):
        row["group"] = group
    assignments = split_groups(records, seed)
    for row in records:
        row["split"] = assignments[row["group"]]
    manifest = {
        "schema_version": "1",
        "dataset_id": DATASET_ID,
        "taxonomy_version": "pcb-ind-v4-1",
        "classes": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
        "coordinate_system": "xyxy_pixels_original_image",
        "seed": seed,
        "source": "https://zenodo.org/records/19723114",
        "license": "CC-BY-4.0",
        "split_policy": "70/15/15 greedy multilabel whole filename-prefix/visual-duplicate components; author splits recombined",
        "records": records,
    }
    validate_manifest(manifest)
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_digest"] = manifest_digest
    audit = {
        "dataset_id": DATASET_ID,
        "manifest_digest": manifest_digest,
        "input_images": len(image_paths),
        "accepted_images": len(records),
        "exclusions": exclusions,
        "orphan_labels": sorted(
            path.relative_to(root).as_posix() for path in all_labels - used_labels
        ),
        "duplicate_audit": duplicate_audit,
        "acquisition_group_metadata": "All matching <numeric>_[tb]_<index> filenames sharing numeric prefix are conservatively grouped; prefix semantics unconfirmed. No explicit board/camera/session metadata, so these proxies cannot certify unseen-board generalization",
        "split_counts": dict(Counter(row["split"] for row in records)),
        "class_instance_counts": {
            name: {
                split: sum(
                    row["labels"].count(index + 1) for row in records if row["split"] == split
                )
                for split in ("train", "val", "test")
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "class_image_counts": {
            name: {
                split: sum(index + 1 in row["labels"] for row in records if row["split"] == split)
                for split in ("train", "val", "test")
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "empty_images": sum(not row["labels"] for row in records),
        "dimension_counts": dict(Counter(f"{row['width']}x{row['height']}" for row in records)),
        "near_duplicate_detection_limits": "dHash plus thumbnail MAE is conservative and may miss crops, rotations, severe illumination changes and different views of the same board",
    }
    _write_immutable(output, manifest)
    _write_immutable(output.with_name(output.stem + ".audit.json"), audit)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/pcb-ind"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/pcb-ind-v4.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_dataset(args.root, args.output, args.seed)
    print(
        json.dumps(
            {"manifest_digest": result["manifest_digest"], "images": len(result["records"])},
            indent=2,
        )
    )
