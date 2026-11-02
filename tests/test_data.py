from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from inspection_dl.data import (
    _fingerprint,
    _write_immutable,
    duplicate_groups,
    parse_yolo,
    prepare_dataset,
    split_groups,
    validate_manifest,
)


def test_yolo_mapping_and_boundaries():
    boxes, labels = parse_yolo("0 0.5 0.5 1 1\n7 0.75 0.25 0.5 0.5", 200, 100)
    assert boxes == [[0, 0, 200, 100], [100, 0, 200, 50]]
    assert labels == [1, 8]
    assert parse_yolo("", 200, 100) == ([], [])


def test_attributed_real_data_fixture_decodes_and_preserves_annotation_geometry():
    root = Path(__file__).parent / "fixtures" / "pcb-ind"
    fixture = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    image_bytes = (root / fixture["image"]).read_bytes()
    annotation_bytes = (root / fixture["annotation"]).read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == fixture["sha256"]
    assert hashlib.sha256(annotation_bytes).hexdigest() == fixture["annotation_sha256"]
    with Image.open(root / fixture["image"]) as image:
        image.load()
        assert image.size == (fixture["width"], fixture["height"])
        assert image.getexif().get(274, 1) == 1
    boxes, labels = parse_yolo(
        annotation_bytes.decode("utf-8-sig"), fixture["width"], fixture["height"]
    )
    assert boxes == fixture["boxes"]
    assert labels == fixture["labels"]


@pytest.mark.parametrize(
    "annotation",
    [
        "0 0.5 0.5 0 0.2",
        "8 0.5 0.5 0.2 0.2",
        "0.2 0.5 0.5 0.2 0.2",
        "0 nan 0.5 0.2 0.2",
        "0 0.5 inf 0.2 0.2",
        "0 0.01 0.5 0.2 0.2",
        "0 0.5 0.5 0.2",
        "cat 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2",
    ],
)
def test_malformed_annotation_rejected(annotation):
    with pytest.raises(ValueError):
        parse_yolo(annotation, 300, 300)


def manifest_fixture():
    return {
        "schema_version": "1",
        "classes": [{"id": 1, "name": "defect"}],
        "records": [
            {
                "id": "a",
                "image": "images/train/a.png",
                "width": 100,
                "height": 100,
                "sha256": "a" * 64,
                "group": "a",
                "split": "train",
                "boxes": [[0, 0, 100, 100]],
                "labels": [1],
            }
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "../outside.png"),
        ("image", "C:\\outside.png"),
        ("image", "/absolute.png"),
        ("boxes", [[0, 0, 101, 100]]),
        ("boxes", [[30, 0, 20, 100]]),
        ("boxes", [[0, 0, float("nan"), 100]]),
        ("labels", [2]),
        ("split", "unassigned"),
    ],
)
def test_manifest_validation_rejects_corrupt_records(field, value):
    manifest = manifest_fixture()
    manifest["records"][0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_cross_split_group_and_hash_leakage():
    for shared_field in ("group", "sha256"):
        manifest = manifest_fixture()
        other = deepcopy(manifest["records"][0])
        other.update(id="b", group="b", sha256="b" * 64, split="test")
        other[shared_field] = manifest["records"][0][shared_field]
        manifest["records"].append(other)
        with pytest.raises(ValueError, match="leakage"):
            validate_manifest(manifest)


def test_manifest_digest_detects_modified_annotations():
    manifest = manifest_fixture()
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    validate_manifest(manifest)
    manifest["records"][0]["boxes"][0][2] = 90
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_manifest(manifest)


def test_filename_prefix_proxy_connects_source_patches():
    records = [
        {"id": "a", "source_group": "prefix-1"},
        {"id": "b", "source_group": "prefix-1"},
        {"id": "c", "source_group": "prefix-2"},
    ]
    images = [Image.new("RGB", (32, 32), color) for color in ("red", "green", "blue")]
    groups, audit = duplicate_groups(records, [_fingerprint(image) for image in images])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert audit["source_prefix_group_count"] == 2


def test_duplicate_groups_use_decoded_pixels_and_near_brightness():
    pattern = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    images = [
        Image.fromarray(pattern),
        Image.fromarray(pattern.copy()),
        Image.fromarray(pattern + 3),
        Image.fromarray(255 - pattern),
    ]
    groups, audit = duplicate_groups(
        [{"id": str(i)} for i in range(4)], [_fingerprint(image) for image in images]
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[0] != groups[3]
    assert len(audit["exact_duplicate_pairs"]) == 1
    assert audit["near_duplicate_pairs"]


def test_group_splits_are_deterministic_and_stratified():
    records = [{"group": f"g{i // 2}", "labels": [1 + (i // 2) % 2]} for i in range(200)]
    split = split_groups(records, 42)
    assert split == split_groups(records, 42)
    assert split != split_groups(records, 43)
    counts = {
        name: sum(split[row["group"]] == name for row in records)
        for name in ("train", "val", "test")
    }
    assert counts == {"train": 140, "val": 30, "test": 30}
    for label in (1, 2):
        assert {split[row["group"]] for row in records if label in row["labels"]} == {
            "train",
            "val",
            "test",
        }


def test_preparation_reports_exclusions_and_freezes_outputs(tmp_path):
    root = tmp_path / "raw"
    images, labels = root / "images" / "train", root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 30), "white").save(images / "good.png")
    Image.new("RGB", (20, 30), "black").save(images / "missing.png")
    (labels / "good.txt").write_text("0 0.5 0.5 0.2 0.4", encoding="utf-8")
    (labels / "orphan.txt").write_text("", encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = prepare_dataset(root, output)
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["boxes"] == [[8, 9, 12, 21]]
    assert record["sha256"] == hashlib.sha256((images / "good.png").read_bytes()).hexdigest()
    audit = json.loads(output.with_name("manifest.audit.json").read_text())
    assert audit["input_images"] == 2
    assert audit["exclusions"][0]["reason"] == "Missing YOLO annotation"
    assert audit["orphan_labels"] == ["labels/train/orphan.txt"]
    assert prepare_dataset(root, output) == result
    with pytest.raises(ValueError, match="frozen"):
        _write_immutable(output, {"different": True})
