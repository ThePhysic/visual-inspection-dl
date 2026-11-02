"""Geometry and real architecture smoke checks on generated synthetic fixtures."""

import json

import pytest
import torch
from PIL import Image

from inspection_dl.model import ARCHITECTURE, build_detector, horizontal_flip, image_tensor, load_checkpoint_model
from inspection_dl.train import DetectionDataset, file_digest, load_manifest, prior_predictions, train_prior


def test_horizontal_flip_reflects_boundary_boxes_and_roundtrips():
    image = torch.arange(3 * 10 * 20).reshape(3, 10, 20)
    boxes = torch.tensor([[0, 0, 7, 10], [15, 3, 20, 8]], dtype=torch.float32)
    flipped_image, flipped_boxes = horizontal_flip(image, boxes)
    assert flipped_boxes.tolist() == [[13, 0, 20, 10], [0, 3, 5, 8]]
    restored_image, restored_boxes = horizontal_flip(flipped_image, flipped_boxes)
    assert torch.equal(restored_image, image)
    assert torch.equal(restored_boxes, boxes)
    assert horizontal_flip(image, torch.empty((0, 4)))[1].shape == (0, 4)


def test_image_tensor_handles_exif_and_rgb_channels():
    image = Image.new("L", (7, 11), color=255)
    exif = image.getexif()
    exif[274] = 6
    image.info["exif"] = exif.tobytes()
    result = image_tensor(image)
    assert result.shape == (3, 7, 11)
    assert result.dtype == torch.float32
    assert result.min() == result.max() == 1


def test_real_model_cpu_inference_save_load_parity(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    model = build_detector(3, pretrained=False, max_detections=100).eval()
    image = torch.rand(3, 71, 113)
    with torch.inference_mode():
        original = model([image])[0]
    checkpoint = tmp_path / "model.pt"
    torch.save({"architecture": ARCHITECTURE, "classes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], "model": model.state_dict()}, checkpoint)
    restored, _ = load_checkpoint_model(checkpoint)
    with torch.inference_mode():
        reloaded = restored([image])[0]
    for key in ("boxes", "labels", "scores"):
        torch.testing.assert_close(original[key], reloaded[key], rtol=0, atol=0)
    boxes = reloaded["boxes"]
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 113).all()
    assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= 71).all()
    assert set(reloaded["labels"].tolist()) <= {1, 2}


def test_manifest_detects_cross_split_leakage_and_invalid_geometry(tmp_path):
    manifest = {"classes": [{"id": 1, "name": "a"}], "records": [{"id": "one", "split": "train", "group": "part01", "width": 10, "height": 10, "boxes": [[0, 0, 10, 10]], "labels": [1]}, {"id": "two", "split": "test", "group": "part01", "width": 10, "height": 10, "boxes": [], "labels": []}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path)
    manifest["records"][1]["group"] = "part02"
    manifest["records"][0]["boxes"] = [[0, 0, 11, 10]]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_manifest(path)


def test_dataset_rejects_path_escape_and_changed_dimensions(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (9, 12)).save(image)
    record = {"id": "fixture", "image": "image.png", "width": 9, "height": 12, "boxes": [], "labels": []}
    dataset = DetectionDataset([record], tmp_path)
    tensor, target, _ = dataset[0]
    assert tensor.shape == (3, 12, 9)
    assert target["boxes"].shape == (0, 4)
    record["width"] = 10
    with pytest.raises(ValueError, match="dimensions"):
        dataset[0]
    record["image"] = "../outside.png"
    with pytest.raises(ValueError, match="escapes"):
        dataset[0]
    assert len(file_digest(image)) == 64


def test_train_only_prior_scales_to_original_frame():
    classes = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    train_records = [{"id": "a", "width": 100, "height": 50, "boxes": [[10, 5, 30, 15]], "labels": [1]}, {"id": "b", "width": 100, "height": 50, "boxes": [], "labels": []}]
    prior = train_prior(train_records, classes)
    result = prior_predictions(prior, [{"id": "test", "width": 200, "height": 100}])["test"]
    assert result == {"boxes": [[20, 10, 60, 30]], "labels": [1], "scores": [.5]}
