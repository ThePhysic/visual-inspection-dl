"""Geometry and real architecture smoke checks on generated synthetic fixtures."""

import json
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.ops import box_iou as torchvision_box_iou

from inspection_dl.metrics import box_iou
from inspection_dl.model import (
    ARCHITECTURE,
    build_detector,
    horizontal_flip,
    image_tensor,
    load_checkpoint_model,
)
from inspection_dl.train import (
    DetectionDataset,
    file_digest,
    load_manifest,
    prior_predictions,
    train_prior,
)


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
    model = build_detector(3, pretrained=False, max_detections=100)
    image = torch.rand(3, 71, 113)
    # A real forward/backward/update on explicitly synthetic data verifies that
    # the detector is trainable, while CI never downloads data or pretrained weights.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    output_bias = model.head.classification_head.module_list[0][-1].bias
    initial_parameter = output_bias.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = model(
        [image], [{"boxes": torch.tensor([[4.0, 6.0, 60.0, 55.0]]), "labels": torch.tensor([1])}]
    )
    total = sum(losses.values())
    assert torch.isfinite(total) and total.item() > 0
    total.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.head.classification_head.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.head.regression_head.parameters()
    )
    optimizer.step()
    assert not torch.equal(initial_parameter, output_bias)
    model.eval()
    with torch.inference_mode():
        original = model([image])[0]
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "classes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "model": model.state_dict(),
        },
        checkpoint,
    )
    restored, _ = load_checkpoint_model(checkpoint)
    with torch.inference_mode():
        reloaded = restored([image])[0]
    for key in ("boxes", "labels", "scores"):
        torch.testing.assert_close(original[key], reloaded[key], rtol=0, atol=0)
    boxes = reloaded["boxes"]
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 113).all()
    assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= 71).all()
    assert set(reloaded["labels"].tolist()) <= {1, 2}


def test_numpy_iou_matches_independent_torchvision_reference():
    generator = torch.Generator().manual_seed(71)
    starts = torch.rand(20, 2, generator=generator, dtype=torch.float64) * 50
    ends = starts + torch.rand(20, 2, generator=generator, dtype=torch.float64) * 30 + 1
    boxes = torch.cat([starts, ends], dim=1)
    reference = torchvision_box_iou(boxes, boxes)
    actual = torch.tensor(
        [box_iou(box.tolist(), boxes.tolist()).tolist() for box in boxes], dtype=torch.float64
    )
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


def test_manifest_detects_cross_split_leakage_and_invalid_geometry(tmp_path):
    manifest = {
        "classes": [{"id": 1, "name": "a"}],
        "records": [
            {
                "id": "one",
                "split": "train",
                "group": "part01",
                "width": 10,
                "height": 10,
                "boxes": [[0, 0, 10, 10]],
                "labels": [1],
            },
            {
                "id": "two",
                "split": "test",
                "group": "part01",
                "width": 10,
                "height": 10,
                "boxes": [],
                "labels": [],
            },
        ],
    }
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
    record = {
        "id": "fixture",
        "image": "image.png",
        "width": 9,
        "height": 12,
        "boxes": [],
        "labels": [],
    }
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
    train_records = [
        {"id": "a", "width": 100, "height": 50, "boxes": [[10, 5, 30, 15]], "labels": [1]},
        {"id": "b", "width": 100, "height": 50, "boxes": [], "labels": []},
    ]
    prior = train_prior(train_records, classes)
    result = prior_predictions(prior, [{"id": "test", "width": 200, "height": 100}])["test"]
    assert result == {"boxes": [[20, 10, 60, 30]], "labels": [1], "scores": [0.5]}


def test_interrupted_epoch_resume_matches_uninterrupted_optimizer_and_sampling(
    tmp_path, monkeypatch
):
    """Resume a real two-epoch CPU detector after an interrupted second epoch."""
    import inspection_dl.train as training

    torch.set_num_threads(2)
    records = []
    rng = np.random.default_rng(17)
    for index in range(4):
        path = tmp_path / f"fixture_{index}.png"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(path)
        records.append(
            {
                "id": str(index),
                "image": path.name,
                "width": 64,
                "height": 48,
                "sha256": file_digest(path),
                "group": str(index),
                "split": "train" if index < 3 else "val",
                "boxes": [[4, 5, 31, 38]],
                "labels": [1],
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "synthetic-ci-only",
                "taxonomy_version": "1",
                "classes": [{"id": 1, "name": "fixture"}],
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "manifest": str(manifest),
        "data_root": str(tmp_path),
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "seed": 41,
        "threads": 2,
        "device": "cpu",
        "horizontal_flip": True,
        "freeze_backbone": True,
        "amp": False,
        "no_pretrained": True,
        "resume": None,
    }
    sampled = []
    original_getitem = training.DetectionDataset.__getitem__

    def track_getitem(self, index):
        sample = original_getitem(self, index)
        if self.flip_probability:
            # Include a pixel so augmentation RNG replay is checked too.
            sampled.append((sample[2], sample[0][0, 0, 0].item()))
        return sample

    monkeypatch.setattr(training.DetectionDataset, "__getitem__", track_getitem)
    full = tmp_path / "full"
    training.train(Namespace(**config, output=str(full)))
    full_sampling = sampled.copy()
    sampled.clear()
    interrupted = tmp_path / "interrupted"
    original_predict = training.predict_records
    calls = 0

    def interrupt_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption after training second epoch")
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(training, "predict_records", interrupt_second_validation)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training.train(Namespace(**config, output=str(interrupted)))
    saved = torch.load(interrupted / "last.pt", weights_only=True)
    assert saved["epoch"] == 0 and (interrupted / "best.pt").exists()
    monkeypatch.setattr(training, "predict_records", original_predict)
    sampled.clear()
    resumed_config = {**config, "resume": str(interrupted / "last.pt"), "output": str(interrupted)}
    training.train(Namespace(**resumed_config))
    assert sampled == full_sampling[3:]
    expected = torch.load(full / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert expected["epoch"] == actual["epoch"] == 1
    for key, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["optimizer"]["param_groups"] == actual["optimizer"]["param_groups"]
    for key, state in expected["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, actual["optimizer"]["state"][key][name], rtol=0, atol=0
            )


def test_nonfinite_training_failure_preserves_batch_audit(tmp_path, monkeypatch):
    import inspection_dl.train as training

    class NonfiniteFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, images, targets):
            return {"classification": self.weight * float("nan")}

    monkeypatch.setattr(training, "build_detector", lambda *args, **kwargs: NonfiniteFixture())
    records = []
    for index, split in enumerate(("train", "val")):
        image = tmp_path / f"fixture_{index}.png"
        Image.new("RGB", (24, 32), color=(index, 0, 0)).save(image)
        records.append(
            {
                "id": str(index),
                "image": image.name,
                "width": 24,
                "height": 32,
                "sha256": file_digest(image),
                "group": str(index),
                "split": split,
                "boxes": [[1, 1, 10, 10]],
                "labels": [1],
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "synthetic-failure-only",
                "taxonomy_version": "1",
                "classes": [{"id": 1, "name": "fixture"}],
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "failed"
    config = Namespace(
        manifest=str(manifest),
        data_root=str(tmp_path),
        output=str(output),
        epochs=1,
        batch_size=1,
        lr=0.0003,
        seed=42,
        threads=2,
        device="cpu",
        horizontal_flip=False,
        freeze_backbone=False,
        amp=False,
        no_pretrained=True,
        resume=None,
    )
    with pytest.raises(RuntimeError, match="nonfinite training loss"):
        training.train(config)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["kind"] == "nonfinite_forward_loss"
    assert failure["epoch"] == failure["batch_index"] == 0
    assert failure["sample_ids"] == ["0"]
    assert failure["losses"] == {"classification": "nan"}
    assert failure["nonfinite_parameter_names"] == []
    assert failure["last_finite_batch"] is None
    assert not (output / "best.pt").exists()
