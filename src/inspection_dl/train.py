"""Bounded, manifest-driven transfer learning: python -m inspection_dl.train."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .metrics import evaluate_detections
from .model import ARCHITECTURE, PREPROCESSING, PRETRAINED_URL, build_detector, horizontal_flip, image_tensor


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_image_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("manifest image path escapes data root")
    return path


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    classes = manifest["classes"]
    if [item["id"] for item in classes] != list(range(1, len(classes) + 1)):
        raise ValueError("class IDs must be contiguous from one, with background implicit")
    if not classes or len({item["name"] for item in classes}) != len(classes):
        raise ValueError("class names must be unique and nonempty")
    records = manifest["records"]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("duplicate record IDs")
    groups, checksums = {}, {}
    for record in records:
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError("invalid split")
        for key, lookup in (("group", groups), ("sha256", checksums)):
            if key in record:
                previous = lookup.setdefault(record[key], record["split"])
                if previous != record["split"]:
                    raise ValueError(f"cross-split {key} leakage")
        for box, label in zip(record["boxes"], record["labels"], strict=True):
            if label not in range(1, len(classes) + 1):
                raise ValueError("unknown target class")
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError("nonfinite target box")
            if not (0 <= box[0] < box[2] <= record["width"] and 0 <= box[1] < box[3] <= record["height"]):
                raise ValueError("target box outside original image frame")
    return manifest


class DetectionDataset(Dataset):
    def __init__(self, records: list[dict], root: str | Path, horizontal_flip_probability: float = 0.0):
        self.records = records
        self.root = Path(root)
        self.flip_probability = horizontal_flip_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(safe_image_path(self.root, record["image"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"image dimensions changed for {record['id']}")
            tensor = image_tensor(image)
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(record["labels"], dtype=torch.int64)
        if random.random() < self.flip_probability:
            tensor, boxes = horizontal_flip(tensor, boxes)
        return tensor, {"boxes": boxes, "labels": labels}, record["id"]


def collate(batch):
    return tuple(map(list, zip(*batch, strict=True)))


@torch.inference_mode()
def predict_records(model, records: list[dict], root: str | Path, device: str, batch_size: int = 8) -> dict:
    model.eval()
    loader = DataLoader(DetectionDataset(records, root), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    predictions = {}
    for images, _, ids in loader:
        outputs = model([image.to(device) for image in images])
        for image_id, output in zip(ids, outputs, strict=True):
            # Zero-area boxes can arise after clipping; reject rather than report them.
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            valid = torch.isfinite(boxes).all(1) & torch.isfinite(scores)
            valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            predictions[image_id] = {"boxes": boxes[valid].tolist(), "labels": labels[valid].tolist(), "scores": scores[valid].tolist()}
    return predictions


def train_prior(train_records: list[dict], classes: list[dict]) -> dict:
    """One median normalized box per class, scored by TRAIN image prevalence."""
    result = {}
    for item in classes:
        class_id = item["id"]
        boxes = [
            np.asarray(box) / [record["width"], record["height"], record["width"], record["height"]]
            for record in train_records
            for box, label in zip(record["boxes"], record["labels"], strict=True)
            if label == class_id
        ]
        if boxes:
            result[str(class_id)] = {"box": np.median(boxes, axis=0).tolist(), "score": sum(class_id in record["labels"] for record in train_records) / len(train_records)}
    return result


def prior_predictions(prior: dict, records: list[dict]) -> dict:
    outputs = {}
    for record in records:
        outputs[record["id"]] = {
            "boxes": [(np.asarray(item["box"]) * [record["width"], record["height"], record["width"], record["height"]]).tolist() for item in prior.values()],
            "labels": [int(key) for key in prior],
            "scores": [item["score"] for item in prior.values()],
        }
    return outputs


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.amp and not args.device.startswith("cuda"):
        raise ValueError("--amp is supported only on CUDA")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = load_manifest(args.manifest)
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    val_records = [record for record in manifest["records"] if record["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError("training and validation splits must both be nonempty")
    # Detect changed files before trusting a frozen manifest.
    for record in train_records + val_records:
        if record.get("sha256") and file_digest(safe_image_path(Path(args.data_root), record["image"])) != record["sha256"]:
            raise ValueError(f"image checksum mismatch: {record['id']}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("resume must continue in the original run directory")
    if args.resume and not (output / "best.pt").is_file():
        raise ValueError("resume requires the original best.pt alongside last.pt")
    if (output / "run.json").exists() and not args.resume:
        raise FileExistsError("run already exists; use a new output directory or explicit --resume")
    configuration = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed, "horizontal_flip_probability": 0.5 if args.horizontal_flip else 0.0, "freeze_backbone": args.freeze_backbone, "amp": args.amp, "pretrained_initialization": not args.no_pretrained, "manifest_sha256": file_digest(args.manifest), "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING, "optimizer": "AdamW", "weight_decay": 0.0005, "scheduler": "CosineAnnealingLR", "batch_norm": "running statistics frozen during fine-tuning; affine params trainable", "selection": "highest validation AP50:.95; earliest epoch wins ties", "threads": args.threads}
    started = time.time()
    model = build_detector(len(manifest["classes"]) + 1, pretrained=not args.no_pretrained and not args.resume, freeze_backbone=args.freeze_backbone).to(args.device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    generator = torch.Generator().manual_seed(args.seed)
    history, first_epoch, best_ap = [], 0, -1.0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous["config"] != configuration:
            raise ValueError("resume configuration differs from original run")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        history, first_epoch, best_ap = previous["history"], previous["epoch"] + 1, previous["best_ap"]
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        generator.set_state(previous["loader_rng"])
        if args.device.startswith("cuda") and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    prior = train_prior(train_records, manifest["classes"])
    atomic_json(output / "prior.json", {"definition": "TRAIN-only per-class median normalized box and image prevalence; same predictions for every image after scaling", "prior": prior})
    baseline = evaluate_detections(val_records, prior_predictions(prior, val_records), manifest["classes"])
    atomic_json(output / "baseline_validation.json", baseline)
    pretrained_cache = Path(torch.hub.get_dir()) / "checkpoints" / PRETRAINED_URL.rsplit("/", 1)[-1]
    provenance = {"run_id": output.name, "dataset_id": manifest["dataset_id"], "classes": manifest["classes"], "taxonomy_version": manifest["taxonomy_version"], "config": configuration, "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda, "device": args.device, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}, "pretrained": {"url": PRETRAINED_URL if not args.no_pretrained else None, "sha256": file_digest(pretrained_cache) if pretrained_cache.exists() and not args.no_pretrained else None, "initialization": "COCO backbone and box regressor; new normal(0,.03) classification head" if not args.no_pretrained else "random initialization"}, "train_images": len(train_records), "validation_images": len(val_records), "holdout_accessed": False}
    atomic_json(output / "run.json", provenance)
    if not args.resume:
        save_checkpoint(output / "initialized.pt", {"architecture": ARCHITECTURE, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "classes": manifest["classes"], "config": configuration, "epoch": -1})
    dataset = DetectionDataset(train_records, args.data_root, configuration["horizontal_flip_probability"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0, generator=generator)
    for epoch in range(first_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        # BatchNorm1d spatial 1x1 with a final singleton batch is otherwise undefined.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        loss_sums = {}
        samples = 0
        skipped_amp_steps = 0
        for images, targets, _ in loader:
            images = [image.to(args.device) for image in images]
            targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", enabled=args.amp and args.device.startswith("cuda")):
                losses = model(images, targets)
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite training loss")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=not scaler.is_enabled()
            )
            # Dynamic scaling may overflow transiently; scaler skips that update.
            if not torch.isfinite(gradient_norm):
                skipped_amp_steps += 1
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach()) * len(images)
            samples += len(images)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        predictions = predict_records(model, val_records, args.data_root, args.device, args.batch_size)
        report = evaluate_detections(val_records, predictions, manifest["classes"])
        current_ap = report["map50_95"]
        if current_ap is None:
            raise ValueError("validation has no ground truth for model selection")
        improved = current_ap > best_ap
        best_ap = max(best_ap, current_ap)
        row = {"epoch": epoch, "learning_rate": learning_rate, "loss": {key: value / samples for key, value in loss_sums.items()}, "samples": samples, "skipped_amp_steps": skipped_amp_steps, "validation_map50": report["map50"], "validation_map50_95": current_ap, "seconds": time.perf_counter() - epoch_start, "selected": improved}
        history.append(row)
        checkpoint = {"architecture": ARCHITECTURE, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "classes": manifest["classes"], "taxonomy_version": manifest["taxonomy_version"], "dataset_id": manifest["dataset_id"], "training_run_id": output.name, "config": configuration, "epoch": epoch, "best_ap": best_ap, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            atomic_json(output / "best_validation.json", report)
            atomic_json(output / "best_validation_predictions.json", predictions)
        atomic_json(output / "history.json", {"epochs": history})
        print(json.dumps(row), flush=True)
    provenance.update({"completed": True, "elapsed_seconds": time.time() - started, "best_validation_map50_95": best_ap, "best_checkpoint_sha256": file_digest(output / "best.pt"), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None})
    atomic_json(output / "run.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="explicit random initialization for smoke tests")
    parser.add_argument("--resume", help="resume last.pt with identical configuration including total epochs")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
