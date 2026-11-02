"""Immutable, checksummed model bundles. Loading never downloads weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARCHITECTURE = "ssdlite320_mobilenet_v3_large"
PREPROCESSING = "ssdlite320_rgb_unit_v1"
COORDINATES = "xyxy_pixels_original_image"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleError(ValueError):
    """A bundle is absent, incompatible, or fails integrity validation."""


class Taxon(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=128)


class BundleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1"] = "1"
    architecture: Literal["ssdlite320_mobilenet_v3_large"] = ARCHITECTURE
    preprocessing: Literal["ssdlite320_rgb_unit_v1"] = PREPROCESSING
    coordinate_system: Literal["xyxy_pixels_original_image"] = COORDINATES
    input_size: tuple[Literal[320], Literal[320]] = (320, 320)
    model_version: str = Field(min_length=1, max_length=128)
    taxonomy_version: str = Field(min_length=1, max_length=128)
    classes: list[Taxon] = Field(min_length=1, max_length=100)
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_run_id: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0, le=1)
    nms_threshold: float = Field(gt=0, le=1)
    max_detections: int = Field(ge=1, le=300)
    runtime: dict[str, str]
    licensing: dict[str, Any]

    @model_validator(mode="after")
    def coherent(self):
        if [item.id for item in self.classes] != list(range(1, len(self.classes) + 1)):
            raise ValueError("taxonomy must have consecutive IDs starting at 1")
        if len({item.name for item in self.classes}) != len(self.classes):
            raise ValueError("taxonomy names must be unique")
        if not all(self.runtime.get(key) for key in ("python", "torch", "torchvision")):
            raise ValueError("runtime versions are required")
        if not self.licensing:
            raise ValueError("licensing and source attribution are required")
        return self


@dataclass
class LoadedBundle:
    model: Any
    metadata: BundleMetadata
    device: str


def _read_json(path: Path, limit: int) -> Any:
    if not path.is_file() or path.is_symlink():
        raise BundleError(f"missing or invalid bundle file: {path.name}")
    if path.stat().st_size > limit:
        raise BundleError(f"bundle file too large: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bundle(path: str | Path) -> BundleMetadata:
    """Validate all persisted semantics and digests before deserialization."""
    directory = Path(path)
    try:
        metadata = BundleMetadata.model_validate(
            _read_json(directory / "metadata.json", 1024 * 1024)
        )
        unsigned = metadata.model_dump(mode="json", exclude={"artifact_digest"})
        if hashlib.sha256(canonical_json(unsigned)).hexdigest() != metadata.artifact_digest:
            raise BundleError("metadata digest mismatch")
        weights = directory / "weights.pt"
        if not weights.is_file() or weights.is_symlink():
            raise BundleError("missing or invalid weights.pt")
        if weights.stat().st_size > 100 * 1024 * 1024:
            raise BundleError("weights exceed supported architecture size")
        if sha256_file(weights) != metadata.weights_sha256:
            raise BundleError("weights digest mismatch")
        _read_json(directory / "evaluation.json", 16 * 1024 * 1024)
        if sha256_file(directory / "evaluation.json") != metadata.evaluation_sha256:
            raise BundleError("evaluation digest mismatch")
        return metadata
    except BundleError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise BundleError("bundle metadata is missing, malformed, or incompatible") from exc


def load_bundle(path: str | Path, device: str = "cpu") -> LoadedBundle:
    import torch
    import torchvision

    from inspection_dl.model import build_detector

    metadata = validate_bundle(path)
    # Minor-version compatibility is explicit; patch updates may include security fixes.
    for key, actual in (("torch", torch.__version__), ("torchvision", torchvision.__version__)):
        expected = metadata.runtime[key].split("+")[0].split(".")[:2]
        if actual.split("+")[0].split(".")[:2] != expected:
            raise BundleError(f"unsupported {key} runtime version")
    if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
        raise BundleError("requested inference device is unavailable")
    try:
        # A tensor-only state dict is the supported serialization format.
        state = torch.load(Path(path) / "weights.pt", map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or not state
            or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in state.items()
            )
        ):
            raise BundleError("weights must be a tensor-only state dict")
        model = build_detector(
            len(metadata.classes) + 1,
            pretrained=False,
            score_threshold=0.001,
            nms_threshold=metadata.nms_threshold,
            max_detections=metadata.max_detections,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return LoadedBundle(model=model, metadata=metadata, device=device)
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError("model weights are incompatible or could not be initialized") from exc


def export_bundle(
    state_dict: dict,
    destination: str | Path,
    *,
    classes: list[dict],
    model_version: str,
    taxonomy_version: str,
    dataset_id: str,
    dataset_manifest_sha256: str,
    training_run_id: str,
    report: dict,
    licensing: dict,
    threshold: float = 0.25,
    nms_threshold: float = 0.55,
    max_detections: int = 100,
) -> BundleMetadata:
    """Write a new release; an existing destination is never overwritten.

    Interrupted exports remain incomplete and cannot pass validate_bundle. The caller
    should use a new release directory for another attempt.
    """
    import torch
    import torchvision

    destination = Path(destination)
    if (
        not isinstance(state_dict, dict)
        or not state_dict
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        )
    ):
        raise BundleError("export requires a tensor-only state dict")
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "weights.pt").open("xb") as stream:
        torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, stream)
    (destination / "evaluation.json").write_bytes(canonical_json(report))
    payload = {
        "schema_version": "1",
        "architecture": ARCHITECTURE,
        "preprocessing": PREPROCESSING,
        "coordinate_system": COORDINATES,
        "input_size": [320, 320],
        "model_version": model_version,
        "taxonomy_version": taxonomy_version,
        "classes": classes,
        "weights_sha256": sha256_file(destination / "weights.pt"),
        "evaluation_sha256": sha256_file(destination / "evaluation.json"),
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "confidence_threshold": float(threshold),
        "nms_threshold": float(nms_threshold),
        "max_detections": max_detections,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "licensing": licensing,
    }
    payload["artifact_digest"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    metadata = BundleMetadata.model_validate(payload)
    (destination / "metadata.json").write_bytes(canonical_json(metadata.model_dump(mode="json")))
    validate_bundle(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--licensing", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    import torch

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key in ("classes", "dataset_id", "taxonomy_version"):
        if checkpoint.get(key) != manifest.get(key):
            raise BundleError(f"checkpoint and release manifest disagree on {key}")
    manifest_sha256 = sha256_file(args.manifest)
    if checkpoint.get("config", {}).get("manifest_sha256") != manifest_sha256:
        raise BundleError("checkpoint was trained with a different dataset manifest")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise BundleError("checkpoint architecture mismatch")
    if checkpoint.get("config", {}).get("preprocessing") != PREPROCESSING:
        raise BundleError("checkpoint preprocessing mismatch")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise BundleError("evaluation report belongs to a different checkpoint")
    if report.get("manifest_sha256") != manifest_sha256:
        raise BundleError("evaluation report belongs to a different dataset manifest")
    if report.get("threshold") != args.threshold:
        raise BundleError("release threshold differs from the supplied evaluation report")
    metadata = export_bundle(
        checkpoint["model"],
        args.destination,
        classes=manifest["classes"],
        model_version=args.model_version,
        taxonomy_version=manifest["taxonomy_version"],
        dataset_id=manifest["dataset_id"],
        dataset_manifest_sha256=manifest_sha256,
        training_run_id=checkpoint.get("training_run_id", args.checkpoint.parent.name),
        report=report,
        licensing=json.loads(args.licensing.read_text(encoding="utf-8")),
        threshold=args.threshold,
    )
    print(json.dumps(metadata.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
