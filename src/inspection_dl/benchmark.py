"""Benchmark local released inference; never interpret these results as an SLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from inspection_dl.bundle import load_bundle
from inspection_dl.metrics import evaluate_detections
from inspection_dl.train import file_digest, load_manifest, safe_image_path


def process_peak_rss() -> int | None:
    """Lifetime high-water RSS, not a per-configuration allocation measurement."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in [
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                ]
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        function = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        if function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    import resource

    measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return measured if platform.system() == "Darwin" else measured * 1024


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def percentiles(values: list[float]) -> dict:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p99_ms": float(np.percentile(values, 99)) if len(values) >= 1000 else None,
    }


def benchmark(
    bundle: Path,
    manifest: Path,
    data_root: Path,
    devices: list[str],
    batch_sizes: list[int],
    count: int = 100,
    threads: int = 4,
) -> dict:
    if count < 10 or threads < 1 or not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("Use at least 10 samples, positive thread count and batch sizes")
    if batch_sizes[0] != 1:
        raise ValueError("Batch size one must be first for the parity reference")
    torch.set_num_threads(threads)
    raw_manifest = manifest.read_bytes()
    spec = load_manifest(manifest)
    records = sorted((r for r in spec["records"] if r["split"] == "val"), key=lambda r: r["id"])[
        :count
    ]
    if not records:
        raise ValueError("No validation records")
    tensors, decoding = [], []
    for record in records:
        image_path = safe_image_path(data_root, record["image"])
        if file_digest(image_path) != record["sha256"]:
            raise ValueError("Benchmark source image checksum mismatch")
        started = time.perf_counter()
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (record["width"], record["height"]):
                raise ValueError("Benchmark source image dimensions mismatch")
            tensors.append(pil_to_tensor(image).float().div(255))
        decoding.append((time.perf_counter() - started) * 1000)
    output = {
        "schema_version": "1",
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cpu_threads": threads,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "split": "val",
        "sample_ids": [r["id"] for r in records],
        "decode_tensor_conversion": percentiles(decoding),
        "results": [],
        "notes": [
            "Local warmed inference; no network or HTTP overhead.",
            "Bundle load timing includes validation and model construction, but excludes Python imports and initial CUDA context creation.",
            "Forward throughput uses preloaded device inputs; input transfer and CPU output materialization are excluded from that rate.",
            "Decode/tensor timing includes local image-file I/O, but excludes the preceding checksum verification.",
            "Forward timing includes torchvision resize/normalize, detector, NMS and coordinate restoration.",
            "Transfer from CPU to GPU measured separately; no claim of kernel-only model time.",
            "p99 omitted below 1000 timed batches.",
            "GPU memory is PyTorch peak allocated/reserved, excluding driver and other processes.",
            "Process peak RSS is the lifetime high-water mark including preloaded samples and earlier configurations.",
            "Batching comparison uses identical model and input samples; parity is measured against batch size one.",
        ],
    }
    with torch.inference_mode():
        for device in devices:
            if device.startswith("cuda") and not torch.cuda.is_available():
                output["results"].append({"device": device, "unavailable": True})
                continue
            synchronize(device)
            started = time.perf_counter()
            loaded = load_bundle(bundle, device=device)
            if loaded.metadata.dataset_manifest_sha256 != output["manifest_sha256"]:
                raise ValueError("Bundle and benchmark manifest digest differ")
            if [item.model_dump() for item in loaded.metadata.classes] != spec["classes"]:
                raise ValueError("Bundle and benchmark taxonomy differ")
            synchronize(device)
            load_ms = (time.perf_counter() - started) * 1000
            model = loaded.model.eval()
            started = time.perf_counter()
            inputs = [tensor.to(device) for tensor in tensors]
            synchronize(device)
            transfer_ms = (time.perf_counter() - started) * 1000
            reference = None
            for batch_size in batch_sizes:
                for _ in range(3):
                    model(inputs[:batch_size])
                synchronize(device)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                durations, outputs = [], []
                for start in range(0, len(inputs), batch_size):
                    synchronize(device)
                    started = time.perf_counter()
                    batch_output = model(inputs[start : start + batch_size])
                    synchronize(device)
                    durations.append((time.perf_counter() - started) * 1000)
                    outputs.extend(
                        [{k: v.detach().cpu() for k, v in item.items()} for item in batch_output]
                    )
                if reference is None:
                    reference = outputs
                parity = {
                    "identical_detection_counts": True,
                    "labels_equal": True,
                    "max_box_abs_difference_pixels": 0.0,
                    "max_score_abs_difference": 0.0,
                }
                for first, second in zip(reference, outputs):
                    if len(first["labels"]) != len(second["labels"]):
                        parity["identical_detection_counts"] = False
                        continue
                    parity["labels_equal"] &= bool(torch.equal(first["labels"], second["labels"]))
                    for name, key in [
                        ("boxes", "max_box_abs_difference_pixels"),
                        ("scores", "max_score_abs_difference"),
                    ]:
                        if first[name].numel():
                            parity[key] = max(
                                parity[key], float((first[name] - second[name]).abs().max())
                            )
                result = {
                    "device": device,
                    "precision": "float32",
                    "batch_size": batch_size,
                    "concurrency": 1,
                    "timed_batches": len(durations),
                    "images": len(inputs),
                    "model_load_ms": load_ms,
                    "input_transfer_total_ms": transfer_ms,
                    "batch_latency": percentiles(durations),
                    "throughput_images_per_second": len(inputs) / (sum(durations) / 1000),
                    "batch_parity_vs_first_configuration": parity,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if device.startswith("cuda")
                    else None,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                    if device.startswith("cuda")
                    else None,
                }
                predictions = {}
                invalid_outputs_rejected = 0
                for record, item in zip(records, outputs, strict=True):
                    valid = torch.isfinite(item["boxes"]).all(1) & torch.isfinite(item["scores"])
                    valid &= (item["boxes"][:, 2] > item["boxes"][:, 0]) & (
                        item["boxes"][:, 3] > item["boxes"][:, 1]
                    )
                    invalid_outputs_rejected += int((~valid).sum())
                    predictions[record["id"]] = {
                        key: value[valid].tolist() for key, value in item.items()
                    }
                quality = evaluate_detections(
                    records, predictions, spec["classes"], loaded.metadata.confidence_threshold
                )
                result["quality_on_identical_validation_subset"] = {
                    key: quality[key]
                    for key in ["images", "map50", "map50_95", "threshold", "operating"]
                }
                result["process_lifetime_peak_rss_bytes"] = process_peak_rss()
                result["invalid_outputs_rejected"] = invalid_outputs_rejected
                output["results"].append(result)
                print(json.dumps(result), flush=True)
            del model, loaded, inputs
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    results = benchmark(
        args.bundle,
        args.manifest,
        args.data_root,
        args.devices,
        args.batch_sizes,
        args.samples,
        args.threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
