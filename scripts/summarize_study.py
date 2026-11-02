"""Generate three evidence-backed study documents; never invent missing results.

The generator reads completed JSON evidence, checks its identities and source
reconstruction, and writes deterministic Markdown without timestamps. It uses no
model execution, GPU, dataset images, or external network connection.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def number(value, digits: int = 4) -> str:
    return "not available" if value is None else f"{value:.{digits}f}"


def mib(value) -> str:
    return "not measured" if value is None else number(value / 2**20, 1)


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


class Evidence:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files = {}

    def path(self, relative: str | Path) -> Path:
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Evidence paths must remain inside the repository")
        return result

    def read(self, relative: str | Path) -> dict:
        path = self.path(relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Required completed evidence missing: {path.relative_to(self.root)}"
            )
        self.files[path.relative_to(self.root).as_posix()] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def include(self, relative: str | Path) -> None:
        path = self.path(relative)
        self.files[path.relative_to(self.root).as_posix()] = sha256(path)


def git_snapshot(root: Path, revision: str) -> tuple[str, dict, dict]:
    paths = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "src/inspection_dl"], cwd=root
        )
        .decode()
        .splitlines()
    )
    digest, hashes, syntax = hashlib.sha256(), {}, {}
    for path in sorted(paths):
        relative = Path(path)
        if relative.suffix != ".py" or len(relative.parts) != 3:
            continue
        content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root)
        digest.update(relative.name.encode() + b"\0" + content + b"\0")
        hashes[relative.name] = hashlib.sha256(content).hexdigest()
        syntax[relative.name] = ast.dump(ast.parse(content), include_attributes=False)
    return digest.hexdigest(), hashes, syntax


def reconstruct_sources(root: Path, runs: list[dict]) -> list[dict]:
    """Find retained exact source snapshots, including runs started from dirty trees."""
    remaining = {run["source"]["package_source_sha256"] for run in runs}
    revisions = (
        subprocess.check_output(["git", "rev-list", "--all", "--", "src/inspection_dl"], cwd=root)
        .decode()
        .splitlines()
    )
    found = {}
    for revision in revisions:
        digest, hashes, syntax = git_snapshot(root, revision)
        if digest in remaining:
            found[digest] = (revision, hashes, syntax)
            remaining.remove(digest)
        if not remaining:
            break
    if remaining:
        raise ValueError(
            f"Training source hashes lack an exact retained Git snapshot: {sorted(remaining)}"
        )
    result = []
    first_syntax = found[runs[0]["source"]["package_source_sha256"]][2]
    for run in runs:
        digest = run["source"]["package_source_sha256"]
        revision, hashes, syntax = found[digest]
        core = ["train.py", "model.py", "metrics.py"]
        equal = all(syntax[name] == first_syntax[name] for name in core)
        if not equal:
            raise ValueError(
                "Training/model/metric syntax differs between controlled-comparison runs"
            )
        result.append(
            {
                "run_id": run["run_id"],
                "recorded_revision": run["source"]["revision"],
                "recorded_dirty": run["source"]["dirty"],
                "package_source_sha256": digest,
                "retained_exact_source_commit": revision,
                "exact_byte_match": True,
                "module_sha256": hashes,
                "training_model_metric_ast_equal_across_runs": equal,
            }
        )
    return result


def load_study(args) -> dict:
    evidence = Evidence(args.repo_root)
    study, release = Path(args.study), Path(args.release)
    required = [
        study / "selection.json",
        study / "initialization-parity.json",
        study / "validation/report.json",
        study / "validation/robustness.json",
        study / "holdout/report.json",
        study / "holdout/per_image_errors.json",
        study / "holdout/examples.json",
        study / "benchmark.json",
        release / "metadata.json",
        release / "evaluation.json",
    ]
    missing = [str(path) for path in required if not evidence.path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Study incomplete; no documents generated. Missing: " + ", ".join(missing)
        )
    manifest = evidence.read(args.manifest)
    audit = evidence.read(Path(args.manifest).with_name(Path(args.manifest).stem + ".audit.json"))
    preregistration = evidence.read(study / "preregistration.json")
    initialization = evidence.read(study / "initialization-parity.json")
    if initialization.get("identical_initial_model_tensors") is not True:
        raise ValueError("Controlled comparison lacks exact initial tensor parity")
    if initialization.get("tensor_count", 0) < 1:
        raise ValueError("Initial tensor parity evidence is empty")
    selection = evidence.read(study / "selection.json")
    validation = evidence.read(study / "validation/report.json")
    robustness = evidence.read(study / "validation/robustness.json")
    control_path = study / "validation/robustness-control.json"
    robustness_control = (
        evidence.read(control_path) if evidence.path(control_path).is_file() else None
    )
    holdout = evidence.read(study / "holdout/report.json")
    errors = evidence.read(study / "holdout/per_image_errors.json")["images"]
    examples = evidence.read(study / "holdout/examples.json")
    benchmark = evidence.read(study / "benchmark.json")
    metadata = evidence.read(release / "metadata.json")
    released_report = evidence.read(release / "evaluation.json")
    diagnosis = evidence.read("reports/training-diagnosis.json")
    failed_run = evidence.read("reports/rtx4060-study-v1/failed-run.json")
    failure = evidence.read("reports/rtx4060-study-v1/failure.json")
    acquisition = evidence.read("data/manifests/pcb-ind-v4.acquisition.json")
    annotation_check = evidence.read("reports/validation/annotation-crosscheck.json")
    metric_check = evidence.read("reports/metric-reference.json")
    environment_report = evidence.read("reports/environment.json")
    manifest_portability = evidence.read("reports/validation/manifest-portability.json")
    manifest_hash = evidence.files[
        evidence.path(args.manifest).relative_to(evidence.root).as_posix()
    ]
    content = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if hashlib.sha256(canonical(content)).hexdigest() != manifest["manifest_digest"]:
        raise ValueError("Canonical manifest content digest mismatch")
    if (
        manifest_portability.get("byte_identical") is not True
        or manifest_portability["working_sha256"] != manifest_hash
        or manifest_portability["committed_blob_sha256"] != manifest_hash
    ):
        raise ValueError("Committed and working manifest bytes differ from training provenance")
    for item in (preregistration, selection, validation, holdout, benchmark):
        if item["manifest_sha256"] != manifest_hash:
            raise ValueError("Study evidence refers to inconsistent manifests")
    if metadata["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("Release refers to a different manifest")
    if (
        metadata["classes"] != manifest["classes"]
        or metadata["taxonomy_version"] != manifest["taxonomy_version"]
    ):
        raise ValueError("Release taxonomy differs from dataset")
    if metadata["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Release dataset identity differs")
    for item in (validation, holdout):
        if [{"id": row["id"], "name": row["name"]} for row in item["per_class"]] != manifest[
            "classes"
        ]:
            raise ValueError("Evaluation taxonomy differs")
    selected = selection["selected"]
    if max(selection["candidates"], key=lambda row: row["validation_map50_95"]) != selected:
        raise ValueError("Selected candidate does not follow preregistered validation rule")
    if not validation["threshold_selected_here_on_validation"] or validation["split"] != "val":
        raise ValueError("Operating threshold was not selected on validation")
    if (
        not holdout["final_holdout"]
        or holdout["split"] != "test"
        or holdout["threshold_selected_here_on_validation"]
    ):
        raise ValueError("Final holdout protocol is inconsistent")
    threshold = selection["threshold"]
    for value in (validation["threshold"], holdout["threshold"], metadata["confidence_threshold"]):
        if value != threshold:
            raise ValueError("Validation, holdout and release thresholds differ")
    expected_threshold = max(
        validation["threshold_sweep"], key=lambda row: (row["f1"], row["threshold"])
    )["threshold"]
    if expected_threshold != threshold:
        raise ValueError("Selected threshold does not maximize recorded validation F1")
    for item in (validation, holdout):
        if (
            item["checkpoint_sha256"] != selected["checkpoint_sha256"]
            or item["training_run_id"] != selected["run_id"]
        ):
            raise ValueError("Evaluation does not refer to the selected checkpoint/run")
    if metadata["training_run_id"] != selected["run_id"] or released_report != holdout:
        raise ValueError("Released model evidence differs from selected holdout result")
    unsigned = {key: value for key, value in metadata.items() if key != "artifact_digest"}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != metadata["artifact_digest"]:
        raise ValueError("Release metadata digest mismatch")
    evidence.include(release / "weights.pt")
    if evidence.files[(release / "weights.pt").as_posix()] != metadata["weights_sha256"]:
        raise ValueError("Released weight digest mismatch")
    if evidence.files[(release / "evaluation.json").as_posix()] != metadata["evaluation_sha256"]:
        raise ValueError("Released evaluation digest mismatch")
    runs, histories = [], []
    expected_configuration = {
        "epochs": preregistration["epochs_per_run"],
        "batch_size": preregistration["batch_size"],
        "learning_rate": preregistration["learning_rate"],
        "seed": preregistration["seed"],
        "amp": preregistration["amp"],
        "architecture": preregistration["architecture"],
    }
    for candidate in selection["candidates"]:
        location = study / "experiments" / candidate["variant"]
        run = evidence.read(location / "run.json")
        history = evidence.read(location / "history.json")["epochs"]
        if not run.get("completed") or len(history) != run["config"]["epochs"]:
            raise ValueError("A comparison run is incomplete")
        if any(run["config"].get(key) != value for key, value in expected_configuration.items()):
            raise ValueError("A completed run differs from the preregistered configuration")
        if run["holdout_accessed"] or run["config"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Training protocol or manifest provenance is inconsistent")
        if candidate["validation_map50_95"] != run["best_validation_map50_95"]:
            raise ValueError("Candidate validation result differs from completed run")
        if not math.isclose(
            max(row["validation_map50_95"] for row in history),
            candidate["validation_map50_95"],
            abs_tol=1e-12,
        ):
            raise ValueError("Selected validation quality differs from history")
        runs.append(run)
        histories.append(history)
    if len(runs) != 2:
        raise ValueError("This report generator expects the preregistered two-variant study")
    if set(initialization["candidates"]) != {run["run_id"] for run in runs}:
        raise ValueError("Initial tensor parity evidence refers to different training runs")
    comparable = [
        {key: value for key, value in run["config"].items() if key != "horizontal_flip_probability"}
        for run in runs
    ]
    if comparable[0] != comparable[1] or sorted(
        run["config"]["horizontal_flip_probability"] for run in runs
    ) != [0, 0.5]:
        raise ValueError("Controlled comparison changes more than horizontal flipping")
    sources = reconstruct_sources(evidence.root, runs)
    holdout_records = [row for row in manifest["records"] if row["split"] == "test"]
    validation_records = sorted(
        [row for row in manifest["records"] if row["split"] == "val"], key=lambda row: row["id"]
    )
    if holdout["images"] != len(holdout_records) or validation["images"] != len(validation_records):
        raise ValueError("Reported split sizes differ from frozen manifest")
    if {row["id"] for row in errors} != {row["id"] for row in holdout_records} or len(
        errors
    ) != len(holdout_records):
        raise ValueError("Per-image error evidence does not cover the entire holdout exactly once")
    for key in ("tp", "fp", "fn"):
        if sum(row[key] for row in errors) != holdout["operating"][key]:
            raise ValueError("Per-image and aggregate operating counts differ")
    for item in holdout["per_class"]:
        if item["ground_truth"] != audit["class_instance_counts"][item["name"]]["test"]:
            raise ValueError("Holdout class instance counts differ from audited data")
    for ids in (robustness["images"], benchmark["sample_ids"]):
        if ids != [row["id"] for row in validation_records[: len(ids)]]:
            raise ValueError(
                "Robustness/benchmark subset differs from declared deterministic validation sampling"
            )
    if robustness["unperturbed_same_subset"]["images"] != len(robustness["images"]):
        raise ValueError("Robustness clean comparison subset differs")
    if robustness_control is not None:
        expected_control = {
            "source_type": "bundle",
            "artifact_digest": metadata["artifact_digest"],
            "weights_sha256": metadata["weights_sha256"],
            "manifest_sha256": manifest_hash,
            "threshold": threshold,
            "batch_size": 1,
            "device": validation["device"],
            "images": robustness["images"],
        }
        if any(robustness_control.get(key) != value for key, value in expected_control.items()):
            raise ValueError("Robustness clean-control identity or batch protocol differs")
        if robustness_control["unperturbed_same_subset"]["images"] != len(robustness["images"]):
            raise ValueError("Robustness clean-control image count differs")
    for item in examples["examples"]:
        evidence.include(study / "holdout" / item["file"])
    for split in ("validation", "holdout"):
        evidence.include(study / split / "precision_recall_and_scores.png")
    for result in benchmark["results"]:
        if not result.get("unavailable") and result["quality_on_identical_validation_subset"][
            "images"
        ] != len(benchmark["sample_ids"]):
            raise ValueError("Benchmark quality does not use the timed subset")
    return locals()


def aggregate_table(validation: dict, holdout: dict) -> str:
    rows = []
    for name, report in (
        ("Validation: train-derived prior baseline", validation["baseline"]),
        ("Validation: trained detector", validation),
        ("Final holdout: train-derived prior baseline", holdout["baseline"]),
        ("Final holdout: trained detector", holdout),
    ):
        operating = report["operating"]
        rows.append(
            [
                name,
                report["images"],
                number(report["map50"]),
                number(report["map50_95"]),
                number(operating["precision"]),
                number(operating["recall"]),
                number(operating["f1"]),
            ]
        )
    return table(["Evaluation", "Images", "AP50", "AP50:.95", "Precision", "Recall", "F1"], rows)


def uncertainty_text(uncertainty: dict) -> str:
    if "precision_95_interval" not in uncertainty:
        return f"Intervals unavailable: {uncertainty.get('status', 'not supplied')}; {uncertainty['groups']} groups."
    precision = ", ".join(number(value) for value in uncertainty["precision_95_interval"])
    recall = ", ".join(number(value) for value in uncertainty["recall_95_interval"])
    return (
        f"A percentile bootstrap over {uncertainty['groups']} held-out source/duplicate groups "
        f"({uncertainty['repetitions']} resamples, seed {uncertainty['seed']}) gives 95% intervals "
        f"for micro precision **[{precision}]** and recall **[{recall}]** at IoU 0.50 and the frozen threshold. "
        "These intervals reflect observed group sampling; they do not estimate training-seed variation, "
        "unseen-board independence or production-domain shift."
    )


def model_card(context: dict) -> str:
    c = context
    manifest, metadata, holdout, validation = (
        c[key] for key in ("manifest", "metadata", "holdout", "validation")
    )
    study, release = c["study"].as_posix(), c["release"].as_posix()
    selected, audit, benchmark = c["selected"], c["audit"], c["benchmark"]
    run = next(run for run in c["runs"] if run["run_id"] == selected["run_id"])
    config = run["config"]
    rows = []
    for item in holdout["per_class"]:
        counts = audit["class_image_counts"][item["name"]]
        rows.append(
            [
                item["name"],
                counts["train"],
                counts["val"],
                counts["test"],
                item["ground_truth"],
                number(item["ap50"]),
                number(item["ap50_95"]),
                number(item["operating"]["precision"]),
                number(item["operating"]["recall"]),
            ]
        )
    candidates = []
    for candidate, candidate_run, history in zip(
        c["selection"]["candidates"], c["runs"], c["histories"], strict=True
    ):
        best = max(history, key=lambda row: row["validation_map50_95"])
        candidates.append(
            [
                candidate["variant"],
                len(history),
                best["epoch"] + 1,
                number(candidate["validation_map50_95"]),
                number(sum(row["seconds"] for row in history) / 60, 2),
                sum(row["skipped_amp_steps"] for row in history),
            ]
        )
    perf, quality = [], []
    for result in benchmark["results"]:
        if result.get("unavailable"):
            perf.append([result["device"], "unavailable", "—", "—", "—", "—", "—", "—", "—"])
            continue
        latency, parity = result["batch_latency"], result["batch_parity_vs_first_configuration"]
        measured = result["quality_on_identical_validation_subset"]
        perf.append(
            [
                result["device"],
                result["batch_size"],
                result["timed_batches"],
                number(result["model_load_ms"], 2),
                number(latency["p50_ms"], 2),
                number(latency["p95_ms"], 2),
                number(result["throughput_images_per_second"], 2),
                mib(result["gpu_peak_allocated_bytes"]),
                mib(result["process_lifetime_peak_rss_bytes"]),
            ]
        )
        quality.append(
            [
                result["device"],
                result["batch_size"],
                number(measured["map50"]),
                number(measured["map50_95"]),
                number(measured["operating"]["precision"]),
                number(measured["operating"]["recall"]),
                parity["identical_detection_counts"],
                parity["labels_equal"],
                number(parity["max_box_abs_difference_pixels"], 6),
                number(parity["max_score_abs_difference"], 8),
            ]
        )
    improvements = []
    for device in sorted({row["device"] for row in benchmark["results"]}):
        configs = {
            row["batch_size"]: row
            for row in benchmark["results"]
            if row["device"] == device and not row.get("unavailable")
        }
        if 1 in configs and 4 in configs:
            first, fourth = configs[1], configs[4]
            ratio = fourth["throughput_images_per_second"] / first["throughput_images_per_second"]
            delta = (
                fourth["quality_on_identical_validation_subset"]["map50_95"]
                - first["quality_on_identical_validation_subset"]["map50_95"]
            )
            first_counts = first["quality_on_identical_validation_subset"]["operating"]
            fourth_counts = fourth["quality_on_identical_validation_subset"]["operating"]
            changes = ", ".join(
                f"{key.upper()} {fourth_counts[key] - first_counts[key]:+d}"
                for key in ("tp", "fp", "fn")
            )
            improvements.append(
                f"{device}: batch 4/batch 1 throughput ratio {ratio:.3f}; "
                f"AP50:.95 change {delta:+.6f}; operating-count changes {changes} on the identical subset."
            )
    parity_issues = [
        f"{row['device']} batch {row['batch_size']}"
        for row in benchmark["results"]
        if not row.get("unavailable")
        and (
            not row["batch_parity_vs_first_configuration"]["labels_equal"]
            or not row["batch_parity_vs_first_configuration"]["identical_detection_counts"]
        )
    ]
    parity_note = (
        "Exact output parity is not established for "
        + ", ".join(parity_issues)
        + ". Raw model output counts and label order are compared before validation and "
        "operating-threshold filtering. Coordinate and score differences compare rows by "
        "score rank; when labels/order differ, the coordinate maximum is not displacement "
        "of matched detections. The AP and operating-count comparisons measure the observed "
        "quality effect independently of this rank comparison."
        if parity_issues
        else "Raw output order/count checks and numerical differences are reported separately from operating quality."
    )
    artifact_rows = [
        ["Model version", metadata["model_version"]],
        ["Artifact digest", metadata["artifact_digest"]],
        ["Weights SHA256", metadata["weights_sha256"]],
        ["Selected checkpoint SHA256", selected["checkpoint_sha256"]],
        ["Dataset ID", manifest["dataset_id"]],
        ["Manifest file SHA256", metadata["dataset_manifest_sha256"]],
        ["Canonical manifest digest", manifest["manifest_digest"]],
        ["Taxonomy version", metadata["taxonomy_version"]],
        ["Training run", selected["run_id"]],
    ]
    return f"""# Model card: {metadata["model_version"]}

This released PyTorch detector proposes eight classes of PCB surface defects with
bounding boxes. It is independently runnable through a local FastAPI service.
The measured task is **300×300 PCB-IND AOI candidate ROI detection**. Other image
dimensions may be accepted by the API, but their detection quality is unvalidated.
Full-board search, other factories/cameras/materials and production false-alarm
rates are outside the demonstrated evidence. Human review remains necessary.

## Model and image processing

Architecture: `{metadata["architecture"]}`; preprocessing version:
`{metadata["preprocessing"]}`. The COCO_V1 pretrained backbone and box regressor
initialize the model; its classification head is replaced for eight defects plus
background. PyTorch **{run["environment"]["torch"]}** and torchvision
**{run["environment"]["torchvision"]}** execute the recorded study.

Input decoding applies EXIF orientation, converts to RGB and creates floating
CHW pixels in [0,1]. Torchvision stretches to 320×320 without letterboxing and
normalizes each channel with mean/std 0.5/0.5. Returned boxes use exclusive-upper-edge
`xyxy` pixels in the original EXIF-oriented image frame. Training augmentation in
the selected run reflects horizontally with probability
**{config["horizontal_flip_probability"]}** and transforms target boxes accordingly.

The model retains candidates above score 0.001, applies class-aware NMS at
IoU **{metadata["nms_threshold"]}**, and caps outputs at
**{metadata["max_detections"]}** detections/image. Serving's default confidence
threshold is **{c["selection"]["threshold"]}**, selected solely by validation micro
F1 at IoU 0.50; higher thresholds win ties. This is a demonstration operating point,
not an industrial acceptance criterion or a calibrated probability cutoff.

## Data and model selection

The author release [PCB-IND v4](https://zenodo.org/records/19723114) supplies real
AOI production-line image patches under CC BY 4.0. The frozen audit accepts
{audit["accepted_images"]} images: {audit["split_counts"]["train"]} train,
{audit["split_counts"]["val"]} validation and {audit["split_counts"]["test"]} test.
{len(audit["exclusions"])} malformed image/annotation record is explicitly excluded.
All shared numeric filename-prefix proxies and detected visual duplicates stay
within one split. These inferred groups reduce obvious correlation leakage;
their physical-board/session semantics are unconfirmed. See [dataset details](DATASET.md).

Two bounded runs share seed {config["seed"]}, batch {config["batch_size"]},
{config["optimizer"]} at learning rate {config["learning_rate"]}, weight decay
{config["weight_decay"]}, cosine scheduling, pretrained initialization and epoch
budget. Only horizontal reflection probability differs. BatchNorm running
statistics are frozen while trainable affine parameters are fine-tuned.
Training uses {"AMP" if config["amp"] else "FP32"}; the earlier failed AMP experiment remains separately reported.
There is one training seed, so the comparison does not establish seed-robust gains.
The [initialization check](../{study}/initialization-parity.json) confirms exact
CPU equality of every initial model parameter and buffer across both runs
({c["initialization"]["tensor_count"]} tensors); matching seeds alone are not used
as evidence that initialization was identical.

{table(["Variant", "Completed epochs", "Best epoch (1-based)", "Validation AP50:.95", "Epoch time total, min", "Skipped AMP steps"], candidates)}

Selected variant: **{selected["variant"]}**. Checkpoint and threshold selection
use validation only, followed by a recorded decision and final holdout evaluation.
The simple baseline proposes one train-derived median normalized box per class,
with its training-image prevalence as confidence. It returns the same normalized
geometry for every image and is a deliberately limited reference.

## Quantitative results

All AP, precision, recall and F1 values below use a 0–1 scale. AP uses class-aware
greedy IoU matching, a 101-point precision envelope, and IoUs 0.50:0.05:0.95.
No crowd/ignore handling or COCO object-area/maxDet averaging is claimed.
Operating precision/recall/F1 use same-class IoU 0.50 matching at the shared frozen
threshold for both baseline and trained model.

{aggregate_table(validation, holdout)}

{table(["Class", "Train images", "Val images", "Test images", "Test GT boxes", "Test AP50", "Test AP50:.95", "Precision", "Recall"], rows)}

{uncertainty_text(holdout["uncertainty"])}

The `open` class has only {audit["class_image_counts"]["open"]["test"]} test images;
per-class results on rare defects have limited precision. Score histograms,
precision–recall curves and binned observed detection precision are descriptive;
scores are not asserted to be calibrated probabilities. Full failure evidence is
in [error analysis](ERROR_ANALYSIS.md), with [machine-readable holdout results](../{study}/holdout/report.json)
and [PR/score plots](../{study}/holdout/precision_recall_and_scores.png).

## Local performance and bounded optimization

Hardware: **{benchmark["hardware"]["gpu"]}**; CPU identification:
**{c["environment_report"]["hardware"]["cpu"].strip()}**; platform `{benchmark["hardware"]["platform"]}`.
The [hardware record](../reports/environment.json) reports
{number(c["environment_report"]["hardware"]["ram_bytes"] / 2**30, 2)}GiB physical RAM
and GPU model/VRAM/driver `{c["environment_report"]["hardware"]["gpu"]}`.
The CPU path uses {benchmark["hardware"]["cpu_threads"]} threads. The benchmark
uses the first {len(benchmark["sample_ids"])} sorted validation IDs, FP32,
concurrency 1, and three untimed warm-up batches for each configuration.
This measures local warmed inference without HTTP/network overhead.

{table(["Device", "Batch", "Timed batches", "Bundle load, ms", "Batch p50, ms", "Batch p95, ms", "Images/s", "Peak allocated VRAM, MiB", "Lifetime peak RSS, MiB"], perf)}

{table(["Device", "Batch", "Subset AP50", "Subset AP50:.95", "Precision", "Recall", "Counts equal vs batch1", "Labels equal", "Max box difference, px", "Max score difference"], quality)}

{" ".join(improvements)} The optimization changes batch policy only; these
quality comparisons use identical images and the same released weights.
{parity_note}
Batch latency is not per-image request latency. Forward timing includes resize,
normalization, model execution, NMS and coordinate restoration, using preloaded
device inputs. Its throughput excludes input transfer and CPU output materialization.
Decoding/tensor
conversion p50/p95 are {number(benchmark["decode_tensor_conversion"]["p50_ms"], 3)}/
{number(benchmark["decode_tensor_conversion"]["p95_ms"], 3)}ms; transfer times are
reported separately in [benchmark JSON](../{study}/benchmark.json). Decode/tensor
timing includes image-file I/O and excludes preceding checksum verification.
Bundle load timing includes integrity validation and model construction, but
excludes Python imports and initial CUDA context creation; cache state is uncontrolled.
It does not establish full process startup. p99 is omitted below 1,000 timed batches.
GPU memory is PyTorch peak allocated/reserved, excluding driver/other processes;
process RSS is a lifetime high-water mark including preloaded data and previous
configurations. These measurements are not production SLA guarantees.

## Release identity and use limits

{table(["Field", "Recorded value"], artifact_rows)}

Load the checked bundle from `{release}`; it includes weights, metadata and the
exact evaluation report. Integrity and compatibility failures prevent readiness;
inference does not silently substitute checkpoints or simulated predictions.
See [reproducibility](REPRODUCIBILITY.md) and [API semantics](API.md).

The source code is MIT; dataset images/labels remain CC BY 4.0 with author
attribution. Pretrained-model and underlying COCO image terms remain applicable;
no unrestricted commercial weight rights are asserted. See
[third-party terms](THIRD_PARTY.md) and the release metadata's licensing record.
This portfolio/research project has no Matroid affiliation, manufacturing deployment
claim or evidence of safety-critical fitness.

Generated deterministically by `scripts/summarize_study.py` from completed,
identity-checked JSON evidence. Input digests are recorded with the study.
"""


def error_analysis(context: dict) -> str:
    c = context
    holdout, validation, errors = c["holdout"], c["validation"], c["errors"]
    study = c["study"].as_posix()
    operating = holdout["operating"]
    categories = []
    for key, description in (
        ("tp", "Matched correct-class boxes"),
        ("fp", "Unmatched/wrong-class predictions"),
        ("fn", "Unmatched ground-truth boxes"),
        ("poor_localization_candidates", "FP with same-class unmatched IoU in [0.1,0.5)"),
        ("overlapping_prediction_pairs", "Prediction pairs with IoU≥0.5, regardless of class"),
    ):
        categories.append(
            [
                key,
                sum(row[key] for row in errors),
                sum(row[key] > 0 for row in errors),
                max(row[key] for row in errors),
                description,
            ]
        )
    per_class = []
    for item in sorted(
        holdout["per_class"],
        key=lambda row: (row["ap50_95"] if row["ap50_95"] is not None else -1, row["name"]),
    ):
        counts = item["operating"]
        per_class.append(
            [
                item["name"],
                item["images_with_class"],
                item["ground_truth"],
                number(item["ap50_95"]),
                counts["tp"],
                counts["fp"],
                counts["fn"],
                number(counts["precision"]),
                number(counts["recall"]),
            ]
        )
    confusion = holdout["confusion"]
    confusions = []
    for row_index, label in enumerate(confusion["labels"][:-1]):
        for column_index, predicted in enumerate(confusion["labels"][:-1]):
            count = confusion["counts"][row_index][column_index]
            if row_index != column_index and count:
                confusions.append((count, label, predicted))
    confusions.sort(key=lambda row: (-row[0], row[1], row[2]))
    confusion_text = (
        table(
            ["Truth", "Predicted", "Class-agnostic matched boxes"],
            [[truth, predicted, count] for count, truth, predicted in confusions[:10]],
        )
        if confusions
        else "No class-to-class off-diagonal matches were observed at this operating point."
    )
    slices = []
    for name, result in holdout["image_slices"]["slices"].items():
        slices.append(
            [
                name,
                result["images"],
                number(result["map50"]),
                number(result["map50_95"]),
                number(result["operating"]["precision"]),
                number(result["operating"]["recall"]),
            ]
        )
    robust_rows = []
    control = c["robustness_control"]
    if control is not None:
        clean = control["unperturbed_same_subset"]
        clean_name = "Unperturbed clean control, batch 1"
        robustness_reference = (
            "The clean reference is a supplementary batch-1 evaluation of the frozen released "
            "weights on the identical IDs. This matches the one-image-at-a-time perturbation "
            "pipeline; its artifact, weight, manifest and threshold identities are checked. "
            "The original batched clean reference remains unchanged in robustness.json as "
            "historical evidence. The companion robustness-control.json supplies the controlled "
            "baseline used for the deltas below. No test images or model/threshold changes "
            "are involved in this additional validation-only control."
        )
    else:
        clean = c["robustness"]["unperturbed_same_subset"]
        clean_name = "Unperturbed original batched reference"
        robustness_reference = (
            "The clean reference reuses predictions from the main batched validation pass, "
            "while perturbations are evaluated one image at a time. Batch-dependent numerical "
            "differences therefore confound the synthetic deltas; they do not isolate a "
            "perturbation-only causal effect."
        )
    for name, result in [
        (clean_name, clean),
        *c["robustness"]["variants"].items(),
    ]:
        robust_rows.append(
            [
                name,
                result["images"],
                number(result["map50"]),
                number(result["map50_95"]),
                number(result["map50_95"] - clean["map50_95"], 6),
                number(result["operating"]["precision"]),
                number(result["operating"]["recall"]),
            ]
        )
    bins = defaultdict(lambda: {"detections": 0, "score_sum": 0.0, "correct_sum": 0.0})
    for item in holdout["per_class"]:
        for bucket in item["confidence_reliability"]:
            count = bucket["detections"]
            if count:
                key = (bucket["lower_inclusive"], bucket["upper"], bucket["upper_inclusive"])
                bins[key]["detections"] += count
                bins[key]["score_sum"] += count * bucket["mean_score"]
                bins[key]["correct_sum"] += count * bucket["observed_precision_iou50"]
    confidence_rows = [
        [
            f"[{lo:.1f},{hi:.1f}{']' if inclusive else ')'}",
            item["detections"],
            number(item["score_sum"] / item["detections"]),
            number(item["correct_sum"] / item["detections"]),
        ]
        for (lo, hi, inclusive), item in sorted(bins.items())
    ]
    threshold_rows = [
        [
            row["threshold"],
            number(row["precision"]),
            number(row["recall"]),
            number(row["f1"]),
            row["tp"],
            row["fp"],
            row["fn"],
        ]
        for row in validation["threshold_sweep"]
    ]
    examples = [
        [
            item["image_id"],
            f"[Image](../{study}/holdout/{item['file']})",
            "; ".join(item["selection"]),
        ]
        for item in c["examples"]["examples"]
    ]
    diagnosis_rows = [
        [
            "AMP" if row["amp"] else "FP32",
            row["learning_rate"],
            row["batches"],
            row["nonfinite_forward_batches"],
            row["nonfinite_gradient_batches"],
            number(row["first_total_loss"]),
            number(row["last_total_loss"]),
        ]
        for row in c["diagnosis"]["summary"]
    ]
    return f"""# Error analysis

The final holdout contains **{holdout["images"]}** images. At the validation-selected
score threshold **{holdout["threshold"]}**, same-class IoU 0.50 matching produces
**{operating["tp"]} TP, {operating["fp"]} FP and {operating["fn"]} FN**;
micro precision/recall/F1 are **{number(operating["precision"])}/
{number(operating["recall"])}/{number(operating["f1"])}**. These measured errors
describe candidate ROI patches from this dataset, not a production defect rate.
Among {sum(row["ground_truth"] == 0 for row in errors)} negative holdout patches,
{sum(row["ground_truth"] == 0 and row["fp"] > 0 for row in errors)} have at least one
false-positive prediction at this threshold. This dataset-specific count does not
estimate an unfiltered production-line false-alarm rate.

## Full-split error accounting

{table(["Category", "Total", "Images with category", "Maximum per image", "Definition"], categories)}

Categories overlap: poor-localization candidates are false positives, and duplicate
or overlapping predictions may also be false positives. A wrong-class prediction
can generate both an FP and an FN. Counts of visually overlapping predictions are
diagnostic; overlaps alone do not establish an error. Invalid predictions rejected
before metric computation: **{holdout["invalid_predictions_rejected"]}**.
[Every image's counts](../{study}/holdout/per_image_errors.json) are retained.

{table(["Class, weakest AP first", "Images", "GT boxes", "AP50:.95", "TP", "FP", "FN", "Precision", "Recall"], per_class)}

Rare-class estimates depend on few samples, including only
{c["audit"]["class_image_counts"]["open"]["test"]} test images for `open`.
{uncertainty_text(holdout["uncertainty"])}

## Confusions and available slices

The separate confusion matrix matches class-agnostically at IoU 0.50, then records
truth rows and predicted columns, including a background/missed category. Its
counts have different semantics from same-class TP/FP/FN. Largest observed
class-to-class confusions are:

{confusion_text}

{table(["Image subset", "Images", "AP50", "AP50:.95", "Precision", "Recall"], slices)}

These are overlapping image subsets containing at least one box of the named
area; all labels in each image remain in the metric. They are not COCO object-area
AP and their results cannot be added together. Natural slice metadata is missing
for {", ".join(holdout["missing_slice_metadata"])}. Filename prefixes are unverified
source proxies, not confirmed physical board, factory or acquisition-session IDs.

## Confidence behavior and threshold choice

The score threshold maximizes recorded validation micro F1; it is frozen before
the final test evaluation. The validation tradeoff is:

{table(["Score threshold", "Precision", "Recall", "F1", "TP", "FP", "FN"], threshold_rows)}

Held-out detection scores across classes are binned below. Observed precision uses
the ranked same-class IoU 0.50 matching outcome and weights each retained detection
equally. Bins include low-score candidates below the operating threshold.

{table(["Score bin", "Detections", "Mean score", "Observed precision"], confidence_rows)}

This is a descriptive confidence analysis. No calibration fit or probability
guarantee is claimed; class imbalance, matching and retained-candidate selection
affect these observations. Per-class curves and histograms remain in
[the complete report](../{study}/holdout/report.json) and its
[plots](../{study}/holdout/precision_recall_and_scores.png).

## Synthetic robustness

The following comparisons all use the same **{len(c["robustness"]["images"])} validation images**:
the first sorted IDs, with unchanged labels and geometry. The reference is their
own clean-subset result, not full-validation performance. Perturbations are
brightness×0.7, Gaussian blur radius 1 and JPEG quality 50. They are artificial
stress tests rather than measured naturally occurring production conditions.
{robustness_reference}

{table(["Variant", "Images", "AP50", "AP50:.95", "ΔAP50:.95 vs clean", "Precision", "Recall"], robust_rows)}

## Qualitative sample selection

The evaluator takes up to 12 evenly spaced images across the entire sorted holdout,
then adds maximal observed TP, FP, FN, poor-localization-candidate and overlapping
prediction cases when the category is nonzero. Ties use a deterministic image-ID
rule. Reasons are preserved in [examples.json](../{study}/holdout/examples.json).
This combines broad coverage with explicit hard cases; it does not present only
successful predictions. Truth is green on the left, model predictions red on the
right, using the frozen operating threshold.

{table(["Image ID", "Rendered comparison", "Selection reason"], examples)}

## Failed experiment and diagnostic limits

The initial `rtx4060-study-v1` attempt used AMP and learning rate
{c["failed_run"]["config"]["learning_rate"]} and stopped before its first completed
epoch with a nonfinite forward loss. It produced no accuracy result and did not
access final holdout images. A bounded diagnostic replay used the same initialized
weights and training-only samples:

{table(["Precision", "Learning rate", "Batches", "Nonfinite forward batches", "Nonfinite-gradient batches", "First total loss", "Last total loss"], diagnosis_rows)}

The replay did not reproduce the original nonfinite forward failure. It changed
precision and learning rate together, so it cannot isolate which caused the
stability difference. Fifty batches per recipe do not establish full-run stability.
The separately named FP32 study has its own completed histories and held-out
results; the failed attempt is retained as a negative finding. See
[diagnostic evidence](../reports/training-diagnosis.json) and
[initial failure record](../reports/rtx4060-study-v1/failure.json).

Generated deterministically from recorded evidence by `scripts/summarize_study.py`.
"""


def reproducibility(context: dict) -> str:
    c = context
    selected, metadata = c["selected"], c["metadata"]
    study, release = c["study"].as_posix(), c["release"].as_posix()
    run = next(row for row in c["runs"] if row["run_id"] == selected["run_id"])
    environment = table(
        ["Recorded field", "Value"],
        [[key, value] for key, value in sorted(run["environment"].items())],
    )
    configs = []
    for candidate_run in c["runs"]:
        config = candidate_run["config"]
        configs.append(
            [
                candidate_run["run_id"],
                config["seed"],
                config["epochs"],
                config["batch_size"],
                config["learning_rate"],
                config["horizontal_flip_probability"],
                config["amp"],
                number(candidate_run["elapsed_seconds"] / 60, 2),
            ]
        )
    source_rows = [
        [
            row["run_id"],
            row["recorded_revision"],
            row["recorded_dirty"],
            row["package_source_sha256"],
            row["retained_exact_source_commit"],
        ]
        for row in c["sources"]
    ]
    digests = table(
        ["Evidence path", "SHA256"],
        [[path, digest] for path, digest in sorted(c["evidence"].files.items())],
    )
    acquisition = c["acquisition"]
    return f"""# Reproducibility report

The released bundle `{metadata["model_version"]}` is tied to completed local GPU
experiments, validation-based selection, one final holdout evaluation, CPU/GPU
benchmarks and recorded provenance. This report is generated from actual JSON
evidence and rejects incomplete runs or mismatched dataset, checkpoint, threshold,
taxonomy, artifact and report identities.

## Recreate the environment and acquire data

Python 3.12 is supported. The demonstrated Windows training environment was:

{environment}

From the repository root on Windows:

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-gpu.txt
.venv-gpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-gpu/Scripts/python.exe scripts/acquire_data.py
.venv-gpu/Scripts/python.exe -m inspection_dl.data --root data/raw/pcb-ind --output data/manifests/reproduced.json --seed 42
.venv-gpu/Scripts/python.exe scripts/crosscheck_annotations.py
```

`requirements-gpu.txt` snapshots the demonstrated Windows CUDA environment.
`requirements-cpu.txt` provides fully hash-pinned serving/CI dependencies.
Linux CPU setup uses `python3.12 -m venv` and `.venv/bin/python`;
see [deployment setup](DEPLOYMENT.md) for platform-specific instructions.
With uv, both official indexes must be considered using
`--index-strategy unsafe-best-match` when installing the requirements snapshots.

Acquisition downloads only the pinned author archive from
{acquisition["record"]}, checks size {acquisition["archive_bytes"]} bytes,
MD5 `{acquisition["archive_md5"]}` and
SHA256 `{acquisition["archive_sha256"]}`, and extracts the YOLO copy and metadata.
Raw data remains outside Git. Verify the reproduced canonical manifest digest
against **`{c["manifest"]["manifest_digest"]}`**. JSON file SHA256 is a separate
byte-level identity, while canonical content hashing ignores formatting.
Frozen outputs refuse conflicting overwrites.
Git attributes preserve checksummed JSON bytes across Windows/Linux checkouts.
A discovered CRLF-to-LF Git normalization mismatch was corrected without changing
the frozen dataset content or bytes used by training; the retained
[portability check](../reports/validation/manifest-portability.json) confirms that
the committed blob and working manifest have the same training-bound file SHA256.

The independent source-format audit checks original COCO/YOLO class and coordinate
semantics, all accepted image and annotation hashes, decoded dimensions and EXIF.
It reports **{c["annotation_check"]["successfully_compared_images"]}** accepted
images with status `{c["annotation_check"]["status"]}`. The metric-reference report
cross-checks supported no-crowd AP behavior against pycocotools using explicitly
synthetic fixtures; it is not evidence of trained accuracy.

## Reproduce bounded training and selection

```powershell
.venv-gpu/Scripts/python.exe scripts/run_study.py --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --epochs {c["preregistration"]["epochs_per_run"]} --batch-size {c["preregistration"]["batch_size"]} --device cuda --learning-rate {c["preregistration"]["learning_rate"]} --study-id reproduced-study --release reproduced-model
```

Fresh study/release IDs preserve existing evidence. The runner performs both
augmentation variants, selects the best validation AP50:.95 checkpoint, selects a
validation micro-F1 threshold, writes the decision, evaluates the final holdout,
packages a bundle, and benchmarks CPU/GPU batch 1 versus 4 on an identical subset.
Before selection, it verifies exact initial parameter/buffer equality and records
`initialization-parity.json`, including tensor count and an initial-tensor digest.
Individual commands and epoch-boundary resume are documented in [TRAINING.md](TRAINING.md).

{table(["Run", "Seed", "Epochs", "Batch", "LR", "Horizontal flip p", "AMP", "Elapsed min"], configs)}

AdamW, weight decay 0.0005 and cosine scheduling are recorded in each run's config.
BatchNorm running statistics remain frozen; gradients are clipped to norm 10.
Checkpoints retain model, optimizer, scheduler, scaler and RNG states. Resume
requires the original configuration, epoch budget, run directory and best/last
checkpoints. Training was local and single-GPU; there is no cloud dependency.
Seeds and deterministic CUDNN settings reduce variability but do not promise
bitwise agreement across CUDA versions, operating systems, devices or libraries.
One training seed does not establish seed variability or generalization certainty.

## Source identity and immutable evidence

The package hash uses sorted direct `src/inspection_dl/*.py` files, appending
each UTF8 basename, NUL, exact file bytes, and NUL into SHA256. Some runs began
with a dirty working tree, so their launch revision alone does not identify the
actual source. The generator independently reconstructs package bytes from
retained Git commits and requires exact recorded-hash matches:

{table(["Run", "Recorded launch revision", "Dirty at launch", "Actual package hash", "Retained exact matching commit"], source_rows)}

Training/model/metric modules are compared by Python AST across the retained
snapshots and must agree, allowing formatting-only changes. Other modules may
change between runs without changing training computation; the complete source
identities remain explicit. Source reconstruction and every documentation input
digest are retained in `{study}/documentation-inputs.json`.

Selected checkpoint SHA256: `{selected["checkpoint_sha256"]}`.
Released weights SHA256: `{metadata["weights_sha256"]}`.
Artifact digest: `{metadata["artifact_digest"]}`.
The checkpoint contains training state; the released weights are a separate
tensor-only state dictionary, so their distinct hashes are expected.
Bundle metadata also pins preprocessing, class mapping, thresholds, runtime,
dataset manifest and evaluation digest. Loading validates identity and rejects
missing, incompatible or corrupted files before serving.

## Standalone inference and evaluation commands

```powershell
py -3.12 -m venv .venv-cpu
.venv-cpu/Scripts/python.exe -m pip install -r requirements-cpu.txt
.venv-cpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-cpu/Scripts/python.exe -m inspection_dl.api --bundle {release} --device cpu
```

The released bundle runs without raw data, training directories or downloaded
pretrained initialization weights. It serves localhost by default. Submit the
attributed sample image from a second terminal:

```powershell
curl.exe --fail-with-body -H "Content-Type: image/jpeg" --data-binary "@tests/fixtures/pcb-ind/5810_t_007.jpg" http://127.0.0.1:8000/v1/predict
```

To repeat evaluation from a fresh clone, use the released bundle and acquired
dataset directly, with a fresh output directory. This CPU validation-split recheck
does not require any ignored training checkpoint or optimizer state:

```powershell
.venv-cpu/Scripts/python.exe -m inspection_dl.evaluate --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --bundle {release} --output reports/release-validation-recheck --split val --device cpu
.venv-gpu/Scripts/python.exe -m inspection_dl.benchmark --bundle {release} --manifest data/manifests/pcb-ind-v4.json --data-root data/raw/pcb-ind --output reports/recomputed-benchmark.json --samples 100 --devices cpu cuda --batch-sizes 1 4
```

Bundle evaluation uses the release's recorded confidence threshold by default
and reports its artifact digest, weights hash and model version as the evaluated
identity. The original training-checkpoint digest remains historical provenance;
it is not presented as an input file when evaluating a bundle. Evaluation reports
also record the evaluator's actual Git revision and source-package hash separately
from the training source. The original study's final holdout remains its frozen
report; this repeatability command evaluates validation images.
Full training is reproducible through the fresh-study command. Raw data and
optimizer checkpoints are ignored by Git; the trained release weights are retained
for direct inference. GPU training, full-data evaluation and local benchmarks
were measured separately from ordinary CPU CI.

## Validation and disclosure

```powershell
.venv-cpu/Scripts/python.exe -m pytest
.venv-cpu/Scripts/python.exe -m ruff check .
.venv-cpu/Scripts/python.exe -m ruff format --check .
python scripts/summarize_study.py --study {study} --release {release}
```

Tests use generated synthetic fixtures and one attributed real image/label pair;
they do not repeat full GPU training. See the committed CI workflow and validation
records for actual checks. The failed initial AMP/0.001 study and confounded
FP32/0.0003 diagnostic replay remain disclosed in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).
Natural acquisition metadata is absent, source grouping is a heuristic, and rare
classes limit inference strength. No production SLA, calibrated probability,
manufacturing deployment or safety-critical claim follows from these results.

The source code is MIT, PCB-IND data is CC BY 4.0, and pretrained/COCO terms remain
separate. Attribution and model-use limitations are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md), release metadata and licensing JSON.

## Documentation input checksums

{digests}

Generated deterministically without current timestamps or external requests by
`scripts/summarize_study.py`. Regeneration requires completed matching evidence;
missing results never become placeholder success claims.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study", type=Path, default=Path("reports/rtx4060-study-v2"))
    parser.add_argument("--release", type=Path, default=Path("releases/pcb-ind-ssdlite-v1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pcb-ind-v4.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    try:
        context = load_study(args)
        for path in (
            "scripts/summarize_study.py",
            "requirements-cpu.txt",
            "requirements-gpu.txt",
            "requirements-cpu.lock",
            "requirements-gpu.lock",
            "pyproject.toml",
            ".gitattributes",
        ):
            context["evidence"].include(path)
        documents = {
            "MODEL_CARD.md": model_card(context),
            "ERROR_ANALYSIS.md": error_analysis(context),
            "REPRODUCIBILITY.md": reproducibility(context),
        }
        output = context["evidence"].path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for filename, content in documents.items():
            (output / filename).write_bytes(content.encode("utf-8"))
        inputs = {
            "generator": "scripts/summarize_study.py",
            "deterministic": True,
            "input_files_sha256": dict(sorted(context["evidence"].files.items())),
            "source_reconstruction": context["sources"],
            "document_sha256": {name: sha256(output / name) for name in documents},
        }
        destination = context["evidence"].path(context["study"] / "documentation-inputs.json")
        destination.write_bytes(
            (json.dumps(inputs, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        )
        print(
            json.dumps(
                {
                    "documents": list(documents),
                    "inputs": str(destination.relative_to(args.repo_root)),
                },
                indent=2,
            )
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Documentation generation stopped: {error}\n")


if __name__ == "__main__":
    main()
