import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from inspection_dl.model import build_detector
from inspection_dl.train import DetectionDataset, collate, load_manifest

torch.set_num_threads(4)
manifest = load_manifest("data/manifests/pcb-ind-v4.json")
records = [r for r in manifest["records"] if r["split"] == "train"]
checkpoint = torch.load(
    "runs/rtx4060-study-v1-no-flip/initialized.pt", weights_only=True, map_location="cpu"
)
results = []
start = time.perf_counter()
for amp, lr in [(True, 0.001), (False, 0.0003)]:
    model = build_detector(9).cuda()
    model.load_state_dict(checkpoint["model"])
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(
        DetectionDataset(records, "data/raw/pcb-ind-ready"),
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=collate,
    )
    for index, (images, targets, ids) in enumerate(loader):
        if index >= 50 or time.perf_counter() - start > 55:
            break
        images = [im.cuda() for im in images]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            losses = model(images, targets)
            total = sum(losses.values())
        row = {
            "amp": amp,
            "lr": lr,
            "batch": index,
            "ids": ids,
            "loss": {
                k: float(v.detach()) if torch.isfinite(v) else str(float(v.detach()))
                for k, v in losses.items()
            },
        }
        if not torch.isfinite(total):
            row["failed"] = True
            results.append(row)
            print(json.dumps(row), flush=True)
            break
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=not amp)
        row["gradient_norm"] = float(norm) if torch.isfinite(norm) else str(float(norm))
        scaler.step(optimizer)
        scaler.update()
        row["scaler"] = scaler.get_scale()
        results.append(row)
        print(json.dumps(row), flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
Path("reports/training-diagnosis.json").write_text(
    json.dumps(
        {
            "definition": "bounded replay of failed-run initialized model and seed42 train sampling; no validation or test accessed",
            "results": results,
            "seconds": time.perf_counter() - start,
        },
        indent=2,
    )
    + "\n"
)
