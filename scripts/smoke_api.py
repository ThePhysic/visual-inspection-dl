"""Capture a real-model response from an already running local inference service."""

import argparse
import json
import urllib.request
from pathlib import Path

from inspection_dl.schemas import Prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        args.image.suffix.lower()
    )
    if media_type is None:
        parser.error("image must have .jpg, .jpeg, or .png extension")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/predict",
        data=args.image.read_bytes(),
        headers={"Content-Type": media_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        prediction = Prediction.model_validate_json(response.read())
    if prediction.simulated or prediction.backend != "trained":
        raise RuntimeError(
            "Smoke evidence requires real trained weights, received a fixture backend"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prediction.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Captured {len(prediction.detections)} detections from {prediction.model_version} to {args.output}"
    )


if __name__ == "__main__":
    main()
