"""Acquire the pinned author release of PCB-IND; data never enters Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/19723114/files/PCB-IND_v4.zip/content"
EXPECTED_MD5 = "1325f8dffeb73bef4db34fbd22b31ec5"
EXPECTED_BYTES = 101_336_845
EXPECTED_SHA256 = "29cf919c64cab7e60e90e16bf3f42cce674688f99295d66a8e10d0066836e3c9"


def acquire(dest: Path, archive: Path | None = None) -> dict:
    dest = dest.resolve()
    archive = archive or dest.parent / "PCB-IND_v4.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        total = 0
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "visual-inspection-dl/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1 << 20):
                    total += len(chunk)
                    if total > EXPECTED_BYTES:
                        raise ValueError("Dataset archive exceeds the pinned release size")
                    out.write(chunk)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    payload = archive.read_bytes()
    if (len(payload) != EXPECTED_BYTES or hashlib.md5(payload).hexdigest() != EXPECTED_MD5
            or hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256):
        raise ValueError("Dataset archive size/checksum mismatch; remove it and retry acquisition")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if sum(item.file_size for item in members) > 1_000_000_000:
            raise ValueError("Dataset uncompressed size exceeds 1 GB")
        for item in members:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename:
                raise ValueError("Unsafe archive member")
            if name.parts and ":" in name.parts[0]:
                raise ValueError("Unsafe archive drive prefix")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are unsupported")
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError("Archive path escapes extraction root")
        dest.mkdir(parents=True, exist_ok=True)
        selected = [item for item in members
                    if item.filename.startswith("YOLO/")
                    or item.filename in {"classes.json", "README.md"}]
        source.extractall(dest, members=selected)
    result = {
        "source": URL,
        "record": "https://zenodo.org/records/19723114",
        "license": "CC-BY-4.0",
        "author": "Hao Yan, Fujian Polytechnic Normal University",
        "archive_bytes": len(payload),
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (dest / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/pcb-ind"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(acquire(args.dest, args.archive), indent=2))
