from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _extract_loudnorm_json(text: str) -> Optional[Dict]:
    # ffmpeg prints JSON on stderr; find the last JSON block
    m = re.findall(r"\{[\s\S]*?\}", text)
    for s in reversed(m):
        try:
            d = json.loads(s)
            if all(k in d for k in ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")):
                return d
        except Exception:
            continue
    return None


def normalize_file(src: Path, dst: Path, I: float = -16.0, TP: float = -1.0, LRA: float = 11.0, bitrate: str = "192k") -> None:
    # First pass: measure
    pass1 = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-filter:a",
        f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    r1 = _run_ffmpeg(pass1)
    meas = _extract_loudnorm_json(r1.stderr)
    # Second pass: apply
    if meas:
        ln = (
            f"loudnorm=I={I}:TP={TP}:LRA={LRA}:"
            f"measured_I={meas['input_i']}:measured_LRA={meas['input_lra']}:"
            f"measured_TP={meas['input_tp']}:measured_thresh={meas['input_thresh']}:"
            f"offset={meas['target_offset']}:linear=true:print_format=summary"
        )
    else:
        ln = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
    pass2 = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-filter:a",
        ln,
        "-b:a",
        bitrate,
        str(dst),
    ]
    r2 = _run_ffmpeg(pass2)
    if r2.returncode != 0:
        # Retry without forcing bitrate/coder
        pass2_fallback = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-filter:a",
            ln,
            str(dst),
        ]
        r2b = _run_ffmpeg(pass2_fallback)
        if r2b.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {src.name}: {r2.stderr}\n{r2b.stderr}")


def normalize_dir(trimmed_dir: Path, backup_dir: Path, I: float = -16.0, TP: float = -1.0, LRA: float = 11.0, bitrate: str = "192k") -> None:
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    mp3s = sorted([p for p in trimmed_dir.iterdir() if p.suffix.lower() == ".mp3"])
    if not mp3s:
        print(f"No mp3 files found in {trimmed_dir}")
        return
    print(f"Normalizing {len(mp3s)} files to I={I} LUFS, TP={TP} dB, LRA={LRA}...")
    for src in mp3s:
        dst_tmp = trimmed_dir / (src.stem + ".normalized.tmp.mp3")
        bak = backup_dir / src.name
        print(f"- {src.name}")
        shutil.copy2(src, bak)
        try:
            normalize_file(bak, dst_tmp, I=I, TP=TP, LRA=LRA, bitrate=bitrate)
            # Overwrite original
            shutil.move(str(dst_tmp), str(src))
        finally:
            if dst_tmp.exists():
                dst_tmp.unlink(missing_ok=True)


def main(argv: List[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Normalize mp3 loudness in a directory (backs up originals)")
    parser.add_argument("dir", nargs="?", default="assets/data/sound/trimmed")
    parser.add_argument("--backup", default="assets/data/sound/backup_originals")
    parser.add_argument("--lufs", type=float, default=-16.0)
    parser.add_argument("--tp", type=float, default=-1.0)
    parser.add_argument("--lra", type=float, default=11.0)
    parser.add_argument("--bitrate", default="192k")
    args = parser.parse_args(argv)
    normalize_dir(Path(args.dir), Path(args.backup), I=args.lufs, TP=args.tp, LRA=args.lra, bitrate=args.bitrate)


if __name__ == "__main__":
    main()

