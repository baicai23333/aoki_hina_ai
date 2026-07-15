#!/usr/bin/env python3
"""Create a lossless XTTS reference without modifying the source recording."""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "new.wav")
    parser.add_argument(
        "--segment",
        action="append",
        default=None,
        help="Reference range as START:END seconds; may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tts_xtts" / "refs_clean",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference source not found: {source}")
    raw_segments = args.segment or ["28.944:38.034"]
    segments = []
    for raw_segment in raw_segments:
        try:
            start_text, end_text = raw_segment.split(":", 1)
            start, end = float(start_text), float(end_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference segment: {raw_segment}") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Invalid reference segment: {raw_segment}")
        segments.append((start, end))

    audio, source_rate = sf.read(source, always_2d=True, dtype="float32")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_paths = []
    for index, (start, end) in enumerate(segments, start=1):
        start_frame = round(start * source_rate)
        end_frame = round(end * source_rate)
        if end_frame > len(audio):
            raise ValueError("Reference range exceeds the source duration.")

        # The source is dual-mono PCM. Downmix once and resample once to XTTS's
        # conditioning rate. No denoising, compression, or gain is applied.
        mono = audio[start_frame:end_frame].mean(axis=1)
        clean = librosa.resample(
            mono,
            orig_sr=source_rate,
            target_sr=22_050,
            res_type="soxr_hq",
        )
        fade_frames = min(round(0.02 * 22_050), len(clean) // 2)
        if fade_frames:
            fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
            clean[:fade_frames] *= fade
            clean[-fade_frames:] *= fade[::-1]
        reference_path = args.output_dir / f"reference_clean_{index:02d}.wav"
        sf.write(reference_path, clean, 22_050, subtype="PCM_16")
        reference_paths.append(reference_path)

    manifest_path = args.output_dir / "refs_manifest.json"
    manifest = {
        "refs": [str(path.resolve()) for path in reference_paths],
        "source": str(source),
        "source_sha256": file_sha256(source),
        "segments": [{"start": start, "end": end} for start, end in segments],
        "processing": {
            "channels": 1,
            "sample_rate": 22_050,
            "resampler": "soxr_hq",
            "fade_ms": 20,
            "denoise": False,
            "gain_db": 0.0,
            "lossy_encoding": False,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
