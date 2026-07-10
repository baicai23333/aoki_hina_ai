#!/usr/bin/env python3
import json
import math
import wave
from pathlib import Path
import audioop

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_WAV = PROJECT_ROOT / "tts_xtts" / "raw" / "new.wav"
REFS_DIR = PROJECT_ROOT / "tts_xtts" / "refs"
MANIFEST_PATH = REFS_DIR / "refs_manifest.json"

TARGET_COUNT = 6
MIN_SEC = 6.0
MAX_SEC = 12.0
WINDOW_SEC = 10.0
CHUNK_SEC = 0.1
SILENCE_DB = -35.0
MIN_SILENCE_SEC = 0.4


def dbfs(rms: int, max_possible: float) -> float:
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / max_possible)


def load_audio_bytes(path: Path):
    with wave.open(str(path), "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    return params, frames


def detect_voiced_segments(params, frames: bytes):
    sr = params.framerate
    sampwidth = params.sampwidth
    n_channels = params.nchannels

    frame_size = sampwidth * n_channels
    chunk_frames = max(1, int(sr * CHUNK_SEC))
    chunk_bytes = chunk_frames * frame_size

    max_possible = float(2 ** (8 * sampwidth - 1))

    segments = []
    in_voiced = False
    start_time = 0.0
    silence_run = 0
    min_silence_chunks = max(1, int(MIN_SILENCE_SEC / CHUNK_SEC))

    total_chunks = math.ceil(len(frames) / chunk_bytes)
    for idx in range(total_chunks):
        chunk = frames[idx * chunk_bytes:(idx + 1) * chunk_bytes]
        if not chunk:
            break
        rms = audioop.rms(chunk, sampwidth)
        chunk_db = dbfs(rms, max_possible)
        voiced = chunk_db > SILENCE_DB
        time_pos = idx * CHUNK_SEC

        if voiced and not in_voiced:
            in_voiced = True
            start_time = time_pos
            silence_run = 0
        elif not voiced and in_voiced:
            silence_run += 1
            if silence_run >= min_silence_chunks:
                end_time = max(start_time, time_pos - silence_run * CHUNK_SEC)
                if end_time > start_time:
                    segments.append((start_time, end_time))
                in_voiced = False
                silence_run = 0
        elif voiced and in_voiced:
            silence_run = 0

    if in_voiced:
        total_sec = len(frames) / frame_size / sr
        segments.append((start_time, total_sec))

    return segments


def split_segments(segments):
    results = []
    for start, end in segments:
        duration = end - start
        if duration < MIN_SEC:
            continue
        if duration <= MAX_SEC:
            results.append((start, end))
        else:
            step = min(5.0, MAX_SEC / 2)
            cursor = start
            while cursor + MIN_SEC <= end:
                window = min(WINDOW_SEC, end - cursor)
                window = min(window, MAX_SEC)
                if window >= MIN_SEC:
                    results.append((cursor, cursor + window))
                cursor += step
        if len(results) >= TARGET_COUNT:
            break
    return results


def fallback_segments(total_sec: float):
    window = min(WINDOW_SEC, max(MIN_SEC, total_sec))
    max_start = max(0.0, total_sec - window)
    if TARGET_COUNT == 1:
        starts = [0.0]
    elif max_start == 0.0:
        starts = [0.0] * TARGET_COUNT
    else:
        step = max_start / (TARGET_COUNT - 1)
        starts = [i * step for i in range(TARGET_COUNT)]
    return [(s, s + window) for s in starts]


def slice_frames(params, frames: bytes, start_sec: float, end_sec: float) -> bytes:
    sr = params.framerate
    frame_size = params.sampwidth * params.nchannels
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)
    total_frames = len(frames) // frame_size

    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(0, min(end_frame, total_frames))

    chunk = frames[start_frame * frame_size:end_frame * frame_size]
    needed_frames = int((end_sec - start_sec) * sr)
    current_frames = len(chunk) // frame_size
    if current_frames < needed_frames:
        pad_frames = needed_frames - current_frames
        chunk += b"\x00" * (pad_frames * frame_size)
    return chunk


def write_wav(path: Path, params, frames: bytes):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(frames)


def main():
    if not RAW_WAV.exists():
        raise FileNotFoundError(f"Missing input wav: {RAW_WAV}")

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    params, frames = load_audio_bytes(RAW_WAV)
    total_sec = len(frames) / (params.sampwidth * params.nchannels) / params.framerate

    segments = detect_voiced_segments(params, frames)
    candidates = split_segments(segments)
    if len(candidates) < TARGET_COUNT:
        candidates = fallback_segments(total_sec)

    candidates = candidates[:TARGET_COUNT]

    manifest = {"refs": [], "segments": []}
    for idx, (start, end) in enumerate(candidates, start=1):
        ref_name = f"ref_{idx:03d}.wav"
        ref_path = REFS_DIR / ref_name
        ref_frames = slice_frames(params, frames, start, end)
        write_wav(ref_path, params, ref_frames)
        manifest["refs"].append(str(ref_path))
        manifest["segments"].append({"file": ref_name, "start": start, "end": end})

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(candidates)} refs at {REFS_DIR}")


if __name__ == "__main__":
    main()
