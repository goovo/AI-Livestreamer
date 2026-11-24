from __future__ import annotations

import logging
import os
import subprocess

import numpy as np

log = logging.getLogger("goovo")

SAMPLE_RATE = 44100
CHANNELS = 1
BYTES_PER_SAMPLE = 2


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def decode_to_pcm(audio: bytes) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=audio,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"音频解码失败: {result.stderr.decode(errors='ignore')[:400]}")
    return normalize_pcm(result.stdout)


def normalize_pcm(pcm: bytes, target_rms: float = 0.08, peak_limit: float = 0.9) -> bytes:
    if not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    if rms < 1e-6 or peak < 1e-6:
        return pcm
    gain = target_rms / rms
    if peak * gain > peak_limit:
        gain = peak_limit / peak
    if abs(gain - 1.0) < 0.01:
        return pcm
    samples = np.clip(samples * gain, -1.0, 1.0)
    return (samples * 32767).astype(np.int16).tobytes()


def pcm_to_wav(pcm: bytes) -> bytes:
    import wave
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    return data if len(data) == size else b""
