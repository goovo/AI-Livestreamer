"""用 whisper 为已缓存的口播音频生成字级时间戳（需 .venv-clone 环境）。

运行：.venv-clone/bin/python tools/align_cache.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from goovo_human.config import load_config
from goovo_human.tts import TTS
from goovo_human.util import pcm_to_wav


def is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def main() -> int:
    from transformers import pipeline

    config = load_config(ROOT / "config.toml")
    tts = TTS(config.voice, ROOT / ".cache" / "tts")
    texts: list[str] = []
    for path in sorted((ROOT / config.brain.scripts_dir).glob("*.md")):
        for chunk in path.read_text(encoding="utf-8").split("\n---"):
            text = " ".join(line.strip() for line in chunk.splitlines() if line.strip()).replace("#", "").strip()
            if text:
                texts.append(text)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    for text in texts:
        pcm_path = tts._cache_path(text)
        if not pcm_path.exists():
            continue
        wav_path = pcm_path.with_suffix(".wav")
        wav_path.write_bytes(pcm_to_wav(pcm_path.read_bytes()))
        result = asr(str(wav_path), return_timestamps="word", generate_kwargs={"task": "transcribe", "language": "zh"})
        chars = [char for char in text if is_chinese(char)]
        boundaries: list[dict] = []
        index = 0
        last_end = 0.0
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            if timestamp[0] is None or timestamp[1] is None:
                continue
            count = sum(is_chinese(char) for char in chunk.get("text", ""))
            if count <= 0:
                continue
            start, end = float(timestamp[0]), float(timestamp[1])
            last_end = end
            span = (end - start) / count
            for offset in range(count):
                if index >= len(chars):
                    break
                boundaries.append({
                    "text": chars[index],
                    "offset_ms": (start + offset * span) * 1000,
                    "duration_ms": span * 1000,
                })
                index += 1
        duration = pcm_path.stat().st_size / 88200
        if index < len(chars):
            span = (duration - last_end) / max(1, len(chars) - index)
            for offset in range(len(chars) - index):
                boundaries.append({
                    "text": chars[index],
                    "offset_ms": (last_end + offset * span) * 1000,
                    "duration_ms": span * 1000,
                })
                index += 1
        pcm_path.with_suffix(".json").write_text(json.dumps(boundaries, ensure_ascii=False), encoding="utf-8")
        wav_path.unlink(missing_ok=True)
        print(f"✓ {pcm_path.name}: {len(boundaries)} 字级时间戳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
