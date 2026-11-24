from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx

from .config import VoiceConfig
from .util import decode_to_pcm

log = logging.getLogger("goovo.tts")


class TTSError(RuntimeError):
    pass


class TTS:
    def __init__(self, config: VoiceConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        key = json.dumps(
            [self.config.engine, self.config.voice, self.config.rate, self.config.volume, text],
            ensure_ascii=False,
        )
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.pcm"

    def synthesize(self, text: str) -> tuple[bytes, list | None]:
        text = text.strip()
        if not text:
            raise TTSError("空文本")
        cached = self._cache_path(text)
        if cached.exists():
            sidecar = cached.with_suffix(".json")
            boundaries = json.loads(sidecar.read_text()) if sidecar.exists() else None
            return cached.read_bytes(), boundaries
        try:
            pcm, boundaries = self._synthesize_engine(text)
        except Exception as exc:
            if self.config.engine != "macos" and self.config.fallback_macos:
                log.warning("TTS 引擎 %s 失败(%s)，回退到 edge 男声", self.config.engine, exc)
                try:
                    pcm, boundaries = self._synthesize_edge(text, self.config.fallback_voice)
                except Exception:
                    log.warning("edge 回退也失败，使用 macOS say")
                    pcm, boundaries = self._synthesize_macos(text)
            else:
                raise TTSError(str(exc)) from exc
        if not pcm:
            raise TTSError("合成结果为空")
        cached.write_bytes(pcm)
        cached.with_suffix(".json").write_text(json.dumps(boundaries, ensure_ascii=False))
        return pcm, boundaries

    def _synthesize_engine(self, text: str) -> tuple[bytes, list | None]:
        if self.config.engine == "edge":
            return self._synthesize_edge(text)
        if self.config.engine == "macos":
            return self._synthesize_macos(text)
        if self.config.engine == "http":
            return self._synthesize_http(text)
        raise TTSError(f"未知 TTS 引擎: {self.config.engine}")

    def _synthesize_edge(self, text: str, voice: str | None = None) -> tuple[bytes, list | None]:
        import edge_tts

        async def run() -> tuple[bytes, list]:
            communicate = edge_tts.Communicate(
                text,
                voice or self.config.voice,
                rate=self.config.rate,
                volume=self.config.volume,
            )
            audio = bytearray()
            boundaries = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    boundaries.append({
                        "text": chunk.get("text", ""),
                        "offset_ms": chunk.get("offset", 0) / 10000,
                        "duration_ms": chunk.get("duration", 0) / 10000,
                    })
            return bytes(audio), boundaries

        audio, boundaries = asyncio.run(run())
        return decode_to_pcm(audio), boundaries

    def _synthesize_macos(self, text: str) -> tuple[bytes, None]:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as handle:
            tmp_path = handle.name
        try:
            subprocess.run(
                ["say", "-v", self.config.macos_voice, "-o", tmp_path, text],
                check=True,
                capture_output=True,
            )
            return decode_to_pcm(Path(tmp_path).read_bytes()), None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _synthesize_http(self, text: str) -> tuple[bytes, None]:
        http = self.config.http
        body = http.json_body.replace("{text}", json.dumps(text, ensure_ascii=False)[1:-1])
        response = httpx.request(
            http.method,
            http.url,
            content=body.encode(),
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        response.raise_for_status()
        return decode_to_pcm(response.content), None
