from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import BrainConfig
from .tts import TTS

log = logging.getLogger("goovo.brain")


@dataclass
class Segment:
    text: str
    pcm: bytes
    source: str
    boundaries: list | None = None


class Brain:
    """口播大脑：维护稿件循环 / LLM 撰稿，预合成音频，供播放线程取用。"""

    def __init__(self, config: BrainConfig, tts: TTS, root: Path):
        self.config = config
        self.tts = tts
        self.root = root
        self.mode = config.mode
        self.paused = False
        self.stop_event = threading.Event()
        self.interrupt_event = threading.Event()
        self._ready: deque[Segment] = deque()
        self._priority: deque[Segment] = deque()
        self._lock = threading.Lock()
        self._playlist: list[str] = []
        self._playlist_index = 0
        self._topic_index = 0
        self.history: deque[dict] = deque(maxlen=60)
        self.current_text = ""
        self.last_error = ""
        self._load_playlist()
        self._producer = threading.Thread(target=self._produce_loop, name="brain-producer", daemon=True)

    def start(self) -> None:
        self._producer.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _load_playlist(self) -> None:
        directory = self.root / self.config.scripts_dir
        segments: list[str] = []
        if directory.exists():
            for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.txt")):
                content = path.read_text(encoding="utf-8")
                for chunk in content.split("\n---"):
                    text = " ".join(line.strip() for line in chunk.splitlines() if line.strip())
                    text = text.replace("#", "").strip()
                    if text:
                        segments.append(text)
        self._playlist = segments
        if not segments:
            log.warning("scripts/ 目录没有口播稿，将使用内置开场白")
            self._playlist = ["大家好，欢迎来到直播间！请在 scripts 目录放入口播稿，或在控制面板直接输入想说的话。"]

    def _next_playlist_text(self) -> str:
        if self.config.shuffle:
            return random.choice(self._playlist)
        text = self._playlist[self._playlist_index % len(self._playlist)]
        self._playlist_index += 1
        return text

    def _next_llm_text(self) -> str:
        llm = self.config.llm
        api_key = os.environ.get(llm.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"未设置环境变量 {llm.api_key_env}，无法使用 LLM 撰稿")
        topic = self.config.topics[self._topic_index % len(self.config.topics)] if self.config.topics else "自由发挥"
        self._topic_index += 1
        recent = "；".join(item["text"][:40] for item in list(self.history)[-5:]) or "（刚开始）"
        prompt = (
            f"当前主题：{topic}。最近讲过的内容：{recent}。"
            f"请生成下一段直播口播稿，不超过{llm.max_chars}字，口语化、自然、有互动感，"
            "不要重复最近内容，不要使用标题、列表或表情符号，直接输出正文。"
        )
        response = httpx.post(
            f"{llm.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": llm.model,
                "temperature": llm.temperature,
                "messages": [
                    {"role": "system", "content": self.config.persona},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=45.0,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return " ".join(text.split())

    def _next_text(self) -> tuple[str, str]:
        if self.mode == "llm":
            try:
                return self._next_llm_text(), "llm"
            except Exception as exc:
                self.last_error = f"LLM 撰稿失败: {exc}"
                log.warning("%s，本段回退到口播稿", self.last_error)
        return self._next_playlist_text(), "playlist"

    def _produce_loop(self) -> None:
        while not self.stop_event.is_set():
            with self._lock:
                ready_count = len(self._ready)
            if ready_count >= max(1, self.config.prefetch):
                self.stop_event.wait(0.3)
                continue
            try:
                text, source = self._next_text()
                pcm, boundaries = self.tts.synthesize(text)
                with self._lock:
                    self._ready.append(Segment(text=text, pcm=pcm, source=source, boundaries=boundaries))
                log.info("已备稿(%s, %.1fs): %s", source, len(pcm) / 88200, text[:40])
            except Exception as exc:
                self.last_error = str(exc)
                log.error("备稿失败: %s", exc)
                self.stop_event.wait(5.0)

    def next_segment(self) -> Segment | None:
        with self._lock:
            if self._priority:
                return self._priority.popleft()
            if self.paused or not self._ready:
                return None
            return self._ready.popleft()

    def playlist_texts(self) -> list[str]:
        return list(self._playlist)

    def say(self, text: str, interrupt: bool = False) -> None:
        text = text.strip()
        if not text:
            return

        def worker() -> None:
            try:
                pcm, boundaries = self.tts.synthesize(text)
            except Exception as exc:
                self.last_error = f"插播合成失败: {exc}"
                log.error(self.last_error)
                return
            with self._lock:
                self._priority.append(Segment(text=text, pcm=pcm, source="manual", boundaries=boundaries))
            if interrupt:
                self.interrupt_event.set()

        threading.Thread(target=worker, name="brain-say", daemon=True).start()

    def mark_started(self, segment: Segment) -> None:
        self.current_text = segment.text
        self.history.append({"text": segment.text, "source": segment.source, "at": time.strftime("%H:%M:%S")})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "paused": self.paused,
                "current": self.current_text,
                "queued": len(self._ready),
                "priority": len(self._priority),
                "playlist_size": len(self._playlist),
                "last_error": self.last_error,
                "history": list(self.history)[-20:][::-1],
            }
