from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .avatar import AvatarMixer
from .brain import Brain, Segment
from .config import Config
from .mouth import MouthDriver
from .util import SAMPLE_RATE, write_all

log = logging.getLogger("goovo.pipeline")

AUDIO_CHUNK_MS = 50
CHUNK_BYTES = SAMPLE_RATE * 2 * AUDIO_CHUNK_MS // 1000
SILENCE = b"\x00" * CHUNK_BYTES


def _bitrate_kbps(value: str) -> int:
    match = re.match(r"(\d+)([kM]?)", value.strip())
    if not match:
        return 2500
    number = int(match.group(1))
    return number * 1000 if match.group(2) == "M" else number


class Pipeline:
    """视频帧 + PCM 音频按墙钟节奏写入 ffmpeg，合成一路连续直播流。"""

    def __init__(self, config: Config, brain: Brain, avatar: AvatarMixer):
        self.config = config
        self.brain = brain
        self.avatar = avatar
        self.stop_event = threading.Event()
        self.speaking = False
        self.driver = MouthDriver()
        self.sink_alive = False
        self.started_at = 0.0
        self.frames_written = 0
        self.audio_seconds = 0.0
        self._cpu_warned = False

    def stop(self) -> None:
        self.stop_event.set()

    def state(self) -> dict:
        return {
            "sink": self.config.stream.sink,
            "sink_alive": self.sink_alive,
            "speaking": self.speaking,
            "uptime": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "frames": self.frames_written,
            "audio_seconds": round(self.audio_seconds, 1),
            "resolution": f"{self.config.avatar.width}x{self.config.avatar.height}",
            "fps": self.config.avatar.fps,
        }

    def run(self) -> None:
        self.started_at = time.time()
        self.avatar.warmup()
        if hasattr(self.avatar, "prerender") and self.config.brain.mode == "playlist":
            self.avatar.prerender(self.brain.playlist_texts(), self.brain.tts)
        while not self.stop_event.is_set():
            self._run_once()
            if self.stop_event.is_set():
                break
            if self.config.stream.sink == "rtmp":
                log.warning("推流中断，3 秒后自动重连…")
                self.stop_event.wait(3.0)
            else:
                break

    def _sink_command(self, video_fd: int, audio_fd: int) -> list[str]:
        stream = self.config.stream
        avatar = self.config.avatar
        bitrate = _bitrate_kbps(stream.video_bitrate)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
            "-thread_queue_size", "64",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-video_size", f"{avatar.width}x{avatar.height}",
            "-framerate", str(avatar.fps),
            "-i", f"/dev/fd/{video_fd}",
            "-thread_queue_size", "512",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-i", f"/dev/fd/{audio_fd}",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-profile:v", "main",
            "-g", str(avatar.fps * 2), "-keyint_min", str(avatar.fps * 2),
            "-sc_threshold", "0",
            "-b:v", stream.video_bitrate,
            "-maxrate", f"{int(bitrate * 1.2)}k",
            "-bufsize", f"{bitrate * 2}k",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", stream.audio_bitrate,
            "-ar", str(SAMPLE_RATE), "-ac", "1",
        ]
        if stream.sink == "rtmp":
            if not stream.rtmp_url:
                raise RuntimeError("stream.rtmp_url 为空：请在视频号助手获取推流地址后填入 config.toml")
            command += ["-f", "flv", "-flvflags", "no_duration_filesize", stream.rtmp_url]
        elif stream.sink == "record":
            target = self.config.path(stream.record_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            command += ["-f", "matroska", str(target)]
        elif stream.sink == "preview":
            command += ["-f", "mpegts", "pipe:1"]
        else:
            raise RuntimeError(f"未知输出类型: {stream.sink}")
        return command

    def _run_once(self) -> None:
        run_stop = threading.Event()
        video_read, video_write = os.pipe()
        audio_read, audio_write = os.pipe()
        command = self._sink_command(video_read, audio_read)
        player = None
        if self.config.stream.sink == "preview":
            scale = self.config.stream.preview_scale
            win_w = max(160, int(self.config.avatar.width * scale))
            win_h = max(160, int(self.config.avatar.height * scale))
            player = subprocess.Popen(
                [
                    "ffplay", "-hide_banner", "-loglevel", "error",
                    "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop",
                    "-window_title", "goovo-human 直播预览",
                    "-x", str(win_w), "-y", str(win_h),
                    "-i", "pipe:0",
                ],
                stdin=subprocess.PIPE,
            )
            sink = subprocess.Popen(command, pass_fds=(video_read, audio_read), stdout=player.stdin)
            player.stdin.close()
        else:
            sink = subprocess.Popen(command, pass_fds=(video_read, audio_read))
        os.close(video_read)
        os.close(audio_read)
        self.sink_alive = True
        log.info("输出已启动: %s", self.config.stream.sink)

        video_thread = threading.Thread(
            target=self._video_loop, args=(video_write, run_stop), name="video", daemon=True
        )
        audio_thread = threading.Thread(
            target=self._audio_loop, args=(audio_write, run_stop), name="audio", daemon=True
        )
        video_thread.start()
        audio_thread.start()
        try:
            while not self.stop_event.is_set() and sink.poll() is None:
                if player is not None and player.poll() is not None:
                    log.info("预览窗口已关闭")
                    self.stop_event.set()
                    break
                time.sleep(0.2)
        finally:
            self.sink_alive = False
            self.speaking = False
            run_stop.set()
            for fd in (video_write, audio_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
            video_thread.join(timeout=3)
            audio_thread.join(timeout=3)
            try:
                sink.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sink.kill()
            if player is not None:
                try:
                    player.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    player.kill()
            log.info("输出已停止")

    def _video_loop(self, fd: int, run_stop: threading.Event) -> None:
        interval = 1.0 / self.config.avatar.fps
        start = time.perf_counter()
        count = 0
        while not run_stop.is_set():
            try:
                frame = self.avatar.read_frame(self.speaking, self.driver)
                write_all(fd, frame)
            except (BrokenPipeError, OSError):
                return
            except Exception as exc:
                log.error("视频线程异常: %s", exc)
                return
            count += 1
            self.frames_written = count
            delay = start + count * interval - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -2.0 and not self._cpu_warned:
                self._cpu_warned = True
                log.warning("编码/解码跟不上实时节奏，建议降低分辨率或帧率")

    def _audio_loop(self, fd: int, run_stop: threading.Event) -> None:
        chunk_seconds = AUDIO_CHUNK_MS / 1000
        start = time.perf_counter()
        count = 0
        current: Segment | None = None
        position = 0
        gap_until = 0.0
        while not run_stop.is_set():
            if self.brain.interrupt_event.is_set():
                self.brain.interrupt_event.clear()
                current = None
                gap_until = 0.0
                if hasattr(self.avatar, "end_segment"):
                    self.avatar.end_segment()
            if current is not None and position >= len(current.pcm):
                current = None
                gap_until = time.perf_counter() + self.config.brain.gap_seconds
                if hasattr(self.avatar, "end_segment"):
                    self.avatar.end_segment()
            if current is None:
                segment = self.brain.next_segment()
                if segment is not None:
                    ready = segment.source == "manual" or time.perf_counter() >= gap_until
                    if ready:
                        current = segment
                        position = 0
                        self.brain.mark_started(segment)
                        if hasattr(self.avatar, "start_segment"):
                            self.avatar.start_segment(segment.pcm, getattr(self.avatar, "idle_count", 0))
                        self.driver.set_segment(segment.text, len(segment.pcm), segment.boundaries, segment.pcm)
                        log.info("口播: %s", segment.text[:40])
            if current is not None:
                data = current.pcm[position:position + CHUNK_BYTES]
                if len(data) < CHUNK_BYTES:
                    data = data + b"\x00" * (CHUNK_BYTES - len(data))
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
                self.driver.update(position / (SAMPLE_RATE * 2), float(np.sqrt(np.mean(samples * samples))))
                position += CHUNK_BYTES
                self.speaking = True
            else:
                data = SILENCE
                self.speaking = False
                self.driver.idle()
            try:
                write_all(fd, data)
            except (BrokenPipeError, OSError):
                return
            count += 1
            self.audio_seconds = count * chunk_seconds
            delay = start + count * chunk_seconds - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
