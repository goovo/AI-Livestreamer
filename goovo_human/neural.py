"""神经网络口型引擎：MuseTalk 1.5 MLX（Apple GPU）生成真人唇形。

M2 Pro 上 MuseTalk 约 5-6 帧/秒，无法边播边渲染，因此采用**预渲染缓存**：
口播稿音频逐帧生成下半脸（缓存于 .cache/mouth/，按音频哈希+idle 版本键控），
播放时零 GPU 开销：取缓存帧 → 按当前头部姿态仿射对齐 → 羽化贴回 idle 画面。
循环稿件第二次起完全命中缓存；新稿件（LLM/插播）首遍用 mouth 引擎兜底并后台渲染。
worker 缺失/崩溃时整体回退 mouth 引擎。
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
import wave
from pathlib import Path

import cv2
import numpy as np

from .avatar import FrameSource
from .config import AvatarConfig
from .mouth import MouthAvatar, _ensure_model, _landmarker

log = logging.getLogger("goovo.neural")


def neural_available(root: Path) -> bool:
    return (
        (root / ".venv-muse" / "bin" / "python").exists()
        and (root / ".cache" / "models" / "musetalk-mlx-fp16" / "unet.safetensors").exists()
        and (root / "tools" / "musetalk_worker.py").exists()
    )


class NeuralMouth:
    """idle 循环画面 + MuseTalk 预渲染唇形，按音频逐帧对齐、按姿态贴回。"""

    def __init__(self, config: AvatarConfig, root: Path):
        self.config = config
        self.root = root
        self.idle = FrameSource(root / config.idle, config.width, config.height, config.fps)
        self.idle_count = 0
        meta = root / "assets" / "mouth" / "meta.json"
        self._fallback = MouthAvatar(config, root) if meta.exists() else None
        self._job: dict | None = None
        self._seg_tick = 0
        self._last_face: np.ndarray | None = None
        self._trio: np.ndarray | None = None
        self._landmarker = None
        self._mp = None
        self._ts = 0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._prepared = threading.Event()
        self._pending: dict[str, dict] = {}
        self._worker_ok = False
        self._idle_key = ""
        self._boxes = None
        self._canon_trio = None
        self._loop = 0
        mask = np.zeros((256, 256), dtype=np.float32)
        cv2.ellipse(mask, (128, int(256 * 0.72)), (int(256 * 0.36), int(256 * 0.24)), 0, 0, 360, 1.0, -1)
        self._mask256 = cv2.GaussianBlur(mask, (0, 0), sigmaX=256 * 0.04, sigmaY=256 * 0.04)
        self._proc: subprocess.Popen | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        log_path = self.root / ".cache" / "musetalk-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = open(log_path, "ab")
        try:
            self._proc = subprocess.Popen(
                [str(self.root / ".venv-muse" / "bin" / "python"), str(self.root / "tools" / "musetalk_worker.py")],
                cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            )
        except OSError as exc:
            log.warning("MuseTalk worker 启动失败，回退 warp 口型: %s", exc)
            return
        threading.Thread(target=self._read_loop, name="musetalk-reader", daemon=True).start()
        if not self._ready.wait(180):
            log.warning("MuseTalk worker 就绪超时，回退 warp 口型")
            self._kill_worker()
            return
        self._worker_ok = True
        self._ensure_prepared()

    def _kill_worker(self) -> None:
        self._worker_ok = False
        for pending in self._pending.values():
            pending["event"].set()
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def _send(self, payload: dict) -> None:
        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                raise RuntimeError("worker 已退出")
            self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            self._proc.stdin.flush()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("event") == "ready":
                self._ready.set()
            elif msg.get("event") == "progress":
                pending = self._pending.get(str(msg.get("id")))
                if pending is not None:
                    pending["done"] = msg.get("done", 0)
            elif msg.get("cmd") == "prepare" and msg.get("ok"):
                self._prepared.set()
            elif msg.get("cmd") == "render":
                pending = self._pending.pop(str(msg.get("id")), None)
                if not msg.get("ok"):
                    log.error("口型渲染失败: %s", msg.get("error"))
                if pending is not None:
                    pending["event"].set()
            elif not msg.get("ok", True):
                log.error("MuseTalk worker: %s", msg.get("error"))
        self._worker_ok = False
        self._ready.set()
        self._prepared.set()
        for pending in self._pending.values():
            pending["event"].set()

    def _ensure_prepared(self) -> None:
        assets = self.root / "assets" / "mouth"
        idle = self.root / self.config.idle
        key = f"{idle.stat().st_size}:{int(idle.stat().st_mtime)}"
        need = True
        if (assets / "neural.npz").exists():
            data = np.load(assets / "neural.npz")
            need = str(data["idle_key"]) != key or "canon_trio" not in data
        if need:
            log.info("为 idle 预计算 MuseTalk 参考 latent（约半分钟）…")
            self._prepared.clear()
            self._send({"cmd": "prepare", "idle": self.config.idle,
                        "width": self.config.width, "height": self.config.height, "fps": self.config.fps})
            if not self._prepared.wait(600):
                log.warning("latent 预计算超时，回退 warp 口型")
                self._kill_worker()
                return
        data = np.load(assets / "neural.npz")
        self._boxes = data["boxes"]
        self._canon_trio = data["canon_trio"]
        self._loop = int(data["boxes"].shape[0])
        self._idle_key = str(data["idle_key"])
        log.info("MuseTalk 口型就绪：idle %d 帧姿态表", self._loop)

    def _cache_dir(self, pcm: bytes) -> Path:
        digest = hashlib.sha1(pcm).hexdigest()[:16]
        idle_hash = hashlib.sha1(self._idle_key.encode()).hexdigest()[:8]
        return self.root / ".cache" / "mouth" / f"{digest}-{idle_hash}"

    def _cache_ready(self, directory: Path, frames: int) -> bool:
        manifest = directory / "manifest.json"
        if not manifest.exists():
            return False
        try:
            return int(json.loads(manifest.read_text())["frames"]) >= frames
        except (ValueError, KeyError):
            return False

    def ensure_render(self, pcm: bytes, frames: int) -> tuple[Path, bool]:
        """缓存命中返回 (dir, True)；否则提交后台渲染并返回 (dir, False)。"""
        directory = self._cache_dir(pcm)
        if self._cache_ready(directory, frames):
            return directory, True
        if not self._worker_ok:
            return directory, False
        key = directory.name
        with self._lock:
            if key in self._pending:
                return directory, False
            directory.mkdir(parents=True, exist_ok=True)
            wav_path = directory / "audio.wav"
            if not wav_path.exists():
                with wave.open(str(wav_path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(44100)
                    handle.writeframes(pcm)
            event = threading.Event()
            self._pending[key] = {"event": event, "done": 0, "frames": frames}
        try:
            self._send({"cmd": "render", "id": key, "wav": str(wav_path),
                        "frames": frames, "out": str(directory)})
        except RuntimeError as exc:
            log.warning("口型渲染提交失败: %s", exc)
            with self._lock:
                self._pending.pop(key, None)
            self._worker_ok = False
        return directory, False

    def prerender(self, texts: list[str], tts) -> None:
        """开播前把循环稿件的口型全部渲染好（仅首次耗时）。"""
        if not self._worker_ok:
            return
        for index, text in enumerate(texts, 1):
            pcm, _ = tts.synthesize(text)
            frames = int(len(pcm) / 88200 * self.config.fps)
            directory, ready = self.ensure_render(pcm, frames)
            if ready:
                continue
            log.info("预渲染口型 [%d/%d] %d 帧（M2 Pro 约 6 帧/秒）…", index, len(texts), frames)
            pending = self._pending.get(directory.name)
            if pending is None:
                continue
            while not pending["event"].wait(5.0):
                log.info("预渲染口型 [%d/%d] %d/%d", index, len(texts), pending["done"], frames)
            if self._cache_ready(directory, frames):
                log.info("预渲染口型 [%d/%d] 完成", index, len(texts))
            else:
                log.warning("预渲染口型 [%d/%d] 未完成，该段首遍用 mouth 兜底", index, len(texts))

    def start_segment(self, pcm: bytes, base_index: int = 0) -> None:
        frames = int(len(pcm) / 88200 * self.config.fps)
        directory, ready = self.ensure_render(pcm, frames)
        self._job = {"dir": directory, "frames": frames,
                     "mode": "neural" if ready else "warp"}
        self._seg_tick = 0
        self._last_face = None
        if not ready:
            log.info("本段口型缓存未就绪，首遍用 mouth 兜底（后台渲染中）")

    def end_segment(self) -> None:
        self._job = None
        self._seg_tick = 0
        self._last_face = None

    def _track(self, frame: bytes) -> np.ndarray | None:
        if self._fallback is not None:
            return self._fallback._track(frame)
        if self._landmarker is None:
            self._landmarker, self._mp = _landmarker(_ensure_model(self.root), video=True)
        width, height = self.config.width, self.config.height
        array = np.frombuffer(frame, dtype=np.uint8).reshape(height * 3 // 2, width)
        half = cv2.resize(array[:height], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(half, cv2.COLOR_YUV2BGR_I420)
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            self._ts,
        )
        self._ts += max(1, int(1000 / self.config.fps))
        if result.face_landmarks:
            points = np.array([(lm.x * (width // 2), lm.y * (height // 2)) for lm in result.face_landmarks[0]])
            return (points[[61, 291, 1]] * 2.0).astype(np.float32)
        return None

    def _face(self, index: int) -> np.ndarray | None:
        job = self._job
        if job is None:
            return None
        path = job["dir"] / f"f_{index:05d}.jpg"
        if path.exists():
            face = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if face is not None:
                self._last_face = face
                return face
        return self._last_face

    def read_frame(self, speaking: bool, driver=None) -> bytes:
        frame = self.idle.read_frame()
        self.idle_count += 1
        job = self._job
        if not speaking or job is None:
            return frame
        trio = self._track(frame)
        if trio is not None:
            self._trio = trio if self._trio is None else self._trio * 0.5 + trio * 0.5
        if job["mode"] == "warp" or not self._worker_ok:
            if self._fallback is not None:
                return self._fallback.render_frame(frame, speaking, driver, trio=self._trio)
            return frame
        if self._boxes is None:
            return frame
        index = self._seg_tick
        self._seg_tick += 1
        if index >= job["frames"]:
            return frame
        face = self._face(index)
        if face is None:
            return frame
        box = self._boxes[(self.idle_count - 1) % self._loop]
        return self._paste(frame, face, box, min(1.0, 0.25 + index / 4))

    def _paste(self, frame: bytes, face: np.ndarray, box, scale: float = 1.0) -> bytes:
        width, height = self.config.width, self.config.height
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0) // 2 * 2, max(0, y0) // 2 * 2
        x1, y1 = min(width, x1) // 2 * 2, min(height, y1) // 2 * 2
        bw, bh = x1 - x0, y1 - y0
        if bw < 16 or bh < 16:
            return frame
        if self._trio is not None:
            local = (self._trio - np.array([x0, y0], dtype=np.float32)) * np.array(
                [256 / bw, 256 / bh], dtype=np.float32)
            matrix = cv2.getAffineTransform(self._canon_trio, local)
            face = cv2.warpAffine(face, matrix, (256, 256), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV_I420)
        fy = cv2.resize(face_yuv[:256], (bw, bh), interpolation=cv2.INTER_LINEAR)
        fu = cv2.resize(face_yuv[256:320].reshape(128, 128), (bw // 2, bh // 2), interpolation=cv2.INTER_LINEAR)
        fv = cv2.resize(face_yuv[320:384].reshape(128, 128), (bw // 2, bh // 2), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(self._mask256, (bw, bh), interpolation=cv2.INTER_LINEAR)
        if scale < 1.0:
            mask = mask * scale
        mask_half = cv2.resize(self._mask256, (bw // 2, bh // 2), interpolation=cv2.INTER_AREA)
        if scale < 1.0:
            mask_half = mask_half * scale
        buffer = bytearray(frame)
        out = np.frombuffer(buffer, dtype=np.uint8).reshape(height * 3 // 2, width)
        ry = out[:height][y0:y1, x0:x1].astype(np.float32)
        total = max(float(mask.sum()), 1e-6)
        shift = float(np.clip((ry * mask).sum() / total - (fy.astype(np.float32) * mask).sum() / total, -14.0, 14.0))
        out[:height][y0:y1, x0:x1] = (ry * (1 - mask) + (fy.astype(np.float32) + shift) * mask).astype(np.uint8)
        uy = out[height:height + height // 4].reshape(height // 2, width // 2)
        vy = out[height + height // 4:height + height // 2].reshape(height // 2, width // 2)
        ru = uy[y0 // 2:y1 // 2, x0 // 2:x1 // 2].astype(np.float32)
        rv = vy[y0 // 2:y1 // 2, x0 // 2:x1 // 2].astype(np.float32)
        half_total = max(float(mask_half.sum()), 1e-6)
        su = float(np.clip((ru * mask_half).sum() / half_total - (fu.astype(np.float32) * mask_half).sum() / half_total, -6.0, 6.0))
        sv = float(np.clip((rv * mask_half).sum() / half_total - (fv.astype(np.float32) * mask_half).sum() / half_total, -6.0, 6.0))
        uy[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = (ru * (1 - mask_half) + (fu.astype(np.float32) + su) * mask_half).astype(np.uint8)
        vy[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = (rv * (1 - mask_half) + (fv.astype(np.float32) + sv) * mask_half).astype(np.uint8)
        return bytes(buffer)

    def warmup(self) -> None:
        self.idle.read_frame()
        self.idle_count += 1

    def close(self) -> None:
        self.end_segment()
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        self._kill_worker()
        if self._fallback is not None:
            self._fallback.close()
        if self._landmarker is not None:
            self._landmarker.close()
        self.idle.close()
        if getattr(self, "_stderr", None):
            self._stderr.close()
