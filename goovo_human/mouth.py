from __future__ import annotations

import bisect
import json
import logging
import math
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .avatar import FrameSource
from .config import AvatarConfig

log = logging.getLogger("goovo.mouth")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_TOP, INNER_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 0, 17
TRIO = [61, 291, 1]
CANON_W, CANON_H = 240, 110

FINAL_VISEME = {
    "a": "a", "ia": "a", "ua": "a", "ai": "a", "uai": "a", "ao": "a", "iao": "a",
    "an": "a", "ian": "a", "uan": "a", "van": "a", "ang": "a", "iang": "a", "uang": "a",
    "e": "a", "en": "a", "eng": "a", "er": "a", "ei": "i", "uei": "i", "ui": "i",
    "o": "o", "uo": "o", "ou": "o", "iou": "o", "iu": "o", "ong": "o", "iong": "o",
    "i": "i", "in": "i", "ing": "i", "ie": "i", "ve": "o", "ue": "o",
    "u": "u", "un": "u", "uen": "u", "v": "u", "ü": "u", "üe": "o",
}
VISEME_OPENNESS = {"a": 1.0, "o": 0.95, "i": 0.85, "u": 0.8}
VISEME_STRETCH = {"a": 1.05, "o": 0.88, "i": 1.16, "u": 0.82, "closed": 1.0}


def _landmarker(model_path: Path, video: bool):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def _ensure_model(root: Path) -> Path:
    target = root / ".cache" / "models" / "face_landmarker.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("下载人脸关键点模型…")
        urllib.request.urlretrieve(MODEL_URL, target)
    return target


def _yuv_to_bgr(frame: bytes, width: int, height: int) -> np.ndarray:
    array = np.frombuffer(frame, dtype=np.uint8).reshape(height * 3 // 2, width)
    return cv2.cvtColor(array, cv2.COLOR_YUV2BGR_I420)


def _mouth_metrics(landmarks, width: int, height: int) -> tuple[float, float, tuple[int, int, int, int]]:
    points = np.array([(lm.x * width, lm.y * height) for lm in landmarks])
    outer = points[OUTER_LIPS]
    mouth_width = np.linalg.norm(points[MOUTH_LEFT] - points[MOUTH_RIGHT])
    mouth_height = np.linalg.norm(points[LIP_TOP] - points[LIP_BOTTOM])
    openness = np.linalg.norm(points[INNER_TOP] - points[INNER_BOTTOM]) / max(mouth_width, 1e-6)
    ratio = mouth_width / max(mouth_height, 1e-6)
    x0, y0 = outer.min(axis=0)
    x1, y1 = outer.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = (x1 - x0) / 2 * 1.5
    half_h = (y1 - y0) / 2 * 1.7
    left = int(max(0, cx - half_w)) // 2 * 2
    top = int(max(0, cy - half_h)) // 2 * 2
    right = int(min(width, cx + half_w)) // 2 * 2
    bottom = int(min(height, cy + half_h)) // 2 * 2
    return float(openness), float(ratio), (left, top, right - left, bottom - top)


def _mouth_center(landmarks, width: int, height: int) -> tuple[float, float, float]:
    points = np.array([(lm.x * width, lm.y * height) for lm in landmarks])
    center = (points[MOUTH_LEFT] + points[MOUTH_RIGHT]) / 2
    mouth_width = float(np.linalg.norm(points[MOUTH_LEFT] - points[MOUTH_RIGHT]))
    return float(center[0]), float(center[1]), mouth_width


def _classify(openness: float, ratio: float) -> str:
    if openness < 0.08:
        return "closed"
    if openness >= 0.18 and ratio < 1.45:
        return "o"
    if openness >= 0.24:
        return "a"
    if openness >= 0.14:
        return "i"
    return "u"


def prepare_mouth_assets(config: AvatarConfig, root: Path) -> Path:
    """从 idle 定位嘴部区域，从 talk 自动提取 a/o/i/u 四类口型素材。"""
    model = _ensure_model(root)
    out_dir = root / "assets" / "mouth"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps = config.width, config.height, config.fps

    idle_source = FrameSource(root / config.idle, width, height, fps)
    detector, mp = _landmarker(model, video=False)
    idle_bgr = _yuv_to_bgr(idle_source.read_frame(), width, height)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(idle_bgr, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        raise RuntimeError("idle 视频首帧未检测到人脸，请确认画面中有清晰正脸")
    _, _, rect = _mouth_metrics(result.face_landmarks[0], width, height)
    center_x, center_y, mouth_width = _mouth_center(result.face_landmarks[0], width, height)
    idle_source.close()
    detector.close()

    talk_path = root / (config.talk or config.idle)
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(talk_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    talk_source = FrameSource(talk_path, width, height, fps)
    detector, mp = _landmarker(model, video=True)
    candidates: dict[str, list] = {name: [] for name in ("a", "o", "i", "u")}
    total_frames = int(duration * fps)
    for index in range(0, total_frames, 2):
        frame = talk_source.read_frame()
        bgr = _yuv_to_bgr(frame, width, height)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            int(index * 1000 / fps),
        )
        if not result.face_landmarks:
            continue
        openness, ratio, box = _mouth_metrics(result.face_landmarks[0], width, height)
        label = _classify(openness, ratio)
        if label == "closed":
            continue
        x, y, w, h = box
        points = np.array([(lm.x * width, lm.y * height) for lm in result.face_landmarks[0]])
        trio = points[TRIO] - np.array([x, y])
        candidates[label].append((openness, ratio, bgr[y:y + h, x:x + w].copy(), trio))
    talk_source.close()
    detector.close()

    sprites = {}
    for label, items in candidates.items():
        if not items:
            continue
        openness_median = np.median([item[0] for item in items])
        ratio_median = np.median([item[1] for item in items])
        best = min(items, key=lambda item: (item[0] - openness_median) ** 2 + (item[1] - ratio_median) ** 2)
        target = out_dir / f"{label}.png"
        cv2.imwrite(str(target), best[2])
        crop_h, crop_w = best[2].shape[:2]
        trio = best[3] * np.array([CANON_W / crop_w, CANON_H / crop_h])
        sprites[label] = {"file": target.name, "trio": trio.tolist()}
    if not sprites:
        raise RuntimeError("talk 视频中没有提取到开口口型，请确认素材中有自然说话画面")
    for label in ("a", "o", "i", "u"):
        if label not in sprites:
            sprites[label] = next(iter(sprites.values()))
    used_files = {info["file"] for info in sprites.values()}
    for old in out_dir.glob("*.png"):
        if old.name not in used_files:
            old.unlink()
    meta = {
        "rect": list(rect),
        "center": [center_x, center_y],
        "mouth_width": mouth_width,
        "sprites": sprites,
        "counts": {k: len(v) for k, v in candidates.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("口型素材就绪: %s 各口型帧数 %s", out_dir, meta["counts"])
    return out_dir


def _final_of(char: str) -> str:
    from pypinyin import lazy_pinyin, Style

    finals = lazy_pinyin(char, style=Style.FINALS, errors=lambda _: "")
    return finals[0] if finals else ""


class MouthDriver:
    """把 TTS 字级时间戳 + 拼音韵母 + 音频包络翻译成逐帧口型指令。"""

    def __init__(self):
        self._starts: list[float] = []
        self._entries: list[tuple[float, str]] = []
        self.env = 0.0
        self.t = 0.0
        self._activity = 0.0
        self._level = 0.0
        self._active = False
        self._silence = 0.0
        self._last_position: float | None = None
        self._env_times: np.ndarray | None = None
        self._env_values: np.ndarray | None = None

    def set_segment(self, text: str, pcm_bytes: int, boundaries: list | None, pcm: bytes | None = None) -> None:
        duration = pcm_bytes / 88200
        self._starts = []
        self._entries = []
        self.env = 0.0
        self.t = 0.0
        self._activity = 0.0
        self._level = 0.0
        self._active = False
        self._silence = 0.0
        self._last_position = None
        self._env_times = None
        self._env_values = None
        spans: list[tuple[float, float, str]] = []
        if boundaries:
            for boundary in boundaries:
                chars = [ch for ch in boundary.get("text", "") if "\u4e00" <= ch <= "\u9fff"]
                if not chars:
                    continue
                start = boundary.get("offset_ms", 0) / 1000
                span = boundary.get("duration_ms", 0) / 1000 / len(chars)
                for offset, char in enumerate(chars):
                    spans.append((start + offset * span, start + (offset + 1) * span, char))
        else:
            chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
            if chars:
                span = duration / len(chars)
                spans = [(i * span, (i + 1) * span, char) for i, char in enumerate(chars)]
        for start, end, char in spans:
            viseme = FINAL_VISEME.get(_final_of(char), "a")
            self._starts.append(start)
            self._entries.append((end, viseme))
        if pcm:
            self._build_envelope(pcm)

    def _build_envelope(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        hop, win = 441, 1102
        if len(samples) < win:
            return
        count = (len(samples) - win) // hop + 1
        starts = np.arange(count) * hop
        frames = np.array([
            float(np.sqrt(np.mean(samples[i:i + win] ** 2))) for i in starts
        ])
        noise = float(np.percentile(frames, 20))
        speech = float(np.percentile(frames, 95))
        norm = np.clip((frames - noise) / max(speech - noise, 1e-6), 0.0, 1.0)
        env = np.zeros_like(norm)
        current = 0.0
        for index, value in enumerate(norm):
            tau = 0.02 if value > current else 0.11
            current += (value - current) * (1 - math.exp(-0.01 / tau))
            env[index] = current
        env = np.convolve(env, np.ones(5) / 5, mode="same") ** 0.7
        self._env_times = (starts + win / 2) / 44100
        self._env_values = env

    def update(self, position_seconds: float, rms: float) -> None:
        if self._last_position is None:
            dt = 0.05
        else:
            dt = min(0.1, max(0.001, position_seconds - self._last_position))
        self._last_position = position_seconds
        self.t = position_seconds
        if rms > 0.028:
            self._active = True
            self._silence = 0.0
        elif rms < 0.010:
            self._silence += dt
            if self._silence >= 0.14:
                self._active = False
        target = 1.0 if self._active else 0.0
        tau = 0.035 if target > self._activity else 0.18
        self._activity += (target - self._activity) * (1 - math.exp(-dt / tau))
        level_tau = 0.09 if rms > self._level else 0.16
        self._level += (rms - self._level) * (1 - math.exp(-dt / level_tau))
        self.env = self._level

    def idle(self) -> None:
        self._active = False
        self._silence = 0.0
        dt = 0.05
        self._activity += (0.0 - self._activity) * (1 - math.exp(-dt / 0.18))
        self._level += (0.0 - self._level) * (1 - math.exp(-dt / 0.16))
        self.env = self._level

    def viseme(self) -> str:
        index = bisect.bisect_right(self._starts, self.t) - 1
        if index < 0:
            return "a"
        end, viseme = self._entries[index]
        return viseme if self.t <= end else "closed"

    def alpha(self) -> float:
        if self._env_values is not None:
            envelope = float(np.interp(self.t, self._env_times, self._env_values))
            level = (0.25 * self._activity + 0.75 * envelope) * 0.95
            viseme = self.viseme()
            if viseme == "closed":
                level *= 0.2
            else:
                level *= VISEME_OPENNESS.get(viseme, 1.0)
            return level
        intensity = min(1.0, max(0.0, (self._level - 0.02) / 0.12))
        level = self._activity * (0.88 + 0.12 * intensity) * 0.95
        if self.viseme() == "closed":
            level *= 0.3
        return level


class MouthAvatar:
    """待机画面 + 音频驱动的口型贴换：嘴型跟随语音节奏与韵母口型。"""

    def __init__(self, config: AvatarConfig, root: Path):
        self.config = config
        self.root = root
        self.idle = FrameSource(root / config.idle, config.width, config.height, config.fps)
        meta = json.loads((root / "assets" / "mouth" / "meta.json").read_text())
        self.rect = tuple(meta["rect"])
        self.ref_center = tuple(meta.get("center", (self.rect[0] + self.rect[2] / 2, self.rect[1] + self.rect[3] / 2)))
        self._canonical_trio = self._reference_trio(meta)
        self._offset = [0.0, 0.0]
        self._tick = 0
        self._ts = 0
        self._landmarker = None
        self._mp = None
        self._blend_planes = None
        self._blend_alpha = 0.0
        self.sprites: dict[str, dict] = {}
        for label, info in meta["sprites"].items():
            bgr = cv2.imread(str(root / "assets" / "mouth" / info["file"]), cv2.IMREAD_COLOR)
            bgr = cv2.resize(bgr, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
            trio = np.array(info["trio"], dtype=np.float32)
            matrix = cv2.getAffineTransform(trio, self._canonical_trio)
            bgr = cv2.warpAffine(
                bgr, matrix, (CANON_W, CANON_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            self.sprites[label] = {"bgr": bgr, "trio": self._canonical_trio.copy()}
        mask = np.zeros((CANON_H, CANON_W), dtype=np.float32)
        cv2.ellipse(mask, (CANON_W // 2, CANON_H // 2), (int(CANON_W * 0.44), int(CANON_H * 0.44)), 0, 0, 360, 1.0, -1)
        kernel = max(3, (CANON_W // 6) | 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), CANON_W / 9)
        border = max(4, int(min(CANON_W, CANON_H) * 0.22))
        ramp = np.linspace(0, 1, border, dtype=np.float32)
        mask[:border, :] *= ramp[:, None]
        mask[-border:, :] *= ramp[::-1][:, None]
        mask[:, :border] *= ramp[None, :]
        mask[:, -border:] *= ramp[::-1][None, :]
        self.mask = mask
        self._trio = None
        self._blend_bgr = None
        self._blend_trio = None
        self._stretch = 1.0

    def _reference_trio(self, meta: dict) -> np.ndarray:
        try:
            frame = self.idle.read_frame()
            bgr = _yuv_to_bgr(frame, self.config.width, self.config.height)
            detector, mp = _landmarker(_ensure_model(self.root), video=False)
            try:
                result = detector.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                ))
            finally:
                detector.close()
            if result.face_landmarks:
                points = np.array([
                    (lm.x * self.config.width, lm.y * self.config.height)
                    for lm in result.face_landmarks[0]
                ], dtype=np.float32)
                x, y, w, h = meta["rect"]
                return (points[TRIO] - np.array([x, y], dtype=np.float32)) * np.array(
                    [CANON_W / w, CANON_H / h], dtype=np.float32
                )
        except Exception as exc:
            log.warning("口型参考姿态获取失败，使用素材平均锚点: %s", exc)
        return np.mean([
            np.array(info["trio"], dtype=np.float32)
            for info in meta["sprites"].values()
        ], axis=0).astype(np.float32)

    def _track(self, frame: bytes) -> np.ndarray | None:
        if self._landmarker is None:
            self._landmarker, self._mp = _landmarker(_ensure_model(self.root), video=True)
        width, height = self.config.width, self.config.height
        array = np.frombuffer(frame, dtype=np.uint8).reshape(height * 3 // 2, width)
        y_half = cv2.resize(array[:height], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        u_half = cv2.resize(array[height:height + height // 4].reshape(height // 2, width // 2), (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        v_half = cv2.resize(array[height + height // 4:height + height // 2].reshape(height // 2, width // 2), (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        half = np.vstack([
            y_half,
            u_half.reshape(height // 8, width // 2),
            v_half.reshape(height // 8, width // 2),
        ])
        bgr = cv2.cvtColor(half, cv2.COLOR_YUV2BGR_I420)
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            self._ts,
        )
        self._ts += max(1, int(1000 / self.config.fps))
        if result.face_landmarks:
            points = np.array([(lm.x * (width // 2), lm.y * (height // 2)) for lm in result.face_landmarks[0]])
            return (points[TRIO] * 2.0).astype(np.float32)
        return None

    def read_frame(self, speaking: bool, driver: MouthDriver | None = None) -> bytes:
        frame = self.idle.read_frame()
        return self.render_frame(frame, speaking, driver)

    def render_frame(self, frame: bytes, speaking: bool, driver: MouthDriver | None = None,
                     trio: np.ndarray | None = None) -> bytes:
        """在给定帧上渲染口型（neural 引擎复用本方法做兜底）。"""
        self._tick += 1
        dt = 1.0 / self.config.fps
        if not speaking or driver is None:
            if self._tick % 6 == 0:
                trio = self._track(frame) if trio is None else trio
                if trio is not None:
                    self._trio = trio if self._trio is None else self._trio * 0.5 + trio * 0.5
            alpha = 0.0
            target = None
        else:
            alpha = driver.alpha()
            viseme = driver.viseme()
            target = self.sprites.get(viseme) or next(iter(self.sprites.values()))
            target_stretch = VISEME_STRETCH.get(viseme, 1.0)
            trio = self._track(frame) if trio is None else trio
            if trio is not None:
                if self._trio is None:
                    self._trio = trio
                else:
                    kt = 1 - math.exp(-dt / 0.08)
                    self._trio += (trio - self._trio) * kt
        ka = 1 - math.exp(-dt / 0.05)
        self._blend_alpha += (alpha - self._blend_alpha) * ka
        ks = 1 - math.exp(-dt / 0.08)
        self._stretch += (target_stretch - self._stretch) * ks if target is not None else (1.0 - self._stretch) * ks
        if target is not None:
            kp = 1 - math.exp(-dt / 0.06)
            if self._blend_bgr is None:
                self._blend_bgr = target["bgr"].astype(np.float32)
                self._blend_trio = target["trio"].copy()
            else:
                self._blend_bgr += (target["bgr"].astype(np.float32) - self._blend_bgr) * kp
                self._blend_trio += (target["trio"] - self._blend_trio) * kp
        return self._render_warp(frame, self._blend_alpha, self._stretch, self._blend_bgr)

    def _render_warp(self, frame: bytes, openness: float, stretch: float, sprite_bgr: np.ndarray | None) -> bytes:
        if openness < 0.015 or self._trio is None or sprite_bgr is None:
            return frame
        width, height = self.config.width, self.config.height
        left, right = self._trio[0], self._trio[1]
        center_x, center_y = (left + right) / 2
        mouth_width = float(np.linalg.norm(left - right))
        if mouth_width < 10:
            return frame
        half_w, half_h = int(mouth_width * 1.15), int(mouth_width * 1.0)
        x0 = int(max(0, center_x - half_w)) // 4 * 4
        y0 = int(max(0, center_y - half_h)) // 4 * 4
        x1 = int(min(width, center_x + half_w)) // 4 * 4
        y1 = int(min(height, center_y + half_h)) // 4 * 4
        rw, rh = x1 - x0, y1 - y0
        if rw < 16 or rh < 16:
            return frame
        array = np.frombuffer(frame, dtype=np.uint8).reshape(height * 3 // 2, width)
        y_roi = array[:height][y0:y1, x0:x1]
        u_roi = array[height:height + height // 4].reshape(height // 2, width // 2)[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
        v_roi = array[height + height // 4:height + height // 2].reshape(height // 2, width // 2)[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
        small = np.vstack([y_roi, u_roi.reshape(rh // 4, rw), v_roi.reshape(rh // 4, rw)])
        bgr = cv2.cvtColor(small, cv2.COLOR_YUV2BGR_I420)
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        rel_x, rel_y = xx + x0 - center_x, yy + y0 - center_y
        weight_x = np.exp(-(rel_x / (mouth_width * 0.62)) ** 2)
        lower = np.clip((rel_y + mouth_width * 0.10) / (mouth_width * 0.55), 0.0, 1.0) ** 1.4
        jaw = openness * mouth_width * 0.24
        delta_y = jaw * weight_x * lower
        upper = np.clip(-rel_y / (mouth_width * 0.22), 0.0, 1.0) * weight_x
        delta_y -= openness * mouth_width * 0.05 * upper
        delta_x = (stretch - 1.0) * rel_x * weight_x * lower
        map_x = (xx - delta_x).astype(np.float32)
        map_y = (yy - delta_y).astype(np.float32)
        warped = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        inner_w = max(8, int(mouth_width * 0.52 * stretch) // 2 * 2)
        inner_h = max(4, int(openness * mouth_width * 0.20) // 2 * 2)
        inner_cx, inner_cy = int(center_x - x0), int(center_y - y0 + jaw * 0.30)
        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(mask, (inner_cx, inner_cy), (inner_w // 2, max(1, inner_h // 2)), 0, 0, 360, 1.0, -1)
        kernel_x = max(3, (inner_w // 4) | 1)
        kernel_y = max(3, (inner_h // 2) | 1)
        mask = cv2.GaussianBlur(mask, (kernel_x, kernel_y), 0)
        texture = cv2.resize(sprite_bgr.astype(np.uint8)[35:85, 60:180], (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        tx0, ty0 = inner_cx - inner_w // 2, inner_cy - inner_h // 2
        tx0 = max(0, min(rw - inner_w, tx0))
        ty0 = max(0, min(rh - inner_h, ty0))
        region = warped[ty0:ty0 + inner_h, tx0:tx0 + inner_w].astype(np.float32)
        weight = mask[ty0:ty0 + inner_h, tx0:tx0 + inner_w, None] * openness
        shift = np.clip(region.mean(axis=(0, 1)) - texture.mean(axis=(0, 1)), -20.0, 20.0)
        blended = region * (1 - weight) + (texture.astype(np.float32) + shift) * weight
        warped[ty0:ty0 + inner_h, tx0:tx0 + inner_w] = blended.astype(np.uint8)
        yuv = cv2.cvtColor(warped, cv2.COLOR_BGR2YUV_I420)
        buffer = bytearray(frame)
        out = np.frombuffer(buffer, dtype=np.uint8).reshape(height * 3 // 2, width)
        out[:height][y0:y1, x0:x1] = yuv[:rh, :rw]
        plane_bytes = rh * rw // 4
        rest = yuv[rh:].ravel()
        out_u = out[height:height + height // 4].reshape(height // 2, width // 2)
        out_v = out[height + height // 4:height + height // 2].reshape(height // 2, width // 2)
        out_u[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = rest[:plane_bytes].reshape(rh // 2, rw // 2)
        out_v[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = rest[plane_bytes:plane_bytes * 2].reshape(rh // 2, rw // 2)
        return bytes(buffer)

    def warmup(self) -> None:
        self.idle.read_frame()

    def close(self) -> None:
        self.idle.close()
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
