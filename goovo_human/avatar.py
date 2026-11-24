from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import AvatarConfig
from .util import read_exact

log = logging.getLogger("goovo.avatar")


class FrameSource:
    """无限循环解码一段视频，输出统一规格的 rawvideo 帧。"""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3 // 2
        self._proc: subprocess.Popen | None = None

    def _spawn(self) -> None:
        self.close()
        filters = (
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
            f"crop={self.width}:{self.height},fps={self.fps},format=yuv420p"
        )
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", str(self.path),
                "-vf", filters,
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read_frame(self) -> bytes:
        for _ in range(2):
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            frame = read_exact(self._proc.stdout, self.frame_size)
            if frame:
                return frame
            log.warning("视频解码中断，重启: %s", self.path)
            self._proc = None
        raise RuntimeError(f"无法从视频读取帧: {self.path}")

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


class AvatarMixer:
    """说话时切到 talk 素材，静默时回到 idle 素材；无 talk 时单视频循环。"""

    def __init__(self, config: AvatarConfig, root: Path):
        self.config = config
        idle_path = root / config.idle
        if not idle_path.exists():
            raise FileNotFoundError(
                f"缺少形象视频 {idle_path}。先运行 `goovo placeholder` 生成示例，"
                "或按 README 录制你自己的视频。"
            )
        self.idle = FrameSource(idle_path, config.width, config.height, config.fps)
        self.talk: FrameSource | None = None
        if config.talk:
            talk_path = root / config.talk
            if talk_path.exists():
                self.talk = FrameSource(talk_path, config.width, config.height, config.fps)
            else:
                log.warning("talk 视频不存在(%s)，退化为单视频循环", talk_path)
        self._active_is_talk = False

    def read_frame(self, speaking: bool, driver=None) -> bytes:
        use_talk = speaking and self.talk is not None
        if use_talk != self._active_is_talk:
            self._active_is_talk = use_talk
        source = self.talk if use_talk else self.idle
        return source.read_frame()

    def warmup(self) -> None:
        self.idle.read_frame()
        if self.talk is not None:
            self.talk.read_frame()

    def close(self) -> None:
        self.idle.close()
        if self.talk is not None:
            self.talk.close()


def make_avatar(config: AvatarConfig, root: Path):
    engine = config.engine
    if engine in ("auto", "neural"):
        from .neural import NeuralMouth, neural_available

        if neural_available(root):
            return NeuralMouth(config, root)
        if engine == "neural":
            raise FileNotFoundError(
                "缺少 MuseTalk MLX 环境或权重：按 README「神经口型」安装 .venv-muse 并下载权重"
            )
    meta = root / "assets" / "mouth" / "meta.json"
    if engine in ("auto", "mouth") and meta.exists():
        from .mouth import MouthAvatar

        return MouthAvatar(config, root)
    if engine == "mouth":
        raise FileNotFoundError("缺少口型素材：先运行 `goovo prepare-mouth` 从你的 talk 视频提取口型")
    return AvatarMixer(config, root)


def generate_placeholders(config: AvatarConfig, root: Path, seconds: int = 8) -> list[Path]:
    """生成示例形象视频（纯色渐变 + 中文提示），仅用于跑通流程。"""
    font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False
    ).stdout
    has_drawtext = " drawtext " in filters and Path(font).exists()
    created = []
    specs = [
        (config.idle or "assets/idle.mp4", "0x2b4a6f|0x1a2c42", "示例形象 · 静默待机", "请替换为你录制的闭嘴视频"),
        (config.talk or "assets/talk.mp4", "0x6f4a2b|0x422c1a", "示例形象 · 说话状态", "请替换为你录制的说话视频"),
    ]
    for name, colors, title, subtitle in specs:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        chain = (
            f"gradients=s={config.width}x{config.height}:speed=0.03:"
            f"c0={colors.split('|')[0]}:c1={colors.split('|')[1]},"
            f"fps={config.fps},format=yuv420p"
        )
        if has_drawtext:
            chain = (
                f"gradients=s={config.width}x{config.height}:speed=0.03:"
                f"c0={colors.split('|')[0]}:c1={colors.split('|')[1]},"
                f"drawtext=fontfile='{font}':text='{title}':fontcolor=white:fontsize=44:"
                f"x=(w-text_w)/2:y=h/2-80,"
                f"drawtext=fontfile='{font}':text='{subtitle}':fontcolor=0xcccccc:fontsize=28:"
                f"x=(w-text_w)/2:y=h/2,"
                f"fps={config.fps},format=yuv420p"
            )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", chain,
                "-t", str(seconds),
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                str(target),
            ],
            check=True,
        )
        created.append(target)
    return created
