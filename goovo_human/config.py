from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AvatarConfig:
    idle: str = "assets/idle.mp4"
    talk: str = ""
    width: int = 720
    height: int = 1280
    fps: int = 25
    engine: str = "auto"


@dataclass
class HttpVoiceConfig:
    url: str = "http://127.0.0.1:9880/tts"
    method: str = "POST"
    json_body: str = '{"text": "{text}"}'


@dataclass
class VoiceConfig:
    engine: str = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    fallback_macos: bool = True
    macos_voice: str = "Tingting"
    fallback_voice: str = "zh-CN-YunxiNeural"
    http: HttpVoiceConfig = field(default_factory=HttpVoiceConfig)


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_chars: int = 140
    temperature: float = 0.8


@dataclass
class BrainConfig:
    mode: str = "playlist"
    scripts_dir: str = "scripts"
    shuffle: bool = False
    gap_seconds: float = 2.0
    prefetch: int = 2
    persona: str = ""
    topics: list[str] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class StreamConfig:
    sink: str = "preview"
    rtmp_url: str = ""
    record_path: str = "output/live.mkv"
    video_bitrate: str = "2500k"
    audio_bitrate: str = "128k"
    preview_scale: float = 0.5


@dataclass
class PanelConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class Config:
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    root: Path = field(default_factory=Path.cwd)

    def path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate


_SECTIONS = {
    "avatar": AvatarConfig,
    "voice": VoiceConfig,
    "brain": BrainConfig,
    "stream": StreamConfig,
    "panel": PanelConfig,
}


def _build(cls, data: dict):
    kwargs = {}
    hints = cls.__dataclass_fields__
    for key, value in data.items():
        if key not in hints:
            continue
        if key == "http" and cls is VoiceConfig:
            kwargs[key] = _build(HttpVoiceConfig, value)
        elif key == "llm" and cls is BrainConfig:
            kwargs[key] = _build(LLMConfig, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        for name in ("config.local.toml", "config.toml"):
            candidate = Path.cwd() / name
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return Config()
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = Config(root=path.resolve().parent)
    for section, cls in _SECTIONS.items():
        if section in raw:
            setattr(config, section, _build(cls, raw[section]))
    return config
