from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .util import setup_logging, pcm_to_wav


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goovo", description="口播数字人智能体（微信视频号直播）")
    parser.add_argument("--config", "-c", help="配置文件路径（默认 config.local.toml / config.toml）")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="启动数字人直播")
    run.add_argument("--sink", choices=["preview", "rtmp", "record"], help="覆盖输出方式")
    run.add_argument("--rtmp", help="覆盖推流地址")
    run.add_argument("--mode", choices=["playlist", "llm"], help="覆盖口播模式")

    say = sub.add_parser("say", help="向运行中的数字人发送插播文本")
    say.add_argument("text")
    say.add_argument("--interrupt", "-i", action="store_true", help="打断当前口播")
    say.add_argument("--port", type=int, default=8765)

    tts = sub.add_parser("tts", help="测试语音合成并保存 wav")
    tts.add_argument("text")
    tts.add_argument("--out", default="output/tts_test.wav")

    sub.add_parser("doctor", help="检查运行环境")
    sub.add_parser("prepare-mouth", help="从你的 idle/talk 视频提取口型素材（音频驱动口型引擎）")
    sub.add_parser("prepare-neural", help="预计算 MuseTalk 神经口型的 idle 逐帧 latent（需 .venv-muse 与权重）")
    voice = sub.add_parser("prepare-voice", help="从 talk.mp4 音轨提取声音克隆参考音频")
    voice.add_argument("--start", type=float, default=4.0)
    voice.add_argument("--seconds", type=float, default=12.0)
    sub.add_parser("warmup", help="预合成全部口播稿到缓存（克隆语音慢时保证开播即播）")
    placeholder = sub.add_parser("placeholder", help="生成示例形象视频（跑通流程用）")
    placeholder.add_argument("--seconds", type=int, default=8)
    return parser


def cmd_run(args) -> int:
    import signal

    from .avatar import make_avatar
    from .brain import Brain
    from .panel import PanelServer
    from .pipeline import Pipeline
    from .tts import TTS

    def _graceful(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful)

    config = load_config(args.config)
    if args.sink:
        config.stream.sink = args.sink
    if args.rtmp:
        config.stream.rtmp_url = args.rtmp
    if args.mode:
        config.brain.mode = args.mode

    tts = TTS(config.voice, config.root / ".cache" / "tts")
    brain = Brain(config.brain, tts, config.root)
    avatar = make_avatar(config.avatar, config.root)
    pipeline = Pipeline(config, brain, avatar)
    panel = PanelServer(config, brain, pipeline)

    brain.start()
    panel.start()
    print(f"▶ 数字人已启动：输出={config.stream.sink} 模式={config.brain.mode}")
    print(f"  控制面板 http://{config.panel.host}:{config.panel.port}  (Ctrl+C 停止)")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        pipeline.stop()
        brain.stop()
        panel.stop()
        avatar.close()
    return 0


def cmd_say(args) -> int:
    import httpx

    response = httpx.post(
        f"http://127.0.0.1:{args.port}/api/say",
        json={"text": args.text, "interrupt": args.interrupt},
        timeout=10,
    )
    data = response.json()
    print("已发送" if data.get("ok") else f"失败: {data}")
    return 0 if data.get("ok") else 1


def cmd_tts(args) -> int:
    from .tts import TTS

    config = load_config(args.config)
    tts = TTS(config.voice, config.root / ".cache" / "tts")
    started = time.time()
    pcm, _ = tts.synthesize(args.text)
    target = config.path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pcm_to_wav(pcm))
    print(f"✓ 合成 {len(pcm) / 88200:.1f}s 音频，耗时 {time.time() - started:.1f}s → {target}")
    return 0


def cmd_doctor(args) -> int:
    config = load_config(args.config)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "✓" if passed else "✗"
        print(f"{mark} {label}" + (f" — {detail}" if detail else ""))
        ok = ok and passed

    check("ffmpeg", shutil.which("ffmpeg") is not None)
    check("ffplay（预览需要）", shutil.which("ffplay") is not None)
    idle = config.path(config.avatar.idle)
    check("idle 形象视频", idle.exists(), str(idle))
    talk = config.path(config.avatar.talk) if config.avatar.talk else None
    check("talk 形象视频", talk is not None and talk.exists(),
          str(talk) if talk else "未配置（可选，缺失时为单视频循环）")
    scripts = config.path(config.brain.scripts_dir)
    count = len(list(scripts.glob("*.md")) + list(scripts.glob("*.txt"))) if scripts.exists() else 0
    check("口播稿", count > 0, f"{scripts} 共 {count} 个文件")
    if config.stream.sink == "rtmp":
        check("推流地址", config.stream.rtmp_url.startswith(("rtmp://", "rtmps://")), "config.toml [stream].rtmp_url")
    if config.brain.mode == "llm":
        import os
        env = config.brain.llm.api_key_env
        check(f"LLM API Key ({env})", bool(os.environ.get(env)))
    try:
        from .tts import TTS
        tts = TTS(config.voice, config.root / ".cache" / "tts")
        pcm, _ = tts.synthesize("测试。")
        check("TTS 合成", len(pcm) > 0, f"{config.voice.engine} / {config.voice.voice}")
    except Exception as exc:
        check("TTS 合成", False, str(exc))
    if config.avatar.engine in ("auto", "mouth"):
        meta = config.root / "assets" / "mouth" / "meta.json"
        check("口型素材（mouth 引擎）", meta.exists(), str(meta) if meta.exists() else "运行 `goovo prepare-mouth` 生成")
    if config.avatar.engine in ("auto", "neural"):
        from .neural import neural_available

        check("神经口型（MuseTalk MLX / Apple GPU）", neural_available(config.root),
              "见 README「神经口型」" if not neural_available(config.root) else "engine=auto 优先启用")
    print("\n环境就绪，运行 `goovo run` 开始。" if ok else "\n请先补齐上面 ✗ 的项目。")
    return 0 if ok else 1


def cmd_placeholder(args) -> int:
    from .avatar import generate_placeholders

    config = load_config(args.config)
    created = generate_placeholders(config.avatar, config.root, seconds=args.seconds)
    for path in created:
        print(f"✓ 已生成 {path}")
    print("这是占位素材，正式直播前请按 README 录制你自己的视频。")
    return 0


def cmd_prepare_mouth(args) -> int:
    from .mouth import prepare_mouth_assets

    config = load_config(args.config)
    target = prepare_mouth_assets(config.avatar, config.root)
    print(f"✓ 口型素材已生成: {target}")
    print("重启 `goovo run` 即自动启用音频驱动口型（engine=auto）。")
    return 0


def cmd_prepare_neural(args) -> int:
    from .neural import NeuralMouth, neural_available

    config = load_config(args.config)
    if not neural_available(config.root):
        print("✗ 缺少 .venv-muse 或 MuseTalk 权重，见 README「神经口型」")
        return 1
    avatar = NeuralMouth(config.avatar, config.root)
    avatar.close()
    print("✓ 神经口型 latent 已就绪，重启 `goovo run` 即启用（engine=auto/neural）。")
    return 0


def cmd_prepare_voice(args) -> int:
    config = load_config(args.config)
    source = config.path(config.avatar.talk or config.avatar.idle)
    target = config.root / "assets" / "voice" / "ref.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(args.start), "-t", str(args.seconds),
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(target),
        ],
        check=True,
    )
    print(f"✓ 参考音频: {target}（{args.seconds}s）")
    print("启动克隆服务: .venv-clone/bin/python tools/clone_server.py")
    return 0


def cmd_warmup(args) -> int:
    from .brain import Brain
    from .tts import TTS

    config = load_config(args.config)
    tts = TTS(config.voice, config.root / ".cache" / "tts")
    brain = Brain(config.brain, tts, config.root)
    total = len(brain._playlist)
    for index, text in enumerate(brain._playlist, 1):
        started = time.time()
        pcm, _ = tts.synthesize(text)
        print(f"[{index}/{total}] {len(pcm) / 88200:.1f}s 音频 / 耗时 {time.time() - started:.1f}s — {text[:24]}")
    print("✓ 全部口播稿已缓存，开播即播。")
    if config.avatar.engine in ("auto", "neural"):
        from .neural import NeuralMouth, neural_available

        if neural_available(config.root):
            avatar = NeuralMouth(config.avatar, config.root)
            avatar.prerender(brain.playlist_texts(), tts)
            avatar.close()
            print("✓ 神经口型缓存已就绪，开播即播。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    handlers = {
        "run": cmd_run,
        "say": cmd_say,
        "tts": cmd_tts,
        "doctor": cmd_doctor,
        "placeholder": cmd_placeholder,
        "prepare-mouth": cmd_prepare_mouth,
        "prepare-neural": cmd_prepare_neural,
        "prepare-voice": cmd_prepare_voice,
        "warmup": cmd_warmup,
    }
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
