"""MuseTalk MLX 口型渲染 worker（独立环境 .venv-muse 运行，Apple GPU）。

协议: stdin/stdout 每行一个 JSON。
  {"cmd":"prepare","idle":...,"width":...,"height":...,"fps":...}
  {"cmd":"render","id":...,"wav":...,"start":...,"frames":...,"out":...}
  {"cmd":"quit"}
事件: {"event":"ready"} / {"event":"progress",...} / {"ok":true,...} / {"ok":false,"error":...}

直接运行 `python tools/musetalk_worker.py proto ...` 可离线生成对比样片。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / ".cache" / "models" / "musetalk-mlx-fp16"
LANDMARK_MODEL = ROOT / ".cache" / "models" / "face_landmarker.task"
TRIO = [61, 291, 1]


def _decode_frames(path: Path, width: int, height: int, fps: int):
    proc = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                   f"crop={width}:{height},fps={fps},format=yuv420p",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    size = width * height * 3 // 2
    while True:
        buf = proc.stdout.read(size)
        if not buf or len(buf) < size:
            break
        array = np.frombuffer(buf, dtype=np.uint8).reshape(height * 3 // 2, width)
        yield cv2.cvtColor(array, cv2.COLOR_YUV2BGR_I420)
    proc.kill()
    proc.wait()


def _face_boxes(frames_bgr, width: int, height: int):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    boxes = []
    trios = []
    for bgr in frames_bgr:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not result.face_landmarks:
            boxes.append(None)
            trios.append(None)
            continue
        pts = np.array([(lm.x * width, lm.y * height) for lm in result.face_landmarks[0]])
        trios.append(pts[TRIO].copy())
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = (x1 - x0) / 2 * 1.12, (y1 - y0) / 2 * 1.12
        boxes.append([cx - half_w, cy - half_h, cx + half_w, cy + half_h])
    detector.close()
    filled = [b for b in boxes if b is not None]
    if not filled:
        raise RuntimeError("idle 视频未检测到人脸")
    median = np.median(filled, axis=0)
    boxes = [b if b is not None else median.tolist() for b in boxes]
    trio_filled = [t for t in trios if t is not None]
    trio_median = np.median(trio_filled, axis=0)
    trios = [t if t is not None else trio_median.copy() for t in trios]
    arr = np.array(boxes, dtype=np.float32)
    smooth = arr.copy()
    for _ in range(3):
        prev = np.roll(smooth, 1, axis=0)
        nxt = np.roll(smooth, -1, axis=0)
        smooth = 0.25 * prev + 0.5 * smooth + 0.25 * nxt
    trio_arr = np.array(trios, dtype=np.float32)
    trio_smooth = trio_arr.copy()
    for _ in range(3):
        trio_smooth = 0.25 * np.roll(trio_smooth, 1, axis=0) + 0.5 * trio_smooth + 0.25 * np.roll(trio_smooth, -1, axis=0)
    return np.clip(smooth, 0, [width, height, width, height]).astype(np.int32), trio_smooth


def prepare_latents(pipe, idle: Path, width: int, height: int, fps: int, out: Path) -> int:
    import mlx.core as mx

    frames = list(_decode_frames(idle, width, height, fps))
    boxes, trios = _face_boxes(frames, width, height)
    center = trios.mean(axis=1)
    canonical = int(np.argmin(np.linalg.norm(center - np.median(center, axis=0), axis=1)))
    x0, y0, x1, y1 = boxes[canonical]
    crop = cv2.resize(frames[canonical][y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
    stack = np.array(pipe.get_latents_for_unet(crop)).astype(np.float16)
    canon_trio = (trios[canonical] - np.array([x0, y0])) * np.array([256 / (x1 - x0), 256 / (y1 - y0)])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, latents=stack, boxes=boxes, trios=trios, canon_trio=canon_trio.astype(np.float32),
             width=width, height=height, fps=fps,
             idle_key=f"{idle.stat().st_size}:{int(idle.stat().st_mtime)}")
    mx.eval(mx.array(stack))
    return len(frames)


def render_segment(pipe, latents: np.ndarray, wav: Path, frames: int,
                   out: Path, fps: int, batch: int = 8, on_progress=None) -> int:
    import mlx.core as mx

    chunks, _ = pipe.encode_audio_from_wav(str(wav), fps=fps)
    count = min(frames, int(chunks.shape[0]))
    if count <= 0:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    stack = mx.concatenate([mx.array(latents)] * count, axis=0)
    started = time.time()
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        faces = pipe.render(stack[begin:end], chunks[begin:end])
        for offset, face in enumerate(faces):
            cv2.imwrite(str(out / f"f_{begin + offset:05d}.jpg"), face,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if on_progress:
            on_progress(end, count, time.time() - started)
    (out / "manifest.json").write_text(json.dumps({"frames": count, "fps": fps}))
    return count


def _serve() -> None:
    import mlx.core as mx
    from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline

    pipe = MuseTalkPipeline.from_pretrained_mlx(WEIGHTS)
    assets = ROOT / "assets" / "mouth"
    cache: dict = {}

    def emit(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def load_cache() -> dict:
        path = assets / "neural.npz"
        if not path.exists():
            return {}
        data = np.load(path)
        if "canon_trio" not in data:
            return {}
        return {"latents": data["latents"], "boxes": data["boxes"],
                "trios": data["trios"], "canon_trio": data["canon_trio"],
                "fps": int(data["fps"]), "idle_key": str(data["idle_key"])}

    cache = load_cache()
    if cache:
        dummy = mx.zeros((8, 50, 384), dtype=mx.float16)
        pipe.render(mx.concatenate([mx.array(cache["latents"])] * 8), dummy)
    emit({"event": "ready", "prepared": bool(cache)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                break
            if cmd == "prepare":
                idle = ROOT / req["idle"]
                count = prepare_latents(pipe, idle, req["width"], req["height"], req["fps"],
                                        assets / "neural.npz")
                cache = load_cache()
                dummy = mx.zeros((8, 50, 384), dtype=mx.float16)
                pipe.render(mx.concatenate([mx.array(cache["latents"])] * 8), dummy)
                emit({"ok": True, "cmd": "prepare", "frames": count})
            elif cmd == "render":
                if not cache:
                    raise RuntimeError("尚未 prepare latent")
                def progress(done, total, elapsed):
                    emit({"event": "progress", "id": req["id"], "done": done,
                          "total": total, "fps": round(done / max(elapsed, 1e-6), 1)})
                count = render_segment(pipe, cache["latents"], Path(req["wav"]),
                                       req["frames"], Path(req["out"]), cache["fps"],
                                       on_progress=progress)
                emit({"ok": True, "cmd": "render", "id": req["id"], "frames": count})
            else:
                emit({"ok": False, "error": f"未知命令 {cmd}"})
        except Exception as exc:
            emit({"ok": False, "id": req.get("id"), "error": f"{type(exc).__name__}: {exc}"})
    mx.clear_cache()


def _proto() -> None:
    """离线样片: python tools/musetalk_worker.py proto <wav> <out.mkv> [seconds]"""
    import mlx.core as mx
    from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline

    wav, out = Path(sys.argv[2]), Path(sys.argv[3])
    limit = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    width, height, fps = 1080, 1920, 30
    pipe = MuseTalkPipeline.from_pretrained_mlx(WEIGHTS)
    assets = ROOT / "assets" / "mouth"
    stale = True
    if (assets / "neural.npz").exists():
        stale = "canon_trio" not in np.load(assets / "neural.npz")
    if stale:
        t0 = time.time()
        count = prepare_latents(pipe, ROOT / "assets" / "idle.mp4", width, height, fps, assets / "neural.npz")
        print(f"prepare {count} frames in {time.time()-t0:.1f}s", flush=True)
    data = np.load(assets / "neural.npz")
    latents, boxes, trios = data["latents"], data["boxes"], data["trios"]
    canon_trio = data["canon_trio"]
    frames = int(limit * fps)
    t0 = time.time()
    render_segment(pipe, latents, wav, frames, ROOT / ".cache" / "diagnostics" / "neural-faces", fps,
                   on_progress=lambda d, t, e: print(f"render {d}/{t} {d/e:.1f} fps", flush=True))
    print(f"render {frames} frames in {time.time()-t0:.1f}s", flush=True)
    faces_dir = ROOT / ".cache" / "diagnostics" / "neural-faces"
    writer = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
         "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(out)],
        stdin=subprocess.PIPE,
    )
    strip = []
    for index, bgr in enumerate(_decode_frames(ROOT / "assets" / "idle.mp4", width, height, fps)):
        if index >= frames:
            break
        face_path = faces_dir / f"f_{index:05d}.jpg"
        if face_path.exists():
            bgr = paste_face(bgr, cv2.imread(str(face_path)), boxes[index], canon_trio, trios[index])
        writer.stdin.write(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).tobytes())
        if index % 12 == 0:
            x0, y0, x1, y1 = boxes[index]
            strip.append(bgr[max(0, y0):y1, max(0, x0):x1])
    writer.stdin.close()
    writer.wait()
    if strip:
        thumbs = [cv2.resize(s, (180, 180)) for s in strip[:40]]
        rows = [np.hstack(thumbs[i:i + 10]) for i in range(0, len(thumbs), 10)]
        cv2.imwrite(str(out.with_name(out.stem + "-strip.jpg")), np.vstack(rows))
    print("done", out, flush=True)


def paste_face(bgr: np.ndarray, face: np.ndarray, box, canon_trio=None, trio=None) -> np.ndarray:
    """把 256² 生成脸贴回原帧：姿态仿射对齐 + 下半脸椭圆羽化 + 肤色均值对齐。"""
    if canon_trio is not None and trio is not None:
        bx0, by0, bx1, by1 = [int(v) for v in box]
        local = (np.asarray(trio, dtype=np.float32) - np.array([bx0, by0], dtype=np.float32)) * np.array(
            [256 / max(bx1 - bx0, 1), 256 / max(by1 - by0, 1)], dtype=np.float32)
        matrix = cv2.getAffineTransform(np.asarray(canon_trio, dtype=np.float32), local)
        face = cv2.warpAffine(face, matrix, (256, 256), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    if bw < 16 or bh < 16:
        return bgr
    height, width = bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    bw, bh = x1 - x0, y1 - y0
    resized = cv2.resize(face, (bw, bh), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((bh, bw), dtype=np.float32)
    cv2.ellipse(mask, (bw // 2, int(bh * 0.72)), (int(bw * 0.36), int(bh * 0.24)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=bw * 0.04, sigmaY=bh * 0.04)
    region = bgr[y0:y1, x0:x1].astype(np.float32)
    weight = mask[..., None]
    total = max(float(mask.sum()), 1e-6)
    shift = np.clip((region * weight).sum(axis=(0, 1)) / total
                    - (resized.astype(np.float32) * weight).sum(axis=(0, 1)) / total, -18.0, 18.0)
    bgr[y0:y1, x0:x1] = (region * (1 - weight) + (resized.astype(np.float32) + shift) * weight).astype(np.uint8)
    return bgr


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "proto":
        _proto()
    else:
        _serve()
