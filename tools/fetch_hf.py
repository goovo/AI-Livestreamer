"""多线程分块下载 HuggingFace 模型文件（单连接被限速时用）。

用法: python tools/fetch_hf.py <repo> <out_dir> <file> [<file> ...]
环境变量: HF_ENDPOINT（默认 https://huggingface.co），FETCH_THREADS（默认 16）
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
THREADS = int(os.environ.get("FETCH_THREADS", "16"))
CHUNK = 8 * 1024 * 1024
SOURCE = os.environ.get("FETCH_SOURCE", "hf")


def _url(repo: str, name: str) -> str:
    if SOURCE == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
    return f"{ENDPOINT}/{repo}/resolve/main/{name}"


def _request(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        length = resp.headers.get("Content-Length")
        if length:
            return int(length)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Range"].split("/")[-1])


def fetch(repo: str, out_dir: Path, name: str) -> None:
    url = _url(repo, name)
    target = out_dir / name
    if target.exists() and target.stat().st_size == _size(url):
        print(f"{name}: 已存在，跳过", flush=True)
        return
    total = _size(url)
    parts = sorted(range(0, total, CHUNK))
    tmp = out_dir / f".{name}.parts"
    tmp.mkdir(parents=True, exist_ok=True)
    done_bytes = 0
    last_print = time.time()

    def grab(index: int) -> int:
        nonlocal done_bytes, last_print
        start = parts[index]
        end = min(start + CHUNK, total) - 1
        part = tmp / f"{index:05d}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return end - start + 1
        for attempt in range(5):
            try:
                data = _request(url, start, end)
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)}")
                part.write_bytes(data)
                return len(data)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        return 0

    started = time.time()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for got in pool.map(grab, range(len(parts))):
            done_bytes += got
            now = time.time()
            if now - last_print > 5:
                last_print = now
                rate = done_bytes / 1024 / max(now - started, 1)
                print(f"{name}: {done_bytes/1e6:.0f}/{total/1e6:.0f} MB  {rate:.0f} KB/s", flush=True)
    with open(target, "wb") as out:
        for index in range(len(parts)):
            out.write((tmp / f"{index:05d}").read_bytes())
    for index in range(len(parts)):
        (tmp / f"{index:05d}").unlink()
    tmp.rmdir()
    print(f"{name}: 完成 {total/1e6:.0f} MB，用时 {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    repo, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[3:]:
        fetch(repo, out, name)
