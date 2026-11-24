"""声音克隆服务：F5-TTS 零样本克隆，参考音频取自你自己的 talk.mp4。

运行（独立 venv）：.venv-clone/bin/python tools/clone_server.py
主程序通过 [voice.http] 指向 http://127.0.0.1:9880/tts 使用。
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "assets" / "voice" / "ref.wav"
REF_TEXT_FILE = ROOT / "assets" / "voice" / "ref.txt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"加载 F5-TTS（device={device}）…", flush=True)
from f5_tts.api import F5TTS

model = F5TTS(model="F5TTS_v1_Base", device=device)

ref_text = ""
if REF_TEXT_FILE.exists():
    ref_text = REF_TEXT_FILE.read_text(encoding="utf-8").strip()
if not ref_text:
    print("转写参考音频（whisper）…", flush=True)
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    ref_text = asr(str(REF), generate_kwargs={"task": "transcribe", "language": "zh"})["text"].strip()
    REF_TEXT_FILE.write_text(ref_text, encoding="utf-8")
print(f"参考文本: {ref_text}", flush=True)

lock = threading.Lock()
app = FastAPI(title="goovo clone voice")


class TTSRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "device": device, "ref_text": ref_text}


@app.post("/tts")
def tts(request: TTSRequest):
    with lock:
        wav, sr, _ = model.infer(
            ref_file=str(REF),
            ref_text=ref_text,
            gen_text=request.text,
            file_wave=None,
            nfe_step=16,
        )
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9880, log_level="warning")
