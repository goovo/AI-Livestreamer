from __future__ import annotations

import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .brain import Brain
from .config import Config
from .pipeline import Pipeline

log = logging.getLogger("goovo.panel")

STATIC_DIR = Path(__file__).parent / "static"


class SayRequest(BaseModel):
    text: str
    interrupt: bool = False


class PauseRequest(BaseModel):
    paused: bool


class ModeRequest(BaseModel):
    mode: str


def create_app(config: Config, brain: Brain, pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="goovo-human 控制台")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status():
        return {**brain.snapshot(), **pipeline.state()}

    @app.post("/api/say")
    def say(request: SayRequest):
        text = request.text.strip()
        if not text:
            return JSONResponse({"ok": False, "error": "文本为空"}, status_code=400)
        brain.say(text, interrupt=request.interrupt)
        return {"ok": True}

    @app.post("/api/pause")
    def pause(request: PauseRequest):
        brain.paused = request.paused
        return {"ok": True, "paused": brain.paused}

    @app.post("/api/mode")
    def mode(request: ModeRequest):
        if request.mode not in ("playlist", "llm"):
            return JSONResponse({"ok": False, "error": "mode 必须是 playlist 或 llm"}, status_code=400)
        brain.mode = request.mode
        return {"ok": True, "mode": brain.mode}

    @app.get("/api/scripts")
    def scripts():
        return {"segments": brain._playlist}

    return app


class PanelServer:
    def __init__(self, config: Config, brain: Brain, pipeline: Pipeline):
        self.config = config
        app = create_app(config, brain, pipeline)
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=config.panel.host,
                port=config.panel.port,
                log_level="warning",
            )
        )
        self._thread = threading.Thread(target=self.server.run, name="panel", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("控制面板: http://%s:%s", self.config.panel.host, self.config.panel.port)

    def stop(self) -> None:
        self.server.should_exit = True
