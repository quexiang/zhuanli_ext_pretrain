"""FastAPI application entry point.

Serves the extraction API and the single-page frontend.
"""

# ⚠️ torch MUST be imported before anything else.
# PaddlePaddle's DLL loading on Windows corrupts the system DLL search path,
# causing torch's shm.dll to fail with "找不到指定的程序" (WinError 127).
# Loading torch first ensures its DLLs are resolved before the path is broken.
import torch  # noqa: F401, E402

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Startup: log output directory."""
        logger.info(
            "%s started — output dir: %s",
            settings.app_name,
            settings.output_dir.resolve(),
        )
        yield  # no shutdown cleanup needed

    app = FastAPI(
        title=settings.app_name,
        version="1.1.0",
        description=(
            "上传专利说明书PDF（单个或ZIP包）或提交HTTP/HTTPS链接，"
            "自动提取文本并生成JSONL格式数据，用于大模型预训练。"
        ),
        lifespan=lifespan,
    )

    # ── Routers ─────────────────────────────────────────────
    app.include_router(extraction.router)

    # ── Static files ────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Index page ──────────────────────────────────────────
    index_html_path = Path(__file__).parent / "templates" / "index.html"
    with open(index_html_path, encoding="utf-8") as f:
        _index_html = f.read()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return HTMLResponse(_index_html)

    return app


app = create_app()
