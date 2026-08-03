"""FastAPI application entry point.

Serves the extraction API and the single-page frontend.
"""

import logging
<<<<<<< HEAD
from contextlib import asynccontextmanager
=======
>>>>>>> 9652865fbd6f3bd0c7da69f3370098de75f83281
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
<<<<<<< HEAD
    # 生命周期函数，替代旧的 on_event
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动逻辑（原startup）
        logger.info(
            "%s started — output dir: %s",
            settings.app_name,
            settings.output_dir.resolve(),
        )
        yield  # 服务正常运行阶段
        # 后续如果需要服务关闭清理代码，写在这里

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="上传专利说明书或标准文献PDF（单个或ZIP包），自动提取文本并生成JSONL格式数据，用于大模型预训练。",
        lifespan=lifespan  # 绑定生命周期
=======
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="上传专利说明书PDF（单个或ZIP包），自动提取文本并生成JSONL格式数据，用于大模型预训练。",
>>>>>>> 9652865fbd6f3bd0c7da69f3370098de75f83281
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

<<<<<<< HEAD
    return app


app = create_app()
=======
    # ── Lifespan ────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        logger.info(
            "%s started — output dir: %s",
            settings.app_name,
            settings.output_dir.resolve(),
        )

    return app


app = create_app()
>>>>>>> 9652865fbd6f3bd0c7da69f3370098de75f83281
