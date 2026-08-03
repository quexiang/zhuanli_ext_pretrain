"""Upload, extraction, and download endpoints with adaptive parallelism."""

from __future__ import annotations

import asyncio
import logging
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import parallel_config, settings
from app.schemas.models import ErrorResponse, ExtractResponse, TaskStatus
from app.services.pdf_extractor import extract_page_images, extract_text_direct
from app.services.ocr_engine import OCREngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["extraction"])

# Lazy‑initialised OCR engine (shared across *async* requests only;
# not used in subprocess workers which create their own instances).
ocr_engine = OCREngine()

# In‑memory task store for background ZIP processing
_task_store: dict[str, TaskStatus] = {}

# Global shared process executor — created once from adaptive config.
# Used for:
#   1. Background ZIP processing (N workers)
#   2. Offloading single-PDF OCR from the event loop (1 worker at a time)
_shared_executor: ProcessPoolExecutor | None = None


def _get_shared_executor() -> ProcessPoolExecutor:
    """Return or create the shared process pool.

    Pool size comes from ``parallel_config.max_workers`` (adaptive).
    """
    global _shared_executor
    if _shared_executor is None:
        _shared_executor = ProcessPoolExecutor(
            max_workers=parallel_config.max_workers,
        )
        logger.info(
            "Created shared executor: %d workers (strategy=%s)",
            parallel_config.max_workers,
            parallel_config.strategy,
        )
    return _shared_executor


# ═══════════════════════════════════════════════════════════════════
# Subprocess worker function (must be at module level for pickle)
# ═══════════════════════════════════════════════════════════════════
def _process_pdf_worker(pdf_bytes: bytes) -> list[dict[str, str]]:
    """Process a single PDF in a subprocess worker.

    Each worker creates its **own** ``OCREngine`` instance with a fresh
    PaddleOCR model — required because PaddleOCR is not thread-safe.
    """
    from app.services.text_processor import process_document

    engine = OCREngine.create_detached()
    page_images = extract_page_images(pdf_bytes, dpi=settings.ocr_dpi)
    if not page_images:
        return []

    full_text = engine.process_pdf(page_images)
    if not full_text.strip():
        return []

    return process_document(
        full_text,
        min_len=settings.segment_min_len,
        max_len=settings.segment_max_len,
        category=settings.category,
    )


# app/routers/extraction.py

# app/routers/extraction.py

def _process_pdf_standard_worker(pdf_bytes: bytes) -> list[dict[str, str]]:
    from app.services.text_processor import clean_standard_text, segment_standard_text
    from app.services.text_quality import is_valid_text_content
    from app.services.pdf_extractor import extract_page_images
    from app.services.ocr_engine import OCREngine

    full_text = extract_text_direct(pdf_bytes)
    stripped = full_text.strip()

    is_valid, reason = is_valid_text_content(stripped)

    if is_valid:
        logger.info("✅ 标准模式：纯文本提取（CJK 合格）")
        cleaned = clean_standard_text(full_text)
        segments = segment_standard_text(cleaned, settings.segment_min_len, settings.segment_max_len)
        return [{"text": seg, "category": "标准文献"} for seg in segments]

    # ===== OCR 降级分支 =====
    logger.warning("⚠️ 标准模式：文本质量不合格（%s），降级到 OCR", reason)

    page_images = extract_page_images(pdf_bytes, dpi=settings.ocr_dpi)
    if not page_images:
        raise RuntimeError("PDF 页面渲染失败，无法进行 OCR 识别。")

    logger.info("📸 标准模式降级 OCR：%d 页图像已渲染，开始识别...", len(page_images))

    engine = OCREngine.create_detached()
    full_text_ocr = engine.process_pdf(page_images)

    if not full_text_ocr.strip():
        raise RuntimeError("OCR 识别未产生任何文本，该 PDF 可能无法识别。")

    # ═══════════════════════════════════════════════════════════
    # 在这里插入 3 行调试日志 ⬇️
    # ═══════════════════════════════════════════════════════════
    logger.info("🔍 调试 A - OCR 原始文本长度: %d 字符", len(full_text_ocr.strip()))

    cleaned = clean_standard_text(full_text_ocr)
    logger.info("🔍 调试 B - 清洗后文本长度: %d 字符", len(cleaned.strip()))

    segments = segment_standard_text(cleaned, settings.segment_min_len, settings.segment_max_len)
    logger.info("🔍 调试 C - 分段后段落数: %d 段", len(segments))
    # ═══════════════════════════════════════════════════════════
    # 调试日志结束 ⬆️
    # ═══════════════════════════════════════════════════════════

    return [{"text": seg, "category": "标准文献"} for seg in segments]

    def _is_garbage_text(text: str) -> bool:
        """Detect if text is likely font-encoding garbage rather than real content.

        Scanned PDFs sometimes cause PyMuPDF to extract font tables
        (ASCII symbols, control chars) instead of actual document text.
        For Chinese-language documents, a very low CJK ratio indicates this.
        """
        if not text:
            return True
        stripped = text.strip()
        if len(stripped) < 100:
            return True
        cjk_count = sum(1 for c in stripped if '一' <= c <= '鿿')
        total_printable = sum(1 for c in stripped if c.isprintable() and not c.isspace())
        if total_printable == 0:
            return True
        cjk_ratio = cjk_count / total_printable
        return cjk_ratio < 0.05  # Less than 5% CJK → likely garbage

    # Fallback to OCR if no text found or text quality is poor
    use_ocr = not full_text or len(full_text.strip()) < 100 or _is_garbage_text(full_text)

    if use_ocr:
        reason = ""
        if not full_text:
            reason = "empty"
        elif len(full_text.strip()) < 100:
            reason = f"short ({len(full_text.strip())} chars)"
        else:
            reason = "garbage (low CJK ratio)"
        logger.info("Direct text extraction %s, falling back to OCR", reason)
        engine = OCREngine.create_detached()
        page_images = extract_page_images(pdf_bytes, dpi=settings.ocr_dpi)
        if not page_images:
            return []
        full_text = engine.process_pdf(page_images)
        if not full_text.strip():
            return []
        # Use patent pipeline for OCR fallback
        from app.services.text_processor import process_document
        return process_document(
            full_text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=settings.category,
        )

    # Standard mode: clean and segment
    cleaned = clean_standard_text(full_text)
    segments = segment_standard_text(cleaned, settings.segment_min_len, settings.segment_max_len)
    return [{"text": seg, "category": settings.category} for seg in segments]


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════
@router.post(
    "/extract",
    response_model=ExtractResponse | TaskStatus,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def extract_pdfs(
    file: UploadFile = File(...),
    mode: str = "patent",
    background_tasks: BackgroundTasks = None,
):
    """Upload a single PDF or a ZIP archive of PDFs for text extraction.

    - **mode=patent** (default): Scan image PDF → OCR → patent text processing.
    - **mode=standard**: Direct PyMuPDF text → standard cleaning & segmentation.
      Falls back to OCR if no embedded text is found (< 100 chars).

    - **Single PDF**: processed asynchronously in the shared process pool.
    - **ZIP archive**: processed in the background with multi-process parallelism.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".zip"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf and .zip are accepted.",
        )

    if mode not in ("patent", "standard"):
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be 'patent' or 'standard'.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    if ext == ".zip":
        return _handle_zip_upload(content, file.filename, mode, background_tasks)
    return await _handle_single_pdf(content, file.filename, mode)


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    """Poll processing progress for a background ZIP extraction task."""
    status = _task_store.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return status


@router.get("/download/{filename:path}")
async def download_jsonl(filename: str):
    """Download a previously generated JSONL file."""
    safe_path = (settings.output_dir / filename).resolve()
    if not str(safe_path).startswith(str(settings.output_dir.resolve())):
        raise HTTPException(status_code=403, detail="Invalid filename.")

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        safe_path,
        media_type="application/octet-stream",
        filename=filename,
    )


# ═══════════════════════════════════════════════════════════════════
# Single-PDF handler (offloaded to process pool)
# ═══════════════════════════════════════════════════════════════════
async def _handle_single_pdf(
    content: bytes,
    filename: str,
    mode: str = "patent",
) -> ExtractResponse:
    """Process a single PDF asynchronously in the shared process pool.

    The CPU-intensive work runs in a subprocess worker so the async
    event loop is not blocked.
    """
    from app.services.text_processor import write_jsonl

    loop = asyncio.get_event_loop()

    worker = _process_pdf_worker if mode == "patent" else _process_pdf_standard_worker

    # Offload processing to a worker process
    records = await loop.run_in_executor(
        _get_shared_executor(),
        worker,
        content,
    )

    if not records:
        if mode == "patent":
            detail = "OCR produced no text. The PDF may be unreadable."
        else:
            detail = "No text extracted. The PDF may contain only images and OCR is unavailable."
        raise HTTPException(status_code=400, detail=detail)

    # Write output file (fast I/O, stay in async context)
    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{stem}_{ts}.jsonl"
    out_path = settings.output_dir / out_name
    write_jsonl(records, out_path)

    preview = records[:3]

    return ExtractResponse(
        filename=out_name,
        record_count=len(records),
        preview=preview,
    )


# ═══════════════════════════════════════════════════════════════════
# ZIP upload & background processing
# ═══════════════════════════════════════════════════════════════════
def _handle_zip_upload(
    content: bytes,
    filename: str,
    mode: str = "patent",
    background_tasks: BackgroundTasks | None = None,
) -> TaskStatus:
    """Start background processing for a ZIP archive."""
    zip_path = settings.upload_dir / f"{uuid.uuid4().hex}_{filename}"
    with open(zip_path, "wb") as f:
        f.write(content)

    task_id = uuid.uuid4().hex
    status = TaskStatus(task_id=task_id, status="processing")
    _task_store[task_id] = status

    if background_tasks:
        background_tasks.add_task(_process_zip_background, task_id, zip_path, mode)
    else:
        _process_zip_background(task_id, zip_path, mode)

    return status


def _process_zip_background(task_id: str, zip_path: Path, mode: str = "patent"):
    """Background task: process all PDFs inside the ZIP in parallel.

    Uses a ``ProcessPoolExecutor`` sized by the adaptive parallel config.
    Each PDF gets its own worker process.
    """
    from app.services.text_processor import write_jsonl

    worker = _process_pdf_worker if mode == "patent" else _process_pdf_standard_worker

    status = _task_store[task_id]
    pdf_files: list[tuple[str, bytes]] = []

    # 1) Extract all PDFs from ZIP into memory
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".pdf"):
                    pdf_files.append((name, zf.read(name)))

        if not pdf_files:
            status.status = "failed"
            status.error = "ZIP contains no PDF files."
            return

        status.total_files = len(pdf_files)
        status.processed_files = 0

        # 2) Submit all PDFs to the process pool
        all_records: list[dict[str, str]] = []
        executor = _get_shared_executor()
        futures = {
            executor.submit(worker, pdf_bytes): pdf_name
            for pdf_name, pdf_bytes in pdf_files
        }

        # 3) Collect results as they complete (unordered)
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            pdf_name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                logger.error("Worker failed for %s: %s", pdf_name, exc)
            # Update progress (unordered, but total count is accurate)
            status.processed_files += 1
            status.current_file = Path(pdf_name).name

        # 4) Write merged output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"batch_{Path(zip_path).stem}_{ts}.jsonl"
        out_path = settings.output_dir / out_name
        if all_records:
            write_jsonl(all_records, out_path)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")

        status.status = "completed"
        status.record_count = len(all_records)
        status.output_filename = out_name

    except zipfile.BadZipFile:
        status.status = "failed"
        status.error = "Invalid ZIP file."
    except Exception as exc:
        status.status = "failed"
        status.error = str(exc)
        logger.exception("Background ZIP processing failed.")
    finally:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass
