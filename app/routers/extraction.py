"""Upload, extraction, and download endpoints with adaptive parallelism."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# Fix Docker container — glibc vfork kills child processes, use spawn instead
multiprocessing.set_start_method("spawn", force=True)

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import parallel_config, settings
from app.schemas.models import ErrorResponse, ExtractResponse, TaskStatus
from app.services.pdf_extractor import extract_page_images
from app.services.ocr_engine import OCREngine
from app.services.standard_processor import process_pdf_bytes as _process_standards
from app.services.regulation_processor import process_regulation_bytes as _process_regulations

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
    import logging
    import traceback

    worker_logger = logging.getLogger(__name__)

    try:
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
    except Exception:
        worker_logger.error(
            "Worker process failed:\n%s", traceback.format_exc()
        )
        raise


# ── Standards worker ─────────────────────────────────────────────
def _process_standards_worker(pdf_bytes: bytes) -> list[dict[str, str]]:
    """Process a single standard PDF in a subprocess worker.

    Uses adaptive parallel config for chapter-level parallelism.
    """
    return _process_standards(
        pdf_bytes,
        chunk_size=settings.standard_chunk_size,
        max_workers=parallel_config.max_workers,
    )


# ── Regulation worker ────────────────────────────────────────────
def _process_regulation_worker(args: tuple[bytes, str]) -> list[dict[str, str]]:
    """Process a single regulation file (PDF or HTML) in a subprocess worker.

    Args:
        args: Tuple of ``(file_bytes, file_format)`` where file_format is
              ``"pdf"`` or ``"html"``.
    """
    file_bytes, file_format = args
    return _process_regulations(
        file_bytes,
        file_format=file_format,
        chunk_size=settings.regulation_chunk_size,
        min_chunk_len=settings.regulation_min_chunk_len,
        max_chunk_len=settings.regulation_max_chunk_len,
        max_workers=parallel_config.max_workers,
    )


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
    background_tasks: BackgroundTasks = None,
):
    """Upload a single PDF or a ZIP archive of PDFs for text extraction.

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

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    if ext == ".zip":
        return _handle_zip_upload(content, file.filename, background_tasks)
    return await _handle_single_pdf(content, file.filename)


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


@router.post(
    "/extract-standards",
    response_model=ExtractResponse | TaskStatus,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def extract_standards(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a single standard PDF or ZIP of standards for text extraction.

    Uses direct PDF text extraction (no OCR) with chapter-aware parsing,
    clause-based chunking, and category inference.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".zip"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf and .zip are accepted.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    if ext == ".zip":
        return _handle_standards_zip_upload(
            content, file.filename, background_tasks
        )
    return await _handle_single_standard(content, file.filename)


# ═══════════════════════════════════════════════════════════════════
# Standards handlers
# ═══════════════════════════════════════════════════════════════════
async def _handle_single_standard(
    content: bytes,
    filename: str,
) -> ExtractResponse:
    """Process a single standard PDF asynchronously.

    Offloaded to the shared process pool so the async event loop
    is not blocked by CPU-intensive processing.
    """
    from app.services.text_processor import write_jsonl

    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(
        _get_shared_executor(),
        _process_standards_worker,
        content,
    )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the standard document.",
        )

    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"std_{stem}_{ts}.jsonl"
    out_path = settings.output_dir / out_name
    write_jsonl(records, out_path)

    return ExtractResponse(
        filename=out_name,
        record_count=len(records),
        preview=records[:3],
    )


def _handle_standards_zip_upload(
    content: bytes,
    filename: str,
    background_tasks: BackgroundTasks | None,
) -> TaskStatus:
    """Start background processing for a ZIP archive of standards."""
    zip_path = settings.upload_dir / f"{uuid.uuid4().hex}_{filename}"
    with open(zip_path, "wb") as f:
        f.write(content)

    task_id = uuid.uuid4().hex
    status = TaskStatus(task_id=task_id, status="processing")
    _task_store[task_id] = status

    if background_tasks:
        background_tasks.add_task(
            _process_standards_zip_background, task_id, zip_path
        )
    else:
        _process_standards_zip_background(task_id, zip_path)

    return status


def _process_standards_zip_background(task_id: str, zip_path: Path):
    """Background task: process all PDFs inside the ZIP in parallel."""
    from app.services.text_processor import write_jsonl

    status = _task_store[task_id]
    pdf_files: list[tuple[str, bytes]] = []

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

        all_records: list[dict[str, str]] = []
        executor = _get_shared_executor()
        futures = {
            executor.submit(_process_standards_worker, pdf_bytes): pdf_name
            for pdf_name, pdf_bytes in pdf_files
        }

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            pdf_name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                logger.error("Standards worker failed for %s: %s", pdf_name, exc)
            status.processed_files += 1
            status.current_file = Path(pdf_name).name

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"batch_std_{Path(zip_path).stem}_{ts}.jsonl"
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
        logger.exception("Background standards ZIP processing failed.")
    finally:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Regulation endpoints & handlers (PDF + HTML, mixed ZIP support)
# ═══════════════════════════════════════════════════════════════════
@router.post(
    "/extract-regulations",
    response_model=ExtractResponse | TaskStatus,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def extract_regulations(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a single regulation file or ZIP archive for text extraction.

    - **Single file**: supports ``.pdf``, ``.html``, ``.htm``.
    - **ZIP archive**: may contain a mix of PDF and HTML files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".html", ".htm", ".zip"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf, .html, .htm, and .zip are accepted.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    if ext == ".zip":
        return _handle_regulations_zip_upload(
            content, file.filename, background_tasks
        )
    # Single file: determine format from extension
    file_format = "html" if ext in (".html", ".htm") else "pdf"
    return await _handle_single_regulation(content, file.filename, file_format)


# ── Regulation handlers ──────────────────────────────────────────
async def _handle_single_regulation(
    content: bytes,
    filename: str,
    file_format: str,
) -> ExtractResponse:
    """Process a single regulation file asynchronously.

    Offloaded to the shared process pool so the async event loop
    is not blocked by CPU-intensive processing.
    """
    from app.services.text_processor import write_jsonl

    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(
        _get_shared_executor(),
        _process_regulation_worker,
        (content, file_format),
    )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the regulation document.",
        )

    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"reg_{stem}_{ts}.jsonl"
    out_path = settings.output_dir / out_name
    write_jsonl(records, out_path)

    return ExtractResponse(
        filename=out_name,
        record_count=len(records),
        preview=records[:3],
    )


def _handle_regulations_zip_upload(
    content: bytes,
    filename: str,
    background_tasks: BackgroundTasks | None,
) -> TaskStatus:
    """Start background processing for a ZIP archive of regulations (mixed PDF+HTML)."""
    zip_path = settings.upload_dir / f"{uuid.uuid4().hex}_{filename}"
    with open(zip_path, "wb") as f:
        f.write(content)

    task_id = uuid.uuid4().hex
    status = TaskStatus(task_id=task_id, status="processing")
    _task_store[task_id] = status

    if background_tasks:
        background_tasks.add_task(
            _process_regulations_zip_background, task_id, zip_path
        )
    else:
        _process_regulations_zip_background(task_id, zip_path)

    return status


def _process_regulations_zip_background(task_id: str, zip_path: Path):
    """Background task: process mixed PDF+HTML files in parallel."""
    from app.services.text_processor import write_jsonl

    status = _task_store[task_id]
    files_to_process: list[tuple[str, bytes, str]] = []  # (name, bytes, format)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                ext = Path(name).suffix.lower()
                if ext == ".pdf":
                    files_to_process.append((name, zf.read(name), "pdf"))
                elif ext in (".html", ".htm"):
                    files_to_process.append((name, zf.read(name), "html"))

        if not files_to_process:
            status.status = "failed"
            status.error = "ZIP contains no PDF or HTML files."
            return

        status.total_files = len(files_to_process)
        status.processed_files = 0

        all_records: list[dict[str, str]] = []
        executor = _get_shared_executor()
        futures = {
            executor.submit(_process_regulation_worker, (fbytes, fmt)): fname
            for fname, fbytes, fmt in files_to_process
        }

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            fname = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                logger.error("Regulation worker failed for %s: %s", fname, exc)
            status.processed_files += 1
            status.current_file = Path(fname).name

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"batch_reg_{Path(zip_path).stem}_{ts}.jsonl"
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
        logger.exception("Background regulations ZIP processing failed.")
    finally:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Single-PDF handler (offloaded to process pool)
# ═══════════════════════════════════════════════════════════════════
async def _handle_single_pdf(
    content: bytes,
    filename: str,
) -> ExtractResponse:
    """Process a single PDF asynchronously in the shared process pool.

    The CPU-intensive OCR runs in a subprocess worker so the async
    event loop is not blocked.
    """
    from app.services.text_processor import process_document, write_jsonl

    loop = asyncio.get_event_loop()

    # Offload OCR + text processing to a worker process
    try:
        records = await loop.run_in_executor(
            _get_shared_executor(),
            _process_pdf_worker,
            content,
        )
    except Exception as exc:
        logger.exception("Patent extraction failed for %s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {type(exc).__name__}: {exc}",
        )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="OCR produced no text. The PDF may be unreadable.",
        )

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
    background_tasks: BackgroundTasks | None,
) -> TaskStatus:
    """Start background processing for a ZIP archive."""
    zip_path = settings.upload_dir / f"{uuid.uuid4().hex}_{filename}"
    with open(zip_path, "wb") as f:
        f.write(content)

    task_id = uuid.uuid4().hex
    status = TaskStatus(task_id=task_id, status="processing")
    _task_store[task_id] = status

    if background_tasks:
        background_tasks.add_task(_process_zip_background, task_id, zip_path)
    else:
        _process_zip_background(task_id, zip_path)

    return status


def _process_zip_background(task_id: str, zip_path: Path):
    """Background task: process all PDFs inside the ZIP in parallel.

    Uses a ``ProcessPoolExecutor`` sized by the adaptive parallel config.
    Each PDF gets its own worker process.
    """
    from app.services.text_processor import write_jsonl

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
            executor.submit(_process_pdf_worker, pdf_bytes): pdf_name
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
