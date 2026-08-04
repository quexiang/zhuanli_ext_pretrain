"""Upload, extraction, URL processing, and download endpoints with adaptive parallelism."""

from __future__ import annotations

import asyncio
import logging
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import parallel_config, settings
from app.schemas.models import (
    ErrorResponse,
    ExtractResponse,
    TaskStatus,
    URLBatchRequest,
    URLBatchTaskStatus,
    URLResult,
    URLSubmitRequest,
)
from app.services.pdf_extractor import extract_page_images
from app.services.ocr_engine import OCREngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["extraction"])


# In‑memory task store for background ZIP processing
_task_store: dict[str, TaskStatus] = {}

# Global shared process executor — created once from adaptive config.
# Used for:
#   1. Background ZIP processing (N workers)
#   2. Offloading single-PDF OCR from the event loop (1 worker at a time)
_shared_executor: ThreadPoolExecutor | None = None


def _get_shared_executor() -> ThreadPoolExecutor:
    """Return or create the shared thread pool.

    Threads are fine because PaddlePaddle GPU releases the GIL during
    GPU computation.  For CPU strategies this still works because
    max_workers is limited to 1 (serial mode).
    Avoids ProcessPoolExecutor DLL issues on Windows.
    """
    global _shared_executor
    if _shared_executor is None:
        _shared_executor = ThreadPoolExecutor(
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
def _process_pdf_worker(
    pdf_bytes: bytes,
    source: str = "local_pdf",
    source_url: str = "",
    doc_type: str = "patent",
) -> list[dict[str, str]]:
    """Process a single PDF in a thread-pool worker.

    Each worker creates its own ``OCREngine`` instance.
    EasyOCR is thread-safe; PaddleOCR gets a fresh instance per thread.
    """
    engine = OCREngine.create_detached()
    page_images = extract_page_images(pdf_bytes, dpi=settings.ocr_dpi)
    if not page_images:
        return []

    full_text = engine.process_pdf(page_images, batch_size=parallel_config.batch_size)
    if not full_text.strip():
        return []

    if doc_type == "regulation":
        from app.services.regulation_processor import process_regulation_document
        return process_regulation_document(
            full_text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=settings.regulation_category,
            source=source,
            source_url=source_url,
        )
    else:
        from app.services.text_processor import process_document
        return process_document(
            full_text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=settings.category,
            source=source,
            source_url=source_url,
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
    doc_type: str = Form("patent"),
    background_tasks: BackgroundTasks = None,
):
    """Upload a single PDF, HTML, or ZIP archive for text extraction.

    - **doc_type=patent**: patent cleaning pipeline (default).
    - **doc_type=regulation**: regulation cleaning pipeline.

    - **Single PDF**: OCR + text processing in the shared process pool.
    - **Single HTML**: body text extraction + text processing.
    - **ZIP archive**: all contained PDFs and HTMLs processed in parallel.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".zip", ".html", ".htm"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf, .html, .htm and .zip are accepted.",
        )

    if doc_type not in ("patent", "regulation"):
        raise HTTPException(
            status_code=400,
            detail="Invalid doc_type. Use 'patent' or 'regulation'.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    if ext == ".zip":
        return _handle_zip_upload(content, file.filename, doc_type, background_tasks)
    if ext in (".html", ".htm"):
        return _handle_single_html_file(content, file.filename, doc_type)
    return await _handle_single_pdf(content, file.filename, doc_type)


@router.get(
    "/status/{task_id}",
    response_model=TaskStatus | URLBatchTaskStatus,
)
async def get_status(task_id: str):
    """Poll processing progress for a background extraction task.

    Works for both ZIP batch and URL batch tasks.
    """
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
    "/extract-url",
    response_model=ExtractResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def extract_from_url(body: URLSubmitRequest):
    """Submit a single HTTP/HTTPS URL pointing to a PDF or HTML page.

    The system downloads the resource, auto-detects its type, then:
    - **PDF**: runs the standard OCR + text-processing pipeline.
    - **HTML**: extracts body text via BeautifulSoup, then runs
      the text-processing pipeline.
    """
    from app.services.url_fetcher import fetch_and_detect

    # 1) Download & detect
    result = fetch_and_detect(body.url)
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process URL: {result.error}",
        )

    if result.source_type == "http_pdf" and result.content_bytes:
        return await _handle_single_pdf_from_bytes(
            result.content_bytes,
            result.filename,
            source="http_pdf",
            source_url=body.url,
            doc_type=body.doc_type,
        )

    if result.source_type == "http_html" and result.text:
        return _handle_html_text(
            result.text,
            result.filename,
            source_url=body.url,
            doc_type=body.doc_type,
        )

    raise HTTPException(
        status_code=400,
        detail=f"URL returned unsupported content: {result.content_type}",
    )


@router.post(
    "/extract-url-batch",
    response_model=URLBatchTaskStatus,
    responses={400: {"model": ErrorResponse}},
)
async def extract_from_url_batch(
    body: URLBatchRequest | None = None,
    file: UploadFile | None = File(None),
):
    """Submit a batch of URLs for parallel processing.

    Two input modes:
    - **JSON body**: ``{"urls": ["https://...", ...]}``
    - **File upload**: a ``.txt``, ``.csv``, or ``.json`` file containing URLs.

    Processing runs in the background.  Poll ``GET /api/status/{task_id}``
    for progress.  Each URL is processed independently — one failure does
    not abort the batch.
    """
    from app.services.url_fetcher import deduplicate_urls, parse_url_list, validate_url

    urls: list[str] = []

    # ── Parse input ──────────────────────────────────────────
    if file and file.filename:
        ext = Path(file.filename).suffix.lower().lstrip(".")
        if ext not in ("txt", "csv", "json"):
            raise HTTPException(
                status_code=400,
                detail="URL list file must be .txt, .csv, or .json.",
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")
        try:
            urls = parse_url_list(content, ext)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse URL list: {exc}",
            )
    elif body and body.urls:
        urls = body.urls
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a JSON body with 'urls' or upload a URL list file.",
        )

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found in input.")

    # ── Validate & deduplicate ──────────────────────────────
    invalid: list[dict] = []
    valid: list[str] = []
    for url in urls:
        ok, err = validate_url(url)
        if ok:
            valid.append(url)
        else:
            invalid.append({"url": url, "error": err})

    valid = deduplicate_urls(valid)

    if not valid:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(urls)} URLs are invalid. "
                   f"First error: {invalid[0]['error'] if invalid else 'unknown'}",
        )

    # ── Start background task ────────────────────────────────
    doc_type = body.doc_type if body else "patent"
    task_id = uuid.uuid4().hex
    status = URLBatchTaskStatus(
        task_id=task_id,
        status="processing",
        total_files=len(valid),
        processed_files=0,
        source="url_batch",
    )
    # Pre-populate with invalid/skipped URLs
    for inv in invalid:
        status.url_results.append(URLResult(
            url=inv["url"],
            status="skipped",
            error=inv["error"],
        ))
    _task_store[task_id] = status

    # Run as a background thread (non-blocking)
    import threading
    threading.Thread(
        target=_process_url_batch_background,
        args=(task_id, valid, bool(invalid), doc_type),
        daemon=True,
    ).start()

    return status


# ═══════════════════════════════════════════════════════════════════
# URL batch background processing
# ═══════════════════════════════════════════════════════════════════
def _process_url_batch_background(
    task_id: str,
    urls: list[str],
    had_skipped: bool = False,
    doc_type: str = "patent",
):
    """Background task: download every URL in parallel, then process results.

    Phase 1 — parallel HTTP download (I/O bound).
    Phase 2 — sequential OCR + text processing (already parallel internally).
    """
    from concurrent.futures import as_completed
    from threading import Lock

    from app.services.text_processor import write_jsonl
    from app.services.url_fetcher import DownloadResult, fetch_and_detect

    status = _task_store[task_id]

    # Ensure this is a URLBatchTaskStatus (idempotent if already is)
    if not isinstance(status, URLBatchTaskStatus):
        status = URLBatchTaskStatus(
            task_id=status.task_id,
            status="processing",
            total_files=len(urls),
            source="url_batch",
        )
        _task_store[task_id] = status

    all_records: list[dict[str, str]] = []
    status_lock = Lock()

    try:
        # ── Phase 1: parallel download ────────────────────────────
        # Use up to 8 download threads (I/O bound, more workers are fine).
        dl_workers = min(len(urls), 8)
        download_results: list[tuple[str, DownloadResult]] = []

        with ThreadPoolExecutor(max_workers=dl_workers) as dl_executor:
            future_map = {dl_executor.submit(fetch_and_detect, url): url for url in urls}

            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = DownloadResult(url=url, success=False, error=str(exc))
                download_results.append((url, result))
                # Report download progress
                with status_lock:
                    status.processed_files = len(download_results)
                    status.current_file = url

        # ── Phase 2: sequential processing ────────────────────────
        for idx, (url, result) in enumerate(download_results):
            status.processed_files = idx

            if not result.success:
                status.url_results.append(URLResult(
                    url=url,
                    status="failed",
                    error=result.error,
                ))
                continue

            if result.source_type == "http_pdf" and result.content_bytes:
                records = _process_single_url_pdf_sync(
                    result.content_bytes,
                    result.filename,
                    result.url,
                    doc_type,
                )
                all_records.extend(records)
                status.url_results.append(URLResult(
                    url=url,
                    status="success",
                    filename=result.filename,
                    record_count=len(records),
                    source_type="http_pdf",
                ))

            elif result.source_type == "http_html" and result.text:
                records = _process_single_url_html_sync(
                    result.text,
                    result.filename,
                    result.url,
                    doc_type,
                )
                all_records.extend(records)
                status.url_results.append(URLResult(
                    url=url,
                    status="success",
                    filename=result.filename,
                    record_count=len(records),
                    source_type="http_html",
                ))
            else:
                status.url_results.append(URLResult(
                    url=url,
                    status="failed",
                    error=f"Unsupported content: {result.content_type}",
                ))

        # ── Write merged output ──────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = uuid.uuid4().hex[:8]
        out_name = f"url_batch_{ts}_{short_hash}.jsonl"
        out_path = settings.output_dir / out_name
        if all_records:
            write_jsonl(all_records, out_path)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")

        status.status = "completed"
        status.record_count = len(all_records)
        status.output_filename = out_name
        status.processed_files = len(urls)

    except Exception as exc:
        status.status = "failed"
        status.error = str(exc)
        logger.exception("URL batch processing failed catastrophically.")


def _process_single_url_pdf_sync(
    pdf_bytes: bytes,
    filename: str,
    source_url: str,
    doc_type: str = "patent",
) -> list[dict[str, str]]:
    """Run the full PDF pipeline synchronously for a downloaded URL PDF."""
    from app.services.text_processor import process_document

    is_regulation = doc_type == "regulation"
    category = settings.regulation_category if is_regulation else settings.category

    engine = OCREngine.create_detached()
    page_images = extract_page_images(pdf_bytes, dpi=settings.ocr_dpi)
    if not page_images:
        return []

    full_text = engine.process_pdf(page_images, batch_size=parallel_config.batch_size)
    if not full_text.strip():
        return []

    if is_regulation:
        from app.services.regulation_processor import process_regulation_document
        return process_regulation_document(
            full_text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=category,
        )
    return process_document(
        full_text,
        min_len=settings.segment_min_len,
        max_len=settings.segment_max_len,
        category=category,
    )


def _process_single_url_html_sync(
    html_text: str,
    filename: str,
    source_url: str,
    doc_type: str = "patent",
) -> list[dict[str, str]]:
    """Run the text-processing pipeline for extracted HTML body text."""
    from app.services.text_processor import process_document

    is_regulation = doc_type == "regulation"
    category = settings.regulation_category if is_regulation else settings.category

    if is_regulation:
        from app.services.regulation_processor import process_regulation_document
        return process_regulation_document(
            html_text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=category,
        )
    return process_document(
        html_text,
        min_len=settings.segment_min_len,
        max_len=settings.segment_max_len,
        category=category,
    )


# ═══════════════════════════════════════════════════════════════════
# Single-PDF handler (offloaded to process pool)
# ═══════════════════════════════════════════════════════════════════
async def _handle_single_pdf(
    content: bytes,
    filename: str,
    doc_type: str = "patent",
) -> ExtractResponse:
    """Process a single PDF asynchronously in the shared process pool.

    The CPU-intensive OCR runs in a subprocess worker so the async
    event loop is not blocked.
    """
    from app.services.text_processor import process_document, write_jsonl

    loop = asyncio.get_event_loop()

    # Offload OCR + text processing to a worker process
    records = await loop.run_in_executor(
        _get_shared_executor(),
        _process_pdf_worker,
        content,
        "local_pdf",
        "",
        doc_type,
    )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="OCR produced no text. The PDF may be unreadable.",
        )

    # Write output file (fast I/O, stay in async context)
    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "reg_" if doc_type == "regulation" else ""
    out_name = f"{prefix}{stem}_{ts}.jsonl"
    out_path = settings.output_dir / out_name
    write_jsonl(records, out_path)

    preview = records[:3]

    return ExtractResponse(
        filename=out_name,
        record_count=len(records),
        preview=preview,
        source="local_pdf",
        source_url="",
    )


async def _handle_single_pdf_from_bytes(
    content: bytes,
    filename: str,
    source: str = "http_pdf",
    source_url: str = "",
    doc_type: str = "patent",
) -> ExtractResponse:
    """Process PDF bytes from a URL download, identical flow to file upload."""
    from app.services.text_processor import process_document, write_jsonl

    loop = asyncio.get_event_loop()

    records = await loop.run_in_executor(
        _get_shared_executor(),
        _process_pdf_worker,
        content,
        source,
        source_url,
        doc_type,
    )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="OCR produced no text. The downloaded PDF may be unreadable.",
        )

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
        source=source,
        source_url=source_url,
    )


def _handle_html_text(
    text: str,
    filename: str,
    source_url: str = "",
    doc_type: str = "patent",
) -> ExtractResponse:
    """Process extracted HTML body text through the text-processing pipeline."""
    from app.services.text_processor import process_document, write_jsonl

    is_regulation = doc_type == "regulation"
    category = settings.regulation_category if is_regulation else settings.category

    if is_regulation:
        from app.services.regulation_processor import process_regulation_document
        records = process_regulation_document(
            text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=category,
        )
    else:
        records = process_document(
            text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=category,
        )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="HTML text extraction produced no usable content.",
        )

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
        source="http_html",
        source_url=source_url,
    )


def _handle_single_html_file(
    content: bytes,
    filename: str,
    doc_type: str = "patent",
) -> ExtractResponse:
    """Process a locally uploaded HTML file — extract body text and run pipeline."""
    from app.services.text_processor import process_document, write_jsonl
    from app.services.url_fetcher import extract_html_text

    try:
        text = extract_html_text(content)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from HTML file: {exc}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="HTML file contains no extractable text.",
        )

    if doc_type == "regulation":
        from app.services.regulation_processor import process_regulation_document
        records = process_regulation_document(
            text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=settings.regulation_category,
            source="local_html",
            source_url=filename,
        )
    else:
        records = process_document(
            text,
            min_len=settings.segment_min_len,
            max_len=settings.segment_max_len,
            category=settings.category,
            source="local_html",
            source_url=filename,
        )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="HTML text extraction produced no usable content.",
        )

    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "reg_" if doc_type == "regulation" else ""
    out_name = f"{prefix}{stem}_{ts}.jsonl"
    out_path = settings.output_dir / out_name
    write_jsonl(records, out_path)

    preview = records[:3]

    return ExtractResponse(
        filename=out_name,
        record_count=len(records),
        preview=preview,
        source="local_html",
        source_url=filename,
    )


# ═══════════════════════════════════════════════════════════════════
# ZIP upload & background processing
# ═══════════════════════════════════════════════════════════════════
def _handle_zip_upload(
    content: bytes,
    filename: str,
    doc_type: str = "patent",
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
        background_tasks.add_task(_process_zip_background, task_id, zip_path, doc_type)
    else:
        _process_zip_background(task_id, zip_path, doc_type)

    return status


class _ZipWithNames:
    """Context manager wrapping a ZipFile with a GBK-aware name map.

    ``disp_name`` (display name) is the correctly-decoded human-readable
    filename.  ``raw_name`` is the internal archive name usable with
    ``zf.read()``.

    Usage::

        with _ZipWithNames.open(zip_path) as zwn:
            for disp, raw in zwn.name_map.items():
                pdf_bytes = zwn.zf.read(raw)
                print(disp)   # correctly-decoded Chinese filename
    """

    def __init__(self, zf: zipfile.ZipFile, name_map: dict[str, str]):
        self.zf = zf
        self.name_map = name_map

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.zf.close()

    @staticmethod
    def open(zip_path: Path) -> "_ZipWithNames":
        import sys

        if sys.version_info >= (3, 11):
            try:
                zf = zipfile.ZipFile(zip_path, "r", metadata_encoding="gbk")
            except (LookupError, ValueError):
                zf = zipfile.ZipFile(zip_path, "r")
            names = zf.namelist()
            return _ZipWithNames(zf, {n: n for n in names})

        # Python < 3.11 — decode GBK filenames into name_map
        zf = zipfile.ZipFile(zip_path, "r")
        name_map: dict[str, str] = {}
        for info in zf.infolist():
            raw_name = info.filename
            if info.flag_bits & 0x800:  # bit 11 = UTF-8
                name_map[raw_name] = raw_name
                continue
            try:
                raw = raw_name.encode("cp437")
                decoded = raw.decode("gbk", errors="replace")
                name_map[decoded] = raw_name
            except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
                # cp437 round-trip failed — keep original name as-is
                name_map[raw_name] = raw_name
        return _ZipWithNames(zf, name_map)


def _process_zip_background(task_id: str, zip_path: Path, doc_type: str = "patent"):
    """Background task: process all PDFs and HTMLs inside the ZIP in parallel.

    - **PDFs** are OCR'd via the shared process pool.
    - **HTMLs** are text-extracted inline (no OCR needed).
    - Results from both types are merged into a single JSONL.
    """
    from app.services.text_processor import write_jsonl
    from app.services.url_fetcher import extract_html_text

    is_regulation = doc_type == "regulation"
    category = settings.regulation_category if is_regulation else settings.category

    status = _task_store[task_id]
    pdf_files: list[tuple[str, bytes]] = []
    html_files: list[tuple[str, str]] = []

    # 1) Extract all files from ZIP into memory
    try:
        with _ZipWithNames.open(zip_path) as zwn:
            for disp_name, raw_name in zwn.name_map.items():
                lower = disp_name.lower()
                # Skip directories
                if disp_name.endswith("/"):
                    continue
                if lower.endswith(".pdf"):
                    pdf_files.append((disp_name, zwn.zf.read(raw_name)))
                elif lower.endswith((".html", ".htm")):
                    try:
                        raw = zwn.zf.read(raw_name)  # bytes — pass directly
                        text = extract_html_text(raw)
                        if text.strip():
                            html_files.append((disp_name, text))
                        else:
                            logger.warning("HTML file '%s' has no extractable text, skipped.", disp_name)
                    except Exception as exc:
                        logger.error("Failed to extract text from HTML '%s': %s", disp_name, exc)

        if not pdf_files and not html_files:
            status.status = "failed"
            status.error = "ZIP contains no PDF or HTML files."
            return

        total = len(pdf_files) + len(html_files)
        status.total_files = total
        status.processed_files = 0

        all_records: list[dict[str, str]] = []

        # 2) Process HTML files first (fast, inline)
        for html_name, html_text in html_files:
            try:
                if is_regulation:
                    from app.services.regulation_processor import process_regulation_document
                    records = process_regulation_document(
                        html_text,
                        min_len=settings.segment_min_len,
                        max_len=settings.segment_max_len,
                        category=category,
                        source="local_html",
                        source_url=html_name,
                    )
                else:
                    from app.services.text_processor import process_document
                    records = process_document(
                        html_text,
                        min_len=settings.segment_min_len,
                        max_len=settings.segment_max_len,
                        category=category,
                        source="local_html",
                        source_url=html_name,
                    )
                all_records.extend(records)
                logger.info(
                    "HTML '%s' → %d records (%s)", html_name, len(records),
                    "regulation" if is_regulation else "local_html",
                )
            except Exception as exc:
                logger.error("HTML processing failed for '%s': %s", html_name, exc)
            status.processed_files += 1
            status.current_file = Path(html_name).name

        # 3) Submit all PDFs to the process pool
        if pdf_files:
            executor = _get_shared_executor()
            futures = {
                executor.submit(
                    _process_pdf_worker, pdf_bytes, "zip_pdf", pdf_name, doc_type,
                ): pdf_name
                for pdf_name, pdf_bytes in pdf_files
            }

            # 4) Collect PDF results as they complete (unordered)
            from concurrent.futures import as_completed

            for future in as_completed(futures):
                pdf_name = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as exc:
                    logger.error("Worker failed for %s: %s", pdf_name, exc)
                status.processed_files += 1
                status.current_file = Path(pdf_name).name

        # 5) Write merged output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "reg_" if is_regulation else "batch_"
        out_name = f"{prefix}{Path(zip_path).stem}_{ts}.jsonl"
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
