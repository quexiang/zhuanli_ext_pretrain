"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, HttpUrl


class ExtractResponse(BaseModel):
    """Returned when extraction completes synchronously (single PDF / single URL)."""

    filename: str
    record_count: int
    preview: list[dict[str, str]] = []
    source: str = ""        # "local_pdf" | "http_pdf" | "http_html"
    source_url: str = ""    # original URL (empty for local files)


class TaskStatus(BaseModel):
    """Returned when a background extraction task is in progress (ZIP / batch URLs)."""

    task_id: str
    status: str  # "processing" | "completed" | "failed"
    total_files: int = 0
    processed_files: int = 0
    current_file: str = ""
    record_count: int = 0
    output_filename: str = ""
    error: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""


# ═══════════════════════════════════════════════════════════════════════
# URL processing models
# ═══════════════════════════════════════════════════════════════════════

class URLSubmitRequest(BaseModel):
    """Single URL extraction request."""
    url: str
    doc_type: str = "patent"  # "patent" | "regulation"


class URLBatchRequest(BaseModel):
    """Batch URL extraction via JSON body — list of URLs."""
    urls: list[str]
    doc_type: str = "patent"  # "patent" | "regulation"


class URLResult(BaseModel):
    """Per-URL processing result."""
    url: str
    status: str = ""         # "success" | "failed" | "skipped"
    filename: str = ""
    record_count: int = 0
    error: str = ""
    source_type: str = ""    # "http_pdf" | "http_html"


class URLBatchTaskStatus(TaskStatus):
    """Extended task status for URL batch processing."""
    url_results: list[URLResult] = []
    source: str = "url_batch"
