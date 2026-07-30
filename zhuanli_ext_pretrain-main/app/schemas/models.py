"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel


class ExtractRequest(BaseModel):
    """Document type for extraction."""

    doc_type: str = "patent"  # "patent" | "standard"


class ExtractResponse(BaseModel):
    """Returned when extraction completes synchronously (single PDF)."""

    filename: str
    record_count: int
    preview: list[dict[str, str]] = []


class TaskStatus(BaseModel):
    """Returned when a background extraction task is in progress (ZIP)."""

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
