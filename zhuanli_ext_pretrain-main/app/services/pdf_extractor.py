"""PDF image extraction service.

Each patent PDF is a scanned-image PDF (no embedded text).
Extract each page as a high-resolution PNG image for OCR processing.
"""

import io
import fitz


def extract_page_images(pdf_bytes: bytes, dpi: int = 300) -> list[bytes]:
    """Render each PDF page as a PNG image.

    Args:
        pdf_bytes: Raw PDF file bytes.
        dpi: Output resolution (dots per inch).  Matches original scan DPI.

    Returns:
        List of PNG image bytes, one per page.
    """
    images: list[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72  # PyMuPDF's base coordinate system is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)

    doc.close()
    return images


def extract_page_images_from_path(pdf_path: str, dpi: int = 300) -> list[bytes]:
    """Open a PDF file from disk and render each page as a PNG image.

    Args:
        pdf_path: Path to the PDF file on disk.
        dpi: Output resolution.

    Returns:
        List of PNG image bytes, one per page.
    """
    with open(pdf_path, "rb") as f:
        return extract_page_images(f.read(), dpi)


def get_pdf_info(pdf_bytes: bytes) -> dict:
    """Extract basic PDF metadata without rendering pages.

    Returns:
        Dict with page_count and file_size info.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    info = {
        "page_count": doc.page_count,
        "has_text": False,
    }
    # Quick check: try to extract text from first page
    if doc.page_count > 0:
        text = doc[0].get_text("text").strip()
        if len(text) > 20:
            info["has_text"] = True
    doc.close()
    return info
