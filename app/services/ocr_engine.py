"""OCR engine service.

Wraps PaddleOCR for Chinese text extraction from scanned patent PDF page images.
PaddleOCR model is loaded lazily on first use to avoid cold-start overhead
during app import.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR wrapper with lazy initialisation.

    PaddleOCR >= 3.7 uses the ``predict()`` method instead of the
    deprecated ``ocr()`` method, and returns a list of ``OCRResult``
    objects whose ``.rec_texts`` attribute holds the recognised text lines.
    """

    def __init__(self):
        self._ocr = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def create_detached(cls) -> "OCREngine":
        """Create an independent OCR engine instance for subprocess workers.

        Each ProcessPoolExecutor worker calls this to get its own
        PaddleOCR model — essential because PaddleOCR is **not**
        thread-safe and the singleton ``ocr_engine`` in ``extraction.py``
        cannot be shared across threads/processes.
        """
        logger.debug("Creating detached OCREngine (lazy init).")
        return cls()

    # Note for future GPU batching:
    #   PaddleOCR's ``predict()`` accepts a single numpy array, not a
    #   batch list.  To batch multiple pages through the GPU, call
    #   ``predict()`` sequentially in a tight loop — the GPU driver
    #   buffers operations internally and achieves near-batch throughput.
    #   When PaddlePaddle adds native batch support, add a
    #   ``process_pdf_batch(self, images, batch_size)`` method here.

    # ------------------------------------------------------------------
    # Lazy property
    # ------------------------------------------------------------------
    @property
    def ocr(self):
        if self._ocr is None:
            logger.info("Initialising PaddleOCR (first call may download models) ...")
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                lang="ch",                     # Simplified Chinese
                use_textline_orientation=True,  # detect & correct rotated text
            )
            logger.info("PaddleOCR initialised.")
        return self._ocr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_page(self, img_bytes: bytes) -> str:
        """Run OCR on a single page image.

        Args:
            img_bytes: PNG image bytes for one page.

        Returns:
            Extracted text as a single string with newline-separated lines,
            or an empty string if nothing was recognised.
        """
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img.convert("RGB"))

        results = self.ocr.predict(img_array)

        lines: list[str] = []
        if results:
            for page_result in results:
                rec_texts = page_result.get("rec_texts", None)
                if rec_texts and isinstance(rec_texts, list):
                    for text in rec_texts:
                        if text and text.strip():
                            lines.append(text.strip())

        return "\n".join(lines)

    def process_pdf(self, page_images: list[bytes]) -> str:
        """Run OCR on all pages of a PDF.

        Inserts a ``\\f`` (form-feed) character between pages so downstream
        processing can treat it as a hard page boundary.

        Args:
            page_images: Output from ``pdf_extractor.extract_page_images()``.

        Returns:
            Full document text with ``\\f`` between pages.
        """
        all_text: list[str] = []
        for i, img in enumerate(page_images):
            page_text = self.process_page(img).strip()
            if page_text:
                all_text.append(page_text)
                if i < len(page_images) - 1:
                    all_text.append("\f")  # page separator
        return "\n".join(all_text)
