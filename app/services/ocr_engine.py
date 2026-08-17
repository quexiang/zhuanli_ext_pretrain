"""OCR engine service.

Wraps PaddleOCR for Chinese text extraction from scanned patent PDF page images.
PaddleOCR model is loaded lazily on first use to avoid cold-start overhead
during app import.
"""

from __future__ import annotations

import io
import logging
import os

import numpy as np
from PIL import Image

from app.config import resource_info

# ── Work around Windows PaddlePaddle PIR + oneDNN compiler bugs ──
#   FLAGS_enable_pir_api=0  →  disable new PIR executor; fall back to old
#   FLAGS_use_onednn_op=0   →  disable oneDNN in both old and new executors
#   ref: https://github.com/PaddlePaddle/Paddle/issues/67288
for _flag in ("FLAGS_enable_pir_api", "FLAGS_use_onednn_op", "FLAGS_use_onednn_graph"):
    os.environ.setdefault(_flag, "0")

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

            # Ensure PIR and oneDNN are disabled before PaddleOCR imports
            # Paddle (belt-and-suspenders — module-level env vars may not
            # propagate to ProcessPoolExecutor workers on Windows spawn).
            for _flag in ("FLAGS_enable_pir_api", "FLAGS_use_onednn_op", "FLAGS_use_onednn_graph"):
                os.environ[_flag] = "0"

            from paddleocr import PaddleOCR

            # Pick the inference device from runtime detection.  paddle_backend
            # is "gpu" whenever PaddlePaddle is compiled with CUDA and a GPU is
            # found.  Without an explicit device, PaddleOCR silently falls back
            # to CPU even when a GPU is available.
            device = "gpu" if resource_info.paddle_backend == "gpu" else "cpu"

            self._ocr = PaddleOCR(
                lang="ch",                     # Simplified Chinese
                use_textline_orientation=True,  # detect & correct rotated text
                enable_mkldnn=False,           # work around PaddlePaddle 3.3.0+ bug
                                               #   ref: https://github.com/PaddlePaddle/Paddle/issues/77340
                device=device,                  # explicit GPU (or CPU) device
            )
            logger.info("PaddleOCR initialised (device=%s).", device)
        return self._ocr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # Maximum image dimension for PaddleOCR.  Images larger than this
    # are downscaled to avoid detection timeouts and OOM errors.
    # PaddleOCR's PP-OCRv6 detection model works best with inputs
    # whose longest side is ≤ 2400 px.
    _MAX_IMAGE_DIM = 2400

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Resize image if needed, then convert to numpy array.

        Large images (e.g. A4 @ 300 DPI = 2480×3509) cause PaddleOCR
        detection to time out or produce empty results.  Resizing keeps
        the longest side ≤ ``_MAX_IMAGE_DIM`` while maintaining aspect ratio.
        """
        w, h = img.size
        longest = max(w, h)
        if longest > self._MAX_IMAGE_DIM:
            scale = self._MAX_IMAGE_DIM / longest
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info(
                "Resizing image from %dx%d → %dx%d for OCR (max=%d)",
                w, h, new_w, new_h, self._MAX_IMAGE_DIM,
            )
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(img.convert("RGB"))

    def process_page(self, img_bytes: bytes) -> str:
        """Run OCR on a single page image.

        Args:
            img_bytes: PNG image bytes for one page.

        Returns:
            Extracted text as a single string with newline-separated lines,
            or an empty string if nothing was recognised.
        """
        img = Image.open(io.BytesIO(img_bytes))
        img_array = self._preprocess_image(img)

        results = self.ocr.predict(img_array)

        lines: list[str] = []
        if results:
            for page_result in results:
                # PaddleOCR >= 3.7 returns OCRResult (dict subclass).
                # Use .get() which works for both dict and dict-like objects.
                rec_texts = page_result.get("rec_texts") if isinstance(page_result, dict) else getattr(page_result, "rec_texts", None)

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
