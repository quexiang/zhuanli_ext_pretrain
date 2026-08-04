"""OCR engine service — PaddleOCR backend."""

from __future__ import annotations

# ⚠️ torch MUST be imported before anything that triggers PaddlePaddle import.
# PaddlePaddle's DLL loading modifies the system DLL search path on Windows,
# which causes torch's shm.dll to fail with "找不到指定的程序" (WinError 127).
import torch  # noqa: F401  — side-effect import: loads torch DLLs early

import io
import logging
import os
import warnings

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Prevent PaddlePaddle oneDNN / PIR crashes on CPU installs
os.environ.setdefault("FLAGS_use_pir", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("GLOG_minloglevel", "2")


class OCREngine:
    """OCR wrapper using PaddleOCR.

    Supports PaddleOCR 2.x and 3.x — parameter compatibility is handled
    automatically.
    """

    def __init__(self):
        self._ocr = None

    @classmethod
    def create_detached(cls) -> "OCREngine":
        """Create a fresh OCR engine for thread-pool workers."""
        return cls()

    @property
    def ocr(self):
        if self._ocr is not None:
            return self._ocr

        logger.info("Initialising PaddleOCR …")
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x / 2.x parameter compatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # PaddleOCR 3.x — show_log removed, use_textline_orientation valid
                self._ocr = PaddleOCR(
                    lang="ch",
                    use_textline_orientation=True,
                )
            except (TypeError, ValueError):
                try:
                    # PaddleOCR 2.x fallback
                    self._ocr = PaddleOCR(
                        lang="ch",
                        use_doc_orientation_classify=True,
                        show_log=False,
                    )
                except (TypeError, ValueError):
                    self._ocr = PaddleOCR(lang="ch")

        logger.info("PaddleOCR initialised.")
        return self._ocr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_page(self, img_bytes: bytes) -> str:
        """OCR a single page image (convenience wrapper)."""
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img.convert("RGB"))
        result = self.ocr.ocr(img_array)
        return self._extract_text(result)

    @staticmethod
    def _extract_text(result) -> str:
        """Extract joined text from PaddleOCR ``ocr()`` output.

        PaddleOCR 3.x ``ocr()`` (det+rec) returns:
        ``[[[box, (text, confidence)], ...]]`` — outer list per image,
        inner list of text regions, each region is ``[box, (text, conf)]``.
        """
        if not result or not isinstance(result, list):
            return ""
        lines: list[str] = []
        # result: [[[box, (text, conf)], ...], ...]  — one entry per image
        for image_regions in result:
            if image_regions is None:
                continue
            if not isinstance(image_regions, list):
                continue
            # image_regions: [[box, (text, conf)], ...]
            for region in image_regions:
                if region is None:
                    continue
                if not isinstance(region, (list, tuple)):
                    continue
                if len(region) < 2:
                    continue
                rec_result = region[1]
                if isinstance(rec_result, (list, tuple)) and len(rec_result) >= 1:
                    text = rec_result[0]
                    if text and isinstance(text, str):
                        lines.append(text.strip())
        return "\n".join(lines)

    def process_pdf(
        self, page_images: list[bytes], batch_size: int = 1,
    ) -> str:
        """OCR every page in *page_images*, processing one page at a time.

        Args:
            page_images: PNG/JPEG bytes for each PDF page.
            batch_size: Ignored — PaddleOCR 3.x ``ocr()`` only accepts
                single images with ``det=True``.  Kept for API compatibility.

        Returns:
            Full text with ``\\f`` separators between pages.
        """
        _ = self.ocr  # eager init once
        page_texts: list[str] = []

        for i, img_bytes in enumerate(page_images):
            try:
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img.convert("RGB"))
                result = self.ocr.ocr(img_array)
                text = self._extract_text(result).strip()
                page_texts.append(text)
            except Exception:
                logger.exception("OCR failed on page %d", i + 1)
                page_texts.append("")

        # Join with form feeds between pages
        return "\n\f\n".join(t for t in page_texts if t)
