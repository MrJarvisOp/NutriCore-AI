"""
ocr_service.py
--------------
Smart OCR layer:
  - If a text layer exists → use PyMuPDF output directly.
  - Otherwise             → run Tesseract on page images.
"""
from __future__ import annotations

import io
from typing import List, Tuple

from app.services.pdf_processor import PDFProcessor
from app.utils.logger import get_logger

log = get_logger(__name__)


class OCRService:
    """Unified text extraction from PDF (text-layer or image-based)."""

    def __init__(self, pdf_path: str):
        self.processor = PDFProcessor(pdf_path)

    # ─── Public ──────────────────────────────────────────────────────────────

    def extract_text(self) -> str:
        """Return full document text, using OCR where needed."""
        pages = self.processor.extract_pages()
        parts: List[str] = []

        for page_no, text, img_bytes in pages:
            if len(text) >= 50:
                # Text layer is good
                log.info(f"Page {page_no}: using embedded text layer")
                parts.append(text)
            elif img_bytes is not None:
                # Fallback to Tesseract
                log.info(f"Page {page_no}: no text layer - running Tesseract OCR")
                ocr_text = self._tesseract_ocr(img_bytes, page_no)
                parts.append(ocr_text)
            else:
                log.warning(f"Page {page_no}: no text and no image - skipping")

        return "\n\n".join(filter(None, parts))

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tesseract_ocr(image_bytes: bytes, page_no: int = 0) -> str:
        """Run Tesseract on raw PNG bytes and return extracted text."""
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            log.error(
                "Tesseract OCR dependencies missing. "
                "Install: pip install pillow pytesseract  (and tesseract binary)"
            )
            return ""

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Use a whitelist-free config; PSM 6 = assume uniform block of text
            text = pytesseract.image_to_string(
                image,
                config="--psm 6 --oem 3",
                lang="eng",
            )
            log.debug(f"Page {page_no} OCR result ({len(text)} chars)")
            return text.strip()
        except Exception as exc:
            log.error(f"Tesseract failed on page {page_no}: {exc}")
            return ""
