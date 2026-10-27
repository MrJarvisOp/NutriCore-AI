"""
pdf_processor.py
-----------------
Loads a PDF and returns page-level content (text or PIL images for OCR).
Uses PyMuPDF (fitz) as the primary driver.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

from app.utils.logger import get_logger

log = get_logger(__name__)


class PDFProcessorError(Exception):
    pass


class PDFProcessor:
    """Extract text and/or images from a PDF file."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise PDFProcessorError(f"PDF not found: {self.pdf_path}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def extract_pages(self) -> List[Tuple[int, str, bytes | None]]:
        """
        Returns a list of (page_no, text, image_bytes) tuples.

        * text       - extracted text layer (may be empty for scanned docs)
        * image_bytes - PNG bytes of the page for OCR fallback (only when text is thin)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise PDFProcessorError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            )

        pages: List[Tuple[int, str, bytes | None]] = []

        with fitz.open(str(self.pdf_path)) as doc:
            log.info(f"PDF opened - {len(doc)} page(s): {self.pdf_path.name}")
            for page_no, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()

                # Render page as image only if text layer is sparse
                img_bytes: bytes | None = None
                if len(text) < 50:
                    mat = fitz.Matrix(2.0, 2.0)           # 2× zoom for better OCR
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("png")
                    log.debug(f"Page {page_no}: sparse text - rendering for OCR")
                else:
                    log.debug(f"Page {page_no}: {len(text)} chars of text found")

                pages.append((page_no, text, img_bytes))

        return pages

    def has_text_layer(self) -> bool:
        """Quick check - does ANY page have a meaningful text layer?"""
        pages = self.extract_pages()
        return any(len(text) > 50 for _, text, _ in pages)

    def get_full_text(self) -> str:
        """Concatenate text from all pages (empty if scanned)."""
        pages = self.extract_pages()
        return "\n\n".join(text for _, text, _ in pages if text)
