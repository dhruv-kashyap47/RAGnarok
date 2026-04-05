"""
pdf_service.py — PDF text extraction using PyMuPDF4LLM.
"""

from __future__ import annotations

from fastapi import HTTPException
from app.core.logger import logger


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from *file_path* as Markdown.

    Raises HTTPException(400) for encrypted/empty PDFs and
    HTTPException(502) for unexpected library errors.
    """
    try:
        import pymupdf4llm  # local import so tests can mock it easily
        md_text = pymupdf4llm.to_markdown(file_path)
    except Exception as exc:
        # Catch both pymupdf errors and unexpected issues; never leak file paths
        msg = str(exc)
        logger.error("PDF extraction failed: %s", msg)
        if any(kw in msg.lower() for kw in ("password", "encrypt", "cannot open")):
            raise HTTPException(
                status_code=400,
                detail="PDF is encrypted or cannot be opened. Please provide an unencrypted PDF.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="Failed to extract text from PDF. The file may be corrupted.",
        ) from exc

    text = str(md_text).strip()
    return text
