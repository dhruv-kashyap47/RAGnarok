import pymupdf4llm


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF4LLM."""
    md_text = pymupdf4llm.to_markdown(file_path)
    return str(md_text)
