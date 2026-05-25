from dataclasses import dataclass
from io import BytesIO
from typing import List
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


@dataclass
class PdfPage:
    page_number: int
    text: str


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _with_pypdf(file_bytes: bytes) -> List[PdfPage]:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append(PdfPage(i, text))
    return pages


def _with_pymupdf(file_bytes: bytes) -> List[PdfPage]:
    if fitz is None:
        return []
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append(PdfPage(i, text))
    return pages


def extract_pdf_pages(file_bytes: bytes) -> List[PdfPage]:
    pages = _with_pypdf(file_bytes)
    if not pages:
        pages = _with_pymupdf(file_bytes)
    if not pages:
        raise ValueError(
            "No readable text was found. This is probably a scanned/image PDF. "
            "Try a text based PDF or add OCR as a future improvement."
        )
    return pages
