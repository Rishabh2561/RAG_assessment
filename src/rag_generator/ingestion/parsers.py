"""Concrete parsers for PDF, plain text / Markdown and DOCX.

Parsers work on bytes (not paths) so uploads never need to touch disk.
"""

from __future__ import annotations

import io
import re

from rag_generator.errors import CorruptDocumentError, EmptyDocumentError, OCRRequiredError
from rag_generator.models import ParsedDocument, Section

# A page with fewer extractable characters than this, but with images, is treated as scanned.
_SCANNED_PAGE_MAX_CHARS = 20
_WHITESPACE_RUNS = re.compile(r"[ \t]+")
_BLANK_LINE_RUNS = re.compile(r"\n{3,}")


def normalise_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and excess blank lines; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINE_RUNS.sub("\n\n", text).strip()


class TextParser:
    """UTF-8 text and Markdown, with a latin-1 fallback for legacy encodings."""

    extensions = ("txt", "md", "markdown")

    def parse(self, data: bytes, source: str) -> ParsedDocument:
        warnings: list[str] = []
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
            warnings.append("not valid UTF-8; decoded as latin-1")
        text = normalise_whitespace(text)
        if not text:
            raise EmptyDocumentError(f"{source}: file contains no text")
        file_type = source.rsplit(".", 1)[-1].lower() if "." in source else "txt"
        return ParsedDocument(
            source=source, file_type=file_type, sections=[Section(text=text)], warnings=warnings
        )


class PdfParser:
    """PDF via PyMuPDF; one section per page so citations can name the page."""

    extensions = ("pdf",)

    def parse(self, data: bytes, source: str) -> ParsedDocument:
        import pymupdf

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # PyMuPDF raises several unrelated types for bad input
            raise CorruptDocumentError(f"{source}: not a readable PDF ({exc})") from exc

        with doc:
            if doc.needs_pass:
                raise CorruptDocumentError(f"{source}: PDF is password-protected")
            sections, scanned_pages = self._extract_pages(doc)

        warnings = []
        if scanned_pages:
            warnings.append(
                f"{len(scanned_pages)} page(s) look scanned (no text layer) and were skipped: "
                f"{scanned_pages[:10]}"
            )
        if not sections:
            if scanned_pages:
                raise OCRRequiredError(
                    f"{source}: no extractable text; the PDF appears to be scanned images. "
                    "OCR is not supported."
                )
            raise EmptyDocumentError(f"{source}: PDF contains no text")
        return ParsedDocument(source=source, file_type="pdf", sections=sections, warnings=warnings)

    @staticmethod
    def _extract_pages(doc) -> tuple[list[Section], list[int]]:
        sections: list[Section] = []
        scanned_pages: list[int] = []
        for page_number, page in enumerate(doc, start=1):
            text = normalise_whitespace(page.get_text("text", sort=True))
            if len(text) >= _SCANNED_PAGE_MAX_CHARS:
                sections.append(Section(text=text, page=page_number))
            elif page.get_images():
                scanned_pages.append(page_number)
            elif text:
                sections.append(Section(text=text, page=page_number))
        return sections, scanned_pages


class DocxParser:
    """DOCX paragraphs followed by table rows (cells joined with ' | ')."""

    extensions = ("docx",)

    def parse(self, data: bytes, source: str) -> ParsedDocument:
        import docx

        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:  # python-docx raises zipfile/KeyError/lxml errors
            raise CorruptDocumentError(f"{source}: not a readable DOCX ({exc})") from exc

        blocks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        text = normalise_whitespace("\n\n".join(blocks))
        if not text:
            raise EmptyDocumentError(f"{source}: DOCX contains no text")
        return ParsedDocument(source=source, file_type="docx", sections=[Section(text=text)])
