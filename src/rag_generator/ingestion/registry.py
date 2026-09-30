"""Extension → parser lookup."""

from __future__ import annotations

from rag_generator.errors import UnsupportedFileTypeError
from rag_generator.ingestion.base import DocumentParser
from rag_generator.ingestion.parsers import (
    CsvParser,
    DocxParser,
    PdfParser,
    PptxParser,
    TextParser,
    XlsxParser,
)
from rag_generator.models import ParsedDocument


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        default = [PdfParser(), TextParser(), DocxParser(), CsvParser(), XlsxParser(), PptxParser()]
        for parser in parsers or default:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        for ext in parser.extensions:
            self._by_extension[ext.lower()] = parser

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(self._by_extension)

    def is_supported(self, source: str) -> bool:
        return _extension(source) in self._by_extension

    def parse(self, data: bytes, source: str) -> ParsedDocument:
        ext = _extension(source)
        parser = self._by_extension.get(ext)
        if parser is None:
            raise UnsupportedFileTypeError(
                f"{source}: unsupported file type '.{ext}'. "
                f"Supported: {', '.join('.' + e for e in self.supported_extensions)}"
            )
        return parser.parse(data, source)


def _extension(source: str) -> str:
    return source.rsplit(".", 1)[-1].lower() if "." in source else ""
