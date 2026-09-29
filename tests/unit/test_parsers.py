import pytest

from rag_generator.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    OCRRequiredError,
    UnsupportedFileTypeError,
)
from rag_generator.ingestion import ParserRegistry
from tests.conftest import SAMPLE_DIR, make_docx, make_image_only_pdf, make_pdf


@pytest.fixture
def registry() -> ParserRegistry:
    return ParserRegistry()


def test_pdf_has_one_section_per_page_with_page_numbers(registry):
    doc = registry.parse(make_pdf(["First page text here.", "Second page text here."]), "a.pdf")
    assert [s.page for s in doc.sections] == [1, 2]
    assert "Second page" in doc.sections[1].text
    assert doc.file_type == "pdf"


def test_text_and_markdown(registry):
    doc = registry.parse(b"# Title\n\nSome   body   text.\r\n", "notes.md")
    assert doc.sections[0].page is None
    assert doc.sections[0].text == "# Title\n\nSome body text."


def test_non_utf8_text_falls_back_to_latin1_with_warning(registry):
    doc = registry.parse("Café prices".encode("latin-1"), "menu.txt")
    assert "Café" in doc.sections[0].text
    assert doc.warnings


def test_docx_paragraphs_and_tables(registry):
    data = make_docx(["Hello paragraph."], table=[["Version", "Date"], ["3.3", "June"]])
    doc = registry.parse(data, "r.docx")
    assert "Hello paragraph." in doc.sections[0].text
    assert "3.3 | June" in doc.sections[0].text


def test_sample_corpus_parses(registry):
    for path in SAMPLE_DIR.iterdir():
        doc = registry.parse(path.read_bytes(), path.name)
        assert doc.char_count > 200, path.name


@pytest.mark.parametrize("name,data", [("empty.txt", b""), ("blank.md", b"  \n\n \t ")])
def test_empty_text_file(registry, name, data):
    with pytest.raises(EmptyDocumentError):
        registry.parse(data, name)


def test_corrupt_pdf(registry):
    with pytest.raises(CorruptDocumentError):
        registry.parse(b"%PDF-1.7 this is not really a pdf", "broken.pdf")


def test_corrupt_docx(registry):
    with pytest.raises(CorruptDocumentError):
        registry.parse(b"PK\x03\x04 not a zip", "broken.docx")


def test_scanned_pdf_requires_ocr(registry):
    with pytest.raises(OCRRequiredError):
        registry.parse(make_image_only_pdf(), "scan.pdf")


def test_blank_pdf_is_empty(registry):
    with pytest.raises(EmptyDocumentError):
        registry.parse(make_pdf([""]), "blank.pdf")


@pytest.mark.parametrize("name", ["data.xlsx", "image.png", "noextension"])
def test_unsupported_types(registry, name):
    with pytest.raises(UnsupportedFileTypeError, match="Supported"):
        registry.parse(b"whatever", name)
