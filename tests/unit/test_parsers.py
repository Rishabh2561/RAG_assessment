import pytest

from rag_generator.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    OCRRequiredError,
    UnsupportedFileTypeError,
)
from rag_generator.ingestion import ParserRegistry
from tests.conftest import (
    SAMPLE_DIR,
    make_docx,
    make_image_only_pdf,
    make_pdf,
    make_pptx,
    make_xlsx,
)


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


@pytest.mark.parametrize("name", ["legacy.xls", "legacy.ppt", "image.png", "noextension"])
def test_unsupported_types(registry, name):
    with pytest.raises(UnsupportedFileTypeError, match="Supported"):
        registry.parse(b"whatever", name)


def test_csv_rows_carry_header_names(registry):
    data = b"Region,Revenue,Owner\nWest,1200,Ana\n\nEast,,Raj\n"
    doc = registry.parse(data, "sales.csv")
    assert doc.file_type == "csv"
    assert doc.sections[0].page is None
    assert doc.sections[0].text.split("\n") == [
        "Row 2: Region: West | Revenue: 1200 | Owner: Ana",
        "Row 4: Region: East | Owner: Raj",
    ]


def test_csv_sniffs_semicolons_and_handles_tsv_and_latin1(registry):
    doc = registry.parse("Name;City\nJosé;Málaga\n".encode("latin-1"), "people.csv")
    assert doc.sections[0].text == "Row 2: Name: José | City: Málaga"
    assert doc.warnings
    tsv = registry.parse(b"a\tb\n1\t2\n", "t.tsv")
    assert tsv.sections[0].text == "Row 2: a: 1 | b: 2"


def test_csv_extra_cells_and_header_only(registry):
    doc = registry.parse(b"id,name\n7,x,overflow\n", "e.csv")
    assert doc.sections[0].text == "Row 2: id: 7 | name: x | Column 3: overflow"
    assert registry.parse(b"id,name\n", "h.csv").sections[0].text == "Columns: id | name"


@pytest.mark.parametrize("data", [b"", b" \n,,\n"])
def test_empty_csv(registry, data):
    with pytest.raises(EmptyDocumentError):
        registry.parse(data, "empty.csv")


def test_xlsx_one_section_per_sheet_with_sheet_name(registry):
    import datetime

    data = make_xlsx(
        {
            "Prices": [["SKU", "Price", "Since"], ["NW-200", 4999.0, datetime.date(2025, 3, 1)]],
            "Empty": [],
            "Stock": [["SKU", "Qty", "Active"], ["NW-C2", 12, True]],
        }
    )
    doc = registry.parse(data, "catalog.xlsx")
    assert doc.file_type == "xlsx"
    assert [s.text for s in doc.sections] == [
        "[Prices] Row 2: SKU: NW-200 | Price: 4999 | Since: 2025-03-01",
        "[Stock] Row 2: SKU: NW-C2 | Qty: 12 | Active: TRUE",
    ]


def test_empty_and_corrupt_xlsx(registry):
    with pytest.raises(EmptyDocumentError):
        registry.parse(make_xlsx({"Sheet1": []}), "blank.xlsx")
    with pytest.raises(CorruptDocumentError):
        registry.parse(b"PK\x03\x04 not a zip", "broken.xlsx")


def test_pptx_slides_are_pages_with_notes_and_tables(registry):
    data = make_pptx(
        [["Quarterly review", "Revenue grew 12%"], [], ["Roadmap"]],
        notes={1: "Mention the NW-200 launch."},
        table=[["Quarter", "Goal"], ["Q3", "Ship firmware 3.3"]],
    )
    doc = registry.parse(data, "deck.pptx")
    assert doc.file_type == "pptx"
    assert [s.page for s in doc.sections] == [1, 3]
    assert "Revenue grew 12%" in doc.sections[0].text
    assert "Notes: Mention the NW-200 launch." in doc.sections[0].text
    assert "Q3 | Ship firmware 3.3" in doc.sections[1].text
    assert any("[2]" in w for w in doc.warnings)


def test_empty_and_corrupt_pptx(registry):
    with pytest.raises(EmptyDocumentError):
        registry.parse(make_pptx([[]]), "blank.pptx")
    with pytest.raises(CorruptDocumentError):
        registry.parse(b"PK\x03\x04 not a zip", "broken.pptx")
