"""IngestionService with real parsers/chunker/store and the hashing embedder."""

from rag_generator.storage import NumpyVectorStore
from tests.conftest import (
    ALT_SAMPLE_DIR,
    SAMPLE_DIR,
    FakeLLM,
    cite_first_source,
    make_image_only_pdf,
    make_pdf,
    make_pptx,
    make_xlsx,
)


def test_ingest_sample_directory(make_app):
    app = make_app()
    report = app.ingestion().ingest_paths([SAMPLE_DIR], "northwind")
    assert report.failed == 0
    assert report.succeeded == len(list(SAMPLE_DIR.iterdir()))
    collection = app.repository.open("northwind")
    assert len(collection.catalog) == report.succeeded
    assert len(collection.store) == sum(r.num_chunks for r in report.results)
    pdf = collection.catalog.find_by_source("nw200_operator_manual.pdf")
    assert pdf.num_pages == 4


def test_reingesting_identical_content_is_a_noop(make_app):
    app = make_app()
    app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    chunks_before = len(app.repository.open("c").store)
    report = app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    assert {r.status for r in report.results} == {"skipped_duplicate"}
    assert len(app.repository.open("c").store) == chunks_before


def test_changed_file_with_same_name_replaces_old_version(make_app):
    app = make_app()
    service = app.ingestion()
    service.ingest_bytes([("policy.txt", b"The refund window is 14 days for all orders.")], "c")
    report = service.ingest_bytes(
        [("policy.txt", b"The refund window is 30 days for all orders.")], "c"
    )
    assert report.results[0].status == "replaced"
    collection = app.repository.open("c")
    assert len(collection.catalog) == 1
    texts = [c.text for c in collection.store.all_chunks()]
    assert texts == ["The refund window is 30 days for all orders."]


def test_bad_files_fail_individually_without_aborting_batch(make_app):
    app = make_app()
    report = app.ingestion().ingest_bytes(
        [
            ("good.txt", b"This document has perfectly good content in it."),
            ("empty.txt", b""),
            ("scan.pdf", make_image_only_pdf()),
            ("broken.pdf", b"%PDF-1.4 garbage"),
            ("sheet.xls", b"legacy binary workbook"),
            ("also_good.pdf", make_pdf(["A valid PDF page with enough text to index."])),
        ],
        "c",
    )
    status = {r.source: (r.status, r.error_type) for r in report.results}
    assert status["good.txt"] == ("ingested", None)
    assert status["also_good.pdf"] == ("ingested", None)
    assert status["empty.txt"] == ("failed", "EmptyDocumentError")
    assert status["scan.pdf"] == ("failed", "OCRRequiredError")
    assert status["broken.pdf"] == ("failed", "CorruptDocumentError")
    assert status["sheet.xls"] == ("failed", "UnsupportedFileTypeError")
    assert len(app.repository.open("c").catalog) == 2


def test_file_size_limit(make_app):
    app = make_app(max_file_mb=0.001)
    report = app.ingestion().ingest_bytes([("big.txt", b"word " * 1000)], "c")
    assert report.results[0].error_type == "FileTooLargeError"


def test_collections_are_isolated_and_persist(make_app, settings):
    app = make_app()
    app.ingestion().ingest_paths([SAMPLE_DIR], "northwind")
    app.ingestion().ingest_paths([ALT_SAMPLE_DIR], "gardening")
    fresh = make_app()  # simulates a process restart: everything reloads from disk
    northwind = {c.source for c in fresh.repository.open("northwind").store.all_chunks()}
    gardening = {c.source for c in fresh.repository.open("gardening").store.all_chunks()}
    assert northwind.isdisjoint(gardening)
    assert "composting_guide.txt" in gardening
    assert sorted(fresh.repository.list_names()) == ["gardening", "northwind"]


def test_delete_document(make_app):
    app = make_app()
    report = app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    doc_id = report.results[0].doc_id
    record = app.ingestion().delete_document("c", doc_id)
    assert record.doc_id == doc_id
    collection = app.repository.open("c")
    assert collection.catalog.get(doc_id) is None
    assert all(c.doc_id != doc_id for c in collection.store.all_chunks())
    assert app.ingestion().delete_document("c", "missing") is None


def test_directory_walk_skips_unsupported_and_hidden_files(make_app, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("Useful markdown content for the index.")
    (tmp_path / "docs" / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "docs" / ".hidden.txt").write_text("should be ignored entirely")
    report = make_app().ingestion().ingest_paths([tmp_path / "docs"], "c")
    assert [r.source for r in report.results] == ["a.md"]


def test_embedding_model_change_is_detected(make_app, settings):
    make_app().ingestion().ingest_paths([ALT_SAMPLE_DIR], "c")
    other = make_app(hashing_dimensions=512)
    collection = other.repository.open("c")
    assert isinstance(collection.store, NumpyVectorStore)
    report = other.ingestion().ingest_bytes([("new.txt", b"Some brand new content here.")], "c")
    assert report.results[0].error_type == "IndexMismatchError"


def test_tabular_and_slide_formats_are_ingested_and_citable(make_app):
    app = make_app()
    report = app.ingestion().ingest_bytes(
        [
            ("prices.csv", b"SKU,Price\nNW-200,4999\nNW-C2,349\n"),
            ("stock.xlsx", make_xlsx({"Stock": [["SKU", "Warehouse"], ["NW-C2", "Leeds"]]})),
            ("deck.pptx", make_pptx([["Intro slide text here"], ["The NW-C2 ships in Q4."]])),
        ],
        "c",
    )
    assert [r.status for r in report.results] == ["ingested"] * 3
    chunks = {c.source: c for c in app.repository.open("c").store.all_chunks()}
    assert chunks["prices.csv"].text.startswith("Row 2: SKU: NW-200 | Price: 4999")
    assert chunks["stock.xlsx"].text == "[Stock] Row 2: SKU: NW-C2 | Warehouse: Leeds"

    answer = make_app(llm=FakeLLM(cite_first_source)).query("c").ask("Which warehouse is NW-C2 in?")
    assert answer.citations
    deck = [p.chunk for p in answer.passages if p.chunk.source == "deck.pptx"]
    assert all(c.page in (1, 2) for c in deck)
