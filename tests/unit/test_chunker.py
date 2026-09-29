from itertools import pairwise

import pytest

from rag_generator.chunking import RecursiveChunker
from rag_generator.models import ParsedDocument, Section

PROSE = " ".join(
    f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(200)
)


def _doc(*sections: Section) -> ParsedDocument:
    return ParsedDocument(source="doc.txt", file_type="txt", sections=list(sections))


@pytest.mark.parametrize("size,overlap", [(200, 40), (500, 100), (1000, 150)])
def test_chunks_respect_size_bound(size, overlap):
    chunks = RecursiveChunker(size, overlap).split_text(PROSE)
    assert len(chunks) > 1
    assert all(len(c) <= size for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = RecursiveChunker(300, 80).split_text(PROSE)
    for previous, current in pairwise(chunks):
        assert current[:20] in previous  # each chunk starts with the previous one's tail


def test_prefers_paragraph_boundaries():
    text = ("Alpha paragraph. " * 8).strip() + "\n\n" + ("Beta paragraph. " * 8).strip()
    chunks = RecursiveChunker(200, 0).split_text(text)
    assert chunks[0].startswith("Alpha") and "Beta" not in chunks[0]
    assert chunks[1].startswith("Beta")


def test_no_separator_text_is_hard_split():
    chunks = RecursiveChunker(100, 10).split_text("x" * 450)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).count("x") >= 450


def test_chunks_never_cross_pages_and_keep_page_numbers():
    doc = _doc(
        Section(text="Page one content. " * 5, page=1),
        Section(text="Page two content. " * 5, page=2),
    )
    chunks = RecursiveChunker(1000, 100).chunk(doc, "abc123")
    assert [c.page for c in chunks] == [1, 2]
    assert "two" not in chunks[0].text


def test_ids_are_deterministic_and_sequential():
    doc = _doc(Section(text=PROSE))
    a = RecursiveChunker(300, 50).chunk(doc, "0123456789abcdef")
    b = RecursiveChunker(300, 50).chunk(doc, "0123456789abcdef")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].chunk_id == "0123456789ab-00000"
    assert [c.index for c in a] == list(range(len(a)))


def test_tiny_fragments_are_dropped():
    doc = _doc(Section(text="3", page=1), Section(text="Real content that matters here.", page=2))
    chunks = RecursiveChunker(300, 50, min_chars=10).chunk(doc, "d")
    assert [c.page for c in chunks] == [2]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        RecursiveChunker(100, 100)


@pytest.mark.parametrize("size,overlap", [(200, 40), (700, 140), (1000, 0)])
def test_no_text_is_lost(size, overlap):
    """Every word of the input appears in some chunk (regression: short trailing lines)."""
    text = ("Policy paragraph with enough words to fill space. " * 13).strip()
    text += "\n\nRefund: 14 days.\n\nContact: help@example.com"
    chunks = RecursiveChunker(size, overlap, min_chars=20).split_text(text)
    joined = " ".join(chunks)
    for word in set(text.split()):
        assert word in joined, word
    assert all(len(c) <= size + 60 for c in chunks)  # small overflow from absorbed tails
