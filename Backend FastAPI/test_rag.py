import sys
from unittest.mock import MagicMock
import pytest

# Patch heavy deps before importing rag so the module-level model loading never runs.
_mock_fitz = MagicMock()
_mock_collection = MagicMock()
_mock_chroma_client = MagicMock()
_mock_chroma_client.get_or_create_collection.return_value = _mock_collection
_mock_embed = MagicMock()
_mock_llm = MagicMock()

sys.modules["fitz"] = _mock_fitz
sys.modules["chromadb"] = MagicMock(PersistentClient=MagicMock(return_value=_mock_chroma_client))
sys.modules["sentence_transformers"] = MagicMock(SentenceTransformer=MagicMock(return_value=_mock_embed))
sys.modules["transformers"] = MagicMock(pipeline=MagicMock(return_value=_mock_llm))

import rag  # noqa: E402 — must come after sys.modules patching


@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_fitz.reset_mock()
    _mock_embed.reset_mock()
    _mock_llm.reset_mock()
    _mock_collection.reset_mock()
    _mock_chroma_client.reset_mock()
    _mock_chroma_client.get_or_create_collection.return_value = _mock_collection


# ---------------------------------------------------------------------------
# break_down_text — pure function, no mocks needed
# ---------------------------------------------------------------------------

class TestBreakDownText:
    def test_basic_chunking(self):
        words = [f"w{i}" for i in range(100)]
        chunks = rag.break_down_text(" ".join(words), chunk_size=10, overlap=0)
        assert len(chunks) == 10
        assert chunks[0] == " ".join(words[:10])
        assert chunks[-1] == " ".join(words[90:])

    def test_overlap_repeats_tail_words(self):
        words = [f"w{i}" for i in range(20)]
        chunks = rag.break_down_text(" ".join(words), chunk_size=10, overlap=2)
        # stride = 10 - 2 = 8, so chunk[1] starts at word index 8
        assert chunks[1].startswith("w8")

    def test_text_shorter_than_chunk_size(self):
        chunks = rag.break_down_text("hello world", chunk_size=500, overlap=50)
        assert chunks == ["hello world"]

    def test_empty_text(self):
        assert rag.break_down_text("") == []

    def test_text_exactly_chunk_size(self):
        words = [f"w{i}" for i in range(10)]
        chunks = rag.break_down_text(" ".join(words), chunk_size=10, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == " ".join(words)

    def test_last_chunk_is_partial(self):
        # 15 words, chunk_size=10, overlap=0 → two chunks: 10 words + 5 words
        words = [f"w{i}" for i in range(15)]
        chunks = rag.break_down_text(" ".join(words), chunk_size=10, overlap=0)
        assert len(chunks) == 2
        assert chunks[1] == " ".join(words[10:])


# ---------------------------------------------------------------------------
# extract_text_from_PDF
# ---------------------------------------------------------------------------

class TestExtractTextFromPDF:
    def test_single_page(self):
        page = MagicMock()
        page.get_text.return_value = "Hello from page 1"
        _mock_fitz.open.return_value = [page]

        result = rag.extract_text_from_PDF(b"fake pdf")
        assert result == "Hello from page 1"

    def test_multi_page_concatenates(self):
        pages = [MagicMock(), MagicMock()]
        pages[0].get_text.return_value = "Page one. "
        pages[1].get_text.return_value = "Page two."
        _mock_fitz.open.return_value = pages

        result = rag.extract_text_from_PDF(b"fake pdf")
        assert result == "Page one. Page two."

    def test_passes_bytes_and_filetype(self):
        _mock_fitz.open.return_value = []
        rag.extract_text_from_PDF(b"data")
        _mock_fitz.open.assert_called_once_with(b"data", filetype="pdf")


# ---------------------------------------------------------------------------
# process_pdf
# ---------------------------------------------------------------------------

class TestProcessPDF:
    def _setup_encode(self, vectors):
        _mock_embed.encode.return_value.tolist.return_value = vectors

    def test_returns_chunk_count(self, monkeypatch):
        chunks = ["chunk1", "chunk2", "chunk3"]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "break_down_text", lambda t, **kw: chunks)
        self._setup_encode([[0.1] * 384] * 3)

        assert rag.process_pdf(b"pdf", "doc.pdf") == 3

    def test_adds_correct_documents_and_ids(self, monkeypatch):
        chunks = ["chunk A", "chunk B"]
        vectors = [[0.1] * 384, [0.2] * 384]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "break_down_text", lambda t, **kw: chunks)
        self._setup_encode(vectors)

        rag.process_pdf(b"pdf", "doc.pdf")

        kwargs = _mock_collection.add.call_args.kwargs
        assert kwargs["documents"] == chunks
        assert kwargs["embeddings"] == vectors
        assert kwargs["ids"] == ["doc.pdf_chunk_0", "doc.pdf_chunk_1"]

    def test_metadata_includes_source_and_chunk_index(self, monkeypatch):
        chunks = ["only chunk"]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "break_down_text", lambda t, **kw: chunks)
        self._setup_encode([[0.0] * 384])

        rag.process_pdf(b"pdf", "paper.pdf")

        meta = _mock_collection.add.call_args.kwargs["metadatas"]
        assert meta == [{"source": "paper.pdf", "chunk_index": 0}]


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def _setup(self, docs, metas):
        _mock_embed.encode.return_value.tolist.return_value = [[0.1] * 384]
        _mock_collection.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
        }
        # generated_text = prompt + answer; use a long prefix so slicing works cleanly.
        _mock_llm.return_value = [{"generated_text": "x" * 5000 + "the answer"}]

    def test_returns_expected_keys(self):
        self._setup(["chunk"], [{"source": "a.pdf", "chunk_index": 0}])
        result = rag.query("What is X?")
        assert set(result.keys()) == {"answer", "sources", "chunks_used"}

    def test_chunks_used_matches_retrieval(self):
        docs = ["chunk one", "chunk two"]
        metas = [{"source": "a.pdf", "chunk_index": 0}, {"source": "a.pdf", "chunk_index": 1}]
        self._setup(docs, metas)
        assert rag.query("question")["chunks_used"] == docs

    def test_deduplicates_sources(self):
        docs = ["c1", "c2", "c3"]
        metas = [
            {"source": "a.pdf", "chunk_index": 0},
            {"source": "a.pdf", "chunk_index": 1},
            {"source": "b.pdf", "chunk_index": 0},
        ]
        self._setup(docs, metas)
        sources = rag.query("question")["sources"]
        assert sorted(sources) == ["a.pdf", "b.pdf"]

    def test_embeds_question_before_querying(self):
        self._setup(["c"], [{"source": "x.pdf", "chunk_index": 0}])
        rag.query("my question")
        _mock_embed.encode.assert_called_once_with(["my question"])

    def test_queries_collection_with_top_k(self):
        self._setup(["c"], [{"source": "x.pdf", "chunk_index": 0}])
        rag.query("q", top_k=5)
        _mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1] * 384],
            n_results=5,
        )


# ---------------------------------------------------------------------------
# clear_doc
# ---------------------------------------------------------------------------

class TestClearDoc:
    def test_deletes_then_recreates_collection(self):
        rag.clear_doc()
        _mock_chroma_client.delete_collection.assert_called_once_with("documents")
        _mock_chroma_client.get_or_create_collection.assert_called_with("documents")


# ---------------------------------------------------------------------------
# list_doc
# ---------------------------------------------------------------------------

class TestListDoc:
    def test_returns_empty_list_when_no_documents(self):
        _mock_collection.count.return_value = 0
        assert rag.list_doc() == []

    def test_returns_unique_filenames(self):
        _mock_collection.count.return_value = 3
        _mock_collection.get.return_value = {
            "metadatas": [
                {"source": "a.pdf", "chunk_index": 0},
                {"source": "a.pdf", "chunk_index": 1},
                {"source": "b.pdf", "chunk_index": 0},
            ]
        }
        assert sorted(rag.list_doc()) == ["a.pdf", "b.pdf"]

    def test_does_not_query_collection_when_empty(self):
        _mock_collection.count.return_value = 0
        rag.list_doc()
        _mock_collection.get.assert_not_called()
