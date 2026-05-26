import sys
import numpy as np
from unittest.mock import MagicMock
import pytest # type: ignore

# Patch heavy deps before importing rag so the module-level model loading never runs.
_mock_fitz = MagicMock()
_mock_collection = MagicMock()
_mock_chroma_client = MagicMock()
_mock_chroma_client.get_or_create_collection.return_value = _mock_collection
_mock_embed = MagicMock()

sys.modules["fitz"] = _mock_fitz
sys.modules["chromadb"] = MagicMock(PersistentClient=MagicMock(return_value=_mock_chroma_client))
sys.modules["sentence_transformers"] = MagicMock(SentenceTransformer=MagicMock(return_value=_mock_embed), CrossEncoder=MagicMock())

import rag  # noqa: E402 — must come after sys.modules patching


@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_fitz.reset_mock()
    _mock_embed.reset_mock()
    _mock_collection.reset_mock()
    _mock_chroma_client.reset_mock()
    _mock_chroma_client.get_or_create_collection.return_value = _mock_collection


# ---------------------------------------------------------------------------
# semantic_chunk
# ---------------------------------------------------------------------------

class TestSemanticChunk:
    def test_empty_text_returns_empty(self):
        assert rag.semantic_chunk("") == []

    def test_single_sentence_returns_one_chunk(self, monkeypatch):
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk("Only one sentence here.")
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_similar_sentences_stay_in_same_chunk(self, monkeypatch):
        # identical vectors → similarity = 1.0 → never splits
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk("First sentence. Second sentence. Third sentence.")
        assert len(result) == 1

    def test_dissimilar_sentences_split_into_chunks(self, monkeypatch):
        # orthogonal vectors → similarity = 0.0 → splits at every sentence boundary
        def mock_encode(sentences):
            vecs = [[1.0, 0.0], [0.0, 1.0]] * len(sentences)
            return np.array(vecs[:len(sentences)])
        monkeypatch.setattr(rag.EMBED, "encode", mock_encode)
        long = ". ".join(["word " * 60] * 6) + "."
        result = rag.semantic_chunk(long, min_words=50)
        assert len(result) > 1

    def test_all_chunks_are_strings(self, monkeypatch):
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk("One. Two. Three.")
        assert all(isinstance(c, str) for c in result)


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
        _mock_fitz.open.assert_called_once_with(stream=b"data", filetype="pdf")


# ---------------------------------------------------------------------------
# process_pdf
# ---------------------------------------------------------------------------

class TestProcessPDF:
    def _setup_encode(self, vectors):
        _mock_embed.encode.return_value.tolist.return_value = vectors

    def test_returns_chunk_count(self, monkeypatch):
        chunks = ["chunk1", "chunk2", "chunk3"]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode([[0.1] * 384] * 3)

        assert rag.process_pdf(b"pdf", "doc.pdf", "session-123") == 3

    def test_adds_correct_documents_and_ids(self, monkeypatch):
        chunks = ["chunk A", "chunk B"]
        vectors = [[0.1] * 384, [0.2] * 384]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode(vectors)

        rag.process_pdf(b"pdf", "doc.pdf", "session-123")

        kwargs = _mock_collection.add.call_args.kwargs
        assert kwargs["documents"] == chunks
        assert kwargs["embeddings"] == vectors
        assert kwargs["ids"] == ["doc.pdf_chunk_0", "doc.pdf_chunk_1"]

    def test_metadata_includes_source_and_chunk_index(self, monkeypatch):
        chunks = ["only chunk"]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: "text")
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode([[0.0] * 384])

        rag.process_pdf(b"pdf", "paper.pdf", "session-123")

        meta = _mock_collection.add.call_args.kwargs["metadatas"]
        assert meta == [{"source": "paper.pdf", "chunk_index": 0}]


# ---------------------------------------------------------------------------
# clear_docs
# ---------------------------------------------------------------------------

class TestClearDocs:
    def test_deletes_then_recreates_collection(self):
        rag.clear_docs("session-123")
        _mock_chroma_client.delete_collection.assert_called_once_with("documents_session-123")
        _mock_chroma_client.get_or_create_collection.assert_called_with("documents_session-123")


# ---------------------------------------------------------------------------
# list_docs
# ---------------------------------------------------------------------------

class TestListDocs:
    def test_returns_empty_list_when_no_documents(self):
        _mock_collection.count.return_value = 0
        assert rag.list_docs("session-123") == []

    def test_returns_unique_filenames(self):
        _mock_collection.count.return_value = 3
        _mock_collection.get.return_value = {
            "metadatas": [
                {"source": "a.pdf", "chunk_index": 0},
                {"source": "a.pdf", "chunk_index": 1},
                {"source": "b.pdf", "chunk_index": 0},
            ]
        }
        assert sorted(rag.list_docs("session-123")) == ["a.pdf", "b.pdf"]

    def test_does_not_query_collection_when_empty(self):
        _mock_collection.count.return_value = 0
        rag.list_docs("session-123")
        _mock_collection.get.assert_not_called()
