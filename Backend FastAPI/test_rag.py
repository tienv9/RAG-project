import re as _re
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
sys.modules["spacy"] = MagicMock()

import rag  # noqa: E402 — must come after sys.modules patching


def _make_nlp_mock(text):
    """Simulate spaCy sentence splitting using regex so tests don't load the real model."""
    doc = MagicMock()
    raw = [s for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    sents = []
    for s in raw:
        m = MagicMock()
        m.text = s
        sents.append(m)
    doc.sents = sents
    return doc


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
    def test_empty_pages_returns_empty(self):
        assert rag.semantic_chunk([]) == []

    def test_single_sentence_returns_one_chunk(self, monkeypatch):
        monkeypatch.setattr(rag, "NLP", _make_nlp_mock)
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk([(1, "Only one sentence here.")])
        assert len(result) == 1
        assert isinstance(result[0][0], str)
        assert result[0][1] == [1]

    def test_similar_sentences_stay_in_same_chunk(self, monkeypatch):
        monkeypatch.setattr(rag, "NLP", _make_nlp_mock)
        # identical vectors → similarity = 1.0 → never splits
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk([(1, "First sentence. Second sentence. Third sentence.")])
        assert len(result) == 1

    def test_dissimilar_sentences_split_into_chunks(self, monkeypatch):
        monkeypatch.setattr(rag, "NLP", _make_nlp_mock)
        # orthogonal vectors → similarity = 0.0 → splits at every sentence boundary
        def mock_encode(sentences):
            vecs = [[1.0, 0.0], [0.0, 1.0]] * len(sentences)
            return np.array(vecs[:len(sentences)])
        monkeypatch.setattr(rag.EMBED, "encode", mock_encode)
        long = ". ".join(["word " * 60] * 6) + "."
        result = rag.semantic_chunk([(1, long)], min_words=50)
        assert len(result) > 1

    def test_chunks_contain_text_and_page_list(self, monkeypatch):
        monkeypatch.setattr(rag, "NLP", _make_nlp_mock)
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk([(1, "One. Two. Three.")])
        assert all(isinstance(c[0], str) for c in result)
        assert all(isinstance(c[1], list) for c in result)

    def test_page_numbers_tracked_across_pages(self, monkeypatch):
        monkeypatch.setattr(rag, "NLP", _make_nlp_mock)
        # identical vectors → all sentences stay in one chunk spanning both pages
        monkeypatch.setattr(rag.EMBED, "encode", lambda s: np.array([[1.0, 0.0]] * len(s)))
        result = rag.semantic_chunk([(1, "First sentence."), (2, "Second sentence.")])
        assert 1 in result[0][1]
        assert 2 in result[0][1]


# ---------------------------------------------------------------------------
# extract_text_from_PDF
# ---------------------------------------------------------------------------

class TestExtractTextFromPDF:
    def test_single_page(self):
        page = MagicMock()
        page.get_text.return_value = "Hello from page 1"
        page.number = 0  # fitz is 0-indexed
        _mock_fitz.open.return_value = [page]

        result = rag.extract_text_from_PDF(b"fake pdf")
        assert result == [(1, "Hello from page 1")]

    def test_multi_page_returns_list_of_tuples(self):
        pages = [MagicMock(), MagicMock()]
        pages[0].get_text.return_value = "Page one. "
        pages[0].number = 0
        pages[1].get_text.return_value = "Page two."
        pages[1].number = 1
        _mock_fitz.open.return_value = pages

        result = rag.extract_text_from_PDF(b"fake pdf")
        assert result == [(1, "Page one. "), (2, "Page two.")]

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
        chunks = [("chunk1", [1]), ("chunk2", [1]), ("chunk3", [2])]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: [(1, "text")])
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode([[0.1] * 384] * 3)

        assert rag.process_pdf(b"pdf", "doc.pdf", "session-123") == 3

    def test_adds_correct_documents_and_ids(self, monkeypatch):
        chunks = [("chunk A", [1]), ("chunk B", [2])]
        vectors = [[0.1] * 384, [0.2] * 384]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: [(1, "text")])
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode(vectors)

        rag.process_pdf(b"pdf", "doc.pdf", "session-123")

        kwargs = _mock_collection.add.call_args.kwargs
        assert kwargs["documents"] == ["chunk A", "chunk B"]
        assert kwargs["embeddings"] == vectors
        assert kwargs["ids"] == ["doc.pdf_chunk_0", "doc.pdf_chunk_1"]

    def test_metadata_includes_source_chunk_index_and_pages(self, monkeypatch):
        chunks = [("only chunk", [3])]
        monkeypatch.setattr(rag, "extract_text_from_PDF", lambda b: [(3, "text")])
        monkeypatch.setattr(rag, "semantic_chunk", lambda t, **kw: chunks)
        self._setup_encode([[0.0] * 384])

        rag.process_pdf(b"pdf", "paper.pdf", "session-123")

        meta = _mock_collection.add.call_args.kwargs["metadatas"]
        assert meta == [{"source": "paper.pdf", "chunk_index": 0, "pages": "3"}]


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
                {"source": "a.pdf", "chunk_index": 0, "pages": "1"},
                {"source": "a.pdf", "chunk_index": 1, "pages": "2"},
                {"source": "b.pdf", "chunk_index": 0, "pages": "1"},
            ]
        }
        assert sorted(rag.list_docs("session-123")) == ["a.pdf", "b.pdf"]

    def test_does_not_query_collection_when_empty(self):
        _mock_collection.count.return_value = 0
        rag.list_docs("session-123")
        _mock_collection.get.assert_not_called()
