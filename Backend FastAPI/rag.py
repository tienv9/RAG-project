import re
import fitz # type: ignore
import chromadb # type: ignore
import ollama # type: ignore
import numpy as np # type: ignore
import spacy # type: ignore
from sentence_transformers import SentenceTransformer, CrossEncoder # type: ignore

OLLAMA_MODEL = "llama3.2"  # change to any model you have pulled locally

# 384-dim embeddings, free, CPU-friendly. Upgrade to text-embedding-3-small (OpenAI) for better recall.
EMBED = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cross-encoder re-ranks candidate chunks by reading question+chunk together — more accurate than cosine similarity alone.
RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Sentence boundary detection — handles abbreviations, decimals, URLs that trip up regex.
# ner/lemmatizer/attribute_ruler disabled since we only need the sentence segmenter.
NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])

# ChromaDB persists vectors to disk so re-ingesting on every restart isn't needed.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(session_id: str):
    return chroma_client.get_or_create_collection(name=f"documents_{session_id}")


def extract_text_from_PDF(file_bytes: bytes) -> list[tuple[int, str]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        # make url a separate chunk to allow llm to search for chunk
        links = [link["uri"] for link in page.get_links() if link.get("uri")]
        if links:
            text += "\nLinks on this page: " + ", ".join(links) + "\n"
        pages.append((page.number + 1, text))  # fitz is 0-indexed; store 1-indexed
    return pages

def semantic_chunk(pages: list[tuple[int, str]], threshold: float = 0.5, max_words: int = 500, min_words: int = 50) -> list[tuple[str, list[int]]]:
    """
        1. Split each page's text into sentences (spaCy, batched by paragraph)
        2. Embed all sentences in one batch
        3. Compute cosine similarity between consecutive sentence embeddings
        4. Start a new chunk where similarity drops below threshold (topic shift)
        5. Merge chunks that are too small into their neighbor
        6. Split chunks that are too large by word count
        Returns list of (chunk_text, [page_numbers]) tuples.

        the 0.5 threshold is a pure guess, would need extensive testing to find perfect threshold for specific type of doc

    """
    # Build (sentence, page_num) pairs, batching by paragraph so spaCy never
    # receives a doc large enough to hit memory limits.
    sentence_data: list[tuple[str, int]] = []
    for page_num, text in pages:
        for para in text.strip().split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) > 100_000:
                # degenerate paragraph (no blank lines in PDF) — regex is good enough here
                for s in re.split(r'(?<=[.!?])\s+', para):
                    if s.strip():
                        sentence_data.append((s.strip(), page_num))
            else:
                for sent in NLP(para).sents:
                    if sent.text.strip():
                        sentence_data.append((sent.text.strip(), page_num))

    if not sentence_data:
        return []

    sentences = [s for s, _ in sentence_data]
    sent_pages = [p for _, p in sentence_data]
    embeddings = EMBED.encode(sentences)

    # compare each sentence to the next one to find where the topic changes
    raw_chunks: list[tuple[str, list[int]]] = []
    current_sents = [sentences[0]]
    current_pages: set[int] = {sent_pages[0]}

    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if sim < threshold:
            raw_chunks.append((" ".join(current_sents), sorted(current_pages)))
            current_sents = [sentences[i]]
            current_pages = {sent_pages[i]}
        else:
            current_sents.append(sentences[i])
            current_pages.add(sent_pages[i])
    raw_chunks.append((" ".join(current_sents), sorted(current_pages)))

    # merge chunks that are too small into their neighbor
    merged: list[tuple[str, list[int]]] = []
    buf_text = ""
    buf_pages: set[int] = set()
    for chunk_text, chunk_pages in raw_chunks:
        buf_text = (buf_text + " " + chunk_text).strip() if buf_text else chunk_text
        buf_pages.update(chunk_pages)
        if len(buf_text.split()) >= min_words:
            merged.append((buf_text, sorted(buf_pages)))
            buf_text = ""
            buf_pages = set()
    if buf_text:
        if merged:
            prev_text, prev_pages = merged[-1]
            merged[-1] = (prev_text + " " + buf_text, sorted(set(prev_pages) | buf_pages))
        else:
            merged.append((buf_text, sorted(buf_pages)))

    # Split any chunk that exceeded max_words back down by word count
    final: list[tuple[str, list[int]]] = []
    for chunk_text, chunk_pages in merged:
        words = chunk_text.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                final.append((" ".join(words[i:i + max_words]), chunk_pages))
        else:
            final.append((chunk_text, chunk_pages))

    return final


def process_pdf(file_bytes: bytes, filename: str, session_id: str) -> int:
    collection = get_collection(session_id)
    pages = extract_text_from_PDF(file_bytes)
    chunks = semantic_chunk(pages)

    texts = [text for text, _ in chunks]  # unpack (text, page_nums) tuples from semantic_chunk
    vectors = EMBED.encode(texts).tolist()  # .tolist() converts numpy array — ChromaDB requires plain Python lists

    collection.add(
        documents=texts,
        embeddings=vectors,
        metadatas=[
            # pages as comma-separated string — ChromaDB metadata values must be scalars, not lists
            {"source": filename, "chunk_index": i, "pages": ",".join(str(p) for p in page_nums)}
            for i, (_, page_nums) in enumerate(chunks)
        ],
        ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
    )

    return len(chunks)

def query_stream(question: str, session_id: str, top_k: int = 3):
    collection = get_collection(session_id)

    # turn the question into a vector so chromadb can compare it to the stored chunks
    question_vector = EMBED.encode([question]).tolist()

    # grab more chunks than needed so the re-ranker can pick the best ones
    # min() avoids asking for more chunks than what actually exists in the collection
    candidate_count = min(top_k * 4, collection.count())
    results = collection.query(
        query_embeddings=question_vector,
        n_results=candidate_count,
    )

    candidates = results["documents"][0]
    candidate_meta = results["metadatas"][0]

    # score each chunk against the question to find which ones actually answer it
    pairs = [[question, chunk] for chunk in candidates]
    scores = RERANKER.predict(pairs)

    # sort by score and take the top chunks with their matching metadata
    ranked = sorted(zip(scores, candidates, candidate_meta), key=lambda x: x[0], reverse=True)
    matched_chunks = [chunk for _, chunk, _ in ranked[:top_k]]
    matched_sources = [meta for _, _, meta in ranked[:top_k]]

    context = "\n\n".join(
        f"[Source: {m['source']} chunk {m['chunk_index']}]\n{chunk}"
        for chunk, m in zip(matched_chunks, matched_sources)
    )

    prompt = (
        "You are a retrieval-augmented assistant.\n"
        "Answer the user's question using ONLY the provided context.\n\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- If the context does not contain enough information, say: \"I don't have enough information to answer that.\"\n"
        "- Do not fabricate facts or details.\n"
        "- Ignore irrelevant context.\n"
        "- Keep the answer concise and accurate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )

    # Deduplicate by (source, pages) — the same page can appear in multiple top-ranked chunks.
    citations = list({
        (m["source"], m["pages"]): {"source": m["source"], "pages": m["pages"]}
        for m in matched_sources
    }.values())

    for chunk in ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=True):
        yield {"type": "token", "content": chunk.response}

    yield {"type": "done", "sources": citations}


def doc_exists(filename: str, session_id: str) -> bool:
    # name-only check — two different files with the same name are treated as duplicates
    result = get_collection(session_id).get(where={"source": filename})
    return bool(result["ids"])

def clear_docs(session_id: str):
    # ChromaDB has no truncate — delete and recreate is the only way to wipe all vectors.
    collection_name = f"documents_{session_id}"
    chroma_client.delete_collection(collection_name)
    chroma_client.get_or_create_collection(collection_name)

def list_docs(session_id: str) -> list[str]:
    all_meta = get_collection(session_id).get(include=["metadatas"])["metadatas"]
    return list({m["source"] for m in all_meta})  # set deduplicates filenames across chunks
