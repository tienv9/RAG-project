import fitz # type: ignore
import chromadb # type: ignore
import ollama # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

OLLAMA_MODEL = "llama3.2"  # change to any model you have pulled locally

# 384-dim embeddings, free, CPU-friendly. Upgrade to text-embedding-3-small (OpenAI) for better recall.
EMBED = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ChromaDB persists vectors to disk so re-ingesting on every restart isn't needed.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def extract_text_from_PDF(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()
        links = [link["uri"] for link in page.get_links() if link.get("uri")]
        if links:
            text += "\nLinks on this page: " + ", ".join(links) + "\n"

    return text

def break_down_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    # Overlap re-includes the last `overlap` words in the next chunk so sentences
    # that fall on a boundary still appear in full in at least one chunk.
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_pdf(file_bytes: bytes, filename: str) -> int:
    rawText = extract_text_from_PDF(file_bytes)
    textChunks = break_down_text(rawText)

    vectors = EMBED.encode(textChunks).tolist()

    collection.add(
        documents=textChunks,
        embeddings=vectors,
        metadatas=[{"source": filename, "chunk_index": i} for i, _ in enumerate(textChunks)],
        ids=[f"{filename}_chunk_{i}" for i, _ in enumerate(textChunks)],
    )

    return len(textChunks)

def query(question: str, top_k: int = 3) -> dict:
    question_vector = EMBED.encode([question]).tolist()

    # Cosine similarity search — returns the top_k most semantically similar chunks.
    results = collection.query(
        query_embeddings=question_vector,
        n_results=top_k,
    )

    matched_chunks = results["documents"][0]
    matched_sources = results["metadatas"][0]

    # Pack retrieved chunks into a labeled context block for the LLM prompt.
    context = "\n\n".join(
        [f"[Source: {m['source']} chunk {m['chunk_index']}]\n{chunk}"
         for chunk, m in zip(matched_chunks, matched_sources)]
    )

    prompt = f'''
                You are a retrieval-augmented assistant.
                Answer the user's question using ONLY the provided context.

                Rules:
                - Do not use outside knowledge.
                - If the context does not contain enough information, say: "I don't have enough information to answer that."
                - Do not fabricate facts or details.
                - Ignore irrelevant context.
                - Keep the answer concise and accurate.

                Context:
                {context}

                Question:
                {question}

                Answer:

                '''

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    answer = response["response"].strip()

    # Deduplicate sources — same file may appear across multiple matched chunks.
    sources = list({m["source"] for m in matched_sources})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": matched_chunks,
    }

def clear_docs():
    # ChromaDB has no truncate — delete and recreate is the only way to wipe all vectors.
    chroma_client.delete_collection("documents")
    chroma_client.get_or_create_collection("documents")

def list_docs() -> list[str]:
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    # Set comprehension deduplicates filenames when multiple chunks share the same source.
    return list({m["source"] for m in all_meta})