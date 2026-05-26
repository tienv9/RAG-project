import fitz # type: ignore
import chromadb # type: ignore
import ollama # type: ignore
from sentence_transformers import SentenceTransformer, CrossEncoder # type: ignore

OLLAMA_MODEL = "llama3.2"  # change to any model you have pulled locally

# 384-dim embeddings, free, CPU-friendly. Upgrade to text-embedding-3-small (OpenAI) for better recall.
EMBED = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cross-encoder re-ranks candidate chunks by reading question+chunk together — more accurate than cosine similarity alone.
RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ChromaDB persists vectors to disk so re-ingesting on every restart isn't needed.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(session_id: str):
    return chroma_client.get_or_create_collection(name=f"documents_{session_id}")


def extract_text_from_PDF(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()
        # make url a separate chunk to allow llm to search for chunk
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


def process_pdf(file_bytes: bytes, filename: str, session_id: str) -> int:
    """
        1. Extract text
        2. Split into chunks
        3. Embed each chunk to vector
        4. store vector in chromadb
    This then return how many chunks were stored.
    """
    
    collection = get_collection(session_id)
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

def query_stream(question: str, session_id: str, top_k: int = 3):
    """
        1. Embed the question into a vector
        2. Fetch top_k * 4 candidate chunks from ChromaDB by cosine similarity
        3. Re-rank candidates with cross-encoder, keep top_k
        4. Build context string from top chunks and send to LLM
        5. Stream LLM response token by token
        6. Yield a final done event with sources
    """
    
    collection = get_collection(session_id)

    # Embed question into a vector so ChromaDB can compare it against stored chunk vectors
    question_vector = EMBED.encode([question]).tolist()

    # Cast a wide net — fetch more candidates than needed so the re-ranker has room to work.
    # min() guards against requesting more results than chunks that actually exist.
    candidate_count = min(top_k * 4, collection.count())
    results = collection.query(
        query_embeddings=question_vector,
        n_results=candidate_count,
    )

    candidates = results["documents"][0]
    candidate_meta = results["metadatas"][0]

    # Re-rank: cross-encoder reads each (question, chunk) pair together and scores relevance.
    # More accurate than cosine similarity because it sees both texts at once instead of separately.
    pairs = [[question, chunk] for chunk in candidates]
    scores = RERANKER.predict(pairs)

    # zip ties each score to its chunk and metadata, sort descending, then unpack top_k
    ranked = sorted(zip(scores, candidates, candidate_meta), key=lambda x: x[0], reverse=True)
    matched_chunks = [chunk for _, chunk, _ in ranked[:top_k]]
    matched_sources = [meta for _, _, meta in ranked[:top_k]]

    # Label each chunk with its source file and position so the LLM can cite them
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

    sources = list({m["source"] for m in matched_sources})

    for chunk in ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=True):
        yield {"type": "token", "content": chunk.response}

    yield {"type": "done", "sources": sources, "chunks_used": matched_chunks}


def doc_exists(filename: str, session_id: str) -> bool:
    # check if it already exist inside the chromadb - this only compare name for now
    collection = get_collection(session_id)
    if collection.count() == 0:
        return False
    result = collection.get(where={"source": filename})
    return len(result["ids"]) > 0

def clear_docs(session_id: str):
    # ChromaDB has no truncate — delete and recreate is the only way to wipe all vectors.
    collection_name = f"documents_{session_id}"
    chroma_client.delete_collection(collection_name)
    chroma_client.get_or_create_collection(collection_name)

def list_docs(session_id: str) -> list[str]:
    # get list of all documents 
    collection = get_collection(session_id)
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    # Set comprehension deduplicates filenames when multiple chunks share the same source.
    return list({m["source"] for m in all_meta})
