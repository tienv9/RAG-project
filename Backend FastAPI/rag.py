import re
import fitz # type: ignore
import chromadb # type: ignore
import ollama # type: ignore
import numpy as np # type: ignore
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

def semantic_chunk(text: str, threshold: float = 0.5, max_words: int = 500, min_words: int = 50) -> list[str]:
    """
        1. Split text into sentences
        2. Embed all sentences in one batch
        3. Compute cosine similarity between consecutive sentence embeddings
        4. Start a new chunk where similarity drops below threshold (topic shift)
        5. Merge chunks that are too small into their neighbor
        6. Split chunks that are too large by word count
        
        This can break since with regex split limitation on thing like e.g. or number in text,
        may also miss context if the context stay on previous chunk like the question is on the paragraph before it.
        the 0.5 threshold is a pure guess, would need extensive testing to find perfect threshold for specific type of doc
        
    """
    # Split using regex on . ! ? followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    embeddings = EMBED.encode(sentences)

    # compare each sentence to the next one to find where the topic changes
    raw_chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if sim < threshold: # topic changed so save the current chunk and start a new one
            raw_chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    raw_chunks.append(" ".join(current))

    # when similarity drops below 0.5, merge sentence into a new chunk to avoid orphan sentences
    merged = []
    buffer = ""
    for chunk in raw_chunks:
        buffer = (buffer + " " + chunk).strip() if buffer else chunk
        if len(buffer.split()) >= min_words:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    # Split any chunk that exceeded max_words back down by word count
    final = []
    for chunk in merged:
        words = chunk.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                final.append(" ".join(words[i:i + max_words]))
        else:
            final.append(chunk)

    return final


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
    textChunks = semantic_chunk(rawText)

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

    # attach the source file and chunk number to each chunk so the llm knows where it came from
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
