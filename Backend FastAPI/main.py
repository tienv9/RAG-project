from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from rag import process_pdf, query, list_docs, clear_docs, doc_exists
app = FastAPI(title="RAG Assistant")

# Allow the React frontend (localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3   # how many chunks to retrieve

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG API is running"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the vector DB.
    Returns how many chunks were created.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if doc_exists(file.filename):
        raise HTTPException(status_code=409, detail=f"{file.filename} is already ingested.")

    file_bytes = await file.read()
    chunk_count = process_pdf(file_bytes, file.filename)

    return {
        "message": f"Ingested {file.filename} successfully.",
        "chunks_created": chunk_count,
        "filename": file.filename,
    }


@app.post("/query", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Ask a question. Returns an answer grounded in your uploaded documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = query(request.question, top_k=request.top_k)
    return result


@app.get("/documents")
def get_docs():
    """List all documents currently in the vector DB."""
    docs = list_docs()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents")
def delete_docs():
    """Clear all documents from the vector DB."""
    clear_docs()
    return {"message": "All documents cleared."}
