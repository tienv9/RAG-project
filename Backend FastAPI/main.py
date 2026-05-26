import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Header # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import StreamingResponse # type: ignore
from pydantic import BaseModel # type: ignore
from rag import process_pdf, query_stream, list_docs, clear_docs, doc_exists

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

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG API is running"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_session_id: str = Header(...),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if doc_exists(file.filename, x_session_id):
        raise HTTPException(status_code=409, detail=f"{file.filename} is already ingested.")

    file_bytes = await file.read()
    chunk_count = process_pdf(file_bytes, file.filename, x_session_id)

    return {
        "message": f"Ingested {file.filename} successfully.",
        "chunks_created": chunk_count,
        "filename": file.filename,
    }


@app.post("/query/stream")
def ask_stream(request: QueryRequest, x_session_id: str = Header(...)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def generate():
        for event in query_stream(request.question, x_session_id, top_k=request.top_k):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/documents")
def get_docs(x_session_id: str = Header(...)):
    docs = list_docs(x_session_id)
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents")
def delete_docs(x_session_id: str = Header(...)):
    clear_docs(x_session_id)
    return {"message": "All documents cleared."}
