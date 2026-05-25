# AskMyDocs

A RAG (Retrieval-Augmented Generation) app that lets you upload any PDF and ask questions about it. Answers are grounded in the document with source citations.

## How it works

1. Upload a PDF — text and hyperlinks are extracted and split into ~500 word chunks
2. Each chunk is embedded into a vector using `all-MiniLM-L6-v2`
3. Vectors are stored in ChromaDB under a per-session collection
4. Ask a question — it's embedded, matched against the closest chunks, and passed to an LLM
5. The LLM answers using only the retrieved context

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs on CPU) |
| LLM | Ollama `llama3.2` (local, swappable) |
| Vector DB | ChromaDB (persistent, per-session) |
| API | FastAPI (Python) |
| Frontend | React + Vite |

## Features

- PDF upload with drag-and-drop
- Hyperlink extraction — ask about URLs referenced in the document
- Per-session document isolation — each browser gets its own private store
- Duplicate detection — re-uploading the same file returns a 409
- Markdown-rendered answers
- Source citations showing which document the answer came from

## Getting started

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

```powershell
ollama pull llama3.2
```

### Backend

```powershell
cd "Backend FastAPI"
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\uvicorn main:app --reload --port 8000
```

### Frontend

```powershell
cd "Frontend React"
npm install
npm run dev
```

Open `http://localhost:5173`, upload a PDF, and start asking questions.

## API

All endpoints require the `X-Session-ID` header.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Upload a PDF, returns chunk count |
| POST | `/query` | Ask a question, returns answer + sources |
| GET | `/documents` | List documents for this session |
| DELETE | `/documents` | Clear all documents for this session |

## Planned improvements

- Semantic chunking instead of fixed word count
- Re-ranking after retrieval
- Source citations with page numbers
- pgvector on AWS RDS for production vector storage
- Evaluation metrics to verify answers are grounded in retrieved context
