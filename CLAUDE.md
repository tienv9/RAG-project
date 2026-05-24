# RAG Study Assistant — project context

## What this is
A RAG (Retrieval-Augmented Generation) app that lets users upload any PDF
and ask questions about it. Answers are grounded in the document with source citations.

## My background
- New CS grad from EWU, strong in full-stack (React, Vue, ASP.NET Core, AWS)
- ML minor — understand embeddings, vectors, cosine similarity, backprop
- Building this as a portfolio project to show AI/ML depth on resume
- First RAG project — learned the concepts from scratch during this build

## How RAG works (my mental model)
1. PDF → extract text → chunk into ~500 word passages with 50 word overlap
2. Each chunk → embedding model → vector (list of 1,536 numbers representing meaning)
3. Store vector + original text in ChromaDB
4. At query time: embed the question → find closest chunk vectors → pack into LLM prompt → answer

Key insight: similar meaning = close vectors. "cat" and "feline" end up near each other
in vector space because the embedding model learned this from billions of sentences.

## Current stack
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace, free, 80MB, runs on CPU)
- **LLM**: `google/flan-t5-base` (HuggingFace, free, 250MB, runs on CPU)
- **Vector DB**: ChromaDB (local, persistent)
- **API**: FastAPI (Python)
- **Frontend**: React (Vite)
- **Deploy target**: AWS

## Project structure
```
rag-app/
├── backend/
│   ├── rag.py          # core pipeline: ingest_pdf(), query(), chunk_text()
│   ├── main.py         # FastAPI server, 4 endpoints
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.jsx     # React UI, file upload + chat interface
└── README.md
```

## API endpoints
- POST /ingest     — upload PDF, returns chunk count
- POST /query      — ask question, returns answer + sources + chunks_used
- GET  /documents  — list all ingested documents
- DELETE /documents — clear all documents

## Shell
Always use PowerShell syntax for commands — never Bash. The user is on Windows and cannot run Bash.
- Use `.\venv\Scripts\python` not `python` or `source venv/bin/activate`
- Use `;` to chain commands, not `&&`
- Use `$env:VAR` for environment variables, not `export VAR=`

## How to run
```powershell
# backend
cd "Backend FastAPI"; .\venv\Scripts\python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\uvicorn main:app --reload --port 8000

# frontend
cd "Frontend React"; npm run dev

# tests
cd "Backend FastAPI"; .\venv\Scripts\python -m pytest test_rag.py -v
```

## Planned improvements (v2)
- Swap flan-t5-base for a better LLM (Mistral or Claude API)
- Add reranking after retrieval
- Semantic chunking instead of fixed word count
- Source citations with page numbers
- pgvector on AWS RDS for production vector storage
- Evaluation metrics — check if answers are grounded in retrieved context
