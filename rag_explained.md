# RAG (Retrieval-Augmented Generation) — explained simply

---

## What is RAG?

You know how if you ask ChatGPT something about your own documents or notes, it has no idea?
That's because LLMs only know what they were trained on. They can't read *your* stuff.

RAG fixes that. It lets you give the LLM your own documents at the time you ask a question,
so it can answer based on *your* content instead of making things up.

Think of it like this:
- Without RAG → you're asking a smart person who's never seen your documents
- With RAG → you hand them the relevant pages first, then ask the question

---

## The core idea (in one sentence)

> Convert your documents into numbers that represent meaning, store them,
> then when a user asks a question — find the closest matching numbers,
> grab the original text, and feed it to the LLM to write an answer.

---

## Two stages

RAG has two stages. You only run Stage 1 once. Stage 2 runs every time a user asks something.

---

## Stage 1 — Ingestion (run once)

This is where you process and store your documents.

### Step 1 — Load the document

Pull the raw text out of your PDF, Word doc, or whatever file you have.

```python
# example
with open("notes.pdf", "rb") as f:
    text = extract_text(f)
```

### Step 2 — Chunk it

Don't embed the whole document as one giant blob. Split it into smaller passages
(around 300-500 words each). Each chunk should cover one idea or topic.

Why not individual words? Because a single word has no context.
The word "bank" alone could mean a river bank or a money bank.
But "the bank approved my loan application" as a chunk is clearly about finance.

Also add a small overlap (~50 words) between chunks so no sentence gets cut off
right at a boundary and loses its meaning.

```
Full document
│
├── Chunk 1: "avoid nuts and shellfish. symptoms include swelling..."
├── Chunk 2: "...swelling of the throat. in severe cases administer..."  ← overlaps with chunk 1
└── Chunk 3: "...administer epinephrine. always carry an EpiPen..."
```

### Step 3 — Embed each chunk

Run each chunk through an embedding model. This converts the text into a list of numbers
called a vector. The vector captures the *meaning* of the chunk, not just the words.

```python
# one API call per chunk
vector = openai.embeddings.create(
    input="avoid nuts and shellfish...",
    model="text-embedding-3-small"
).data[0].embedding

# vector looks like: [0.82, -0.14, 0.55, 0.31, -0.67, ...]
# OpenAI gives you 1,536 numbers per chunk
```

Similar meaning = similar numbers = close together in vector space.
"cat" and "feline" end up near each other. "cat" and "skyscraper" end up far apart.

The model learned this on its own by training on billions of sentences —
nobody hand-wrote the numbers. It figured out that words appearing in similar
contexts should have similar vectors.

### Step 4 — Store in a vector database

Save both the vector AND the original text together. You need both:
- the vector for searching
- the original text to send to the LLM later (you can't send raw numbers to it)

```python
collection.add(
    documents=["avoid nuts and shellfish..."],   # original text
    embeddings=[[0.82, -0.14, 0.55, ...]],       # vector
    metadatas=[{"source": "notes.pdf", "page": 1}],
    ids=["chunk_1"]
)
```

---

## Stage 2 — Query (runs every time a user asks something)

### Step 1 — Embed the question

Take the user's question and run it through the exact same embedding model.
No chunking needed — the question is already short.

```python
query = "what foods should I avoid with a nut allergy?"

query_vector = openai.embeddings.create(
    input=query,
    model="text-embedding-3-small"   # same model as ingestion
).data[0].embedding
```

### Step 2 — Search for closest chunks

Compare the query vector against every stored vector using cosine similarity.
This is just math — it returns a score between 0 and 1 for how close two vectors are.

- Score near 1.0 → very similar meaning
- Score near 0.0 → totally unrelated

Grab the top 3-5 closest chunks.

```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3   # top 3 closest chunks
)
```

### Step 3 — Build the prompt

Pack the retrieved chunks + the original question into one prompt.
The key instruction is telling the LLM to only use the provided context.
This is what stops it from hallucinating.

```
Answer using only the context below. If you don't know, say so.

Context:
[chunk 1]: "avoid nuts and shellfish. symptoms include swelling of the throat..."
[chunk 2]: "in severe cases, administer epinephrine immediately..."

Question: what foods should I avoid with a nut allergy?
```

### Step 4 — Send to LLM and return the answer

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

answer = response.choices[0].message.content
# "You should avoid nuts and shellfish. In severe reactions,
#  epinephrine should be administered immediately. [Source: notes.pdf p.1]"
```

---

## Full picture

```
INGESTION (once)
PDF → chunk into passages → embed each chunk → store vector + text in ChromaDB

QUERY (every request)
question → embed question → find closest chunks → build prompt → LLM → answer
```

---

## Stack (fits your resume)

| Layer | Tool | Why |
|---|---|---|
| RAG pipeline | Python | entire ML ecosystem lives here |
| Embeddings | OpenAI text-embedding-3-small | cheap, fast, great quality |
| Vector DB | ChromaDB | free, local, zero setup |
| API | FastAPI | like ASP.NET Core but Python |
| Frontend | React | you already know this |
| Deploy | AWS | you already know this |

---

## Where it gets harder (v2 improvements)

- **Smarter chunking** — split at paragraph/section boundaries instead of fixed word count
- **Reranking** — after retrieval, run a second model to re-score chunks for actual relevance
- **Citations** — return which chunk the answer came from so users can verify it
- **Evaluation** — measure whether answers are actually grounded in the retrieved context
- **HyDE** — generate a fake answer first, embed that, search with it (works better for vague questions)

None of these are needed for v1. Get the basic pipeline working first.

---

## Why this is a good portfolio project

- You built an ML pipeline end to end, not just called an API
- You can explain embeddings, vector search, and retrieval at a technical level
- The ML minor on your resume now has a project behind it
- Easy to demo live in an interview — upload a doc, ask a question, show the source citation
