const API = "http://localhost:8000";

export async function fetchDocuments(sessionId) {
  const res = await fetch(`${API}/documents`, {
    headers: { "X-Session-ID": sessionId },
  });
  return res.json();
}

export async function ingestFile(file, sessionId) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/ingest`, {
    method: "POST",
    body: form,
    headers: { "X-Session-ID": sessionId },
  });
  return { ok: res.ok, status: res.status, data: await res.json() };
}

export async function queryDocuments(question, topK, sessionId) {
  const res = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-ID": sessionId },
    body: JSON.stringify({ question, top_k: topK }),
  });
  return res.json();
}

export async function deleteDocuments(sessionId) {
  await fetch(`${API}/documents`, {
    method: "DELETE",
    headers: { "X-Session-ID": sessionId },
  });
}
