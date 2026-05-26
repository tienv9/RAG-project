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

export async function queryDocumentsStream(question, topK, sessionId, { onToken, onDone, signal }) {
  const res = await fetch(`${API}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-ID": sessionId },
    body: JSON.stringify({ question, top_k: topK }),
    signal,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();

    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      const data = JSON.parse(part.slice(6));
      if (data.type === "token") onToken(data.content);
      else if (data.type === "done") onDone(data);
    }
  }
}

export async function deleteDocuments(sessionId) {
  await fetch(`${API}/documents`, {
    method: "DELETE",
    headers: { "X-Session-ID": sessionId },
  });
}
