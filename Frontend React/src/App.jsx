import "./App.css";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API = "http://localhost:8000";

// Retrieve or create a persistent session ID so each browser gets its own
// isolated document store on the server.

//this can be a problem if edit session ID, extremely unlikely and wont need a fix this project
const SESSION_ID = (() => {
  let id = localStorage.getItem("rag_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("rag_session_id", id);
  }
  return id;
})();

const SUGGESTIONS = [
  "Summarize the main topics",
  "What are the key concepts?",
  "Explain the most important points",
  "What should I focus on?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [docs, setDocs] = useState([]); // { name, chunks }
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [connected, setConnected] = useState(false);
  const [checking, setChecking] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null); // { type: "error"|"success", text }
  const chatRef = useRef(null);
  const textareaRef = useRef(null);
  const addFileRef = useRef(null);

  async function checkConnection() {
    setChecking(true);
    try {
      const res = await fetch(`${API}/documents`, { headers: { "X-Session-ID": SESSION_ID } });
      const data = await res.json();
      setDocs(data.documents.map((name) => ({ name })));
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      setChecking(false);
    }
  }

  // load existing documents on mount — also verifies this session ID has data on the server
  useEffect(() => {
    (async () => {
      setChecking(true);
      try {
        const res = await fetch(`${API}/documents`, { headers: { "X-Session-ID": SESSION_ID } });
        const data = await res.json();
        setDocs(data.documents.map((name) => ({ name })));
        setConnected(true);
      } catch {
        setConnected(false);
      } finally {
        setChecking(false);
      }
    })();
  }, []);

  // auto scroll to bottom when messages change
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  async function uploadFile(file) {
    if (!file || !file.name.endsWith(".pdf")) {
      alert("Please upload a PDF file.");
      return;
    }

    setUploading(true);
    setUploadMsg(null);
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/ingest`, { method: "POST", body: form, headers: { "X-Session-ID": SESSION_ID } });
      const data = await res.json();

      if (res.status === 409) {
        setUploadMsg({ type: "error", text: `"${file.name}" is already loaded.` });
        setTimeout(() => setUploadMsg(null), 2000);
        return;
      }
      if (!res.ok) {
        setUploadMsg({ type: "error", text: data.detail ?? "Upload failed." });
        setTimeout(() => setUploadMsg(null), 2000);
        return;
      }

      setDocs((prev) => [...prev, { name: data.filename, chunks: data.chunks_created }]);
      setUploadMsg({ type: "success", text: `"${data.filename}" added successfully.` });
      setTimeout(() => setUploadMsg(null), 2000);
    } catch {
      setUploadMsg({ type: "error", text: "Upload failed. Make sure the backend is running." });
      setTimeout(() => setUploadMsg(null), 2000);
    } finally {
      setUploading(false);
    }
  }

  async function ask(q) {
    const text = q || question.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-ID": SESSION_ID },
        body: JSON.stringify({ question: text, top_k: 3 }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Could not reach the backend. Make sure the FastAPI server is running on port 8000.",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function clearDocs() {
    if (!confirm("Clear all documents?")) return;
    await fetch(`${API}/documents`, { method: "DELETE", headers: { "X-Session-ID": SESSION_ID } });
    setDocs([]);
    setMessages([]);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    uploadFile(file);
  }

  return (
    <div className="app">
      <main className="main">
        <div className="topbar">
          <div className="topbar-left">
            <span className="logo-icon">📄</span>
            <div>
              <div className="topbar-title">Ask your documents</div>
              <div className="topbar-sub">
                {docs.length === 0
                  ? "No documents loaded"
                  : `${docs.length} document${docs.length > 1 ? "s" : ""} loaded`}
              </div>
            </div>
            {docs.length > 0 && (
              <>
                <input
                  ref={addFileRef}
                  type="file"
                  accept=".pdf"
                  style={{ display: "none" }}
                  onChange={(e) => { uploadFile(e.target.files[0]); e.target.value = ""; }}
                />
                <button
                  className="add-doc-btn"
                  onClick={() => addFileRef.current.click()}
                  disabled={uploading}
                >
                  {uploading ? "Uploading…" : "+ Add PDF"}
                </button>
              </>
            )}
          </div>
          <div className="topbar-right">
            {docs.length > 0 && (
              <button className="clear-btn" onClick={clearDocs}>
                ✕ clear all
              </button>
            )}
            <div className="topbar-sub">
              <span className={`status-dot ${connected ? "" : "disconnected"}`} />
              {connected ? "RAG pipeline ready" : "RAG pipeline not connected"}
              <button
                className={`refresh-btn ${checking ? "spinning" : ""}`}
                onClick={checkConnection}
                disabled={checking}
                title="Check connection"
              >↻</button>
            </div>
          </div>
        </div>

        {/* Chat */}
        <div className="chat" ref={chatRef}>
          {messages.length === 0 && !loading ? (
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <div className="empty-title">Ask anything about your PDFs</div>
              <div className="empty-sub">
                Upload a document below, then ask questions.
                Answers will cite which document they came from.
              </div>

              <div
                className={`upload-zone ${dragging ? "dragging" : ""} ${uploading ? "uploading" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <input
                  type="file"
                  accept=".pdf"
                  disabled={uploading}
                  onChange={(e) => uploadFile(e.target.files[0])}
                />
                {uploading ? (
                  <div className="upload-loading">
                    <div className="upload-spinner" />
                    <div className="upload-loading-text">Processing PDF…</div>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon">⬆️</div>
                    <div className="upload-hint">
                      <strong>Click or drag</strong> a PDF here
                    </div>
                  </>
                )}
                {uploading && (
                  <div className="uploading-bar">
                    <div className="uploading-bar-fill" />
                  </div>
                )}
              </div>

              {docs.length > 0 && (
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      className="suggestion-chip"
                      onClick={() => ask(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="bubble"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources">
                      {msg.sources.map((s) => (
                        <span key={s} className="source-tag">
                          ◆ {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="message assistant">
                  <div className="thinking">
                    <div className="dots">
                      <div className="dot" />
                      <div className="dot" />
                      <div className="dot" />
                    </div>
                    searching documents...
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Input */}
        <div className="input-area">
          <div className="input-wrap">
            <textarea
              ref={textareaRef}
              placeholder={
                docs.length === 0
                  ? "Upload a PDF first..."
                  : "Ask a question about your documents..."
              }
              value={question}
              onChange={(e) => {
                const val = e.target.value;
                if (val.length <= 2000) setQuestion(val);
              }}
              onKeyDown={onKeyDown}
              rows={1}
              disabled={docs.length === 0 || loading}
            />
            {question.length > 0 && (
              <span className="char-count">{question.length} / 2000</span>
            )}
          </div>
          <button
            className="send-btn"
            onClick={() => ask()}
            disabled={!question.trim() || loading || docs.length === 0}
          >
            ↑
          </button>
        </div>
      </main>

      {uploadMsg && (
        <div className={`toast toast--${uploadMsg.type}`}>
          {uploadMsg.text}
        </div>
      )}
    </div>
  );
}
