import "./App.css";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API = "http://localhost:8000";

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
  const chatRef = useRef(null);
  const textareaRef = useRef(null);

  async function checkConnection() {
    setChecking(true);
    try {
      const res = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocs(data.documents.map((name) => ({ name })));
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      setChecking(false);
    }
  }

  // load existing documents on mount
  useEffect(() => {
    (async () => {
      setChecking(true);
      try {
        const res = await fetch(`${API}/documents`);
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
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
      const data = await res.json();

      setDocs((prev) => {
        // don't add duplicate
        if (prev.find((d) => d.name === data.filename)) return prev;
        return [...prev, { name: data.filename, chunks: data.chunks_created }];
      });
    } catch {
      alert("Upload failed. Make sure the backend is running.");
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
        headers: { "Content-Type": "application/json" },
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
    await fetch(`${API}/documents`, { method: "DELETE" });
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
    </div>
  );
}
