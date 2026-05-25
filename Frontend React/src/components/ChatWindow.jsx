import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const SUGGESTIONS = [
  "Summarize the main topics",
  "What are the key concepts?",
  "Explain the most important points",
  "What should I focus on?",
];

export default function ChatWindow({ messages, loading, docs, uploading, dragging, onUploadFile, onSetDragging, onAsk }) {
  const chatRef = useRef(null);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
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
            onDragOver={(e) => { e.preventDefault(); onSetDragging(true); }}
            onDragLeave={() => onSetDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              onSetDragging(false);
              onUploadFile(e.dataTransfer.files[0]);
            }}
          >
            <input
              type="file"
              accept=".pdf"
              disabled={uploading}
              onChange={(e) => onUploadFile(e.target.files[0])}
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
                <button key={s} className="suggestion-chip" onClick={() => onAsk(s)}>
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
                    <span key={s} className="source-tag">◆ {s}</span>
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
  );
}
