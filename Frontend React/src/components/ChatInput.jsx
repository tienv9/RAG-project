import { useRef, useState } from "react";

export default function ChatInput({ question, setQuestion, docs, loading, onAsk, onAbort }) {
  const textareaRef = useRef(null);
  const [showDocs, setShowDocs] = useState(false);

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (loading) onAbort();
      else onAsk();
    }
  }

  return (
    <div className="input-area">
      <div className="docs-btn-wrap">
        <button
          className="docs-toggle-btn"
          onClick={() => setShowDocs((v) => !v)}
          title="View uploaded documents"
        >
          ☰
        </button>
        {showDocs && (
          <div className="docs-popup">
            <div className="docs-popup-header">
              <span>Uploaded PDFs</span>
              <button className="docs-popup-close" onClick={() => setShowDocs(false)}>✕</button>
            </div>
            <div className="docs-popup-list">
              {docs.length === 0 ? (
                <p className="docs-popup-empty">No documents uploaded yet.</p>
              ) : (
                docs.map((doc) => (
                  <div key={doc.name} className="docs-popup-item">
                    <span className="docs-popup-name">{doc.name}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
      <div className="input-wrap">
        <textarea
          ref={textareaRef}
          placeholder={docs.length === 0 ? "Upload a PDF first..." : "Ask a question about your documents..."}
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
        className={`send-btn${loading ? " send-btn--stop" : ""}`}
        onClick={loading ? onAbort : onAsk}
        disabled={loading ? false : (!question.trim() || docs.length === 0)}
      >
        {loading ? "■" : "↑"}
      </button>
    </div>
  );
}

