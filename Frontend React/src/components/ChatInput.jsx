import { useRef } from "react";

export default function ChatInput({ question, setQuestion, docs, loading, onAsk, onAbort }) {
  const textareaRef = useRef(null);

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (loading) onAbort();
      else onAsk();
    }
  }

  return (
    <div className="input-area">
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
