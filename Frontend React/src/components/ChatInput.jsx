import { useRef } from "react";

export default function ChatInput({ question, setQuestion, docs, loading, onAsk }) {
  const textareaRef = useRef(null);

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onAsk();
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
        className="send-btn"
        onClick={onAsk}
        disabled={!question.trim() || loading || docs.length === 0}
      >
        ↑
      </button>
    </div>
  );
}
