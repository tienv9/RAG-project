import { useRef } from "react";

export default function TopBar({ docs, uploading, connected, checking, onAddFile, onClearDocs, onCheckConnection }) {
  const addFileRef = useRef(null);

  return (
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
              onChange={(e) => { onAddFile(e.target.files[0]); e.target.value = ""; }}
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
          <button className="clear-btn" onClick={onClearDocs}>
            ✕ clear all
          </button>
        )}
        <div className="topbar-sub">
          <span className={`status-dot ${connected ? "" : "disconnected"}`} />
          {connected ? "RAG pipeline ready" : "RAG pipeline not connected"}
          <button
            className={`refresh-btn ${checking ? "spinning" : ""}`}
            onClick={onCheckConnection}
            disabled={checking}
            title="Check connection"
          >↻</button>
        </div>
      </div>
    </div>
  );
}
