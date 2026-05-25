import "./App.css";
import { useState, useEffect } from "react";
import TopBar from "./components/TopBar";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import { fetchDocuments, ingestFile, queryDocuments, deleteDocuments } from "./api";

//this can be a problem if edit session ID, extremely unlikely and wont need a fix this project
const SESSION_ID = (() => {
  let id = localStorage.getItem("rag_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("rag_session_id", id);
  }
  return id;
})();

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [connected, setConnected] = useState(false);
  const [checking, setChecking] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null); // { type: "error"|"success", text }

  async function checkConnection() {
    setChecking(true);
    try {
      const data = await fetchDocuments(SESSION_ID);
      setDocs(data.documents.map((name) => ({ name })));
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => {
    (async () => {
      setChecking(true);
      try {
        const data = await fetchDocuments(SESSION_ID);
        setDocs(data.documents.map((name) => ({ name })));
        setConnected(true);
      } catch {
        setConnected(false);
      } finally {
        setChecking(false);
      }
    })();
  }, []);

  async function uploadFile(file) {
    if (!file || !file.name.endsWith(".pdf")) {
      alert("Please upload a PDF file.");
      return;
    }

    setUploading(true);
    setUploadMsg(null);

    try {
      const { ok, status, data } = await ingestFile(file, SESSION_ID);

      if (status === 409) {
        setUploadMsg({ type: "error", text: `"${file.name}" is already loaded.` });
        setTimeout(() => setUploadMsg(null), 2000);
        return;
      }
      if (!ok) {
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
      const data = await queryDocuments(text, 3, SESSION_ID);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Could not reach the backend. Make sure the FastAPI server is running on port 8000.",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function clearDocs() {
    if (!confirm("Clear all documents?")) return;
    await deleteDocuments(SESSION_ID);
    setDocs([]);
    setMessages([]);
  }

  return (
    <div className="app">
      <main className="main">
        <TopBar
          docs={docs}
          uploading={uploading}
          connected={connected}
          checking={checking}
          onAddFile={uploadFile}
          onClearDocs={clearDocs}
          onCheckConnection={checkConnection}
        />
        <ChatWindow
          messages={messages}
          loading={loading}
          docs={docs}
          uploading={uploading}
          dragging={dragging}
          onUploadFile={uploadFile}
          onSetDragging={setDragging}
          onAsk={ask}
        />
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          docs={docs}
          loading={loading}
          onAsk={ask}
        />
      </main>

      {uploadMsg && (
        <div className={`toast toast--${uploadMsg.type}`}>
          {uploadMsg.text}
        </div>
      )}
    </div>
  );
}
