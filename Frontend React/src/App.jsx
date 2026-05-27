import "./App.css";
import { useState, useEffect, useRef } from "react";
import TopBar from "./components/TopBar";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import { fetchDocuments, ingestFile, queryDocumentsStream, deleteDocuments } from "./api";

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
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [checking, setChecking] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null); // { type: "error"|"success", text }

  function showMsg(type, text) {
    setUploadMsg({ type, text });
    setTimeout(() => setUploadMsg(null), 2000);
  }

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

  useEffect(() => { checkConnection(); }, []);

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
        showMsg("error", `"${file.name}" is already loaded.`);
        return;
      }
      if (!ok) {
        showMsg("error", data.detail ?? "Upload failed.");
        return;
      }

      setDocs((prev) => [...prev, { name: data.filename, chunks: data.chunks_created }]);
      showMsg("success", `"${data.filename}" added successfully.`);
    } catch {
      showMsg("error", "Upload failed. Make sure the backend is running.");
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
    setStreaming(false);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let firstToken = true;
      await queryDocumentsStream(text, 3, SESSION_ID, {
        onToken(token) {
          if (firstToken) {
            firstToken = false;
            setStreaming(true);
            setMessages((prev) => [...prev, { role: "assistant", content: token, sources: [] }]);
          } else {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = { ...last, content: last.content + token };
              return updated;
            });
          }
        },
        onDone({ sources }) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = { ...last, sources };
            return updated;
          });
        },
        signal: controller.signal,
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Could not reach the backend. Make sure the FastAPI server is running on port 8000.",
            sources: [],
          },
        ]);
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
      setStreaming(false);
    }
  }

  function abort() {
    abortRef.current?.abort();
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
          streaming={streaming}
          docs={docs}
          uploading={uploading}
          onUploadFile={uploadFile}
          onAsk={ask}
        />
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          docs={docs}
          loading={loading}
          onAsk={ask}
          onAbort={abort}
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
