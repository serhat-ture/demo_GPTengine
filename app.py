# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Lock

from core import RAGService, OPENAI_MODEL

app = FastAPI(title="opentrue-ai RAG API")

# CORS (gerekirse domainlerinle sınırla)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag = None
_rag_lock = Lock()

def get_rag():
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                print("🟡 Initializing RAGService (lazy)...")
                _rag = RAGService()
                print("🟢 RAGService ready.")
    return _rag

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"ok": True, "service": "opentrue-ai", "hint": "See /health, /chat"}

@app.get("/health")
def health():
    global _rag
    if _rag is None:
        # Henüz ilk istek gelmemiş (lazy init bekliyor)
        return {"status": "booting", "model": OPENAI_MODEL}
    try:
        return {"status": "ok", "model": OPENAI_MODEL, "chunks": len(_rag.chunks), "top_k": 3}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/chat")
def chat(req: ChatRequest):
    rag = get_rag()
    answer = rag.answer(req.query)
    return {"answer": answer}
