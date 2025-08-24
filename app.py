# app.py (küçük ekleme)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import RAGService, OPENAI_MODEL

app = FastAPI(title="openai_Assistant_RAG API")

# CORS - geliştirme için tüm origin'lere izin (isteğe göre daralt)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # prod'da kendi domainlerinle sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
rag = None
ready = False

@app.on_event("startup")
def _startup():
    global rag, ready
    try:
        rag = RAGService()
        ready = True
    except Exception as e:
        ready = False
        raise e

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {
        "status": "ok" if ready else "initializing",
        "model": OPENAI_MODEL,
        "chunks": 0 if not ready else len(rag.chunks),
        "top_k": 3
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not ready:
        raise HTTPException(status_code=503, detail="Service is initializing, try again in a minute.")
    answer = rag.answer(req.query)
    return {"answer": answer}
