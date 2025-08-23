# app.py
from fastapi import FastAPI
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

# App start: PDF + rules yükle, index hazırla (ilk açılışta biraz sürebilir)
rag = RAGService()

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "chunks": len(rag.chunks),
        "top_k": 3
    }

@app.post("/chat")
def chat(req: ChatRequest):
    answer = rag.answer(req.query)
    return {"answer": answer}
