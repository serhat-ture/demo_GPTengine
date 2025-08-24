# core.py
# -*- coding: utf-8 -*-
"""
Core RAG service for OpenAI Assistant RAG (FastAPI backend).
- Artefact (önceden hazırlanmış) dosyaları varsa onları yükler,
  yoksa PDF'den üretir ve kaydeder.
- Yol/klasör karmaşasını engellemek için mutlak/bağıl path çözümlemesi yapar.
- Ayrıntılı tanılama (debug) çıktılarını loga yazar.

ÖNEMLİ: OPENAI_MODEL = 'gpt-5-nano' DEĞİŞTİRİLMEDİ.
"""

from __future__ import annotations

import os
import json
from typing import List
import numpy as np
import pdfplumber
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
#   SABİTLER / AYARLAR
# =========================

# Kullanıcı kodundan gelen isimler korunuyor:
PDF_PATH = "Keep_Me_Certified_MA_Exam_Prep_Book.pdf"
RULES_PATH = "rules.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"     # SBERT model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
OPENAI_MODEL = "gpt-5-nano"          # !!! DEĞİŞTİRME !!!

# Baz dizin (bu dosyanın bulunduğu yer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artefact klasörü (App Settings > ART_DIR ile override edilebilir)
_raw_art_dir = os.getenv("ART_DIR", os.path.join("cache", "artifacts"))
ART_DIR = _raw_art_dir if os.path.isabs(_raw_art_dir) else os.path.join(BASE_DIR, _raw_art_dir)

# Artefact dosyaları
CHUNKS_JSON = os.path.join(ART_DIR, "chunks.json")
EMB_NPY     = os.path.join(ART_DIR, "embeddings.npy")
FAISS_IDX   = os.path.join(ART_DIR, "index.faiss")

# OpenAI istemci (API key .env ya da App Settings'ten gelir)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY (env veya Azure App Settings).")
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
#   YARDIMCI FONKSİYONLAR
# =========================

def _abs_path(p: str) -> str:
    """Bağıl verilirse BASE_DIR altına çevir, mutlaksa dokunma."""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

def _debug_paths() -> None:
    """Log'a bakılan yolları ve mevcut/eksik durumlarını dök."""
    print("🔎 PATH DEBUG")
    print("  BASE_DIR   :", BASE_DIR)
    print("  ART_DIR    :", repr(ART_DIR))
    print("  chunks.json:", os.path.exists(CHUNKS_JSON), CHUNKS_JSON)
    print("  embeddings :", os.path.exists(EMB_NPY),     EMB_NPY)
    print("  index.faiss:", os.path.exists(FAISS_IDX),   FAISS_IDX)

def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="📄 Extracting text"):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0
    i = 0
    # Overlap'lı güvenli bölümleme
    step = max(1, size - overlap)
    total = max(1, (len(text) + step - 1) // step)
    for _ in tqdm(range(total), desc="✂️ Chunking text"):
        end = min(start + size, len(text))
        if start >= len(text):
            break
        chunk = f"[Chunk {i}]\n{text[start:end]}"
        chunks.append(chunk)
        start += step
        i += 1
    return chunks

def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    return np.array(model.encode(chunks))

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query: str, model: SentenceTransformer, index, chunks: List[str], top_k: int = TOP_K) -> List[str]:
    query_embedding = model.encode([query])
    _, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

def get_openai_response(messages, model_name: str = OPENAI_MODEL) -> str:
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(❌ OpenAI Error: {e})"


# =========================
#       RAG SERVİSİ
# =========================

class RAGService:
    """
    Başlatırken:
      1) Artefact'lar mevcutsa (chunks.json, embeddings.npy, index.faiss) -> doğrudan yükler (hızlı).
      2) Değilse PDF'yi işler, artefact'ları üretip kaydeder (bir defalık yavaş).
    """

    def __init__(self,
                 pdf_path: str = PDF_PATH,
                 rules_path: str = RULES_PATH,
                 embed_model_name: str = EMBED_MODEL):

        # Yol çözümleme (Linux case‑sensitive!)
        pdf_path   = _abs_path(pdf_path)
        rules_path = _abs_path(rules_path)

        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"❌ Rules file not found: {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules_text = f.read()

        # Küçük bir model; ilk indirme cache'e alınır (HF_HOME/SENTENCE_TRANSFORMERS_HOME ile yönlendirilebilir)
        self.embed_model = SentenceTransformer(embed_model_name)

        # Artefact kontrolü + tanılama
        _debug_paths()
        artifacts_exist = all(map(os.path.exists, [CHUNKS_JSON, EMB_NPY, FAISS_IDX]))

        if artifacts_exist:
            print(f"✅ Loading artifacts from: {ART_DIR}")
            with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(EMB_NPY)
            self.index = faiss.read_index(FAISS_IDX)
        else:
            print("🚧 Artifacts not found; building from PDF (slow)")
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")

            text = extract_text(pdf_path)
            self.chunks = chunk_text(text)
            self.embeddings = embed_chunks(self.chunks, self.embed_model)
            self.index = build_faiss_index(self.embeddings)

            os.makedirs(ART_DIR, exist_ok=True)
            print(f"💾 Saving artifacts to: {ART_DIR}")
            with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
            np.save(EMB_NPY, self.embeddings)
            faiss.write_index(self.index, FAISS_IDX)

    def answer(self, query: str) -> str:
        relevant_chunks = retrieve_relevant_chunks(query, self.embed_model, self.index, self.chunks)
        context = "\n\n".join(relevant_chunks)
        system_prompt = f"""You are a helpful real estate assistant.
Always follow these rules:

{self.rules_text}

Use this book content to help answer:

{context}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return get_openai_response(messages)
