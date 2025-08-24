# core.py
# -*- coding: utf-8 -*-
"""
Core RAG service extracted from your existing script.
IMPORTANT: Keeps OPENAI_MODEL='gpt-5-nano' unchanged.
"""

# core.py (güncelle)
import os
import json
import pdfplumber
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

PDF_PATH = "Keep_Me_Certified_MA_Exam_Prep_Book.pdf"
RULES_PATH = "rules.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
OPENAI_MODEL = "gpt-5-nano"

# Artefact yolları
ART_DIR = os.path.join("cache", "artifacts")
CHUNKS_JSON = os.path.join(ART_DIR, "chunks.json")
EMB_NUMPY = os.path.join(ART_DIR, "embeddings.npy")
FAISS_IDX = os.path.join(ART_DIR, "index.faiss")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY (env or App Settings)")
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="📄 Extracting text"):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    i = 0
    total_chunks = max(1, (len(text) - overlap) // (size - overlap) + 1)
    for _ in tqdm(range(total_chunks), desc="✂️ Chunking text"):
        end = min(start + size, len(text))
        chunk = f"[Chunk {i}]\n{text[start:end]}"
        chunks.append(chunk)
        start += size - overlap
        i += 1
        if start >= len(text):
            break
    return chunks

def embed_chunks(chunks, model: SentenceTransformer):
    return np.array(model.encode(chunks), dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query: str, model: SentenceTransformer, index, chunks, top_k: int = TOP_K):
    query_embedding = np.array(model.encode([query]), dtype=np.float32)
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

def get_openai_response(messages, model_name: str = OPENAI_MODEL) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(❌ OpenAI Error: {e})"

class RAGService:
    def __init__(self,
                 pdf_path: str = PDF_PATH,
                 rules_path: str = RULES_PATH,
                 embed_model_name: str = EMBED_MODEL):
        os.makedirs(ART_DIR, exist_ok=True)

        # Rules
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"❌ Rules file not found: {rules_path}")
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules_text = f.read()

        self.embed_model = SentenceTransformer(embed_model_name)

        # 1) Varsa artefact’ları yükle
        if all(os.path.exists(p) for p in [CHUNKS_JSON, EMB_NUMPY, FAISS_IDX]):
            with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(EMB_NUMPY)
            self.index = faiss.read_index(FAISS_IDX)
        else:
            # 2) Yoksa üret ve kaydet
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
            text = extract_text(pdf_path)
            self.chunks = chunk_text(text)
            self.embeddings = embed_chunks(self.chunks, self.embed_model)
            self.index = build_faiss_index(self.embeddings)

            # Kaydet
            with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
            np.save(EMB_NUMPY, self.embeddings)
            faiss.write_index(self.index, FAISS_IDX)

    def answer(self, query: str) -> str:
        relevant = retrieve_relevant_chunks(query, self.embed_model, self.index, self.chunks)
        context = "\n\n".join(relevant)
        system_prompt = f"""You are a helpful real estate assistant.
Always follow these rules:

{self.rules_text}

Use this book content to help answer:

{context}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        return get_openai_response(messages)
