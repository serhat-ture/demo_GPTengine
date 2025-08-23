# core.py
# -*- coding: utf-8 -*-
"""
Core RAG service extracted from your existing script.
IMPORTANT: Keeps OPENAI_MODEL='gpt-5-nano' unchanged.
"""

import os
import pdfplumber
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIGURATION (same as your script) ===
PDF_PATH = "Keep_Me_Certified_MA_Exam_Prep_Book.pdf"
RULES_PATH = "rules.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"  # SBERT embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
OPENAI_MODEL = "gpt-5-nano"  # DO NOT CHANGE without your approval

# === LOAD OPENAI CLIENT ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY (env or App Settings)")
client = OpenAI(api_key=OPENAI_API_KEY)


# === Original functions (kept same behavior) ===
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
    total_chunks = (len(text) - overlap) // (size - overlap) + 1
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
    # SentenceTransformer returns float32 by default; keep as numpy
    return np.array(model.encode(chunks))


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(query: str, model: SentenceTransformer, index, chunks, top_k: int = TOP_K):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
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
    """
    Loads PDF + rules on startup, builds embeddings & FAISS index,
    and answers queries using your system prompt + retrieved context.
    """
    def __init__(self,
                 pdf_path: str = PDF_PATH,
                 rules_path: str = RULES_PATH,
                 embed_model_name: str = EMBED_MODEL):
        # Validate files
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"❌ Rules file not found: {rules_path}")

        # Load rules
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules_text = f.read()

        # Build chunks & embeddings
        text = extract_text(pdf_path)
        self.chunks = chunk_text(text)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embeddings = embed_chunks(self.chunks, self.embed_model)

        # Build FAISS index
        self.index = build_faiss_index(self.embeddings)

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
            {"role": "user", "content": query}
        ]
        return get_openai_response(messages)
