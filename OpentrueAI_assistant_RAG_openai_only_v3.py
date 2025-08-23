# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:33:43 2025

@author: dagnew
"""

# -*- coding: utf-8 -*-
"""
AI chatbot for real estate customer guidance
Author: Tewodros M. Dagnew (dagnewtewodrosm@gmail.com)
Year: Aug. 2025
"""

import os
import sys
import pdfplumber
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIGURATION ===
PDF_PATH = "Keep_Me_Certified_MA_Exam_Prep_Book.pdf"
RULES_PATH = "rules.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"  # SBERT embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
OPENAI_MODEL = "gpt-5-nano"  # Default model

# === LOAD OPENAI CLIENT ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file")
client = OpenAI(api_key=OPENAI_API_KEY)

# === STEP 1: Extract Text ===
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="📄 Extracting text"):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# === STEP 2: Chunk Text ===
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
    return chunks

# === STEP 3: Embed Chunks ===
def embed_chunks(chunks, model):
    return np.array(model.encode(chunks))

# === STEP 4: Build FAISS Index ===
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# === STEP 5: Retrieve Relevant Chunks ===
def retrieve_relevant_chunks(query, model, index, chunks, top_k=TOP_K):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# === STEP 6: Get OpenAI Response ===
def get_openai_response(messages, model_name=OPENAI_MODEL):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(❌ OpenAI Error: {e})"

# === STEP 7: Chat Interface ===
def chat(index, embed_model, chunks, rules_text):
    print(f"📚 Opentrue Real Estate Assistant is ready supported by (OpenAI: {OPENAI_MODEL}) AI engine. Type 'exit' to quit.\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"You: {query}")
        respond_to_query(query, index, embed_model, chunks, rules_text)
        return

    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        respond_to_query(query, index, embed_model, chunks, rules_text)

def respond_to_query(query, index, embed_model, chunks, rules_text):
    relevant_chunks = retrieve_relevant_chunks(query, embed_model, index, chunks)
    context = "\n\n".join(relevant_chunks)

    system_prompt = f"""You are a helpful real estate assistant.
Always follow these rules:

{rules_text}

Use this book content to help answer:

{context}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    answer = get_openai_response(messages)
    print("📘 AI:", answer)

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF file '{PDF_PATH}' not found.")
        exit(1)

    if not os.path.exists(RULES_PATH):
        print(f"❌ Rules file '{RULES_PATH}' not found.")
        exit(1)

    print("🔍 Extracting PDF...")
    text = extract_text(PDF_PATH)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("🔎 Embedding chunks...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(chunks, embed_model)

    print("📂 Building search index...")
    index = build_faiss_index(embeddings)

    rules_text = open(RULES_PATH, "r", encoding="utf-8").read()
    chat(index, embed_model, chunks, rules_text)
