# scripts/chat_rag_ollama.py
import json
import subprocess
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# -------------------------
# --- Load embeddings ---
# -------------------------
data = []
embedding_files = [
    Path("./docs/embeddings/EGamma_pages_embeddings_local.json"),
    Path("./docs/embeddings/MuonPerformanceAnalysis_embeddings_local.json")
]

for fname in embedding_files:
    with open(fname, "r", encoding="utf-8") as f:
        data.extend(json.load(f))

print(f"Loaded {len(data)} chunks of embeddings")

# -------------------------
# --- Cleaning function ---
# -------------------------
def clean_text(raw):
    text = raw.replace("(BUTTON)", "")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d{2,}", "", text)
    return text.strip()

# -------------------------
# --- FAISS index setup ---
# -------------------------
embedding_dim = len(data[0]["embedding"])
embeddings_matrix = np.ascontiguousarray(
    np.array([chunk["embedding"] for chunk in data], dtype=np.float32)
)

# L2 distance index (safer than IndexFlatIP for large arrays)
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_matrix)

# -------------------------
# --- Retrieval using FAISS ---
# -------------------------
def retrieve(query_embedding, top_k=3):
    query_vec = np.ascontiguousarray(np.array(query_embedding, dtype=np.float32).reshape(1, -1))
    D, I = index.search(query_vec, top_k)
    return [data[i] for i in I[0]]

# -------------------------
# --- Ask LLM ---
# -------------------------
def ask_llm(context: str, question: str):
    context_clean = context.replace('"', '\\"')
    prompt = f'Context: {context_clean}\nQuestion: {question}\nAnswer:'
    result = subprocess.run(
        ['bash', '-c', f'echo "{prompt}" | ollama run llama3.1'],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# -------------------------
# --- Initialize embedding model ---
# -------------------------
MODEL = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL)

def get_query_embedding(query):
    """Generate embedding for the user query using the same model as the page embeddings."""
    embedding = embed_model.encode(query)
    return embedding.tolist()  # convert to list to match FAISS input

# -------------------------
# --- Interactive loop ---
# -------------------------
if __name__ == "__main__":
    print("Mini-RAG Chatbot with FAISS (Ctrl+C to quit)")
    while True:
        query = input("\nYour question> ").strip()
        if not query:
            continue

        query_emb = get_query_embedding(query)
        retrieved_chunks = retrieve(query_emb, top_k=3)
        context = " ".join([clean_text(chunk["text"][:1000]) for chunk in retrieved_chunks])
        answer = ask_llm(context, query)
        print("\n--- Answer ---\n", answer)
