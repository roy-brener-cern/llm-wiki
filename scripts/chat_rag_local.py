# chat_rag_local.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import subprocess
import tempfile
import re



EMB_FILE = "MuonPerformanceAnalysis_embeddings_local.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
EXCERPT_LEN = 2000  # how much of each chunk to feed the LLM

# --- Load embeddings ---
data = json.loads(Path(EMB_FILE).read_text(encoding="utf-8"))
texts = [d["text"] for d in data]
filenames = [d["chunk_file"] for d in data]
embs = np.array([d["embedding"] for d in data], dtype=np.float32)
embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

# --- Load model for embedding queries ---
model = SentenceTransformer(MODEL_NAME)


def clean_text(raw):
    text = raw.replace("(BUTTON)", "")
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    text = re.sub(r"\d{2,}", "", text)          # remove long numbers/IDs
    return text.strip()


def embed_query(q: str):
    v = model.encode(q)
    v = np.array(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def retrieve(q: str, k=TOP_K):
    qv = embed_query(q)
    sims = embs_norm @ qv
    idx = np.argsort(-sims)[:k]
    results = [(filenames[i], sims[i], texts[i]) for i in idx]
    return results

def ask_llm(context: str, question: str):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Write prompt to temporary file
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(prompt)
        tmp_path = Path(f.name)
    
    result = subprocess.run(
        ["ollama", "run", "llama3.1", "--prompt-file", str(tmp_path)],
        capture_output=True
    )
    
    tmp_path.unlink()
    return result.stdout.decode("utf-8").strip()


if __name__ == "__main__":
    print("Mini-RAG Chatbot (Ctrl+C to quit)")
    while True:
        query = input("\nYour question> ").strip()
        if not query:
            continue
        
        retrieved = retrieve(query)  # top 3 chunks, for example
        answers = []

        for _, _, txt in retrieved:
            context = clean_text(txt[:2000])  # clean + limit length
            ans = ask_llm(context, query)
            answers.append(ans)

        # Combine or pick the best answer
        final_answer = "\n---\n".join([a for a in answers if a])
        print(final_answer)

