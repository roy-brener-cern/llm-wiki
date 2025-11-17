# query_embeddings_local.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_FILE = "MuonPerformanceAnalysis_embeddings_local.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
EXCERPT_LEN = 300

# Load embeddings
data = json.loads(Path(EMB_FILE).read_text(encoding="utf-8"))
texts = [d["text"] for d in data]
filenames = [d["chunk_file"] for d in data]
embs = np.array([d["embedding"] for d in data], dtype=np.float32)

# normalize stored embeddings
embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

# load model
model = SentenceTransformer(MODEL_NAME)

def embed_query(q: str):
    v = model.encode(q)
    v = np.array(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def search(q: str, k=TOP_K):
    qv = embed_query(q)
    sims = (embs_norm @ qv)  # cosine via dot of normalized vectors
    idx = np.argsort(-sims)[:k]
    results = [(filenames[i], sims[i], texts[i]) for i in idx]
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query> ").strip()
    if not query:
        print("No query provided. Exiting.")
        raise SystemExit(0)

    results = search(query)
    for rank, (fname, score, txt) in enumerate(results, start=1):
        print(f"\n--- Rank {rank} | score={score:.4f} | file={fname} ---")
        excerpt = txt.replace("\n", " ").strip()[:EXCERPT_LEN]
        print(excerpt + ("..." if len(txt) > EXCERPT_LEN else ""))

