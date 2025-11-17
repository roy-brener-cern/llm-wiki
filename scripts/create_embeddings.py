# scripts/create_embeddings.py
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 2000  # max characters per chunk

# Paths relative to project root
BASE_DIR = Path(__file__).parent.parent  # parent of scripts/
PAGE_FOLDER = BASE_DIR / "docs/pages"    # where HTML pages are stored
OUTPUT_FILE = BASE_DIR / "docs/embeddings/EGamma_pages_embeddings_local.json"

# Make sure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load model
model = SentenceTransformer(MODEL)

# -------------------------
# PROCESS PAGES
# -------------------------
embeddings_data = []

html_files = sorted(PAGE_FOLDER.glob("*.html"))
for page_file in tqdm(html_files, desc="Processing pages"):
    text = page_file.read_text(encoding="utf-8")
    
    # split text into chunks
    for i in range(0, len(text), CHUNK_SIZE):
        chunk_text = text[i:i+CHUNK_SIZE]
        embedding = model.encode(chunk_text).tolist()
        embeddings_data.append({
            "chunk_file": str(page_file),
            "text": chunk_text,
            "embedding": embedding
        })
    print(f"Processed {page_file.name}")

# -------------------------
# SAVE EMBEDDINGS
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f)

print(f"Created embeddings for {len(embeddings_data)} chunks in {OUTPUT_FILE}")
