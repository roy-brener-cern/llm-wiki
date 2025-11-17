# split_text.py
from pathlib import Path

def split_text(file_path, chunk_size=1000, overlap=200):
    text = Path(file_path).read_text(encoding="utf-8")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap  # overlap between chunks
    return chunks

chunks = split_text("MuonPerformanceAnalysis.txt")

# Save each chunk to a separate file
for idx, chunk in enumerate(chunks):
    Path(f"MuonPerformanceAnalysis_chunk{idx+1}.txt").write_text(chunk, encoding="utf-8")

print(f"Created {len(chunks)} chunks.")

