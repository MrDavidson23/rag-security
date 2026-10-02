import pathlib
import chromadb

client = chromadb.PersistentClient(path="db")
col = client.get_or_create_collection("security_docs")

def chunk(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

for f in pathlib.Path("docs").glob("*"):
    if f.suffix not in {".md", ".txt"}:
        continue
    pieces = chunk(f.read_text(encoding="utf-8"))
    col.add(
        ids=[f"{f.name}-{i}" for i in range(len(pieces))],
        documents=pieces,
        metadatas=[{"source": f.name}] * len(pieces),
    )
    print(f"{f.name}: {len(pieces)} chunks")