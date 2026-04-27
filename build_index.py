# build_index.py — reads from data/raw/ AND data/raw/web/
import os
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
RAW_DATA_DIR = "data/raw"
WEB_DATA_DIR = "data/raw/web"
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "cinema_knowledge"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("  ✓ Embedder ready\n")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
try:
    client.delete_collection(COLLECTION_NAME)
    print("  ℹ Deleted existing collection")
except:
    pass
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print("  ✓ Collection created\n")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ── Collect ALL files from both directories ───────────────────────────────────

all_files = []

# Wikipedia files in data/raw/
for f in os.listdir(RAW_DATA_DIR):
    if f.endswith(".txt"):
        all_files.append((RAW_DATA_DIR, f))

# Web corpus files in data/raw/web/
if os.path.exists(WEB_DATA_DIR):
    for f in os.listdir(WEB_DATA_DIR):
        if f.endswith(".txt"):
            all_files.append((WEB_DATA_DIR, f))

wiki_count = len([f for d, f in all_files if d == RAW_DATA_DIR])
web_count = len([f for d, f in all_files if d == WEB_DATA_DIR])

print(f"Wikipedia files: {wiki_count}")
print(f"Web corpus files: {web_count}")
print(f"Total files: {len(all_files)}\n")
print("Processing and chunking...\n")

all_chunks = []
all_ids = []
all_metadata = []

for dirpath, filename in sorted(all_files):
    filepath = os.path.join(dirpath, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    category = "unknown"
    subject = filename.replace(".txt", "")
    source_type = "web" if dirpath == WEB_DATA_DIR else "wikipedia"

    for line in text.split("\n")[:6]:
        if line.startswith("CATEGORY:"):
            category = line.replace("CATEGORY:", "").strip().lower()
        if line.startswith("SUBJECT:") or line.startswith("TITLE:"):
            subject = line.split(":", 1)[1].strip()

    chunks = chunk_text(text)
    print(f"  [{source_type}] {filename[:50]} → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{filename}__chunk_{i:04d}")
        all_metadata.append({
            "source": filename,
            "subject": subject,
            "category": category,
            "source_type": source_type,
            "chunk_index": i,
        })

print(f"\nTotal chunks to embed: {len(all_chunks)}")
print("\nEmbedding and storing in ChromaDB...\n")

BATCH_SIZE = 50
for batch_start in range(0, len(all_chunks), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(all_chunks))
    batch_embeddings = embedder.encode(all_chunks[batch_start:batch_end]).tolist()
    collection.add(
        documents=all_chunks[batch_start:batch_end],
        embeddings=batch_embeddings,
        ids=all_ids[batch_start:batch_end],
        metadatas=all_metadata[batch_start:batch_end],
    )
    print(f"  Stored chunks {batch_start}–{batch_end} of {len(all_chunks)}")

print("\n=== Build Complete ===\n")
print(f"✓ Total chunks in ChromaDB: {collection.count()}")
print(f"✓ Wikipedia chunks: {wiki_count} files")
print(f"✓ Web corpus chunks: {web_count} files")

print("\nSanity check: 'directorial style and visual language'...\n")
test_vec = embedder.encode(["directorial style and visual language"]).tolist()
results = collection.query(query_embeddings=test_vec, n_results=3)
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"Result {i+1}: [{meta['subject']}] [{meta['source_type']}]")
    print(f"  {doc[:150]}...")
    print()