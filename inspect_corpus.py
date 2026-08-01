"""
Inspect what's actually in the ChromaDB collection.
Ground truth for what the eval can reference.
"""

import chromadb
from pathlib import Path
from collections import Counter

CHROMA_PATH = Path("./data/chroma_db")
COLLECTION_NAME = "cinema_knowledge"  # adjust if your build_index used a different name

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Total chunks in collection: {collection.count()}\n")

# Pull all metadata to see what sources exist
all_data = collection.get(include=["metadatas"])
sources = [m.get("source", "unknown") for m in all_data["metadatas"]]

source_counts = Counter(sources)
print(f"Unique sources: {len(source_counts)}\n")
print("Chunks per source:")
for source, count in sorted(source_counts.items()):
    print(f"  {count:3d}  {source}")