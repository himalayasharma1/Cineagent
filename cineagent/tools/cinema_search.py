"""
Tool 1: search_cinema_knowledge

Wraps the local ChromaDB built by CineQuery. Given a natural-language query,
returns the top-k most relevant chunks from the Wikipedia world cinema corpus.

This is a pure-local tool — no network calls, no API keys.
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
# Path to the ChromaDB built by CineQuery's build_index.py.
# Resolved relative to the project root so this works regardless
# of where Python is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "cinema_knowledge"  # must match build_index.py

# Embedding model — must match what was used to build the index.
# all-MiniLM-L6-v2, 384 dimensions, same as CineQuery.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval defaults
DEFAULT_K = 5

# Similarity threshold — chunks with cosine *distance* above this are dropped.
# (ChromaDB returns distance, not similarity. Lower distance = more similar.)
# 0.7 matches CineQuery's existing guardrail.
SIMILARITY_THRESHOLD = 0.7

# ---------------------------------------------------------------
# One-time setup: load embedder and connect to ChromaDB.
# These are module-level so they're loaded once per process,
# not once per tool call. Important for performance.
# ---------------------------------------------------------------
_embedder = None
_collection = None


def _get_embedder():
    """Lazy-load the embedding model on first call."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    """Lazy-load the ChromaDB collection on first call."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------
def search_cinema_knowledge(query: str, k: int = DEFAULT_K) -> dict:
    """
    Search the local Wikipedia world cinema corpus for chunks relevant to a query.

    Args:
        query: Natural-language question or topic to search for.
        k: Number of top results to return (default 5).

    Returns:
        A dict with one of two shapes:

        On success:
        {
            "status": "ok",
            "query": <the original query>,
            "results": [
                {
                    "text": <chunk text>,
                    "source": <source filename>,
                    "chunk_index": <position within source>,
                    "distance": <cosine distance, lower = more similar>
                },
                ...
            ]
        }

        On no results above threshold:
        {
            "status": "no_results",
            "query": <the original query>,
            "message": "No chunks above similarity threshold."
        }
    """
    # Validate input
    if not query or not query.strip():
        return {
            "status": "error",
            "query": query,
            "message": "Query is empty.",
        }

    embedder = _get_embedder()
    collection = _get_collection()

    # Embed the query
    query_vec = embedder.encode([query]).tolist()

    # Retrieve from ChromaDB
    raw = collection.query(
        query_embeddings=query_vec,
        n_results=k,
    )

    # ChromaDB returns parallel arrays. Unpack the first (and only) query.
    docs = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # Apply the similarity threshold guardrail
    filtered = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        if dist <= SIMILARITY_THRESHOLD:
            filtered.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": round(dist, 4),
            })

    if not filtered:
        return {
            "status": "no_results",
            "query": query,
            "message": (
                f"No chunks found within similarity threshold "
                f"(distance ≤ {SIMILARITY_THRESHOLD})."
            ),
        }

    return {
        "status": "ok",
        "query": query,
        "results": filtered,
    }