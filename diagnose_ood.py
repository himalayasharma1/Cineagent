"""
Diagnose the OOD failures: what distances did the absent-director
queries actually return, and from which sources?
"""

from cineagent.tools.cinema_search import search_cinema_knowledge

# Temporarily bypass the threshold by calling with a high k and
# inspecting raw distances. We import the internals directly.
from cineagent.tools.cinema_search import _get_embedder, _get_collection

queries = [
    ("Bong Joon-ho and Parasite", "OOD — should be absent"),
    ("Greta Gerwig directing style", "OOD — should be absent"),
    ("how to change a car tyre", "OOD — correctly rejected"),
    ("Christopher Nolan nonlinear storytelling", "in-domain — for comparison"),
]

embedder = _get_embedder()
collection = _get_collection()

for query, label in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: {query!r}")
    print(f"({label})")
    print("-" * 60)

    vec = embedder.encode([query]).tolist()
    raw = collection.query(query_embeddings=vec, n_results=5)

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    for i, (meta, dist) in enumerate(zip(metas, dists), 1):
        src = meta.get("source", "unknown")
        flag = "  <-- under 0.7 threshold" if dist <= 0.7 else ""
        print(f"  {i}. dist={dist:.4f}  {src}{flag}")