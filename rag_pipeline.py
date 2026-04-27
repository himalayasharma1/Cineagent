# rag_pipeline.py
# The core CineQuery brain — retrieves relevant chunks, feeds to Mistral, returns answer

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "cinema_knowledge"
MODEL_PATH = "models/gemma-3-4b-instruct-q4.gguf"

N_RESULTS = 4          # how many chunks to retrieve (your PM decision)
TEMPERATURE = 0.3      # low = factual and consistent (your PM decision)
MAX_TOKENS = 512       # maximum length of the generated answer
N_CTX = 8192   # Gemma can handle much more — give it room

# ── Load models (done once, reused for every query) ───────────────────────────

print("Loading embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("  ✓ Embedder ready")

print("Loading Gemma 3 4B (this takes ~10 seconds)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=-1,   # all layers on Metal GPU
    verbose=False
)
print("  ✓ Mistral ready")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"  ✓ ChromaDB ready ({collection.count()} chunks)\n")

# ── The retriever ─────────────────────────────────────────────────────────────

def retrieve(question, n_results=N_RESULTS):
    """
    Converts the question to a vector, finds the closest chunks in ChromaDB.
    Returns the chunk texts and their metadata (source file, subject).
    """
    # Embed the question using the same model that embedded the corpus
    # CRITICAL: you must use the same embedder for both indexing and querying
    # Using a different embedder would be like translating to French then
    # searching a Spanish dictionary — the vectors live in different spaces
    question_vector = embedder.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=question_vector,
        n_results=n_results,
    )
    
    chunks = results["documents"][0]      # list of chunk texts
    metadatas = results["metadatas"][0]   # list of metadata dicts
    distances = results["distances"][0]   # similarity scores (lower = more similar)
    
    return chunks, metadatas, distances


# ── The prompt builder ────────────────────────────────────────────────────────

def build_prompt(question, chunks, metadatas):
    """
    Assembles retrieved chunks into a structured prompt for Mistral.
    
    The prompt structure is critical — it tells Mistral:
    1. What role to play (cinema expert)
    2. What knowledge to use (only the retrieved chunks)
    3. What to answer (the user's question)
    4. How to behave if it doesn't know (say so honestly)
    """
    
    # Build the context block from retrieved chunks
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        source_label = f"[Source {i+1}: {meta['subject']}]"
        context_parts.append(f"{source_label}\n{chunk}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Mistral instruct format: [INST] ... [/INST]
    prompt = f"""<start_of_turn>user
You are CineQuery, an expert cinema assistant with deep knowledge of world cinema, directors, and films.

Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information about that in my knowledge base" — do not make things up.
Where relevant, mention specific films or directors from the context to support your answer.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, helpful answer in 2-4 sentences.
<end_of_turn>
<start_of_turn>model
"""
    
    return prompt


# ── The generator ─────────────────────────────────────────────────────────────

def generate(prompt):
    """
    Passes the assembled prompt to Mistral and returns the generated answer.
    """
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["<end_of_turn>"],    # stop if it tries to generate a new question
    )
    
    answer = response["choices"][0]["text"].strip()
    tokens_used = response["usage"]["completion_tokens"]
    
    return answer, tokens_used


# ── The full RAG pipeline ─────────────────────────────────────────────────────

# Minimum similarity score to trust retrieved chunks
# Cosine distance in ChromaDB: lower = more similar (0 = identical, 2 = opposite)
# 0.7 distance = 0.3 similarity — below this we don't trust the context
SIMILARITY_THRESHOLD = 0.7

# Keywords that signal out-of-domain questions
OUT_OF_DOMAIN_TRIGGERS = [
    "weather", "temperature", "forecast", "stock", "price",
    "news", "sports score", "recipe", "math", "calculate",
    "translate", "time", "date", "location", "map"
]

def ask(question, verbose=False):
    """
    Full RAG pipeline with guardrails:
    1. Topic scope check — reject obviously out-of-domain questions
    2. Similarity threshold — reject if retrieved chunks aren't relevant enough
    3. Retrieve → Prompt → Generate
    """

    # ── Guardrail 1: Topic scope check ────────────────────────────────────────
    question_lower = question.lower()
    for trigger in OUT_OF_DOMAIN_TRIGGERS:
        if trigger in question_lower:
            if verbose:
                print(f"  ⚠ Out-of-domain trigger detected: '{trigger}'")
            return (
                "I'm CineQuery — I can only answer questions about cinema, "
                "directors, films, and actors. That question seems outside my domain.",
                [],
                []
            )

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    chunks, metadatas, distances = retrieve(question)

    if verbose:
        print(f"\n--- Retrieved {len(chunks)} chunks ---")
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            print(f"  {i+1}. [{meta['subject']}] distance: {dist:.3f}")

    # ── Guardrail 2: Similarity threshold ─────────────────────────────────────
    # ChromaDB returns cosine DISTANCE not similarity
    # Distance 0 = identical, Distance 2 = opposite
    # We reject if even the best chunk has distance > threshold
    best_distance = min(distances)

    if best_distance > SIMILARITY_THRESHOLD:
        if verbose:
            print(f"  ⚠ Best distance {best_distance:.3f} exceeds threshold {SIMILARITY_THRESHOLD}")
        return (
            "I don't have reliable information about that in my knowledge base. "
            "Try asking about a specific director, film, actor, or cinematic style.",
            chunks,
            metadatas
        )

    # ── Step 2: Build prompt ──────────────────────────────────────────────────
    prompt = build_prompt(question, chunks, metadatas)

    if verbose:
        print(f"\n--- Prompt length: {len(prompt.split())} words ---")

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    answer, tokens = generate(prompt)

    if verbose:
        print(f"--- Tokens generated: {tokens} ---\n")

    return answer, chunks, metadatas


# ── Test it directly ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    test_questions = [
        "How does Scorsese use music in his films?",
        "What should I watch if I love mind-bending narratives?",
        "Tell me about Irrfan Khan's acting style",
        "What are the themes in Satyajit Ray's work?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        
        answer, chunks, metadatas = ask(question, verbose=True)
        
        print(f"\nA: {answer}")
        print(f"\nSources used: {[m['subject'] for m in metadatas]}")