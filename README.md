# CineAgent

An agentic AI for world cinema knowledge, built on the **CineQuery** retrieval foundation. CineAgent extends a local-first RAG system into a tool-using agent that can reason about films, directors, and — as tools come online — real-time data like streaming availability and current film metadata.

> **Status:** Active development. The CineQuery RAG foundation and the first agent tool are complete and covered by an evaluation harness. The agent reasoning loop and remaining tools are in progress — see the [Roadmap](#roadmap). This README tracks the *actual* state of the code and is updated as each phase ships.

---

## Why this project exists

Most RAG demos stop at "embed, retrieve, generate." CineAgent is an attempt to go one layer deeper — into agentic tool use — while keeping the engineering honest: local-first inference, hand-rolled where understanding matters, and **measurement before features**. Every capability ships with evaluation coverage before it's considered done.

The project is deliberately built in two stages:

1. **CineQuery** — a single-shot RAG pipeline over a curated world-cinema corpus. The foundation.
2. **CineAgent** — a tool-using agent built on top, where a local LLM decides which tool to call, observes the result, and reasons toward an answer.

---

## Architecture

### CineQuery (the foundation) — a fixed RAG pipeline

A query is embedded, matched against a local vector store, filtered by a similarity threshold, and passed as context to a local LLM. One model call per query, deterministic flow.

```
Query → Embed → Retrieve top-k (ChromaDB) → Threshold filter → Build prompt → LLM → Answer
```

### CineAgent — a reasoning loop with tool use

Instead of a fixed pipeline, a local LLM sits at the centre of a loop and *decides* what to do at each step: call a tool, observe the result, and either continue or answer. Variable model calls per query, branching flow.

```
Query → LLM reasons → Need a tool? ──no──→ Answer
                          │ yes
                          ▼
                  Pick + call a tool ──→ Observe result ──→ Enough info? ──no──→ (loop back)
                                                                 │ yes
                                                                 ▼
                                                             Final answer
```

The architectural shift from CineQuery to CineAgent is from a **fixed assembly line** to a **reasoning loop with a decision fork** — which is what makes it an agent rather than a pipeline, and also what expands the failure surface (tool selection, termination, error recovery) that the evaluation harness exists to measure.

---

## The corpus

A curated world-cinema knowledge base of **2,485 chunks across 90 sources**, spanning two sub-corpora:

- **Wikipedia articles** — director and actor profiles across Hollywood, Indian, and international cinema (Nolan, Scorsese, Kurosawa, Satyajit Ray, Shah Rukh Khan, Guru Dutt, and more).
- **Senses of Cinema essays** — critical, essayistic writing on art-house and auteur directors (Akerman, Antonioni, Almodóvar, Wong Kar-wai, Pasolini, and more).

The two sub-corpora have different textual character — factual vs. critical — which makes retrieval behaviour richer to evaluate, including a set of figures (e.g. Guru Dutt, Mrinal Sen) present in *both*.

---

## What's built

### CineQuery — complete
- Local-first RAG over the 2,485-chunk corpus
- ChromaDB persistent vector store
- `all-MiniLM-L6-v2` embeddings (384-dim)
- Similarity-threshold guardrail for out-of-domain rejection

### CineAgent Tool 1 — `search_cinema_knowledge` — complete, eval-covered
- Self-contained retrieval tool over the local corpus (independent of CineQuery's own code)
- Returns structured results with source attribution and distances
- Enforces the similarity-threshold guardrail at the tool layer
- Distinguishes three states: `ok`, `no_results`, `error`

### Local LLM inference — verified
- **Qwen3-4B-Instruct-2507** (Q4_K_M GGUF) via `llama.cpp` with Metal GPU acceleration
- Tool-call emission verified (`<tool_call>` format) with a custom parser — chosen over library auto-parsing for explicit control

### Evaluation harness — complete
The measurement substrate, built *before* the agent it will eventually score. An assertion-driven runner over hand-crafted test cases, grouped by category, reporting per-category pass rates with named, documented failures.

A representative finding from development: the harness surfaced that the retrieval guardrail passed two famous directors *not* in the corpus. Pulling the actual embedding distances revealed two distinct failure modes — one a near-miss fixable by an evidence-based threshold change (0.7 → 0.62, verified against the full suite to confirm no recall regression), and one that landed inside the in-domain range because the query semantically matched a *concept* present in the corpus. The latter was documented as an expected tool-layer limitation and deferred to the agent layer, rather than hidden. This is the intended workflow: **measure → diagnose with data → fix with evidence → verify no regression.**

---

## Roadmap

Planned and in progress. Clearly *not yet built* — this section shrinks as features move up into [What's built](#whats-built).

- [ ] **Tool 2 — `get_film_details`** (TMDB): current film metadata — cast, runtime, ratings, plot — with API error handling and disambiguation
- [ ] **Tool 3 — `streaming_lookup`**: real-time streaming availability
- [ ] **Hand-rolled ReAct loop** (Version A): the reasoning loop, built from scratch for full control and understanding
- [ ] **Guardrails layer**: iteration cap, loop detection, tool-error recovery
- [ ] **Observability**: per-iteration tracing of the agent's Thought → Action → Observation cycle
- [ ] **LangGraph rebuild** (Version B): the same agent on a framework, as a deliberate build-vs-framework comparison

Each tool follows a **"done = evaluated"** rule: it is not considered complete until its evaluation cases are written and passing.

---

## Tech stack

| Layer | Choice |
|---|---|
| Local LLM | Qwen3-4B-Instruct-2507 (Q4_K_M GGUF) |
| Inference runtime | llama.cpp (via `llama-cpp-python`), Metal-accelerated |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent) |
| Language | Python 3.11 |

---

## Repository layout

```
cinequery/
├── build_index.py            # Builds the ChromaDB index from the corpus
├── rag_pipeline.py           # CineQuery RAG pipeline
├── app.py                    # CineQuery interface
├── fetch_corpus.py           # Wikipedia corpus fetch
├── fetch_web_corpus.py       # Senses of Cinema corpus fetch
├── data/
│   ├── raw/                  # Source corpus (Wikipedia + Senses of Cinema)
│   └── chroma_db/            # Persistent vector store (gitignored)
└── cineagent/
    ├── tools/
    │   └── cinema_search.py  # Tool 1 — search_cinema_knowledge
    └── tests/
        ├── eval_cases.py     # Evaluation cases
        └── run_eval.py       # Assertion-driven eval runner
```

*Model files, the vector store, the virtual environment, and secrets are gitignored — the corpus and index are reproducible from the fetch and build scripts.*

---

## Running it

> Setup notes are being expanded as the project stabilises.

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the index (regenerates the ChromaDB store from the corpus)
python build_index.py

# Run the evaluation harness against the retrieval tool
python -m cineagent.tests.run_eval
```

A local GGUF model and a `.env` file (for forthcoming API-based tools) are required for the full agent; details will be documented as those components land.

---

## Design principles

- **Local-first** — inference and retrieval run on-device; external calls are added only per-capability where the data genuinely requires it (e.g. real-time streaming availability), not by default.
- **Measurement before features** — the evaluation harness was built before the agent it scores. No tool is "done" without eval coverage.
- **Hand-rolled where understanding matters** — the agent loop is built from scratch before any framework comparison, so every decision is explicable rather than delegated.
- **Honest about limitations** — known failures are documented in the eval, not hidden.

---

*This project is in active development. The README reflects the current, verified state of the code and is updated as each phase ships.*
