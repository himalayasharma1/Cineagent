markdown# CineAgent

An agentic AI for world cinema knowledge, built on the **CineQuery** retrieval foundation. CineAgent extends a local-first RAG system into a tool-using agent that reasons about films and directors, and pulls real-time data like current film metadata and streaming availability.

> **Status:** Active development. The CineQuery RAG foundation and all three agent tools (local retrieval, TMDB film details, streaming availability) are complete and covered by a multi-tool evaluation harness (25/26, with one documented, intentional failure). The agent reasoning loop that orchestrates these tools is next — see the [Roadmap](#roadmap). This README tracks the *actual* state of the code and is updated as each phase ships.

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
Query → Embed → Retrieve top-k (ChromaDB) → Threshold filter → Build prompt → LLM → Answer

### CineAgent — a reasoning loop with tool use

Instead of a fixed pipeline, a local LLM sits at the centre of a loop and *decides* what to do at each step: call a tool, observe the result, and either continue or answer. Variable model calls per query, branching flow.
Query → LLM reasons → Need a tool? ──no──→ Answer
│ yes
▼
Pick + call a tool ──→ Observe result ──→ Enough info? ──no──→ (loop back)
│ yes
▼
Final answer

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

### CineAgent Tool 2 — `get_film_details` (TMDB) — complete, eval-covered
- Fetches current film metadata from TMDB: director, cast, runtime, genres, rating, plot
- Curated 9-field response shaping (reduces TMDB's 50+ raw fields to keep agent context lean)
- Disambiguation via an optional `year` parameter plus a surfaced list of alternative matches
- Graceful failure handling: network timeouts and API errors return structured errors rather than crashing the agent
- Complements Tool 1 by design: fetches current, real-world data the local corpus structurally cannot contain

### CineAgent Tool 3 — `streaming_lookup` (TMDB watch-providers) — complete, eval-covered
- Finds where a film can be streamed, rented, or bought, by country
- Sourced from TMDB's watch-providers endpoint (JustWatch data) — reuses the existing TMDB key, avoiding a second vendor, secret, or scraping dependency
- Country-aware: availability is regional, so country is a first-class parameter (defaults to India)
- Four-state contract: distinguishes "available," "film exists but streams nowhere here" (`no_availability`), "film not found" (`no_results`), and error — so the agent can phrase each situation differently
- Graceful network-failure handling, consistent with Tool 2

### Local LLM inference — verified
- **Qwen3-4B-Instruct-2507** (Q4_K_M GGUF) via `llama.cpp` with Metal GPU acceleration
- Tool-call emission verified (`<tool_call>` format) with a custom parser — chosen over library auto-parsing for explicit control

### Evaluation harness — complete, multi-tool
The measurement substrate, built *before* the agent it will eventually score. An assertion-driven runner over hand-crafted test cases, grouped by category, reporting per-category pass rates with named, documented failures. The runner dispatches each case to the tool it targets, so a single run scores all three tools together.

A representative finding from development: the harness surfaced that the retrieval guardrail passed two famous directors *not* in the corpus. Pulling the actual embedding distances revealed two distinct failure modes — one a near-miss fixable by an evidence-based threshold change (0.7 → 0.62, verified against the full suite to confirm no recall regression), and one that landed inside the in-domain range because the query semantically matched a *concept* present in the corpus. The latter was documented as an expected tool-layer limitation and deferred to the agent layer, rather than hidden. This is the intended workflow: **measure → diagnose with data → fix with evidence → verify no regression.**

The harness also demonstrates knowing *when to deviate from a default*: the tools are evaluated against live APIs, but the one state that can't be reliably triggered against live data — a film that exists but streams nowhere in a given country — is verified with a deterministic probe instead, because a flaky test is worse than a targeted one.

Current suite: **26 cases across three tools and 13 categories, 25 passing**, with the single remaining failure being the documented, intentional one described above.

---

## Roadmap

Planned and in progress. Clearly *not yet built* — this section shrinks as features move up into [What's built](#whats-built).

- [ ] **Hand-rolled ReAct loop** (Version A): the reasoning loop that orchestrates the three tools, built from scratch for full control and understanding
- [ ] **Guardrails layer**: iteration cap, loop detection, tool-error recovery
- [ ] **Observability**: per-iteration tracing of the agent's Thought → Action → Observation cycle
- [ ] **LangGraph rebuild** (Version B): the same agent on a framework, as a deliberate build-vs-framework comparison

Each capability follows a **"done = evaluated"** rule: it is not considered complete until its evaluation cases are written and passing.

---

## Tech stack

| Layer | Choice |
|---|---|
| Local LLM | Qwen3-4B-Instruct-2507 (Q4_K_M GGUF) |
| Inference runtime | llama.cpp (via `llama-cpp-python`), Metal-accelerated |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent) |
| External data | TMDB API (film metadata + watch-providers / JustWatch streaming data) |
| Language | Python 3.11 |

---

## Repository layout
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
│   ├── cinema_search.py     # Tool 1 — search_cinema_knowledge
│   ├── film_details.py      # Tool 2 — get_film_details (TMDB)
│   └── streaming_lookup.py  # Tool 3 — streaming_lookup (TMDB watch-providers)
└── tests/
├── eval_cases.py     # Evaluation cases (multi-tool)
└── run_eval.py       # Assertion-driven, tool-dispatching eval runner

*Model files, the vector store, the virtual environment, and secrets (`.env`) are gitignored — the corpus and index are reproducible from the fetch and build scripts.*

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

# Run the evaluation harness across all three tools
python -m cineagent.tests.run_eval
```

A local GGUF model is required for inference, and a `.env` file with a `TMDB_API_KEY` is required for the film-details and streaming tools:
TMDB_API_KEY=your_key_here

The full agent loop (in progress) will document any additional configuration as those components land.

---

## Design principles

- **Local-first** — inference and retrieval run on-device; external calls are added only per-capability where the data genuinely requires it (current film metadata, real-time streaming availability), not by default.
- **Measurement before features** — the evaluation harness was built before the agent it scores. No tool is "done" without eval coverage.
- **Hand-rolled where understanding matters** — the agent loop is built from scratch before any framework comparison, so every decision is explicable rather than delegated.
- **Honest about limitations** — known failures are documented in the eval, not hidden.

---

*This project is in active development. The README reflects the current, verified state of the code and is updated as each phase ships.*