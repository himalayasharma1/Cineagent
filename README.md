markdown# CineAgent

An agentic AI for world cinema knowledge, built on the **CineQuery** retrieval foundation. CineAgent extends a local-first RAG system into a tool-using agent that reasons about films and directors, and pulls real-time data like current film metadata and streaming availability.

> **Status:** Active development. The CineQuery RAG foundation, all three agent tools, and a working hand-rolled ReAct loop are complete and stress-tested, with boundaries documented in [Known limitations & scope](#known-limitations--scope). The tools are covered by a multi-tool evaluation harness (26/27, one documented intentional failure). Next: agent-level evaluation and observability — see the Roadmap.

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

### CineAgent ReAct loop (Version A, hand-rolled) — working
- A reasoning loop that orchestrates the three tools: the model reasons, calls one tool at a time, observes the result, and repeats until it can answer.
- Four hand-built components: a system prompt defining the protocol, a parser that extracts tool calls from the model's output, a dispatcher that safely routes calls to the real tools, and a loop executor that ties them together.
- Three stop conditions: natural termination (model answers), a hard iteration cap, and no-progress detection (identical repeated tool call).
- Every malformed input — bad JSON, unknown tool, missing argument — becomes recoverable feedback the model can correct, never a crash.
- Returns a full trace of every reasoning step, tool call, and observation, so the loop's decisions are legible rather than inferred.
- Runs entirely on local inference (Qwen3-4B via llama.cpp); external calls happen only inside the tools.

### Local LLM inference — verified
- **Qwen3-4B-Instruct-2507** (Q4_K_M GGUF) via `llama.cpp` with Metal GPU acceleration
- Tool-call emission verified (`<tool_call>` format) with a custom parser — chosen over library auto-parsing for explicit control

### Evaluation harness — complete, multi-tool
The measurement substrate, built *before* the agent it will eventually score. An assertion-driven runner over hand-crafted test cases, grouped by category, reporting per-category pass rates with named, documented failures. The runner dispatches each case to the tool it targets, so a single run scores all three tools together.

A representative finding from development: the harness surfaced that the retrieval guardrail passed two famous directors *not* in the corpus. Pulling the actual embedding distances revealed two distinct failure modes — one a near-miss fixable by an evidence-based threshold change (0.7 → 0.62, verified against the full suite to confirm no recall regression), and one that landed inside the in-domain range because the query semantically matched a *concept* present in the corpus. The latter was documented as an expected tool-layer limitation and deferred to the agent layer, rather than hidden. This is the intended workflow: **measure → diagnose with data → fix with evidence → verify no regression.**

The harness also demonstrates knowing *when to deviate from a default*: the tools are evaluated against live APIs, but the one state that can't be reliably triggered against live data — a film that exists but streams nowhere in a given country — is verified with a deterministic probe instead, because a flaky test is worse than a targeted one.

Current suite: **26 cases across three tools and 13 categories, 25 passing**, with the single remaining failure being the documented, intentional one described above. Agent-level evaluation (tool-selection accuracy, termination behaviour) is the next phase.

---

## Roadmap

Planned and in progress. Clearly *not yet built* — this section shrinks as features move up into [What's built](#whats-built).

- [ ] **Agent-level evaluation**: extend the harness to score the loop itself — tool-selection accuracy, termination behaviour, and relevance judgment on out-of-corpus queries
- [ ] **Guardrails layer**: formalize iteration cap, loop detection, and tool-error recovery as a named layer
- [ ] **Observability**: per-iteration tracing of the agent's Thought → Action → Observation cycle
- [ ] **LangGraph rebuild** (Version B): the same agent on a framework, as a deliberate build-vs-framework comparison

Each capability follows a **"done = evaluated"** rule: it is not considered complete until its evaluation cases are written and passing.

- [ ] **Tool 4 — director filmography** (candidate): a `get_director_filmography` tool (TMDB `/person/{id}/movie_credits`) to answer "latest/newest film by X" questions with current data, closing the confabulation gap found in testing
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
├── agent/
│   ├── dispatcher.py     # Tool schemas + safe tool dispatch
│   ├── parser.py         # Extracts tool calls / reasoning from model output
│   ├── prompt.py         # Builds the ReAct system prompt
│   └── loop.py           # The ReAct loop executor
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

---

## Design principles

- **Local-first** — inference and retrieval run on-device; external calls are added only per-capability where the data genuinely requires it (current film metadata, real-time streaming availability), not by default.
- **Measurement before features** — the evaluation harness was built before the agent it scores. No tool is "done" without eval coverage.
- **Hand-rolled where understanding matters** — the agent loop is built from scratch before any framework comparison, so every decision is explicable rather than delegated.
- **Honest about limitations** — known failures are documented in the eval, not hidden.

---

*This project is in active development. The README reflects the current, verified state of the code and is updated as each phase ships.*


## Known limitations & scope

These are deliberate boundaries of the current agent, documented rather than hidden:

- **Per-country streaming only.** `streaming_lookup` answers "is this film available in country X." It does not answer "which countries is this available in" globally. Asked a global-availability question, the agent answers for the default country and flags that the result is country-specific. TMDB's data would support an all-countries mode; it's deliberately out of scope for a "help me decide what to watch here" use case rather than built speculatively.
- **Out-of-corpus analysis (documented in the eval).** For analysis questions about people not in the local corpus, retrieval can return topically-similar-but-wrong results. The agent is prompted to detect this and decline rather than fabricate — verified in testing (e.g. it correctly refuses to invent an answer about a director absent from the corpus). This is the agent-layer half of a deliberate layered defense; the tool-layer half is the similarity threshold.
- **Literal (not semantic) loop detection.** The loop stops if the agent makes the *identical* tool call twice in a row. It does not yet detect *semantic* repetition — reformulating the same unanswerable query with different wording and getting substantially the same results. In testing, a question whose answer wasn't in the corpus produced three near-identical searches before the agent correctly gave up. It reaches the right answer (an honest "I don't have that") and terminates naturally, but less efficiently than ideal. Detecting substantially-repeated *results* (not just identical *calls*) is a planned improvement.
- **No current-filmography lookup — "latest/newest" questions can be stale.** The agent has no tool that lists a director's films by date, and the local corpus is frozen at scrape time. Asked for someone's "latest" film, the model may answer from its own training data rather than verifying — and because the tool it *does* call confirms correct details about that (stale) guess, the answer looks fully sourced while the underlying "this is the latest" premise goes unchecked. This is the most subtle failure mode found in testing: dangerous precisely because every visible fact is correct and only the unstated premise is wrong. Fixing it requires a director-filmography tool with current data (TMDB supports this) plus prompt work to treat currency questions as requiring premise verification, not just detail confirmation.
- **Answers are bounded by tool + corpus coverage.** More generally, the agent is only as current and complete as its three tools and static corpus. It's built to *decline* when retrieved results are visibly irrelevant (verified), but it cannot detect when a tool confirms a subtly wrong premise. Questions requiring capabilities the tools don't have (global availability, current filmographies) are out of scope by design rather than answered speculatively.