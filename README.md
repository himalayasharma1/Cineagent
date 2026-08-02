# CineAgent

An agentic AI for world cinema knowledge, built on the **CineQuery** retrieval foundation. CineAgent extends a local-first RAG system into a tool-using agent that can reason about films, directors, and — as tools come online — real-time data like streaming availability and current film metadata.

> **Status:** Active development. The CineQuery RAG foundation and the first two agent tools (local retrieval + TMDB film details) are complete and covered by a multi-tool evaluation harness (20/21, with one documented, intentional failure). The agent reasoning loop and the remaining tool are in progress — see the [Roadmap](#roadmap). This README tracks the *actual* state of the code and is updated as each phase ships.

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