# University RAG Assistant

A production-minded RAG backend that answers prospective student questions about study programs, admissions, financing, and application process.

> **In one paragraph:** A hybrid RAG backend for answering prospective student questions about university admissions. It combines dense vector search (ChromaDB) and BM25 keyword search to retrieve the right *combination* of general policy, program-specific rules, and exception documents — then assembles them into a structured, labelled context block before passing to GPT-4o. The system supports multi-turn conversation with persistent user profiles, multi-stage guardrails across the pipeline, and a 13-case evaluation suite.

---

## How it works

This system uses hybrid retrieval — combining dense vector search (ChromaDB) with sparse keyword search (BM25) — to assemble the right combination of general policy, program-specific rules, and exception documents before generating an answer.

Not only the most semantically similar chunks, this system retrieves the most *relevant combination*: a general policy doc, a program-specific override, and an exception document may all be needed to answer a single question correctly.

Key design decisions:
- Metadata pre-filtering (before similarity search) narrows ChromaDB to relevant chunks before ranking — filtering by detected program (e.g. only MSc DS docs) and region (e.g. only non-EU or general docs), so similarity scoring operates on a focused candidate pool rather than all 72 chunks.
- RRF fusion combines dense and sparse rankings without requiring weight tuning.
- A metadata scoring heuristic boosts exception and program-specific chunks when the query intent signals they are needed.
- Evidence buckets structure context into labelled sections before passing to GPT-4o, so the model sees [General Policy] / [Program-Specific] / [Exceptions] / [Process] explicitly rather than an unordered chunk dump.
- Multi-stage guardrails protect quality: a **scope guardrail** fires before retrieval (off-topic questions are rejected without making any embedding or database call), and a **confidence guardrail** fires after retrieval (if no chunk scored above a minimum threshold, a fallback response is returned instead of asking GPT-4o to hallucinate from weak evidence).
- The system prompt instructs GPT-4o: exception policy takes precedence over general policy when they conflict.

---

## Project structure

```
uni_rag_assistant/
├── app.py                  FastAPI app, GET / + POST /ask
├── config.py               Settings, env vars, constants
├── requirements.txt
├── .env.example
├── static/
│   └── index.html          Vanilla HTML chat UI
├── data/
│   ├── documents.py        All 18 documents (source of truth)
│   └── chunks.py           Section-based chunker
├── src/
│   ├── models.py           Pydantic schemas + dataclasses
│   ├── intent_parser.py    Rule-based intent extraction + LLM fallback
│   ├── state.py            In-memory conversation state
│   ├── indexer.py          One-time embed + ChromaDB + BM25 build
│   ├── retriever.py        HybridRetriever (async dense + sparse + RRF)
│   ├── context_builder.py  Evidence buckets → structured context string
│   ├── prompt_builder.py   Final system + user prompt assembly
│   ├── generator.py        GPT-4o call (async) + source extraction
│   └── guardrails.py       Confidence check + scope detection
├── evaluation/
│   ├── eval_cases.json     13 test cases (clarification, single-turn, multi-turn, insufficient info, out-of-scope)
│   ├── run_eval.py         Eval runner
│   └── demo.txt            Sample session transcript
└── examples/
    ├── sample_requests.json
    └── sample_responses.json
```

## 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-...

# Build the index (embeds all the chunks, builds ChromaDB + BM25 index)
python -m src.indexer
```

---

## 2. How to run

```bash
# Start the API server
uvicorn app:app --reload

# Test with curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "question": "Can I apply for the MSc Data Science with an economics degree?"}'
```

Response shape:
```json
{
  "answer": "...",
  "sources": ["gen_001", "prog_ds_001", "ctx_001"]
}
```

```bash
# Run evaluation suite (13 test cases)
python evaluation/run_eval.py
```

Example requests and expected responses are in `examples/sample_requests.json` and `examples/sample_responses.json`.

---

## Chat UI

A simple browser-based chat interface is available at http://localhost:8000 after starting the server. It demonstrates multi-turn conversation with source attribution.

---

## 3. Architecture & design decisions

### 4-layer document design

The dataset is intentionally structured across four layers of precedence:

| Priority | Layer | Purpose |
|---|---|---|
| 3 | Exception | Overrides everything — handles edge cases and special rules |
| 2 | Program-specific | Overrides general policy for a specific program |
| 1 | General | Default baseline policy |
| 0 | FAQ/Process | Supporting procedural information |

Documents are authored with **designed conflicts** (e.g. MBA scholarship exclusion overriding the general scholarship doc) because resolving these conflicts is exactly what the system must do well.

### Hybrid retrieval rationale

Dense-only retrieval (embeddings) misses exact policy terminology like "APS certificate" — these are keyword-specific and won't surface well from semantic search alone. BM25 provides the keyword precision that fills this gap. RRF fuses both without requiring weight tuning.

### Vector store — development vs production

ChromaDB was chosen for development simplicity: zero infrastructure, local disk persistence, and native metadata pre-filtering. Hybrid retrieval is assembled manually — dense retrieval via ChromaDB, sparse BM25 via a separate pickle file, and RRF fusion written in Python.

For production, the migration targets are **Weaviate** (BM25 native, hybrid fusion built-in) or **Qdrant** (sparse + dense hybrid, async Python client). Both collapse the separate BM25 index and manual RRF code into a single API call. The retrieval *architecture* — dense + sparse + metadata boost — stays identical; only the implementation layer changes.

The manual implementation was chosen deliberately for this challenge: it makes every retrieval decision explicit and explainable, whereas a native hybrid API call abstracts the mechanism away.

### Chunking strategy

Documents are split using **structural chunking** — paragraph boundaries (double newlines), with short paragraphs under 80 characters merged forward to avoid trivially small chunks, capped at 4 sections per document. This produces 72 chunks from 18 documents.

This approach works well here because the documents were authored with deliberate paragraph structure — each paragraph covers one policy point, so structural boundaries align with semantic ones by design.

For production with unstructured raw documents (PDFs, scraped pages), **sentence-embedding-based semantic chunking** would be preferred: embed each sentence, compute similarity between consecutive sentences, and cut where similarity drops below a threshold. This detects topic shifts regardless of physical formatting, at the cost of one extra embedding pass per document at index time and a similarity threshold hyperparameter to tune.

### Intent parser

A lightweight rule-based parser extracts intent type, program, region, and audience from the question text. For queries where rule-based matching returns UNKNOWN, a single GPT-4o-mini call (temperature=0, max_tokens=20) classifies the intent — adding no latency to the common case. The structured `QueryIntent` object drives both metadata filtering and context assembly.

Two OpenAI models are used: GPT-4o for answer generation, GPT-4o-mini for intent classification fallback.

### Evidence buckets

Chunks are distributed into four labeled buckets (General / Program-Specific / Exceptions / Process) before being passed to GPT-4o. This structure reduces the model's reasoning burden and produces more consistent answers than a raw chunk dump.

### Multi-turn conversation

Conversation state is maintained via a persistent `UserProfile` per `user_id`. Each detected signal — program, region, audience — is stored on first mention and inherited unconditionally on every subsequent turn via profile fallback (`program = detected or profile.program`). A follow-up like “Does that apply if I’m from India?” on turn 2 inherits `program: msc_ds` from turn 1 automatically.

A `needs_exception` flag is also inherited: once a user is identified as non-EU, exception documents receive a scoring boost on all subsequent turns — not only turns where the user explicitly mentions their region.

Prior conversation turns (last 3) are included in the prompt verbatim. This helps the model maintain continuity for follow-up questions but introduces a known limitation: prior generated answers re-enter as implicit context on subsequent turns. If an earlier answer contained a simplification, that text is treated as evidence rather than output. Production fix: summarise prior turns into a compact user-context block rather than including raw Q/A pairs.

### Guardrails

The system implements guardrails at three stages of the pipeline, each catching a different failure mode:

| Guardrail | Where | Catches | Response |
|---|---|---|---|
| Scope check | Before retrieval | Off-topic questions — detects out-of-domain queries before any embedding or database call is made | Canned “I can only help with IU programs” response |
| Retrieval confidence | After retrieval | Low retrieval scores — if no chunk scored above the minimum threshold, nothing reliable was found | Fallback “I couldn’t find enough information, contact IU admissions directly” |
| Source grounding | System prompt | Hallucination — model instructed to use only retrieved context and never invent policies, fees, or dates | Model returns “I don’t have enough information” and directs to admissions |
| Output validation | After generation | Hallucinated citations — model may cite doc_ids not in the retrieved context | Any cited source ID not present in the prompt is silently dropped before the response is returned |

The confidence threshold (0.10) is calibrated to the post-RRF scoring scale rather than raw cosine similarity. A score below this threshold means retrieval found nothing worth passing to the model — not even the program-match or exception boosts fired.

---

## 4. Retrieval & prompting approach

### Retrieval pipeline

```
question + session profile
    ↓
intent_parser  →  QueryIntent (intent, program, region, audience)
    ↓
dense_retrieve (ChromaDB + metadata pre-filter) + sparse_retrieve (BM25)  [concurrent]
    ↓
RRF fusion
    ↓
metadata scoring boost:
  +0.15  program match
  +0.12  exception boost (if needs_exception)
  +0.10  topic match
  +0.08  region match
  +0.06  audience match
    ↓
top 6 selected → assembled into 4 evidence buckets
```

This constitutes a four-stage retrieval pipeline: broad recall (dense + sparse), fusion, metadata reranking, and evidence selection — a lightweight form of staged retrieval where each stage refines the candidate set before the next.

### Metadata pre-filter

ChromaDB filters the candidate pool *before* similarity ranking. The `$in` clause always includes `"all"` alongside the specific program/region value, so general policy documents (which have `program="all"`) are never excluded — only other programs' specific docs are filtered out.

For `compare_options` intent, `intent.program` is set to `None` so no filter is applied and both program docs can be retrieved.

### Prompt structure

GPT-4o receives a structured context block with labelled sections:

```
[General Policy]
[Program-Specific Policy]
[Exceptions / Overrides]
[Process / Next Steps]
```

The system prompt explicitly instructs the model: "When general policy and program-specific or exception policy conflict, the more specific or exception document takes precedence." This mirrors the priority system in retrieval.

---

## Evaluation

The system is evaluated against 13 test cases via `evaluation/run_eval.py`, which hits the live API and checks source attribution and answer behaviour. All 13 pass.

| Category | Cases | What is tested |
|---|---|---|
| Single-turn eligibility | TC02–TC08, TC11 | Multi-doc retrieval, exception overrides, program comparison, non-EU routing |
| Multi-turn state | TC09, TC10 | Profile inheritance across 3 turns, mid-conversation program pivot |
| Clarification behaviour | TC01 | Underspecified question — model must ask rather than assume |
| Insufficient information | TC12 | No matching docs — model must direct to admissions rather than hallucinate |
| Out-of-scope | TC13 | Scope guardrail fires, off-topic question deflected gracefully |

---

## 5. Assumptions & trade-offs

- **In-memory state** — conversation history and user profiles are stored in a Python dict for challenge scope simplicity. In production, Redis with TTL is the standard choice: it survives server restarts, supports horizontal scaling across multiple instances, and allows session expiry.
- **Fixed dataset** — 18 documents, 72 chunks. No live data ingestion.
- **No authentication** — any `user_id` string creates a session.
- **Structural chunking** — documents are split on paragraph boundaries (`\n\n`). Paragraph boundaries in the authored documents align with semantic boundaries by design — in production with unstructured raw documents, sentence-embedding-based semantic chunking would be used instead to detect topic shifts regardless of physical formatting.
- **`valid_from` metadata field is stored but not wired** — Every chunk carries a `valid_from` date in its metadata. In production with periodically updated policy documents, a date-range filter on retrieval would exclude docs not yet in effect and allow automatic expiry of outdated policies via a `valid_until` field — without any document changes.
- **Confidence threshold manually calibrated** — The retrieval confidence check uses a threshold calibrated to the post-RRF scoring scale, not raw cosine similarity. A score below the threshold means no metadata boost fired — the system found nothing meaningful. The limitation is that this is a hand-tuned constant: if boost values or the scoring pipeline change, the threshold becomes wrong silently. Production improvement: set the threshold dynamically based on observed score distributions rather than a fixed value.
- **Rule-based intent parser** — keyword matching handles the common case; LLM fallback covers novel phrasings with a single GPT-4o-mini call.
- **Synthetic dataset** — Documents are realistic but simulated, inspired by typical university admissions processes.

---

## 6. Limitations & improvements

| Limitation | Improvement |
|---|---|
| In-memory sessions (lost on restart) | Redis with TTL for production multi-user deployments |
| ChromaDB queries are sync (wrapped in executor) | Native async vector store (e.g. Qdrant async client) for higher throughput |
| Rule-based intent parser | LLM fallback already implemented for UNKNOWN intents; full LLM classification for higher accuracy |
| Manually authored metadata | LLM-based metadata extraction pipeline for raw PDF/text ingestion in production |
| No cross-encoder reranking | Add a cross-encoder pass after RRF for higher precision |
| Fixed dataset | Document ingestion pipeline with re-indexing support |
| No streaming | `StreamingResponse` would improve perceived UX for a student-facing chatbot — GPT-4o takes 1–3s to generate; streaming lets the answer appear word-by-word rather than all at once after a blank wait |
| Conversation history includes raw prior answers | Production system would summarise prior turns to avoid reintroducing generated text as implicit evidence |
| Structural chunking (paragraph boundaries) | Semantic chunking using sentence embeddings and similarity-drop detection — splits on topic shifts rather than physical structure, more robust for unstructured raw documents like PDFs where paragraph breaks don't reliably align with semantic boundaries |
| No query rewriting | In production with real student queries, an LLM rewriting step would handle colloquial language and resolve ambiguous references before retrieval |
| Whitespace tokenization for BM25 | Phrase-level tokenization (bigrams for domain concepts like "professional experience pathway" or "credit transfer") — highest-value BM25 upgrade; simple tokenization is sufficient for this clean dataset but misses multi-word concepts |
| Indexer has no persisted chunk artifact or index manifest | For production, chunk generation should produce a canonical artifact (e.g. JSONL) consumed by both dense and sparse indexers from the same snapshot. An index manifest storing chunk count, doc count, embedding model, tokenizer version, and build timestamp would prevent drift between indexes, enable reproducibility, and make incremental re-indexing of changed documents possible without rebuilding everything. |
| Source extraction via regex — if model omits or reformats the `Sources:` line, fallback returns all candidate source IDs (over-credits but never fails silently) | OpenAI structured outputs — constrain the model to return `answer` and `sources` as separate JSON fields, eliminating the regex and the fallback entirely |
| Strict deduplication by doc_id — only the highest-scoring chunk per document is sent to the LLM | Top-N per document (e.g. top 2) would allow multi-faceted documents to contribute more than one chunk when a query requires multiple aspects from the same source. Currently a document covering both IELTS requirements and conditional admission would only contribute its highest-scoring chunk, potentially dropping relevant detail for multi-part questions. |
| Score-only token budget trimming — lowest-scoring chunks dropped globally with no guarantee of structural coverage | Three production alternatives: (A) **bucket-aware** — guarantee at least one chunk per non-empty evidence type (general, specific, exception, process) before filling remaining budget by score, so no category is entirely crowded out; (B) **intent-weighted** — allocate token budget proportionally by query type (eligibility queries get more exception/specific, application queries get more process); (C) **chunk compression** — summarise lower-priority chunks with a short LLM call instead of dropping them, minimising information loss at the cost of extra latency; (D) **exact token counting** with tiktoken instead of the chars/4 heuristic, which matters when running close to model context limits. |

---
