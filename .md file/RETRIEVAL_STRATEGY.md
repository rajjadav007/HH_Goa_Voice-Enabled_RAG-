# RETRIEVAL_STRATEGY.md

# HH Goa 2026 — Retrieval Strategy

## 1. Purpose

This document defines the retrieval architecture for the HH Goa 2026 Voice-Enabled RAG system.

The retrieval system is responsible for finding the most relevant information from the AI4Bharat MSMARCO-XI dataset before the answer is generated.

The core principle is:

> The LLM should answer from retrieved dataset context, not from its general knowledge.

The retrieval pipeline is:

```text
User Query
    ↓
Query Processing
    ↓
Query Embedding
    ↓
┌───────────────────┬───────────────────┐
│                   │                   │
▼                   ▼                   │
Qdrant             BM25                 │
Vector Search      Keyword Search       │
│                   │                   │
└─────────┬─────────┘                   │
          ▼                             │
     Result Fusion                      │
          ↓                             │
         RRF                            │
          ↓                             │
   Candidate Results                    │
          ↓                             │
      Reranking                         │
          ↓                             │
    Context Filtering                  │
          ↓                             │
    Final Top-K Context                │
          ↓                             │
       Gemini                          │
```

---

## 2. Production Embedding Specification (Phase 4.1)

- **Selected Model**: `intfloat/multilingual-e5-small`
- **Vector Dimension**: `384`
- **Similarity Metric**: `cosine`
- **Normalization**: L2 normalized (`normalize_embeddings=True`)
- **Query Prefix**: `"query: "`
- **Passage/Chunk Prefix**: `"passage: "`
- **Batch Processing**: Resumable vector generation stored in `data/embeddings/vectors.jsonl`.