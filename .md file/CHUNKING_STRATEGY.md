# CHUNKING_STRATEGY.md

# HH Goa 2026 — Chunking Strategy

## 1. Purpose

This document defines the chunking strategy for the HH Goa 2026 Voice-Enabled RAG system.

The challenge explicitly requires a thoughtful chunking approach rather than a single naive fixed-size splitter.

Therefore, the project will evaluate multiple chunking strategies and select the final strategy based on:

- Retrieval quality
- Context relevance
- Answer quality
- Chunk quality
- Index size
- Retrieval latency
- End-to-end latency

The final chunking strategy MUST be based on actual analysis of the AI4Bharat MSMARCO-XI dataset.

---

# 2. Dataset

## Dataset

AI4Bharat MSMARCO-XI

## Source

https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

The dataset is the primary knowledge source for the RAG system.

The chunking pipeline must preserve the relationship between:

```text
Original Dataset Record
        ↓
Document / Passage
        ↓
Chunk
        ↓
Embedding
        ↓
Vector Index
        ↓
Retrieved Chunk
        ↓
Final Answer