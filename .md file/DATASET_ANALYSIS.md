# DATASET_ANALYSIS.md

# HH Goa 2026 — MSMARCO-XI Dataset Analysis

## 1. Purpose

This document defines how the AI4Bharat MSMARCO-XI dataset will be inspected, analyzed, processed, and prepared for the voice-enabled RAG system.

The dataset is the primary knowledge source for the project.

The RAG system MUST retrieve information from this dataset before generating an answer.

The LLM must not be used as a replacement for the dataset.

---

# 2. Dataset Information

## Dataset Name

AI4Bharat MSMARCO-XI

## Source

Hugging Face:

https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

## Role in the Project

The dataset provides the knowledge/context that the RAG pipeline retrieves to answer user questions.

The basic architecture is:

```text
MSMARCO-XI
    ↓
Dataset Analysis
    ↓
Preprocessing
    ↓
Chunking
    ↓
Embeddings
    ↓
Qdrant
    +
BM25
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Relevant Context
    ↓
Gemini
    ↓
Grounded Answer
```

---

## 3. Phase 2.3 Preprocessing Decisions & Summary

### 3.1 Preprocessing Pipeline
- **Raw Data Loader**: Reused `MSMARCODatasetLoader` (Phase 2.1).
- **Text Normalization**: Unicode NFC normalization, ASCII control character removal (except `\n`, `\t`), whitespace collapse (`\s+` -> ` `). No aggressive lowercasing, stopword stripping, or HTML removal.
- **Stable Document ID**: Deterministic format `doc_{query_id}_{passage_idx}_{sha256(text)[:12]}`.
- **Ground-Truth & Relationships**: Preserved `is_selected` (0/1) per document and mapping between `query_id` and `relevant_document_ids`.

### 3.2 Preprocessing Statistics (Run Output)
- **Input Raw Records**: 1,000
- **Processed Queries**: 1,000
- **Processed Documents**: 9,982
- **Rejected Records**: 0
- **Deduplicated Passages**: 6

### 3.3 Output Artifacts
- `data/processed/queries.jsonl` — Normalized query records with ground-truth document IDs.
- `data/processed/documents.jsonl` — Normalized document records ready for Milestone 3 chunking.
- `data/processed/manifest.json` — Preprocessing execution manifest and configuration.