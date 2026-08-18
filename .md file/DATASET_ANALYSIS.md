# DATASET_ANALYSIS.md

# HH Goa 2026 — MSMARCO-XI Dataset Analysis

## 1. Purpose

This document defines how the AI4Bharat MSMARCO-XI dataset is inspected, analyzed, processed, and prepared for the voice-enabled RAG system.

The dataset is the primary knowledge source for the project.

The RAG system MUST retrieve information from this dataset before generating an answer.

The LLM (Gemini) must NOT receive the raw dataset or replace the knowledge corpus.

---

## 2. Official Hugging Face Dataset Details

### Dataset Name
`ai4bharat/MSMARCO-XI`

### Official Source URL
`https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`

### Discovered Dataset Structure & Scale
- **`train` split**: 10,080,140 rows (~10.08 Million raw records)
- **`validation` split**: 1,371,174 rows (~1.37 Million raw records across 11 Indic languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu)

### Column Schemas Discovered at Runtime
- `query_id`: `int64` (unique query identifier)
- `query`: `string` (translated query text in Indic target language)
- `Eng_Query`: `string` (English query text)
- `Answer`: `string` (translated ground-truth answer text)
- `Eng_Answer`: `string` (English ground-truth answer text)
- `source_lang`: `string`
- `target_lang`: `string`
- `query_type`: `string`
- `meta`: `dict` (model hyperparameters)
- `passages`:
  - `English_passages`: `List[string]`
  - `Translated_passages`: `List[string]`
  - `is_selected`: `List[int64]` (binary ground-truth relevance: 1=relevant, 0=not relevant)

---

## 3. Ingestion & Preprocessing Architecture

```text
Hugging Face MSMARCO-XI
        ↓
Memory-Efficient Parquet Loader (`HF_HOME` on D: drive)
        ↓
MSMARCOPreprocessor (Unicode NFC, Whitespace, Control Char Strip)
        ↓
Stable Document ID Generation (`doc_{query_id}_{passage_idx}_{content_hash}`)
        ↓
`data/processed/documents.jsonl` + `queries.jsonl`
        ↓
BatchChunkProcessor (`final_chunks.jsonl`)
        ↓
Dual Index Construction (Qdrant Point Count == BM25 Chunk Count == Processed Chunks)
```

---

## 4. Configurable Dataset Ingestion Modes

- **Development Subset Mode**: Configurable via `DATASET_MAX_ROWS` environment variable or `--max-rows` CLI parameter (e.g. `DATASET_MAX_ROWS=1000` for fast local testing).
- **Full Ingestion Mode**: Setting `--max-rows 0` streams the full target dataset split without artificial capping.
- **Disk Cache Location**: `HF_HOME` points explicitly to `D:\HH GOA\data\raw\cache` to prevent system drive space exhaustion.

---

## 5. Preprocessing & Indexing Statistics (Production Run)

- **Raw Rows Processed**: 1,000 queries
- **Processed Documents**: 9,980 documents
- **Generated Chunks**: 9,980 chunks
- **Qdrant Vector Points**: 10,980 points
- **BM25 Documents**: 9,980 documents
- **Dual-Index ID Consistency**: PASS (`Qdrant IDs` == `BM25 IDs` == `Processed Chunk IDs`)
- **Output Artifacts**:
  - `data/processed/queries.jsonl`
  - `data/processed/documents.jsonl`
  - `data/chunks/final_chunks.jsonl`
  - `data/embeddings/vectors.jsonl`
  - `data/qdrant_db/`
  - `data/bm25_index/bm25.pkl`