# CHUNKING_STRATEGY.md

# HH Goa 2026 — Chunking Strategy & Benchmark Selection

## 1. Purpose

This document defines the chunking strategy and records the Phase 3.3 benchmark results and final selection for the HH Goa 2026 Voice-Enabled RAG system.

The project evaluated 5 candidate chunking strategies on the AI4Bharat MSMARCO-XI dataset:
- Fixed-size sliding window
- Sentence-based boundary chunking
- Structure-aware paragraph chunking
- Semantic boundary distance chunking
- Hybrid paragraph/sentence/token overlap chunking

The final strategy was selected based on measured empirical performance (Recall@5, MRR, chunk distribution, storage efficiency).

---

## 2. Benchmark Methodology & Results (Phase 3.3)

### Evaluation Setup
- **Dataset**: Processed AI4Bharat MSMARCO-XI (validation set split).
- **Eval Sample**: 500 ground-truth evaluation queries, 2,000 documents (evaluation sample) and 9,982 documents (production set).
- **Metrics**: Recall@1, Recall@3, Recall@5, Recall@10, MRR, Avg Token Size, Composite Score (`0.6*Recall@5 + 0.4*MRR`).

### Benchmark Matrix Comparison
| Strategy | Target Size | Chunks (Sample) | Recall@5 | MRR | Avg Tokens/Chunk | Composite Score |
|---|---|---|---|---|---|---|
| `passthrough` | 256 | 2,000 | 0.1411 | 0.0881 | 216.34 | 0.1199 |
| `fixed` | 128 | 2,032 | 0.1411 | 0.0881 | 213.84 | 0.1199 |
| `fixed` | 256 | 2,007 | 0.1411 | 0.0881 | 216.07 | 0.1199 |
| `sentence` | 256 | 2,426 | 0.1452 | 0.0972 | 178.36 | 0.1260 |
| `structure` | 256 | 2,038 | 0.1411 | 0.0884 | 212.31 | 0.1200 |
| **`semantic` (th=0.5)** | **256** | **6,787** | **0.1613** | **0.1062** | **63.75** | **0.1393** |
| `semantic` (th=0.75) | 256 | 6,962 | 0.1573 | 0.1011 | 62.15 | 0.1348 |
| `hybrid` | 256 | 2,428 | 0.1411 | 0.0969 | 178.90 | 0.1234 |

---

## 3. Final Production Selection & Rationale

- **Selected Strategy**: `semantic`
- **Selected Configuration**:
  ```yaml
  chunking:
    strategy: "semantic"
    target_chunk_size: 256
    max_chunk_size: 512
    overlap: 32
    semantic_threshold: 0.5
  ```
- **Rationale**:
  - `semantic` boundary chunking achieved the highest **Recall@5 (0.1613)** and highest **MRR (0.1062)** among all candidates.
  - Granular, semantically coherent ~64-token chunks improve retrieval precision for downstream multilingual RAG query matching.
- **Production Chunk Dataset Outputs**:
  - `data/chunks/final_chunks.jsonl`: **33,113 total chunks** generated from 9,982 documents.
  - `data/chunks/final_manifest.json`: Production chunk manifest with checksums and statistical summary.