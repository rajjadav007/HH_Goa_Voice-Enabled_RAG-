# IMPLEMENTATION PLAN

# HH Goa 2026 — Voice-Enabled RAG System

**Project:** HH Goa 2026 Shortlisting Task 2
**Primary Dataset:** AI4Bharat MSMARCO-XI
**STT:** Sarvam AI
**LLM:** Gemini
**Vector Database:** Qdrant
**Keyword Retrieval:** BM25
**Backend:** Python + FastAPI
**Frontend:** React + Vite + TypeScript
**Target Online RAG Latency:** < 200 ms
**Implementation Status:** Not Started

---

# 1. Purpose

This document defines the exact implementation sequence for the HH Goa 2026 Voice-Enabled RAG project.

The implementation MUST be performed incrementally.

The coding agent must not attempt to build the entire project in a single step.

The implementation must progress through the following stages:

```text
Phase 0  → Project Setup
Phase 1  → Dataset Analysis
Phase 2  → Data Preprocessing
Phase 3  → Multi-Strategy Chunking
Phase 4  → Embedding Pipeline
Phase 5  → Qdrant Vector Index
Phase 6  → BM25 Index
Phase 7  → Basic Text RAG
Phase 8  → Hybrid Retrieval
Phase 9  → Reranking
Phase 10 → Gemini Generation
Phase 11 → RAG Harness
Phase 12 → Guardrails
Phase 13 → Evaluation System
Phase 14 → Sarvam Voice
Phase 15 → FastAPI Production API
Phase 16 → React Frontend
Phase 17 → Latency Optimization
Phase 18 → Final Benchmarking
Phase 19 → Deployment
Phase 20 → Final QA
Phase 21 → Submission Preparation
```

---

# 2. Implementation Rules

Before starting any phase, the coding agent must:

1. Read `PRD.md`.
2. Read `ARCHITECTURE.md`.
3. Read `AGENTS.md`.
4. Check the current project state.
5. Identify dependencies for the phase.
6. Implement only the requested phase.
7. Run relevant tests.
8. Validate the result.
9. Update documentation if necessary.
10. Mark the phase complete only when acceptance criteria are satisfied.

---

# 3. Golden Implementation Principle

The system must be built from the inside out:

```text
DATA
 ↓
INDEX
 ↓
RETRIEVAL
 ↓
RAG
 ↓
GUARDRAILS
 ↓
VOICE
 ↓
API
 ↓
UI
 ↓
OPTIMIZATION
 ↓
DEPLOYMENT
```

Do not start with the frontend.

Do not start with animations.

Do not start with voice.

First make the dataset-grounded RAG pipeline work correctly.

---

# 4. Phase 0 — Project Setup

## Objective

Create the basic repository structure and development environment.

---

## Tasks

### 4.1 Initialize Git

Create:

```text
.git/
```

Initialize the repository.

---

### 4.2 Create Root Documentation

Create:

```text
README.md
PRD.md
ARCHITECTURE.md
AGENTS.md
IMPLEMENTATION_PLAN.md
```

---

### 4.3 Create Root Structure

Create:

```text
hh-goa-voice-rag/
│
├── backend/
├── frontend/
├── ingestion/
├── evaluation/
├── scripts/
├── tests/
├── docs/
│
├── README.md
├── PRD.md
├── ARCHITECTURE.md
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── Dockerfile
```

---

## 4.4 Backend Setup

Initialize Python environment.

Use a supported Python version compatible with selected dependencies.

Create:

```text
backend/
├── app/
├── tests/
├── requirements.txt
└── ...
```

---

## 4.5 Frontend Setup

Initialize:

```text
React
Vite
TypeScript
```

Add only necessary UI dependencies.

---

## 4.6 Environment Configuration

Create:

```text
.env.example
```

Initial variables:

```text
SARVAM_API_KEY=
GEMINI_API_KEY=

QDRANT_URL=
QDRANT_API_KEY=

EMBEDDING_MODEL=
RERANKER_MODEL=
LLM_MODEL=

ENVIRONMENT=
LOG_LEVEL=
```

Do not add actual credentials.

---

## 4.7 Initial Backend Health Endpoint

Implement:

```text
GET /api/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

---

## Phase 0 Acceptance Criteria

```text
[ ] Repository created
[ ] Git initialized
[ ] Backend starts
[ ] Frontend starts
[ ] FastAPI health endpoint works
[ ] .env.example exists
[ ] .env is ignored
[ ] No secrets committed
[ ] Basic README exists
```

---

# 5. Phase 1 — Dataset Analysis

## Objective

Understand the MSMARCO-XI dataset before implementing retrieval.

Dataset:

```text
https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
```

---

## Tasks

### 5.1 Load Dataset

Implement:

```text
ingestion/load_dataset.py
```

---

### 5.2 Inspect Dataset Schema

Determine:

* Available fields
* Dataset splits
* Record types
* IDs
* Text fields
* Query fields
* Metadata

Do not assume field names.

---

### 5.3 Dataset Statistics

Calculate:

```text
Number of records
Number of documents
Number of queries where available
Text length distribution
Missing values
Duplicate records
Language distribution
```

---

### 5.4 Analyze Text Structure

Determine:

* Average text length
* Median text length
* Maximum text length
* Sentence distribution
* Paragraph structure
* Metadata structure

---

### 5.5 Create Dataset Analysis Report

Create:

```text
DATASET_ANALYSIS.md
```

Include:

```text
Dataset source
Schema
Statistics
Text characteristics
Metadata
Duplicate analysis
Missing-value analysis
Chunking recommendations
Retrieval recommendations
```

---

## Phase 1 Acceptance Criteria

```text
[ ] Dataset successfully loaded
[ ] Schema documented
[ ] Record counts measured
[ ] Text statistics measured
[ ] Missing values analyzed
[ ] Duplicates analyzed
[ ] Language analyzed where applicable
[ ] DATASET_ANALYSIS.md created
```

Do not start chunking until this phase is complete.

---

# 6. Phase 2 — Data Preprocessing

## Objective

Create clean and deterministic input data for chunking.

---

## Tasks

Implement:

```text
ingestion/preprocess.py
```

---

## Processing

Pipeline:

```text
Raw Dataset
 ↓
Schema Validation
 ↓
Invalid Record Removal
 ↓
Text Cleaning
 ↓
Unicode Normalization
 ↓
Whitespace Normalization
 ↓
Metadata Preservation
 ↓
Duplicate Handling
 ↓
Stable IDs
 ↓
Processed Dataset
```

---

## Requirements

Do not destroy useful semantic information.

Do not aggressively remove words simply because they look unusual.

Preserve original text where practical for traceability.

---

## Stable IDs

Every record must have a stable:

```text
document_id
```

---

## Phase 2 Acceptance Criteria

```text
[ ] Preprocessing script works
[ ] Invalid records handled
[ ] Text normalized
[ ] Duplicates handled
[ ] Stable IDs generated
[ ] Metadata preserved
[ ] Processed dataset can be regenerated
[ ] Unit tests pass
```

---

# 7. Phase 3 — Multi-Strategy Chunking

## Objective

Implement a serious chunking system rather than a single naive fixed-size splitter.

---

# 7.1 Chunking Interface

Create a common interface:

```text
Chunker
```

Possible structure:

```text
Chunker
├── FixedChunker
├── SentenceChunker
├── StructureAwareChunker
├── SemanticChunker
└── HybridChunker
```

---

# 7.2 Fixed Chunking

Implement configurable:

```text
chunk_size
overlap
```

This is the baseline only.

---

# 7.3 Sentence Chunking

Split according to sentence boundaries.

Avoid splitting important semantic units unnecessarily.

---

# 7.4 Structure-Aware Chunking

If dataset records contain meaningful structure, preserve it.

Examples:

```text
Paragraph
Section
Title
Passage
Metadata
```

Only implement fields that actually exist in MSMARCO-XI.

---

# 7.5 Semantic Chunking

Implement semantic splitting only if dataset structure and computational cost justify it.

Do not introduce a heavy semantic chunker merely to appear advanced.

---

# 7.6 Hybrid Chunking

Combine useful signals such as:

```text
Structure
+
Sentence boundaries
+
Token length
+
Semantic boundaries
```

---

# 7.7 Chunk Metadata

Every chunk must contain:

```text
chunk_id
document_id
chunk_index
text
chunk_strategy
metadata
```

---

# 7.8 Chunk Benchmark

Compare strategies using:

```text
Chunk count
Average chunk size
Index size
Retrieval quality
Retrieval latency
Context quality
Answer quality
```

---

# 7.9 Select Final Strategy

Select the best strategy based on actual evaluation.

Do not select based on intuition alone.

---

## Phase 3 Acceptance Criteria

```text
[ ] Chunking interface exists
[ ] At least 2 meaningful strategies implemented
[ ] Chunk metadata exists
[ ] Chunking is configurable
[ ] Strategies benchmarked
[ ] Final strategy selected
[ ] CHUNKING_STRATEGY.md created
[ ] Chunking tests pass
```

---

# 8. Phase 4 — Embedding Pipeline

## Objective

Convert chunks into vectors efficiently.

---

## 8.1 Embedding Interface

Create:

```text
EmbeddingService
```

Methods:

```text
embed_text()
embed_batch()
get_dimension()
```

---

## 8.2 Model Selection

Start with a fast local embedding model.

Evaluate:

```text
Quality
Latency
Memory
Vector dimension
Deployment complexity
```

---

## 8.3 Batch Embedding

The ingestion pipeline must embed chunks in batches.

Do not embed one record at a time if batch inference is available.

---

## 8.4 Embedding Cache

Optionally cache generated embeddings during ingestion to prevent unnecessary recomputation.

---

## Phase 4 Acceptance Criteria

```text
[ ] Embedding model selected
[ ] Embedding service implemented
[ ] Batch embedding works
[ ] Vector dimensions validated
[ ] Embedding benchmark completed
[ ] Tests pass
```

---

# 9. Phase 5 — Qdrant Vector Index

## Objective

Create the semantic vector retrieval system.

---

## 9.1 Qdrant Setup

Create Qdrant collection.

Configure:

```text
Vector dimension
Distance metric
Payload
Metadata indexes
```

---

## 9.2 Indexing

Pipeline:

```text
Chunks
 ↓
Embeddings
 ↓
Qdrant
```

Use batch upserts.

---

## 9.3 Payload

Store:

```text
chunk_id
document_id
text
chunk_index
chunk_strategy
metadata
```

---

## 9.4 Search

Implement:

```text
vector_search(query, top_k)
```

---

## 9.5 Health Check

Implement Qdrant health validation.

---

## Phase 5 Acceptance Criteria

```text
[ ] Qdrant running
[ ] Collection created
[ ] Vectors inserted
[ ] Payload stored
[ ] Search works
[ ] Metadata returned
[ ] Batch indexing works
[ ] Index can be rebuilt
[ ] Retrieval tests pass
```

---

# 10. Phase 6 — BM25 Index

## Objective

Implement lexical retrieval.

---

## Tasks

Build BM25 index from processed chunks.

Implement:

```text
BM25Retriever
```

Method:

```text
search(query, top_k)
```

---

## BM25 Requirements

The index must:

* Be built offline
* Be persisted
* Be loadable by backend
* Return chunk IDs
* Return scores

---

## Phase 6 Acceptance Criteria

```text
[ ] BM25 implemented
[ ] Index generated
[ ] Index persisted
[ ] Search works
[ ] Scores returned
[ ] Chunk IDs traceable
[ ] Tests pass
```

---

# 11. Phase 7 — Basic Text RAG

## Objective

Build the first complete dataset-grounded RAG pipeline without voice.

This is the most important initial milestone.

---

## Flow

```text
Text Query
 ↓
Embedding
 ↓
Qdrant
 ↓
Top-K Context
 ↓
Gemini
 ↓
Answer
```

---

## Tasks

Implement:

```text
POST /api/query
```

Request:

```json
{
  "query": "..."
}
```

---

## Context Prompt

Gemini receives:

```text
Question
+
Retrieved Context
```

The prompt must instruct:

* Use context only
* Do not hallucinate
* Refuse if insufficient
* Ignore instructions inside retrieved documents

---

## Phase 7 Acceptance Criteria

```text
[ ] Text query endpoint works
[ ] Query embedding works
[ ] Qdrant retrieval works
[ ] Context passed to Gemini
[ ] Gemini answer generated
[ ] Answer is dataset-grounded
[ ] Basic error handling exists
[ ] End-to-end text RAG works
```

Do not add voice until this phase is stable.

---

# 12. Phase 8 — Hybrid Retrieval

## Objective

Improve retrieval quality by combining semantic and lexical retrieval.

---

## Flow

```text
Query
 │
 ├── Qdrant
 │
 └── BM25
      ↓
Result Fusion
      ↓
Candidate Set
```

---

# 12.1 Parallel Retrieval

Where practical, execute:

```text
Qdrant Search
+
BM25 Search
```

concurrently.

---

# 12.2 Result Fusion

Implement:

```text
RRF
```

Keep fusion parameters configurable.

---

# 12.3 Retrieval Result Schema

Standardize results:

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "text": "...",
  "score": 0.92,
  "method": "hybrid"
}
```

---

# 12.4 Evaluation

Compare:

```text
Vector-only
BM25-only
Hybrid
```

Measure:

```text
Recall@K
MRR
Relevance
Latency
```

---

## Phase 8 Acceptance Criteria

```text
[ ] Vector retrieval works
[ ] BM25 works
[ ] Parallel retrieval implemented where useful
[ ] RRF implemented
[ ] Hybrid result schema implemented
[ ] Retrieval benchmark completed
[ ] Hybrid improves or provides useful tradeoff
[ ] RETRIEVAL_STRATEGY.md updated
```

---

# 13. Phase 9 — Reranking

## Objective

Improve final context quality.

---

## Flow

```text
Hybrid Retrieval
 ↓
Candidate Set
 ↓
Reranker
 ↓
Top 3–5
```

---

## Tasks

Implement:

```text
Reranker
```

Use a lightweight local model initially.

---

## Benchmark

Compare:

```text
Without reranker
vs
With reranker
```

Measure:

```text
Retrieval quality
Answer quality
Latency
```

---

## Phase 9 Acceptance Criteria

```text
[ ] Reranker interface exists
[ ] Local reranker implemented
[ ] Candidate count configurable
[ ] Benchmark completed
[ ] Quality impact measured
[ ] Latency impact measured
[ ] Final reranker decision documented
```

If reranking significantly harms latency with little quality improvement, do not force it into the final online path.

---

# 14. Phase 10 — Gemini Generation

## Objective

Build a robust structured answer-generation layer.

---

## 14.1 LLM Service

Create:

```text
LLMService
```

Implementation:

```text
GeminiLLMService
```

---

## 14.2 Prompt Management

Keep prompts outside business logic.

Recommended:

```text
backend/app/prompts/
```

---

## 14.3 Structured Output

Expected structure:

```json
{
  "answer": "...",
  "grounded": true,
  "confidence": 0.92,
  "source_chunk_ids": []
}
```

---

## 14.4 Response Validation

Validate:

* Schema
* Answer presence
* Source IDs
* Grounded flag
* Confidence range

---

## Phase 10 Acceptance Criteria

```text
[ ] Gemini service implemented
[ ] Prompt separated
[ ] Structured output implemented
[ ] Response validation implemented
[ ] Invalid responses handled
[ ] Dataset context always passed
[ ] Tests pass
```

---

# 15. Phase 11 — RAG Harness

## Objective

Create proper orchestration around the model and retrieval system.

This is a major challenge requirement.

---

# 15.1 Orchestrator

Create:

```text
RAGOrchestrator
```

Pipeline:

```text
Input
 ↓
Validate
 ↓
Preprocess
 ↓
Embed
 ↓
Retrieve
 ↓
Fuse
 ↓
Rerank
 ↓
Context Validate
 ↓
Generate
 ↓
Validate Output
 ↓
Grounding
 ↓
Response
```

---

# 15.2 Request State

Use a structured state object.

It should track:

```text
request_id
query
transcript
retrieval_results
context
generation
grounding
latency
errors
```

---

# 15.3 Retry Handling

Implement bounded retries for transient failures.

---

# 15.4 Error Recovery

Handle:

```text
STT failure
Embedding failure
Qdrant failure
BM25 failure
Reranker failure
Gemini failure
Grounding failure
Timeout
```

---

## Phase 11 Acceptance Criteria

```text
[ ] RAG orchestrator exists
[ ] Pipeline stages are explicit
[ ] Request state exists
[ ] Retry logic exists
[ ] Error classification exists
[ ] Structured response exists
[ ] Logs contain request IDs
[ ] Harness tests pass
```

---

# 16. Phase 12 — Guardrails

## Objective

Ensure the system knows when not to answer.

---

# 16.1 Input Guardrail

Implement checks for:

* Empty input
* Excessive input
* Invalid input
* Prompt injection
* Unsafe input where applicable

---

# 16.2 Retrieval Guardrail

Check:

```text
Number of results
Similarity
Relevance
Context availability
```

---

# 16.3 Insufficient Context

If evidence is insufficient:

```text
Do not call Gemini
```

or return a controlled answer according to the final architecture.

Preferred:

```text
I couldn't find enough relevant information in the provided dataset to answer that.
```

---

# 16.4 Grounding Validator

After Gemini:

```text
Answer
 ↓
Grounding Validator
 ↓
Supported?
```

If not:

```text
Retry
 ↓
Validate again
 ↓
Reject if still unsupported
```

---

# 16.5 Prompt Injection

Retrieved content must never override system instructions.

---

## Phase 12 Acceptance Criteria

```text
[ ] Input guardrail works
[ ] Retrieval guardrail works
[ ] Insufficient-context behavior works
[ ] Prompt injection protection works
[ ] Grounding validation works
[ ] Retry/rejection works
[ ] Guardrail tests pass
```

---

# 17. Phase 13 — Evaluation System

## Objective

Create a reproducible benchmark system.

---

# 17.1 Test Query Dataset

Create:

```text
evaluation/test_queries.json
```

Include representative queries.

Categories:

```text
Normal
Semantic
Keyword-heavy
Short
Long
Ambiguous
No-answer
Off-topic
Adversarial
Prompt injection
```

---

# 17.2 Retrieval Evaluation

Measure:

```text
Recall@K
MRR
Precision@K where possible
Context relevance
```

---

# 17.3 Answer Evaluation

Measure where ground truth permits:

```text
Correctness
Relevance
Groundedness
Source correctness
Hallucination rate
```

---

# 17.4 Failure Evaluation

Measure:

```text
Guardrail rejection rate
Retry rate
Failure rate
Invalid output rate
```

---

## Phase 13 Acceptance Criteria

```text
[ ] Test query dataset created
[ ] Retrieval evaluation implemented
[ ] Answer evaluation implemented
[ ] Grounding evaluation implemented
[ ] Failure evaluation implemented
[ ] Evaluation report generated
```

---

# 18. Phase 14 — Sarvam Voice Integration

## Objective

Add voice input only after text RAG is stable.

---

# 18.1 Sarvam Service

Create:

```text
SpeechToTextService
```

Implementation:

```text
SarvamSpeechToTextService
```

---

# 18.2 Audio Validation

Validate:

```text
MIME type
File size
Duration
Format
Empty audio
```

---

# 18.3 Voice Pipeline

Implement:

```text
Audio
 ↓
Validation
 ↓
Sarvam
 ↓
Transcript
 ↓
RAGOrchestrator
 ↓
Answer
```

---

# 18.4 Voice Endpoint

Implement:

```text
POST /api/voice/query
```

Response:

```json
{
  "request_id": "...",
  "transcript": "...",
  "answer": "...",
  "grounded": true,
  "sources": [],
  "latency": {}
}
```

---

## Phase 14 Acceptance Criteria

```text
[ ] Sarvam integrated
[ ] Audio validation implemented
[ ] Transcription works
[ ] Voice query works
[ ] Transcript returned
[ ] Transcript passed to RAG
[ ] Error handling implemented
[ ] Voice integration tests pass
```

---

# 19. Phase 15 — FastAPI Production API

## Objective

Convert the internal pipeline into clean production APIs.

---

# 19.1 Endpoints

Implement:

```text
GET  /api/health
POST /api/query
POST /api/voice/query
GET  /api/metrics
```

---

# 19.2 Request Validation

Use Pydantic.

Validate:

* Query length
* Audio
* Content type
* Request size

---

# 19.3 Error Responses

Use standardized errors.

Example:

```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_CONTEXT",
    "message": "The dataset does not contain enough relevant information."
  }
}
```

---

# 19.4 CORS

Configure frontend origin correctly.

Do not use unrestricted production CORS without justification.

---

# 19.5 Rate Limiting

Add rate limiting if required for deployment.

---

## Phase 15 Acceptance Criteria

```text
[ ] API endpoints implemented
[ ] Pydantic validation works
[ ] Standard responses work
[ ] Error responses work
[ ] CORS configured
[ ] Rate limiting considered
[ ] API tests pass
```

---

# 20. Phase 16 — React Frontend

## Objective

Build the user-facing voice RAG interface.

---

# 20.1 UI Structure

Create:

```text
VoiceRecorder
TranscriptPanel
AnswerPanel
SourcePanel
LatencyPanel
StatusIndicator
ErrorDisplay
```

---

# 20.2 Application States

Implement:

```text
IDLE
RECORDING
TRANSCRIBING
RETRIEVING
GENERATING
VALIDATING
SUCCESS
ERROR
```

---

# 20.3 Voice Recording

The user must be able to:

```text
Start
 ↓
Record
 ↓
Stop
 ↓
Submit
```

Handle microphone permissions.

---

# 20.4 Transcript

Show:

```text
Your question
```

after Sarvam transcription.

---

# 20.5 Answer

Display the answer prominently.

---

# 20.6 Sources

Display relevant source/chunk information where appropriate.

---

# 20.7 Latency

Show:

```text
Total latency
```

and optionally stage-level latency in developer mode.

---

# 20.8 Error States

Display friendly messages.

Examples:

```text
Unable to process audio.
```

```text
No relevant information was found in the dataset.
```

---

## Phase 16 Acceptance Criteria

```text
[ ] Frontend starts
[ ] Microphone works
[ ] Recording state works
[ ] Voice API connected
[ ] Transcript displayed
[ ] Answer displayed
[ ] Sources displayed
[ ] Latency displayed
[ ] Errors displayed
[ ] Responsive UI
[ ] Accessibility basics implemented
```

---

# 21. Phase 17 — Latency Instrumentation

## Objective

Measure every important stage.

---

# 21.1 Stage Timers

Measure:

```text
STT
Query Processing
Embedding
Vector Search
BM25
Fusion
Reranking
Context Preparation
Gemini
Grounding
Total
```

---

# 21.2 Request Metrics

Store:

```text
request_id
timestamp
stage
duration_ms
status
```

---

# 21.3 Latency Report

Create:

```text
LATENCY.md
```

Document:

* Benchmark environment
* Number of queries
* Methodology
* P50
* P70
* P100
* Mean
* Minimum
* Maximum

---

## Phase 17 Acceptance Criteria

```text
[ ] Stage-level timers implemented
[ ] Total latency measured
[ ] Metrics collected
[ ] Benchmark runner implemented
[ ] LATENCY.md created
```

---

# 22. Phase 18 — Latency Optimization

## Objective

Optimize the online pipeline toward <200 ms.

---

# 22.1 First Measure

Do not optimize before obtaining baseline measurements.

---

# 22.2 Identify Bottlenecks

Example:

```text
STT          40ms
Embedding    12ms
Qdrant        8ms
BM25          3ms
Reranker     25ms
Gemini       90ms
Grounding    12ms
```

These are examples only.

Actual values must be measured.

---

# 22.3 Optimization Areas

Potential optimizations:

```text
Local embeddings
Async retrieval
Parallel Qdrant/BM25
Connection reuse
Smaller context
Lightweight reranker
Caching
Model optimization
Reduced network calls
```

---

# 22.4 Quality Protection

After every optimization:

```text
Benchmark latency
+
Benchmark retrieval
+
Benchmark answer quality
```

Do not keep an optimization that causes unacceptable quality degradation.

---

## Phase 18 Acceptance Criteria

```text
[ ] Baseline latency measured
[ ] Bottleneck identified
[ ] Optimizations tested
[ ] Retrieval quality rechecked
[ ] Answer quality rechecked
[ ] Final configuration selected
```

---

# 23. Phase 19 — Final Benchmarking

## Objective

Produce the official performance numbers.

---

# 23.1 Test Set

Use a reasonable test set.

Recommended:

```text
100+ queries
```

---

# 23.2 Required Metrics

Calculate:

```text
P50
P70
P100
```

Also calculate:

```text
Mean
Min
Max
Standard deviation
```

---

# 23.3 Retrieval Metrics

Calculate:

```text
Recall@K
MRR
Precision@K where possible
```

---

# 23.4 Grounding Metrics

Calculate:

```text
Grounded answer rate
Ungrounded answer rate
Rejected answer rate
Retry rate
```

---

# 23.5 Final Benchmark Report

Create/update:

```text
EVALUATION.md
LATENCY.md
```

---

# 23.6 Honesty Requirement

Never modify or remove results to make the benchmark look better.

Report actual values.

If the target is not achieved, document:

```text
Target
Actual
Bottleneck
Reason
Future optimization
```

---

## Phase 19 Acceptance Criteria

```text
[ ] 100+ queries tested
[ ] P50 calculated
[ ] P70 calculated
[ ] P100 calculated
[ ] Retrieval metrics calculated
[ ] Grounding metrics calculated
[ ] Failure metrics calculated
[ ] Results documented
```

---

# 24. Phase 20 — Deployment

## Objective

Deploy a working public version.

---

# 24.1 Backend

Deploy:

```text
FastAPI
```

---

# 24.2 Vector Database

Deploy:

```text
Qdrant
```

Ensure production index is loaded.

---

# 24.3 Frontend

Deploy:

```text
React
```

---

# 24.4 Environment Variables

Configure production:

```text
SARVAM_API_KEY
GEMINI_API_KEY
QDRANT_URL
QDRANT_API_KEY
EMBEDDING_MODEL
RERANKER_MODEL
LLM_MODEL
```

Never put private credentials in frontend environment variables.

---

# 24.5 Production Test

Test:

```text
Frontend
 ↓
Backend
 ↓
Sarvam
 ↓
Retrieval
 ↓
Gemini
 ↓
Grounding
 ↓
Answer
```

---

## Phase 20 Acceptance Criteria

```text
[ ] Frontend deployed
[ ] Backend deployed
[ ] Qdrant available
[ ] Production environment variables configured
[ ] Health endpoint works
[ ] Text query works
[ ] Voice query works
[ ] Grounding works
[ ] Production latency measured
```

---

# 25. Phase 21 — Final QA

## Objective

Perform complete system validation before submission.

---

# 25.1 Functional QA

Test:

```text
Normal voice query
Normal text query
Long query
Short query
No-answer query
Off-topic query
Prompt injection
Invalid audio
Empty audio
API failure
Qdrant failure
Gemini failure
Sarvam failure
```

---

# 25.2 Security QA

Check:

```text
No API keys in Git
No secrets in frontend
.env ignored
.env.example complete
Production CORS configured
Request limits configured
```

---

# 25.3 Performance QA

Verify:

```text
P50
P70
P100
```

---

# 25.4 Retrieval QA

Verify:

```text
Vector retrieval
BM25
Hybrid retrieval
Reranking
Source IDs
```

---

# 25.5 Grounding QA

Verify:

```text
Correct answer
Unsupported answer rejection
Insufficient context
Prompt injection
```

---

# 25.6 UI QA

Verify:

```text
Microphone
Recording
Transcript
Answer
Sources
Latency
Errors
Responsive layout
```

---

## Phase 21 Acceptance Criteria

```text
[ ] All functional tests pass
[ ] Security check passes
[ ] Performance check completed
[ ] Retrieval check completed
[ ] Grounding check completed
[ ] UI check completed
[ ] Live demo tested
```

---

# 26. Submission Preparation

After final QA, prepare:

```text
GitHub Repository
Live URL
Team/Process Video
Demo Video
Submission Form
```

---

# 27. Repository Cleanup

Before submission:

Remove:

```text
debug files
temporary scripts
large unused files
secrets
unused dependencies
unused imports
test credentials
local configuration
```

Keep:

```text
PRD.md
ARCHITECTURE.md
AGENTS.md
IMPLEMENTATION_PLAN.md
README.md
DATASET_ANALYSIS.md
CHUNKING_STRATEGY.md
RETRIEVAL_STRATEGY.md
EVALUATION.md
LATENCY.md
GUARDRAILS.md
```

where relevant.

---

# 28. README Requirements

The README must explain:

## Project

What the system does.

## Architecture

How the system works.

## Dataset

Why MSMARCO-XI is used.

## Pipeline

```text
Voice
 ↓
Sarvam
 ↓
Hybrid RAG
 ↓
Gemini
 ↓
Grounding
```

## Setup

How to run locally.

## Environment Variables

Which variables are required.

## Dataset Ingestion

How to build indexes.

## Development

How to start backend/frontend.

## Testing

How to run tests.

## Evaluation

How benchmarks are run.

## Latency

P50/P70/P100.

## Deployment

How the live system is deployed.

---

# 29. Documentation Completion

Before submission, ensure these files exist where applicable:

```text
PRD.md
ARCHITECTURE.md
AGENTS.md
IMPLEMENTATION_PLAN.md
README.md
DATASET_ANALYSIS.md
CHUNKING_STRATEGY.md
RETRIEVAL_STRATEGY.md
GUARDRAILS.md
EVALUATION.md
LATENCY.md
```

---

# 30. Suggested Development Milestones

The following milestones should be used to track progress.

---

## Milestone 1 — Foundation

```text
[ ] Repository
[ ] Backend
[ ] Frontend
[ ] Environment
[ ] Health endpoint
```

---

## Milestone 2 — Dataset Ready

```text
[ ] Dataset loaded
[ ] Dataset analyzed
[ ] Preprocessing complete
```

---

## Milestone 3 — Index Ready

```text
[ ] Chunking complete
[ ] Embeddings complete
[ ] Qdrant complete
[ ] BM25 complete
```

---

## Milestone 4 — RAG Ready

```text
[ ] Text query
[ ] Retrieval
[ ] Hybrid retrieval
[ ] Reranking
[ ] Gemini
```

---

## Milestone 5 — Reliable RAG

```text
[ ] Harness
[ ] Error handling
[ ] Retry
[ ] Guardrails
[ ] Grounding
```

---

## Milestone 6 — Voice Ready

```text
[ ] Sarvam
[ ] Voice endpoint
[ ] End-to-end voice
```

---

## Milestone 7 — UI Ready

```text
[ ] Voice recorder
[ ] Transcript
[ ] Answer
[ ] Sources
[ ] Latency
```

---

## Milestone 8 — Performance Ready

```text
[ ] Instrumentation
[ ] Benchmark
[ ] P50
[ ] P70
[ ] P100
[ ] Optimization
```

---

## Milestone 9 — Production Ready

```text
[ ] Deployment
[ ] Security
[ ] Monitoring
[ ] Production test
```

---

## Milestone 10 — Submission Ready

```text
[ ] Repository cleaned
[ ] README complete
[ ] Live link working
[ ] Demo video
[ ] Process video
[ ] Social posts prepared
[ ] #RAGInGoa included
```

---

# 31. Recommended Implementation Order Inside Each Phase

Every phase should follow:

```text
1. Understand
      ↓
2. Design
      ↓
3. Implement
      ↓
4. Unit Test
      ↓
5. Integration Test
      ↓
6. Benchmark if applicable
      ↓
7. Document
      ↓
8. Commit
```

---

# 32. Recommended Coding-Agent Workflow

The coding agent should work in small prompts.

Example:

### Prompt 1

```text
Read PRD.md, ARCHITECTURE.md, AGENTS.md and IMPLEMENTATION_PLAN.md.

Implement only Phase 0.

Do not implement dataset ingestion, RAG, voice, Gemini, or frontend features beyond the basic setup.

After implementation, run validation and report what was completed.
```

Then:

### Prompt 2

```text
Phase 0 is complete.

Now implement only Phase 1: Dataset Analysis.

First inspect the actual MSMARCO-XI dataset structure.

Do not assume field names.

Generate DATASET_ANALYSIS.md with real findings.

Do not implement retrieval or frontend features yet.
```

Then:

### Prompt 3

```text
Phase 1 is complete.

Now implement only Phase 2: Data Preprocessing.

Use the actual dataset schema discovered during Phase 1.

Do not modify unrelated components.

Run tests after implementation.
```

Continue phase by phase.

---

# 33. Do Not Ask the Agent to Build Everything

Avoid prompts such as:

```text
Build the entire HH Goa RAG project.
```

Instead use:

```text
Implement Phase 3 only.
```

This reduces:

* Bugs
* Hallucinated architecture
* Unnecessary dependencies
* Broken integrations
* Uncontrolled code generation
* Difficult debugging

---

# 34. Dependency Order

The coding agent must respect dependencies.

```text
Dataset
   ↓
Preprocessing
   ↓
Chunking
   ↓
Embedding
   ↓
Index
   ↓
Retrieval
   ↓
RAG
   ↓
Guardrails
   ↓
Voice
   ↓
Frontend
```

Do not implement a dependent component before its prerequisite works.

---

# 35. Critical Dependency Rules

## Chunking depends on:

```text
Dataset Analysis
```

## Embeddings depend on:

```text
Chunking
```

## Qdrant depends on:

```text
Embeddings
```

## BM25 depends on:

```text
Processed Chunks
```

## RAG depends on:

```text
Retrieval
```

## Guardrails depend on:

```text
Retrieval + Generation
```

## Voice depends on:

```text
Stable Text RAG
```

## Frontend depends on:

```text
Stable API
```

## Latency optimization depends on:

```text
Working complete pipeline
+
Baseline measurements
```

---

# 36. Performance Gate

Do not optimize before the system works correctly.

Required order:

```text
Correctness
 ↓
Retrieval Quality
 ↓
Grounding
 ↓
Measurement
 ↓
Optimization
```

---

# 37. Quality Gate

Before moving to the next major phase:

```text
Tests pass
+
No critical bugs
+
Documentation updated
+
Current feature manually verified
```

---

# 38. Production Gate

Before deployment:

```text
All tests
+
Security review
+
Latency benchmark
+
Grounding benchmark
+
Failure testing
+
Environment validation
```

---

# 39. Final System Checklist

The final system must contain:

```text
VOICE
✓ Sarvam STT

DATA
✓ MSMARCO-XI
✓ Preprocessing
✓ Multi-strategy chunking

RETRIEVAL
✓ Embeddings
✓ Qdrant
✓ BM25
✓ Hybrid retrieval
✓ RRF
✓ Reranking

GENERATION
✓ Gemini
✓ Structured output

HARNESS
✓ Orchestrator
✓ Retry
✓ Error recovery
✓ Request state

GUARDRAILS
✓ Input validation
✓ Retrieval threshold
✓ Prompt injection protection
✓ Grounding validation
✓ Refusal behavior

PERFORMANCE
✓ Stage latency
✓ P50
✓ P70
✓ P100

FRONTEND
✓ Voice recorder
✓ Transcript
✓ Answer
✓ Sources
✓ Latency

DEPLOYMENT
✓ Backend
✓ Frontend
✓ Qdrant
✓ Environment configuration
```

---

# 40. Final Definition of Done

The project is considered implementation-complete only when:

```text
┌─────────────────────────────────────────┐
│          HH GOA VOICE RAG               │
├─────────────────────────────────────────┤
│                                         │
│  MSMARCO-XI                             │
│       ↓                                 │
│  Multi-Strategy Chunking                │
│       ↓                                 │
│  Embeddings                             │
│       ↓                                 │
│  Qdrant + BM25                          │
│       ↓                                 │
│  Hybrid Retrieval                       │
│       ↓                                 │
│  Reranking                              │
│       ↓                                 │
│  Gemini                                 │
│       ↓                                 │
│  Grounding Validation                   │
│       ↓                                 │
│  Sarvam Voice                           │
│       ↓                                 │
│  FastAPI                                │
│       ↓                                 │
│  React Frontend                         │
│       ↓                                 │
│  Latency Analytics                      │
│                                         │
└─────────────────────────────────────────┘
```

And:

```text
[✓] Dataset grounded
[✓] Voice works
[✓] Retrieval works
[✓] Hybrid retrieval works
[✓] Reranking evaluated
[✓] Gemini works
[✓] Harness implemented
[✓] Guardrails implemented
[✓] Grounding verified
[✓] P50 measured
[✓] P70 measured
[✓] P100 measured
[✓] 100+ queries benchmarked
[✓] Frontend works
[✓] Backend works
[✓] Production deployment works
[✓] Secrets protected
[✓] Documentation complete
[✓] Demo ready
```

---

# 41. Final Instruction

The coding agent must never interpret this implementation plan as permission to skip validation.

The correct development philosophy is:

```text
BUILD
 ↓
TEST
 ↓
MEASURE
 ↓
VALIDATE
 ↓
DOCUMENT
 ↓
OPTIMIZE
```

not:

```text
BUILD EVERYTHING
 ↓
HOPE IT WORKS
```

The final goal is not to produce the largest codebase.

The final goal is to produce the **best-performing, most grounded, reliable, measurable, and demonstrable voice-enabled RAG system possible within the HH Goa 2026 challenge requirements.**
