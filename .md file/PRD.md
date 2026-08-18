# Product Requirements Document (PRD)

## HH Goa 2026 Shortlisting Task 2 — Voice-Enabled RAG System

**Project Name:** VoiceRAG — HH Goa 2026
**Challenge:** HH Goa 2026 Shortlisting Task 2
**Challenge Type:** Voice-Enabled Retrieval-Augmented Generation (RAG)
**Status:** Development
**Primary Dataset:** AI4Bharat MSMARCO-XI
**Speech-to-Text:** Sarvam AI
**Primary LLM:** Gemini
**Vector Database:** Qdrant
**Backend:** Python + FastAPI
**Frontend:** React + Vite + TypeScript
**Target Latency:** Under 200 ms for the defined online RAG pipeline
**Deadline:** August 22, 2026, 11:59 PM

---

# 1. Product Overview

## 1.1 Purpose

Build a production-oriented, voice-enabled Retrieval-Augmented Generation system for the HH Goa 2026 Shortlisting Task 2.

The system allows a user to ask a question using their voice. The system must:

1. Capture the user's voice.
2. Convert the voice into text using Sarvam AI.
3. Validate and preprocess the query.
4. Retrieve relevant information from the provided MSMARCO-XI dataset.
5. Use multiple retrieval techniques rather than relying only on naive vector search.
6. Provide the retrieved context to an LLM.
7. Generate an answer grounded strictly in the retrieved dataset context.
8. Detect unsupported, irrelevant, unsafe, or unanswerable questions.
9. Measure and report latency.
10. Provide P50, P70, and P100 latency metrics.
11. Operate through a structured orchestration/harness rather than a single prompt-to-answer call.

The product should demonstrate strong AI engineering rather than simply connecting a speech API to an LLM.

---

# 2. Challenge Requirements

The implementation MUST satisfy the following requirements from the HH Goa 2026 task.

## 2.1 Dataset

The provided dataset must be used as the primary knowledge source:

AI4Bharat MSMARCO-XI:

https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

The system must not replace the provided dataset with a custom unrelated dataset.

Any preprocessing, chunking, indexing, retrieval, and answer generation must ultimately operate over the provided dataset.

---

## 2.2 Speech-to-Text

The system must use one of the challenge-approved speech-to-text providers:

* Sarvam
* ElevenLabs

### Selected provider

**Sarvam AI**

The system will use Sarvam for voice-to-text conversion.

Architecture:

```text
User Voice
    ↓
Sarvam STT
    ↓
Transcribed Query
```

Speech-to-text must be isolated as its own service/module so it can be replaced later if necessary.

---

## 2.3 Chunking

The system MUST NOT use only one naive fixed-size chunking strategy.

The chunking engine must support multiple strategies and allow them to be evaluated.

Potential strategies include:

* Fixed-size chunking
* Sentence-based chunking
* Paragraph/structure-aware chunking
* Overlap-aware chunking
* Semantic chunking
* Metadata-aware chunking
* Hybrid/adaptive chunking

The actual strategies implemented must be based on the real structure of MSMARCO-XI.

The system must not implement unnecessary chunking methods simply for appearance. Each strategy should have a measurable purpose.

The system must provide a way to compare strategies using retrieval and performance metrics.

---

# 3. Problem Statement

Traditional voice assistants typically follow:

```text
Voice
 ↓
Speech-to-Text
 ↓
LLM
 ↓
Answer
```

This architecture does not guarantee that the answer comes from the provided dataset.

It can result in:

* Hallucination
* Unsupported answers
* Irrelevant answers
* Lack of traceability
* Poor dataset grounding
* Difficulty measuring retrieval quality

The proposed system solves this by introducing a retrieval layer:

```text
Voice
 ↓
Speech-to-Text
 ↓
Query Processing
 ↓
Dataset Retrieval
 ↓
Relevant Context
 ↓
LLM
 ↓
Grounded Answer
```

The system's source of truth is the MSMARCO-XI dataset.

---

# 4. Product Vision

Build a fast and reliable voice-first RAG system where:

> A user can speak a question and receive an answer that is generated from relevant information retrieved from the provided dataset, with measurable retrieval quality, latency, and grounding guarantees.

The product should demonstrate:

* Strong retrieval
* Advanced chunking
* Fast inference
* Robust orchestration
* Grounded generation
* Error recovery
* Guardrails
* Benchmarking
* Production-quality engineering

---

# 5. Core Product Principle

## Dataset First

The MSMARCO-XI dataset is the knowledge source.

Gemini must NOT be treated as the primary knowledge source.

Correct:

```text
Question
   ↓
Retrieve MSMARCO-XI context
   ↓
Gemini
   ↓
Answer based on context
```

Incorrect:

```text
Question
   ↓
Gemini
   ↓
General knowledge answer
```

The system must be designed so that the LLM cannot freely invent information when relevant dataset context is unavailable.

---

# 6. Target Users

## Primary User

A hackathon evaluator who wants to test the system by asking natural-language questions using voice.

## Secondary Users

* Developers
* Technical reviewers
* Hackathon judges
* RAG/AI researchers
* Team members testing retrieval quality

---

# 7. User Experience

## 7.1 Main Flow

The user opens the application.

They see a microphone interface.

The user presses the microphone button and asks a question.

Example:

```text
"What is ...?"
```

The system:

```text
Voice
 ↓
Sarvam STT
 ↓
Transcription
 ↓
Query Validation
 ↓
RAG Retrieval
 ↓
Context Selection
 ↓
Gemini
 ↓
Grounding Validation
 ↓
Answer
```

The UI displays:

* Transcribed question
* Final answer
* Relevant source/context information
* Confidence/grounding status where appropriate
* Latency information
* Error state when the system cannot answer

---

# 8. Functional Requirements

## FR-001 — Voice Input

The application must provide a microphone-based voice input interface.

The user must be able to:

* Start recording
* Stop recording
* Submit voice input
* See recording status
* Handle microphone permission errors

---

## FR-002 — Speech-to-Text

The backend must send audio to Sarvam's speech-to-text service.

The service must return:

* Transcribed text
* Relevant metadata provided by the API
* Error information when applicable

The system must not expose API credentials to the frontend.

---

## FR-003 — Query Validation

Before retrieval, the query must be validated.

Validation should handle:

* Empty input
* Extremely short input
* Extremely long input
* Invalid input
* Unsupported requests
* Potential prompt injection attempts
* Unsafe or inappropriate queries where applicable

---

## FR-004 — Query Preprocessing

The system should normalize the query before retrieval where useful.

Possible operations:

* Whitespace normalization
* Unicode normalization
* Text cleanup
* Query normalization
* Language normalization where required

Preprocessing must not destroy important query semantics.

---

# 9. Dataset Requirements

## 9.1 Dataset Ingestion

The system must provide a repeatable ingestion pipeline.

Expected pipeline:

```text
MSMARCO-XI
    ↓
Load
    ↓
Schema Validation
    ↓
Cleaning
    ↓
Normalization
    ↓
Deduplication
    ↓
Metadata Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
Indexing
```

---

## 9.2 Dataset Analysis

Before production ingestion, the system must analyze the dataset.

The analysis should identify:

* Number of records
* Dataset splits
* Available fields
* Document structure
* Query structure
* Languages
* Text lengths
* Missing values
* Duplicate records
* Metadata
* Potential retrieval fields

A dataset analysis report must be generated.

Recommended file:

```text
DATASET_ANALYSIS.md
```

---

# 10. Preprocessing Requirements

The preprocessing pipeline must be deterministic and repeatable.

It should support:

* Cleaning
* Normalization
* Duplicate detection
* Invalid-record handling
* Metadata preservation
* Stable IDs

The preprocessing process must not accidentally remove important semantic information.

---

# 11. Chunking Engine

## 11.1 Requirement

The chunking engine must be modular.

Recommended architecture:

```text
ChunkingEngine
    │
    ├── FixedChunker
    ├── SentenceChunker
    ├── StructureAwareChunker
    ├── SemanticChunker
    ├── MetadataAwareChunker
    └── HybridChunker
```

Not every strategy must be used in the final production pipeline.

Each strategy must be evaluated.

---

## 11.2 Chunk Metadata

Every chunk should contain sufficient metadata for retrieval and debugging.

Recommended metadata:

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "source_id": "...",
  "chunk_strategy": "...",
  "chunk_index": 0,
  "text": "...",
  "language": "...",
  "metadata": {}
}
```

Only fields actually supported by the dataset should be populated.

---

## 11.3 Chunk Evaluation

Chunking strategies should be compared using:

* Retrieval relevance
* Context completeness
* Number of chunks
* Average chunk size
* Storage size
* Retrieval latency
* Answer quality

The final chunking strategy must be selected based on measurable results rather than arbitrary preference.

---

# 12. Embedding System

The system must convert dataset chunks into vector embeddings.

Pipeline:

```text
Chunk
 ↓
Embedding Model
 ↓
Vector
 ↓
Vector Database
```

The embedding layer must be abstracted behind a service interface.

Example:

```text
EmbeddingService
    ├── embed_text()
    ├── embed_batch()
    └── get_dimension()
```

The implementation should initially benchmark a fast local embedding model against any required API-based alternative.

The final choice must consider:

* Retrieval quality
* Latency
* Memory usage
* Cost
* Deployment complexity

---

# 13. Vector Database

## Selected Technology

**Qdrant**

Qdrant will store chunk embeddings and metadata.

Expected architecture:

```text
Chunk
 ↓
Embedding
 ↓
Qdrant
```

The system must support:

* Collection creation
* Collection health check
* Batch insertion
* Upsert
* Vector search
* Metadata filtering
* Top-K retrieval
* Stable document/chunk IDs

---

# 14. Keyword Retrieval

The system must also support keyword-based retrieval.

Recommended technology:

**BM25**

Purpose:

* Exact keyword matching
* Entity matching
* Rare-term matching
* Complement semantic search

Architecture:

```text
Query
 ↓
BM25
 ↓
Keyword Candidates
```

---

# 15. Hybrid Retrieval

The primary retrieval architecture should combine semantic and keyword retrieval.

```text
                    Query
                      │
            ┌─────────┴─────────┐
            ↓                   ↓
      Vector Retrieval        BM25
            ↓                   ↓
       Semantic Results    Keyword Results
            └─────────┬─────────┘
                      ↓
                Result Fusion
                      ↓
                  Reranking
                      ↓
                 Top Context
```

The fusion algorithm should be configurable.

A suitable initial approach is Reciprocal Rank Fusion (RRF).

---

# 16. Reranking

The system should support a lightweight reranking stage.

Purpose:

* Improve relevance
* Remove weak candidates
* Select the best context for the LLM

Example:

```text
Vector + BM25
      ↓
20 candidates
      ↓
Reranker
      ↓
Top 3–5
```

The reranker must be benchmarked because an overly expensive reranker could violate the latency requirement.

---

# 17. Retrieval Configuration

The following parameters must be configurable rather than hardcoded:

* Top-K vector results
* Top-K BM25 results
* Fusion weights
* RRF parameters
* Reranker candidate count
* Final context count
* Similarity threshold
* Maximum context length

These should be stored in configuration/environment settings where appropriate.

---

# 18. RAG Orchestration

The system must use a proper orchestration layer.

It must NOT be implemented as a single function such as:

```python
answer = llm(prompt)
```

The orchestrator should manage the complete flow.

Recommended structure:

```text
RAGOrchestrator
    │
    ├── validate_input()
    ├── transcribe()
    ├── preprocess_query()
    ├── embed_query()
    ├── retrieve_vector()
    ├── retrieve_bm25()
    ├── fuse_results()
    ├── rerank()
    ├── validate_context()
    ├── generate_answer()
    ├── verify_grounding()
    ├── retry_if_needed()
    └── build_response()
```

---

# 19. LLM Answer Generation

## Selected Model Provider

**Gemini API**

Gemini will be used as the answer-generation layer.

The LLM must receive:

1. User question
2. Retrieved context
3. System instructions
4. Output schema

The model must be instructed to:

* Answer using retrieved context
* Avoid unsupported claims
* Avoid using external knowledge
* State when the context is insufficient
* Return structured output
* Identify supporting chunks when possible

---

# 20. Structured LLM Output

The answer-generation layer should use structured output.

Example:

```json
{
  "answer": "Generated answer",
  "grounded": true,
  "confidence": 0.92,
  "source_chunk_ids": [
    "chunk_123",
    "chunk_456"
  ]
}
```

The exact schema can be adjusted based on the selected Gemini API capabilities.

The backend must validate the model response before returning it to the frontend.

---

# 21. Guardrails

Guardrails are mandatory.

The system must demonstrate that it knows when NOT to answer.

## 21.1 Input Guardrail

Detect:

* Empty queries
* Invalid input
* Excessive input
* Prompt injection attempts
* Unsafe or inappropriate requests where applicable

---

## 21.2 Retrieval Guardrail

After retrieval, check whether sufficient relevant context exists.

Example:

```text
Similarity < threshold
        ↓
Insufficient context
        ↓
Do not generate unsupported answer
```

The system should return a controlled response such as:

> "I couldn't find enough relevant information in the provided dataset to answer that."

---

## 21.3 Grounding Guardrail

The generated answer must be checked against retrieved context.

The system should detect:

* Unsupported claims
* Missing evidence
* Contradictions
* Context-free answers

If grounding fails:

```text
Grounding failure
       ↓
Retry / regenerate
       ↓
Check again
       ↓
If still invalid
       ↓
Reject answer
```

---

# 22. Prompt Injection Protection

Retrieved dataset content must be treated as untrusted data.

The system must distinguish:

```text
SYSTEM INSTRUCTIONS
        ↓
USER QUERY
        ↓
RETRIEVED DATA
```

Retrieved text must never be allowed to override system instructions.

The model must not follow instructions embedded inside retrieved documents.

---

# 23. Error Handling

The system must implement structured error handling.

Potential failures:

### Sarvam failure

```text
Sarvam unavailable
 ↓
Return controlled error
```

### Embedding failure

```text
Embedding failure
 ↓
Retry
 ↓
Controlled error
```

### Qdrant failure

```text
Qdrant unavailable
 ↓
Retry / fallback
 ↓
Controlled error
```

### Gemini failure

```text
Generation failure
 ↓
Retry where appropriate
 ↓
Controlled error
```

### Grounding failure

```text
Unsupported answer
 ↓
Regenerate
 ↓
Reject if necessary
```

Errors must not expose secrets, stack traces, API keys, or internal infrastructure details to the user.

---

# 24. Retry Policy

Retries must be controlled.

The system must avoid infinite retries.

Recommended:

* Maximum retry count configurable
* Exponential backoff where appropriate
* Retry only transient failures
* Do not retry invalid user input
* Do not repeatedly regenerate hallucinated answers without limits

---

# 25. Latency Requirements

## Primary target

The project must target an online pipeline latency of:

**< 200 ms**

The exact benchmark definition must be documented.

Latency must not be claimed as end-to-end if important stages have been excluded.

---

# 26. Latency Instrumentation

Every request must record component-level latency.

Minimum measurements:

```text
STT latency
Query preprocessing latency
Embedding latency
Vector retrieval latency
BM25 latency
Fusion latency
Reranking latency
LLM generation latency
Grounding validation latency
Total latency
```

Example internal telemetry:

```json
{
  "request_id": "...",
  "stt_ms": 40,
  "embedding_ms": 10,
  "vector_search_ms": 8,
  "bm25_ms": 4,
  "rerank_ms": 10,
  "generation_ms": 50,
  "grounding_ms": 5,
  "total_ms": 127
}
```

---

# 27. Latency Analytics

The system must benchmark a reasonable number of test queries.

The benchmark must calculate:

* P50
* P70
* P100

The benchmark must not be based on a single best-case query.

Recommended initial benchmark:

**100+ representative queries**

The exact number can increase depending on available resources.

Results must be reproducible.

---

# 28. Retrieval Evaluation

The system should measure retrieval quality.

Recommended metrics:

* Recall@K
* Precision@K where ground truth is available
* MRR
* Context relevance
* Retrieval latency

The evaluation system should support a known test-query set.

---

# 29. Answer Evaluation

Where ground-truth answers or suitable evaluation data are available, evaluate:

* Answer correctness
* Answer relevance
* Groundedness
* Citation/source correctness
* Hallucination rate

If exact ground truth is unavailable, clearly document the evaluation methodology.

---

# 30. Benchmarking Chunking Strategies

The evaluation framework must allow comparison between chunking approaches.

Example:

```text
Strategy
    ↓
Index
    ↓
Evaluation queries
    ↓
Retrieval metrics
    ↓
Latency
    ↓
Answer quality
```

The selected final strategy must be supported by benchmark results.

---

# 31. Frontend Requirements

## Technology

React + Vite + TypeScript.

The frontend should be lightweight and focused on the core demonstration.

---

## Main UI

The application should include:

### Voice control

* Start recording
* Stop recording
* Recording state
* Microphone permission handling

### Transcription

Display the text returned from speech-to-text.

### Answer

Display the generated answer clearly.

### Sources

Display relevant retrieved context/source identifiers where appropriate.

### Metrics

Display useful latency information.

Example:

```text
Transcription
"What is ...?"

Answer
"Based on the retrieved information..."

Sources
3 relevant chunks

Latency
147 ms
```

---

# 32. Backend Requirements

## Technology

Python + FastAPI.

The backend must separate:

* API layer
* Orchestration
* Retrieval
* Chunking
* Embeddings
* LLM
* STT
* Guardrails
* Analytics
* Configuration

Recommended architecture:

```text
backend/
└── app/
    ├── api/
    ├── core/
    ├── models/
    ├── services/
    ├── orchestration/
    ├── retrieval/
    ├── chunking/
    ├── embeddings/
    ├── guardrails/
    ├── analytics/
    └── utils/
```

---

# 33. API Requirements

The backend should provide APIs such as:

```text
GET  /api/health
POST /api/query
POST /api/voice/query
GET  /api/metrics
```

Additional internal/admin endpoints may be added if necessary.

---

# 34. Query API

A text query endpoint should support testing without voice.

Example:

```http
POST /api/query
```

Request:

```json
{
  "query": "User question"
}
```

Response:

```json
{
  "answer": "...",
  "grounded": true,
  "sources": [],
  "latency_ms": 120
}
```

This endpoint is important because it allows RAG testing independently of the voice interface.

---

# 35. Voice Query API

Example:

```http
POST /api/voice/query
```

Input:

* Audio file or supported audio payload

Processing:

```text
Audio
 ↓
Sarvam
 ↓
Text
 ↓
RAG
 ↓
Answer
```

Response should contain:

```json
{
  "transcript": "...",
  "answer": "...",
  "grounded": true,
  "sources": [],
  "latency_ms": 150
}
```

---

# 36. Health Monitoring

The backend must expose health information for:

* API
* Qdrant
* Embedding service
* Gemini
* Sarvam where practical

Example:

```text
API       ✓
Qdrant    ✓
Embedding ✓
Gemini    ✓
Sarvam    ✓
```

---

# 37. Configuration Management

All credentials and sensitive configuration must come from environment variables.

Never hardcode:

* API keys
* Tokens
* Passwords
* Database credentials
* Private URLs
* Secrets

Required configuration should be represented in:

```text
.env.example
```

Example categories:

```text
SARVAM_API_KEY=
GEMINI_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
EMBEDDING_MODEL=
LLM_MODEL=
```

Actual `.env` files must not be committed to Git.

---

# 38. Security Requirements

The system must:

* Never expose API keys to the frontend
* Validate uploaded audio
* Limit audio size
* Validate MIME types
* Limit request size
* Sanitize user input
* Protect internal errors
* Protect secrets
* Treat retrieved content as untrusted
* Protect against prompt injection
* Avoid logging sensitive user data unnecessarily

---

# 39. Logging

The backend must provide structured logs.

Each request should have a request ID.

Example:

```text
request_id
timestamp
stage
latency
status
error_type
```

Logs should not contain API keys or sensitive credentials.

---

# 40. Caching

Caching should be considered for latency optimization.

Potential caching targets:

* Frequently repeated queries
* Query embeddings
* Retrieval results
* Configuration
* Static dataset metadata

Caching must not compromise correctness.

A cache should only be introduced where benchmarking demonstrates meaningful benefit.

---

# 41. Offline vs Online Architecture

## Offline pipeline

Expensive operations should happen before user queries whenever possible.

```text
MSMARCO-XI
 ↓
Preprocessing
 ↓
Chunking
 ↓
Embedding
 ↓
Qdrant
 ↓
BM25 index
```

## Online pipeline

Only latency-sensitive operations should happen during a user request.

```text
Voice
 ↓
STT
 ↓
Query processing
 ↓
Embedding
 ↓
Hybrid retrieval
 ↓
Reranking
 ↓
LLM
 ↓
Grounding
 ↓
Answer
```

---

# 42. Performance Principles

The implementation must prioritize:

1. Precompute expensive operations.
2. Batch embedding operations during ingestion.
3. Keep vector search local/low latency where possible.
4. Avoid unnecessary external API calls.
5. Use lightweight reranking.
6. Keep retrieved context small and relevant.
7. Cache repeated operations where useful.
8. Measure every stage.
9. Optimize based on real benchmark data.
10. Never sacrifice grounding merely to reduce latency.

---

# 43. Production Architecture

High-level architecture:

```text
                         USER
                          │
                          ▼
                    React Frontend
                          │
                          ▼
                     FastAPI API
                          │
                          ▼
                  RAG Orchestrator
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
        Sarvam        Retrieval      Guardrails
            │             │             │
            │       ┌─────┴─────┐       │
            │       ▼           ▼       │
            │    Qdrant        BM25     │
            │       │           │       │
            │       └─────┬─────┘       │
            │             ▼             │
            │         Reranker          │
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                     Gemini LLM
                          │
                          ▼
                  Grounding Validator
                          │
                          ▼
                    Final Response
```

---

# 44. Repository Structure

The final repository should approximately follow:

```text
hh-goa-voice-rag/
│
├── README.md
├── PRD.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
├── AGENTS.md
├── DATASET_ANALYSIS.md
├── EVALUATION.md
├── CHUNKING_STRATEGY.md
├── RETRIEVAL_STRATEGY.md
├── LATENCY.md
├── GUARDRAILS.md
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   ├── orchestration/
│   │   ├── retrieval/
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── guardrails/
│   │   └── analytics/
│   ├── tests/
│   └── requirements.txt
│
├── ingestion/
│   ├── load_dataset.py
│   ├── preprocess.py
│   ├── metadata.py
│   ├── chunking/
│   ├── embeddings.py
│   └── index.py
│
├── evaluation/
│   ├── test_queries.json
│   ├── benchmark.py
│   ├── latency.py
│   ├── retrieval_eval.py
│   └── answer_eval.py
│
├── scripts/
│
├── tests/
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── Dockerfile
```

The exact structure may evolve during implementation, but architectural boundaries must remain clear.

---

# 45. Development Phases

The system MUST be implemented incrementally.

## Phase 1 — Project Setup

Tasks:

* Repository setup
* Python environment
* React environment
* Configuration system
* Documentation
* Logging
* Testing infrastructure

---

## Phase 2 — Dataset Analysis

Tasks:

* Download/load MSMARCO-XI
* Inspect schema
* Analyze records
* Analyze text lengths
* Analyze language
* Identify metadata
* Identify duplicates/missing data

Deliverable:

```text
DATASET_ANALYSIS.md
```

---

## Phase 3 — Data Preprocessing

Tasks:

* Cleaning
* Normalization
* Deduplication
* Metadata extraction
* Stable IDs

---

## Phase 4 — Chunking Engine

Tasks:

* Implement multiple strategies
* Create configurable chunking
* Generate chunk metadata
* Benchmark chunking strategies

---

## Phase 5 — Embeddings

Tasks:

* Select embedding model
* Implement embedding service
* Batch embedding
* Benchmark embedding latency
* Validate vector dimensions

---

## Phase 6 — Vector Index

Tasks:

* Deploy/configure Qdrant
* Create collection
* Batch upload
* Metadata indexing
* Search testing

---

## Phase 7 — Basic RAG

Tasks:

* Query embedding
* Vector search
* Context selection
* Basic answer generation
* Text-only query endpoint

Milestone:

```text
Text Query
 ↓
Retrieved Context
 ↓
Answer
```

---

## Phase 8 — Hybrid Retrieval

Tasks:

* BM25
* Vector retrieval
* Result fusion
* Reranking
* Retrieval evaluation

---

## Phase 9 — Harness

Tasks:

* Orchestrator
* Structured input/output
* Retry logic
* Error handling
* Stage-level logging
* Request IDs

---

## Phase 10 — Guardrails

Tasks:

* Input validation
* Relevance threshold
* Prompt injection protection
* Grounding validation
* Regeneration/rejection logic

---

## Phase 11 — Voice

Tasks:

* Sarvam integration
* Audio handling
* Transcription
* Voice query endpoint
* Error handling

---

## Phase 12 — Frontend

Tasks:

* Microphone UI
* Recording
* API integration
* Transcript display
* Answer display
* Sources
* Latency display
* Error states

---

## Phase 13 — Evaluation

Tasks:

* Build representative test set
* Retrieval evaluation
* Answer evaluation
* Grounding evaluation
* Failure testing

---

## Phase 14 — Latency Optimization

Tasks:

* Profile every stage
* Remove unnecessary network calls
* Optimize embedding
* Optimize retrieval
* Optimize reranking
* Optimize generation
* Add caching where useful
* Measure P50/P70/P100

---

## Phase 15 — Deployment

Tasks:

* Dockerize backend
* Deploy Qdrant
* Deploy backend
* Deploy frontend
* Configure environment variables
* Test production system

---

## Phase 16 — Final QA

Verify:

* Voice works
* Dataset retrieval works
* Multiple chunking strategies work
* Hybrid retrieval works
* Reranking works
* Gemini generation works
* Guardrails work
* Grounding works
* Latency metrics are real
* P50/P70/P100 are documented
* Live URL works
* Repository is clean
* Secrets are removed
* README is complete

---

# 46. Acceptance Criteria

The project is considered complete only when all of the following are true.

## Dataset

* [ ] MSMARCO-XI is used as the primary knowledge source.
* [ ] Dataset ingestion is reproducible.
* [ ] Dataset structure has been analyzed.
* [ ] Preprocessing is implemented.

## Chunking

* [ ] Multiple chunking strategies exist.
* [ ] Chunk strategies are configurable.
* [ ] Chunking strategies are benchmarked.
* [ ] Final strategy is selected based on evidence.

## Retrieval

* [ ] Vector retrieval works.
* [ ] Qdrant works.
* [ ] BM25 works.
* [ ] Hybrid retrieval works.
* [ ] Reranking works.
* [ ] Retrieval quality is evaluated.

## Generation

* [ ] Gemini is integrated.
* [ ] Gemini receives retrieved context.
* [ ] Structured output is validated.
* [ ] Answers are grounded in retrieved context.

## Voice

* [ ] Sarvam STT works.
* [ ] Voice query works end-to-end.
* [ ] Transcription is shown to the user.

## Harness

* [ ] Structured orchestration exists.
* [ ] Errors are handled.
* [ ] Retries are controlled.
* [ ] Structured responses are used.
* [ ] Stage-level logging exists.

## Guardrails

* [ ] Off-topic queries are handled.
* [ ] Insufficient context is handled.
* [ ] Prompt injection is addressed.
* [ ] Unsupported answers are rejected or regenerated.
* [ ] Grounding is checked.

## Performance

* [ ] Latency is instrumented.
* [ ] P50 is calculated.
* [ ] P70 is calculated.
* [ ] P100 is calculated.
* [ ] Benchmark uses multiple queries.
* [ ] The defined latency target is tested honestly.

## Security

* [ ] API keys are never hardcoded.
* [ ] Secrets are stored in environment variables.
* [ ] `.env` is ignored by Git.
* [ ] `.env.example` contains required variable names.
* [ ] Frontend never receives private API keys.

## Deployment

* [ ] Backend is deployed.
* [ ] Frontend is deployed.
* [ ] Vector database is available.
* [ ] Production environment variables are configured.
* [ ] Live end-to-end test succeeds.

---

# 47. Non-Functional Requirements

## Performance

The system should minimize latency without compromising retrieval quality or grounding.

## Reliability

Temporary API failures should not crash the application.

## Maintainability

Components must be modular and independently testable.

## Scalability

The ingestion and retrieval architecture should support a larger dataset without requiring a complete rewrite.

## Security

No secret or credential may be hardcoded.

## Observability

The system must provide sufficient logs and metrics to identify latency and failure points.

## Reproducibility

Dataset ingestion, indexing, and evaluation should be reproducible.

---

# 48. Out of Scope

The following should not be prioritized unless required for the core system:

* Generic personal assistant features
* General web search
* Open-domain chatbot behavior
* Training a large language model from scratch
* Fine-tuning a large LLM without measurable benefit
* Unnecessary multi-agent complexity
* Large social features
* User accounts unless required
* Unrelated datasets
* Excessive frontend animations
* Features that increase latency without improving evaluation results

The primary goal is a high-quality, fast, grounded voice RAG system.

---

# 49. Design Principles

## Principle 1 — Retrieval First

The quality of the answer depends on retrieving the right context.

## Principle 2 — Dataset Is the Source of Truth

The LLM must not replace the dataset.

## Principle 3 — Measure Before Optimizing

Do not assume a component is slow. Measure it.

## Principle 4 — Local Where Practical

Avoid unnecessary external APIs when local processing can achieve the same result.

## Principle 5 — Ground Everything

No evidence means no confident answer.

## Principle 6 — Modular Architecture

Every major component should have a replaceable interface.

## Principle 7 — Build Incrementally

Do not implement the entire system in one coding-agent prompt.

---

# 50. Recommended Technology Stack

| Component         | Technology                           |
| ----------------- | ------------------------------------ |
| Frontend          | React                                |
| Build Tool        | Vite                                 |
| Language          | TypeScript                           |
| Styling           | Tailwind CSS                         |
| UI Components     | shadcn/ui                            |
| Backend           | Python                               |
| API Framework     | FastAPI                              |
| STT               | Sarvam AI                            |
| LLM               | Gemini                               |
| Vector DB         | Qdrant                               |
| Keyword Retrieval | BM25                                 |
| Embeddings        | Fast local embedding model initially |
| Reranking         | Lightweight local reranker           |
| Analytics         | Python                               |
| Testing           | Pytest + frontend testing framework  |
| Containerization  | Docker                               |
| Version Control   | Git + GitHub                         |

Technology choices may be changed only after benchmarking or a documented technical reason.

---

# 51. Environment Variables

The project must use environment variables for all secrets and configurable infrastructure.

Example:

```text
SARVAM_API_KEY=
GEMINI_API_KEY=

QDRANT_URL=
QDRANT_API_KEY=

EMBEDDING_MODEL=
RERANKER_MODEL=
LLM_MODEL=

LOG_LEVEL=
ENVIRONMENT=
```

The actual values must never be committed.

---

# 52. Git Requirements

Use Git throughout development.

Recommended workflow:

```text
main
 │
 ├── feature/dataset-analysis
 ├── feature/chunking
 ├── feature/embeddings
 ├── feature/retrieval
 ├── feature/rag-generation
 ├── feature/guardrails
 ├── feature/voice
 └── feature/frontend
```

Every major milestone should have a meaningful commit.

---

# 53. Testing Strategy

Testing must happen continuously.

## Unit Tests

Test:

* Chunking
* Preprocessing
* Embeddings interface
* Retrieval
* Reranking
* Guardrails
* Response validation

## Integration Tests

Test:

```text
Query
 ↓
Embedding
 ↓
Qdrant
 ↓
Retrieval
 ↓
Gemini
```

## End-to-End Tests

Test:

```text
Voice
 ↓
Sarvam
 ↓
RAG
 ↓
Gemini
 ↓
Grounding
 ↓
Answer
```

## Failure Tests

Test:

* API unavailable
* Empty query
* No relevant context
* Invalid audio
* LLM failure
* Qdrant failure
* Grounding failure
* Prompt injection

---

# 54. Evaluation Dashboard

The application may expose a developer/evaluation view showing:

```text
Dataset
────────────────
Documents: ...
Chunks: ...
Index: Ready

Retrieval
────────────────
Vector: ...
BM25: ...
Reranker: ...

Latency
────────────────
P50: ...
P70: ...
P100: ...

Grounding
────────────────
Grounded: ...
Rejected: ...
Regenerated: ...
```

This should be optional for the public user interface.

---

# 55. Submission Requirements

The final project must support the HH Goa submission requirements:

* GitHub repository link
* Live working link
* Team/process video
* Demo video
* Submission form

The two videos must be posted to:

* Instagram
* X
* LinkedIn

Every team member must post both videos individually.

Every required post must contain:

```text
#RAGInGoa
```

At least one Instagram account must be public.

---

# 56. Demo Requirements

The final demo should demonstrate the complete pipeline.

Recommended demo flow:

```text
1. Open application
2. Press microphone
3. Ask a question
4. Show transcription
5. Show retrieval
6. Show grounded answer
7. Show source/context
8. Show latency
9. Demonstrate an unanswerable/off-topic query
10. Show guardrail response
```

The demo must be real and live rather than a pre-recorded fake interaction.

---

# 57. Success Definition

The project succeeds when a judge can:

1. Open the live application.
2. Speak a natural-language question.
3. See accurate transcription.
4. Observe retrieval from MSMARCO-XI.
5. Receive a useful answer.
6. Verify that the answer is grounded in retrieved context.
7. See that the system refuses to hallucinate when information is unavailable.
8. Observe measurable latency.
9. Understand that multiple chunking and retrieval strategies were considered.
10. Inspect the GitHub repository and understand the architecture.

---

# 58. Final Product Definition

The final product is NOT:

```text
Voice → Gemini → Answer
```

The final product IS:

```text
                       VOICE RAG

User Voice
    ↓
Sarvam STT
    ↓
Query Validation
    ↓
Query Processing
    ↓
Hybrid Retrieval
    ├── Vector Search
    └── BM25
    ↓
Result Fusion
    ↓
Reranking
    ↓
Relevant MSMARCO-XI Context
    ↓
Context Validation
    ↓
Gemini
    ↓
Grounding Validation
    ↓
Final Dataset-Grounded Answer
    ↓
Latency + Evaluation Metrics
```

---

# 59. Implementation Priority

The implementation order MUST be:

```text
1. Project Setup
       ↓
2. Dataset Analysis
       ↓
3. Data Preprocessing
       ↓
4. Multi-Strategy Chunking
       ↓
5. Embeddings
       ↓
6. Qdrant Index
       ↓
7. Basic Text RAG
       ↓
8. Hybrid Retrieval
       ↓
9. Reranking
       ↓
10. Gemini Generation
       ↓
11. Harness
       ↓
12. Guardrails
       ↓
13. Retrieval/Answer Evaluation
       ↓
14. Sarvam Voice
       ↓
15. FastAPI Production API
       ↓
16. React Frontend
       ↓
17. Latency Optimization
       ↓
18. P50/P70/P100 Benchmark
       ↓
19. Deployment
       ↓
20. Final QA
       ↓
21. Demo + Submission
```

This order should not be skipped unless a technical dependency requires it.

---

# 60. Definition of Done

The project is DONE only when:

```text
[✓] MSMARCO-XI integrated
[✓] Dataset analyzed
[✓] Dataset preprocessing implemented
[✓] Multiple chunking strategies implemented
[✓] Chunking benchmark completed
[✓] Embedding pipeline implemented
[✓] Qdrant indexed
[✓] BM25 implemented
[✓] Hybrid retrieval implemented
[✓] Reranking implemented
[✓] Gemini integrated
[✓] Structured output implemented
[✓] RAG harness implemented
[✓] Retry/error handling implemented
[✓] Input guardrails implemented
[✓] Retrieval guardrails implemented
[✓] Grounding guardrails implemented
[✓] Sarvam STT integrated
[✓] Voice query works
[✓] FastAPI backend works
[✓] React frontend works
[✓] Latency instrumentation implemented
[✓] P50 calculated
[✓] P70 calculated
[✓] P100 calculated
[✓] Retrieval evaluation completed
[✓] Grounding evaluation completed
[✓] Production deployment completed
[✓] Secrets removed from repository
[✓] README completed
[✓] Live link tested
[✓] GitHub repository cleaned
[✓] Demo video ready
[✓] Process video ready
[✓] Submission requirements verified
```

---

# 61. Vibe-Coding Rules

This project will be developed using AI-assisted/vibe coding.

The coding agent MUST:

1. Read `PRD.md` before implementation.
2. Read `ARCHITECTURE.md` before changing architecture.
3. Read `AGENTS.md` before modifying project files.
4. Implement one milestone at a time.
5. Never implement the entire application in one step.
6. Run tests after every significant change.
7. Explain architectural changes before making them.
8. Avoid unnecessary dependencies.
9. Never hardcode secrets.
10. Keep backend and frontend separated.
11. Keep offline ingestion separate from online query processing.
12. Keep retrieval modular.
13. Keep model providers replaceable.
14. Add latency instrumentation to performance-critical components.
15. Never remove guardrails simply to improve benchmark numbers.
16. Never fabricate evaluation results.
17. Never claim the system meets the 200 ms target without actual measurements.
18. Preserve working functionality while adding new features.
19. Prefer measurable engineering decisions over assumptions.
20. Treat the provided dataset as the primary source of truth.

---

# 62. Primary Engineering Goal

The primary engineering goal is:

> Build the fastest, most reliable, well-grounded voice-enabled RAG system possible using the provided MSMARCO-XI dataset while demonstrating advanced chunking, strong retrieval, structured orchestration, guardrails, and real latency measurements.

The project should optimize for:

```text
                    QUALITY
                       +
                    GROUNDING
                       +
                    RETRIEVAL
                       +
                    LATENCY
                       +
                   RELIABILITY
```

rather than simply maximizing the number of technologies used.

---

# 63. Final Guiding Architecture

```text
                         ┌─────────────────────┐
                         │    MSMARCO-XI       │
                         │   Source Dataset    │
                         └──────────┬──────────┘
                                    │
                             OFFLINE PIPELINE
                                    │
                                    ▼
                           Data Preprocessing
                                    │
                                    ▼
                        Multi-Strategy Chunking
                                    │
                                    ▼
                              Embeddings
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                       Qdrant                 BM25
                    Vector Index          Keyword Index
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    │
                         ONLINE QUERY PIPELINE
                                    │
User ──────── Voice ────────────────┤
                                    ▼
                              Sarvam STT
                                    │
                                    ▼
                              Text Query
                                    │
                                    ▼
                           Input Guardrail
                                    │
                                    ▼
                         Hybrid Retrieval
                                    │
                                    ▼
                              Reranking
                                    │
                                    ▼
                           Context Validation
                                    │
                                    ▼
                             Gemini LLM
                                    │
                                    ▼
                         Grounding Validation
                                    │
                           ┌────────┴────────┐
                           │                 │
                         PASS              FAIL
                           │                 │
                           ▼                 ▼
                       Answer          Retry / Reject
                           │
                           ▼
                     React Frontend
                           │
                           ▼
                    Latency Analytics
```

**This PRD is the source of truth for the implementation. Any implementation decision should be checked against this document before code is written.**
