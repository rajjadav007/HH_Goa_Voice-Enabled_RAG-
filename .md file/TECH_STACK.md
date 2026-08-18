# TECH_STACK.md

# HH Goa 2026 — Voice-Enabled RAG System

## 1. Project Overview

This document defines the technology stack for the HH Goa 2026 Shortlisting Task 2 project.

The system is a voice-enabled Retrieval-Augmented Generation (RAG) application built around the provided AI4Bharat MSMARCO-XI dataset.

The primary pipeline is:

User Voice
↓
Sarvam Speech-to-Text
↓
Query Processing
↓
Hybrid Retrieval
↓
Qdrant + BM25
↓
Reranking
↓
Retrieved Dataset Context
↓
Gemini
↓
Grounding Validation
↓
Final Answer

---

# 2. Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React | User interface |
| Frontend Language | TypeScript | Type-safe frontend development |
| Frontend Build Tool | Vite | Fast development/build |
| UI Styling | Tailwind CSS | Styling |
| UI Components | shadcn/ui | Reusable UI components |
| Backend | Python | AI/RAG backend |
| API Framework | FastAPI | Backend API |
| Validation | Pydantic | Request/response validation |
| Dataset | AI4Bharat MSMARCO-XI | Primary knowledge source |
| Dataset Processing | Hugging Face Datasets | Dataset loading/processing |
| Speech-to-Text | Sarvam AI | Voice transcription |
| LLM | Gemini | Answer generation |
| Vector Database | Qdrant | Semantic vector retrieval |
| Keyword Retrieval | BM25 | Lexical retrieval |
| Embeddings | Local embedding model | Text/vector representation |
| Reranking | Local lightweight reranker | Candidate ranking |
| Orchestration | Custom Python RAG Orchestrator | Pipeline coordination |
| Testing | Pytest | Backend testing |
| Frontend Testing | Vitest / React Testing Library | Frontend testing |
| Containerization | Docker | Deployment consistency |
| Version Control | Git | Source control |
| Repository | GitHub | Code hosting |
| Configuration | Environment Variables | Secrets/configuration |

---

# 3. Frontend Stack

## 3.1 React

### Technology

React

### Purpose

React is used to build the user-facing voice RAG interface.

The frontend is responsible for:

- Voice recording
- Recording status
- Sending audio to backend
- Displaying transcription
- Displaying answer
- Displaying retrieved sources
- Displaying latency
- Displaying errors
- Showing system status

The frontend must not contain private API credentials.

---

# 4. TypeScript

## Technology

TypeScript

## Purpose

TypeScript provides type safety for the frontend.

Use TypeScript for:

- API response types
- Component props
- Application state
- RAG response structures
- Error structures

Avoid unnecessary use of:

```ts
any