# HH Goa 2026 — Voice-Enabled RAG System

Voice-enabled Retrieval-Augmented Generation (RAG) system built for the HH Goa 2026 Shortlisting Task 2.

## Technology Stack

- **Dataset:** AI4Bharat MSMARCO-XI
- **Speech-to-Text:** Sarvam AI
- **LLM:** Gemini
- **Vector Database:** Qdrant
- **Keyword Retrieval:** BM25
- **Backend:** Python + FastAPI
- **Frontend:** React + Vite + TypeScript + Tailwind CSS

## Repository Structure

```text
.
├── backend/            # FastAPI backend application
├── frontend/           # React + Vite + TypeScript frontend application
├── ingestion/          # Offline dataset processing & indexing pipeline
├── evaluation/         # Retrieval, answer, and latency benchmarks
├── scripts/            # Helper scripts and automation tools
├── docs/               # System documentation & specifications
├── tests/              # Top-level integration test suite
├── docker-compose.yml  # Docker multi-container setup
└── Dockerfile          # Backend production container configuration
```

## Quick Start (Development)

### Backend

```bash
cd backend
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend health check: `http://localhost:8000/api/health`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend interface: `http://localhost:5173`
