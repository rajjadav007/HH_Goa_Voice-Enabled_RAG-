"""Batch chunk processor for streaming processed documents and producing validated chunks."""

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from ingestion.chunking.base import BaseChunker, validate_chunk
from ingestion.chunking.models import Chunk, ChunkingConfig
from ingestion.chunking.registry import ChunkerRegistry
from ingestion.preprocessor import ProcessedDocument

logger = logging.getLogger(__name__)


class BatchChunkProcessor:
    """Batch processor to execute chunking on ProcessedDocument streams."""

    def __init__(self, chunker: Optional[BaseChunker] = None):
        self.chunker = chunker or ChunkerRegistry.get("passthrough")

    def process_documents(
        self, documents: Iterable[ProcessedDocument]
    ) -> List[Chunk]:
        """Process an iterable stream of ProcessedDocument objects into Chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunker.process(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def process_jsonl_file(
        self,
        input_documents_jsonl: str,
        output_chunks_jsonl: str,
        output_manifest_json: Optional[str] = None,
        max_documents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stream ProcessedDocuments from a JSONL file, chunk them, and write JSONL output."""
        if not os.path.exists(input_documents_jsonl):
            raise FileNotFoundError(f"Input documents JSONL not found: {input_documents_jsonl}")

        os.makedirs(os.path.dirname(os.path.abspath(output_chunks_jsonl)), exist_ok=True)

        input_doc_count = 0
        total_chunks_count = 0
        total_tokens = 0
        total_chars = 0
        rejected_chunks_count = 0

        strategy_name = self.chunker.name

        with open(input_documents_jsonl, "r", encoding="utf-8") as in_f, open(
            output_chunks_jsonl, "w", encoding="utf-8"
        ) as out_f:

            for line in in_f:
                if max_documents is not None and input_doc_count >= max_documents:
                    break
                line = line.strip()
                if not line:
                    continue

                input_doc_count += 1
                doc_dict = json.loads(line)

                # Reconstruct ProcessedDocument object
                doc = ProcessedDocument(
                    document_id=doc_dict["document_id"],
                    text=doc_dict["text"],
                    english_text=doc_dict.get("english_text"),
                    source_query_id=doc_dict["source_query_id"],
                    passage_index=doc_dict["passage_index"],
                    is_selected=doc_dict["is_selected"],
                    language=doc_dict["language"],
                    metadata=doc_dict.get("metadata", {}),
                )

                try:
                    chunks = self.chunker.process(doc)
                    for c in chunks:
                        out_f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                        total_chunks_count += 1
                        total_tokens += c.token_count
                        total_chars += c.character_count
                except Exception as exc:
                    rejected_chunks_count += 1
                    logger.warning(
                        f"Chunking failed for document '{doc.document_id}': {exc}"
                    )

        manifest = {
            "chunk_strategy": strategy_name,
            "input_documents_file": input_documents_jsonl,
            "output_chunks_file": output_chunks_jsonl,
            "input_document_count": input_doc_count,
            "output_chunk_count": total_chunks_count,
            "rejected_document_count": rejected_chunks_count,
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "avg_tokens_per_chunk": round(total_tokens / max(1, total_chunks_count), 2),
            "avg_chars_per_chunk": round(total_chars / max(1, total_chunks_count), 2),
            "config": asdict(self.chunker.config),
        }

        if output_manifest_json:
            os.makedirs(os.path.dirname(os.path.abspath(output_manifest_json)), exist_ok=True)
            with open(output_manifest_json, "w", encoding="utf-8") as mf_f:
                json.dump(manifest, mf_f, indent=2, ensure_ascii=False)

        logger.info(
            f"Chunking complete [{strategy_name}]. Documents: {input_doc_count}, Chunks: {total_chunks_count}"
        )
        return manifest
