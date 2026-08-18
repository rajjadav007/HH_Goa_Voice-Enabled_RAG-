"""Dataset Preprocessor Module for AI4Bharat MSMARCO-XI.

Converts raw MSMARCO-XI records into clean, traceable, normalized
ProcessedDocument and ProcessedQuery representations with stable IDs,
ground-truth preservation, and quality filtering ready for chunking.
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
)


@dataclass
class PreprocessingConfig:
    """Configurable options for dataset preprocessing."""

    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    strip_control_chars: bool = True
    min_passage_char_length: int = 10
    deduplicate_passages: bool = True
    batch_size: int = 1000
    processed_dir: str = DEFAULT_PROCESSED_DIR


@dataclass
class ProcessedDocument:
    """Normalized document representation for chunking & retrieval."""

    document_id: str
    text: str
    english_text: Optional[str]
    source_query_id: int
    passage_index: int
    is_selected: int  # Ground-truth relevance (1=relevant, 0=not relevant)
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedQuery:
    """Normalized query representation for RAG evaluation & retrieval."""

    query_id: int
    query_text: str
    eng_query_text: str
    answer_text: Optional[str]
    eng_answer_text: Optional[str]
    query_type: str
    source_lang: str
    target_lang: str
    relevant_document_ids: List[str] = field(default_factory=list)
    all_document_ids: List[str] = field(default_factory=list)


@dataclass
class ProcessingManifest:
    """Machine-readable manifest describing preprocessing output & stats."""

    dataset: str
    preprocessing_version: str
    config: Dict[str, Any]
    input_records: int
    processed_queries: int
    processed_documents: int
    rejected_records: int
    rejected_reasons: Dict[str, int]
    duplicate_passages_deduped: int
    language_distribution: Dict[str, int]
    output_files: Dict[str, str]


def generate_stable_document_id(
    query_id: int, passage_idx: int, text: str, lang: str = "eng"
) -> str:
    """Generate deterministic, stable document ID from query_id, index, and content hash."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"doc_{query_id}_{passage_idx}_{content_hash}"


def normalize_text(
    text: str,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    strip_control_chars: bool = True,
) -> str:
    """Apply safe text normalization without destroying search-critical tokens."""
    if not text:
        return ""

    normalized = text

    if normalize_unicode:
        normalized = unicodedata.normalize("NFC", normalized)

    if strip_control_chars:
        # Strip ASCII control chars except newline (\n) and tab (\t)
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

    if normalize_whitespace:
        # Replace multiple whitespace/newlines with single space
        normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


class MSMARCOPreprocessor:
    """Preprocessing engine for AI4Bharat MSMARCO-XI dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        os.makedirs(self.config.processed_dir, exist_ok=True)

    def process_record(
        self, record: Dict[str, Any]
    ) -> Tuple[Optional[ProcessedQuery], List[ProcessedDocument], Optional[str]]:
        """Process a single raw record into ProcessedQuery and ProcessedDocuments."""
        # Quality check: query_id required
        qid = record.get("query_id")
        if qid is None:
            return None, [], "missing_query_id"

        # Extract & normalize queries
        raw_q = record.get("query") or ""
        raw_eq = record.get("Eng_Query") or ""

        norm_q = normalize_text(
            raw_q,
            self.config.normalize_unicode,
            self.config.normalize_whitespace,
            self.config.strip_control_chars,
        )
        norm_eq = normalize_text(
            raw_eq,
            self.config.normalize_unicode,
            self.config.normalize_whitespace,
            self.config.strip_control_chars,
        )

        if not norm_q and not norm_eq:
            return None, [], "empty_query_text"

        # Answers
        raw_a = record.get("Answer") or ""
        raw_ea = record.get("Eng_Answer") or ""
        norm_a = normalize_text(raw_a) if raw_a else None
        norm_ea = normalize_text(raw_ea) if raw_ea else None

        # Metadata
        source_lang = str(record.get("source_lang", "unknown"))
        target_lang = str(record.get("target_lang", "unknown"))
        query_type = str(record.get("query_type", "unknown"))
        meta_dict = record.get("meta") if isinstance(record.get("meta"), dict) else {}

        # Passages processing
        passages_struct = record.get("passages")
        if not isinstance(passages_struct, dict):
            return None, [], "missing_or_invalid_passages_struct"

        eng_passages = passages_struct.get("English_passages") or []
        trans_passages = passages_struct.get("Translated_passages") or []
        is_selected_list = passages_struct.get("is_selected") or []

        if not eng_passages and not trans_passages:
            return None, [], "empty_passages_list"

        docs: List[ProcessedDocument] = []
        all_doc_ids: List[str] = []
        relevant_doc_ids: List[str] = []

        num_passages = max(len(eng_passages), len(trans_passages))

        for idx in range(num_passages):
            raw_p_eng = eng_passages[idx] if idx < len(eng_passages) else ""
            raw_p_trans = trans_passages[idx] if idx < len(trans_passages) else ""
            is_sel = is_selected_list[idx] if idx < len(is_selected_list) else 0

            norm_p_eng = normalize_text(
                raw_p_eng,
                self.config.normalize_unicode,
                self.config.normalize_whitespace,
                self.config.strip_control_chars,
            )
            norm_p_trans = normalize_text(
                raw_p_trans,
                self.config.normalize_unicode,
                self.config.normalize_whitespace,
                self.config.strip_control_chars,
            )

            # Use translated passage if present, else English passage as primary text
            primary_text = norm_p_trans if norm_p_trans else norm_p_eng
            english_text = norm_p_eng if norm_p_eng else None

            if not primary_text or len(primary_text) < self.config.min_passage_char_length:
                continue

            doc_id = generate_stable_document_id(
                query_id=int(qid),
                passage_idx=idx,
                text=primary_text,
                lang=target_lang,
            )

            doc = ProcessedDocument(
                document_id=doc_id,
                text=primary_text,
                english_text=english_text,
                source_query_id=int(qid),
                passage_index=idx,
                is_selected=int(is_sel),
                language=target_lang if norm_p_trans else "eng_Latn",
                metadata={
                    "query_type": query_type,
                    "model_name": meta_dict.get("model_name"),
                },
            )
            docs.append(doc)
            all_doc_ids.append(doc_id)
            if is_sel == 1:
                relevant_doc_ids.append(doc_id)

        if not docs:
            return None, [], "all_passages_below_length_threshold"

        query_obj = ProcessedQuery(
            query_id=int(qid),
            query_text=norm_q if norm_q else norm_eq,
            eng_query_text=norm_eq if norm_eq else norm_q,
            answer_text=norm_a,
            eng_answer_text=norm_ea,
            query_type=query_type,
            source_lang=source_lang,
            target_lang=target_lang,
            relevant_document_ids=relevant_doc_ids,
            all_document_ids=all_doc_ids,
        )

        return query_obj, docs, None

    def process_dataset_stream(
        self, record_stream: Any, max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process an iterable stream of raw records in batches and save JSONL outputs."""
        queries_file = os.path.join(self.config.processed_dir, "queries.jsonl")
        documents_file = os.path.join(self.config.processed_dir, "documents.jsonl")
        manifest_file = os.path.join(self.config.processed_dir, "manifest.json")

        input_count = 0
        processed_queries_count = 0
        processed_docs_count = 0
        rejected_count = 0
        deduped_docs_count = 0

        rejected_reasons: Dict[str, int] = {}
        language_dist: Dict[str, int] = {}
        seen_doc_ids: Set[str] = set()

        with open(queries_file, "w", encoding="utf-8") as q_out, open(
            documents_file, "w", encoding="utf-8"
        ) as d_out:

            for rec in record_stream:
                if max_records is not None and input_count >= max_records:
                    break

                input_count += 1
                q_obj, docs, rej_reason = self.process_record(dict(rec))

                if rej_reason or not q_obj:
                    rejected_count += 1
                    reason = rej_reason or "unknown"
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    continue

                # Deduplicate documents if enabled
                valid_docs: List[ProcessedDocument] = []
                for d in docs:
                    if self.config.deduplicate_passages:
                        if d.document_id in seen_doc_ids:
                            deduped_docs_count += 1
                            continue
                        seen_doc_ids.add(d.document_id)
                    valid_docs.append(d)

                if not valid_docs:
                    rejected_count += 1
                    rejected_reasons["all_passages_deduplicated"] = (
                        rejected_reasons.get("all_passages_deduplicated", 0) + 1
                    )
                    continue

                # Write Query JSONL
                q_out.write(json.dumps(asdict(q_obj), ensure_ascii=False) + "\n")
                processed_queries_count += 1

                # Write Documents JSONL
                for d in valid_docs:
                    d_out.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
                    processed_docs_count += 1
                    lang = d.language
                    language_dist[lang] = language_dist.get(lang, 0) + 1

        manifest = ProcessingManifest(
            dataset="ai4bharat/MSMARCO-XI",
            preprocessing_version="1.0.0",
            config=asdict(self.config),
            input_records=input_count,
            processed_queries=processed_queries_count,
            processed_documents=processed_docs_count,
            rejected_records=rejected_count,
            rejected_reasons=rejected_reasons,
            duplicate_passages_deduped=deduped_docs_count,
            language_distribution=language_dist,
            output_files={
                "queries": queries_file,
                "documents": documents_file,
                "manifest": manifest_file,
            },
        )

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)

        logger.info(
            f"Preprocessing complete. Queries: {processed_queries_count}, Docs: {processed_docs_count}, Rejected: {rejected_count}"
        )
        return asdict(manifest)
