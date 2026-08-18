"""Qdrant service abstraction for collection lifecycle, indexing, and vector search."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from ingestion.chunking.models import Chunk
from retrieval.embeddings.models import VectorArtifact
from retrieval.vector_db.models import QdrantConfig, SearchResultPoint

logger = logging.getLogger(__name__)

# Fixed namespace UUID for deterministic, idempotent point ID generation from chunk_id
CHUNK_POINT_NAMESPACE = uuid.UUID("6ba7b810-9ed0-11d1-80b4-00c04fd430c8")


def generate_deterministic_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk_id for idempotent Qdrant points."""
    return str(uuid.uuid5(CHUNK_POINT_NAMESPACE, chunk_id))


class QdrantService:
    """Production Qdrant service handling collection management and vector operations."""

    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig()
        self.client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self):
        """Connect to Qdrant using URL, local path, or memory mode."""
        url = self.config.url or os.getenv("QDRANT_URL")
        if url:
            logger.info(f"Connecting to remote Qdrant server at '{url}'...")
            self.client = QdrantClient(url=url)
        elif self.config.path == ":memory:":
            logger.info("Initializing Qdrant client in in-memory mode...")
            self.client = QdrantClient(":memory:")
        else:
            local_path = os.path.abspath(self.config.path or "data/qdrant_db")
            os.makedirs(local_path, exist_ok=True)
            logger.info(f"Initializing local Qdrant client at storage path '{local_path}'...")
            self.client = QdrantClient(path=local_path)

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        c_name = collection_name or self.config.collection_name
        try:
            return self.client.collection_exists(c_name)
        except Exception:
            try:
                collections = self.client.get_collections().collections
                return any(c.name == c_name for c in collections)
            except Exception as exc:
                logger.error(f"Error checking collection existence: {exc}")
                return False

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        recreate: bool = False,
    ) -> bool:
        """Create or verify Qdrant collection with configured vector size & distance."""
        c_name = collection_name or self.config.collection_name
        distance_enum = (
            rest_models.Distance.COSINE
            if self.config.distance.lower() == "cosine"
            else rest_models.Distance.DOT
        )

        exists = self.collection_exists(c_name)
        if exists:
            if recreate:
                logger.warning(f"Recreating collection '{c_name}'...")
                self.client.delete_collection(c_name)
            else:
                logger.info(f"Collection '{c_name}' already exists. Validating configuration...")
                info = self.client.get_collection(c_name)
                # Handle vector config check
                vectors_cfg = info.config.params.vectors
                if isinstance(vectors_cfg, rest_models.VectorParams):
                    actual_dim = vectors_cfg.size
                    if actual_dim != self.config.vector_size:
                        raise ValueError(
                            f"Collection dimension mismatch! Existing: {actual_dim}, Configured: {self.config.vector_size}"
                        )
                return True

        logger.info(
            f"Creating Qdrant collection '{c_name}' [dim={self.config.vector_size}, distance={self.config.distance}]..."
        )
        self.client.create_collection(
            collection_name=c_name,
            vectors_config=rest_models.VectorParams(
                size=self.config.vector_size,
                distance=distance_enum,
            ),
        )

        # Create payload indexes for metadata filtering
        self._create_payload_indexes(c_name)
        return True

    def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes on frequently filtered fields."""
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="chunk_id",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="source_query_id",
                field_schema=rest_models.PayloadSchemaType.INTEGER,
            )
            logger.info(f"Payload indexes created for collection '{collection_name}'.")
        except Exception as exc:
            logger.warning(f"Payload index creation note: {exc}")

    def count(self, collection_name: Optional[str] = None) -> int:
        """Get total point count in collection."""
        c_name = collection_name or self.config.collection_name
        if not self.collection_exists(c_name):
            return 0
        try:
            res = self.client.count(collection_name=c_name, exact=True)
            return res.count
        except Exception as exc:
            logger.error(f"Error counting points: {exc}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant connectivity."""
        try:
            collections = self.client.get_collections().collections
            c_names = [c.name for c in collections]
            point_cnt = self.count() if self.config.collection_name in c_names else 0
            return {
                "status": "healthy",
                "collections_count": len(c_names),
                "target_collection": self.config.collection_name,
                "target_point_count": point_cnt,
                "vector_dimension": self.config.vector_size,
                "distance_metric": self.config.distance,
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    def upsert_batch(
        self,
        artifacts: List[VectorArtifact],
        chunks_map: Optional[Dict[str, Chunk]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Idempotent batch upsert of vector artifacts into Qdrant."""
        if not artifacts:
            return True

        c_name = collection_name or self.config.collection_name
        if not self.collection_exists(c_name):
            self.create_collection(collection_name=c_name, recreate=False)

        points: List[rest_models.PointStruct] = []
        for art in artifacts:
            point_id = generate_deterministic_point_id(art.chunk_id)
            chunk_text = ""
            chunk_strategy = "unknown"
            chunk_idx = 0

            if chunks_map and art.chunk_id in chunks_map:
                chunk_obj = chunks_map[art.chunk_id]
                chunk_text = chunk_obj.text
                chunk_strategy = chunk_obj.chunk_strategy
                chunk_idx = chunk_obj.chunk_index
            else:
                chunk_text = art.metadata.get("text", "")
                chunk_strategy = art.metadata.get("chunk_strategy", "semantic")
                chunk_idx = art.metadata.get("chunk_index", 0)

            payload = {
                "chunk_id": art.chunk_id,
                "document_id": art.document_id,
                "text": chunk_text,
                "chunk_index": chunk_idx,
                "chunk_strategy": chunk_strategy,
                "source_query_id": art.metadata.get("source_query_id"),
                "is_selected": art.metadata.get("is_selected"),
                "language": art.metadata.get("language"),
                "embedding_model": art.embedding_model,
            }

            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=art.vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=c_name, points=points, wait=True)
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultPoint]:
        """Perform vector search on Qdrant collection."""
        c_name = collection_name or self.config.collection_name
        if not self.collection_exists(c_name):
            logger.warning(f"Collection '{c_name}' does not exist for search.")
            return []

        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                must_conditions.append(
                    rest_models.FieldCondition(
                        key=k,
                        match=rest_models.MatchValue(value=v),
                    )
                )
            qdrant_filter = rest_models.Filter(must=must_conditions)

        if hasattr(self.client, "query_points"):
            search_res = self.client.query_points(
                collection_name=c_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            search_result = search_res.points
        else:
            search_result = self.client.search(
                collection_name=c_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )

        results: List[SearchResultPoint] = []
        for hit in search_result:
            p = hit.payload or {}
            res = SearchResultPoint(
                chunk_id=p.get("chunk_id", ""),
                document_id=p.get("document_id", ""),
                score=float(hit.score),
                text=p.get("text", ""),
                chunk_index=p.get("chunk_index", 0),
                chunk_strategy=p.get("chunk_strategy", "unknown"),
                metadata={
                    "source_query_id": p.get("source_query_id"),
                    "is_selected": p.get("is_selected"),
                    "language": p.get("language"),
                    "point_id": str(hit.id),
                },
            )
            results.append(res)

        return results
