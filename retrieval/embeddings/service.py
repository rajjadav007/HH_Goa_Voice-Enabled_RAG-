"""Production embedding service supporting batch vector inference."""

import logging
from typing import Any, Dict, List, Optional, Union


import numpy as np

from retrieval.embeddings.models import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Production embedding service converting texts into normalized dense vectors."""

    _model_cache: Dict[str, Any] = {}

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None
        self._device = self._resolve_device(self.config.device)
        self._query_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 1024
        self._init_model()

    def _resolve_device(self, device_setting: str) -> str:
        if device_setting == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_setting

    def _init_model(self):
        cache_key = f"{self.config.model_name}:{self._device}"
        if cache_key in EmbeddingService._model_cache:
            logger.info(f"Reusing cached embedding model '{self.config.model_name}' on device '{self._device}'...")
            cached = EmbeddingService._model_cache[cache_key]
            self._model = cached["model"]
            self._dimension = cached["dimension"]
            return

        logger.info(f"Initializing embedding model '{self.config.model_name}' on device '{self._device}'...")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.model_name, device=self._device)
            # Detect model dimension
            dummy_vec = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = int(dummy_vec.shape[1])
            logger.info(f"Embedding model loaded successfully. Dimension: {self._dimension}")
            EmbeddingService._model_cache[cache_key] = {"model": self._model, "dimension": self._dimension}
        except Exception as exc:
            logger.warning(f"Failed to load via sentence_transformers ({exc}). Falling back to HuggingFace transformers...")
            self._init_hf_fallback()


    def _init_hf_fallback(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._hf_model = AutoModel.from_pretrained(self.config.model_name).to(self._device)
        self._hf_model.eval()

        dummy_input = self._tokenizer(["test"], return_tensors="pt", padding=True, truncation=True).to(self._device)
        with torch.no_grad():
            out = self._hf_model(**dummy_input)
            pooled = out.last_hidden_state.mean(dim=1)
            self._dimension = int(pooled.shape[1])
        logger.info(f"HuggingFace fallback model loaded successfully. Dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ValueError("Embedding model dimension is not initialized.")
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Embed a single text string with LRU query caching."""
        clean_text = text.strip()
        if is_query and clean_text in self._query_cache:
            return self._query_cache[clean_text]

        results = self.embed_batch([text], is_query=is_query)
        vec = results[0]

        if is_query and clean_text:
            if len(self._query_cache) >= self._max_cache_size:
                # Evict oldest entry
                first_key = next(iter(self._query_cache))
                del self._query_cache[first_key]
            self._query_cache[clean_text] = vec

        return vec

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []

        prefix = self.config.query_prefix if is_query else self.config.passage_prefix
        prefixed_texts = [f"{prefix}{t if t.strip() else ' '}" for t in texts]

        if self._model is not None:
            raw_embeddings = self._model.encode(
                prefixed_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
            )
            arr = np.array(raw_embeddings, dtype=np.float32)
        else:
            arr = self._hf_encode_batch(prefixed_texts)

        # Ensure values are finite and zero out NaNs
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.tolist()

    def _hf_encode_batch(self, prefixed_texts: List[str]) -> np.ndarray:
        import torch

        inputs = self._tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._hf_model(**inputs)
            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            if self.config.normalize_embeddings:
                mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            return mean_pooled.cpu().numpy()
