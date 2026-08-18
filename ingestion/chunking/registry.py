"""Chunking strategy registry for selecting and instantiating chunkers."""

import logging
from typing import Dict, List, Optional, Type

from ingestion.chunking.base import BaseChunker, PassthroughChunker
from ingestion.chunking.models import ChunkingConfig

logger = logging.getLogger(__name__)


class StrategyNotFoundError(Exception):
    """Raised when an unregistered chunking strategy is requested."""


class ChunkerRegistry:
    """Registry for registering and resolving chunking strategy implementations."""

    _registry: Dict[str, Type[BaseChunker]] = {}

    @classmethod
    def register(cls, name: str, chunker_cls: Type[BaseChunker]) -> None:
        """Register a new chunker strategy implementation class."""
        if not issubclass(chunker_cls, BaseChunker):
            raise ValueError(f"Class {chunker_cls} must subclass BaseChunker")
        cls._registry[name.lower()] = chunker_cls
        logger.debug(f"Registered chunking strategy: '{name.lower()}'")

    @classmethod
    def get(
        cls, name: str, config: Optional[ChunkingConfig] = None
    ) -> BaseChunker:
        """Resolve and instantiate a registered chunking strategy."""
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise StrategyNotFoundError(
                f"Chunking strategy '{name}' not found in registry. Registered strategies: {available}"
            )
        chunker_cls = cls._registry[key]
        cfg = config or ChunkingConfig(strategy=key)
        return chunker_cls(config=cfg)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List names of all registered chunking strategies."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def clear(cls) -> None:
        """Clear registry (primarily for testing)."""
        cls._registry.clear()


# Register default foundation strategy
ChunkerRegistry.register("passthrough", PassthroughChunker)
