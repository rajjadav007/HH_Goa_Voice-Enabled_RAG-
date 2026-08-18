"""Strategy implementations for chunking framework."""

from ingestion.chunking.registry import ChunkerRegistry
from ingestion.chunking.strategies.fixed import FixedSizeChunker
from ingestion.chunking.strategies.sentence import SentenceChunker
from ingestion.chunking.strategies.structure import StructureAwareChunker
from ingestion.chunking.strategies.semantic import SemanticChunker
from ingestion.chunking.strategies.hybrid import HybridChunker

# Auto-register all 5 chunking strategies in ChunkerRegistry
ChunkerRegistry.register("fixed", FixedSizeChunker)
ChunkerRegistry.register("sentence", SentenceChunker)
ChunkerRegistry.register("structure", StructureAwareChunker)
ChunkerRegistry.register("semantic", SemanticChunker)
ChunkerRegistry.register("hybrid", HybridChunker)

__all__ = [
    "FixedSizeChunker",
    "SentenceChunker",
    "StructureAwareChunker",
    "SemanticChunker",
    "HybridChunker",
]
