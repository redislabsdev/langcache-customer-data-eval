"""
Embedding interface for semantic cache evaluation.

This module provides a unified EmbeddingModel class that works with any embedding provider
(HuggingFace, OpenAI, Gemini) and delegates matching logic to SimilarityMatcher.
"""

from typing import Optional

import numpy as np

from src.customer_analysis.embedding_providers import (
    EmbeddingProvider,
    get_embedding_provider,
)
from src.customer_analysis.similarity_matcher import SimilarityMatcher


class EmbeddingModel:
    """
    Unified embedding model that works with any EmbeddingProvider.

    This class provides a simple interface for embedding and similarity matching,
    delegating actual work to the underlying provider and SimilarityMatcher.
    """

    def __init__(self, provider: EmbeddingProvider):
        """Initialize with an embedding provider."""
        self._provider = provider
        self._matcher = SimilarityMatcher(provider)
        self.embeddings: Optional[dict[str, list[float]]] = None

    @property
    def model(self):
        """Return underlying model for HuggingFace providers."""
        return getattr(self._provider, "model", None)

    @property
    def client(self):
        """Return underlying client for API providers."""
        return getattr(self._provider, "client", None)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return getattr(self._provider, "model_name", "unknown")

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """Encode sentences to embeddings."""
        batch_size = kwargs.get("batch_size", 32)
        return self._provider.encode(sentences, batch_size=batch_size, normalize=False)

    def embed_all_sentences(self, sentences: list[str], batch_size: int) -> dict[str, list[float]]:
        """Embed all unique sentences."""
        return self._matcher.embed_all_sentences(sentences, batch_size)

    def calculate_best_matches(
        self,
        sentences: list[str],
        batch_size: int = 32,
        large_dataset: bool = False,
        early_stop: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate best similarity match for each sentence against all others."""
        result = self._matcher.calculate_best_matches(sentences, batch_size, large_dataset, early_stop)
        self.embeddings = self._matcher.embeddings
        return result

    def calculate_best_matches_from_embeddings(
        self,
        embeddings: dict[str, list[float]],
        sentences: list[str],
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate best matches using pre-computed embeddings."""
        return self._matcher.calculate_best_matches_from_embeddings(embeddings, sentences, batch_size)

    def calculate_best_matches_with_cache(
        self,
        sentences: list[str],
        cache: list[str],
        batch_size: int = 1024,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate best matches for each sentence against cache entries."""
        return self._matcher.calculate_best_matches_with_cache(sentences, cache, batch_size, k)

    def calculate_best_matches_from_embeddings_with_cache(
        self,
        cache_embeddings: dict[str, list[float]],
        sentence_embeddings: dict[str, list[float]],
        sentences: list[str],
        cache: list[str],
        batch_size: int = 1024,
        sentence_offset: int = 0,
        mask_self_similarity: bool = False,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate best matches using pre-computed embeddings."""
        return self._matcher.calculate_best_matches_from_embeddings_with_cache(
            cache_embeddings, sentence_embeddings, sentences, cache,
            batch_size, sentence_offset, mask_self_similarity, k,
        )

    def calculate_best_matches_with_cache_large_dataset(
        self,
        queries: list[str],
        cache: list[str],
        batch_size: int = 1024,
        *,
        memmap_dir: Optional[str] = None,
        dtype: np.dtype = np.float32,
        sentence_offset: int = 0,
        early_stop: int = 0,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Large-dataset matching using memory-mapped files."""
        return self._matcher.calculate_best_matches_with_cache_large_dataset(
            queries, cache, batch_size,
            memmap_dir=memmap_dir, dtype=dtype,
            sentence_offset=sentence_offset, early_stop=early_stop, k=k,
        )


def get_embedding_model(model_name: str, device: str = "cpu") -> EmbeddingModel:
    """
    Factory function to create an EmbeddingModel based on model name.

    Args:
        model_name: Model name with optional prefix:
            - 'openai/...' for OpenAI models
            - 'gemini/...' for Gemini models
            - Otherwise, assumes HuggingFace SentenceTransformer
        device: Device to use for local models ('cuda' or 'cpu').

    Returns:
        An EmbeddingModel instance.
    """
    provider = get_embedding_provider(model_name, device)
    return EmbeddingModel(provider)
