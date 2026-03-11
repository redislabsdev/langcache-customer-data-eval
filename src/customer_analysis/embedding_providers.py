"""
Embedding provider interface and implementations.

This module defines an abstract base class for embedding providers and
concrete implementations for HuggingFace (SentenceTransformer), OpenAI, and Gemini.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_on: Callable[[Exception], bool] = lambda e: False,
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute.
        max_retries: Maximum number of retry attempts.
        retry_on: A function that returns True if the exception should trigger a retry.

    Returns:
        The result of the function.

    Raises:
        The last exception if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if retry_on(e) and attempt < max_retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
            else:
                raise


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity."""
    if len(embeddings) == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embeddings / norms


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement the encode() and get_embedding_dim() methods.
    The encode() method should return normalized embeddings for cosine similarity.
    """

    @abstractmethod
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode sentences to embeddings.

        Args:
            sentences: A list of sentences to encode.
            batch_size: Batch size for encoding.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
            show_progress: Whether to show a progress bar.

        Returns:
            np.ndarray: A (N, D) array of embeddings where N is the number of sentences
                        and D is the embedding dimension.
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the embedding dimension for this provider.

        Returns:
            int: The embedding dimension.
        """
        pass

    @property
    def provider_name(self) -> str:
        """Return the name of this provider for decision_methods tracking."""
        return "unknown"


class HuggingFaceProvider(EmbeddingProvider):
    """Embedding provider using HuggingFace SentenceTransformer models."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(
            model_name, device=device, local_files_only=False, trust_remote_code=True
        )
        self._embedding_dim: Optional[int] = None

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.get_embedding_dim())

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32, copy=False)

    def get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            probe = self.model.encode(["test"], convert_to_numpy=True)
            self._embedding_dim = int(probe.shape[1])
        return self._embedding_dim

    @property
    def provider_name(self) -> str:
        return "neural"


class APIProvider(EmbeddingProvider):
    """Base class for API-based embedding providers with common functionality."""

    MODEL_DIMENSIONS: dict[str, int] = {}

    def __init__(self, model_name: str):
        self._embedding_dim: Optional[int] = None
        self.model_name = self._extract_model_name(model_name)

    def _extract_model_name(self, model_name: str) -> str:
        """Extract model name by removing provider prefix."""
        prefixes = ["openai/", "gemini/"]
        for prefix in prefixes:
            if model_name.startswith(prefix):
                return model_name[len(prefix) :]
        return model_name

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if exception is a rate limit error."""
        error_str = str(e).lower()
        return "rate" in error_str or "quota" in error_str or "rate_limit" in error_str

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        """Encode a single batch. Must be implemented by subclasses."""
        raise NotImplementedError

    def _encode_with_batching(
        self, sentences: list[str], batch_size: int, show_progress: bool
    ) -> list[list[float]]:
        """Encode sentences with batching and retry logic."""
        from tqdm import tqdm

        all_embeddings: list[list[float]] = []
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Embedding sentences ({self.provider_name})...")

        for i in iterator:
            batch = sentences[i : i + batch_size]
            batch_embeddings = retry_with_backoff(
                func=lambda b=batch: self._encode_batch(b),
                max_retries=3,
                retry_on=self._is_rate_limit_error,
            )
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 100,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.get_embedding_dim())

        all_embeddings = self._encode_with_batching(sentences, batch_size, show_progress)
        embeddings = np.array(all_embeddings, dtype=np.float32)

        if normalize:
            embeddings = normalize_embeddings(embeddings)

        return embeddings

    def get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            if self.model_name in self.MODEL_DIMENSIONS:
                self._embedding_dim = self.MODEL_DIMENSIONS[self.model_name]
            else:
                probe = self.encode(["test"], normalize=False)
                self._embedding_dim = probe.shape[1]
        return self._embedding_dim


class OpenAIProvider(APIProvider):
    """Embedding provider using OpenAI's API."""

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it to use OpenAI embeddings."
            )

        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=batch, model=self.model_name)
        return [item.embedding for item in response.data]

    @property
    def provider_name(self) -> str:
        return "openai"


class GeminiProvider(APIProvider):
    """Embedding provider using Google's Gemini API."""

    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        "text-embedding-005": 768,
        "embedding-001": 768,
    }
    MAX_BATCH_SIZE = 100  # Gemini API limit

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini embeddings. "
                "Install it with: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
                "Please set it to use Gemini embeddings."
            )

        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)
        self.model_path = self._get_model_path()

    def _get_model_path(self) -> str:
        if self.model_name.startswith("models/"):
            return self.model_name
        return f"models/{self.model_name}"

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model_path,
            contents=batch,
        )
        return [embedding.values for embedding in response.embeddings]

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 100,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        # Enforce Gemini's batch size limit
        batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        return super().encode(sentences, batch_size, normalize, show_progress)

    @property
    def provider_name(self) -> str:
        return "gemini"


def get_embedding_provider(model_name: str, device: str = "cpu") -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider based on model name.

    Args:
        model_name: Model name with optional prefix:
            - 'openai/text-embedding-3-small' for OpenAI models
            - 'gemini/text-embedding-004' for Gemini models
            - Otherwise, assumes HuggingFace SentenceTransformer model
        device: Device to use for local models ('cuda' or 'cpu').

    Returns:
        An EmbeddingProvider instance.
    """
    if model_name.startswith("openai/"):
        return OpenAIProvider(model_name, device=device)
    elif model_name.startswith("gemini/"):
        return GeminiProvider(model_name, device=device)
    else:
        return HuggingFaceProvider(model_name, device=device)
