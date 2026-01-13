from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _is_api_model(model_name: str) -> bool:
    """Check if model name refers to an API-based model (OpenAI or Gemini)."""
    return model_name.startswith("openai/") or model_name.startswith("gemini/")


@dataclass
class RedisVectorIndex:
    """
    RedisVL vector index backed by either a local SentenceTransformer model,
    OpenAI embeddings API, or Gemini embeddings API.

    Parameters
    ----------
    col_query : str
        Name of the text column whose *embedding* is built and searched.
    index_name : str
        RediSearch index name.
    prefix : str
        Document key prefix (e.g. "cache:").
    model_name : str
        SentenceTransformer model name, local path (e.g. "all-MiniLM-L6-v2"),
        OpenAI model with 'openai/' prefix (e.g. "openai/text-embedding-3-small"),
        or Gemini model with 'gemini/' prefix (e.g. "gemini/text-embedding-004").
    redis_url : str
        Redis connection URL (default "redis://localhost:6379").
    device : str
        "cuda" or "cpu". If None, auto-selects CUDA when available.
        Ignored for API-based models.
    batch_size : int
        Batch size for encoder.encode().
    additional_fields : list[dict]
        Extra schema fields if needed (e.g., [{"name":"row_index","type":"numeric"}]).
    """

    col_query: str
    index_name: str
    prefix: str
    model_name: str
    redis_url: str = "redis://localhost:6379"
    device: str | None = None
    batch_size: int = 256
    additional_fields: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self._is_api = _is_api_model(self.model_name)
        
        # Use the new embedding provider architecture
        from src.customer_analysis.embedding_providers import get_embedding_provider
        
        device = self.device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
        self._provider = get_embedding_provider(self.model_name, device=device)
        
        # Get embedding dimension
        self.embed_dim = self._provider.get_embedding_dim()

        # Ensure Redis index exists (schema dims come from the model)
        schema_dict = {
            "index": {"name": self.index_name, "prefix": self.prefix},
            "fields": [
                {"name": self.col_query, "type": "text"},
                {
                    "name": f"{self.col_query}_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self.embed_dim,
                        "distance_metric": "COSINE",
                        "algorithm": "FLAT",
                        "datatype": "FLOAT32",
                    },
                },
                *self.additional_fields,
            ],
        }
        schema = IndexSchema.from_dict(schema_dict)
        self.index: SearchIndex = SearchIndex(schema, redis_url=self.redis_url)
        # Always overwrite to ensure schema matches the current model dimensions
        self.index.create(overwrite=True)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using the configured provider."""
        vecs = self._provider.encode(
            texts,
            batch_size=self.batch_size,
            normalize=False,  # leave unnormalized; Redis uses true cosine
            show_progress=False,
        )
        # ensure float32
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)
        return vecs

    def load_texts_and_vecs(self, texts: List[str], vecs: np.ndarray):
        docs = []
        for i, (txt, vec) in enumerate(zip(texts, vecs)):
            docs.append(
                {
                    "id": f"{self.prefix}{i}",
                    self.col_query: txt,
                    f"{self.col_query}_embedding": vec.tobytes(),
                }
            )
        self.index.load(docs)

    def load(self, docs: List[Dict[str, Any]]):
        self.index.load(docs)

    def query_vector_topk(self, query_vec: np.ndarray, k: int = 1):
        vq = VectorQuery(
            vector_field_name=f"{self.col_query}_embedding",
            vector=query_vec.astype(np.float32).tobytes(),
            num_results=k,
            return_score=True,
            return_fields=[
                "id",
                self.col_query,
                "vector_distance",
                *[f["name"] for f in self.additional_fields],
            ],
        )
        return self.index.query(vq)

    def drop(self):
        try:
            self.index.drop(delete_documents=True)
        except Exception:
            try:
                self.index.delete()
            except Exception:
                pass
        # best-effort free model memory on CUDA
        if not self._is_api:
            try:
                if _HAS_TORCH and hasattr(self._provider, 'model'):
                    model = self._provider.model
                    if hasattr(model, 'device') and model.device.type == "cuda":
                        del self._provider
                        torch.cuda.empty_cache()
            except Exception:
                pass
