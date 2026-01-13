"""
Similarity matching logic for semantic cache evaluation.

This module contains the SimilarityMatcher class which performs similarity-based
matching between queries and cache entries using embedding providers.
"""

import os
import tempfile
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.customer_analysis.embedding_providers import EmbeddingProvider, normalize_embeddings


# ------------------------------
# Top-K Helper Functions
# ------------------------------


def find_top1(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the best match (top-1) for each row in similarity matrix."""
    best_idx = np.argmax(sim, axis=1)
    best_val = sim[np.arange(sim.shape[0]), best_idx]
    return best_idx, best_val.astype(np.float32, copy=False)


def find_topk(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the top-k matches for each row in similarity matrix."""
    if sim.shape[1] <= k:
        # Fewer candidates than k - return all sorted
        top_k_idx = np.argsort(-sim, axis=1)
        top_k_val = np.take_along_axis(sim, top_k_idx, axis=1)
    else:
        # Use partial sort for efficiency
        part_idx = np.argpartition(-sim, k, axis=1)[:, :k]
        top_k_val = np.take_along_axis(sim, part_idx, axis=1)
        sorted_sub_idx = np.argsort(-top_k_val, axis=1)
        top_k_val = np.take_along_axis(top_k_val, sorted_sub_idx, axis=1)
        top_k_idx = np.take_along_axis(part_idx, sorted_sub_idx, axis=1)
    return top_k_idx, top_k_val


def merge_topk(
    current_idx: np.ndarray,
    current_val: np.ndarray,
    new_idx: np.ndarray,
    new_val: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two sets of top-k results and keep the best k."""
    combined_vals = np.concatenate([current_val, new_val], axis=1)
    combined_idxs = np.concatenate([current_idx, new_idx], axis=1)
    best_args = np.argsort(-combined_vals, axis=1)[:, :k]
    return (
        np.take_along_axis(combined_idxs, best_args, axis=1),
        np.take_along_axis(combined_vals, best_args, axis=1),
    )


def update_best_k1(
    chunk_idx: np.ndarray,
    chunk_val: np.ndarray,
    block_idx: np.ndarray,
    block_val: np.ndarray,
    col_offset: int,
) -> None:
    """Update best scores in-place for k=1 case."""
    better = block_val > chunk_val
    chunk_val[better] = block_val[better]
    chunk_idx[better] = col_offset + block_idx[better]


# ------------------------------
# Self-Similarity Masking
# ------------------------------


def mask_self_similarity_block(
    sim: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    sentence_offset: int,
) -> None:
    """Mask diagonal entries in similarity matrix to avoid self-matching."""
    row_global_start = row_start + sentence_offset
    row_global_end = row_end + sentence_offset
    overlap_start = max(row_global_start, col_start)
    overlap_end = min(row_global_end, col_end)

    if overlap_start < overlap_end:
        row_local = np.arange(overlap_start - row_global_start, overlap_end - row_global_start)
        col_local = np.arange(overlap_start - col_start, overlap_end - col_start)
        sim[row_local, col_local] = -np.inf


# ------------------------------
# Result Initialization
# ------------------------------


def init_results(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize result arrays for indices and scores."""
    if k == 1:
        return np.zeros(n, dtype=np.int32), np.full(n, -np.inf, dtype=np.float32)
    return np.zeros((n, k), dtype=np.int32), np.full((n, k), -np.inf, dtype=np.float32)


# ------------------------------
# SimilarityMatcher Class
# ------------------------------


class SimilarityMatcher:
    """
    Similarity matcher for finding best matches between queries and cache entries.

    This class handles all the matching logic, including:
    - Embedding all sentences
    - Computing cosine similarity
    - Finding top-k matches
    - Memory-efficient large dataset handling with memmaps
    """

    def __init__(self, provider: EmbeddingProvider):
        """Initialize the SimilarityMatcher with an embedding provider."""
        self.provider = provider
        self.embeddings: Optional[dict[str, list[float]]] = None

    def embed_all_sentences(
        self, sentences: list[str], batch_size: int = 32
    ) -> dict[str, list[float]]:
        """Embed all unique sentences and return a dictionary mapping sentences to embeddings."""
        sentence_to_embeddings: dict[str, list[float]] = {}
        sentence_list = list(set(sentences))
        total = len(sentence_list)

        print(f"Embedding {total} unique sentences in batches of {batch_size} ...")

        for start in tqdm(range(0, total, batch_size), desc="Embedding sentences..."):
            end = min(start + batch_size, total)
            batch = sentence_list[start:end]
            batch_embs = self.provider.encode(batch, batch_size=batch_size, normalize=False)
            for sent, emb in zip(batch, batch_embs):
                sentence_to_embeddings[sent] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        return sentence_to_embeddings

    def calculate_best_matches(
        self,
        sentences: list[str],
        batch_size: int = 32,
        large_dataset: bool = False,
        early_stop: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the best similarity match for each sentence against all other sentences."""
        if not large_dataset:
            self.embeddings = self.embed_all_sentences(sentences, batch_size)
            return self.calculate_best_matches_from_embeddings(self.embeddings, sentences, batch_size)
        return self._calculate_best_matches_large_dataset(sentences, batch_size, early_stop=early_stop)

    def calculate_best_matches_from_embeddings(
        self,
        embeddings: dict[str, list[float]],
        sentences: list[str],
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate best similarity matches using pre-computed embeddings (self-matching)."""
        best_indices, best_scores = init_results(len(sentences), k=1)
        decision_methods = np.full(len(sentences), self.provider.provider_name, dtype=object)

        for batch_start in tqdm(range(0, len(sentences), batch_size), desc="Calculating best matches..."):
            batch_end = min(batch_start + batch_size, len(sentences))
            out = self.calculate_best_matches_from_embeddings_with_cache(
                cache_embeddings=embeddings,
                sentence_embeddings=embeddings,
                sentences=sentences[batch_start:batch_end],
                cache=sentences,
                batch_size=batch_size,
                sentence_offset=batch_start,
                mask_self_similarity=True,
            )
            best_indices[batch_start:batch_end] = out[0]
            best_scores[batch_start:batch_end] = out[1]

        return best_indices, best_scores, decision_methods

    def calculate_best_matches_with_cache(
        self,
        sentences: list[str],
        cache: list[str],
        batch_size: int = 1024,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the best similarity match for each sentence against cache entries."""
        cache_embeddings = self.embed_all_sentences(cache, batch_size)
        sentence_embeddings = self.embed_all_sentences(sentences, batch_size)
        return self.calculate_best_matches_from_embeddings_with_cache(
            cache_embeddings=cache_embeddings,
            sentence_embeddings=sentence_embeddings,
            sentences=sentences,
            cache=cache,
            batch_size=batch_size,
            sentence_offset=0,
            k=k,
        )

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
        """Calculate the best similarity match using pre-computed embeddings."""
        # Build and normalize embedding matrices
        cache_matrix = normalize_embeddings(
            np.asarray([cache_embeddings[s] for s in cache], dtype=np.float32)
        )
        sentence_matrix = normalize_embeddings(
            np.asarray([sentence_embeddings[s] for s in sentences], dtype=np.float32)
        )

        best_indices, best_scores = init_results(len(sentences), k)
        decision_methods = np.full(len(sentences), self.provider.provider_name, dtype=object)

        for start in tqdm(
            range(0, len(sentences), batch_size),
            desc="Calculating best matches with cache...",
            disable=len(sentences) // batch_size < 10,
        ):
            end = min(start + batch_size, len(sentences))
            batch_sims = sentence_matrix[start:end] @ cache_matrix.T

            if mask_self_similarity:
                self._mask_batch_self_similarity(batch_sims, start, end, sentence_offset, len(cache))

            if k == 1:
                idx, val = find_top1(batch_sims)
            else:
                idx, val = find_topk(batch_sims, k)

            best_indices[start:end] = idx
            best_scores[start:end] = val

        return best_indices, best_scores, decision_methods

    def _mask_batch_self_similarity(
        self, batch_sims: np.ndarray, start: int, end: int, offset: int, cache_len: int
    ) -> None:
        """Mask self-similarity in a batch similarity matrix."""
        row_indices = np.arange(end - start)
        col_indices = np.arange(start, end) + offset
        valid = col_indices < cache_len
        if np.any(valid):
            batch_sims[row_indices[valid], col_indices[valid]] = -np.inf

    # ------------------------------
    # Large dataset methods
    # ------------------------------

    def _infer_embedding_dim(self, sentences: list[str]) -> int:
        return self.provider.get_embedding_dim()

    def _prepare_memmap_dir(self, memmap_dir: Optional[str]) -> tuple[bool, str, str]:
        """Ensure a directory exists for memmap files."""
        created = memmap_dir is None
        if created:
            memmap_dir = tempfile.mkdtemp(prefix="embedding_eval_memmap_")
        else:
            os.makedirs(memmap_dir, exist_ok=True)
        return created, memmap_dir, os.path.join(memmap_dir, "embeddings.dat")

    def _write_embeddings_memmap(
        self,
        sentences: list[str],
        emb_path: str,
        num_sentences: int,
        embedding_dim: int,
        batch_size: int,
        dtype: np.dtype,
    ) -> None:
        """Encode sentences and write normalized embeddings to memmap."""
        mm = np.memmap(emb_path, mode="w+", dtype=dtype, shape=(num_sentences, embedding_dim))
        print(f"Encoding and writing {num_sentences} embeddings to memmap at {emb_path} ...")

        for start in tqdm(range(0, num_sentences, batch_size), desc="Encoding (memmap)..."):
            end = min(start + batch_size, num_sentences)
            batch_embs = self.provider.encode(sentences[start:end], batch_size=batch_size, normalize=True)
            mm[start:end] = batch_embs.astype(dtype, copy=False)

        mm.flush()
        del mm

    def _choose_block_sizes(self, batch_size: int) -> tuple[int, int]:
        """Pick conservative row/col block sizes to bound peak memory."""
        max_block_bytes = 128 * 1024 * 1024
        row_block = min(batch_size, 4096)
        col_block = max(512, min(batch_size, int(max_block_bytes / 4 / max(1, row_block))))
        return row_block, col_block

    def _compute_blockwise_best_matches(
        self,
        emb_path: str,
        num_sentences: int,
        embedding_dim: int,
        row_block: int,
        col_block: int,
        dtype: np.dtype,
        early_stop: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blockwise exact nearest-neighbour (k=1) with self-similarity masking."""
        n = early_stop if early_stop > 0 else num_sentences
        best_indices, best_scores = init_results(n, k=1)
        mm = np.memmap(emb_path, mode="r", dtype=dtype, shape=(n, embedding_dim))

        for row_start in tqdm(range(0, n, row_block), desc="Row blocks"):
            row_end = min(row_start + row_block, n)
            row_emb = np.asarray(mm[row_start:row_end])
            chunk_idx, chunk_val = init_results(row_end - row_start, k=1)

            for col_start in range(0, n, col_block):
                col_end = min(col_start + col_block, n)
                sim = row_emb @ np.asarray(mm[col_start:col_end]).T
                mask_self_similarity_block(sim, row_start, row_end, col_start, col_end, 0)
                block_idx, block_val = find_top1(sim)
                update_best_k1(chunk_idx, chunk_val, block_idx, block_val, col_start)

            best_indices[row_start:row_end] = chunk_idx
            best_scores[row_start:row_end] = chunk_val

        del mm
        return best_indices, best_scores

    def _cleanup_memmap(self, created: bool, memmap_dir: str, emb_path: str) -> None:
        """Best-effort cleanup of memmap files."""
        if not created:
            return
        try:
            if os.path.exists(emb_path):
                os.remove(emb_path)
            os.rmdir(memmap_dir)
        except Exception:
            pass

    def _compute_blockwise_best_matches_two_sets(
        self,
        row_emb_path: str,
        num_rows: int,
        col_emb_path: str,
        num_cols: int,
        embedding_dim: int,
        row_block: int,
        col_block: int,
        dtype: np.dtype,
        *,
        mask_self_similarity: bool = False,
        sentence_offset: int = 0,
        early_stop: int = 0,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blockwise nearest-neighbour where rows and columns come from two sets."""
        n_rows = early_stop if early_stop > 0 else num_rows
        best_indices, best_scores = init_results(n_rows, k)

        rows_mm = np.memmap(row_emb_path, mode="r", dtype=dtype, shape=(n_rows, embedding_dim))
        cols_mm = np.memmap(col_emb_path, mode="r", dtype=dtype, shape=(num_cols, embedding_dim))

        for row_start in tqdm(range(0, n_rows, row_block), desc="Row blocks (two-sets)"):
            row_end = min(row_start + row_block, n_rows)
            row_emb = np.asarray(rows_mm[row_start:row_end])
            chunk_idx, chunk_val = init_results(row_end - row_start, k)

            for col_start in range(0, num_cols, col_block):
                col_end = min(col_start + col_block, num_cols)
                sim = row_emb @ np.asarray(cols_mm[col_start:col_end]).T

                if mask_self_similarity:
                    mask_self_similarity_block(sim, row_start, row_end, col_start, col_end, sentence_offset)

                if k == 1:
                    block_idx, block_val = find_top1(sim)
                    update_best_k1(chunk_idx, chunk_val, block_idx, block_val, col_start)
                else:
                    block_idx, block_val = find_topk(sim, k)
                    chunk_idx, chunk_val = merge_topk(chunk_idx, chunk_val, block_idx + col_start, block_val, k)

            best_indices[row_start:row_end] = chunk_idx
            best_scores[row_start:row_end] = chunk_val

        del rows_mm, cols_mm
        return best_indices, best_scores

    def _calculate_best_matches_large_dataset(
        self,
        sentences: list[str],
        batch_size: int = 1024,
        *,
        memmap_dir: Optional[str] = None,
        dtype: np.dtype = np.float32,
        early_stop: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Memory-efficient exact similarity search using disk-backed memmap."""
        if len(sentences) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=object)

        embedding_dim = self._infer_embedding_dim(sentences)
        created, memmap_dir, emb_path = self._prepare_memmap_dir(memmap_dir)

        self._write_embeddings_memmap(sentences, emb_path, len(sentences), embedding_dim, batch_size, dtype)

        print("Finding best matches with blockwise dot-products ...")
        row_block, col_block = self._choose_block_sizes(batch_size)
        best_indices, best_scores = self._compute_blockwise_best_matches(
            emb_path, len(sentences), embedding_dim, row_block, col_block, dtype, early_stop
        )

        self._cleanup_memmap(created, memmap_dir, emb_path)
        return best_indices, best_scores, np.full(len(sentences), self.provider.provider_name, dtype=object)

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
        """Large-dataset variant: find best cache match for each query using memmaps."""
        num_queries, num_cache = len(queries), len(cache)
        if num_queries == 0 or num_cache == 0:
            idx, scores = init_results(num_queries, k)
            return idx, scores, np.zeros(num_queries, dtype=object)

        embedding_dim = self._infer_embedding_dim(queries)
        created, memmap_dir, _ = self._prepare_memmap_dir(memmap_dir)
        row_path = os.path.join(memmap_dir, "rows_embeddings.dat")
        col_path = os.path.join(memmap_dir, "cols_embeddings.dat")

        self._write_embeddings_memmap(queries, row_path, num_queries, embedding_dim, batch_size, dtype)
        self._write_embeddings_memmap(cache, col_path, num_cache, embedding_dim, batch_size, dtype)

        row_block, col_block = self._choose_block_sizes(batch_size)
        best_indices, best_scores = self._compute_blockwise_best_matches_two_sets(
            row_path, num_queries, col_path, num_cache, embedding_dim,
            row_block, col_block, dtype,
            mask_self_similarity=(queries is cache or queries == cache),
            sentence_offset=sentence_offset,
            early_stop=early_stop,
            k=k,
        )

        # Cleanup
        for path in [row_path, col_path]:
            try:
                os.remove(path)
            except Exception:
                pass
        if created:
            try:
                os.rmdir(memmap_dir)
            except Exception:
                pass

        return best_indices, best_scores, np.full(num_queries, self.provider.provider_name, dtype=object)
