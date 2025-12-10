import os
import tempfile
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class NeuralEmbedding:
    """
    A placeholder for a neural embedding model that will use a Hugging Face model.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the NeuralEmbedding model.
        """
        self.model = SentenceTransformer(model_name, device=device, local_files_only=False, trust_remote_code=True)
        self.embeddings = None

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """
        A placeholder 'encode' method to maintain compatibility with the evaluation
        pipeline, which expects a model to have this method.

        This method does not generate meaningful embeddings. Instead, it returns
        a zero vector for each sentence. The actual similarity logic is handled
        by the `calculate_best_matches` method.

        Args:
            sentences (list[str]): A list of sentences to "encode".
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            np.ndarray: A numpy array of zero vectors.
        """
        return self.model.encode(sentences)

    def calculate_best_matches(
        self, sentences: list[str], batch_size: int = 32, large_dataset: bool = False, early_stop: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the best similarity match for each sentence against all other
        sentences using a neural embedding model.

        Args:
            sentences (list[str]): The list of sentences to compare.
            batch_size (int): The batch size to use for the similarity search.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - best_indices (np.ndarray): An array of indices for the best match of each sentence.
                - best_scores (np.ndarray): An array of similarity scores (0-1) for the best matches.
                - decision_methods (np.ndarray): An array with the string value "neural" for every sentence.
        """
        if not large_dataset:
            self.embeddings = self.embed_all_sentences(sentences, batch_size)
            return self.calculate_best_matches_from_embeddings(self.embeddings, sentences, batch_size)
        else:
            return self._calculate_best_matches_large_dataset(sentences, batch_size, early_stop=early_stop)

    def embed_all_sentences(self, sentences: list[str], batch_size: int) -> dict[str, list[float]]:
        """Embed all unique sentences with the provided model."""
        sentence_to_embeddings: dict[str, list[float]] = {}
        sentence_list = list(set(sentences))
        total = len(sentence_list)

        print(f"Embedding {total} unique sentences in batches of {batch_size} ...")

        for start in tqdm(range(0, total, batch_size), desc="Embedding sentences..."):
            end = min(start + batch_size, total)
            batch = sentence_list[start:end]

            batch_embs = self.model.encode(batch)
            for sent, emb in zip(batch, batch_embs):
                sentence_to_embeddings[sent] = emb.tolist() if hasattr(emb, "tolist") else emb
        return sentence_to_embeddings

    def calculate_best_matches_from_embeddings(
        self, embeddings: dict[str, list[float]], sentences: list[str], batch_size: int = 1024
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the best similarity match for each sentence without building a full similarity matrix.
        """
        best_indices = np.zeros(len(sentences), dtype=np.int32)
        best_scores = np.zeros(len(sentences), dtype=np.float32)
        decision_methods = np.full(len(sentences), "neural", dtype=object)

        for sentence_batch_idx in tqdm(
            range(0, len(sentences), batch_size), desc="Calculating best matches with cache..."
        ):
            max_index = min(sentence_batch_idx + batch_size, len(sentences))

            out = self.calculate_best_matches_from_embeddings_with_cache(
                cache_embeddings=embeddings,
                sentence_embeddings=embeddings,
                sentences=sentences[sentence_batch_idx:max_index],
                cache=sentences,
                batch_size=batch_size,
                sentence_offset=sentence_batch_idx,
                mask_self_similarity=True,
            )
            best_indices_batch, best_scores_batch, decision_methods_batch = out

            best_indices[sentence_batch_idx:max_index] = best_indices_batch
            best_scores[sentence_batch_idx:max_index] = best_scores_batch
            decision_methods[sentence_batch_idx:max_index] = decision_methods_batch

        return best_indices, best_scores, decision_methods

    # ------------------------------
    # Large dataset helper methods
    # ------------------------------
    def _infer_embedding_dim(self, sentences: list[str]) -> int:
        """Return the embedding dimension for the current model."""
        # Use probing to get the actual dimension as some models report incorrect config dimensions
        try:
             probe = self.model.encode([sentences[0] if sentences else "test"])
             return int(probe.shape[1])
        except Exception:
             # Fallback to config if probing fails
             return int(self.model.get_sentence_embedding_dimension())

    def _prepare_memmap_dir(self, memmap_dir: Optional[str]) -> tuple[bool, str, str]:
        """Ensure a directory exists for memmap files and return path components.

        Returns (created_tmpdir, directory_path, embeddings_path).
        """
        created_tmpdir = False
        if memmap_dir is None:
            memmap_dir = tempfile.mkdtemp(prefix="embedding_eval_memmap_")
            created_tmpdir = True
        else:
            os.makedirs(memmap_dir, exist_ok=True)
        emb_path = os.path.join(memmap_dir, "embeddings.dat")
        return created_tmpdir, memmap_dir, emb_path

    def _write_embeddings_memmap(
        self,
        sentences: list[str],
        emb_path: str,
        num_sentences: int,
        embedding_dim: int,
        batch_size: int,
        dtype: np.dtype,
    ) -> None:
        """Encode sentences in batches, normalize, and write to a memmap file."""
        embeddings_mm = np.memmap(emb_path, mode="w+", dtype=dtype, shape=(num_sentences, embedding_dim))
        print(f"Encoding and writing {num_sentences} embeddings to memmap at {emb_path} ...")
        for start in tqdm(range(0, num_sentences, batch_size), desc="Encoding (memmap)..."):
            end = min(start + batch_size, num_sentences)
            batch = sentences[start:end]
            batch_embs = self.model.encode(batch, normalize_embeddings=True).astype(dtype, copy=False)
            embeddings_mm[start:end] = batch_embs
        embeddings_mm.flush()
        del embeddings_mm

    def _choose_block_sizes(self, batch_size: int) -> tuple[int, int]:
        """Pick conservative row/col block sizes to bound peak memory."""
        max_block_bytes = 128 * 1024 * 1024  # ~128MB per similarity block
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
        """Blockwise exact nearest-neighbour by cosine-similarity via dot-products."""
        n = early_stop if early_stop > 0 else num_sentences
        best_scores = np.full(n, -np.inf, dtype=np.float32)
        best_indices = np.zeros(n, dtype=np.int32)

        embeddings_mm = np.memmap(emb_path, mode="r", dtype=dtype, shape=(n, embedding_dim))
        for row_start in tqdm(range(0, n, row_block), desc="Row blocks"):
            row_end = min(row_start + row_block, n)
            row_emb = np.asarray(embeddings_mm[row_start:row_end])

            chunk_best_scores = np.full(row_end - row_start, -np.inf, dtype=np.float32)
            chunk_best_indices = np.zeros(row_end - row_start, dtype=np.int32)

            for col_start in range(0, n, col_block):
                col_end = min(col_start + col_block, n)
                col_emb = np.asarray(embeddings_mm[col_start:col_end])

                sim = row_emb @ col_emb.T

                # mask diagonal for overlapping region to avoid self-match
                overlap_start = max(row_start, col_start)
                overlap_end = min(row_end, col_end)
                if overlap_start < overlap_end:
                    i = np.arange(overlap_start, overlap_end)
                    sim[i - row_start, i - col_start] = -np.inf

                block_idx = np.argmax(sim, axis=1)
                block_val = sim[np.arange(sim.shape[0]), block_idx].astype(np.float32, copy=False)

                better = block_val > chunk_best_scores
                if np.any(better):
                    chunk_best_scores[better] = block_val[better]
                    chunk_best_indices[better] = col_start + block_idx[better]

            best_scores[row_start:row_end] = chunk_best_scores
            best_indices[row_start:row_end] = chunk_best_indices

        del embeddings_mm
        return best_indices, best_scores

    def _cleanup_memmap(self, created_tmpdir: bool, memmap_dir: str, emb_path: str) -> None:
        """Best-effort cleanup of memmap file and temp directory if created here."""
        if not created_tmpdir:
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
        """Blockwise nearest-neighbour where rows and columns come from two sets.

        If mask_self_similarity is True, rows are assumed to correspond to a
        contiguous slice of the columns starting at `sentence_offset`, and the
        diagonal entries for that alignment will be masked to -inf.
        """
        n_rows = early_stop if early_stop > 0 else num_rows
        
        if k == 1:
            best_scores = np.full(n_rows, -np.inf, dtype=np.float32)
            best_indices = np.zeros(n_rows, dtype=np.int32)
        else:
            best_scores = np.full((n_rows, k), -np.inf, dtype=np.float32)
            best_indices = np.zeros((n_rows, k), dtype=np.int32)

        rows_mm = np.memmap(row_emb_path, mode="r", dtype=dtype, shape=(n_rows, embedding_dim))
        cols_mm = np.memmap(col_emb_path, mode="r", dtype=dtype, shape=(num_cols, embedding_dim))

        for row_start in tqdm(range(0, n_rows, row_block), desc="Row blocks (two-sets)"):
            row_end = min(row_start + row_block, n_rows)
            row_emb = np.asarray(rows_mm[row_start:row_end])

            if k == 1:
                chunk_best_scores = np.full(row_end - row_start, -np.inf, dtype=np.float32)
                chunk_best_indices = np.zeros(row_end - row_start, dtype=np.int32)
            else:
                chunk_best_scores = np.full((row_end - row_start, k), -np.inf, dtype=np.float32)
                chunk_best_indices = np.zeros((row_end - row_start, k), dtype=np.int32)

            for col_start in range(0, num_cols, col_block):
                col_end = min(col_start + col_block, num_cols)
                col_emb = np.asarray(cols_mm[col_start:col_end])

                sim = row_emb @ col_emb.T

                # Mask diagonal if needed to avoid self-similarity
                if mask_self_similarity:
                    # Calculate the overlap between row indices and column indices
                    row_global_start = row_start + sentence_offset
                    row_global_end = row_end + sentence_offset
                    overlap_start = max(row_global_start, col_start)
                    overlap_end = min(row_global_end, col_end)

                    if overlap_start < overlap_end:
                        # Map global indices to local block indices
                        row_local_indices = np.arange(overlap_start - row_global_start, overlap_end - row_global_start)
                        col_local_indices = np.arange(overlap_start - col_start, overlap_end - col_start)
                        sim[row_local_indices, col_local_indices] = -np.inf

                if k == 1:
                    block_idx = np.argmax(sim, axis=1)
                    block_val = sim[np.arange(sim.shape[0]), block_idx].astype(np.float32, copy=False)

                    for i in range(len(block_val)):
                        if block_val[i] > chunk_best_scores[i]:
                            chunk_best_scores[i] = block_val[i]
                            chunk_best_indices[i] = col_start + block_idx[i]
                else:
                    # Top-k logic
                    # If columns in this block < k, take all valid
                    curr_block_size = col_end - col_start
                    if curr_block_size <= k:
                        top_k_in_block_idx = np.argsort(-sim, axis=1) # Sort all
                        top_k_in_block_val = np.take_along_axis(sim, top_k_in_block_idx, axis=1)
                        # Might have fewer than k if block is small
                    else:
                        # Use argpartition for top k
                        # We want largest k
                        part_idx = np.argpartition(-sim, k, axis=1)[:, :k]
                        top_k_in_block_val = np.take_along_axis(sim, part_idx, axis=1)
                        
                        # Sort them to have ordered top-k (optional but good for merging)
                        sorted_sub_idx = np.argsort(-top_k_in_block_val, axis=1)
                        top_k_in_block_val = np.take_along_axis(top_k_in_block_val, sorted_sub_idx, axis=1)
                        top_k_in_block_idx = np.take_along_axis(part_idx, sorted_sub_idx, axis=1)
                    
                    # Merge with accumulated bests
                    # chunk_best_scores: (batch, k)
                    # top_k_in_block_val: (batch, min(block, k))
                    
                    # Adjust indices to global column indices
                    top_k_in_block_idx_global = top_k_in_block_idx + col_start
                    
                    combined_vals = np.concatenate([chunk_best_scores, top_k_in_block_val], axis=1)
                    combined_idxs = np.concatenate([chunk_best_indices, top_k_in_block_idx_global], axis=1)
                    
                    # Find top k in combined
                    best_combined_args = np.argsort(-combined_vals, axis=1)[:, :k]
                    
                    chunk_best_scores = np.take_along_axis(combined_vals, best_combined_args, axis=1)
                    chunk_best_indices = np.take_along_axis(combined_idxs, best_combined_args, axis=1)


            best_scores[row_start:row_end] = chunk_best_scores
            best_indices[row_start:row_end] = chunk_best_indices

        del rows_mm
        del cols_mm
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
        """Memory-efficient exact similarity search using a disk-backed memmap."""
        num_sentences = len(sentences)
        if num_sentences == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=object),
            )

        # Determine embedding dimension and memmap paths
        embedding_dim = self._infer_embedding_dim(sentences)
        created_tmpdir = False
        if memmap_dir is None:
            memmap_dir = tempfile.mkdtemp(prefix="embedding_eval_memmap_")
            created_tmpdir = True
        os.makedirs(memmap_dir, exist_ok=True)
        emb_path = os.path.join(memmap_dir, "embeddings.dat")

        # Phase 1: write normalized embeddings to disk
        self._write_embeddings_memmap(
            sentences=sentences,
            emb_path=emb_path,
            num_sentences=num_sentences,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            dtype=dtype,
        )

        # Phase 2: blockwise nearest neighbour search
        print("Finding best matches with blockwise dot-products ...")
        row_block, col_block = self._choose_block_sizes(batch_size)
        best_indices, best_scores = self._compute_blockwise_best_matches(
            emb_path=emb_path,
            num_sentences=num_sentences,
            embedding_dim=embedding_dim,
            row_block=row_block,
            col_block=col_block,
            dtype=dtype,
            early_stop=early_stop,
        )

        decision_methods = np.full(num_sentences, "neural", dtype=object)
        self._cleanup_memmap(created_tmpdir, memmap_dir, emb_path)
        return best_indices, best_scores, decision_methods

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
        """Large-dataset variant: find best cache match for each sentence using memmaps.

        Writes two memmaps (rows for sentences, cols for cache), normalised, and
        performs blockwise dot-products. If `sentence_offset` is provided and the
        cache corresponds to the same corpus, the self-similarity diagonal is masked.
        """
        num_sentences = len(queries)
        num_cache = len(cache)
        if num_sentences == 0 or num_cache == 0:
            return (
                np.zeros(num_sentences, dtype=np.int32),
                np.zeros(num_sentences, dtype=np.float32),
                np.zeros(num_sentences, dtype=object),
            )

        embedding_dim = self._infer_embedding_dim(queries)

        created_tmpdir = False
        if memmap_dir is None:
            memmap_dir = tempfile.mkdtemp(prefix="embedding_eval_memmap_")
            created_tmpdir = True
        os.makedirs(memmap_dir, exist_ok=True)

        row_emb_path = os.path.join(memmap_dir, "rows_embeddings.dat")
        col_emb_path = os.path.join(memmap_dir, "cols_embeddings.dat")

        # Write sentence and cache embeddings
        self._write_embeddings_memmap(
            sentences=queries,
            emb_path=row_emb_path,
            num_sentences=num_sentences,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            dtype=dtype,
        )
        # For cache we might reuse the same model; normalisation happens inside
        self._write_embeddings_memmap(
            sentences=cache,
            emb_path=col_emb_path,
            num_sentences=num_cache,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            dtype=dtype,
        )

        row_block, col_block = self._choose_block_sizes(batch_size)
        best_indices, best_scores = self._compute_blockwise_best_matches_two_sets(
            row_emb_path=row_emb_path,
            num_rows=num_sentences,
            col_emb_path=col_emb_path,
            num_cols=num_cache,
            embedding_dim=embedding_dim,
            row_block=row_block,
            col_block=col_block,
            dtype=dtype,
            mask_self_similarity=(queries is cache or queries == cache),
            sentence_offset=sentence_offset,
            early_stop=early_stop,
            k=k,
        )

        decision_methods = np.full(num_sentences, "neural", dtype=object)
        # Cleanup
        try:
            if os.path.exists(row_emb_path):
                os.remove(row_emb_path)
            if os.path.exists(col_emb_path):
                os.remove(col_emb_path)
            if created_tmpdir:
                os.rmdir(memmap_dir)
        except Exception:
            pass

        return best_indices, best_scores, decision_methods

    def calculate_best_matches_with_cache(
        self, sentences: list[str], cache: list[str], batch_size: int = 1024, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the best similarity match for each sentence against all other
        sentences using a neural embedding model.
        """
        cache_embeddings = self.embed_all_sentences(cache, batch_size)
        sentence_embeddings = self.embed_all_sentences(sentences, batch_size)

        out = self.calculate_best_matches_from_embeddings_with_cache(
            cache_embeddings=cache_embeddings,
            sentence_embeddings=sentence_embeddings,
            sentences=sentences,
            cache=cache,
            batch_size=batch_size,
            sentence_offset=0,
            k=k,
        )

        best_indices, best_scores, decision_methods = out

        return best_indices, best_scores, decision_methods

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
        """
        Calculate the best similarity match for each sentence against all other
        sentences using a neural embedding model.
        """
        cache_embeddings_matrix = np.asarray([cache_embeddings[s] for s in cache], dtype=np.float32)
        sentence_embeddings_matrix = np.asarray([sentence_embeddings[s] for s in sentences], dtype=np.float32)

        norms = np.linalg.norm(sentence_embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        sentence_embeddings_matrix /= norms

        norms = np.linalg.norm(cache_embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        cache_embeddings_matrix /= norms

        if k == 1:
            best_indices = np.zeros(len(sentences), dtype=np.int32)
            best_scores = np.zeros(len(sentences), dtype=np.float32)
        else:
            best_indices = np.zeros((len(sentences), k), dtype=np.int32)
            best_scores = np.zeros((len(sentences), k), dtype=np.float32)
            
        decision_methods = np.full(len(sentences), "neural", dtype=object)

        for start in tqdm(
            range(0, len(sentences), batch_size),
            desc="Calculating best matches with cache...",
            disable=len(sentences) // batch_size < 10,
        ):
            end = min(start + batch_size, len(sentences))
            sentence_embedding = sentence_embeddings_matrix[start:end]

            batch_sims = sentence_embedding @ cache_embeddings_matrix.T  # (batch_size, cache_size)
            row_indices = np.arange(end - start)  # (batch_size)
            col_indices = np.arange(start, end)

            if (
                sentence_offset
            ):  # if we are calculating the best matches for a subset of sentences, we need to ignore the self-similarity
                col_indices += sentence_offset

            if mask_self_similarity:
                batch_sims[row_indices, col_indices] = -np.inf

            if k == 1:
                best_indices_batch = np.argmax(
                    batch_sims, axis=1
                )  # we want to find the best match for each sentence in the batch (batch_size)
                best_scores_batch = batch_sims[
                    row_indices, best_indices_batch
                ]  # we want to find the best score for each sentence in the batch (batch_size)
            else:
                # Top k
                if batch_sims.shape[1] <= k:
                    # Less candidates than k
                     best_indices_batch = np.argsort(-batch_sims, axis=1)
                     best_scores_batch = np.take_along_axis(batch_sims, best_indices_batch, axis=1)
                else:
                    part_idx = np.argpartition(-batch_sims, k, axis=1)[:, :k]
                    top_k_val = np.take_along_axis(batch_sims, part_idx, axis=1)
                    sorted_sub_idx = np.argsort(-top_k_val, axis=1)
                    best_scores_batch = np.take_along_axis(top_k_val, sorted_sub_idx, axis=1)
                    best_indices_batch = np.take_along_axis(part_idx, sorted_sub_idx, axis=1)

            best_indices[start:end] = best_indices_batch
            best_scores[start:end] = best_scores_batch
            decision_methods[start:end] = "neural"

        return best_indices, best_scores, decision_methods
