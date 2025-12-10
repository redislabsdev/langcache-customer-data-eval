import pandas as pd
import numpy as np
import torch

from src.customer_analysis.embedding_interface import NeuralEmbedding
from src.customer_analysis.file_handler import FileHandler
from src.customer_analysis.query_engine import RedisVectorIndex

RANDOM_SEED = 42


def run_matching_redis(queries: pd.DataFrame, cache: pd.DataFrame, args):
    """
    SentenceTransformer + RedisVL implementation of your 5 steps:
      1) initialize the index
      2) embed + load the cache into Redis
      3) embed and search all queries against the cache
      4) attach results: best_scores (cosine sim) & matches (text)
      5) drop index and free memory (DataFrame results persist)
    Expects:
      - args.sentence_column
      - args.model_name (SentenceTransformer name or local path)
      - optional: args.n_samples, args.index_name, args.prefix, args.redis_url, args.batch_size, args.device
    """
    text_col = args.sentence_column

    # Determine k for retrieval
    k = 1
    cross_encoder = None
    if getattr(args, "cross_encoder_model", None):
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(
                args.cross_encoder_model, 
                device=getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            k = getattr(args, "rerank_k", 10)
            print(f"Using Cross-Encoder reranking: {args.cross_encoder_model} (top-{k})")
        except ImportError:
            print("Warning: sentence_transformers not found or CrossEncoder import failed. Skipping reranking.")

    rindex = RedisVectorIndex(
        col_query=text_col,
        index_name=getattr(args, "redis_index_name", "idx_cache_match"),
        prefix=getattr(args, "redis_doc_prefix", "cache:"),
        model_name=args.model_name,
        redis_url=getattr(args, "redis_url", "redis://localhost:6379"),
        device=getattr(args, "device", None),
        batch_size=getattr(args, "redis_batch_size", 256),
        additional_fields=[],
    )

    try:
        # 2) embed + load cache
        cache_texts = cache[text_col].tolist()
        cache_vecs = rindex._embed_batch(cache_texts)  # (M, D)
        
        # Fix: Ensure vectors are normalized and float32
        norms = np.linalg.norm(cache_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        cache_vecs = (cache_vecs / norms).astype(np.float32)
        
        rindex.load_texts_and_vecs(cache_texts, cache_vecs)

        # 3) embed queries and search top-k
        query_texts = queries[text_col].tolist()
        query_vecs = rindex._embed_batch(query_texts)
        
        # Normalize queries too and ensure float32
        norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        query_vecs = (query_vecs / norms).astype(np.float32)

        best_scores: list[float] = []
        matches: list[str] = []

        if cross_encoder and k > 1:
            all_pairs = []
            query_candidate_counts = []
            candidates_list = []  # store candidates for each query to retrieve text later
            
            print("Retrieving candidates from Redis...")
            for i, qv in enumerate(query_vecs):
                resp = rindex.query_vector_topk(qv, k=k)
                if not resp:
                    query_candidate_counts.append(0)
                    candidates_list.append([])
                    continue
                
                q_text = query_texts[i]
                cands = [r[text_col] for r in resp]
                candidates_list.append(cands)
                
                for c_text in cands:
                    all_pairs.append([q_text, c_text])
                
                query_candidate_counts.append(len(cands))
                
            if all_pairs:
                print(f"Reranking {len(all_pairs)} pairs with Cross-Encoder...")
                all_scores = cross_encoder.predict(all_pairs, batch_size=32, show_progress_bar=True)
                
                # Reassemble
                score_idx = 0
                for i, count in enumerate(query_candidate_counts):
                    if count == 0:
                        best_scores.append(0.0)
                        matches.append("")
                        continue
                        
                    # Get scores for this query
                    q_scores = all_scores[score_idx : score_idx + count]
                    score_idx += count
                    
                    best_idx = np.argmax(q_scores)
                    best_scores.append(float(q_scores[best_idx]))
                    matches.append(candidates_list[i][best_idx])
            else:
                best_scores = [0.0] * len(queries)
                matches = [""] * len(queries)

        else:
            for qv in query_vecs:
                resp = rindex.query_vector_topk(qv, k=1)
                if not resp:
                    best_scores.append(0.0)
                    matches.append("")
                    continue

                hit = resp[0]
                cosine_sim = 1.0 - float(hit["vector_distance"])  # convert to similarity
                best_scores.append(cosine_sim)
                matches.append(hit[text_col])

        # 4) attach outputs
        out = queries.copy()
        out["best_scores"] = best_scores
        out["matches"] = matches

    finally:
        # 5) clear index + free GPU mem (if any)
        rindex.drop()

    return out


def run_matching(queries, cache, args):
    embedding_model = NeuralEmbedding(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine k for retrieval
    k = 1
    cross_encoder = None
    if getattr(args, "cross_encoder_model", None):
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(
                args.cross_encoder_model, 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            k = getattr(args, "rerank_k", 10)
            print(f"Using Cross-Encoder reranking: {args.cross_encoder_model} (top-{k})")
        except ImportError:
            print("Warning: sentence_transformers not found or CrossEncoder import failed. Skipping reranking.")

    queries["best_scores"] = 0

    query_list = queries[args.sentence_column].to_list()
    cache_list = cache[args.sentence_column].to_list()

    best_indices, best_scores, decision_methods = embedding_model.calculate_best_matches_with_cache_large_dataset(
        queries=query_list,
        cache=cache_list,
        batch_size=512,
        k=k
    )

    if cross_encoder and k > 1:
        print("Reranking results with Cross-Encoder...")
        # best_indices is (N, k)
        all_pairs = []
        N = len(query_list)
        
        for i in range(N):
            q_text = query_list[i]
            for idx in best_indices[i]:
                all_pairs.append([q_text, cache_list[idx]])
                
        if all_pairs:
            all_scores = cross_encoder.predict(all_pairs, batch_size=128, show_progress_bar=True)
            all_scores = all_scores.reshape(N, k)
            
            best_idx_in_k = np.argmax(all_scores, axis=1) # (N,)
            
            final_scores = all_scores[np.arange(N), best_idx_in_k]
            final_cache_indices = best_indices[np.arange(N), best_idx_in_k]
            
            queries["best_scores"] = final_scores
            queries["matches"] = [cache_list[i] for i in final_cache_indices]
        else:
            queries["best_scores"] = 0.0
            queries["matches"] = ""
    else:
        queries["best_scores"] = best_scores
        queries["matches"] = cache.iloc[best_indices][args.sentence_column].to_list()

    del embedding_model
    torch.cuda.empty_cache()

    return queries


def postprocess_results_for_metrics(queries, llm_df, args):
    llm_df_deduped = llm_df.drop_duplicates(subset=[args.sentence_column, "matches"])
    final_df = pd.merge(
        queries,
        llm_df_deduped,
        left_on=[args.sentence_column, "matches"],
        right_on=[args.sentence_column, "matches"],
        how="left",
    )
    # rename columns in case they exist in both dataframes
    # this is to avoid merge errors
    if "similarity_score" in final_df.columns:
        final_df.rename(columns={"similarity_score": "similarity_score_old"}, inplace=True)
    if "actual_label" in final_df.columns:
        final_df.rename(columns={"actual_label": "actual_label_old"}, inplace=True)
    final_df.rename(columns={"response": "actual_label", "best_scores": "similarity_score"}, inplace=True)

    final_df["actual_label"] = (final_df["actual_label"] == "yes").astype(int)

    return final_df[[args.sentence_column, "matches", "similarity_score", "actual_label"]]


def load_data(query_log_path: str, cache_path: str = None, n_samples: int = 100):
    """
    Load query log and cache from either local paths or S3 URIs.
    """
    # Load the query log
    query_log = FileHandler.read_csv(query_log_path)

    # Load cache if provided
    if cache_path:
        cache = FileHandler.read_csv(cache_path)
    else:
        cache = None

    # Sample n_samples for queries
    if n_samples < len(query_log):
        queries = query_log.sample(n_samples, random_state=RANDOM_SEED)
    else:
        queries = query_log

    # If no explicit cache, use remaining rows
    if cache is None:
        cache = query_log.drop(queries.index)

    # Shuffle cache to remove order bias
    cache = cache.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return queries, cache
