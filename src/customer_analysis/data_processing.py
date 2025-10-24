import pandas as pd
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
        rindex.load_texts_and_vecs(cache_texts, cache_vecs)

        # 3) embed queries and search top-1
        query_texts = queries[text_col].tolist()
        query_vecs = rindex._embed_batch(query_texts)

        best_scores: list[float] = []
        matches: list[str] = []

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

    queries["best_scores"] = 0

    best_indices, best_scores, decision_methods = embedding_model.calculate_best_matches_with_cache_large_dataset(
        queries=queries[args.sentence_column].to_list(),
        cache=cache[args.sentence_column].to_list(),
        batch_size=512,
    )

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
