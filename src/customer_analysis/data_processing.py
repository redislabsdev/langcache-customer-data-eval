import pandas as pd
from src.customer_analysis.file_handler import FileHandler

RANDOM_SEED = 42


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


def load_data(query_log_path, cache_path: str = None, n_samples: int = 100):
    """
    Load query log and cache from either local paths or S3 URIs.
    """
    # Load the query log
    query_log_handler = FileHandler(query_log_path)
    query_log = query_log_handler.read_csv()

    # Load cache if provided
    if cache_path:
        cache_handler = FileHandler(cache_path)
        cache = cache_handler.read_csv()
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
