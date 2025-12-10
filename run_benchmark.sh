# Example usage:
uv run run_benchmark.py \
    --dataset_dir "dataset" \
    --output_dir "cross_encoder_results" \
    --models "Alibaba-NLP/gte-modernbert-base" "redis/langcache-embed-v1" "redis/langcache-embed-v3-small" \
    --dataset_names "vizio_unique_medium.csv" "axis_bank_unique_sentences.csv"\
    --sentence_column "sentence" \
    --n_runs 10 \
    --n_samples 10000 \
    --sample_ratio 0.8 \
    --llm_name "tensoropera/Fox-1-1.6B" \
    --full \
    --use_redis \
    # --cross_encoder_models "redis/langcache-reranker-v1-softmnrl-triplet" "Alibaba-NLP/gte-reranker-modernbert-base" \
    # --rerank_k 5