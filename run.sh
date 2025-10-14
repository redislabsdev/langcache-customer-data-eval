# Local usage:
uv run customer_evaluation.py \
    --query_log_path "7_day_cache.csv" \
    --sentence_column transcription \
    --output_dir vizio/v1/customer_analysis/langcache-minilm \
    --n_samples 1000 \
    --model_name redis/langcache-embed-experimental

# S3 usage example:
# uv run customer_evaluation.py \
#     --query_log_path "s3://redis-ai-research-customer-datasets/7_day_cache.csv" \
#     --sentence_column transcription \
#     --output_dir "s3://redis-ai-research-customer-datasets/vizio/v1/customer_analysis/langcache-minilm" \
#     --n_samples 1000 \
#     --model_name redis/langcache-embed-experimental