# Local usage:
uv run cache_evaluation.py \
    --query_log_path "dataset/chatgpt.csv" \
    --sentence_column sentence2 \
    --output_dir test_outputs \
    --n_samples 100 \
    --model_name redis/langcache-embed-v1
    # --cache_path "dataset/cache.csv" \

# S3 usage example:
# uv run cache_evaluation.py \
#     --query_log_path "s3://redis-ai-research-customer-datasets/7_day_cache.csv" \
#     --sentence_column transcription \
#     --output_dir "s3://redis-ai-research-customer-datasets/vizio/v1/customer_analysis/langcache-minilm" \
#     --n_samples 1000 \
#     --model_name redis/langcache-embed-experimental