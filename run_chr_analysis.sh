uv run evaluation.py \
  --query_log_path ./dataset/queries.csv \
  --cache_path ./dataset/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1"

uv run evaluation.py \
  --query_log_path ./dataset/queries.csv \
  --cache_path ./dataset/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1" \
  --use_redis

uv run evaluation.py \
  --query_log_path ./dataset/queries.csv \
  --cache_path ./dataset/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1" \
  --full

uv run evaluation.py \
  --query_log_path ./dataset/queries.csv \
  --cache_path ./dataset/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1" \
  --full \
  --use_redis

uv run evaluation.py \
  --query_log_path ./dataset/chatgpt.csv \
  --sentence_column sentence2 \
  --output_dir ./outputs \
  --n_samples 20 \
  --model_name "redis/langcache-embed-v3.1" \
  --full \
  --use_redis
