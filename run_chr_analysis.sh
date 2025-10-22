uv run evaluation.py --query_log_path dataset/chatgpt.csv \
    --sentence_column text \
    --output_dir output_files \
    --n_samples 500 \
    --model_name redis/langcache-embed-v3-mini \
    --use_redis \

