uv run evaluation.py \
  --query_log_path dataset/mangoes_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./mangoes/v2 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v2" \
  --full \

uv run evaluation.py \
  --query_log_path dataset/mangoes_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./mangoes/v3 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v3" \
  --full \

uv run evaluation.py \
  --query_log_path dataset/mangoes_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./mangoes/v3.1 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v3.1" \
  --full \

uv run evaluation.py \
  --query_log_path dataset/mangoes_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./mangoes/v1 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --full 


# ================================

uv run evaluation.py \
  --query_log_path dataset/chatgpt_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./rado_synthetic/v1 \
  --n_samples 500 \
  --model_name "redis/langcache-embed-v1" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/chatgpt_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./rado_synthetic/v2 \
  --n_samples 500 \
  --model_name "redis/langcache-embed-v2" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/chatgpt_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./rado_synthetic/v3 \
  --n_samples 500 \
  --model_name "redis/langcache-embed-v3" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/chatgpt_unique_sentences.csv \
  --sentence_column sentence \
  --output_dir ./rado_synthetic/v3.1 \
  --n_samples 500 \
  --model_name "redis/langcache-embed-v3.1" \
  --full 

# ================================

uv run evaluation.py \
  --query_log_path dataset/vizio_unique_sentences.csv \
  --sentence_column transcription \
  --output_dir ./vizio/v1 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/vizio_unique_sentences.csv \
  --sentence_column transcription \
  --output_dir ./vizio/v2 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v2" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/vizio_unique_sentences.csv \
  --sentence_column transcription \
  --output_dir ./vizio/v3 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v3" \
  --full 

uv run evaluation.py \
  --query_log_path dataset/vizio_unique_sentences.csv \
  --sentence_column transcription \
  --output_dir ./vizio/v3.1 \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v3.1" \
  --full 