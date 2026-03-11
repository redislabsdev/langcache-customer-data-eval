# Example usage:
uv run run_benchmark.py \
    --dataset_dir "dataset" \
    --output_dir "limitations-experiments-gte-modernbert-base-lora" \
    --models "redis/model-a-baseline" "redis/model-b-structured" \
    --dataset_names "sentencepairs_v3_unique_sentences.csv" "vizio_unique_medium.csv"\
    --sentence_column "sentence" \
    --n_runs 3 \
    --n_samples 16384 \
    --sample_ratio 0.8 \
    --llm_name "tensoropera/Fox-1-1.6B" \
    --full \
    --use_redis \
    # --cross_encoder_models "gemini/text-embedding-001" \
    # --rerank_k 1