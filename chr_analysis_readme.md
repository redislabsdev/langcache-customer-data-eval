## Cache Hit Ratio Analysis (`chr_analysis.py`)

For faster analysis without the LLM-as-a-Judge step, use `chr_analysis.py`. This script focuses purely on **cache hit ratio** analysis by:

1. Running embedding-based matching to find the best cache entry for each query
2. Sweeping over threshold values to compute cache hit ratios
3. Generating plots and CSV outputs

**Key differences from `customer_evaluation.py`:**
- ❌ No LLM judge step → much faster, no GPU/LLM needed for judgment
- ✅ Only evaluates cache hit ratio as a function of similarity threshold
- ✅ Uses the same embedding and matching infrastructure
- ✅ Supports local and S3 paths

### Usage

```bash
uv run chr_analysis.py \
  --data_path ./dataset/chatgpt.csv \
  --sentence_column sentence1 \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1" \
  --sweep_steps 200
```

**S3 example**

```bash
uv run chr_analysis.py \
  --data_path s3://my-bucket/data.csv \
  --sentence_column text \
  --output_dir s3://my-bucket/chr-results
```

### Example output

```
==================================================
Cache Hit Ratio Analysis Summary
==================================================
Total queries analyzed: 100
Similarity score range: [0.5234, 0.9876]
Mean similarity score: 0.7845
Median similarity score: 0.7923

Cache Hit Ratios at common thresholds:
  Threshold 0.5: 98.00%
  Threshold 0.6: 94.00%
  Threshold 0.7: 76.00%
  Threshold 0.8: 45.00%
  Threshold 0.9: 12.00%
==================================================
```
### Arguments

| Flag                | Type | Required | Default                          | Description                                                    |
| ------------------- | ---: | :------: | -------------------------------- | -------------------------------------------------------------- |
| `--data_path`       |  str |     ✅    | —                                | Path to the CSV file with sentences (local or `s3://…`).       |
| `--sentence_column` |  str |     ✅    | —                                | Name of the column containing sentences to analyze.            |
| `--output_dir`      |  str |     ✅    | —                                | Where to write CSVs/plots (local or `s3://…`).                 |
| `--n_samples`       |  int |          | `100`                            | Number of queries to analyze (taken from start of dataset).    |
| `--model_name`      |  str |          | `"redis/langcache-embed-v3.1"`   | Embedding model name used to perform the ranking               |
| `--sweep_steps`     |  int |          | `200`                            | Number of threshold steps in the sweep.                        |

---
### How it works

1. **Load data**: Reads `--data_path` and splits it into:
   - **Queries**: First `n_samples` rows
   - **Cache**: Remaining rows

2. **Run matching**: Uses `NeuralEmbedding.calculate_best_matches_with_cache_large_dataset()` to find the best cache match for each query.

3. **Threshold sweep**: Sweeps from `min(similarity_scores)` to `1.0` in `--sweep_steps` increments, computing cache hit ratio at each threshold.

4. **Generate outputs**:
   - `chr_matches.csv` — `[<sentence_column>, matches, best_scores]`
   - `chr_sweep.csv` — `[threshold, cache_hit_ratio]`
   - `chr_vs_threshold.png` — Plot of cache hit ratio vs threshold

5. **Summary statistics**: Prints a summary including:
   - Similarity score distribution (min, max, mean, median)
   - Cache hit ratios at common thresholds (0.5, 0.6, 0.7, 0.8, 0.9)

--- 
### When to use `chr_analysis.py` vs `customer_evaluation.py`

**Use `chr_analysis.py` when:**
- You want to quickly understand cache hit ratio characteristics
- You don't need precision/recall metrics (only CHR)
- You want to avoid the overhead of running an LLM judge
- You're doing exploratory analysis on embedding model performance

**Use `customer_evaluation.py` when:**
- You need precision, recall, and F-scores
- You have labeled data or need LLM-judged similarity labels
- You want the full evaluation pipeline with quality metrics
