# Customer Evaluation Pipeline

Run evaluations for semantic caching with neural embeddings and optional LLM-as-a-Judge:

- **`customer_evaluation.py`**: Full pipeline with LLM judge → precision, recall, F1, and cache hit ratio metrics
- **`chr_analysis.py`**: Fast cache hit ratio analysis without LLM judge → threshold sweep and CHR plots

Both scripts support **local or S3** inputs/outputs and optional **GPU acceleration**.

## ✨ Features

* **Two evaluation modes**
  Choose full LLM-judged metrics (`customer_evaluation.py`) or fast cache-hit-ratio-only analysis (`chr_analysis.py`).
* **Two‑stage scoring** *(customer_evaluation.py only)*
  Neural embedding matching followed by **LLM‑as‑a‑Judge** for higher‑quality similarity signals.
* **Metrics & plots out of the box**
  Saves CSVs and generates threshold‑sweep visualizations to help tune decision thresholds.
* **Local *and* S3 I/O**
  Read inputs and write outputs to `s3://…` or local paths; no code changes needed.
* **Deterministic runs**
  Seeds are set for `random`, `numpy`, and `torch` to improve reproducibility.
* **GPU‑aware**
  Uses CUDA automatically when available; falls back to CPU otherwise.


---

## Quick Start

### 1) Installation

This project uses **[uv](https://docs.astral.sh/uv/)** for fast, reliable Python package management.

**Install uv** (if you don't have it yet):

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install project dependencies**:

```bash
# Clone the repository (if needed)
git clone <your-repo-url>
cd langcache-customer-data-eval

# Install dependencies and sync the environment
uv sync

# Or if you want to include dev dependencies
uv sync --all-groups
```

> **Note:** To install `llm-sim-eval`, you need to follow the installation steps in [the Redis artifactory](https://artifactory.dev.redislabs.com/ui/packages/pypi:%2F%2Fllm-sim-eval).

### 2) Start Redis (Optional but Recommended)

The matching pipeline can use Redis for fast vector similarity search. To enable Redis-based matching, add the `--use_redis` flag and ensure Redis is running:

```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 redis/redis-stack:latest

# Or install Redis locally
# macOS: brew install redis && redis-server
# Ubuntu: sudo apt-get install redis-server && redis-server
# Windows: Download from https://redis.io/download
```

> **Note:** Redis connection defaults to `redis://localhost:6379`. You can customize this with `--redis_url`.

### 3) Prepare your data

* **Queries CSV** (`--query_log_path`): must include your **sentence column** (name passed via `--sentence_column`).
* **Cache CSV** (`--cache_path`): a catalog of reference sentences/utterances. Must at least include the **same sentence column**.

**Example (minimal)**

```csv
# queries.csv
id,text
1,"how do I reset my password?"
2,"store hours on sunday"

# cache.csv
id,text
101,"reset your password"
102,"our store hours"
```

### 4) Run

```bash
# With Redis (recommended for better performance)
uv run customer_evaluation.py \
  --query_log_path ./data/queries.csv \
  --cache_path ./data/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --llm_name "microsoft/Phi-4-mini-instruct" \
  --use_redis

# Without Redis (in-memory matching)
uv run customer_evaluation.py \
  --query_log_path ./data/queries.csv \
  --cache_path ./data/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --llm_name "microsoft/Phi-4-mini-instruct"
```

**S3 example**

```bash
uv run customer_evaluation.py \
  --query_log_path s3://my-bucket/eval/queries.csv \
  --cache_path s3://my-bucket/eval/cache.csv \
  --sentence_column text \
  --output_dir s3://my-bucket/eval/results
```

> S3 access relies on your environment (e.g., `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).

---

## What the script does

> High‑level flow (see code for details):

1. **Load data**
   `load_data(query_log_path, cache_path, n_samples)` → `queries` and `cache` DataFrames.

2. **Stage 1 — Matching**
   Uses `NeuralEmbedding(model_name, device=auto)` and
   `calculate_best_matches_with_cache_large_dataset(...)` to find top matches for each query (batch size `512`, early stop at `n_samples`).
   **Output:** `matches.csv` with `[<sentence_column>, matches, best_scores]`.

3. **Stage 2 — LLM‑as‑a‑Judge**
   Pairs `(query, match)` are scored by `run_llm_local_sim_prediction_pipeline(...)` using an **empty prompt** loaded via `DEFAULT_PROMPTS`.
   **Output:** `llm_as_a_judge_results.csv` with `[<sentence_column>, matches, similarity_score, actual_label]` after post‑processing.

4. **Stage 3 — Metrics & threshold sweep**
   `postprocess_results_for_metrics(...)` prepares a final frame, then `sweep_thresholds_on_results(...)` evaluates metrics across thresholds.
   **Output:** `threshold_sweep_results.csv`.

5. **Stage 4 — Plotting**
   `generate_plots(...)` writes:

   * `precision_vs_cache_hit_ratio.png`
   * `metrics_over_threshold.png`

Finally prints `Done!`.

---

## Command‑line arguments

### customer_evaluation.py

| Flag                   | Type | Required | Default                           | Description                                                         |
| ---------------------- | ---: | :------: | --------------------------------- | ------------------------------------------------------------------- |
| `--query_log_path`     |  str |     ✅    | —                                 | Path to the **queries CSV** (local or `s3://…`).                    |
| `--sentence_column`    |  str |     ✅    | —                                 | Name of the text column to evaluate (must exist in both CSVs).      |
| `--output_dir`         |  str |     ✅    | —                                 | Where to write CSVs/plots (local or `s3://…`).                      |
| `--n_samples`          |  int |          | `1000`                            | Max queries to evaluate (also used as an "early stop" in matching). |
| `--model_name`         |  str |          | `"redis/langcache-embed-v1"`      | Embedding model passed to `NeuralEmbedding`.                        |
| `--cache_path`         |  str |          | `None`                            | Path to the **cache CSV** (local or `s3://…`).                      |
| `--llm_name`           |  str |          | `"microsoft/Phi-4-mini-instruct"` | Local LLM identifier used by the judging pipeline.                  |
| `--use_redis`          | flag |          | `False`                           | Use Redis for vector matching (default: in-memory matching).        |
| `--redis_url`          |  str |          | `"redis://localhost:6379"`        | Redis connection URL for vector search.                             |
| `--redis_index_name`   |  str |          | `"idx_cache_match"`               | Redis index name for vector storage.                                |
| `--redis_doc_prefix`   |  str |          | `"cache:"`                        | Redis document key prefix.                                          |
| `--redis_batch_size`   |  int |          | `256`                             | Batch size for Redis vector operations.                             |

> Tip: `--model_name` and `--llm_name` must be supported by your environment/backends. The script auto‑selects `cuda` when `torch.cuda.is_available()` returns true.

---

## Inputs & outputs

**Inputs**

* `queries.csv` — must include `--sentence_column` (e.g., `text`). 
* `cache.csv` — must include the same `--sentence_column` used by queries.

**Outputs** (written under `--output_dir`)

* `matches.csv` — `[<sentence_column>, matches, best_scores]`
* `llm_as_a_judge_results.csv` — `[<sentence_column>, matches, similarity_score, actual_label]`
* `threshold_sweep_results.csv` — thresholded metrics across the sweep
* `precision_vs_cache_hit_ratio.png` and `metrics_over_threshold.png`

> When `--output_dir` starts with `s3://`, paths are joined with forward slashes; local directories use `os.path.join` logic.

---

## Performance & tuning

* **GPU**: The embedding stage will use CUDA if available; otherwise CPU. Free device memory after matching is reclaimed (`torch.cuda.empty_cache()`).
* **Batch sizes**: Matching uses `batch_size=512`. The LLM stage uses a conservative `batch_size=2` to reduce memory spikes; increase if your hardware allows.
* **Early stop**: `--n_samples` limits how many queries pass through heavy stages (useful for quick iteration).

---

## Extending the pipeline

* **Custom prompts**: Swap out the default *empty* prompt (`DEFAULT_PROMPTS["empty_prompt"]`) with your own prompt via `Prompt.load(...)`.
* **Alternative scorers**: Replace `NeuralEmbedding` or its `calculate_best_matches_with_cache_large_dataset` with your own implementation.
* **Metrics**: Extend `postprocess_results_for_metrics` and/or `sweep_thresholds_on_results` to add domain‑specific KPIs.

---

## Troubleshooting

* *"Connection refused" or "Error connecting to Redis"*:
  Ensure Redis is running on the specified `--redis_url`. Test with: `redis-cli ping` (should return `PONG`).
* *My run finishes matching but shows "Number of discarded queries…"*:
  That count is how many judge calls produced unusable/failed responses; the pipeline continues with successful ones.
* *Plots are empty or flat*:
  Ensure your inputs contain valid ground‑truth signals (`actual_label` or whatever your metrics function expects), and that scores vary across pairs.
* *S3 permissions errors*:
  Confirm AWS credentials in the environment and that your `FileHandler` is configured for those credentials/regions.
* *Redis index conflicts*:
  If you see errors about existing indexes, change `--redis_index_name` to a unique value or manually delete the old index with: `redis-cli FT.DROPINDEX <index_name> DD`

---

## Contributing

We welcome improvements — add new metrics, plots, or backends and send a PR. A helpful contribution flow is:

1. Create a branch:

   ```bash
   git checkout -b feat/<your-feature>
   ```
2. Add tests and docs (e.g., examples under a `docs/` folder).
3. Ensure type checks / formatting / tests pass.
4. Open a pull request with a clear description and before/after examples.

(Contributing steps follow the same spirit as the “development setup” section and workflow in the Redis Model Store README.) ([GitHub][1])

---

## Acknowledgments

* README structure inspired by the **Redis Model Store** project’s README. ([GitHub][1])

---

**CLI help**

```bash
# Full evaluation pipeline with LLM-as-a-Judge
uv run customer_evaluation.py -h

# Cache hit ratio analysis only
uv run chr_analysis.py -h
```

[1]: https://github.com/redis-applied-ai/redis-model-store "GitHub - redis-applied-ai/redis-model-store: AI/ML model versioning and storage engine backed by Redis."
