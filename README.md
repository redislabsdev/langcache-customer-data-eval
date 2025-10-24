# Customer Evaluation Pipeline
Evaluate how a semantic cache performs on your dataset by computing key KPIs over a threshold sweep and producing plots/CSVs:

The pipeline finds nearest matches for each user query using text embeddings, optionally asks an LLM to judge similarity, computes metrics across score thresholds, and generates plots — with support for **local or S3** inputs/outputs and optional **GPU acceleration**.

> Why does the full analysis mode require an LLM? We use an **LLM-as-a-Judge** to produce proxy ground‑truth labels for each `(query, match)` pair, so you can calculate precision without manual annotation.

## ✨ Features

* **Two evaluation modes in one script**
  Choose full LLM-judged metrics (`evaluation.py --full`) or fast cache-hit-ratio-only analysis (`evaluation.py`).
* **Conditional LLM dependency**
  The `llm-sim-eval` package is only required for `--full` mode. Run CHR-only analysis without it.
* **Two‑stage scoring** *(--full mode only)*
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
**Install LLM-as-a-Judge**:
- Configure `~/.pip/pip.conf`
    - [Locate the package](https://artifactory.dev.redislabs.com/ui/packages/pypi:%2F%2Fllm-sim-eval/0.2.0)
    - Set me up (Client: `pip`)
    - If you want to use `uv`
        `uv add llm-sim-eval==x.x.x --index=...`

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
> In all cases the script can be run with and without an explicit cache set. If you pass only `--query_log_path` and omit `--cache_path` as an argument, the script randomly samples `n_samples` to use as queries and uses the rest as cache.

**CHR-only mode (default - no LLM required):**

```bash
# Fast cache hit ratio analysis
uv run evaluation.py \
  --query_log_path ./data/queries.csv \
  --cache_path ./data/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 100 \
  --model_name "redis/langcache-embed-v3.1"
```

**Full evaluation mode (requires llm-sim-eval):**

```bash
# With Redis (recommended for better performance)
uv run evaluation.py \
  --query_log_path ./data/queries.csv \
  --cache_path ./data/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --llm_name "microsoft/Phi-4-mini-instruct" \
  --full \
  --use_redis

# Without Redis (in-memory matching)
uv run evaluation.py \
  --query_log_path ./data/queries.csv \
  --cache_path ./data/cache.csv \
  --sentence_column text \
  --output_dir ./outputs \
  --n_samples 1000 \
  --model_name "redis/langcache-embed-v1" \
  --llm_name "microsoft/Phi-4-mini-instruct" \
  --full
```

**S3 example**

```bash
# CHR-only with S3
uv run evaluation.py \
  --query_log_path s3://my-bucket/eval/queries.csv \
  --cache_path s3://my-bucket/eval/cache.csv \
  --sentence_column text \
  --output_dir s3://my-bucket/eval/results

# Full evaluation with S3
uv run evaluation.py \
  --query_log_path s3://my-bucket/eval/queries.csv \
  --cache_path s3://my-bucket/eval/cache.csv \
  --sentence_column text \
  --output_dir s3://my-bucket/eval/results \
  --full
```

> S3 access relies on your environment (e.g., `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).

---

## What the script does

> High‑level flow (see code for details):

### Common stages (both modes):

1. **Load data**
   `load_data(query_log_path, cache_path, n_samples)` → `queries` and `cache` DataFrames.

2. **Stage 1 — Matching**
   Uses `NeuralEmbedding(model_name, device=auto)` and
   `calculate_best_matches_with_cache_large_dataset(...)` to find top matches for each query (batch size `512`).

### CHR-only mode (default):

3. **Threshold sweep for cache hit ratios**
   Sweeps thresholds from min score to 1.0, calculating CHR at each point.
   **Output:** `chr_sweep.csv`, `chr_matches.csv`

4. **Plotting & summary**
   Generates `chr_vs_threshold.png` and prints summary statistics (score distribution, CHR at common thresholds).

### Full mode (--full flag):

3. **Stage 2 — LLM‑as‑a‑Judge**
   Pairs `(query, match)` are scored by `run_llm_local_sim_prediction_pipeline(...)` using an **empty prompt** loaded via `DEFAULT_PROMPTS`.
   **Output:** `llm_as_a_judge_results.csv` with `[<sentence_column>, matches, similarity_score, actual_label]` after post‑processing.

4. **Stage 3 — Metrics & threshold sweep**
   `postprocess_results_for_metrics(...)` prepares a final frame, then `sweep_thresholds_on_results(...)` evaluates metrics across thresholds.
   **Output:** `threshold_sweep_results.csv`, `matches.csv`

5. **Stage 4 — Plotting**
   `generate_plots(...)` writes:

   * `precision_vs_cache_hit_ratio.png`
   * `metrics_over_threshold.png`

Finally prints `Done!`.

---

## Command‑line arguments

### evaluation.py

| Flag                   | Type | Required | Default                           | Description                                                         |
| ---------------------- | ---: | :------: | --------------------------------- | ------------------------------------------------------------------- |
| `--query_log_path`     |  str |     ✅    | —                                 | Path to the **queries CSV** (local or `s3://…`).                    |
| `--sentence_column`    |  str |     ✅    | —                                 | Name of the text column to evaluate (must exist in both CSVs).      |
| `--output_dir`         |  str |     ✅    | —                                 | Where to write CSVs/plots (local or `s3://…`).                      |
| `--n_samples`          |  int |          | `100`                             | Number of samples to evaluate (default: 100).                       |
| `--model_name`         |  str |          | `"redis/langcache-embed-v3.1"`    | Embedding model passed to `NeuralEmbedding`.                        |
| `--cache_path`         |  str |          | `None`                            | Path to the **cache CSV** (local or `s3://…`).                      |
| `--full`               | flag |          | `False`                           | Run full evaluation with LLM-as-a-Judge (requires llm-sim-eval).    |
| `--llm_name`           |  str |          | `"microsoft/Phi-4-mini-instruct"` | Local LLM identifier (only used with `--full`).                     |
| `--sweep_steps`        |  int |          | `200`                             | Number of threshold steps in sweep (default: 200).                  |
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

### CHR-only mode (default):
* `chr_matches.csv` — `[<sentence_column>, matches, best_scores]`
* `chr_sweep.csv` — `[threshold, cache_hit_ratio]`
* `chr_vs_threshold.png` — Plot of cache hit ratio vs threshold

### Full mode (--full):
* `matches.csv` — `[<sentence_column>, matches, best_scores]`
* `llm_as_a_judge_results.csv` — `[<sentence_column>, matches, similarity_score, actual_label]`
* `threshold_sweep_results.csv` — thresholded metrics across the sweep
* `precision_vs_cache_hit_ratio.png` and `metrics_over_threshold.png`

### Metrics, charts, and files generated

* **Metrics** (computed per threshold and saved in `threshold_sweep_results.csv`):
  * `threshold`
  * `precision`, `recall`, `f1_score`, `f0_5_score`
  * `f05_chr_score` — harmonic mean of precision and cache hit ratio (β=0.5)
  * `cache_hit_ratio`
  * `tp`, `fp`, `fn`, `tn`, `accuracy`

* **Charts** (saved under `--output_dir`):
  * `precision_vs_cache_hit_ratio.png` — Precision vs Cache Hit Ratio
  * `metrics_over_threshold.png` — Over threshold: Precision, Cache Hit Ratio, and `precision * cache_hit_ratio`

* **Files** (saved under `--output_dir`):
  * `matches.csv` — `[<sentence_column>, matches, best_scores]`
  * `llm_as_a_judge_results.csv` — `[<sentence_column>, matches, similarity_score, actual_label]`
  * `threshold_sweep_results.csv` — one row per threshold with the metrics listed above

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

## Choosing the right mode

**Use CHR-only mode (default) when:**
- You want to quickly understand cache hit ratio characteristics
- You don't need precision/recall metrics (only CHR)
- You want to avoid the overhead of running an LLM judge
- You're doing exploratory analysis on embedding model performance
- The `llm-sim-eval` package is not available in your environment

**Use full mode (--full) when:**
- You need precision, recall, and F-scores
- You have labeled data or need LLM-judged similarity labels
- You want the full evaluation pipeline with quality metrics
- You need to understand both cache efficiency (CHR) and accuracy (precision)

**CLI help**

```bash
# See all available options
uv run evaluation.py -h
```

[1]: https://github.com/redis-applied-ai/redis-model-store "GitHub - redis-applied-ai/redis-model-store: AI/ML model versioning and storage engine backed by Redis."
