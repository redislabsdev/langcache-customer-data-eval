import os
import re


def get_model_name_from_dir(dir_name):
    # Try to reconstruct meaningful model name from directory name
    # e.g. neural_redis_langcache-embed-v1 -> redis/langcache-embed-v1
    # e.g. Alibaba-NLP_gte-modernbert-base -> Alibaba-NLP/gte-modernbert-base

    # If dir_name starts with neural_, strip it
    name = dir_name
    if name.startswith("neural_"):
        name = name[7:]

    # Reconstruct slash
    known_prefixes = ["redis", "Alibaba-NLP"]
    for prefix in known_prefixes:
        if name.startswith(prefix + "_"):
            return name.replace(prefix + "_", prefix + "/", 1)

    # If we have timestamps in the name (old format), strip them
    # Pattern: name_YYYY_MM_DD...
    match = re.search(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", name)
    if match:
        name = name[: match.start()]
        # Try prefix fix again
        for prefix in known_prefixes:
            if name.startswith(prefix + "_"):
                return name.replace(prefix + "_", prefix + "/", 1)

    return name


def crawl_results(base_dir):
    # Structure: base_dir / dataset / model / run_N / timestamp / csv
    # OR old structure: base_dir / dataset / model_timestamp / csv

    results = {}  # dataset -> model -> list of run paths

    if not os.path.exists(base_dir):
        return results

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        results[dataset] = {}

        # List subdirs (models)
        for model_dir in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            # Check if this is a model directory (contains run_X) or a timestamped dir (old format)
            if any(sub.startswith("run_") for sub in os.listdir(model_path)):
                # New format: model_dir is the model name (safe)
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []

                for run_dir in os.listdir(model_path):
                    if not run_dir.startswith("run_"):
                        continue
                    run_path = os.path.join(model_path, run_dir)
                    # Inside run_dir, there should be a timestamped dir
                    timestamp_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
                    if timestamp_dirs:
                        # Take the latest one if multiple?
                        timestamp_dirs.sort()
                        final_path = os.path.join(run_path, timestamp_dirs[-1])
                        results[dataset][model_name].append(final_path)

            elif "threshold_sweep_results.csv" in os.listdir(model_path):
                # Old format: model_path IS the run path
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []
                results[dataset][model_name].append(model_path)

    return results
