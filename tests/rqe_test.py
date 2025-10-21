# test_matching.py

import pandas as pd
import pytest
import redis

from chr_analysis import run_matching_redis


class DummyArgs:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_index_name = "test_idx"
        self.redis_doc_prefix = "cache:"
        self.distance_metric = "COSINE"
        self.index_algorithm = "FLAT"  # or “HNSW”
        self.n_samples = 1
        self.sentence_column = "sentence"


@pytest.fixture(scope="module")
def redis_client():
    r = redis.Redis(host="localhost", port=6379, decode_responses=False)
    yield r
    # clean up: drop index, flush docs
    try:
        r.ft("test_idx").dropindex(delete_documents=True)
    except Exception:
        pass
    r.flushdb()


def test_run_matching_basic(redis_client):
    args = DummyArgs()
    # define small dummy cache and queries
    cache = pd.DataFrame({"sentence": ["apple pie recipe", "how to bake bread", "best chocolate cake"]})
    queries = pd.DataFrame({"sentence": ["baking bread instructions", "making chocolate dessert"]})

    # run matching
    out_df = run_matching_redis(queries.copy(), cache.copy(), args)

    # Check output columns
    assert "matches" in out_df.columns
    assert "best_scores" in out_df.columns

    # Check data types
    assert len(out_df) == len(queries)
    for idx, row in out_df.iterrows():
        assert isinstance(row["matches"], str) or row["matches"] is None
        # Score should be numeric or None
        assert isinstance(row["best_scores"], float) or row["best_scores"] is None

    # Check that the matches make sense: for query[0] about baking bread, expect “how to bake bread”
    assert "bread" in out_df.iloc[0]["matches"].lower()


def test_switch_to_hnsw(redis_client):
    args = DummyArgs()
    args.index_algorithm = "HNSW"
    args.n_samples = 2

    cache = pd.DataFrame({"sentence": ["cat video", "dog video", "funny pet clip"]})
    queries = pd.DataFrame({"sentence": ["funny dog clip", "cute cat video"]})

    out_df = run_matching_redis(queries.copy(), cache.copy(), args)

    assert len(out_df) == 2
    # Ensure HNSW didn't break: scores should still exist
    assert out_df["best_scores"].notnull().any()
