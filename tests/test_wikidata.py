import json

import pytest

from src.viberec import wikidata as wd


def test_resumes_after_a_crash(tmp_path):
    ids = [f"tt{i:07d}" for i in range(10)]
    calls = []

    def flaky(batch):
        calls.append(batch[0])
        if batch[0] == "tt0000006":
            raise ConnectionError("network down")
        return [{"imdb": i} for i in batch]

    with pytest.raises(RuntimeError):
        wd.fetch_in_batches(ids, tmp_path, flaky, batch_size=3, pause=0, retries=2, backoff=0)
    assert sorted(p.name for p in tmp_path.glob("batch_*.json")) == ["batch_00000.json", "batch_00001.json"]
    assert not list(tmp_path.glob("*.tmp"))                  # no half-written file left behind

    calls.clear()
    fetched, skipped = wd.fetch_in_batches(ids, tmp_path, lambda b: [{"imdb": i} for i in b],
                                           batch_size=3, pause=0)
    assert (fetched, skipped) == (2, 2)                     # only the missing batches are fetched
    assert sorted(wd.load_batches(tmp_path)["imdb"]) == ids


def test_retry_then_success(tmp_path):
    attempts = {"n": 0}

    def once_failing(batch):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("blip")
        return [{"imdb": i} for i in batch]

    assert wd.fetch_in_batches(["tt1", "tt2"], tmp_path, once_failing, batch_size=5, pause=0, backoff=0) == (1, 0)


def test_refuses_a_different_id_list(tmp_path):
    wd.fetch_in_batches(["tt1", "tt2"], tmp_path, lambda b: [], batch_size=5, pause=0)
    with pytest.raises(ValueError):
        wd.fetch_in_batches(["tt1", "tt3"], tmp_path, lambda b: [], batch_size=5, pause=0)
