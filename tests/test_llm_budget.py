import pytest

from src.viberec import llm_budget as lb

M = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class FakeClient:
    def __init__(self):
        self.calls = 0

    def converse(self, **kw):
        self.calls += 1
        return {"usage": {"inputTokens": 1000, "outputTokens": 200}, "output": {}}


def test_cost_and_ledger_survive_restarts(tmp_path):
    g = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=1.0)
    resp = lb.guarded_converse(FakeClient(), g, "test", M, [{"role": "user", "content": [{"text": "hi"}]}], {"maxTokens": 300})
    assert g.spent() == pytest.approx((1000 * 1 + 200 * 5) / 1e6)                 # $0.002
    assert lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=1.0).spent() == g.spent()


def test_refuses_before_sending_when_cap_would_be_passed(tmp_path):
    g = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=0.0021)
    client = FakeClient()
    lb.guarded_converse(client, g, "first", M, [{"role": "user", "content": [{"text": "hi"}]}], {"maxTokens": 100})
    with pytest.raises(lb.BudgetExceeded):
        lb.guarded_converse(client, g, "second", M, [{"role": "user", "content": [{"text": "hi"}]}], {"maxTokens": 100})
    assert client.calls == 1                                                          # the second call never went out


class Throttle(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "ThrottlingException"}}


class FlakyClient(FakeClient):
    def __init__(self, fails):
        super().__init__()
        self.fails = fails

    def converse(self, **kw):
        if self.fails:
            self.fails -= 1
            raise Throttle()
        return super().converse(**kw)


def test_throttling_is_retried_and_only_success_is_billed(tmp_path):
    g = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=1.0)
    client = FlakyClient(fails=2)
    lb.guarded_converse(client, g, "t", M, [{"role": "user", "content": [{"text": "hi"}]}], {"maxTokens": 10}, base_wait=0.0)
    assert client.calls == 1 and len(g._ledger()["calls"]) == 1


def test_other_errors_are_not_retried(tmp_path):
    g = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=1.0)

    class Broken:
        calls = 0

        def converse(self, **kw):
            Broken.calls += 1
            raise ValueError("bad request")
    with pytest.raises(ValueError):
        lb.guarded_converse(Broken(), g, "t", M, [{"role": "user", "content": [{"text": "hi"}]}], {"maxTokens": 10}, base_wait=0.0)
    assert Broken.calls == 1 and g.spent() == 0


def test_parallel_records_are_all_kept(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    guard = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=15.0)
    usage = {"inputTokens": 1000, "outputTokens": 100}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: guard.record(list(lb.PRICES)[0], usage, "t"), range(50)))
    assert len(guard._ledger()["calls"]) == 50
