"""Hard spending cap for LLM calls (Adil: at most $20 on this project).

Every Bedrock call goes through BudgetGuard:
  1. before the call: estimate its worst-case cost; refuse if spent + estimate would pass the cap;
  2. after the call: record the real token usage and its cost in a ledger file on disk.
The ledger survives restarts, so the cap holds across notebooks and sessions. AWS billing updates hours late,
so this local guard is the fast stop; an AWS Budget action is the backstop.

Prices are Anthropic's list prices (USD per million tokens). Bedrock's own table could not be read from here;
compare the ledger total with the AWS bill after the first runs and update PRICES if they differ.
"""
import fcntl
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PRICES = {  # model id -> USD per million tokens
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
}
CHARS_PER_TOKEN = 3.0      # deliberately low (English is ~4), so input estimates err on the high side


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    _lock = threading.Lock()            # parallel calls in one process: one ledger update at a time

    def __init__(self, ledger_path, cap_usd=15.0):
        self.path = Path(ledger_path)
        self.cap = cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"cap_usd": cap_usd, "calls": []}, indent=1))

    def _ledger(self):
        return json.loads(self.path.read_text())

    def spent(self):
        return round(sum(c["cost_usd"] for c in self._ledger()["calls"]), 6)

    @staticmethod
    def cost(model_id, usage):
        p = PRICES[model_id]
        return (usage.get("inputTokens", 0) * p["input"] + usage.get("outputTokens", 0) * p["output"]
                + usage.get("cacheReadInputTokens", 0) * p["cache_read"]
                + usage.get("cacheWriteInputTokens", 0) * p["cache_write"]) / 1e6

    def estimate(self, model_id, prompt_chars, max_output_tokens):
        """Worst case: input estimated generously, output at the full max_tokens."""
        return self.cost(model_id, {"inputTokens": prompt_chars / CHARS_PER_TOKEN, "outputTokens": max_output_tokens})

    def check(self, estimate_usd):
        spent = self.spent()
        if spent + estimate_usd > self.cap:
            raise BudgetExceeded(f"LLM budget: ${spent:.4f} spent of ${self.cap:.2f}; this call could cost up to "
                                 f"${estimate_usd:.4f}, so it was not sent")

    def record(self, model_id, usage, purpose):
        """Threads in one process take the in-process lock; separate processes (several app workers, a notebook) take a
        file lock, so no ledger update is lost."""
        with self._lock, open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._record(model_id, usage, purpose)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _record(self, model_id, usage, purpose):
        ledger = self._ledger()
        cost = self.cost(model_id, usage)
        ledger["calls"].append({"time": datetime.now(timezone.utc).isoformat(), "purpose": purpose, "model": model_id,
                                "usage": {k: usage.get(k, 0) for k in ("inputTokens", "outputTokens",
                                                                        "cacheReadInputTokens", "cacheWriteInputTokens")},
                                "cost_usd": cost})
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=1))
        tmp.replace(self.path)
        return cost


def _is_throttle(err):
    code = getattr(err, "response", {}).get("Error", {}).get("Code", "")
    return code in ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException")


def guarded_converse(client, guard, purpose, modelId, messages, inferenceConfig, retries=8, base_wait=2.0, **kwargs):
    """client.converse(...) behind the budget guard. Returns the Bedrock response.

    When Bedrock says "too many requests", wait (2, 4, 8 ... s, at most 60 s) and retry; a refused call is not
    billed and is not recorded. Other errors are raised at once."""
    chars = len(json.dumps(messages)) + len(json.dumps(kwargs.get("system", ""))) + len(json.dumps(kwargs.get("toolConfig", "")))
    guard.check(guard.estimate(modelId, chars, inferenceConfig.get("maxTokens", 1000)))
    for attempt in range(retries):
        try:
            resp = client.converse(modelId=modelId, messages=messages, inferenceConfig=inferenceConfig, **kwargs)
            break
        except Exception as err:
            if not _is_throttle(err) or attempt == retries - 1:
                raise
            time.sleep(min(60.0, base_wait * 2 ** attempt))
    guard.record(modelId, resp["usage"], purpose)
    return resp
