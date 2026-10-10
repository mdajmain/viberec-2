"""VibeRec API (Step 13, app v2). Run locally:  customengine/bin/uvicorn app.main:app --port 8000

Settings come from the environment, so the same code runs on the Mac and on EC2:
  VIBEREC_DB_URL      PostgreSQL + pgvector (local Docker, or the database on the server)
  VIBEREC_DATA_DIR    where the LLM spend ledger and the Claude answer cache live
  VIBEREC_API_KEY     when set, every call except /health needs the header X-VibeRec-Key with this value; only the
                      web app's server (Vercel) knows it, so strangers cannot call the API and spend Claude credits
  AWS_REGION          Bedrock region (us-east-2); credentials: AWS_PROFILE on the Mac, the instance role on EC2
  VIBEREC_LLM_CAP     the spend cap the budget guard enforces (default 15 USD)
"""
import hmac
import os
import sys
from pathlib import Path
from typing import Literal

import boto3
import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.viberec import store                                        # noqa: E402
from src.viberec.llm_budget import BudgetExceeded, BudgetGuard       # noqa: E402
from src.viberec.serve import VERDICTS, RateLimited, VibeRecService  # noqa: E402

DB_URL = os.environ.get("VIBEREC_DB_URL", "postgresql://viberec:viberec-local-only@localhost:5433/viberec")
API_KEY = os.environ.get("VIBEREC_API_KEY", "")

app = FastAPI(title="VibeRec API", description="Vibe-first movie recommendations: search, taste quiz, personal feed.")
_service = None


def service():
    global _service
    if _service is None:
        data = Path(os.environ.get("VIBEREC_DATA_DIR", ROOT / "artifacts"))
        pool = ConnectionPool(DB_URL, min_size=1, max_size=4, open=True)
        with psycopg.connect(DB_URL) as conn:                        # its own connection: it registers the vector type
            taste = store.taste_model(conn)
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None,
                                region_name=os.environ.get("AWS_REGION", "us-east-2"))
        guard = BudgetGuard(data / "llm_spend_ledger.json", cap_usd=float(os.environ.get("VIBEREC_LLM_CAP", "15")))
        _service = VibeRecService(pool, session.client("bedrock-runtime"), guard, data / "app_cache", taste_model=taste)
    return _service


def require_key(x_viberec_key: str = Header(default="")):
    if API_KEY and not hmac.compare_digest(x_viberec_key, API_KEY):
        raise HTTPException(401, "missing or wrong API key")


def run(fn, *args, **kwargs):
    """Call the service and turn its errors into HTTP answers."""
    try:
        return fn(*args, **kwargs)
    except RateLimited as err:
        raise HTTPException(429, str(err))
    except BudgetExceeded:
        raise HTTPException(503, "The LLM budget for this demo is used up.")
    except KeyError as err:
        raise HTTPException(404, str(err).strip("'"))
    except ValueError as err:
        raise HTTPException(422, str(err))


class ProfileIn(BaseModel):
    guest_id: str | None = Field(default=None, description="the browser's guest id (g_ + 32 hex)")
    clerk_user_id: str | None = Field(default=None, description="set by the web app's server after Clerk verified the session")


class ReactIn(BaseModel):
    profile_id: str
    movie_id: int
    verdict: Literal["like", "dislike", "unseen", "not_this_vibe"]
    source: Literal["quiz", "search", "feed"]
    context_id: str | None = None


class SearchIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=500, examples=["Something like How to Lose a Guy in 10 Days, but more urban"])
    k: int = Field(default=5, ge=1, le=10)
    profile_id: str | None = None


class ProfileRef(BaseModel):
    profile_id: str


class FeedbackIn(BaseModel):                                   # the first page's buttons (kept until the new site is live)
    prompt: str
    movie_id: int
    verdict: str = Field(description=f"one of {sorted(VERDICTS)}")


@app.get("/health")
def health():
    with service().pool.connection() as conn:
        n = conn.execute("SELECT count(*) FROM movies").fetchone()[0]
    return {"ok": True, "movies": n}


@app.post("/profile", dependencies=[Depends(require_key)])
def profile(body: ProfileIn):
    if not body.guest_id and not body.clerk_user_id:
        raise HTTPException(422, "guest_id or clerk_user_id is required")
    return run(service().profile, body.guest_id, body.clerk_user_id)


@app.get("/quiz/next", dependencies=[Depends(require_key)])
def quiz_next(profile_id: str):
    return run(service().quiz_next, profile_id)


@app.post("/react", dependencies=[Depends(require_key)])
def react(body: ReactIn):
    return run(service().react, body.profile_id, body.movie_id, body.verdict, body.source, body.context_id)


@app.get("/feed", dependencies=[Depends(require_key)])
def feed(profile_id: str, k: int = 10):
    return run(service().feed, profile_id, max(1, min(k, 20)))


@app.post("/search", dependencies=[Depends(require_key)])
def search(body: SearchIn):
    return run(service().search, body.prompt.strip(), body.k, body.profile_id)


@app.get("/taste", dependencies=[Depends(require_key)])
def taste(profile_id: str):
    return run(service().taste_summary, profile_id)


@app.get("/history", dependencies=[Depends(require_key)])
def history(profile_id: str):
    return run(service().history, profile_id)


@app.post("/forget", dependencies=[Depends(require_key)])
def forget(body: ProfileRef):
    return run(service().forget, body.profile_id)


@app.post("/feedback", dependencies=[Depends(require_key)])
def feedback(body: FeedbackIn):
    run(service().feedback, body.prompt, body.movie_id, body.verdict)
    return {"ok": True}


@app.get("/")
def page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
