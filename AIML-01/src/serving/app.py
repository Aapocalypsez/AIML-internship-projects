"""FastAPI scoring API; production startup fails if champion/Redis are unavailable."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from src.serving.scorer import Scorer, Transaction

SCORES = Counter("fraud_scores_total", "Successful score requests")
FAILURES = Counter("fraud_scoring_errors_total", "Failed score requests")
LATENCY = Histogram(
    "fraud_scoring_seconds", "Scoring latency", buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1)
)


def create_app(scorer=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.scorer = scorer or Scorer()
        yield

    app = FastAPI(title="Financial Fraud Detection", version="1.0", lifespan=lifespan)

    @app.post("/score")
    def score(transaction: Transaction):
        try:
            with LATENCY.time():
                result = app.state.scorer.score(transaction.model_dump())
            SCORES.inc()
            return result
        except (ValueError, ValidationError) as exc:
            FAILURES.inc()
            raise HTTPException(409 if "already exists" in str(exc) else 422, str(exc)) from exc
        except Exception as exc:
            FAILURES.inc()
            raise HTTPException(503, "Scoring temporarily unavailable; retry") from exc

    @app.get("/health")
    def health():
        try:
            app.state.scorer.redis.ping()
            return {"status": "ok", "model_version": app.state.scorer.version}
        except Exception as exc:
            raise HTTPException(503, "Redis unavailable") from exc

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
