import os
import time
import json
import pickle
import logging
import importlib
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    redis_lib = importlib.import_module("redis")
    REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None
    REDIS_AVAILABLE = False

try:
    faiss = importlib.import_module("faiss")
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = int(os.getenv("REDIS_TTL", 3600))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "data")
TOP_K = int(os.getenv("TOP_K", 10))
INTERACTION_THRESHOLD = int(os.getenv("INTERACTION_THRESHOLD", 3))

STATE = {
    "als_model": None,
    "als_user_to_idx": None,
    "als_item_to_idx": None,
    "als_user_ids": None,
    "als_item_ids": None,
    "als_user_item": None,
    "item_factors_norm": None,
    "user_factors_norm": None,
    "faiss_index": None,
    "faiss_item_ids": None,
    "faiss_user_to_idx": None,
    "lfm_model": None,
    "lfm_item_features": None,
    "lfm_user_map": None,
    "lfm_item_ids_list": None,
    "n_lfm_items": 0,
    "purchases": {},
    "als_users": set(),
    "lfm_users": set(),
    "redis_client": None,
    "memory_cache": {},
    "cache_lock": Lock(),
    "startup_time": None,
}


def artifact_path(filename: str) -> str:
    return os.path.join(ARTIFACTS_DIR, filename)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load serving artifacts once at application startup."""
    logger.info("Loading recommendation artifacts...")
    started = time.time()

    try:
        with open(artifact_path("als_artifacts.pkl"), "rb") as file:
            als = pickle.load(file)

        STATE["als_model"] = als["model"]
        STATE["als_user_to_idx"] = als["user_to_idx"]
        STATE["als_item_to_idx"] = als["item_to_idx"]
        STATE["als_user_ids"] = als["user_ids"]
        STATE["als_item_ids"] = als["item_ids"]
        STATE["als_user_item"] = als.get(
            "train_user_item_matrix", als["user_item_matrix"]
        )
        logger.info(
            "ALS loaded: %s users, %s items",
            f"{len(als['user_ids']):,}",
            f"{len(als['item_ids']):,}",
        )

        with open(artifact_path("faiss_artifacts.pkl"), "rb") as file:
            faiss_art = pickle.load(file)

        STATE["item_factors_norm"] = faiss_art["item_factors_norm"]
        STATE["user_factors_norm"] = faiss_art["user_factors_norm"]
        STATE["faiss_item_ids"] = faiss_art["item_ids"]
        STATE["faiss_user_to_idx"] = faiss_art["user_to_idx"]

        if FAISS_AVAILABLE:
            try:
                STATE["faiss_index"] = faiss.read_index(
                    artifact_path("faiss_index.bin")
                )
                logger.info(
                    "FAISS index loaded: %s items",
                    f"{STATE['faiss_index'].ntotal:,}",
                )
            except Exception as error:
                logger.warning("FAISS index unavailable (%s); using NumPy fallback", error)

        logger.info("Loading LightFM serving artifact...")
        lightfm_started = time.time()

        with open(artifact_path("lightfm_serving.pkl"), "rb") as file:
            lfm = pickle.load(file)

        STATE["lfm_model"] = lfm["model_hybrid"]
        STATE["lfm_item_features"] = lfm["item_features_matrix"]
        STATE["lfm_user_map"] = lfm["lfm_user_map"]
        STATE["lfm_item_ids_list"] = lfm["lfm_item_ids_list"]
        STATE["n_lfm_items"] = int(lfm["n_lfm_items"])

        logger.info(
            "LightFM serving artifact unpickled in %.2fs; %s items loaded",
            time.time() - lightfm_started,
            f"{STATE['n_lfm_items']:,}",
        )

        with open(artifact_path("routing_split.pkl"), "rb") as file:
            routing = pickle.load(file)

        STATE["als_users"] = set(routing["als_users"])
        STATE["lfm_users"] = set(routing["lightfm_users"])

        import pandas as pd
        events = pd.read_csv(artifact_path("events.csv"))
        STATE["purchases"] = (
            events.loc[events["event"] == "transaction"]
            .groupby("visitorid")["itemid"]
            .apply(set)
            .to_dict()
        )
        logger.info("Purchase history loaded: %s users", f"{len(STATE['purchases']):,}")

    except Exception as error:
        logger.exception("Artifact loading failed: %s", error)
        raise

    if REDIS_AVAILABLE:
        try:
            STATE["redis_client"] = redis_lib.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=1,
            )
            STATE["redis_client"].ping()
            logger.info("Redis connected: %s:%s", REDIS_HOST, REDIS_PORT)
        except Exception as error:
            logger.warning("Redis unavailable (%s); using in-memory TTL cache", error)
            STATE["redis_client"] = None

    STATE["startup_time"] = time.time() - started
    logger.info("Startup complete in %.2fs", STATE["startup_time"])

    yield

    logger.info("Shutting down recommendation API")


app = FastAPI(
    title="Hybrid Recommender API",
    description="ALS + FAISS for eligible returning users; LightFM fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = 10
    use_cache: Optional[bool] = True


class RecommendResponse(BaseModel):
    user_id: int
    engine: str
    recommendations: List[int]
    scores: List[float]
    cached: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    als_loaded: bool
    lfm_loaded: bool
    redis_connected: bool
    startup_time_s: float


def _cache_key(user_id: int, top_k: int) -> str:
    return f"rec:{user_id}:{top_k}"


def _get_from_cache(key: str):
    if STATE["redis_client"] is not None:
        try:
            cached_value = STATE["redis_client"].get(key)
            if cached_value:
                return json.loads(cached_value)
        except Exception:
            pass

    now = time.time()
    with STATE["cache_lock"]:
        cached = STATE["memory_cache"].get(key)
        if cached is None:
            return None
        expiry_time, value = cached
        if now >= expiry_time:
            del STATE["memory_cache"][key]
            return None
        return value.copy()


def _set_cache(key: str, value: dict):
    if STATE["redis_client"] is not None:
        try:
            STATE["redis_client"].setex(key, REDIS_TTL, json.dumps(value))
        except Exception:
            pass

    with STATE["cache_lock"]:
        STATE["memory_cache"][key] = (time.time() + REDIS_TTL, value.copy())


def _get_als_recs(user_id: int, top_k: int = 30):
    user_to_idx = STATE["faiss_user_to_idx"] or {}
    if user_id not in user_to_idx:
        return [], []

    user_index = user_to_idx[user_id]
    if user_index >= len(STATE["user_factors_norm"]):
        return [], []

    user_vector = STATE["user_factors_norm"][user_index:user_index + 1].astype(np.float32)

    if STATE["faiss_index"] is not None:
        scores, item_indices = STATE["faiss_index"].search(user_vector, top_k)
        valid_pairs = [
            (int(STATE["faiss_item_ids"][item_index]), float(score))
            for item_index, score in zip(item_indices[0], scores[0])
            if 0 <= item_index < len(STATE["faiss_item_ids"])
        ]
        if not valid_pairs:
            return [], []
        items, result_scores = zip(*valid_pairs)
        return list(items), list(result_scores)

    raw_scores = (STATE["item_factors_norm"] @ user_vector.T).flatten()
    item_indices = np.argsort(raw_scores)[::-1][:top_k]
    return (
        [int(STATE["faiss_item_ids"][index]) for index in item_indices],
        [float(raw_scores[index]) for index in item_indices],
    )


def _get_lfm_recs(user_id: int, top_k: int = 30):
    user_map = STATE["lfm_user_map"] or {}
    if user_id not in user_map:
        return [], []

    user_index = user_map[user_id]
    n_items = STATE["n_lfm_items"]

    scores = STATE["lfm_model"].predict(
        user_ids=np.full(n_items, user_index),
        item_ids=np.arange(n_items),
        item_features=STATE["lfm_item_features"],
        num_threads=2,
    )

    top_indices = np.argsort(scores)[::-1][:top_k]
    return (
        [int(STATE["lfm_item_ids_list"][index]) for index in top_indices],
        [float(scores[index]) for index in top_indices],
    )


def _exclude_purchased(items, scores, user_id):
    bought_items = STATE["purchases"].get(user_id, set())
    if not bought_items:
        return list(items), list(scores)

    filtered = [
        (item_id, score)
        for item_id, score in zip(items, scores)
        if item_id not in bought_items
    ]
    if not filtered:
        return [], []

    filtered_items, filtered_scores = zip(*filtered)
    return list(filtered_items), list(filtered_scores)


def _normalize(scores):
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return []
    low, high = values.min(), values.max()
    if high == low:
        return [0.5] * len(values)
    return ((values - low) / (high - low)).tolist()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        als_loaded=STATE["als_model"] is not None,
        lfm_loaded=STATE["lfm_model"] is not None,
        redis_connected=STATE["redis_client"] is not None,
        startup_time_s=STATE["startup_time"] or 0.0,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    started = time.perf_counter()
    user_id = request.user_id
    top_k = request.top_k or TOP_K
    cache_key = _cache_key(user_id, top_k)

    if request.use_cache:
        cached = _get_from_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            cached["latency_ms"] = (time.perf_counter() - started) * 1000
            return RecommendResponse(**cached)

    interaction_count = 0
    als_index_map = STATE["faiss_user_to_idx"] or {}
    if user_id in als_index_map and STATE["als_user_item"] is not None:
        user_index = als_index_map[user_id]
        if user_index < STATE["als_user_item"].shape[0]:
            interaction_count = STATE["als_user_item"][user_index].nnz

    use_als = (
        user_id in STATE["als_users"]
        and user_id in als_index_map
        and interaction_count >= INTERACTION_THRESHOLD
    )

    if use_als:
        engine = "ALS + FAISS"
        raw_items, raw_scores = _get_als_recs(user_id, top_k=top_k * 3)
    else:
        engine = "LightFM Hybrid"
        raw_items, raw_scores = _get_lfm_recs(user_id, top_k=top_k * 3)

    if not raw_items:
        raise HTTPException(status_code=404, detail=f"No recommendations for user {user_id}")

    raw_items, raw_scores = _exclude_purchased(raw_items, raw_scores, user_id)
    if not raw_items:
        raise HTTPException(
            status_code=404,
            detail=f"No unseen recommendations remain for user {user_id}",
        )

    final_items = raw_items[:top_k]
    final_scores = _normalize(raw_scores)[:top_k]

    response_data = {
        "user_id": user_id,
        "engine": engine,
        "recommendations": final_items,
        "scores": final_scores,
        "cached": False,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }

    if request.use_cache:
        _set_cache(cache_key, response_data)

    return RecommendResponse(**response_data)


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend_get(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    use_cache: bool = Query(default=True),
):
    return await recommend(
        RecommendRequest(user_id=user_id, top_k=top_k, use_cache=use_cache)
    )


@app.delete("/cache/{user_id}")
async def clear_user_cache(user_id: int):
    keys_deleted = 0
    for top_k in [5, 10, 20, 50]:
        key = _cache_key(user_id, top_k)
        if STATE["redis_client"] is not None:
            try:
                keys_deleted += STATE["redis_client"].delete(key)
            except Exception:
                pass
        with STATE["cache_lock"]:
            if key in STATE["memory_cache"]:
                del STATE["memory_cache"][key]
                keys_deleted += 1
    return {"user_id": user_id, "keys_deleted": keys_deleted}


@app.get("/stats")
async def stats():
    """Return system statistics."""
    redis_stats = {}

    if STATE["redis_client"] is not None:
        try:
            info = STATE["redis_client"].info("stats")

            redis_stats = {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }

        except Exception:
            pass

    return {
        "als_users": len(STATE["als_users"]),
        "lfm_users": len(STATE["lfm_users"]),
        "faiss_items": (
            len(STATE["faiss_item_ids"])
            if STATE["faiss_item_ids"] is not None
            else 0
        ),
        "lfm_items": STATE["n_lfm_items"],
        "redis_available": STATE["redis_client"] is not None,
        "redis_stats": redis_stats,
        "startup_time_s": STATE["startup_time"],
    }