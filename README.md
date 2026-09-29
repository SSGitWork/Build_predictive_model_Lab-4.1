# Lab 4.1 — Hybrid Recommender API

## Purpose
This lab packages the recommender system into a FastAPI service and prepares a slim LightFM serving artifact.

It includes:
- a production-style recommendation API
- ALS + FAISS routing for returning users
- LightFM fallback for cold-start users
- caching with Redis or in-memory fallback
- purchase exclusion and normalized scores

## Setup
- Python 3.11+
- Required packages from `requirements-app.txt` for the API runtime
- Required packages from `requirements.txt` for the full lab environment
- Input artifacts expected under `data/`:
  - `als_artifacts.pkl`
  - `faiss_artifacts.pkl`
  - `faiss_index.bin`
  - `lightfm_artifacts.pkl`
  - `routing_split.pkl`
  - `events.csv`

## How to Run
First, create the slim LightFM serving artifact:

```bash
python create_lightfm_serving_artifact.py
```

Then start the API:

```bash
uvicorn app:app --reload
```

The service exposes health and recommendation endpoints for online inference.

## Outputs
The scripts create or use:
- `data/lightfm_serving.pkl`
- cached recommendation responses in Redis or memory
- API logs in the console
- any runtime artifacts written to `output/`

## Key Design Choices
- Separated offline training artifacts from online serving artifacts.
- Loaded heavy models once at startup using FastAPI lifespan hooks.
- Routed returning users to ALS + FAISS when they meet the interaction threshold.
- Routed other users to LightFM hybrid recommendations.
- Excluded already purchased items before returning results.
- Added Redis caching with an in-memory fallback.
- Normalized scores before returning them to clients.

## Key Findings
Typical outcomes from this lab include:
- The API can serve personalized recommendations with low latency.
- Routing improves relevance by matching users to the right engine.
- Caching reduces repeated request cost.
- A slim serving artifact makes deployment more practical.

## Extra Info
- The API reads configuration from environment variables such as `REDIS_HOST`, `REDIS_PORT`, `REDIS_TTL`, `ARTIFACTS_DIR`, `TOP_K`, and `INTERACTION_THRESHOLD`.
- If Redis is unavailable, the service falls back to in-memory caching.
- Run the prerequisite recommender labs first so all artifacts are available.
- `requirements-app.txt` is the preferred dependency set for the API runtime.
