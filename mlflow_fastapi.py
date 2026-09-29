# =============================================================================
# MODULE 4 | LAB 4.1
# File: mlflow_fastapi.py
# Purpose: MLflow run tracking + FastAPI endpoint + latency benchmark
# Python: 3.13-compatible, memory-conscious serving version
# =============================================================================

import os
import sys
import time
import gc
import pickle
import shutil
import subprocess
from pathlib import Path

import requests
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import mlflow

print("=" * 60)
print("  MODULE 4 | LAB 4.1")
print("  MLflow Registry + FastAPI Endpoint + Latency Benchmark")
print("=" * 60)

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
N_COLD_REQUESTS = 25
N_WARM_REQUESTS = 25


def load_pickle_checked(path):
    """Load required non-empty pickle artifacts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Artifact is empty: {path}")
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except EOFError as error:
        raise ValueError(f"Artifact is incomplete/corrupted: {path}") from error


def wait_for_server(url, attempts=180, pause_seconds=1):
    """Wait for FastAPI to finish startup and return a healthy response."""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return response.json(), attempt
            print(f"    Waiting... attempt {attempt}/{attempts} (HTTP {response.status_code})")
        except requests.RequestException:
            if attempt == 1 or attempt % 5 == 0:
                print(f"    Waiting for artifact loading... attempt {attempt}/{attempts}")
        time.sleep(pause_seconds)
    return None, attempts


def prepare_local_runtime_artifacts():
    """Copy only required serving files from Drive to fast local Colab storage."""
    source_data_dir = Path("data")
    local_data_dir = Path("/content/runtime_bundle/data")
    local_data_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "als_artifacts.pkl",
        "lightfm_serving.pkl",
        "faiss_artifacts.pkl",
        "routing_split.pkl",
        "faiss_index.bin",
        "events.csv",
    ]

    print("\n    Preparing local API runtime bundle...")
    for filename in required_files:
        source_path = source_data_dir / filename
        target_path = local_data_dir / filename

        if not source_path.exists():
            raise FileNotFoundError(f"Required runtime file is missing: {source_path}")

        source_size = source_path.stat().st_size
        if target_path.exists() and target_path.stat().st_size == source_size:
            print(f"      Reusing local file: {filename}")
            continue

        print(f"      Copying: {filename} ({source_size / 1024 / 1024:.1f} MB)")
        shutil.copy2(source_path, target_path)

    return str(local_data_dir)


# ---------------------------------------------------------------------------
# SECTION 1: MLflow Setup
# ---------------------------------------------------------------------------
print("\n[1] Setting up MLflow...")
print(f"    Python executable : {sys.executable}")
print(f"    MLflow version    : {mlflow.__version__}")

mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.set_experiment("hybrid-recommender")
print("    Experiment        : hybrid-recommender")


# ---------------------------------------------------------------------------
# SECTION 2: Load only compact metadata-bearing artifacts
# ---------------------------------------------------------------------------
print("\n[2] Loading model metadata artifacts...")

# Do not load lightfm_serving.pkl here: it can be multi-GB. Uvicorn is the
# sole process that should deserialize it during serving startup.
als_art = load_pickle_checked("data/als_artifacts.pkl")
fai_art = load_pickle_checked("data/faiss_artifacts.pkl")

als_model = als_art["model"]
best_ndcg = float(
    als_art.get(
        "best_ndcg_at_10",
        als_art.get("baseline_ndcg_at_10", als_art.get("best_ndcg", 0.017)),
    )
)

# Values are intentionally stored as deployment metadata, avoiding a second
# in-memory deserialization of lightfm_serving.pkl in this launcher process.
LIGHTFM_LOSS = "warp"
LIGHTFM_COMPONENTS = 64
hybrid_p10 = 0.0

print(f"    ALS model loaded   : factors={als_model.factors}")
print(f"    LightFM serving    : components={LIGHTFM_COMPONENTS} (loaded by API only)")
print(f"    ALS NDCG@10        : {best_ndcg:.4f}")
print(f"    LightFM P@10       : {hybrid_p10:.4f}")


# ---------------------------------------------------------------------------
# SECTION 3: Track ALS in MLflow without duplicating binary artifacts
# ---------------------------------------------------------------------------
print("\n[3] Registering ALS model in MLflow...")

with mlflow.start_run(run_name="als_personalization_engine"):
    mlflow.log_params({
        "model_type": "ALS",
        "factors": int(als_model.factors),
        "iterations": int(als_model.iterations),
        "regularization": float(als_model.regularization),
        "n_users": int(len(als_art["user_ids"])),
        "n_items": int(len(als_art["item_ids"])),
    })
    mlflow.log_metric("ndcg_at_10", best_ndcg)
    mlflow.set_tag("als_artifact_path", os.path.abspath("data/als_artifacts.pkl"))
    mlflow.set_tag("faiss_artifact_path", os.path.abspath("data/faiss_artifacts.pkl"))
    mlflow.log_dict({
        "model_type": "ALS",
        "artifact_path": os.path.abspath("data/als_artifacts.pkl"),
        "faiss_artifact_path": os.path.abspath("data/faiss_artifacts.pkl"),
        "ndcg_at_10": best_ndcg,
    }, "model/als_model_reference.json")
    print(f"    ALS run ID         : {mlflow.active_run().info.run_id}")


# ---------------------------------------------------------------------------
# SECTION 4: Track LightFM serving reference without copying model binaries
# ---------------------------------------------------------------------------
print("\n[4] Registering LightFM model in MLflow...")

with mlflow.start_run(run_name="lightfm_coldstart_engine"):
    mlflow.log_params({
        "model_type": "LightFM Hybrid",
        "loss": LIGHTFM_LOSS,
        "no_components": LIGHTFM_COMPONENTS,
        "n_items": int(fai_art["n_items"]),
    })
    mlflow.log_metric("precision_at_10", hybrid_p10)
    mlflow.set_tag(
        "lightfm_serving_artifact_path",
        os.path.abspath("data/lightfm_serving.pkl"),
    )
    mlflow.log_dict({
        "model_type": "LightFM Hybrid",
        "artifact_path": os.path.abspath("data/lightfm_serving.pkl"),
        "loss": LIGHTFM_LOSS,
        "no_components": LIGHTFM_COMPONENTS,
        "precision_at_10": hybrid_p10,
    }, "model/lightfm_model_reference.json")
    print(f"    LightFM run ID     : {mlflow.active_run().info.run_id}")


# Release ALS objects before starting Uvicorn to minimize parent-process RAM.
del als_model, als_art, fai_art
gc.collect()


# ---------------------------------------------------------------------------
# SECTION 5: Show MLflow Run Metadata
# ---------------------------------------------------------------------------
print("\n[5] MLflow experiment runs:")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("hybrid-recommender")

if experiment is not None:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    for run in runs:
        print(f"\n    Run name : {run.data.tags.get('mlflow.runName', 'Unnamed')}")
        print(f"    Run ID   : {run.info.run_id}")
        print(f"    Status   : {run.info.status}")
        print(f"    Params   : {run.data.params}")
        print(f"    Metrics  : {run.data.metrics}")


# ---------------------------------------------------------------------------
# SECTION 6: Start FastAPI Server
# ---------------------------------------------------------------------------
print("\n[6] Starting FastAPI server...")
local_artifacts_dir = prepare_local_runtime_artifacts()
print(f"    Runtime artifacts : {local_artifacts_dir}")
print(f"    Starting Uvicorn on {BASE_URL} ...")

server_log_path = "output/fastapi_server.log"
server_log = open(server_log_path, "w", encoding="utf-8")
server_environment = os.environ.copy()
server_environment["ARTIFACTS_DIR"] = local_artifacts_dir

server_process = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--log-level", "info",
    ],
    cwd=os.getcwd(),
    env=server_environment,
    stdout=server_log,
    stderr=subprocess.STDOUT,
)

print("    Waiting for server to be ready...")
health, attempts = wait_for_server(BASE_URL)

if health is None:
    print("    WARNING: Server did not start successfully.")
    if server_process.poll() is not None:
        print(f"    Process exit code: {server_process.returncode}")
    server_log.close()
    if os.path.exists(server_log_path):
        print("\n    FastAPI server log:")
        with open(server_log_path, "r", encoding="utf-8") as file:
            print(file.read()[-5000:])
    raise RuntimeError("FastAPI failed to start. Review output/fastapi_server.log.")

print(f"    Server ready after {attempts} attempts")
print(f"    Redis connected : {health.get('redis_connected', False)}")
print(f"    ALS loaded      : {health.get('als_loaded', False)}")
print(f"    LightFM loaded  : {health.get('lfm_loaded', False)}")


# ---------------------------------------------------------------------------
# SECTION 7: Endpoint Smoke Test
# ---------------------------------------------------------------------------
print("\n[7] Testing /recommend endpoint...")

# Use stable known ALS user IDs from a compact artifact only after server starts.
# Reloading ALS is cheap (~8 MB) and occurs after the child server is alive.
als_smoke_art = load_pickle_checked("data/als_artifacts.pkl")
sample_users = [int(user_id) for user_id in list(als_smoke_art["user_ids"])[:3]]
del als_smoke_art
gc.collect()

for user_id in sample_users:
    try:
        response = requests.get(
            f"{BASE_URL}/recommend/{user_id}",
            params={"top_k": 5, "use_cache": True},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        print(f"\n    User ID         : {user_id}")
        print(f"    Engine          : {payload.get('engine')}")
        print(f"    Cached          : {payload.get('cached')}")
        print(f"    Latency         : {payload.get('latency_ms', 0):.2f} ms")
        print(f"    Recommendations : {payload.get('recommendations')}")
    except requests.RequestException as error:
        print(f"    Request failed for user {user_id}: {error}")


# ---------------------------------------------------------------------------
# SECTION 8: Latency Benchmark
# ---------------------------------------------------------------------------
print(f"\n[8] Latency benchmark ({N_COLD_REQUESTS + N_WARM_REQUESTS} requests)...")
rng = np.random.default_rng(42)
bench_users = rng.choice(sample_users, size=max(N_COLD_REQUESTS, N_WARM_REQUESTS), replace=True)

cold_latencies = []
warm_latencies = []

print("    Running cold requests (cache miss)...")
for user_id in bench_users[:N_COLD_REQUESTS]:
    try:
        requests.delete(f"{BASE_URL}/cache/{user_id}", timeout=15)
        started = time.perf_counter()
        response = requests.get(
            f"{BASE_URL}/recommend/{user_id}",
            params={"top_k": 10, "use_cache": True},
            timeout=90,
        )
        response.raise_for_status()
        cold_latencies.append((time.perf_counter() - started) * 1000)
    except requests.RequestException:
        pass

print("    Running warm requests (cache hit)...")
for user_id in bench_users[:N_WARM_REQUESTS]:
    try:
        started = time.perf_counter()
        response = requests.get(
            f"{BASE_URL}/recommend/{user_id}",
            params={"top_k": 10, "use_cache": True},
            timeout=30,
        )
        response.raise_for_status()
        if response.json().get("cached") is True:
            warm_latencies.append((time.perf_counter() - started) * 1000)
    except requests.RequestException:
        pass

percentiles = [50, 75, 90, 95, 99]
if cold_latencies and warm_latencies:
    cold_percentile_values = np.percentile(cold_latencies, percentiles)
    warm_percentile_values = np.percentile(warm_latencies, percentiles)
    print("\n    Latency Summary")
    print("    " + "-" * 52)
    print("    Percentile | Cold / Cache Miss | Warm / Cache Hit")
    print("    " + "-" * 52)
    for percentile, cold_value, warm_value in zip(percentiles, cold_percentile_values, warm_percentile_values):
        print(f"    p{percentile:<9} | {cold_value:>8.2f} ms      | {warm_value:>8.2f} ms")
else:
    print("    WARNING: Insufficient latency samples collected.")
    cold_percentile_values = np.zeros(len(percentiles))
    warm_percentile_values = np.zeros(len(percentiles))


# ---------------------------------------------------------------------------
# SECTION 9: Latency Visualization
# ---------------------------------------------------------------------------
print("\n[9] Plotting latency results...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Lab 4.1: FastAPI Endpoint Latency Benchmark\nCold (cache miss) vs Warm (cache hit)",
    fontsize=12,
    fontweight="bold",
)

if cold_latencies:
    axes[0].hist(cold_latencies, bins=min(25, len(cold_latencies)), alpha=0.65, label="Cold / Cache Miss", color="steelblue")
if warm_latencies:
    axes[0].hist(warm_latencies, bins=min(25, len(warm_latencies)), alpha=0.65, label="Warm / Cache Hit", color="seagreen")
axes[0].axvline(50, color="red", linestyle="--", linewidth=2, label="p99 SLA target: 50 ms")
axes[0].set_title("Latency Distribution")
axes[0].set_xlabel("Latency (ms)")
axes[0].set_ylabel("Requests")
axes[0].legend()
axes[0].grid(alpha=0.25)

x = np.arange(len(percentiles))
bar_width = 0.38
axes[1].bar(x - bar_width / 2, cold_percentile_values, width=bar_width, label="Cold / Cache Miss", color="steelblue")
axes[1].bar(x + bar_width / 2, warm_percentile_values, width=bar_width, label="Warm / Cache Hit", color="seagreen")
axes[1].axhline(50, color="red", linestyle="--", linewidth=2, label="SLA target: 50 ms")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"p{value}" for value in percentiles])
axes[1].set_title("Percentile Comparison")
axes[1].set_xlabel("Percentile")
axes[1].set_ylabel("Latency (ms)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("output/01_fastapi_latency.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved -> output/01_fastapi_latency.png")


# ---------------------------------------------------------------------------
# Shutdown Server
# ---------------------------------------------------------------------------
if server_process is not None:
    server_process.terminate()
    try:
        server_process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_process.kill()

server_log.close()
print("\n    Server process terminated")
print("    Server log saved -> output/fastapi_server.log")
