# =============================================================================
# Utility: Create a lightweight LightFM serving artifact
# Keeps only objects required by FastAPI inference.
# =============================================================================

import os
import gc
import pickle
import sys
from pathlib import Path

import numpy as np

SOURCE_PATH = "data/lightfm_artifacts.pkl"
TARGET_PATH = "data/lightfm_serving.pkl"
TEMP_PATH = "data/lightfm_serving_temp.pkl"

print("=" * 60)
print("Creating slim LightFM serving artifact")
print("=" * 60)

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(f"Missing source artifact: {SOURCE_PATH}")

print(f"Python: {sys.version.split()[0]}")
print(f"Loading source: {SOURCE_PATH}")
print(f"Source size: {os.path.getsize(SOURCE_PATH) / 1024**3:.2f} GB")

with open(SOURCE_PATH, "rb") as f:
    lfm_art = pickle.load(f)

print("Original artifact loaded.")

# Extract only serving-time components.
model_hybrid = lfm_art["model_hybrid"]
item_features_matrix = lfm_art["item_features_matrix"]

# Obtain mappings once, then save simple dict/list objects instead of the
# full LightFM Dataset object and all Lab 2.3 training matrices.
dataset = lfm_art["dataset"]
user_map, _, item_map, _ = dataset.mapping()

lfm_user_map = dict(user_map)
lfm_item_ids_list = list(item_map.keys())

serving_artifact = {
    "model_hybrid": model_hybrid,
    "item_features_matrix": item_features_matrix,
    "lfm_user_map": lfm_user_map,
    "lfm_item_ids_list": lfm_item_ids_list,
    "n_lfm_items": len(lfm_item_ids_list),

    # Preserve only lightweight metadata useful for Lab 4 tracking.
    "hybrid_test_precision_at_10": float(
        lfm_art.get(
            "hybrid_test_precision_at_10",
            lfm_art.get("hybrid_test_precision", 0.0)
        )
    ),
    "hybrid_train_precision_at_10": float(
        lfm_art.get(
            "hybrid_train_precision_at_10",
            lfm_art.get("hybrid_train_precision", 0.0)
        )
    ),
    "config": lfm_art.get("config", {})
}

print("Saving slim serving artifact...")

with open(TEMP_PATH, "wb") as f:
    pickle.dump(
        serving_artifact,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(TEMP_PATH, TARGET_PATH)

del serving_artifact
del model_hybrid
del item_features_matrix
del lfm_user_map
del lfm_item_ids_list
del dataset
del lfm_art
gc.collect()

new_size_gb = os.path.getsize(TARGET_PATH) / 1024**3

print(f"\nSaved: {TARGET_PATH}")
print(f"Serving artifact size: {new_size_gb:.2f} GB")
print("Keep this file for FastAPI inference.")
