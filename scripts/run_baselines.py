"""
Step 2: fit the baselines and store their validation / test probabilities.

    python scripts/run_baselines.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np

from solartriage import CLASSES
from solartriage.baselines import fit_logistic, fit_random_forest, hotspot_score, thermal_features
from solartriage.config import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config
from solartriage.data import load_processed


def main():
    cfg = load_config()
    ensure_dirs()
    out = RESULTS_DIR / "logits"
    out.mkdir(exist_ok=True)
    df, images = load_processed(cfg["data"]["processed_dir"])
    y = df["label"].map(CLASSES.index).to_numpy()
    p = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}
    seed = cfg["seed"]

    for split in ("val", "test"):
        np.save(out / f"hotspot_{split}.npy", hotspot_score(images[p[split]]))

    print("Logistic regression ...", flush=True)
    lr, lr_info = fit_logistic(images[p["train"]], y[p["train"]], images[p["val"]], y[p["val"]], seed)
    for split in ("val", "test"):
        np.save(out / f"logreg_{split}.npy", lr.predict_proba(images[p[split]].reshape(p[split].sum(), -1)))

    print("Random forest ...", flush=True)
    rf, rf_info = fit_random_forest(images[p["train"]], y[p["train"]], images[p["val"]], y[p["val"]], seed,
                                    n_jobs=int(os.environ.get("BASELINE_JOBS", "4")))
    for split in ("val", "test"):
        np.save(out / f"rf_{split}.npy", rf.predict_proba(thermal_features(images[p[split]])))
    joblib.dump(rf, MODELS_DIR / "baseline_random_forest.joblib", compress=3)

    info = {"logistic_regression": lr_info, "random_forest": rf_info}
    (RESULTS_DIR / "baseline_selection.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
