"""
Step 4: leave-one-fault-type-out ("open-set") experiment.

Question: if a fault type never appeared in training, does the triage model still flag it?
For each of the 11 anomaly classes c:
  * remove every image of class c from TRAIN and VALIDATION;
  * train the CNN (and the Random Forest baseline) on the remaining classes;
  * pick the threshold that gives 95 % recall on the *seen* anomalies of the validation split;
  * measure the share of class-c images that get flagged. All class-c images can be used,
    because none of them was used for training or threshold selection.
The hot-spot rule needs no training, so its recall per class is reported as a reference.

    python scripts/open_set_experiment.py              # all 11 classes
    python scripts/open_set_experiment.py --classes Soiling Diode-Multi --threads 4   # one parallel worker
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from solartriage import CLASSES, NORMAL_CLASS
from solartriage.baselines import hotspot_score, thermal_features
from solartriage.config import RESULTS_DIR, ensure_dirs, load_config
from solartriage.data import binary_labels, load_processed
from solartriage.metrics import anomaly_score, threshold_for_recall
from solartriage.training import ThermalDataset, predict_logits, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    if args.threads:
        cfg["train"]["num_threads"] = args.threads
    ensure_dirs()
    os_cfg = cfg["open_set"]
    target = cfg["triage"]["target_recall"]
    out_dir = RESULTS_DIR / "open_set"   # one file per class, so several workers can run in parallel
    out_dir.mkdir(exist_ok=True)

    df, images = load_processed(cfg["data"]["processed_dir"])
    y = df["label"].map(CLASSES.index).to_numpy()
    yb = binary_labels(df["label"])
    split = df["split"].to_numpy()

    hot = hotspot_score(images)
    va_seen_all = split == "val"
    hot_thr = threshold_for_recall(hot[va_seen_all], yb[va_seen_all], target)

    for held in args.classes or [c for c in CLASSES if c != NORMAL_CLASS]:
        is_held = (df["label"] == held).to_numpy()
        tr = (split == "train") & ~is_held
        va = (split == "val") & ~is_held
        print(f"=== held-out class: {held} ({is_held.sum()} images) ===", flush=True)

        model, info = train_model(images[tr], y[tr], images[va], y[va], cfg, os_cfg["seed"],
                                  max_epochs=os_cfg["max_epochs"], patience=os_cfg["patience"], verbose=False)

        def cnn_scores(mask):
            ds = ThermalDataset(images[mask], y[mask], info["mean"], info["std"])
            return anomaly_score(softmax(predict_logits(model, DataLoader(ds, batch_size=512)), axis=1))

        thr = threshold_for_recall(cnn_scores(va), yb[va], target)
        cnn_recall = float((cnn_scores(is_held) >= thr).mean())
        test_normal = (split == "test") & (df["label"] == NORMAL_CLASS).to_numpy()
        cnn_fpr = float((cnn_scores(test_normal) >= thr).mean())

        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, class_weight="balanced_subsample",
                                    n_jobs=max(1, args.threads or 4), random_state=os_cfg["seed"])
        rf.fit(thermal_features(images[tr]), y[tr])
        rf_score = lambda m: anomaly_score(_full_proba(rf, thermal_features(images[m])))
        rf_thr = threshold_for_recall(rf_score(va), yb[va], target)
        rf_recall = float((rf_score(is_held) >= rf_thr).mean())

        result = {
            "n_images": int(is_held.sum()),
            "cnn_recall_unseen": cnn_recall,
            "cnn_fpr_test_normal": cnn_fpr,
            "cnn_best_epoch": info["best_epoch"],
            "rf_recall_unseen": rf_recall,
            "hotspot_recall": float((hot[is_held] >= hot_thr).mean()),
        }
        print(held, json.dumps(result), flush=True)
        (out_dir / f"{held}.json").write_text(json.dumps(result, indent=2))


def _full_proba(model, x):
    """predict_proba with a column for every class (the held-out class gets probability 0)."""
    p = np.zeros((len(x), len(CLASSES)))
    p[:, model.classes_] = model.predict_proba(x)
    return p


if __name__ == "__main__":
    main()
