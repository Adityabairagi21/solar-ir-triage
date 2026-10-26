"""
Step 3: train SolarTriageNet with several random seeds.

    python scripts/train_cnn.py            # all seeds from configs/default.yaml
    python scripts/train_cnn.py --seeds 42 # quick single run

For every seed: train on the TRAIN split, early-stop on VALIDATION macro-F1, and save
validation + test logits (test logits are only *stored* here; every decision below uses
validation data). The deployed model is the seed with the best validation macro-F1; its
temperature and triage thresholds are fitted on validation data and saved with it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

from solartriage import CLASSES, __version__
from solartriage.config import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config
from solartriage.data import binary_labels, load_processed
from solartriage.metrics import anomaly_score, fit_temperature, threshold_for_recall
from solartriage.training import ThermalDataset, predict_logits, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for this process (0 = config)")
    ap.add_argument("--no-select", action="store_true",
                    help="only train the given seeds (lets several seeds run in parallel processes)")
    ap.add_argument("--select-only", action="store_true",
                    help="skip training; choose the deployed model from saved seed checkpoints")
    args = ap.parse_args()

    cfg = load_config()
    if args.threads:
        cfg["train"]["num_threads"] = args.threads
    ensure_dirs()
    logit_dir = RESULTS_DIR / "logits"
    logit_dir.mkdir(exist_ok=True)

    df, images = load_processed(cfg["data"]["processed_dir"])
    y = df["label"].map(CLASSES.index).to_numpy()
    parts = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}

    runs = []
    for seed in ([] if args.select_only else (args.seeds or cfg["train"]["seeds"])):
        t0 = time.perf_counter()
        model, info = train_model(images[parts["train"]], y[parts["train"]],
                                  images[parts["val"]], y[parts["val"]], cfg, seed)
        info["train_minutes"] = (time.perf_counter() - t0) / 60
        for split in ("val", "test"):
            ds = ThermalDataset(images[parts[split]], y[parts[split]], info["mean"], info["std"])
            np.save(logit_dir / f"cnn_seed{seed}_{split}.npy", predict_logits(model, DataLoader(ds, batch_size=512)))
        torch.save({"state_dict": model.state_dict(), "info": info}, MODELS_DIR / f"cnn_seed{seed}.pt")
        (RESULTS_DIR / f"history_seed{seed}.json").write_text(json.dumps(info["history"], indent=1))
        print(f"seed {seed}: best epoch {info['best_epoch']}, val macro-F1 {info['best_val_macro_f1']:.4f}, "
              f"{info['train_minutes']:.1f} min", flush=True)
        runs.append(info)
    if args.no_select:
        return

    # ---- choose the deployed model on VALIDATION performance only ----
    runs = [torch.load(MODELS_DIR / f"cnn_seed{s}.pt", weights_only=False)["info"] for s in cfg["train"]["seeds"]]
    best = max(runs, key=lambda r: r["best_val_macro_f1"])
    seed = best["seed"]
    val_logits = np.load(logit_dir / f"cnn_seed{seed}_val.npy")
    y_val = y[parts["val"]]
    temperature = fit_temperature(val_logits, y_val)
    val_scores = anomaly_score(softmax(val_logits / temperature, axis=1))
    yb_val = binary_labels(df.loc[parts["val"], "label"])
    recalls = [cfg["triage"]["target_recall"], *cfg["triage"]["extra_recalls"]]
    thresholds = {f"{r:.2f}": threshold_for_recall(val_scores, yb_val, r) for r in recalls}

    ckpt = torch.load(MODELS_DIR / f"cnn_seed{seed}.pt", weights_only=False)
    package = {
        "version": __version__,
        "state_dict": ckpt["state_dict"],
        "classes": CLASSES,
        "model_cfg": cfg["model"],
        "mean": best["mean"],
        "std": best["std"],
        "temperature": temperature,
        "thresholds": thresholds,
        "default_recall": f"{cfg['triage']['target_recall']:.2f}",
        "real_world_prevalence": cfg["real_world_prevalence"],
        "seed": seed,
        "best_epoch": best["best_epoch"],
        "val_macro_f1": best["best_val_macro_f1"],
    }
    torch.save(package, MODELS_DIR / "solartriage_cnn.pt")
    summary = {"selected_seed": seed, "temperature": temperature, "thresholds": thresholds,
               "n_parameters": sum(v.numel() for k, v in ckpt["state_dict"].items()
                                   if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))),
               "runs": [{k: v for k, v in r.items() if k != "history"} for r in runs]}
    (RESULTS_DIR / "cnn_training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
