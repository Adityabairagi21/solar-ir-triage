"""
Step 5: evaluate every model on the untouched TEST split and write all tables and figures.

    python scripts/evaluate.py

Rules: every threshold and the calibration temperature come from the VALIDATION split; the
test split is used only here, once, for reporting.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from solartriage import CLASSES, NORMAL_CLASS
from solartriage import plots
from solartriage.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, load_config
from solartriage.data import binary_labels, load_processed
from solartriage.metrics import (anomaly_score, ap_at_prevalence, binary_summary, bootstrap_ci,
                                 deployment_view, expected_calibration_error, multiclass_summary,
                                 precision_recall_at_prevalence, rates_at, threshold_for_recall)
from solartriage.model import SolarTriageNet, count_parameters

CNN, RF, LR, HOT = "SolarTriageNet (CNN)", "Random Forest", "Logistic Regression", "Hot-spot rule"


def main():
    cfg = load_config()
    prev = cfg["real_world_prevalence"]
    op = cfg["triage"]["target_recall"]
    recalls = sorted([op, *cfg["triage"]["extra_recalls"]])
    L = RESULTS_DIR / "logits"
    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df, images = load_processed(cfg["data"]["processed_dir"])
    y = df["label"].map(CLASSES.index).to_numpy()
    split = df["split"].to_numpy()
    yv, yt = y[split == "val"], y[split == "test"]
    ybv, ybt = binary_labels(df.loc[split == "val", "label"]), binary_labels(df.loc[split == "test", "label"])
    test_imgs = images[split == "test"]
    test_ids = df.loc[split == "test", "image_id"].to_numpy()

    pkg = torch.load(MODELS_DIR / "solartriage_cnn.pt", weights_only=False)
    T = pkg["temperature"]
    summary_train = json.loads((RESULTS_DIR / "cnn_training_summary.json").read_text())

    # ---------------- probabilities / scores ----------------
    probs = {
        CNN: (softmax(np.load(L / f"cnn_seed{pkg['seed']}_val.npy") / T, 1),
              softmax(np.load(L / f"cnn_seed{pkg['seed']}_test.npy") / T, 1)),
        RF: (np.load(L / "rf_val.npy"), np.load(L / "rf_test.npy")),
        LR: (np.load(L / "logreg_val.npy"), np.load(L / "logreg_test.npy")),
    }
    scores = {name: (anomaly_score(pv), anomaly_score(pt)) for name, (pv, pt) in probs.items()}
    scores[HOT] = (np.load(L / "hotspot_val.npy"), np.load(L / "hotspot_test.npy"))

    results = {"prevalence": prev, "operating_recall": op, "temperature": T, "selected_seed": pkg["seed"]}

    # ---------------- 12-class fault-type results ----------------
    mc = {}
    for name, (_, pt) in probs.items():
        mc[name] = multiclass_summary(yt, pt.argmax(1))
    seeds = [r["seed"] for r in summary_train["runs"]]
    seed_rows = []
    for s in seeds:
        pt = softmax(np.load(L / f"cnn_seed{s}_test.npy"), 1)
        pv = softmax(np.load(L / f"cnn_seed{s}_val.npy"), 1)
        sv, st = anomaly_score(pv), anomaly_score(pt)
        tpr, fpr = rates_at(st, ybt, threshold_for_recall(sv, ybv, op))
        seed_rows.append({"seed": s, **multiclass_summary(yt, pt.argmax(1)),
                          "roc_auc": roc_auc_score(ybt, st),
                          **{k: v for k, v in deployment_view(tpr, fpr, prev).items()
                             if k in ("recall", "review_workload", "precision_at_prevalence")}})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(tables / "cnn_seeds_test.csv", index=False)
    results["cnn_seed_mean_std"] = {c: [float(seed_df[c].mean()), float(seed_df[c].std(ddof=1))]
                                    for c in seed_df.columns if c != "seed"}
    pd.DataFrame(mc).T.to_csv(tables / "multiclass_test.csv")
    results["multiclass_test"] = mc

    per_class = {}
    for name in (CNN, RF):
        rep = classification_report(yt, probs[name][1].argmax(1), labels=range(len(CLASSES)),
                                    target_names=CLASSES, output_dict=True, zero_division=0)
        per_class[name] = {c: rep[c] for c in CLASSES}
    pc = pd.DataFrame({(m, k): {c: per_class[m][c][k] for c in CLASSES}
                       for m in per_class for k in ("precision", "recall", "f1-score", "support")})
    pc.to_csv(tables / "per_class_test.csv")
    results["per_class_test"] = per_class

    # ---------------- binary triage results ----------------
    binres, oprows = {}, []
    for name, (sv, st) in scores.items():
        binres[name] = binary_summary(st, ybt, prev)
        for r in recalls:
            thr = threshold_for_recall(sv, ybv, r)
            dv = deployment_view(*rates_at(st, ybt, thr), prev)
            oprows.append({"model": name, "target_recall_on_val": r, "threshold": thr, **dv})
    op_df = pd.DataFrame(oprows)
    op_df.to_csv(tables / "operating_points_test.csv", index=False)
    pd.DataFrame(binres).T.to_csv(tables / "binary_test.csv")
    results["binary_test"] = binres
    results["operating_points_test"] = oprows

    # ---------------- bootstrap confidence intervals (paired CNN vs RF) ----------------
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["seed"]
    thr_cnn = threshold_for_recall(scores[CNN][0], ybv, op)
    thr_rf = threshold_for_recall(scores[RF][0], ybv, op)
    pc_cnn, pc_rf = probs[CNN][1].argmax(1), probs[RF][1].argmax(1)
    st_cnn, st_rf = scores[CNN][1], scores[RF][1]

    def workload(st, thr, idx):
        return deployment_view(*rates_at(st[idx], ybt[idx], thr), prev)["review_workload"]

    ci = {
        "cnn_macro_f1": bootstrap_ci(lambda i: f1_score(yt[i], pc_cnn[i], average="macro"), len(yt), nb, seed, yt),
        "rf_macro_f1": bootstrap_ci(lambda i: f1_score(yt[i], pc_rf[i], average="macro"), len(yt), nb, seed, yt),
        "diff_macro_f1_cnn_minus_rf": bootstrap_ci(lambda i: f1_score(yt[i], pc_cnn[i], average="macro")
                                                   - f1_score(yt[i], pc_rf[i], average="macro"), len(yt), nb, seed, yt),
        "cnn_roc_auc": bootstrap_ci(lambda i: roc_auc_score(ybt[i], st_cnn[i]), len(yt), nb, seed, yt),
        "cnn_recall_at_op": bootstrap_ci(lambda i: rates_at(st_cnn[i], ybt[i], thr_cnn)[0], len(yt), nb, seed, yt),
        "cnn_workload_at_op": bootstrap_ci(lambda i: workload(st_cnn, thr_cnn, i), len(yt), nb, seed, yt),
        "rf_workload_at_op": bootstrap_ci(lambda i: workload(st_rf, thr_rf, i), len(yt), nb, seed, yt),
        "diff_workload_cnn_minus_rf": bootstrap_ci(lambda i: workload(st_cnn, thr_cnn, i) - workload(st_rf, thr_rf, i),
                                                   len(yt), nb, seed, yt),
    }
    results["bootstrap_95ci"] = ci

    # ---------------- per-class miss analysis at the operating point ----------------
    flagged = st_cnn >= thr_cnn
    flagged_rf = st_rf >= thr_rf
    miss = {c: {"n": int((yt == k).sum()), "cnn_flagged": float(flagged[yt == k].mean()),
                "rf_flagged": float(flagged_rf[yt == k].mean())} for k, c in enumerate(CLASSES)}
    results["flag_rate_by_class_at_op"] = miss
    pd.DataFrame(miss).T.to_csv(tables / "flag_rate_by_class_at_op.csv")

    # ---------------- calibration ----------------
    raw_test = softmax(np.load(L / f"cnn_seed{pkg['seed']}_test.npy"), 1)
    results["calibration"] = {"ece_before": expected_calibration_error(raw_test, yt),
                              "ece_after": expected_calibration_error(probs[CNN][1], yt)}

    # ---------------- confusion pairs ----------------
    cm = confusion_matrix(yt, pc_cnn, labels=range(len(CLASSES)))
    pairs = [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i in range(len(CLASSES)) for j in range(len(CLASSES))
             if i != j and cm[i, j] > 0]
    results["top_confusions"] = sorted(pairs, key=lambda p: -p[2])[:10]
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(tables / "confusion_cnn_test.csv")

    # ---------------- model size and CPU latency ----------------
    model = SolarTriageNet(len(CLASSES), tuple(pkg["model_cfg"]["channels"]), pkg["model_cfg"]["dropout"])
    model.load_state_dict(pkg["state_dict"]); model.eval()
    x1 = torch.randn(1, 1, 40, 24); xb = torch.randn(512, 1, 40, 24)
    with torch.no_grad():
        for _ in range(10): model(x1)
        t0 = time.perf_counter(); [model(x1) for _ in range(200)]; single = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter(); [model(xb) for _ in range(5)]; batch = (time.perf_counter() - t0) / 5 / 512
    results["efficiency"] = {"parameters": count_parameters(model),
                             "model_file_mb": (MODELS_DIR / "solartriage_cnn.pt").stat().st_size / 1e6,
                             "cpu_ms_per_image_batch1": single * 1000, "cpu_ms_per_image_batch512": batch * 1000,
                             "cpu_threads": torch.get_num_threads()}

    # ---------------- open-set (if available) ----------------
    os_files = sorted((RESULTS_DIR / "open_set").glob("*.json"))
    if os_files:
        results["open_set"] = {f.stem: json.loads(f.read_text()) for f in os_files}
        pd.DataFrame(results["open_set"]).T.to_csv(tables / "open_set.csv")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    # ---------------- figures ----------------
    labels_all = df["label"].to_numpy()
    plots.class_examples(images, labels_all, CLASSES, FIGURES_DIR / "01_class_examples.png")
    plots.class_distribution(df["label"].value_counts().reindex(CLASSES).to_dict(),
                             FIGURES_DIR / "02_class_distribution.png", prev)
    hist = json.loads((RESULTS_DIR / f"history_seed{pkg['seed']}.json").read_text())
    plots.learning_curves(hist, FIGURES_DIR / "03_learning_curves.png", pkg["seed"])
    plots.confusion(cm / cm.sum(1, keepdims=True), CLASSES, FIGURES_DIR / "04_confusion_cnn.png",
                    "SolarTriageNet confusion matrix (test, row-normalised)")
    curves = {name: precision_recall_at_prevalence(st, ybt, prev) for name, (_, st) in scores.items()}
    plots.deployment_curves(curves, FIGURES_DIR / "05_precision_workload_vs_recall.png", prev, op)
    anomaly_classes = [c for c in CLASSES if c != NORMAL_CLASS]
    plots.grouped_bars(anomaly_classes, {CNN: [per_class[CNN][c]["f1-score"] for c in anomaly_classes],
                                         RF: [per_class[RF][c]["f1-score"] for c in anomaly_classes]},
                       FIGURES_DIR / "06_per_class_f1.png", "Per-class F1 on the test set", "F1-score",
                       fmt="{:.2f}")
    plots.reliability(raw_test, probs[CNN][1], yt, FIGURES_DIR / "07_reliability.png")
    if "open_set" in results:
        os_res = results["open_set"]
        cls = [c for c in anomaly_classes if c in os_res]
        plots.grouped_bars(cls, {CNN: [os_res[c]["cnn_recall_unseen"] for c in cls],
                                 RF: [os_res[c]["rf_recall_unseen"] for c in cls],
                                 HOT: [os_res[c]["hotspot_recall"] for c in cls]},
                           FIGURES_DIR / "08_open_set_recall.png",
                           "Share of a NEVER-SEEN fault type that still gets flagged (threshold: 95 % recall on seen faults)",
                           "Flagged share of unseen class")
    # error gallery: missed anomalies (false negatives) and strongest false alarms at the operating point
    fn = np.flatnonzero((ybt == 1) & ~flagged)
    fn = fn[np.argsort(st_cnn[fn])][:16]
    fp = np.flatnonzero((ybt == 0) & flagged)
    fp = fp[np.argsort(-st_cnn[fp])][:8]
    sel = np.concatenate([fn, fp])
    titles = [f"{'MISSED' if k in fn else 'FALSE ALARM'}\ntrue: {CLASSES[yt[k]]}\nscore {st_cnn[k]:.2f}\nid {test_ids[k]}"
              for k in sel]
    plots.gallery(test_imgs[sel], titles, FIGURES_DIR / "09_error_gallery.png",
                  f"Errors at the {op:.0%}-recall operating point (test): lowest-scored missed faults and top false alarms")

    # console summary
    print(pd.DataFrame(mc).T.round(4).to_string())
    print(pd.DataFrame(binres).T.round(4).to_string())
    print(op_df[op_df.target_recall_on_val == op].round(4).to_string(index=False))
    print(json.dumps({k: results[k] for k in ("bootstrap_95ci", "calibration", "efficiency", "top_confusions")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
