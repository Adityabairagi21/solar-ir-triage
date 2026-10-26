"""Figures for the report. Colours are a colour-blind-checked categorical palette; each model
keeps the same colour in every chart."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MODEL_COLORS = {"SolarTriageNet (CNN)": "#2a78d6", "Random Forest": "#eb6834",
                "Logistic Regression": "#1baf7a", "Hot-spot rule": "#eda100"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 9.5, "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def class_examples(images, labels, classes, path, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, len(classes), figsize=(len(classes) * 0.95, per_class * 1.45))
    for j, c in enumerate(classes):
        idx = rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
        for i, k in enumerate(idx):
            ax = axes[i, j]
            ax.imshow(images[k], cmap="inferno", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, j].set_title(c, fontsize=7.5, rotation=30, ha="left", loc="left")
    fig.suptitle("Example thermal crops per class (colour map: inferno; brighter = warmer)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, path)


def class_distribution(counts: dict, path, real_prev):
    names = list(counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names[::-1], [counts[n] for n in names[::-1]], color="#2a78d6", height=0.65)
    for b in bars:
        ax.annotate(f"{int(b.get_width()):,}", (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set(title="Images per class after cleaning",
           xlabel=f"Images  (dataset is 50 % anomalous; real farms ≈ {real_prev:.1%})")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    save(fig, path)


def learning_curves(history, path, seed):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], color="#2a78d6", lw=2, label="Training")
    axes[0].plot(ep, [h["val_loss"] for h in history], color="#eb6834", lw=2, label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(ep, [h["train_macro_f1"] for h in history], color="#2a78d6", lw=2, label="Training")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], color="#eb6834", lw=2, label="Validation")
    best = int(np.argmax([h["val_macro_f1"] for h in history]))
    axes[1].axvline(ep[best], color=MUTED, ls="--", lw=1)
    axes[1].annotate(f"kept epoch {ep[best]}", (ep[best], history[best]["val_macro_f1"]),
                     xytext=(-90, -30), textcoords="offset points", color=INK2)
    axes[1].set(title="Macro-F1 (12 classes)", xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle(f"SolarTriageNet learning curves (seed {seed})", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", y=1.03)
    save(fig, path)


def confusion(cm_norm, classes, path, title):
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Share of the true class (row-normalised)")
    save(fig, path)


def deployment_curves(curves: dict, path, prevalence, op_recall):
    """curves[name] = (recall, precision_at_prevalence, workload)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, (rec, prec, work) in curves.items():
        c = MODEL_COLORS[name]
        o = np.argsort(rec)
        axes[0].plot(rec[o], prec[o], color=c, lw=2, label=name)
        axes[1].plot(rec[o], 100 * work[o], color=c, lw=2, label=name)
    for ax in axes:
        ax.axvline(op_recall, color=MUTED, ls="--", lw=1)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Anomaly recall (share of faulty modules caught)")
    axes[0].set(title=f"Precision at real-world prevalence ({prevalence:.1%})", ylabel="Precision", ylim=(0, 1.02))
    axes[1].set(title="Expert review workload", ylabel="Modules flagged for review (% of all modules)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")
    axes[0].text(op_recall, 0.03, f" operating point: {op_recall:.0%} recall", color=INK2, fontsize=8.5)
    save(fig, path)


def grouped_bars(labels, series: dict, path, title, ylabel, fmt="{:.0%}", ylim=(0, 1.08)):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for k, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (k - (n - 1) / 2) * w, vals, w, label=name, color=MODEL_COLORS[name],
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK, rotation=90)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set(title=title, ylabel=ylabel, ylim=ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=n)
    save(fig, path)


def reliability(probs_before, probs_after, y, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    edges = np.linspace(0, 1, n_bins + 1)
    for probs, name, color in ((probs_before, "Before temperature scaling", "#eb6834"),
                               (probs_after, "After temperature scaling", "#2a78d6")):
        conf, correct = probs.max(1), probs.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.text(0.62, 0.55, "perfect calibration", color=MUTED, rotation=40)
    ax.set(title="Reliability of predicted confidence (test)", xlabel="Predicted confidence",
           ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    save(fig, path)


def gallery(images, titles, path, title, ncols=8):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.25, nrows * 2.25))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255)
        ax.set_title(t, fontsize=6.5, loc="center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, path)
