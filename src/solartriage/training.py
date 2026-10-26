"""Dataset, augmentation, training loop with early stopping, and prediction helpers."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from solartriage import CLASSES
from solartriage.model import SolarTriageNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ThermalDataset(Dataset):
    """uint8 images (N,40,24) + integer labels -> float tensors (1,40,24) in [0, 1]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, mean: float, std: float):
        self.x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
        self.y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Augmentation is applied per batch in `augment_batch` (much faster on CPU)
        return self.x[i], self.y[i]

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def augment_batch(x: torch.Tensor, a: dict) -> torch.Tensor:
    """Label-preserving changes that a different flight / camera could plausibly produce.
    x: (B,1,H,W) in [0,1]; every image gets its own random flip / shift / brightness / contrast."""
    b = x.shape[0]
    if a.get("hflip"):
        m = torch.rand(b) < 0.5
        x[m] = x[m].flip(-1)
    if a.get("vflip"):
        m = torch.rand(b) < 0.5
        x[m] = x[m].flip(-2)
    s = a.get("max_shift_px", 0)
    if s:
        h, w = x.shape[-2:]
        padded = F.pad(x, (s, s, s, s), mode="replicate")
        off = torch.randint(0, 2 * s + 1, (b, 2))
        x = torch.stack([padded[i, :, oy:oy + h, ox:ox + w] for i, (oy, ox) in enumerate(off.tolist())])
    c = 1 + (torch.rand(b, 1, 1, 1) * 2 - 1) * a.get("contrast", 0)
    br = (torch.rand(b, 1, 1, 1) * 2 - 1) * a.get("brightness", 0)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean + br).clamp(0, 1)


def class_weights(labels: np.ndarray, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    # a class absent from training (open-set hold-out) gets weight 0, not inf
    w = np.zeros_like(counts)
    present = counts > 0
    w[present] = (counts.sum() / (len(CLASSES) * counts[present])) ** power
    return torch.tensor(w / w[present].mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    ds = loader.dataset
    return torch.cat([model(ds.normalise(x)) for x, _ in loader]).numpy()


def train_model(train_imgs, train_y, val_imgs, val_y, cfg: dict, seed: int, max_epochs=None,
                patience=None, verbose=True) -> tuple[nn.Module, dict]:
    """Train from scratch; keep the epoch with the best validation macro-F1."""
    set_seed(seed)
    t = cfg["train"]
    if t.get("num_threads"):
        torch.set_num_threads(t["num_threads"])
    max_epochs = max_epochs or t["max_epochs"]
    patience = patience or t["patience"]

    mean = float(train_imgs.mean() / 255.0)
    std = float(train_imgs.std() / 255.0)          # statistics from the TRAINING split only
    g = torch.Generator().manual_seed(seed)
    train_ds = ThermalDataset(train_imgs, train_y, mean, std)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, generator=g)
    # Learning curves use a fixed 3,000-image sample of the training split (saves CPU time)
    sub = np.random.default_rng(seed).choice(len(train_y), size=min(3000, len(train_y)), replace=False)
    train_eval_dl = DataLoader(ThermalDataset(train_imgs[sub], train_y[sub], mean, std), batch_size=512)
    val_dl = DataLoader(ThermalDataset(val_imgs, val_y, mean, std), batch_size=512)

    model = SolarTriageNet(len(CLASSES), tuple(cfg["model"]["channels"]), cfg["model"]["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_y, t["class_weight_power"]),
                                  label_smoothing=t["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    history, best, best_state, wait = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running, n = 0.0, 0
        for x, y in train_dl:
            x = train_ds.normalise(augment_batch(x.clone(), cfg["augment"]))
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        sched.step()

        tr_logits, va_logits = predict_logits(model, train_eval_dl), predict_logits(model, val_dl)
        row = {
            "epoch": epoch,
            "train_loss_aug": running / n,
            "train_loss": float(F.cross_entropy(torch.from_numpy(tr_logits), torch.from_numpy(train_y[sub]))),
            "val_loss": float(F.cross_entropy(torch.from_numpy(va_logits), torch.from_numpy(val_y))),
            "train_macro_f1": f1_score(train_y[sub], tr_logits.argmax(1), average="macro"),
            "val_macro_f1": f1_score(val_y, va_logits.argmax(1), average="macro"),
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        if verbose:
            print(f"  seed {seed} epoch {epoch:2d} | train loss {row['train_loss']:.3f} F1 {row['train_macro_f1']:.3f}"
                  f" | val loss {row['val_loss']:.3f} F1 {row['val_macro_f1']:.3f} | {row['seconds']:.1f}s", flush=True)
        if row["val_macro_f1"] > best:
            best, best_state, wait = row["val_macro_f1"], copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    info = {"seed": seed, "mean": mean, "std": std, "best_epoch": best_epoch,
            "best_val_macro_f1": best, "history": history}
    return model, info
