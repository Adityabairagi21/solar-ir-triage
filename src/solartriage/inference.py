"""
Inference on new thermal crops with the saved model package (models/solartriage_cnn.pt).

    from solartriage.inference import TriageModel
    tm = TriageModel.load()
    result = tm.predict_files(["module_001.jpg"])

Each result contains the triage decision (FLAG for expert review / CLEAR), the calibrated
anomaly probability, the most likely fault types and any input warnings.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from scipy.special import softmax

from solartriage import CLASSES, IMG_H, IMG_W, NORMAL_IDX
from solartriage.config import MODELS_DIR
from solartriage.model import SolarTriageNet

MAX_UPLOAD_PIXELS = 4_000_000   # refuse absurdly large images (decompression-bomb guard)

# Coarse, illustrative routing of fault types to the kind of field action they usually need.
# This grouping is the author's own simplification for the demo, not an industry standard.
ACTION_HINT = {
    "Cell": "Electrical / module inspection", "Cell-Multi": "Electrical / module inspection",
    "Hot-Spot": "Electrical / module inspection", "Hot-Spot-Multi": "Electrical / module inspection",
    "Cracking": "Physical damage: inspect / replace module",
    "Diode": "Electrical: check bypass diode / junction box", "Diode-Multi": "Electrical: check bypass diodes / junction box",
    "Offline-Module": "Electrical: module not producing (wiring / connector)",
    "Shadowing": "Site: check for obstruction / shading", "Vegetation": "Site: vegetation management",
    "Soiling": "Site: cleaning", "No-Anomaly": "No action",
}


class InputError(ValueError):
    """Raised for files that cannot be used as thermal crops."""


@dataclass
class Prediction:
    name: str
    decision: str
    anomaly_probability: float
    top_classes: list[tuple[str, float]]
    action_hint: str
    warnings: list[str] = field(default_factory=list)


def load_image(data: bytes | str | Path, name: str = "") -> tuple[np.ndarray, list[str]]:
    """Read any common image format, validate it and return a (40, 24) uint8 array + warnings."""
    warnings = []
    try:
        img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InputError(f"{name or 'file'}: not a readable image ({e.__class__.__name__}).") from e
    w, h = img.size
    if w * h > MAX_UPLOAD_PIXELS:
        raise InputError(f"{name}: image is too large ({w}x{h}); upload a single-module crop.")
    if w < 8 or h < 8:
        raise InputError(f"{name}: image is too small ({w}x{h}).")
    if img.mode not in ("L", "I;16", "I", "F"):
        warnings.append(f"converted from {img.mode} to greyscale; the model expects single-channel thermal intensity")
    img = img.convert("L")
    if (w, h) != (IMG_W, IMG_H):
        if w > h:
            warnings.append("image is landscape; rotated 90° to match the portrait 24x40 training crops")
            img = img.rotate(90, expand=True)
            w, h = img.size
        aspect = (h / w) / (IMG_H / IMG_W)
        if not 0.75 <= aspect <= 1.33:
            warnings.append(f"aspect ratio differs from a module crop (h/w = {h / w:.2f}, expected {IMG_H / IMG_W:.2f})")
        warnings.append(f"resized from {w}x{h} to {IMG_W}x{IMG_H}; predictions are most reliable on native crops")
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.std() < 1.0:
        warnings.append("image is almost uniform; there may be no thermal content")
    return arr, warnings


class TriageModel:
    def __init__(self, package: dict, recall_key: str | None = None):
        self.pkg = package
        self.model = SolarTriageNet(len(CLASSES), tuple(package["model_cfg"]["channels"]),
                                    package["model_cfg"]["dropout"])
        self.model.load_state_dict(package["state_dict"])
        self.model.eval()
        self.recall_key = recall_key or package["default_recall"]

    @classmethod
    def load(cls, path: str | Path = MODELS_DIR / "solartriage_cnn.pt", recall_key: str | None = None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Run  python scripts/train_cnn.py  first.")
        return cls(torch.load(path, map_location="cpu", weights_only=False), recall_key)

    @property
    def threshold(self) -> float:
        return self.pkg["thresholds"][self.recall_key]

    @torch.no_grad()
    def probabilities(self, images: np.ndarray) -> np.ndarray:
        """images: (N, 40, 24) uint8 -> calibrated class probabilities (N, 12)."""
        x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
        x = (x - self.pkg["mean"]) / self.pkg["std"]
        logits = torch.cat([self.model(x[i:i + 512]) for i in range(0, len(x), 512)]).numpy()
        return softmax(logits / self.pkg["temperature"], axis=1)

    def predict_arrays(self, images: np.ndarray, names=None, warnings=None) -> list[Prediction]:
        probs = self.probabilities(images)
        out = []
        for i, p in enumerate(probs):
            p_anom = float(1 - p[NORMAL_IDX])
            order = np.argsort(-p)
            top = [(CLASSES[k], float(p[k])) for k in order[:3]]
            fault = next(CLASSES[k] for k in order if k != NORMAL_IDX)
            flag = p_anom >= self.threshold
            out.append(Prediction(
                name=names[i] if names else f"image_{i}",
                decision="FLAG for expert review" if flag else "CLEAR",
                anomaly_probability=p_anom,
                top_classes=top,
                action_hint=ACTION_HINT[fault] if flag else ACTION_HINT["No-Anomaly"],
                warnings=(warnings[i] if warnings else []),
            ))
        return out

    def predict_files(self, files) -> tuple[list[Prediction], list[str]]:
        """files: paths or (name, bytes) tuples. Returns predictions and per-file error messages."""
        arrays, names, warns, errors = [], [], [], []
        for f in files:
            name, data = (f if isinstance(f, tuple) else (Path(f).name, f))
            try:
                arr, w = load_image(data, name)
            except InputError as e:
                errors.append(str(e))
                continue
            arrays.append(arr); names.append(name); warns.append(w)
        preds = self.predict_arrays(np.stack(arrays), names, warns) if arrays else []
        return preds, errors
