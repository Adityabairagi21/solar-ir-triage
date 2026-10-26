"""
Data acquisition, cleaning, leakage-safe splitting and the PyTorch Dataset.

Pipeline (scripts/prepare_data.py):
    download zip -> verify SHA-256 -> safe extract -> load all images
    -> drop exact duplicates whose copies disagree on the label (label noise)
    -> keep one copy of agreeing exact duplicates
    -> group near-duplicates so that all copies land in the same split
    -> stratified, group-aware 70 / 15 / 15 split  -> data/processed/splits.csv
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from solartriage import CLASSES, IMG_H, IMG_W, NORMAL_CLASS

DATASET_DIRNAME = "InfraredSolarModules"


# ---------------------------------------------------------------------------
# Download + extraction
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(url: str, raw_dir: Path, expected_sha256: str) -> Path:
    """Download the official zip once and verify its checksum."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "InfraredSolarModules.zip"
    if not zpath.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zpath)
    digest = sha256_of(zpath)
    if digest != expected_sha256:
        raise ValueError(f"Checksum mismatch for {zpath}: {digest} (expected {expected_sha256}). "
                         "Delete the file and download again.")
    return zpath


def safe_extract(zpath: Path, out_dir: Path) -> Path:
    """Extract only the dataset folder, refusing absolute or parent-relative paths."""
    target = out_dir / DATASET_DIRNAME
    if (target / "module_metadata.json").exists():
        return target
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if member.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(f"Unsafe path in archive: {member}")
            if parts and parts[0] == DATASET_DIRNAME:
                z.extract(member, out_dir)
    return target


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_images(dataset_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return metadata (one row per image) and a uint8 array of shape (N, 40, 24)."""
    meta = json.loads((dataset_dir / "module_metadata.json").read_text())
    rows, arrays = [], []
    for key, v in meta.items():
        img = Image.open(dataset_dir / v["image_filepath"]).convert("L")
        if img.size != (IMG_W, IMG_H):
            raise ValueError(f"Unexpected size {img.size} for {v['image_filepath']}")
        arrays.append(np.asarray(img, dtype=np.uint8))
        rows.append({"image_id": int(key), "path": v["image_filepath"], "label": v["anomaly_class"]})
    df = pd.DataFrame(rows)
    unknown = set(df["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown classes in metadata: {unknown}")
    order = np.argsort(df["image_id"].to_numpy())
    return df.iloc[order].reset_index(drop=True), np.stack(arrays)[order]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def dhash64(images: np.ndarray) -> np.ndarray:
    """64-bit difference hash per image (resize to 9x8, compare neighbours)."""
    out = np.empty(len(images), dtype=np.uint64)
    weights = (1 << np.arange(64, dtype=np.uint64)).astype(np.uint64)
    for i, a in enumerate(images):
        small = np.asarray(Image.fromarray(a).resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).flatten().astype(np.uint64)
        out[i] = np.sum(bits * weights, dtype=np.uint64)
    return out


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    while np.any(x):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


def near_duplicate_pairs(images: np.ndarray, max_hamming: int, max_mad: float,
                         chunk: int = 1000) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, that are near-identical by hash AND by pixel difference."""
    hashes = dhash64(images)
    flat = images.reshape(len(images), -1).astype(np.int16)
    pairs = []
    for start in range(0, len(images), chunk):
        block = hashes[start:start + chunk]
        dist = _popcount64(block[:, None] ^ hashes[None, :])
        ii, jj = np.nonzero(dist <= max_hamming)
        ii = ii + start
        keep = jj > ii
        for i, j in zip(ii[keep], jj[keep]):
            if np.abs(flat[i] - flat[j]).mean() <= max_mad:
                pairs.append((int(i), int(j)))
    return pairs


def connected_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Union-find: group id per item, linking every pair."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def clean_duplicates(df: pd.DataFrame, images: np.ndarray, max_hamming: int, max_mad: float
                     ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    1. Exact duplicates (identical pixels) with DIFFERENT labels -> remove all copies.
    2. Exact duplicates with the SAME label -> keep the first copy.
    3. Near-duplicates -> keep all, but give them a shared `group` so the split
       never separates them (otherwise the test set would contain images the
       model has effectively already seen).
    """
    report = {"images_raw": int(len(df))}
    digests = [hashlib.md5(a.tobytes()).hexdigest() for a in images]
    df = df.assign(digest=digests)

    label_sets = df.groupby("digest")["label"].nunique()
    conflicting = label_sets[label_sets > 1].index
    conflict_mask = df["digest"].isin(conflicting)
    report["exact_dup_groups_conflicting_labels"] = int(len(conflicting))
    report["images_removed_label_conflict"] = int(conflict_mask.sum())

    dup_mask = df.duplicated("digest", keep="first") & ~conflict_mask
    report["images_removed_exact_duplicate"] = int(dup_mask.sum())

    keep = ~(conflict_mask | dup_mask)
    df, images = df[keep].reset_index(drop=True), images[keep.to_numpy()]

    pairs = near_duplicate_pairs(images, max_hamming, max_mad)
    groups = connected_groups(len(df), pairs)
    df["group"] = groups
    sizes = pd.Series(groups).value_counts()
    report["near_dup_pairs"] = len(pairs)
    report["near_dup_groups_size_gt1"] = int((sizes > 1).sum())
    report["images_in_near_dup_groups"] = int(sizes[sizes > 1].sum())
    report["largest_near_dup_group"] = int(sizes.max())
    pair_labels = [(df.at[i, "label"], df.at[j, "label"]) for i, j in pairs]
    report["near_dup_pairs_with_different_labels"] = int(sum(a != b for a, b in pair_labels))
    report["images_clean"] = int(len(df))
    return df.drop(columns="digest"), images, report


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def group_stratified_split(df: pd.DataFrame, n_folds: int, test_folds: int, val_folds: int,
                           seed: int) -> pd.Series:
    """Assign 'train' / 'val' / 'test' with class stratification and no group crossing splits."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
        fold[idx] = k
    split = np.where(fold < test_folds, "test", np.where(fold < test_folds + val_folds, "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def binary_labels(labels) -> np.ndarray:
    """1 = anomaly (any fault class), 0 = No-Anomaly."""
    return (np.asarray(labels) != NORMAL_CLASS).astype(int)


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the prepared split table and image tensor written by scripts/prepare_data.py."""
    splits = processed_dir / "splits.csv"
    arr = processed_dir / "images.npy"
    if not splits.exists() or not arr.exists():
        raise FileNotFoundError("Processed data not found. Run:  python scripts/prepare_data.py")
    return pd.read_csv(splits), np.load(arr)
