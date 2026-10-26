"""
Automated checks.   python -m pytest -q
Tests that need prepared data or the trained model are skipped when those files are absent.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solartriage import CLASSES  # noqa: E402
from solartriage.config import MODELS_DIR, load_config  # noqa: E402
from solartriage.data import (clean_duplicates, connected_groups, group_stratified_split,  # noqa: E402
                              safe_extract)
from solartriage.inference import InputError, TriageModel, load_image  # noqa: E402
from solartriage.metrics import (ap_at_prevalence, deployment_view, rates_at,  # noqa: E402
                                 threshold_for_recall)
from solartriage.model import SolarTriageNet  # noqa: E402
from solartriage.training import augment_batch  # noqa: E402

CFG = load_config()
PROCESSED = CFG["data"]["processed_dir"] / "splits.csv"
MODEL = MODELS_DIR / "solartriage_cnn.pt"


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------- data ----------------
def test_safe_extract_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("InfraredSolarModules/../../evil.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(z, tmp_path / "out")


def test_connected_groups_links_chains():
    g = connected_groups(5, [(0, 1), (1, 2)])
    assert g[0] == g[1] == g[2] and g[3] != g[0] and g[4] != g[3]


def test_clean_duplicates_drops_conflicting_labels():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (5, 40, 24), dtype=np.uint8)
    imgs[1] = imgs[0]          # same image, different label -> both removed
    imgs[3] = imgs[2]          # same image, same label -> one kept
    df = pd.DataFrame({"image_id": range(5), "path": "", "label": ["Cell", "Diode", "Soiling", "Soiling", "Cell"]})
    out, _, rep = clean_duplicates(df, imgs, max_hamming=0, max_mad=0)
    assert rep["images_removed_label_conflict"] == 2
    assert rep["images_removed_exact_duplicate"] == 1
    assert sorted(out["image_id"]) == [2, 4]


def test_group_split_never_separates_a_group():
    df = pd.DataFrame({"label": np.repeat(["A", "B"], 200), "group": np.arange(400) // 4})
    s = group_stratified_split(df, 20, 3, 3, seed=0)
    assert df.assign(s=s).groupby("group")["s"].nunique().max() == 1
    assert set(s) == {"train", "val", "test"}


@pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/prepare_data.py first")
def test_prepared_splits_have_no_leakage():
    df = pd.read_csv(PROCESSED)
    assert df["image_id"].is_unique
    assert df.groupby("group")["split"].nunique().max() == 1
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["test"] - 0.15) < 0.02 and abs(frac["val"] - 0.15) < 0.02
    for c in CLASSES:  # every class present in every split
        assert (df[df.label == c]["split"].nunique()) == 3


# ---------------- metrics ----------------
def test_threshold_for_recall_meets_target():
    rng = np.random.default_rng(1)
    s = rng.random(1000); y = (rng.random(1000) < 0.5).astype(int)
    thr = threshold_for_recall(s, y, 0.95)
    assert rates_at(s, y, thr)[0] >= 0.95


def test_deployment_view_arithmetic():
    d = deployment_view(tpr=0.9, fpr=0.1, prevalence=0.02)
    assert d["review_workload"] == pytest.approx(0.02 * 0.9 + 0.98 * 0.1)
    assert d["precision_at_prevalence"] == pytest.approx(0.018 / (0.018 + 0.098))
    assert d["missed_per_10k"] == pytest.approx(20)


def test_ap_at_prevalence_perfect_and_lower_for_noisy():
    y = np.array([0] * 50 + [1] * 50)
    assert ap_at_prevalence(y.astype(float), y, 0.02) == pytest.approx(1.0)
    noisy = y + np.random.default_rng(0).normal(0, 0.6, 100)
    assert ap_at_prevalence(noisy, y, 0.02) < ap_at_prevalence(noisy, y, 0.5)


# ---------------- model / augmentation ----------------
def test_model_output_shape():
    out = SolarTriageNet()(torch.zeros(3, 1, 40, 24))
    assert out.shape == (3, len(CLASSES))


def test_augmentation_keeps_shape_and_range():
    x = torch.rand(16, 1, 40, 24)
    y = augment_batch(x.clone(), CFG["augment"])
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


# ---------------- inference input validation ----------------
def test_load_image_rejects_non_images():
    with pytest.raises(InputError):
        load_image(b"this is not an image", "notes.txt")


def test_load_image_native_size_has_no_warnings():
    arr, warns = load_image(_png_bytes(np.random.default_rng(0).integers(0, 255, (40, 24), dtype=np.uint8)))
    assert arr.shape == (40, 24) and warns == []


def test_load_image_resizes_and_rotates_with_warnings():
    rgb = np.random.default_rng(0).integers(0, 255, (48, 80, 3), dtype=np.uint8)  # landscape colour image
    arr, warns = load_image(_png_bytes(rgb))
    assert arr.shape == (40, 24)
    assert any("rotated" in w for w in warns) and any("greyscale" in w for w in warns)


# ---------------- trained model ----------------
@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/train_cnn.py first")
def test_trained_model_predicts_valid_output():
    tm = TriageModel.load()
    samples = sorted((ROOT / "samples").glob("*.jpg"))[:10]
    preds, errors = tm.predict_files(samples)
    assert not errors and len(preds) == len(samples)
    for p in preds:
        assert 0 <= p.anomaly_probability <= 1
        assert p.decision in ("CLEAR", "FLAG for expert review")
        assert abs(sum(prob for _, prob in p.top_classes)) <= 1 + 1e-6


@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/train_cnn.py first")
def test_thresholds_increase_recall_monotonically():
    th = TriageModel.load().pkg["thresholds"]
    ordered = [th[k] for k in sorted(th, key=float)]
    assert all(a >= b for a, b in zip(ordered, ordered[1:]))  # higher recall -> lower threshold


@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/train_cnn.py first")
def test_streamlit_app_runs_on_samples():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Modules analysed" for m in at.metric)
