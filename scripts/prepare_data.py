"""
Step 1: download, verify, clean and split the InfraredSolarModules dataset.

    python scripts/prepare_data.py

Writes data/processed/{images.npy, splits.csv, cleaning_report.json}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from solartriage.config import load_config
from solartriage.data import (clean_duplicates, download_dataset, group_stratified_split,
                              load_images, safe_extract)


def main():
    cfg = load_config()
    d = cfg["data"]
    zpath = download_dataset(d["url"], d["raw_dir"], d["sha256"])
    dataset_dir = safe_extract(zpath, d["raw_dir"] / "extracted")
    print("Loading images ...")
    df, images = load_images(dataset_dir)

    print("Removing duplicates and grouping near-duplicates ...")
    df, images, report = clean_duplicates(df, images, d["near_dup_max_hamming"], d["near_dup_max_mad"])

    df["split"] = group_stratified_split(df, d["n_split_folds"], d["test_folds"], d["val_folds"], cfg["seed"])

    # Leakage checks: no image id and no near-duplicate group appears in two splits
    assert df["image_id"].is_unique
    assert df.groupby("group")["split"].nunique().max() == 1, "a near-duplicate group crosses splits"

    out = d["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", images)
    df.to_csv(out / "splits.csv", index=False)

    report["split_sizes"] = df["split"].value_counts().to_dict()
    report["class_counts_by_split"] = df.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict("index")
    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        if k != "class_counts_by_split":
            print(f"  {k}: {v}")
    print(f"Saved {len(df)} images to {out}")


if __name__ == "__main__":
    main()
