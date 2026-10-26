"""Copy 2 TEST-split images per class into samples/ for the demo (file name = <id>_<class>.jpg)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from solartriage import CLASSES
from solartriage.config import ROOT, load_config


def main(per_class: int = 2):
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_dir"] / "splits.csv")
    src = cfg["data"]["raw_dir"] / "extracted" / "InfraredSolarModules"
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    test = df[df["split"] == "test"]
    for c in CLASSES:
        for _, r in test[test["label"] == c].sample(per_class, random_state=cfg["seed"]).iterrows():
            shutil.copyfile(src / r["path"], out / f"{r['image_id']}_{c}.jpg")
    print(f"wrote {len(list(out.glob('*.jpg')))} sample images to {out}")


if __name__ == "__main__":
    main()
