"""
SolarTriage demo: upload thermal crops of solar modules and get a triage decision.

    streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from solartriage.inference import TriageModel

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MAX_FILES = 500

st.set_page_config(page_title="SolarTriage", page_icon="☀️", layout="wide")


@st.cache_resource
def get_model():
    return TriageModel.load()


try:
    base = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # corrupted file / version mismatch
    st.error(f"Could not load the model: {e}")
    st.stop()

st.title("☀️ SolarTriage: thermal-image triage for solar modules")
st.write(
    "Upload infrared crops of individual solar modules (as produced by a drone thermography survey). "
    "Each module is either **cleared** or **flagged for expert review**, with the most likely fault type. "
    "The goal is to cut the number of images a human expert has to inspect while still catching most faults."
)

with st.sidebar:
    st.header("Operating point")
    keys = sorted(base.pkg["thresholds"], key=float)
    key = st.select_slider("Target share of faulty modules to catch (recall)", options=keys,
                           value=base.pkg["default_recall"],
                           format_func=lambda k: f"{float(k):.0%}")
    st.caption("Thresholds were chosen on the validation set. Higher recall catches more faults "
               "but sends more healthy modules to the expert.")
    model = TriageModel(base.pkg, key)
    st.metric("Decision threshold on P(anomaly)", f"{model.threshold:.3f}")
    st.divider()
    st.caption(f"Model: SolarTriageNet (seed {base.pkg['seed']}), temperature-calibrated "
               f"(T = {base.pkg['temperature']:.2f}).")

tab_upload, tab_samples = st.tabs(["Upload images", "Try sample images"])
files = []
with tab_upload:
    uploads = st.file_uploader("Thermal module crops (JPG / PNG / TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                               accept_multiple_files=True)
    if uploads:
        if len(uploads) > MAX_FILES:
            st.warning(f"Only the first {MAX_FILES} files are processed.")
        files = [(u.name, u.getvalue()) for u in uploads[:MAX_FILES]]
with tab_samples:
    sample_paths = sorted(SAMPLES.glob("*.jpg"))
    if not sample_paths:
        st.info("No sample images found in samples/.")
    elif st.button(f"Run triage on {len(sample_paths)} sample test images", type="primary"):
        files = [(p.name, p.read_bytes()) for p in sample_paths]
        st.caption("Samples are held-out TEST images; the true class is the part of the file name after the id.")

if files:
    preds, errors = model.predict_files(files)
    for e in errors:
        st.error(e)
    if preds:
        n = len(preds)
        flagged = [p for p in preds if p.decision.startswith("FLAG")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Modules analysed", n)
        c2.metric("Flagged for expert review", f"{len(flagged)} ({len(flagged) / n:.0%})")
        c3.metric("Cleared automatically", n - len(flagged))

        rows = [{"file": p.name, "decision": p.decision, "P(anomaly)": round(p.anomaly_probability, 3),
                 "most likely": f"{p.top_classes[0][0]} ({p.top_classes[0][1]:.0%})",
                 "2nd": f"{p.top_classes[1][0]} ({p.top_classes[1][1]:.0%})",
                 "suggested action (illustrative)": p.action_hint,
                 "warnings": "; ".join(p.warnings)} for p in preds]
        table = pd.DataFrame(rows).sort_values("P(anomaly)", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", table.to_csv(index=False).encode(), "solartriage_results.csv",
                           "text/csv")

        st.subheader("Flagged modules")
        show = sorted(flagged, key=lambda p: -p.anomaly_probability)[:24]
        lookup = dict(files)
        cols = st.columns(8)
        for i, p in enumerate(show):
            with cols[i % 8]:
                st.image(lookup[p.name], width=72)
                st.caption(f"{p.top_classes[0][0]}\n{p.anomaly_probability:.2f}")
        if not show:
            st.write("No module was flagged.")

st.divider()
st.caption(
    "**Limitations.** Trained on 24×40-pixel crops from one public dataset (Raptor Maps InfraredSolarModules, "
    "MIT licence). Real deployments differ in camera, altitude, climate and module technology, and an expert "
    "normally compares a module with its neighbours, which a single crop cannot show. Flagged modules must be "
    "checked by a qualified person; 'cleared' does not guarantee a healthy module. Suggested actions are an "
    "illustrative grouping, not maintenance advice."
)
