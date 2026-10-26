# SolarTriage

Workload-aware deep-learning triage of aerial thermal images of solar modules (InfraredSolarModules, 24 × 40 px, 12 classes).

A compact residual CNN (SolarTriageNet, 299k parameters, 1.2 MB) classifies each module crop and decides whether an expert needs to look at it. Evaluation is done at the real-world anomaly rate of about 2.2 %, with a leakage-controlled split and a leave-one-fault-type-out test.

## Results (test set)

| | SolarTriageNet (CNN) | Random Forest | Logistic Regression | Hot-spot rule |
|---|---|---|---|---|
| 12-class macro-F1 | **0.688** | 0.441 | 0.248 | – |
| ROC-AUC (anomaly vs. healthy) | **0.979** | 0.907 | 0.775 | 0.823 |
| Review workload at 95 % recall, 2.2 % prevalence | **16.7 %** | 47.9 % | 73.4 % | 74.1 % |

- Calibration error (ECE) 0.150 → 0.030 after temperature scaling.
- Never-seen fault types are still flagged 89 % of the time on average, but only 46 % for Offline-Module and 66 % for Shadowing.

Full write-up: [`report/SolarTriage_Report.pdf`](report/SolarTriage_Report.pdf).

## Run

```
python -m venv .venv
.venv\Scripts\activate                      (Windows)   |   source .venv/bin/activate   (Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py                        # uses the trained model in models/
```

Reproduce everything from scratch:

```
python scripts/prepare_data.py              # download (15.5 MB), verify, clean, split
python scripts/train_cnn.py                 # 3 seeds, about 45 min on a 14-thread CPU
python scripts/run_baselines.py
python scripts/open_set_experiment.py       # 11 retrainings, about 60 min
python scripts/evaluate.py                  # tables in results/, figures in results/figures/
python -m pytest -q
```

## Data

InfraredSolarModules by Raptor Maps, MIT licence: https://github.com/RaptorMaps/InfraredSolarModules. The dataset is downloaded by `scripts/prepare_data.py` and is not stored in this repository.
