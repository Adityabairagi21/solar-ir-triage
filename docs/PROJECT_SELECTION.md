# Project Selection: SolarTriage

**Title:** SolarTriage: workload-aware deep-learning triage of aerial thermal images of solar modules at realistic fault prevalence

## 1. The real-world problem

Utility-scale solar farms are inspected with infrared (thermal) cameras on drones or piloted aircraft. Faulty modules (activated bypass diodes, hot cells, cracks, soiling, shading, modules that have gone offline) show up as abnormal heat patterns. The dataset authors at Raptor Maps describe the current bottleneck: *"Currently, trained experts are required to analyze this data. The authors seek to automate this process with machine learning to allow experts to focus on restoring power production, rather than manually searching through imagery."* (Millendorf et al., 2020). Survey images are cropped to modules and then classified by experts.

Only a small share of modules is faulty. The same paper cites a global aerial-inspection report finding that **2.2 % of all modules have anomalies**. An expert therefore spends most of the time looking at healthy modules.

**Who experiences it:** thermography analysts and O&M (operations and maintenance) teams at solar farms and inspection service providers. Missed faults cost energy and revenue; false alarms cost analyst time and truck rolls.

## 2. Evidence that the problem and the gap exist (verified sources)

| Evidence | Source |
|---|---|
| Experts are currently needed to analyse thermal imagery; the aim of automation is to save their time | Millendorf, Obropta & Vadhavkar, *Infrared Solar Module Dataset for Anomaly Detection*, ICLR 2020 AI4Earth workshop. https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf |
| Real farms: 2.2 % of modules anomalous; the dataset is deliberately ~50 % anomalous; the authors "highlight the class imbalance that is prevalent in solar PV anomaly detection" | same paper, Section 3.2 |
| Published results on this dataset are mostly overall accuracy (e.g. 85 %, 88 %, 92.5 %, 93 %), and a plain Random Forest reaches 84.0 %. A single crop also lacks the neighbouring-panel context that experts use | Corley et al., *Aerial Infrared Health Monitoring of Solar Photovoltaic Farms at Scale*, arXiv:2503.02128 (2025). https://arxiv.org/abs/2503.02128 |
| Domain shift between PV plants and unknown anomaly types are recognised open problems (studied there on a private 2.92-million-image dataset) | Bommes et al., *Anomaly Detection in IR Images of PV Modules using Supervised Contrastive Learning*, arXiv:2112.02922 (2021). https://arxiv.org/abs/2112.02922 |
| The dataset itself: 20,000 crops, 24×40 px, 12 classes, MIT licence | https://github.com/RaptorMaps/InfraredSolarModules |

## 3. Existing solutions and their limitations

- **Commercial inspection platforms** (for example, Raptor Maps itself) combine software and human analysts. Their models are proprietary and cannot be evaluated publicly.
- **Academic work on InfraredSolarModules** focuses on raising overall or multi-class accuracy on the balanced dataset. Accuracy on a 50 %-anomalous test set says little about a farm where 97.8 % of modules are healthy. There, the number of false alarms per true fault, and therefore the analyst workload, is what matters.
- **Rule-based hot-spot thresholding** (flag a module that is much hotter in one place than overall) is simple, but it cannot tell cell faults from shading, soiling or vegetation.
- Some reported results augment or oversample **before** splitting the data, which leaks near-copies of test images into training and inflates scores (noted in our literature search; individual papers were not all verified).

## 4. The specific gap this project addresses

In our literature search we did **not find** a public, reproducible evaluation on InfraredSolarModules that:
1. reports performance in **deployment terms**: the share of all modules an expert still has to review, and precision at the real-world 2.2 % prevalence, at a recall level fixed in advance;
2. checks whether a model still **flags a fault type it has never seen** (leave-one-fault-type-out), which matters because new or rare fault types appear in the field; and
3. uses a **leakage-controlled split**: duplicates with conflicting labels removed, near-duplicates kept in the same split, and thresholds and calibration chosen on validation data only.

We do **not** claim to be the first to classify this dataset, and we do not claim state-of-the-art accuracy.

## 5. What is different about this approach

- The model is a **triage tool**, not just a classifier: one compact CNN gives a calibrated P(anomaly) for the flag/clear decision plus the likely fault type, so flagged modules can be routed to the right kind of crew.
- The operating point is set **in advance on validation data** (95 % anomaly recall), and the results are translated into analyst workload, false alarms and missed faults per 10,000 modules.
- Fair comparison against three baselines on the **same split**: a hot-spot rule, logistic regression and a Random Forest (the strongest classical result reported for this dataset).
- An open-set (leave-one-fault-type-out) robustness experiment.
- An edge-friendly model (about 0.3 M parameters), matching the dataset authors' stated milestone of *"real-time classification that can be used by low performance edge-computing devices"*.

## 6. Why deep learning

The fault types are spatial patterns: a hot strip covering one-third of the module (diode), checkerboard hot cells, gradients from shading or vegetation, and full-module heating. Convolutional networks learn such local and translation-tolerant patterns directly from pixels. Classical models on raw pixels treat each pixel independently of its position, and fixed rules see only a single statistic. Whether the CNN's advantage is real is **tested, not assumed**: the Random Forest baseline is strong, and we report the difference with bootstrap confidence intervals.

## 7. Dataset

| Property | Value |
|---|---|
| Name | InfraredSolarModules (Raptor Maps) |
| Source | https://github.com/RaptorMaps/InfraredSolarModules (zip 15.5 MB, SHA-256 verified in `prepare_data.py`) |
| Licence | MIT (Copyright 2020 Raptor Maps, Inc.) |
| Content | 20,000 single-channel 24×40 px JPEG crops of individual modules; 11 anomaly classes + No-Anomaly (10,000) |
| Labels | `module_metadata.json`, one class per image, labelled by Raptor Maps experts using thermal and visible imagery |
| Known limitations | No site, flight or date identifiers (a split by site is impossible); unknown radiometric scaling (pixel values are not temperatures); class counts differ slightly from the paper; 22 exact-duplicate groups, 6 with conflicting labels (found in this project) |

## 8. Architecture, metrics and baselines

- **SolarTriageNet:** 3 residual stages (32-64-128 channels), max-pooling 40×24 → 5×3, global average pooling, dropout and a 12-way softmax; about 0.3 M parameters; temperature-calibrated.
- **Metrics:** 12-class macro-F1, balanced accuracy and per-class F1. For triage: ROC-AUC, average precision at 2.2 % prevalence, and, at the pre-set 95 % recall operating point, review workload, precision at prevalence, false alarms and missed faults per 10k modules. Also calibration (ECE), open-set recall, and CPU latency.
- **Baselines:** hot-spot rule, logistic regression and Random Forest, all on the same splits.

## 9. Feasibility, challenges and scope

- **Feasible on a laptop CPU:** 15.5 MB of data and about 20 s per training epoch on 14 CPU threads.
- **Challenges:** severe class imbalance (175 Diode-Multi vs 10,000 normal images), tiny images, no site identifiers, and unknown temperature calibration.
- **Out of scope:** detection on full drone mosaics, combining visible and thermal imagery, a site-level split, and field validation. These are listed as future work.

## 10. Candidates considered

Each candidate was scored 1 (poor) to 5 (strong). "Researched" means sources were checked during this project; "screened" means it was assessed from general knowledge only and rejected early.

| # | Candidate | Importance | Evidence | Data access / licence | CPU feasibility | Not a tutorial | Measurable | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | **Solar thermal-image triage at real prevalence** | 4 | 5 | 5 (MIT, 15 MB) | 5 | 4 | 5 | **Selected** (researched) |
| 2 | Wearable fall detection: simulated → real falls | 5 | 5 | 1 (FARSEEING real falls need an access request) | 4 | 4 | 3 | Rejected (researched) |
| 3 | Electricity-theft detection (SGCC smart meters) | 4 | 3 | 2 (redistribution and licence unclear) | 4 | 3 | 4 | Rejected (researched) |
| 4 | ECG arrhythmia, inter-patient split (MIT-BIH) | 4 | 5 | 5 | 4 | 2 (very common) | 4 | Rejected (screened) |
| 5 | Bearing faults across loads (CWRU) | 3 | 4 | 4 | 5 | 2 | 4 | Rejected (screened) |
| 6 | Road-damage detection (RDD2022) | 4 | 4 | 4 | 1 (GB-scale detection, GPU) | 3 | 4 | Rejected (screened) |
| 7 | Mosquito species from wingbeat audio | 4 | 3 | 3 | 2 (large audio) | 4 | 4 | Rejected (screened) |
| 8 | Machine-sound anomaly detection (MIMII / DCASE) | 3 | 4 | 4 | 2 (multi-GB) | 3 | 4 | Rejected (screened) |
| 9 | Sign-language recognition | 4 | 3 | 2 (licences / logins vary) | 3 | 2 | 3 | Rejected (screened) |
| 10 | Air-quality forecasting with LSTMs | 4 | 4 | 4 | 5 | 2 | 3 (deep learning often does not beat GBMs on tabular data) | Rejected (screened) |
| 11 | PV electroluminescence cell defects (ELPV) | 3 | 4 | 3 (non-commercial licence) | 4 | 3 | 4 | Runner-up (screened) |
