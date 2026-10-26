"""SolarTriage: workload-aware deep-learning triage of aerial thermal images of solar modules."""

CLASSES = [
    "No-Anomaly", "Cell", "Cell-Multi", "Cracking", "Hot-Spot", "Hot-Spot-Multi",
    "Shadowing", "Diode", "Diode-Multi", "Vegetation", "Soiling", "Offline-Module",
]
NORMAL_CLASS = "No-Anomaly"
NORMAL_IDX = CLASSES.index(NORMAL_CLASS)
IMG_H, IMG_W = 40, 24  # images are 24 px wide and 40 px tall

__version__ = "1.0.0"
