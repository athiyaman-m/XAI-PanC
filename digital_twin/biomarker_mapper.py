"""
biomarker_mapper.py — Map lab biomarkers to specific organs.

Provides bidirectional lookup:
  - Which organs are affected by a given biomarker?
  - Which biomarkers are relevant to a given organ?
"""

BIOMARKER_ORGAN_MAP = {
    "CA 19-9 (U/mL)": ["Pancreas", "Bile Duct", "Liver"],
    "CEA (ng/mL)": ["Pancreas", "Liver", "Lungs"],
    "Bilirubin (mg/dL)": ["Bile Duct", "Liver"],
    "Amylase (U/L)": ["Pancreas", "Duodenum"],
    "Weight Loss (%)": ["Pancreas", "Stomach", "Duodenum"],
}

ORGAN_BIOMARKER_MAP: dict[str, list[str]] = {}
for bio, organs in BIOMARKER_ORGAN_MAP.items():
    for org in organs:
        ORGAN_BIOMARKER_MAP.setdefault(org, []).append(bio)


def get_affected_organs(biomarker_name: str) -> list[str]:
    """Return organ names affected by *biomarker_name*."""
    return BIOMARKER_ORGAN_MAP.get(biomarker_name, [])


def get_organ_biomarkers(organ_name: str) -> list[str]:
    """Return biomarker names relevant to *organ_name*."""
    return ORGAN_BIOMARKER_MAP.get(organ_name, [])


def compute_biomarker_impact(biomarker_name: str, value: float,
                             threshold: float = None) -> float:
    """
    Return a normalised impact score (0-1) for a biomarker value
    exceeding its threshold.  Used to modulate organ health scores.
    """
    thresholds = {
        "CA 19-9 (U/mL)": 37,
        "CEA (ng/mL)": 5,
        "Bilirubin (mg/dL)": 1.2,
        "Amylase (U/L)": 110,
        "Weight Loss (%)": 5,
    }
    th = threshold if threshold is not None else thresholds.get(biomarker_name, 0)
    if th == 0:
        return 0.0
    if value <= th:
        return 0.0
    # Normalised excess (capped at 3x threshold → 1.0)
    ratio = (value - th) / th
    return min(1.0, ratio / 3.0)
