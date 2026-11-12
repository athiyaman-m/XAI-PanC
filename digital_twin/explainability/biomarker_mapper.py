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
    return BIOMARKER_ORGAN_MAP.get(biomarker_name, [])


def get_organ_biomarkers(organ_name: str) -> list[str]:
    return ORGAN_BIOMARKER_MAP.get(organ_name, [])


def compute_biomarker_impact(biomarker_name: str, value: float,
                             threshold: float = None) -> float:
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
    ratio = (value - th) / th
    return min(1.0, ratio / 3.0)
