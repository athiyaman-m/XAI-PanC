import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_ORGANS = [
    "Pancreas", "Liver", "Gallbladder", "Bile Duct",
    "Duodenum", "Stomach", "Spleen", "Lungs",
    "Lymph Nodes", "Heart", "Kidneys",
]

ORGAN_CENTRES = {
    "Pancreas":     (0.0, -0.10, 0.05),
    "Liver":        (0.55, 0.18, 0.55),
    "Gallbladder":  (0.78, 0.32, 0.28),
    "Bile Duct":    (0.50, 0.24, 0.12),
    "Duodenum":     (0.38, 0.04, -0.04),
    "Stomach":      (-0.26, 0.26, 0.12),
    "Spleen":       (-1.02, -0.08, 0.35),
    "Lungs":        (0.0, 0.04, 1.15),
    "Lymph Nodes":  (0.0, -0.08, 0.08),
    "Heart":        (-0.10, 0.22, 1.00),
    "Kidneys":      (0.0, -0.18, 0.0),
}

SPREAD_PATHS = {
    "Pancreas":  ["Lymph Nodes", "Liver", "Duodenum", "Stomach"],
    "Liver":     ["Lungs", "Bile Duct"],
    "Duodenum":  ["Stomach", "Lymph Nodes"],
    "Lymph Nodes": ["Lungs", "Liver", "Pancreas"],
}

SEED = 42
