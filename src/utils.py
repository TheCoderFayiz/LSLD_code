import os
import random
import numpy as np
import torch

# ---------------------------
# Device & Reproducibility
# ---------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------
# Sociocultural encodings
# ---------------------------
gender_mapping = {'Woman': 0, 'Man': 1}

locale_mapping = {'IN': 0, 'US': 1}

race_mapping = {
    'Asian/Asian subcontinent': 1,
    'White': 5,
    'LatinX, Latino, Hispanic or Spanish Origin': 2,
    'Multiracial': 3,
    'Black/African American': 0,
    'Other': 4
}

age_mapping = {'gen z': 0, 'millenial': 1, 'gen x+': 2}

education_mapping = {
    'High school or below': 0,
    'College degree or higher': 1,
    'Other': 2
}

# ---------------------------
# Calibration ranges
# ---------------------------
# symmetric percentile intervals across median to find calibration
quantile_ranges = [
    (0, 0),
    (0.475, 0.525),
    (0.45, 0.55),
    (0.40, 0.60),
    (0.35, 0.65),
    (0.30, 0.70),
    (0.25, 0.75),
    (0.20, 0.80),
    (0.15, 0.85),
    (0.10, 0.90),
    (0.05, 0.95),
    (0.025, 0.975),
    (0, 1),
]

def compute_quantile_intervals(values, quantile_ranges, mean_value):
    """
    For a list of predicted probabilities for an item, compute symmetric
    quantile intervals and count how many contain the item's mean label.
    Returns:
        hits_per_range: list[int] of len(quantile_ranges), 1 if mean inside interval else 0
        intervals: list[tuple(lower, upper, contains_mean)]
    """
    hits_per_range = []
    intervals = []
    arr = np.asarray(values, dtype=float)
    for (lo_q, hi_q) in quantile_ranges:
        lower = np.quantile(arr, lo_q)
        upper = np.quantile(arr, hi_q)
        contains = (lower <= mean_value) and (mean_value <= upper)
        hits_per_range.append(1 if contains else 0)
        intervals.append((float(lower), float(upper), bool(contains)))
    return hits_per_range, intervals