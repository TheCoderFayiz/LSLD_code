import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr, spearmanr

from .dataset import CustomDataset
from .utils import quantile_ranges, compute_quantile_intervals
from .visualization import plot_item_boxplots, plot_calibration


def _iqr_overlap_fraction(probabilities, threshold):
    """
    Given a list of probabilities, compute the IQR and its overlap with [threshold,1] and [0,threshold].
    Returns fraction favoring the positive side: overlap_high / (overlap_high + overlap_low)
    """
    q1, q3 = np.percentile(probabilities, [25, 75])
    overlap_high = max(0.0, min(q3, 1.0) - max(q1, threshold))
    overlap_low  = max(0.0, min(q3, threshold) - max(q1, 0.0))
    denom = (overlap_high + overlap_low)
    return (overlap_high / denom) if denom > 0 else 0.5, overlap_high, overlap_low


def evaluate_model(model, dataloader_test, test_df, device, threshold=0.406):
    """
    Full evaluation mirroring the notebook:
      - Per-item probability distributions
      - Pearson/Spearman between per-item mean prob and true mean label
      - Majority-label prediction via IQR overlap; macro F1
      - Boxplot coverage (% of red dots within whiskers)
      - Calibration coverage across symmetric quantile bands; slope/intercept
      - Overall ROC-AUC on all test annotations
    """
    model.eval()
    # -------------------------------
    # 1) AUC on the whole test set
    # -------------------------------
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for target_embedding, other_features, labels, unique_id, avg_label in dataloader_test:
            probs = model(target_embedding, other_features).detach().cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

    if len(set(all_labels)) > 1:
        auc_score = roc_auc_score(np.asarray(all_labels), np.asarray(all_probs))
    else:
        auc_score = float("nan")
    print(f"AUC Score (all annotations): {auc_score:.6f}")

    # -------------------------------------------------------
    # 2) Per-item distributions, correlations, majority F1
    # -------------------------------------------------------
    per_item_probs = {}   # item_id -> list of probs
    per_item_truth = {}   # item_id -> mean of labels
    per_item_majority_truth = {}  # 0/1 majority label by true mean > 0.5

    grouped = test_df.groupby("item_id")
    with torch.no_grad():
        for item_id, gdf in grouped:
            X = gdf.drop("Q_overall", axis=1, inplace=False)
            y = gdf["Q_overall"]
            tempset = CustomDataset(X, y, device)
            temploader = DataLoader(tempset, batch_size=len(tempset), shuffle=False)

            for target_embedding, other_features, labels, unique, avgz in temploader:
                probs = model(target_embedding, other_features).detach().cpu().numpy().flatten().tolist()
                per_item_probs[item_id] = probs
                labels_np = labels.detach().cpu().numpy().astype(float).flatten()
                per_item_truth[item_id] = float(labels_np.mean())
                per_item_majority_truth[item_id] = 1 if per_item_truth[item_id] > 0.5 else 0

    # Pearson/Spearman between per-item mean prob and per-item true mean
    mpd_pred = [float(np.mean(per_item_probs[i])) for i in per_item_probs.keys()]
    mpd_label = [per_item_truth[i] for i in per_item_probs.keys()]
    if len(mpd_pred) >= 2 and np.std(mpd_pred) > 0 and np.std(mpd_label) > 0:
        pear = pearsonr(mpd_pred, mpd_label)[0]
        spear = spearmanr(mpd_pred, mpd_label).correlation
    else:
        pear, spear = float("nan"), float("nan")

    print(f"Pearson Correlation (item means): {pear}")
    print(f"Spearman Correlation (item ranks): {spear}")

    # Majority label prediction via IQR overlap
    pred_majority = {}       # item_id -> 0/1
    pred_majority_prob = {}  # item_id -> fraction favoring positive
    for iid, probs in per_item_probs.items():
        frac_pos, hi, lo = _iqr_overlap_fraction(probs, threshold)
        pred_majority_prob[iid] = frac_pos
        pred_majority[iid] = 1 if hi > lo else 0

    y_true_m = [per_item_majority_truth[i] for i in sorted(per_item_probs.keys())]
    y_pred_m = [pred_majority[i] for i in sorted(per_item_probs.keys())]
    majority_f1 = f1_score(y_true_m, y_pred_m, average='macro')
    print(f"Majority-label Macro F1: {majority_f1:.6f}")

    # -------------------------------------------------------
    # 3) Boxplot coverage (red truth dot within whiskers)
    # -------------------------------------------------------
    item_ids_sorted = sorted(per_item_probs.keys())
    y_values = [per_item_probs[i] for i in item_ids_sorted]
    red_dots = [per_item_truth[i] for i in item_ids_sorted]
    whisker_coverage_pct = plot_item_boxplots(item_ids_sorted, y_values, red_dots)
    print(f"Percentage of ground truths within whisker limits: {whisker_coverage_pct:.2f}%")

    # -------------------------------------------------------
    # 4) Calibration coverage across quantile bands
    # -------------------------------------------------------
    # Accumulate hits per band across items
    total_items = len(per_item_probs)
    band_hits = np.zeros(len(quantile_ranges), dtype=int)

    for iid in item_ids_sorted:
        probs = per_item_probs[iid]
        mean_truth = per_item_truth[iid]
        hits, _ = compute_quantile_intervals(probs, quantile_ranges, mean_truth)
        band_hits += np.asarray(hits, dtype=int)

    # Normalize by item count to get coverage fraction per band
    coverage_fractions = (band_hits / max(total_items, 1)).tolist()

    # x-points like the notebook (0, 0.05, 0.1, ..., 1)
    x_points = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    slope, intercept = plot_calibration(x_points, coverage_fractions)
    print(f"Calibration best-fit slope, intercept: ({slope:.6f}, {intercept:.6f})")

    # Summary dictionary if the caller wants to log/save
    return {
        "auc": auc_score,
        "pearson": pear,
        "spearman": spear,
        "majority_f1": majority_f1,
        "whisker_coverage_pct": whisker_coverage_pct,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "coverage_fractions": coverage_fractions,
    }
