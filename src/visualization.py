import numpy as np
import matplotlib.pyplot as plt

def plot_item_boxplots(item_ids, per_item_probs, true_means):
    """
    Draw boxplots of per-item predicted probabilities with red dots at true mean label per item.
    Args:
        item_ids: list of item ids (sorted order)
        per_item_probs: list of lists; per_item_probs[i] corresponds to probabilities for item_ids[i]
        true_means: list of floats; true_means[i] is mean label for item_ids[i]
    Returns:
        coverage_percentage_within_whiskers: float
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    boxplot = ax.boxplot(per_item_probs, showfliers=True)

    ax.set_xticks(range(1, len(item_ids) + 1))
    ax.set_xticklabels(item_ids, rotation=90)
    ax.set_xlabel('Text Item by IDs')
    ax.set_ylabel('Probability')

    # Red dots at ground-truth mean label proportions
    for i, m in enumerate(true_means):
        ax.plot(i + 1, m, 'ro')

    plt.tight_layout()
    plt.show()

    # Compute whisker coverage of red dots
    whiskers = boxplot['whiskers']
    whisker_limits = []
    for i in range(0, len(whiskers), 2):
        lower_whisker = whiskers[i].get_ydata()[1]
        upper_whisker = whiskers[i + 1].get_ydata()[1]
        whisker_limits.append((lower_whisker, upper_whisker))

    inside_whiskers_count = 0
    for i, m in enumerate(true_means):
        low, up = whisker_limits[i]
        if low <= m <= up:
            inside_whiskers_count += 1

    total = len(true_means)
    coverage_pct = (inside_whiskers_count / total) * 100.0 if total > 0 else 0.0
    return coverage_pct


def plot_calibration(quantile_points, coverage_fractions):
    """
    Plot calibration curve: empirical coverage (y) vs quantile-band width about median (x).
    Args:
        quantile_points: list of x positions (e.g., [0, 0.05, ... 1])
        coverage_fractions: list of y coverage (0..1), same length as quantile_points
    Returns:
        slope, intercept from linear fit
    """
    x = np.asarray(quantile_points, dtype=float)
    y = np.asarray(coverage_fractions, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data Points')

    # Best-fit line
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    plt.plot(x, poly(x), label='Model Calibration fit')

    # Ideal x=y line
    plt.plot(x, x, linestyle='--', label='Ideal Calibration fit')

    plt.xlabel('Quantiles about median')
    plt.ylabel('Coverage')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(False)
    plt.show()

    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return slope, intercept
