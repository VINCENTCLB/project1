import matplotlib.pyplot as plt
import numpy as np

def plot_feature_distribution(feature_values, feature_name):
    """
    Plots the distribution of a single feature.

    Args:
        feature_values (np.array): Values of the feature
        feature_name (str): Name of the feature
    """
    plt.figure(figsize=(10, 6))
    plt.hist(feature_values, bins=30, alpha=0.7)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature_name}')
    plt.show()
    

def plot_statistics(
mean, std, min_val, max_val, median_val, feature_names, log_scale=False):
    """Plots summary statistics as clean range intervals (Min-Max, Mean ± Std,

    Median).
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)
    median_val = np.asarray(median_val)

    num_features = len(feature_names)
    y_indices = np.arange(num_features)

    fig, ax = plt.subplots(figsize=(10, max(5, num_features * 0.6)))

    # 1. Intervalle total [Min, Max] (Ligne fine)
    ax.hlines(
        y_indices,
        min_val,
        max_val,
        color="gray",
        alpha=0.6,
        linewidth=2,
        label="Range [Min, Max]",
    )

    # 2. Intervalle [Mean - Std, Mean + Std] (Ligne epaisse)
    ax.hlines(
        y_indices,
        mean - std,
        mean + std,
        color="#2b5c8f",
        linewidth=6,
        alpha=0.7,
        label="Mean ± Std",
    )

    # 3. Points pour la Moyenne et la Médiane
    ax.plot(
        mean,
        y_indices,
        "o",
        color="#2b5c8f",
        markersize=7,
        label="Mean",
        zorder=5,
    )
    ax.plot(
        median_val,
        y_indices,
        "d",
        color="#d73027",
        markersize=7,
        label="Median",
        zorder=5,
    )

    ax.set_yticks(y_indices)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Value")
    ax.set_title("Feature Summary Statistics (Range Plot)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    if log_scale:
        ax.set_xscale("symlog")

    plt.tight_layout()
    plt.show()
    
def plot_correlation_matrix(corr_matrix, feature_names):
    """
    Plots the correlation matrix as a heatmap.

    Args:
        corr_matrix (np.array): Correlation matrix of shape (d, d)
        feature_names (list): List of feature names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Correlation Matrix')
    plt.show()
    
def plot_target_distribution(distribution):
    """
    Plots the distribution of the target variable.

    Args:
        distribution (dict): Dictionary with keys as unique values and values as their counts.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.xlabel('Target Values')
    plt.ylabel('Counts')
    plt.title('Target Variable Distribution')
    plt.show()
    
def plot_loss_curve(losses):
    """
    Plots the loss curve over iterations.

    Args:
        losses (list): List of loss values over iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    