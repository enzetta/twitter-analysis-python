import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Define file path
file_name = "2025-02-19_20-17-10_network_climate_metrics.csv"
output_path = os.path.join("data", "output", file_name)

# Specify metrics to analyze
selected_metrics = [
    "modularity",
    "network_avg_toxicity",
    "transitivity",
    "assortativity",
    "max_core_number",
    "rich_club_coefficient",
    "average_clustering",
    "connected_components",
    "density",
]


def normalize_series(series):
    """Normalize a series using z-score normalization."""
    return (series - series.mean()) / series.std()


def load_and_prepare_data(data, selected_metrics):
    """Load and prepare the network metrics data."""
    df = pd.read_csv(data)
    df["month_start"] = pd.to_datetime(df["month_start"])

    # Ensure all selected metrics exist in the dataframe
    available_metrics = [m for m in selected_metrics if m in df.columns]
    if len(available_metrics) != len(selected_metrics):
        missing = set(selected_metrics) - set(available_metrics)
        print(f"Warning: Some metrics not found in data: {missing}")

    return df[["month_start"] + available_metrics]


def calculate_statistical_measures(df, metrics, normalize=True):
    """Calculate various statistical measures between metrics."""
    results = {}

    for i, metric1 in enumerate(metrics):
        results[metric1] = {}
        for metric2 in metrics[i + 1 :]:  # Avoid self-comparisons and duplicates
            # Prepare data
            x = df[metric1].values
            y = df[metric2].values

            if normalize:
                x = normalize_series(pd.Series(x))
                y = normalize_series(pd.Series(y))

            # Calculate correlations
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            # Calculate R-squared using numpy's polyfit
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

            results[metric1][metric2] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "r_squared": r_squared,
                "slope": z[0],
                "intercept": z[1],
            }

    return results


def calculate_lagged_correlations(df, metrics, max_lag=3, normalize=True):
    """Calculate lagged correlations between selected metrics."""
    lag_correlations = {}

    for metric1 in metrics:
        lag_correlations[metric1] = {}
        for metric2 in metrics:
            if metric1 != metric2:
                lag_correlations[metric1][metric2] = []
                for lag in range(max_lag + 1):
                    if lag == 0:
                        x = df[metric1].values
                        y = df[metric2].values
                    else:
                        x = df[metric1][lag:].values
                        y = df[metric2][:-lag].values

                    if normalize:
                        x = normalize_series(pd.Series(x))
                        y = normalize_series(pd.Series(y))

                    # Calculate R-squared
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    y_pred = p(x)
                    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum(
                        (y - y.mean()) ** 2
                    )
                    corr = stats.spearmanr(x, y)[0]

                    lag_correlations[metric1][metric2].append(
                        {"lag": lag, "correlation": corr, "r_squared": r_squared}
                    )

    return lag_correlations


def plot_correlation_heatmap(df, metrics, title, normalize=True):
    """Plot correlation heatmap for selected metrics."""
    if normalize:
        df_norm = df.copy()
        for metric in metrics:
            df_norm[metric] = normalize_series(df[metric])
        correlations = df_norm[metrics].corr(method="spearman")
    else:
        correlations = df[metrics].corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_metric_trends(df, metrics):
    """Plot trends of selected metrics over time."""
    plt.figure(figsize=(15, 8))
    for metric in metrics:
        # Always normalize for trend comparison
        normalized = normalize_series(df[metric])
        plt.plot(df["month_start"], normalized, label=metric)

    plt.title("Normalized Metric Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_regression_scatter(df, metric1, metric2, stats_results, normalize=True):
    """Plot scatter plot with regression line and statistics."""
    plt.figure(figsize=(10, 6))

    # Prepare data
    x = df[metric1]
    y = df[metric2]

    if normalize:
        x = normalize_series(x)
        y = normalize_series(y)

    # Plot scatter points
    plt.scatter(x, y, alpha=0.5)

    # Plot regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)

    # Add statistics to plot
    stats_text = f"R² = {stats_results['r_squared']:.3f}\n"
    stats_text += f"Pearson r = {stats_results['pearson_r']:.3f} (p = {stats_results['pearson_p']:.3f})\n"
    stats_text += f"Spearman r = {stats_results['spearman_r']:.3f} (p = {stats_results['spearman_p']:.3f})"

    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title(f"{metric1} vs {metric2}")
    plt.xlabel(f"{metric1} {'(normalized)' if normalize else ''}")
    plt.ylabel(f"{metric2} {'(normalized)' if normalize else ''}")
    plt.tight_layout()
    plt.show()


def analyze_network_metrics(data_path, selected_metrics, normalize=True):
    """Main analysis function."""
    # Load and prepare data
    df = load_and_prepare_data(data_path, selected_metrics)

    # Calculate statistical measures
    stats_results = calculate_statistical_measures(
        df, selected_metrics, normalize=normalize
    )

    # Calculate lagged correlations
    lag_correlations = calculate_lagged_correlations(
        df, selected_metrics, normalize=normalize
    )

    # Generate visualizations
    plot_correlation_heatmap(
        df, selected_metrics, "Selected Metrics Correlations", normalize=normalize
    )
    plot_metric_trends(df, selected_metrics)  # Always normalized for comparison

    # Plot regression scatter plots for each pair
    for metric1 in selected_metrics:
        for metric2 in selected_metrics:
            if metric1 < metric2:  # Avoid duplicates
                if metric1 in stats_results and metric2 in stats_results[metric1]:
                    plot_regression_scatter(
                        df,
                        metric1,
                        metric2,
                        stats_results[metric1][metric2],
                        normalize=normalize,
                    )

    return stats_results, lag_correlations


if __name__ == "__main__":
    # Run analysis
    stats_results, lag_correlations = analyze_network_metrics(
        output_path, selected_metrics, normalize=True
    )

    # Print detailed results
    print("\nDetailed Statistical Measures:")
    for metric1 in stats_results:
        for metric2 in stats_results[metric1]:
            print(f"\n{metric1} vs {metric2}:")
            for measure, value in stats_results[metric1][metric2].items():
                print(f"  {measure}: {value:.4f}")

            # Print strongest lagged correlation
            max_lag = max(
                lag_correlations[metric1][metric2], key=lambda x: abs(x["correlation"])
            )
            print(
                f"  Strongest lag: {max_lag['lag']} months "
                f"(correlation: {max_lag['correlation']:.4f}, "
                f"R²: {max_lag['r_squared']:.4f})"
            )
