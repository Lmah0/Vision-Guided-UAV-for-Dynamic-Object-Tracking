"""
Minimal results exporter: CSV tables + matplotlib plots only.
No verbose narrative - just clean data and visualizations.
"""

import csv
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def export_results(all_summaries: Dict) -> Path:
    """
    Export benchmark results as CSV tables and matplotlib plots.
    
    Args:
        all_summaries: Dict of {test_name: summary_dict} from benchmark collector
        
    Returns:
        Path to output directory
    """
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse factorial design results
    # Expected naming: factorial_{model}_{entropy}
    results = {}
    for test_name, summary in all_summaries.items():
        if "factorial_" in test_name and summary:
            parts = test_name.split("_")
            if len(parts) >= 3:
                model = parts[1]
                entropy = parts[2]
                results[(model, entropy)] = summary
    
    if not results:
        print("⚠️  No factorial design results found.")
        return output_dir
    
    # Extract unique models and entropy levels
    models = sorted(set(k[0] for k in results.keys()))
    entropy_levels = sorted(set(k[1] for k in results.keys()))
    
    # ========================================================================
    # CSV EXPORT: Three tables (Mean Latency, P95 Latency, RSS Memory)
    # ========================================================================
    
    _export_csv_table(
        output_dir / "mean_latency_ms.csv",
        models, entropy_levels, results,
        metric_key="latency",
        value_key="mean_ms",
        ci_lower_key="mean_ci_lower",
        ci_upper_key="mean_ci_upper",
        title="Mean Inference Latency (ms)"
    )
    
    _export_csv_table(
        output_dir / "p95_latency_ms.csv",
        models, entropy_levels, results,
        metric_key="latency",
        value_key="p95_ms",
        ci_lower_key="p95_ci_lower",
        ci_upper_key="p95_ci_upper",
        title="P95 Inference Latency (ms)"
    )
    
    _export_csv_table(
        output_dir / "rss_memory_mb.csv",
        models, entropy_levels, results,
        metric_key="memory",
        value_key="rss_mean_mb",
        ci_lower_key="rss_ci_lower",
        ci_upper_key="rss_ci_upper",
        title="RSS Memory (MB)"
    )
    
    # ========================================================================
    # MATPLOTLIB PLOTS: One plot per metric
    # ========================================================================
    
    if MATPLOTLIB_AVAILABLE:
        _plot_metric(
            output_dir / "plot_mean_latency.png",
            models, entropy_levels, results,
            metric_key="latency",
            value_key="mean_ms",
            ci_lower_key="mean_ci_lower",
            ci_upper_key="mean_ci_upper",
            ylabel="Mean Latency (ms)",
            title="Mean Inference Latency by Model & Entropy"
        )
        
        _plot_metric(
            output_dir / "plot_p95_latency.png",
            models, entropy_levels, results,
            metric_key="latency",
            value_key="p95_ms",
            ci_lower_key="p95_ci_lower",
            ci_upper_key="p95_ci_upper",
            ylabel="P95 Latency (ms)",
            title="P95 Inference Latency by Model & Entropy"
        )
        
        _plot_metric(
            output_dir / "plot_rss_memory.png",
            models, entropy_levels, results,
            metric_key="memory",
            value_key="rss_mean_mb",
            ci_lower_key="rss_ci_lower",
            ci_upper_key="rss_ci_upper",
            ylabel="RSS Memory (MB)",
            title="Memory Usage by Model & Entropy"
        )
    
    print(f"✓ Results exported to {output_dir}/")
    print(f"  - mean_latency_ms.csv")
    print(f"  - p95_latency_ms.csv")
    print(f"  - rss_memory_mb.csv")
    if MATPLOTLIB_AVAILABLE:
        print(f"  - plot_mean_latency.png")
        print(f"  - plot_p95_latency.png")
        print(f"  - plot_rss_memory.png")
    
    return output_dir


def _export_csv_table(
    filepath: Path,
    models: List[str],
    entropy_levels: List[str],
    results: Dict,
    metric_key: str,
    value_key: str,
    ci_lower_key: str,
    ci_upper_key: str,
    title: str
) -> None:
    """Export a single metric as CSV with CI bounds."""
    
    rows = []
    
    # Header
    rows.append(["Model"] + entropy_levels)
    
    # Data rows
    for model in models:
        row = [model]
        for entropy in entropy_levels:
            key = (model, entropy)
            if key in results:
                summary = results[key]
                if metric_key in summary:
                    metric = summary[metric_key]
                    val = metric.get(value_key, None)
                    ci_low = metric.get(ci_lower_key, None)
                    ci_high = metric.get(ci_upper_key, None)
                    
                    if val is not None and ci_low is not None and ci_high is not None:
                        # Format: value [ci_low - ci_high]
                        cell = f"{val:.2f} [{ci_low:.2f}-{ci_high:.2f}]"
                    else:
                        cell = "N/A"
                else:
                    cell = "N/A"
            else:
                cell = "—"
            row.append(cell)
        rows.append(row)
    
    # Write CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"  ✓ {filepath.name}")


def _plot_metric(
    filepath: Path,
    models: List[str],
    entropy_levels: List[str],
    results: Dict,
    metric_key: str,
    value_key: str,
    ci_lower_key: str,
    ci_upper_key: str,
    ylabel: str,
    title: str
) -> None:
    """Create a grouped bar plot for a metric across all combinations."""
    
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, entropy in enumerate(entropy_levels):
        means = []
        errors = []
        
        for model in models:
            key = (model, entropy)
            if key in results:
                summary = results[key]
                if metric_key in summary:
                    metric = summary[metric_key]
                    val = metric.get(value_key, np.nan)
                    ci_low = metric.get(ci_lower_key, val)
                    ci_high = metric.get(ci_upper_key, val)
                    means.append(val)
                    errors.append(val - ci_low)  # Error bar size
                else:
                    means.append(np.nan)
                    errors.append(0)
            else:
                means.append(np.nan)
                errors.append(0)
        
        offset = width * (i - len(entropy_levels)/2 + 0.5)
        ax.bar(x + offset, means, width, label=entropy, yerr=errors, capsize=5)
    
    ax.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title="Scene Entropy", loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    print(f"  ✓ {filepath.name}")
