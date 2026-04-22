"""
Statistical analysis utilities for performance benchmark results.

Provides:
  - Confidence interval calculations (95% CI using t-distribution and bootstrap)
  - Summary statistics with CI bounds
  - Comparison tests between groups
  - Result aggregation and serialization
"""

import statistics
from typing import List, Dict, Tuple, Optional
import numpy as np
https://file+.vscode-resource.vscode-cdn.net/Users/lionelhasan/Capstone/Capstone-LOCK-2A/benchmark_results/plot_mean_latency.png?version%3D1776890267497

class PerformanceStats:
    """
    Calculate statistics with confidence intervals for latency and resource metrics.
    Uses scipy.stats.t for small sample CI calculation (robust for n < 30).
    """

    @staticmethod
    def calculate_ci(
        values: List[float],
        confidence: float = 0.95,
        method: str = "t"
    ) -> Dict[str, float]:
        """
        Calculate mean and confidence interval for a list of values.
        
        Args:
            values: Latency or resource measurements
            confidence: CI level (default 0.95 = 95%)
            method: "t" (t-distribution, better for small n) or "bootstrap"
            
        Returns:
            {
                "mean": float,
                "std": float,
                "ci_lower": float,
                "ci_upper": float,
                "margin_of_error": float,
                "n": int
            }
        """
        if len(values) < 2:
            raise ValueError("Need at least 2 values for CI calculation")
        
        values = np.array(values)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # sample std
        
        if method == "t":
            # t-distribution CI (more accurate for small samples)
            from scipy import stats
            alpha = 1 - confidence
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            margin = t_crit * (std / np.sqrt(n))
        elif method == "bootstrap":
            # Bootstrap CI
            rng = np.random.default_rng(seed=42)
            bootstrap_means = []
            for _ in range(10000):
                sample = rng.choice(values, size=n, replace=True)
                bootstrap_means.append(np.mean(sample))
            bootstrap_means = np.array(bootstrap_means)
            alpha = 1 - confidence
            ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
            margin = (ci_upper - ci_lower) / 2
            return {
                "mean": mean,
                "std": std,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "margin_of_error": margin,
                "n": n
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "mean": mean,
            "std": std,
            "ci_lower": float(mean - margin),
            "ci_upper": float(mean + margin),
            "margin_of_error": float(margin),
            "n": n
        }

    @staticmethod
    def percentile_ci(
        values: List[float],
        percentile: float = 95,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate percentile (e.g., P95) with confidence interval using bootstrap.
        """
        if len(values) < 2:
            raise ValueError("Need at least 2 values for CI calculation")
        
        values = np.array(values)
        n = len(values)
        p_value = float(np.percentile(values, percentile))
        
        # Bootstrap CI for percentile
        rng = np.random.default_rng(seed=42)
        bootstrap_percentiles = []
        for _ in range(10000):
            sample = rng.choice(values, size=n, replace=True)
            bootstrap_percentiles.append(np.percentile(sample, percentile))
        bootstrap_percentiles = np.array(bootstrap_percentiles)
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_percentiles, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_percentiles, (1 - alpha/2) * 100)
        
        return {
            "percentile": percentile,
            "value": p_value,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n": n
        }

    @staticmethod
    def ttest_independent(
        group1: List[float],
        group2: List[float]
    ) -> Dict[str, float]:
        """
        Welch's t-test (doesn't assume equal variances).
        Returns: t-statistic, p-value, significant (p < 0.05)
        """
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n1": len(group1),
            "n2": len(group2)
        }

    @staticmethod
    def effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
        """Cohen's d: standardized mean difference."""
        g1 = np.array(group1)
        g2 = np.array(group2)
        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        return float((np.mean(g1) - np.mean(g2)) / pooled_std)


class BenchmarkResult:
    """
    Aggregates a single benchmark result (one model/resolution/entropy combination).
    """
    
    def __init__(
        self,
        name: str,
        latencies_ms: List[float],
        rss_mb: Optional[List[float]] = None,
        cpu_pct: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            name: Test identifier (e.g., "yolo26n_720p_high-entropy")
            latencies_ms: Inference latencies for each frame
            rss_mb: RSS memory readings (optional)
            cpu_pct: CPU usage % readings (optional)
            metadata: Arbitrary metadata dict
        """
        self.name = name
        self.latencies_ms = list(latencies_ms)
        self.rss_mb = list(rss_mb) if rss_mb else None
        self.cpu_pct = list(cpu_pct) if cpu_pct else None
        self.metadata = metadata or {}
    
    def summary(self) -> Dict:
        """Return full summary with CIs."""
        stats = PerformanceStats()
        result = {
            "name": self.name,
            "metadata": self.metadata,
        }
        
        # Latency stats
        latency_ci = stats.calculate_ci(self.latencies_ms, method="t")
        latency_p95 = stats.percentile_ci(self.latencies_ms, percentile=95)
        result["latency"] = {
            "mean_ms": latency_ci["mean"],
            "mean_ci_lower": latency_ci["ci_lower"],
            "mean_ci_upper": latency_ci["ci_upper"],
            "mean_std": latency_ci["std"],
            "p95_ms": latency_p95["value"],
            "p95_ci_lower": latency_p95["ci_lower"],
            "p95_ci_upper": latency_p95["ci_upper"],
            "n_samples": latency_ci["n"]
        }
        
        # Memory stats
        if self.rss_mb:
            rss_ci = stats.calculate_ci(self.rss_mb, method="t")
            result["memory"] = {
                "rss_mean_mb": rss_ci["mean"],
                "rss_ci_lower": rss_ci["ci_lower"],
                "rss_ci_upper": rss_ci["ci_upper"],
                "rss_std": rss_ci["std"],
                "n_samples": rss_ci["n"]
            }
        
        # CPU stats
        if self.cpu_pct:
            cpu_ci = stats.calculate_ci(self.cpu_pct, method="t")
            result["cpu"] = {
                "mean_pct": cpu_ci["mean"],
                "ci_lower": cpu_ci["ci_lower"],
                "ci_upper": cpu_ci["ci_upper"],
                "std": cpu_ci["std"],
                "n_samples": cpu_ci["n"]
            }
        
        return result


class BenchmarkComparison:
    """
    Compares two BenchmarkResult objects with statistical tests.
    """
    
    def __init__(self, baseline: BenchmarkResult, test: BenchmarkResult):
        self.baseline = baseline
        self.test = test
        self.stats = PerformanceStats()
    
    def compare_latency(self) -> Dict:
        """Welch's t-test + Cohen's d for latency difference."""
        ttest = self.stats.ttest_independent(
            self.baseline.latencies_ms,
            self.test.latencies_ms
        )
        cohens_d = self.stats.effect_size_cohens_d(
            self.baseline.latencies_ms,
            self.test.latencies_ms
        )
        
        mean_diff = np.mean(self.test.latencies_ms) - np.mean(self.baseline.latencies_ms)
        pct_diff = (mean_diff / np.mean(self.baseline.latencies_ms)) * 100
        
        return {
            "baseline_mean": float(np.mean(self.baseline.latencies_ms)),
            "test_mean": float(np.mean(self.test.latencies_ms)),
            "mean_diff_ms": float(mean_diff),
            "pct_diff": float(pct_diff),
            "t_statistic": ttest["t_statistic"],
            "p_value": ttest["p_value"],
            "significant": ttest["significant"],
            "cohens_d": float(cohens_d)
        }


# Example usage in fixtures / test utilities
def collect_latencies(frames: List, model, conf: float, iou: float) -> List[float]:
    """Helper to collect latencies from a batch of frames."""
    import time
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, iou=iou, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies
