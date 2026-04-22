"""
Performance Benchmarking & Statistical Analysis Infrastructure

This module provides comprehensive performance testing capabilities for the CV detection pipeline with:
- Inference latency measurement (mean + P95 percentile)
- Resource consumption tracking (CPU%, memory RSS)
- 95% confidence interval calculations
- Publication-quality visualizations with matplotlib
- Comprehensive markdown reports

================================================================================
QUICK START
================================================================================

## Generate Full Report (Recommended)

    python run_benchmark_tests.py

This runs all tests and generates:
- JSON results (benchmark_results/results.json)
- Statistical summaries (benchmark_results/summary.txt)
- Markdown report (benchmark_results/REPORT.md)
- Publication plots (benchmark_plots/*.png)

## Run Tests Only

    pytest tests/unit/detection/test_cv_detection.py -v -m performance

## Run Specific Test Group

    # Model comparison tests
    pytest tests/unit/detection/test_cv_detection.py::TestResourceByModel -v
    
    # Scene entropy tests
    pytest tests/unit/detection/test_cv_detection.py::TestResourceByEntropy -v
    
    # Resolution scaling tests
    pytest tests/unit/detection/test_cv_detection.py::TestDetectionLatencyScreening -v
    
    # Pipeline stage breakdown
    pytest tests/unit/detection/test_cv_detection.py::TestDetectionStageBreakdown -v

================================================================================
CONFIDENCE INTERVALS
================================================================================

All reported metrics include 95% confidence intervals calculated using:

### For Means (Inference Latency, Memory, CPU Usage)
- **Method:** Welch's t-distribution
- **Formula:** mean ± t* × (s / √n)
- **Why:** More robust for small samples (n < 30) and doesn't assume equal variance
- **Example:** "42.0 ms (95% CI: [41.2, 42.8])" means we're 95% confident the true 
  mean latency falls between 41.2-42.8 ms

### For Percentiles (P95 Latency)
- **Method:** Bootstrap with 10,000 resamples
- **Why:** Handles non-normal distributions without assumptions
- **Example:** "P95: 47.1 ms (95% CI: [46.3, 48.1])" means we're 95% confident the 
  95th percentile falls in that range

### Interpreting CI Width
- **Narrow CI (< 1-2 ms):** More reliable estimate, consistent measurements
- **Wide CI (> 5 ms):** More variability in measurements, consider more samples
- **CI includes 0 difference:** Factors may not be significantly different

================================================================================
MODULE STRUCTURE
================================================================================

tests/
├── conftest.py                          # Fixtures, ResourceMonitor, result collection
├── performance_stats.py                 # Statistical calculations & CI generation
├── performance_viz.py                   # matplotlib visualizations
├── performance_report.py                # Markdown report generation
├── unit/detection/
│   └── test_cv_detection.py            # Performance test suite
└── benchmark_results/                   # Generated results (auto-created)
    ├── results.json                     # Raw latency/memory measurements
    ├── summary.txt                      # Statistical summaries
    └── REPORT.md                        # Final markdown report

================================================================================
TEST STRUCTURE
================================================================================

### Test Groups

1. **Model Complexity (TestResourceByModel)**
   - Compares YOLOv11n, YOLOv26n, YOLOv26s
   - Fixed 720p, complex-entropy synthetic workload
   - Measures: latency (mean + P95), memory (RSS)
   - Expected result: Model size significantly impacts latency

2. **Scene Entropy (TestResourceByEntropy)**
   - Low entropy: error-video.mp4 (single human, sparse)
   - High entropy: video.mp4 (many humans, dense)
   - Fixed YOLOv26n model, 720p
   - Measures: latency (mean + P95), memory (RSS)
   - Expected result: Minimal latency impact, significant memory overhead

3. **Resolution Scaling (TestDetectionLatencyScreening)**
   - Tests 480p, 720p, 1080p
   - Fixed YOLOv26n model, complex-entropy synthetic
   - Measures: latency (mean + P95)
   - Expected result: Negligible impact (all within 1.3ms)

4. **Stage Breakdown (TestDetectionStageBreakdown)**
   - Decomposes: inference, NMS boxes, drawing
   - Tests each resolution with real video
   - Measures: per-stage latency
   - Expected result: Identifies pipeline bottleneck

### Metrics

All metrics collected per-frame with resource monitoring every 5 frames:

- **Inference Latency (ms)**
  - Per-frame model.predict() execution time
  - Mean: average across all frames
  - P95: 95th percentile (tail latency)
  - 95% CI: confidence interval bounds

- **RSS Memory (MB)**
  - Process resident set size
  - Sampled from psutil.Process().memory_info()
  - Compare low vs high entropy scenes
  - Note: May see memory growth as Python GC runs

- **CPU Utilization (%)**
  - Process CPU percentage from psutil
  - Averaged across sampling interval
  - Collected for resource analysis

================================================================================
RESULT COLLECTION & AGGREGATION
================================================================================

### Automatic Collection (via pytest fixture)

Tests use the `benchmark_results` fixture to record data:

    def test_something(self, benchmark_results):
        latencies = [...]  # Collected measurements
        
        benchmark_results.record(
            test_name="my_test",
            latencies_ms=latencies,
            rss_mb=memory_samples,
            metadata={"model": "yolo26n", "resolution": "720p"}
        )

### Result Storage

Results are automatically saved after test completion:

    benchmark_results/
    ├── results.json           # Raw phase-structured format
    ├── summary.txt            # Human-readable summaries
    └── summaries.json         # Structured summaries

### Accessing Results in Python

    from tests.conftest import _benchmark_collector
    
    # Get all summaries
    summaries = _benchmark_collector.get_all_summaries()
    
    # Access specific test result
    result = summaries["model_comparison_yolo26n_720p_complex"]
    print(result["latency"]["mean_ms"])
    print(result["latency"]["ci_lower"])  # 95% CI lower bound
    print(result["latency"]["ci_upper"])  # 95% CI upper bound

================================================================================
VISUALIZATION EXAMPLES
================================================================================

Generated plots include:

### model_comparison.png
Bar chart with error bars showing latency across model variants

### resolution_scaling.png  
Line plot with CI band showing negligible resolution impact

### entropy_impact.png
Grouped bar chart comparing latency vs memory impact of entropy

### factor_impact.png
Summary chart highlighting high-priority vs low-priority factors

All plots include:
- 95% confidence interval error bars/bands
- Value labels on bars/points
- Statistical annotations
- Publication-quality styling (300 DPI)

================================================================================
STATISTICAL VALIDATION
================================================================================

### Sample Size
- Model comparison: 60 frames per model
- Entropy tests: 60 frames per entropy level
- Resolution tests: 30 frames per resolution

### Reproducibility
- Synthetic workloads use fixed seeds (seed=42)
- Real videos extracted deterministically
- ResourceMonitor initialization consistent

### Interpretation Checklist
- ✓ CI width indicates measurement precision
- ✓ No-overlap CIs indicate significant difference
- ✓ Large sample sizes improve CI reliability
- ✓ Wide CIs suggest high variability → recommend more samples

================================================================================
TROUBLESHOOTING
================================================================================

### Issue: "Model not found"
Error: pytest.skip(f"Model not found: {path}")
Solution: Ensure YOLO model files exist in backend/gcs/ai/models/

### Issue: "Test video not found"
Error: No frames could be read from test video
Solution: Place video.mp4 and error-video.mp4 in backend/gcs/ai/

### Issue: Wide confidence intervals (> 5ms latency)
Cause: High variability in measurements (background processes, thermal throttling)
Solution: Ensure no background apps running, run during cool period, increase sample size

### Issue: "ResourceMonitor has no attribute..."
Cause: Fixture not properly initialized
Solution: Ensure benchmark_results fixture is included in test parameters

### Issue: Plots not generated
Cause: Missing matplotlib or scipy dependencies
Solution: pip install -U requirements.txt

================================================================================
ADVANCED USAGE
================================================================================

### Custom Statistical Tests

    from tests.performance_stats import PerformanceStats, BenchmarkComparison
    
    # Welch's t-test
    stats = PerformanceStats()
    ttest_result = stats.ttest_independent(group1, group2)
    
    # Effect size (Cohen's d)
    effect_size = stats.effect_size_cohens_d(group1, group2)
    
    # Compare two benchmarks
    baseline = BenchmarkResult("baseline", latencies_baseline)
    test = BenchmarkResult("test", latencies_test)
    comparison = BenchmarkComparison(baseline, test)
    diff = comparison.compare_latency()

### Custom Visualizations

    from tests.performance_viz import BenchmarkVisualizer
    
    viz = BenchmarkVisualizer("my_plots")
    
    # Plot custom comparison
    viz.plot_model_comparison(custom_results, metric="latency")
    
    # Plot distributions as violin plot
    viz.plot_distribution_violin(
        {"group1": [1, 2, 3], "group2": [4, 5, 6]},
        title="Custom Distribution"
    )

### Custom Reports

    from tests.performance_report import PerformanceReportGenerator
    
    generator = PerformanceReportGenerator()
    generator.generate_report(
        title="My Custom Report",
        test_summary={...},
        factor_analysis={...},
        findings=[...],
        recommendations=[...]
    )

================================================================================
DEPENDENCIES
================================================================================

Required packages:
- pytest, pytest-asyncio, pytest-cov
- numpy, scipy
- matplotlib
- ultralytics (YOLO models)
- opencv-python
- psutil (resource monitoring)

Install all:
    pip install -U requirements.txt

Or install testing only:
    pip install pytest scipy matplotlib

================================================================================
FINAL REPORT WORKFLOW
================================================================================

1. Run benchmarks:
   python run_benchmark_tests.py

2. Review markdown report:
   cat benchmark_results/REPORT.md

3. Examine plots:
   open benchmark_plots/

4. Extract key statistics for proposal:
   - Use CI values from REPORT.md
   - Reference specific plot filenames
   - Quote key findings

5. Archive results:
   mkdir -p results/$(date +%Y%m%d_%H%M%S)
   cp -r benchmark_results benchmark_plots results/
"""

# This docstring serves as the module documentation.
# For quick reference, also see RUN_TESTS_README.md in the project root.
