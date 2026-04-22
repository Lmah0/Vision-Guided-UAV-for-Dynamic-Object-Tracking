#!/usr/bin/env python3
"""
Performance benchmark execution and report generation script.

Usage:
    python run_benchmark_tests.py

This script:
1. Runs all performance benchmark tests with instrumentation
2. Collects results with 95% confidence intervals
3. Generates publication-quality visualizations
4. Produces comprehensive markdown report
"""

import subprocess
import sys
from pathlib import Path
import json
import time


def run_tests():
    """Execute pytest suite for performance benchmarks."""
    print("\n" + "=" * 80)
    print("RUNNING PERFORMANCE BENCHMARK TESTS")
    print("=" * 80 + "\n")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/detection/test_cv_detection.py",
        "-v", "--tb=short",
        "-m", "performance"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def generate_visualizations():
    """Generate performance plots from collected results."""
    print("\n" + "=" * 80)
    print("GENERATING PERFORMANCE VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    try:
        from tests.performance_report import generate_experimental_report
        report_path = generate_experimental_report()
        
        if report_path:
            print(f"\n✓ Report generated: {report_path}")
            return True
        else:
            print("Warning: Could not generate visualizations")
            return False
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PERFORMANCE BENCHMARK SUITE" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    start_time = time.time()
    
    # Run tests
    print("📊 Stage 1: Running benchmark tests...")
    if not run_tests():
        print("✗ Test suite failed. Aborting report generation.")
        return 1
    
    print("\n✓ Test suite completed successfully")
    
    # Generate report and visualizations
    print("\n📊 Stage 2: Generating report and visualizations...")
    if not generate_visualizations():
        print("⚠ Warning: Some visualizations failed, but report may still be available")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print("=" * 80)
    print(f"✓ Elapsed time: {elapsed:.1f} seconds")
    print("\n📁 Output Artifacts:")
    print("   - benchmark_results/results.json       : Raw benchmark data")
    print("   - benchmark_results/summary.txt        : Statistical summaries")
    print("   - benchmark_results/REPORT.md          : Final markdown report")
    print("   - benchmark_plots/                     : Publication-quality plots")
    print("       ├── model_comparison.png")
    print("       ├── resolution_scaling.png")
    print("       ├── entropy_impact.png")
    print("       └── factor_impact.png")
    print("\n💡 Next steps:")
    print("   1. Review benchmark_results/REPORT.md for findings")
    print("   2. Check benchmark_plots/ for visualizations")
    print("   3. Use results for final proposal document")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
