# tests/unit/detection/test_cv_detection.py
"""
Detection Performance Evaluation - Factorial Design
=====================================================

Following the project methodology (Section 5), this test suite evaluates the 
YOLO-based detection module using a Full Factorial Design:

Factors:
  - Model Complexity: YOLOv5n, YOLOv8n, YOLOv11n, YOLOv26n, YOLOv26s
  - Scene Entropy: Low (sparse), Medium, High (dense), Very High

Metrics Collected:
  - Inference Latency (Mean): Average per-frame processing time
  - Tail Latency (P95): 95th percentile latency
  - Resource Consumption (RSS): Peak memory usage in MB

Results include 95% confidence intervals on all metrics.
"""

import sys
import time
from pathlib import Path
import io
import contextlib

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import ResourceMonitor
from tests.performance_stats import PerformanceStats

MODELS_DIR = ROOT / "backend" / "gcs" / "ai" / "models"

# YOLO models to test: Mix of YOLOv8, YOLOv11, and YOLOv26 variants
# Pre-downloaded and cached locally in backend/gcs/ai/models/
MODELS = [
    "yolo26n",      # YOLOv26 nano
    "yolo26s",      # YOLOv26 small
    "yolo11n",      # YOLOv11 nano
    "yolov8n",      # YOLOv8 nano
    "yolov8s",      # YOLOv8 small
]
ENTROPY_LEVELS = ["low", "medium", "high", "very_high"]

pytestmark = pytest.mark.performance


# ============================================================================
# HELPERS
# ============================================================================

def _load_model(model_name: str):
    """Load YOLO model from local cache or auto-download."""
    from ultralytics import YOLO
    
    # Try loading from local models directory first
    local_paths = [
        MODELS_DIR / f"{model_name}.pt",
        ROOT / "backend" / "gcs" / "ai" / "utils" / f"{model_name}.pt",
        ROOT / "backend" / "gcs" / "ai" / "models" / f"{model_name}.pt",
    ]
    
    for path in local_paths:
        if path.exists():
            try:
                model = YOLO(str(path))
                return model
            except Exception as e:
                continue
    
    # Fall back to auto-download from Ultralytics
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")


def _get_entropy_video(entropy_level: str):
    """Get video frames for entropy level: real videos or synthetic."""
    from tests.conftest import generate_synthetic_frame
    
    # First try real videos (if available)
    real_videos = {
        "low": ROOT / "backend" / "gcs" / "ai" / "error-video.mp4",      # Single human
        "high": ROOT / "backend" / "gcs" / "ai" / "video.mp4",           # Many humans
    }
    
    path = real_videos.get(entropy_level)
    if path and path.exists():
        # Use real video for low/high
        return _load_video_frames(path, 60)
    
    # Generate synthetic frames for all entropy levels (for consistency/reproducibility)
    # This ensures medium/very_high also have consistent, repeatable complexity
    width, height = 1280, 720
    frames = []
    for i in range(60):
        # Use incremented seed to get different frames but deterministic results
        frame = generate_synthetic_frame(width, height, entropy_level, seed=42 + i)
        frames.append(frame)
    
    return frames


def _load_video_frames(video_path: Path, n_frames: int = 60) -> list:
    """Load frames from a video file."""
    if not video_path.exists():
        pytest.skip(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pytest.skip(f"Could not open video: {video_path}")
    
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        pytest.skip(f"No frames read from {video_path}")
    
    return frames


# ============================================================================
# FACTORIAL DESIGN TEST: MODEL COMPLEXITY × SCENE ENTROPY
# ============================================================================

class TestFactorialDesign:
    """
    Full Factorial Design: Tests all combinations of model complexity and scene entropy.
    
    Per Section 5.5, this design allows analysis of interaction effects between
    model size and environmental complexity.
    """
    
    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("entropy_level", ENTROPY_LEVELS)
    def test_model_entropy_combination(
        self, model_name, entropy_level, benchmark_results
    ):
        """
        Test a specific model × entropy combination.
        
        Following Section 5.7:
        - Warm-up: 100 cold inference passes
        - Execution: Process pre-defined workload
        - Data Collection: Per-frame latency and RSS
        - Analysis: Aggregated mean/P95 statistics
        """
        from backend.gcs.ai.AIEngine import TrackingConfig
        
        # Load model (auto-downloads if needed)
        model = _load_model(model_name)
        
        # Get frames for entropy level
        frames = _get_entropy_video(entropy_level)
        
        # Inference parameters (from TrackingConfig)
        conf_threshold = TrackingConfig.CONFIDENCE_THRESHOLD
        iou_threshold = TrackingConfig.MODEL_IOU
        
        # ================================================================
        # WARM-UP PHASE (Per Section 5.7)
        # ================================================================
        # 100 cold inference passes to prime caches and reach steady state
        with contextlib.redirect_stderr(io.StringIO()):
            for i in range(min(100, len(frames))):
                model.predict(
                    frames[i % len(frames)],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
        
        # ================================================================
        # EXECUTION & DATA COLLECTION PHASE (Per Section 5.7)
        # ================================================================
        monitor = ResourceMonitor(sample_every_n_frames=5)
        latencies_ms = []
        rss_samples = []
        
        monitor.start()
        with contextlib.redirect_stderr(io.StringIO()):
            for i, frame in enumerate(frames):
                t0 = time.perf_counter()
                model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                rss_samples.append(monitor._process.memory_info().rss / (1024 ** 2))
                monitor.sample(i)
        
        s = monitor.summary()
        
        # ================================================================
        # ANALYSIS PHASE (Per Section 5.7)
        # ================================================================
        stats = PerformanceStats()
        
        # Calculate confidence intervals
        latency_ci = stats.calculate_ci(latencies_ms, method="t")
        p95_ci = stats.percentile_ci(latencies_ms, percentile=95)
        rss_ci = stats.calculate_ci(rss_samples, method="t")
        
        # Record results
        test_name = f"factorial_{model_name.replace('.pt', '')}_{entropy_level}"
        benchmark_results.record(
            test_name=test_name,
            latencies_ms=latencies_ms,
            rss_mb=rss_samples,
            metadata={
                "model": model_name,
                "entropy": entropy_level,
                "model_version": model_name[4],  # Extract version: 'v' in yolov8n → '8'
                "model_size": model_name[-1],     # Extract size: last char 'n' or 's'
            }
        )
        
        # Console output for debugging
        print(f"\n{model_name:15} × {entropy_level:12}")
        print(f"  Mean Latency:  {latency_ci['mean']:7.2f} ms (95% CI: [{latency_ci['ci_lower']:7.2f}, {latency_ci['ci_upper']:7.2f}])")
        print(f"  P95 Latency:   {p95_ci['value']:7.2f} ms (95% CI: [{p95_ci['ci_lower']:7.2f}, {p95_ci['ci_upper']:7.2f}])")
        print(f"  RSS Memory:    {rss_ci['mean']:7.1f} MB (95% CI: [{rss_ci['ci_lower']:7.1f}, {rss_ci['ci_upper']:7.1f}])")
        
        # Basic sanity check
        assert latency_ci['mean'] < 1000, f"Latency unusually high: {latency_ci['mean']:.1f}ms"

