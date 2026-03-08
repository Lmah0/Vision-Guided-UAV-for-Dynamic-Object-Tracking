# tests/unit/detection/test_cv_detection.py
"""
Experiment 1: Detection Performance
====================================
Performance evaluation of YOLO detection models across models, scene entropy, and video resolutions.

Test Groups (tab-separated output for easy table generation):
  1. Model Comparison      — 2 models (yolo11n, yolo26n) at 720p complex-entropy
  2. Scene Entropy Impact  — 2 entropy levels (low/high) with yolo26n at 720p
  3. Resolution Scaling    — 3 resolutions (480p/720p/1080p) with yolo26n at complex-entropy
  4. Stage Breakdown       — Inference/boxes/drawing cost per resolution with yolo26n

Metrics: Inference latency (mean, p95), RSS memory (MB), CPU utilization (%)
Video Sources:
  - real video: video.mp4 (high entropy, many humans), error-video.mp4 (low entropy, single human)
  - synthetic: procedural frames (uniform, controllable)
Total Tests: 2 + 2 + 3 + 3 = 10
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

MODELS_DIR = ROOT / "backend" / "gcs" / "ai" / "models"
AVAILABLE_MODELS = ["yolo11n.pt", "yolo26n.pt", "yolo26s.pt"]

pytestmark = pytest.mark.performance


# ============================================================================
# HELPERS
# ============================================================================

def _model_path(filename: str) -> str:
    path = MODELS_DIR / filename
    if not path.exists():
        pytest.skip(f"Model not found: {path}")
    return str(path)


def _run_inference_batch(model, frames: list, conf: float, iou: float) -> list:
    """Run model.predict on each frame; return list of latency_ms values."""
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, iou=iou, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


# ============================================================================
# MODEL COMPARISON (synthetic 720p, controlled)
# ============================================================================
# CSV Format: Model | Mean Latency | P95 Latency | RSS Mean

class TestResourceByModel:
    """
    Head-to-head resource comparison across nano model variants (yolo11n, yolo26n)
    at fixed 720p complex-entropy workload. Isolates model architecture cost.
    """

    @pytest.mark.parametrize("model_filename", AVAILABLE_MODELS)
    def test_resource_and_latency_by_model(self, model_filename, synthetic_video_factory):
        from ultralytics import YOLO
        from backend.gcs.ai.AIEngine import TrackingConfig

        model   = YOLO(_model_path(model_filename))
        frames  = synthetic_video_factory("720p", "complex", n_frames=60)
        monitor = ResourceMonitor(sample_every_n_frames=5)

        model.predict(frames[0], conf=TrackingConfig.CONFIDENCE_THRESHOLD,
                      iou=TrackingConfig.MODEL_IOU, verbose=False)  # warm-up

        latencies_ms = []
        monitor.start()
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            model.predict(frame, conf=TrackingConfig.CONFIDENCE_THRESHOLD,
                          iou=TrackingConfig.MODEL_IOU, verbose=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            monitor.sample(i)

        s       = monitor.summary()
        mean_ms = float(np.mean(latencies_ms))
        p95_ms  = float(np.percentile(latencies_ms, 95))

        print(f"\n{model_filename}\t{mean_ms:.1f}\t{p95_ms:.1f}\t{s['rss_mean_mb']:.1f}")
        assert mean_ms < 500
        assert s["rss_delta_mb"] < 200


# ============================================================================
# RESOURCE CONSUMPTION BY SCENE ENTROPY
# ============================================================================

class TestResourceByEntropy:
    """
    Isolates the effect of real scene entropy on resource cost using yolo26n.
    low  — error-video.mp4: single human, sparse detections
    high — video.mp4:       many humans, dense detections
    """
    
    _header_printed = False

    def test_resource_by_entropy(self, real_frames_by_entropy):
        from ultralytics import YOLO

        # Print header once
        if not TestResourceByEntropy._header_printed:
            print("\n" + "Model".ljust(30) + "Entropy".ljust(15) + "Mean Latency (ms)".ljust(20) + "RSS Mean (MB)")
            TestResourceByEntropy._header_printed = True

        model_filename = "yolo26s.pt"
        entropy, frames = real_frames_by_entropy
        
        # YOLO inference parameters (hardcoded to avoid AIEngine dependency)
        conf_threshold = 0.25
        iou_threshold = 0.7
        
        with contextlib.redirect_stderr(io.StringIO()):
            model   = YOLO(_model_path(model_filename))
            monitor = ResourceMonitor(sample_every_n_frames=5)

            model.predict(frames[0], conf=conf_threshold,
                          iou=iou_threshold, verbose=False)  # warm-up

            latencies_ms = []
            monitor.start()
            for i, frame in enumerate(frames):
                t0 = time.perf_counter()
                model.predict(frame, conf=conf_threshold,
                              iou=iou_threshold, verbose=False)
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                monitor.sample(i)

        s       = monitor.summary()
        mean_ms = float(np.mean(latencies_ms))

        print(f"{model_filename:<30}{entropy:<15}{mean_ms:<20.1f}{s['rss_mean_mb']:<15.1f}")
        assert mean_ms < 500


# ============================================================================
# DETECTION LATENCY SCREENING (resolution × entropy matrix)
# ============================================================================

class TestDetectionLatencyScreening:
    """
    Tests latency across resolutions at fixed high entropy using yolo26n.
    Only varies resolution (480p, 720p, 1080p); entropy fixed to high.
    """
    
    _header_printed = False

    @pytest.mark.parametrize("resolution_label", ["480p", "720p", "1080p"])
    def test_latency_by_resolution(
        self, resolution_label, synthetic_video_factory
    ):
        from ultralytics import YOLO

        # Print header once
        if not TestDetectionLatencyScreening._header_printed:
            print("\n" + "Resolution".ljust(15) + "Mean Latency (ms)".ljust(20) + "P95 Latency (ms)")
            TestDetectionLatencyScreening._header_printed = True

        conf_threshold = 0.25
        iou_threshold = 0.7

        with contextlib.redirect_stderr(io.StringIO()):
            model  = YOLO(_model_path("yolo26n.pt"))
            frames = synthetic_video_factory(resolution_label, "complex", n_frames=30)

            model.predict(frames[0], conf=conf_threshold,
                          iou=iou_threshold, verbose=False)  # warm-up

            latencies_ms = _run_inference_batch(
                model, frames, conf_threshold, iou_threshold
            )

        mean_ms = float(np.mean(latencies_ms))
        p95_ms  = float(np.percentile(latencies_ms, 95))
        max_ms  = float(np.max(latencies_ms))

        print(f"{resolution_label:<15}{mean_ms:<20.1f}{p95_ms:<15.1f}")
        threshold = {"480p": 200, "720p": 300, "1080p": 500}[resolution_label]
        assert mean_ms < threshold, (
            f"Mean latency {mean_ms:.1f}ms exceeded threshold {threshold}ms"
        )


# ============================================================================
# PIPELINE STAGE BREAKDOWN (inference / boxes / drawing)
# ============================================================================

class TestDetectionStageBreakdown:
    """
    Decompose total frame time into distinct pipeline stages using real frames
    resized to each resolution. Pinpoints which stage bottleneck changes with
    resolution.
    """

    @pytest.mark.parametrize("resolution_label", ["480p", "720p", "1080p"])
    def test_stage_breakdown_on_real_frames(self, resolution_label, real_video_frames):
        from ultralytics import YOLO
        from backend.gcs.ai.AIEngine import ProcessingState, process_detection_mode

        w, h   = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}[resolution_label]
        model  = YOLO(_model_path("yolo26n.pt"))
        state  = ProcessingState()
        frames = [cv2.resize(f, (w, h)) for f in real_video_frames[:30]]

        inference_ms, drawing_ms, boxes_ms = [], [], []
        for frame in frames:
            process_detection_mode(frame, model, state, cursor_pos=None, click_pos=None)
            if state.detection_ran_this_frame:
                inference_ms.append(state.profile_inference_ms)
                drawing_ms.append(state.profile_drawing_ms)
                boxes_ms.append(state.profile_boxes_ms)
            state.increment_frame()

        if not inference_ms:
            pytest.skip("No detection frames collected.")

        mean_infer = float(np.mean(inference_ms))
        mean_draw  = float(np.mean(drawing_ms))
        mean_boxes = float(np.mean(boxes_ms))
        total      = mean_infer + mean_draw + mean_boxes

        print(f"\n{resolution_label}\t{mean_infer:.2f}\t{mean_boxes:.2f}\t{mean_draw:.2f}\t{total:.2f}")
        assert mean_draw <= mean_infer * 2, (
            f"Drawing ({mean_draw:.1f}ms) exceeds 2× inference ({mean_infer:.1f}ms)"
        )
