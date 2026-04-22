# tests/conftest.py
"""
Shared fixtures and utilities for CV performance experiments.

Provides:
  ResourceMonitor           — samples CPU% + RSS at regular frame intervals
  real_video_frames         — extracts frames from video.mp4 (high entropy, many humans)
  error_video_frames        — extracts frames from error-video.mp4 (low entropy, single human)
  real_frames_by_entropy    — parametrized fixture: (label, frames) for low/high entropy videos
  synthetic_video_factory   — procedural frames for controlled resolution screening
  tracking_sequence_factory — synthetic motion sequences with ground-truth bboxes
  compute_iou               — IoU helper for tracking accuracy tests
"""

from pathlib import Path
import json
from typing import Optional, Dict, List
import cv2
import numpy as np
import psutil
import pytest

# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parents[1]
VIDEO_PATH      = ROOT / "backend" / "gcs" / "ai" / "video.mp4"
ERROR_VIDEO_PATH = ROOT / "backend" / "gcs" / "ai" / "error-video.mp4"

RESOLUTIONS = {
    "480p":  (854,  480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
}

# Global results collector
BENCHMARK_RESULTS = {}


# ============================================================================
# RESOURCE MONITOR
# ============================================================================

class ResourceMonitor:
    """
    Samples process RSS memory (MB) and CPU% at regular frame intervals.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        for i, frame in enumerate(frames):
            do_work(frame)
            monitor.sample(i)
        print(monitor.summary())
    """

    def __init__(self, sample_every_n_frames: int = 5):
        self._process  = psutil.Process()
        self._interval = sample_every_n_frames
        self._samples  = []

    def start(self):
        """Prime the CPU% counter (first call always returns 0.0 by design)."""
        self._samples.clear()
        self._process.cpu_percent(interval=None)

    def sample(self, frame_idx: int):
        if frame_idx % self._interval == 0:
            self._samples.append({
                "frame":   frame_idx,
                "rss_mb":  self._process.memory_info().rss / (1024 ** 2),
                "cpu_pct": self._process.cpu_percent(interval=None),
            })

    def summary(self) -> dict:
        if not self._samples:
            return {}
        rss = [s["rss_mb"]  for s in self._samples]
        cpu = [s["cpu_pct"] for s in self._samples]
        return {
            "rss_mean_mb":  float(np.mean(rss)),
            "rss_max_mb":   float(np.max(rss)),
            "rss_delta_mb": float(rss[-1] - rss[0]),
            "cpu_mean_pct": float(np.mean(cpu)),
            "cpu_max_pct":  float(np.max(cpu)),
            "n_samples":    len(self._samples),
        }


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def compute_iou(b1: tuple, b2: tuple) -> float:
    """IoU between two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# FRAME / VIDEO GENERATORS
# ============================================================================

def generate_synthetic_frame(width: int, height: int, scene_entropy: str = "simple", seed: int = 42) -> np.ndarray:
    """
    Generate synthetic frames with varying complexity levels.
    
    'low'       = minimal texture (open sky)
    'medium'    = moderate complexity (10 random rectangles)
    'high'      = high complexity (20 random rectangles, denser)
    'very_high' = maximum complexity (40+ rectangles, very dense texture)
    
    Args:
        seed: Random seed for reproducibility (incremented per frame enables variation)
    """
    rng = np.random.default_rng(seed)
    
    if scene_entropy == "low":
        # Uniform sky-blue gradient (minimal texture)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 135
        frame[:, :, 1] = 206
        frame[:, :, 2] = 235
    elif scene_entropy == "medium":
        # Moderate noise + 10 rectangles
        frame = rng.integers(50, 150, (height, width, 3), dtype=np.uint8)
        for _ in range(10):
            x1 = int(rng.integers(0, max(1, width  - 50)))
            y1 = int(rng.integers(0, max(1, height - 50)))
            x2 = int(rng.integers(x1 + 10, min(x1 + 100, width)))
            y2 = int(rng.integers(y1 + 10, min(y1 + 100, height)))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    elif scene_entropy == "high":
        # Higher noise + 20 rectangles (previous "complex")
        frame = rng.integers(50, 200, (height, width, 3), dtype=np.uint8)
        for _ in range(20):
            x1 = int(rng.integers(0, max(1, width  - 50)))
            y1 = int(rng.integers(0, max(1, height - 50)))
            x2 = int(rng.integers(x1 + 10, min(x1 + 100, width)))
            y2 = int(rng.integers(y1 + 10, min(y1 + 100, height)))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    elif scene_entropy == "very_high":
        # Maximum noise + 40+ overlapping rectangles (very dense)
        frame = rng.integers(50, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(40):
            x1 = int(rng.integers(0, max(1, width  - 30)))
            y1 = int(rng.integers(0, max(1, height - 30)))
            x2 = int(rng.integers(x1 + 5, min(x1 + 80, width)))
            y2 = int(rng.integers(y1 + 5, min(y1 + 80, height)))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # Add additional texture overlay
        noise = rng.integers(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.7, noise, 0.3, 0)
    else:
        raise ValueError(f"Unknown scene_entropy: '{scene_entropy}'")
    return frame


def generate_tracking_sequence(
    width: int,
    height: int,
    n_frames: int,
    motion_speed: int,
    bbox_size: tuple,
    occlusion_fraction: float = 0.0,
) -> tuple:
    """
    Build a synthetic motion sequence for CSRT tracking tests.

    Target: a brightly coloured rectangle moving horizontally across a
    per-frame random-noise background. The distinct colour boundary gives
    CSRT a reliable feature to latch onto, and ground-truth positions are
    exact.

    Returns:
        (frames, initial_bbox, ground_truth_bboxes)
        where bboxes are (x, y, w, h) tuples.
    """
    bw, bh   = bbox_size
    start_x  = bw
    start_y  = max(0, (height - bh) // 2)
    frames, gt_bboxes = [], []

    for i in range(n_frames):
        rng    = np.random.default_rng(seed=i)
        bg     = rng.integers(60, 140, (height, width, 3), dtype=np.uint8)
        tx     = min(start_x + i * motion_speed, width - bw - 1)
        ty     = start_y
        frame  = bg.copy()

        # Solid target with inner pattern for texture richness
        cv2.rectangle(frame, (tx, ty), (tx + bw, ty + bh), (0, 140, 255), -1)
        cv2.rectangle(frame, (tx + 4, ty + 4), (tx + bw - 4, ty + bh - 4), (255, 100, 0), 2)

        # Partial occlusion: black bar covers left fraction of target
        if occlusion_fraction > 0:
            occ_w = int(bw * occlusion_fraction)
            cv2.rectangle(frame, (tx, ty), (tx + occ_w, ty + bh), (10, 10, 10), -1)

        frames.append(frame)
        gt_bboxes.append((tx, ty, bw, bh))

    return frames, gt_bboxes[0], gt_bboxes


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def resource_monitor():
    return ResourceMonitor(sample_every_n_frames=5)


@pytest.fixture
def real_video_frames():
    """
    Extracts up to 120 frames from video.mp4 — the actual project test video
    used by mouse_hover.py. Contains real-world content including humans.
    Skips if the file is absent.
    """
    if not VIDEO_PATH.exists():
        pytest.skip(f"Test video not found: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        pytest.skip(f"Could not open test video: {VIDEO_PATH}")
    frames = []
    while len(frames) < 120:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    cap.release()
    if not frames:
        pytest.skip("No frames could be read from test video.")
    return frames


@pytest.fixture(params=["480p", "720p", "1080p"])
def resolution(request):
    return request.param


@pytest.fixture
def error_video_frames():
    """
    Extracts up to 120 frames from error-video.mp4 — low scene-entropy footage
    containing a single human subject. Skips if the file is absent.
    """
    if not ERROR_VIDEO_PATH.exists():
        pytest.skip(f"Error video not found: {ERROR_VIDEO_PATH}")
    cap = cv2.VideoCapture(str(ERROR_VIDEO_PATH))
    if not cap.isOpened():
        pytest.skip(f"Could not open error video: {ERROR_VIDEO_PATH}")
    frames = []
    while len(frames) < 120:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    cap.release()
    if not frames:
        pytest.skip("No frames could be read from error video.")
    return frames


@pytest.fixture(
    params=[
        ("low",  ERROR_VIDEO_PATH),   # error-video.mp4 — single human
        ("high", VIDEO_PATH),          # video.mp4       — many humans
    ],
    ids=["entropy=low", "entropy=high"],
)
def real_frames_by_entropy(request):
    """
    Parametrized fixture yielding (entropy_label, frames) pairs.
    ``low``  — error-video.mp4 (single human, sparse detections)
    ``high`` — video.mp4       (many humans, dense detections)
    """
    label, path = request.param
    if not path.exists():
        pytest.skip(f"Video not found for entropy={label}: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.skip(f"Could not open video for entropy={label}: {path}")
    frames = []
    while len(frames) < 60:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    cap.release()
    if not frames:
        pytest.skip(f"No frames read for entropy={label}.")
    return label, frames


@pytest.fixture(params=["simple", "complex"])
def scene_entropy(request):
    return request.param


@pytest.fixture
def synthetic_frame_factory():
    """factory(resolution_label, entropy) -> np.ndarray"""
    def _factory(resolution_label: str = "720p", entropy: str = "simple") -> np.ndarray:
        w, h = RESOLUTIONS[resolution_label]
        return generate_synthetic_frame(w, h, entropy)
    return _factory


@pytest.fixture
def synthetic_video_factory():
    """factory(resolution_label, entropy, n_frames) -> list of identical frames."""
    def _factory(
        resolution_label: str = "720p",
        entropy: str = "simple",
        n_frames: int = 30,
    ) -> list:
        w, h  = RESOLUTIONS[resolution_label]
        base  = generate_synthetic_frame(w, h, entropy)
        return [base.copy() for _ in range(n_frames)]
    return _factory


@pytest.fixture
def tracking_sequence_factory():
    """
    factory(resolution_label, motion_speed, bbox_size, occlusion_fraction, n_frames)
        -> (frames, initial_bbox, ground_truth_bboxes)
    """
    def _factory(
        resolution_label:   str   = "720p",
        motion_speed:       int   = 4,
        bbox_size:          tuple = (128, 128),
        occlusion_fraction: float = 0.0,
        n_frames:           int   = 60,
    ) -> tuple:
        w, h = RESOLUTIONS[resolution_label]
        return generate_tracking_sequence(
            w, h, n_frames, motion_speed, bbox_size, occlusion_fraction
        )
    return _factory


# ============================================================================
# BENCHMARK RESULT COLLECTION
# ============================================================================

class BenchmarkResultCollector:
    """
    Centralized benchmark result storage and reporting.
    Collects latency, memory, and CPU metrics across all test runs.
    """
    
    def __init__(self):
        self.results = {}  # {test_name: {latencies, memory, cpu, metadata}}
    
    def record(
        self,
        test_name: str,
        latencies_ms: List[float],
        rss_mb: Optional[List[float]] = None,
        cpu_pct: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ):
        """Record benchmark result for a single test."""
        self.results[test_name] = {
            "latencies_ms": list(latencies_ms),
            "rss_mb": list(rss_mb) if rss_mb else None,
            "cpu_pct": list(cpu_pct) if cpu_pct else None,
            "metadata": metadata or {}
        }
    
    def save_json(self, filepath: Path):
        """Save all results to JSON."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved benchmark results to {output_path}")
    
    def get_summary(self, test_name: str) -> Dict:
        """Get statistical summary for a single test."""
        try:
            from tests.performance_stats import BenchmarkResult
            result = self.results[test_name]
            br = BenchmarkResult(
                name=test_name,
                latencies_ms=result["latencies_ms"],
                rss_mb=result["rss_mb"],
                cpu_pct=result["cpu_pct"],
                metadata=result["metadata"]
            )
            return br.summary()
        except Exception as e:
            print(f"Warning: Could not generate summary for {test_name}: {e}")
            return {}
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get statistical summaries for all tests."""
        return {name: self.get_summary(name) for name in self.results}


# Global collector instance
_benchmark_collector = BenchmarkResultCollector()


@pytest.fixture(scope="session")
def benchmark_results():
    """Fixture providing access to the global results collector."""
    return _benchmark_collector


# Hook to save results after all tests
def pytest_sessionfinish(session, exitstatus):
    """Export benchmark results as CSV tables and matplotlib plots only."""
    try:
        from tests.results_exporter import export_results
        export_results(_benchmark_collector.get_all_summaries())
    except Exception as e:
        print(f"Warning: Could not export results: {e}")

