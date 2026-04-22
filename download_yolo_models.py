#!/usr/bin/env python3
"""
Download YOLO models for testing.
Run this once to cache all models locally before running tests.
"""

from ultralytics import YOLO
import sys

# Models to download: YOLOv8 variants (pre-downloaded yolo26/yolo11 handled separately)
MODELS_TO_DOWNLOAD = [
    "yolov8n",      # YOLOv8 nano
    "yolov8s",      # YOLOv8 small
]

print("=" * 80)
print("DOWNLOADING YOLO MODELS FOR PERFORMANCE TESTING")
print("=" * 80)

for i, model_name in enumerate(MODELS_TO_DOWNLOAD, 1):
    print(f"\n[{i}/{len(MODELS_TO_DOWNLOAD)}] Downloading {model_name}...")
    try:
        model = YOLO(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY")
print("=" * 80)
print("\nReady to run tests:")
print("  pytest tests/unit/detection/test_cv_detection.py::TestFactorialDesign -v")
