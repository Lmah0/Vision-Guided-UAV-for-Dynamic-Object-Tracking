"""
Interactive object detection and tracking with cv2 window interface.
Uses TrackingEngine for code sharing but inlines hot path for performance.
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AIEngine import TrackingEngine, ProcessingState, process_detection_mode, process_tracking_mode

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Interactive object detection and tracking')
parser.add_argument('--stats', action='store_true', default=False, help='Enable statistics collection and reporting')
args = parser.parse_args()
COLLECT_STATS = args.stats

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
ai_dir = os.path.dirname(script_dir)  # Parent directory (backend/gcs/ai/)

# Use paths relative to the AI module directory
VIDEO_PATH = os.path.join(ai_dir, "video.mp4")

# Initialize engine (reuses code with GCS backend)
engine = TrackingEngine()

# Direct references to engine attributes for hot path access (bypasses property overhead)
model = engine.model

# Initialize processing state
state = ProcessingState()

# Check if video file exists, with helpful error message
if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file not found at {VIDEO_PATH}")
    print(f"Expected location: {VIDEO_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    sys.exit(1)

print(f"✓ Video loaded: {VIDEO_PATH}")

cursor_x, cursor_y = 0, 0
click_flag = False

# Performance statistics (track last 100 frames)
STATS_WINDOW = 100
frame_times = deque(maxlen=STATS_WINDOW)

def mouse_event(event, x, y, flags, param):
    global cursor_x, cursor_y, click_flag
    cursor_x, cursor_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_flag = True

cv2.namedWindow("Interactive Segmentation")
cv2.setMouseCallback("Interactive Segmentation", mouse_event)

def print_performance_stats():
    """Print performance statistics with detailed profiling breakdown for DETECTION mode"""
    if not COLLECT_STATS:
        return
    
    if len(frame_times) < 3:
        return
    
    avg_frame_time = np.mean(frame_times)
    fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
    mode = "TRACKING" if state.tracking else "DETECTION"
    
    print("\n" + "="*60)
    print(f"[Frame {state.frame_count}] [{mode} MODE] FPS: {fps:.2f} (avg frame: {avg_frame_time:.2f}ms)")
    
    # Print detailed breakdown for DETECTION mode
    if not state.tracking:
        if state.detection_ran_this_frame:
            print(f"Input Profiling:")
            print(f"  Frame: {state.profile_frame_shape} dtype={state.profile_frame_dtype} device={state.profile_frame_device}")
            print(f"  Model Device: {state.profile_model_device}")
            print(f"Detection Profiling (detection ran this frame):")
            print(f"  ├─ Model Inference:  {state.profile_inference_ms:.2f}ms")
            print(f"  │  ├─ Frame Prep:      {state.profile_frame_prep_ms:.2f}ms")
            print(f"  │  ├─ Model Predict:   {state.profile_model_predict_ms:.2f}ms (BOTTLENECK CHECK)")
            print(f"  │  └─ Results Proc:    {state.profile_results_process_ms:.2f}ms")
            print(f"  ├─ Extract Boxes:    {state.profile_boxes_ms:.2f}ms")
            print(f"  └─ Drawing/Overlay:  {state.profile_drawing_ms:.2f}ms")
            total_detection_time = state.profile_inference_ms + state.profile_boxes_ms + state.profile_drawing_ms
            print(f"  Total Detection:     {total_detection_time:.2f}ms")
        else:
            print(f"(Using cached detection from previous frame - no inference run)")
    
    print("="*60)

while cap.isOpened():
    if COLLECT_STATS:
        frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    state.increment_frame()
    annotated_frame = None

    if not state.tracking:
        # --- DETECTION MODE ---
        output_frame, _, mode_changed = process_detection_mode(
            frame, model, state, (cursor_x, cursor_y), (cursor_x, cursor_y) if click_flag else None
        )
        annotated_frame = output_frame
        if mode_changed:
            click_flag = False
    else:
        # --- TRACKING MODE ---
        output_frame, tracking_succeeded, mode_changed = process_tracking_mode(frame, state)
        annotated_frame = output_frame
        if mode_changed:
            # Switched back to detection mode
            pass
    
    # Display frame
    if annotated_frame is None:
        cv2.imshow("Interactive Segmentation", frame)
    else:
        cv2.imshow("Interactive Segmentation", annotated_frame)
    
    # Record frame time and print stats every 100 frames
    if COLLECT_STATS:
        frame_time = (time.time() - frame_start_time) * 1000
        frame_times.append(frame_time)
    
    if COLLECT_STATS and state.frame_count % 100 == 0:
        print(f"[Frame {state.frame_count}] ", end="")
        print_performance_stats()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print final statistics
if COLLECT_STATS:
    print("\n" + "="*60)
    print("Final Performance Statistics")
    print_performance_stats()
