"""
GCS Backend AI Processor: WebSocket-based detection and tracking.
Uses TrackingEngine for code sharing - EXACT SAME approach as mouse_hover_refactored.py
"""
import time
import traceback
import numpy as np
from collections import deque
from .AIEngine import TelemetryRecorder, TrackingEngine, ProcessingState, CursorHandler, process_detection_mode, process_tracking_mode
from GeoLocate import locate, locate_with_fixed_gimbal

ENGINE = TrackingEngine()
STATE = ProcessingState()
CURSOR_HANDLER = CursorHandler()
TELEMETRY_RECORDER = TelemetryRecorder()

# Basic FPS tracking (minimal overhead)
STATS_WINDOW = 100
frame_times = deque(maxlen=STATS_WINDOW)
last_fps_print = time.time()

# Track WebSocket timing
get_frame_times = deque(maxlen=100)
send_frame_times = deque(maxlen=100)

def print_fps():
    """Print FPS statistics with detailed profiling breakdown for DETECTION mode"""
    if len(frame_times) < 3:
        return
    
    avg_frame_time = np.mean(frame_times)
    fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
    mode = "TRACKING" if STATE.tracking else "DETECTION"
    
    avg_get = np.mean(get_frame_times) if len(get_frame_times) > 0 else 0
    avg_send = np.mean(send_frame_times) if len(send_frame_times) > 0 else 0

    print(f"\n[Frame {STATE.frame_count}] [{mode} MODE] FPS: {fps:.1f} (avg frame: {avg_frame_time:.2f}ms)")

    # WebSocket timing
    print(f"  WebSocket I/O:")
    print(f"    ├─ Get Frame:  {avg_get:.2f}ms")
    print(f"    └─ Send Frame: {avg_send:.2f}ms")

    # Print detailed breakdown for DETECTION mode
    if not STATE.tracking:
        if STATE.detection_ran_this_frame:
            print(f"  Input Profiling:")
            print(f"    Frame: {STATE.profile_frame_shape} dtype={STATE.profile_frame_dtype} device={STATE.profile_frame_device}")
            print(f"    Model Device: {STATE.profile_model_device}")
            print(f"  Detection Profiling (detection ran this frame):")
            print(f"    ├─ Model Inference:  {STATE.profile_inference_ms:.2f}ms")
            print(f"    │  ├─ Frame Prep:      {STATE.profile_frame_prep_ms:.2f}ms")
            print(f"    │  ├─ Model Predict:   {STATE.profile_model_predict_ms:.2f}ms (BOTTLENECK CHECK)")
            print(f"    │  └─ Results Proc:    {STATE.profile_results_process_ms:.2f}ms")
            print(f"    ├─ Extract Boxes:    {STATE.profile_boxes_ms:.2f}ms")
            print(f"    └─ Drawing/Overlay:  {STATE.profile_drawing_ms:.2f}ms")
            total_detection_time = STATE.profile_inference_ms + STATE.profile_boxes_ms + STATE.profile_drawing_ms
            print(f"    Total Detection:     {total_detection_time:.2f}ms")
        else:
            print(f"  (Using cached detection from previous frame - no inference run)")
    print()

print("AI Processor initialized, ready to process frames...")

def process_frame(frame, metadata, cursor_pos=None, click_pos=None):
    """Process a single frame through the AI pipeline and return the annotated frame"""
    try:
        
        frame_start_time = time.time()
        
        if frame is None:
            return None

        STATE.increment_frame()
        
        cursor_x, cursor_y = cursor_pos if cursor_pos else (0, 0)

        # --- DETECTION MODE or TRACKING MODE ---
        if not STATE.tracking:
            # --- DETECTION MODE ---
            output_frame, _, mode_changed = process_detection_mode(frame, ENGINE.model, STATE, (cursor_x, cursor_y), click_pos)
        else:
            # --- TRACKING MODE ---
            output_frame, tracking_succeeded, mode_changed = process_tracking_mode(frame, STATE)
            
            # Geolocation processing - only run every N frames to reduce computational load
            if tracking_succeeded and STATE.frame_count % 5 == 0:
                x, y, w, h = STATE.tracked_bbox

                current_alt = metadata["altitude"]
                current_lat = metadata["latitude"]
                current_lon = metadata["longitude"]
                heading = metadata["heading"]

                # Information for fixed gimbal
                roll = metadata["roll"]
                pitch = metadata["pitch"]
                yaw = metadata["yaw"]

                # --- Calculate Target Location ---
                image_center_x = frame.shape[1] / 2
                image_center_y = frame.shape[0] / 2
                
                bbox_center_x = x + w/2
                bbox_center_y = y + h/2
                
                obj_x_px = bbox_center_x - image_center_x
                obj_y_px = bbox_center_y - image_center_y
                
                # target_lat, target_lon = locate(current_lat, current_lon, current_alt, heading, obj_x_px, obj_y_px) # TODO: Will need this back when we switch to 2D gimbal
                target_lat, target_lon = locate_with_fixed_gimbal(bbox_center_x, bbox_center_y, current_lat, current_lon, current_alt, roll, pitch, yaw)
                STATE.last_target_lat = target_lat
                STATE.last_target_lon = target_lon
                
                if TELEMETRY_RECORDER.is_recording: 
                    metadata['longitude'] = target_lon
                    metadata['latitude'] = target_lat
                    TELEMETRY_RECORDER.record_telemetry(metadata)

                print(f"Target Found at relative latitude, longitude: {target_lat}, {target_lon}")
        
        # Return annotated frame or original if no annotation
        display_frame = output_frame if output_frame is not None else frame
        
        # Track FPS
        frame_time = (time.time() - frame_start_time) * 1000
        frame_times.append(frame_time)
        
        # Print FPS every 2 seconds
        global last_fps_print
        if time.time() - last_fps_print > 2.0:
            # print_fps()
            last_fps_print = time.time()
        
        return display_frame

    except Exception as e:
        print(f"\nERROR processing frame: {e}")
        traceback.print_exc()
        return frame  # Return original frame on error