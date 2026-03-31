import os
import time
import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO

class CursorHandler:
    """Handles cursor position and click events from user"""
    def __init__(self):
        self.cursor_pos = None  # (x, y) or None
        self.click_pos = None   # (x, y) or None

    def update_cursor(self, x: int, y: int):
        """Update current cursor position"""
        self.cursor_pos = (x, y)

    def register_click(self, x: int, y: int):
        """Register a click event at (x, y)"""
        self.click_pos = (x, y)

    def clear_click(self):
        """Clear the registered click event after processing"""
        self.click_pos = None

class TrackingConfig:
    """Centralized configuration for all tracking and detection parameters"""
    # --- Frame Skipping ---
    DETECTION_FRAME_SKIP = 1  # Skip N frames during detection phase (0=every frame, 1=every 2nd, 2=every 3rd)
    TRACKER_FRAME_SKIP =   1    # Skip N frames during tracking phase (0=every frame, 1=every 2nd)
    
    # --- Detection Parameters ---
    CONFIDENCE_THRESHOLD = 0.3    # YOLO detection confidence threshold
    MODEL_IOU = 0.5               # NMS IOU threshold for YOLO
    
    # --- Tracking Confidence ---
    MAX_CONSECUTIVE_TRACKING_FAILURES = 5  # Stop tracking if tracker fails this many frames in a row
    
    # --- Tracker Configuration ---
    PREFER_GPU_TRACKER = True     # Use VitTrack if available, otherwise use CSRT
    TRACKER_TYPE = None           # Will be auto-detected: 'vittrack' or 'csrt'
    VITTRACK_MODEL = None          # Path to VitTrack model file

class TelemetryRecorder:
    def __init__(self):
        self.is_recording = False
        self.recorded_data = []

    def start(self):
        self.is_recording = True
        self.recorded_data.clear()

    def stop_and_get_data(self):
        self.is_recording = False
        data = self.recorded_data.copy()
        self.recorded_data.clear()
        return data

    def record_telemetry(self, data: dict):
        """
        Record telemetry only called when:
        - recording enabled
        - tracking enabled
        """
        point = {
            "timestamp": data["receive_time"],
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "speed": math.sqrt(float(data["dlat"])**2 + float(data["dlon"])**2 + float(data["dalt"])**2),
            "heading": float(data["heading"]),
        }
        self.recorded_data.append(point)

def _init_tracker_config():
    """Initialize tracker type based on GPU availability"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vittrack_model_path = os.path.join(base_dir, "models", "object_tracking_vittrack_2023sep.onnx")
    gpu_available = torch.cuda.is_available()
    
    # Check GPU first, then VitTrack model availability
    if gpu_available and TrackingConfig.PREFER_GPU_TRACKER and os.path.exists(vittrack_model_path):
        TrackingConfig.TRACKER_TYPE = 'vittrack'
        TrackingConfig.VITTRACK_MODEL = vittrack_model_path
        print(f"✓ GPU available and VitTrack model found. Using VitTrack tracker (GPU-optimized)")
    else:
        if gpu_available and TrackingConfig.PREFER_GPU_TRACKER and not os.path.exists(vittrack_model_path):
            print(f"⚠ GPU available but VitTrack model not found at {vittrack_model_path}. Using CSRT tracker.")
        elif not gpu_available and TrackingConfig.PREFER_GPU_TRACKER:
            print("⚠ No GPU available. Using CSRT tracker (CPU)")
        TrackingConfig.TRACKER_TYPE = 'csrt'
        print("Using CSRT tracker (CPU)")

_init_tracker_config()


class TrackingEngine:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolo26n.pt')
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            print("YOLO will attempt to download the model...")
        else:
            print(f"Loading model from: {model_path}")

        # Public attributes for high-performance direct access (hot path)
        self.model = YOLO(model_path)
        self.tracker = None  # Created on-demand in start_tracking()
        self.tracker_type = TrackingConfig.TRACKER_TYPE
        
        # State
        self.is_tracking = False
        self.tracked_bbox = None
        self.tracked_class = None

    def detect_objects(self, frame):
        """Run YOLO detection"""
        if frame is None or frame.size == 0:
            return None
        results = self.model.predict(frame, conf=TrackingConfig.CONFIDENCE_THRESHOLD, iou=TrackingConfig.MODEL_IOU, verbose=False)
        return results[0]

    def _load_vittrack(self):
        """Initialize VitTrack tracker with GPU acceleration if available"""
        try:
            params = cv2.TrackerVit_Params()
            params.net = TrackingConfig.VITTRACK_MODEL
            
            # GPU acceleration via OpenCV DNN if available
            if torch.cuda.is_available():
                params.backend = cv2.dnn.DNN_BACKEND_CUDA
                params.target = cv2.dnn.DNN_TARGET_CUDA
                print("VitTrack: Using CUDA acceleration")
            else:
                params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
                params.target = cv2.dnn.DNN_TARGET_CPU
            
            return cv2.TrackerVit.create(params)
        except Exception as e:
            print(f"VitTrack initialization failed: {e}. Falling back to CSRT.")
            return None

    def start_tracking(self, frame, bbox, class_id):
        """Initialize tracker (VitTrack if available, else CSRT)"""
        if self.tracker_type == 'vittrack':
            self.tracker = self._load_vittrack()
        
        # Fall back to CSRT if VitTrack failed or not available
        if self.tracker is None:
            self.tracker = cv2.TrackerCSRT.create()
            self.tracker_type = 'csrt'
        
        self.tracker.init(frame, bbox)
        self.tracked_bbox = bbox
        self.tracked_class = class_id
        self.is_tracking = True
        print(f"Engine: Started tracking Class {class_id} with {self.tracker_type.upper()} tracker")


# ============================================================================
# SHARED RENDERING AND INTERACTION LOGIC
# ============================================================================

class ProcessingState:
    """Manages state for detection/tracking processing"""
    def __init__(self):
        self.tracking = False
        self.tracker = None
        self.tracked_class = None
        self.tracked_bbox = None
        self.frame_count = 0
        self.last_detection_results = None
        self.last_tracker_bbox = None
        self.last_rendered_tracking_frame = None  # Cache rendered tracking frame
        self.target_latitude = None
        self.target_longitude = None
        
        # Last target geolocation
        self.last_target_lat = None
        self.last_target_lon = None
        
        # Tracking confidence monitoring
        self.consecutive_tracking_failures = 0  # Track consecutive frames where tracker fails

        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        
        # Fine-grained profiling timings (in ms)
        self.profile_inference_ms = 0.0      # YOLO model inference time
        self.profile_boxes_ms = 0.0          # Box extraction and processing time
        self.profile_drawing_ms = 0.0        # Drawing/visualization time
        self.profile_frame_to_gpu_ms = 0.0   # Frame CPU->GPU transfer time
        self.profile_results_to_cpu_ms = 0.0 # Results GPU->CPU transfer time
        self.detection_ran_this_frame = False  # Track if detection actually ran this frame
        
        # Detailed inference breakdown
        self.profile_frame_prep_ms = 0.0     # Frame preparation before model
        self.profile_model_predict_ms = 0.0  # Actual model.predict() call
        self.profile_results_process_ms = 0.0  # Processing results after model
        
        # Input profiling
        self.profile_frame_shape = None
        self.profile_frame_dtype = None
        self.profile_frame_device = "unknown"  # CPU or GPU
        self.profile_model_device = "unknown"
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking = False
        self.tracker = None
        self.tracked_class = None
        self.tracked_bbox = None
        self.last_tracker_bbox = None
        self.last_rendered_tracking_frame = None
        self.target_latitude = None
        self.target_longitude = None
        self.consecutive_tracking_failures = 0
    
    def start_tracking(self, frame, bbox, class_id):
        """Initialize tracking from a detection"""
        if TrackingConfig.TRACKER_TYPE == 'dasiamrpn':
            # For DaSiamRPN - would need full implementation
            # For now, fall back to CSRT in ProcessingState
            self.tracker = cv2.TrackerCSRT.create()
        else:
            self.tracker = cv2.TrackerCSRT.create()
        
        self.tracker.init(frame, bbox)
        self.tracked_class = class_id
        self.tracked_bbox = bbox
        self.tracking = True
        self.last_tracker_bbox = (True, bbox)
        print(f"Started tracking object, class {self.tracked_class}")
    
    def increment_frame(self):
        """Increment frame counter"""
        self.frame_count += 1


def process_detection_mode(frame, model, state, cursor_pos, click_pos):
    """
    Process frame in detection mode.
    
    Args:
        frame: Input frame
        model: YOLO model instance
        state: ProcessingState object
        cursor_pos: Tuple (x, y) of cursor position or None
        click_pos: Tuple (x, y) of click position or None
    
    Returns:
        Tuple (output_frame, detection_results, mode_changed)
        - output_frame: Annotated frame or None if unchanged
        - detection_results: Latest detection results
        - mode_changed: True if mode switched to tracking
    """
    output_frame = None
    mode_changed = False
    
    # Determine if we should run detection this frame
    should_detect = (state.frame_count % (TrackingConfig.DETECTION_FRAME_SKIP + 1)) == 0
    
    # Reset timings for this frame
    state.profile_inference_ms = 0.0
    state.profile_boxes_ms = 0.0
    state.profile_drawing_ms = 0.0
    state.profile_frame_prep_ms = 0.0
    state.profile_results_process_ms = 0.0
    state.detection_ran_this_frame = False
    
    if should_detect:
        state.detection_ran_this_frame = True
        
        # Profile frame inputs
        state.profile_frame_shape = frame.shape
        state.profile_frame_dtype = str(frame.dtype)
        state.profile_frame_device = "GPU" if state.gpu_available else "CPU"
        
        # Profile model device
        try:
            if hasattr(model, 'device'):
                state.profile_model_device = str(model.device)
            else:
                # Check the underlying model's device
                for param in model.model.parameters():
                    state.profile_model_device = str(param.device)
                    break
        except:
            state.profile_model_device = "unknown"
        
        # Time the actual model inference
        # YOLO handles frame format conversion internally, optimized for numpy HWC format
        # Use device=0 to keep operations on GPU, half=True for fp16 memory efficiency
        t_model_start = time.time()
        results = model.predict(frame, conf=TrackingConfig.CONFIDENCE_THRESHOLD,
                               iou=TrackingConfig.MODEL_IOU, 
                               device=0 if state.gpu_available else 'cpu',
                               half=state.gpu_available, 
                               verbose=False)
        state.profile_model_predict_ms = (time.time() - t_model_start) * 1000
        
        state.last_detection_results = results
        state.profile_inference_ms = state.profile_model_predict_ms
        
        # Periodic GPU memory optimization
        if state.gpu_available and state.frame_count % 100 == 0:
            torch.cuda.empty_cache()
    else:
        results = state.last_detection_results
    
    # Process bounding boxes - convert GPU tensors to numpy only when needed
    if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
        t_boxes_start = time.time()
        # Safe conversion: handle both GPU tensors and numpy arrays
        xyxy = results[0].boxes.xyxy
        cls_vals = results[0].boxes.cls
        
        # Convert to numpy if needed (GPU tensor -> CPU numpy)
        if hasattr(xyxy, 'cpu'):
            boxes = xyxy.cpu().numpy().astype(np.int32)
        elif hasattr(xyxy, 'numpy'):
            boxes = xyxy.numpy().astype(np.int32)
        else:
            boxes = np.array(xyxy).astype(np.int32)
            
        if hasattr(cls_vals, 'cpu'):
            classes = cls_vals.cpu().numpy()
        elif hasattr(cls_vals, 'numpy'):
            classes = cls_vals.numpy()
        else:
            classes = np.array(cls_vals)
        
        # Filter to only humans (class_id == 0 in COCO dataset)
        human_mask = classes == 0
        boxes = boxes[human_mask]
        classes = classes[human_mask]
            
        state.profile_boxes_ms = (time.time() - t_boxes_start) * 1000
        
        cursor_x, cursor_y = cursor_pos if cursor_pos else (0, 0)
        
        # Draw human detections only
        t_drawing_start = time.time()
        if len(boxes) > 0:
            output_frame = frame.copy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                # Draw filled transparent box
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                output_frame = cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0)
                
                # Draw outline
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                class_id = int(classes[i])
                class_name = model.names[class_id]
                cv2.putText(output_frame, class_name, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Click on detection to start tracking
                if click_pos is not None:
                    click_x, click_y = click_pos
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        state.start_tracking(frame, (x1, y1, x2 - x1, y2 - y1), class_id)
                        mode_changed = True
                        break
        state.profile_drawing_ms = (time.time() - t_drawing_start) * 1000
    
    return output_frame, results, mode_changed


def process_tracking_mode(frame, state):
    """
    Process frame in tracking mode.
    
    Args:
        frame: Input frame
        state: ProcessingState object
    
    Returns:
        Tuple (output_frame, tracking_succeeded, mode_changed)
        - output_frame: Annotated frame or None if tracking lost
        - tracking_succeeded: True if tracking succeeded
        - mode_changed: True if mode switched back to detection
    """
    output_frame = None
    mode_changed = False
    
    # Determine if we should update tracker this frame
    should_track = (state.frame_count % (TrackingConfig.TRACKER_FRAME_SKIP + 1)) == 0
    if should_track:
        success, bbox = state.tracker.update(frame)
        state.last_tracker_bbox = (success, bbox)
        
        # Check if object has left the frame (bbox completely out of bounds)
        if success and bbox is not None:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            frame_height, frame_width = frame.shape[:2]
            
            # Object is out of frame if bbox is completely outside frame boundaries
            if (x + w < 0 or x > frame_width or 
                y + h < 0 or y > frame_height):
                print("Object left the frame, stopping tracking")
                success = False
        
        # Track consecutive tracking failures
        if not success:
            state.consecutive_tracking_failures += 1
        else:
            state.consecutive_tracking_failures = 0
    else:
        success, bbox = state.last_tracker_bbox if state.last_tracker_bbox else (False, None)
    
    # Stop tracking if too many consecutive failures (low tracking confidence)
    if state.consecutive_tracking_failures >= TrackingConfig.MAX_CONSECUTIVE_TRACKING_FAILURES:
        print(f"Tracking confidence too low ({state.consecutive_tracking_failures} consecutive failures), stopping tracking")
        state.reset_tracking()
        return None, False, True
    
    if success and bbox is not None:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        state.tracked_bbox = (x, y, w, h)
        
        # Only render when we actually update the tracker
        if should_track:
            output_frame = frame.copy()
            
            # Draw gradient fill with transparency
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
            output_frame = cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0)
            
            # Draw outline
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 200, 200), 2)
            
            state.last_rendered_tracking_frame = output_frame
        else:
            # Reuse cached frame on skipped frames
            output_frame = state.last_rendered_tracking_frame
        
        return output_frame, True, False
    else:
        print("Lost tracking, resuming detection")
        state.reset_tracking()
        return None, False, True