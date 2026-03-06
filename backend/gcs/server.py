"""Main server for Ground Control Station (GCS) backend."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import csv
import uvicorn
import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
import os
import websockets
import cv2
import time
import random
from database import get_all_objects, delete_object, record_telemetry_data
from ai.AI import ENGINE, STATE, CURSOR_HANDLER, process_frame, TELEMETRY_RECORDER
from dotenv import load_dotenv
from GeoLocate import calculate_horizontal_distance
from webrtc import webrtc_router, write_frame, get_peer_connections
from receiveVideoStream import VideoStreamReceiver
import threading

load_dotenv(dotenv_path="../../.env")

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai", "error-video.mp4")

active_connections: List[WebSocket] = []
flight_comp_ws: WebSocket = None

# Video stream configuration
GCS_VIDEO_PORT = os.getenv("GCS_VIDEO_PORT", 5000)
STREAM_URL = "udp://"+ os.getenv(
        "FLIGHT_COMP_IP", "192.168.1.66")+":" + str(GCS_VIDEO_PORT)  # Video from drone
FLIGHT_COMP_URL = f"ws://{os.getenv('FLIGHT_COMP_IP')}:{os.getenv('RPI_BACKEND_PORT', '5555')}/ws/flight-computer"
newest_telemetry = {}

TELEMETRY_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.csv")
TELEMETRY_STREAM_MIN_INTERVAL_SEC = float(os.getenv("TELEMETRY_STREAM_MIN_INTERVAL_SEC", "1.0"))

telemetry_event = asyncio.Event()

process_frame_executor: Optional[ThreadPoolExecutor] = None
csv_telemetry_data: List[Dict[str, Any]] = []

# Non-tracking idle telemetry behavior
IDLE_ATTITUDE_JITTER = 0.0001
IDLE_SPEED_JITTER = 0.01
IDLE_ALTITUDE_JITTER = 0.01


def load_csv_telemetry() -> List[Dict[str, Any]]:
    """Load and normalize telemetry rows from CSV one time at startup."""
    rows: List[Dict[str, Any]] = []

    if not os.path.exists(TELEMETRY_CSV_PATH):
        print(f"Telemetry CSV not found: {TELEMETRY_CSV_PATH}")
        return rows

    with open(TELEMETRY_CSV_PATH, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                timestamp_ms = float(row.get("timestamp(ms)", 0.0))
                altitude = float(row.get("BARO[1].Alt", 0.0))
                latitude = float(row.get("GPS[0].Lat", 0.0)) / 1e7
                longitude = float(row.get("GPS[0].Lng", 0.0)) / 1e7
                pitch = float(row.get("ATT.Pitch", 0.0))
                yaw = float(row.get("ATT.Yaw", 0.0))
                roll = float(row.get("ATT.Roll", 0.0))
                speed = float(row.get("GPS[0].Spd", 0.0))
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "altitude": altitude,
                    "latitude": latitude,
                    "longitude": longitude,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "speed": speed,
                }
            )

    print(f"Loaded {len(rows)} telemetry rows from CSV.")
    return rows


def build_telemetry_message(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map CSV telemetry row to frontend websocket payload format."""
    return {
        "timestamp": row["timestamp_ms"],
        "altitude": row["altitude"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "pitch": row["pitch"],
        "yaw": row["yaw"],
        "roll": row["roll"],
        "speed": row["speed"],
        # Frontend derives speed using sqrt(dlat^2 + dlon^2 + dalt^2)
        "dlat": row["speed"],
        "dlon": 0.0,
        "dalt": 0.0,
        "flight_mode": -1,
        "battery_remaining": None,
        "battery_voltage": None,
        "tracking": STATE.tracking,
        "tracked_class": ENGINE.model.names[STATE.tracked_class]
        if STATE.tracked_class is not None and ENGINE.model is not None
        else None,
        "distance_to_target": None,
        "is_recording": TELEMETRY_RECORDER.is_recording,
    }


def build_idle_telemetry_message(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """Build stable synthetic telemetry for non-tracking state."""
    idle_roll = random.uniform(-IDLE_ATTITUDE_JITTER, IDLE_ATTITUDE_JITTER)
    idle_pitch = random.uniform(-IDLE_ATTITUDE_JITTER, IDLE_ATTITUDE_JITTER)
    idle_yaw = random.uniform(-IDLE_ATTITUDE_JITTER, IDLE_ATTITUDE_JITTER)
    idle_speed = max(0.0, random.uniform(0.0, IDLE_SPEED_JITTER))
    idle_altitude = base_row["altitude"] + random.uniform(-IDLE_ALTITUDE_JITTER, IDLE_ALTITUDE_JITTER)

    return {
        "timestamp": time.time() * 1000.0,
        "altitude": idle_altitude,
        "latitude": base_row["latitude"],
        "longitude": base_row["longitude"],
        "pitch": idle_pitch,
        "yaw": idle_yaw,
        "roll": idle_roll,
        "speed": idle_speed,
        # Frontend derives speed using sqrt(dlat^2 + dlon^2 + dalt^2)
        "dlat": idle_speed,
        "dlon": 0.0,
        "dalt": 0.0,
        "flight_mode": -1,
        "battery_remaining": None,
        "battery_voltage": None,
        "tracking": False,
        "tracked_class": None,
        "distance_to_target": None,
        "is_recording": TELEMETRY_RECORDER.is_recording,
    }


async def csv_telemetry_stream_task():
    """Stream idle telemetry when not tracking; stream CSV telemetry only while tracking."""
    if not csv_telemetry_data:
        print("CSV telemetry stream not started: no telemetry data loaded.")
        return

    base_idle_row = csv_telemetry_data[0]
    print("Starting telemetry stream task (idle + tracking CSV)...")

    csv_index = 0
    was_tracking = False
    while True:
        try:
            # Transition into tracking: restart CSV playback from the beginning.
            if STATE.tracking and not was_tracking:
                csv_index = 0

            if not STATE.tracking:
                await send_data_to_connections(build_idle_telemetry_message(base_idle_row))
                await asyncio.sleep(TELEMETRY_STREAM_MIN_INTERVAL_SEC)
                was_tracking = False
                continue

            row = csv_telemetry_data[csv_index]
            await send_data_to_connections(build_telemetry_message(row))

            # Follow CSV timing as closely as possible, but never faster than configured minimum.
            if csv_index < len(csv_telemetry_data) - 1:
                next_row = csv_telemetry_data[csv_index + 1]
                delta_sec = max(
                    0.0,
                    (next_row["timestamp_ms"] - row["timestamp_ms"]) / 1000.0,
                )
                await asyncio.sleep(max(TELEMETRY_STREAM_MIN_INTERVAL_SEC, delta_sec))
                csv_index += 1
            else:
                # Loop CSV while staying in tracking mode.
                csv_index = 0
                await asyncio.sleep(TELEMETRY_STREAM_MIN_INTERVAL_SEC)

            was_tracking = True
        except asyncio.CancelledError:
            print("CSV telemetry stream task cancelled.")
            raise
        except Exception as e:
            print(f"CSV telemetry stream error: {e}")
            await asyncio.sleep(1)

async def flight_computer_background_task():
    """Background task that connects to flight computer and listens for telemetry"""
    global flight_comp_ws
    while True:
        try:
            async with websockets.connect(FLIGHT_COMP_URL) as ws:
                flight_comp_ws = ws
                print("Connected to flight computer")   
                async for message in ws:
                    try:
                        data = json.loads(message)

                        data["is_recording"] = TELEMETRY_RECORDER.is_recording

                        data["tracking"] = STATE.tracking # Add tracking state
                        if STATE.tracked_class is not None and ENGINE.model is not None:
                            data["tracked_class"] = ENGINE.model.names[STATE.tracked_class]
                        else:
                            data["tracked_class"] = None
                        
                        # Calculate distance from drone to target if tracking
                        if STATE.tracking and STATE.target_latitude is not None and STATE.target_longitude is not None:
                            drone_lat = data.get("latitude")
                            drone_lon = data.get("longitude")
                            if drone_lat is not None and drone_lon is not None:
                                distance_meters = calculate_horizontal_distance(drone_lat, drone_lon, STATE.target_latitude, STATE.target_longitude)
                                data["distance_to_target"] = distance_meters
                            else:
                                data["distance_to_target"] = None
                        else:
                            data["distance_to_target"] = None
                        
                        await send_data_to_connections(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Flight computer connection error: {e}, retrying in 5s")
            flight_comp_ws = None
            await asyncio.sleep(5)


video_stop_event = threading.Event()
video_receiver = VideoStreamReceiver(STREAM_URL)
async def video_streaming_task():
    """Background task that reads video, processes through AI, and streams via WebRTC"""
    print("Starting receive video stream background task...")
    global newest_telemetry
    # Start Live Receiver
    await asyncio.to_thread(video_receiver.start)    
    # Target 60 FPS for the loop
    target_interval = 1.0 / 60.0

    # Start fallback video once at the beginning asynchronously
    cap = await asyncio.to_thread(cv2.VideoCapture, VIDEO_PATH)
    fallback_available = cap.isOpened()
    if not fallback_available:
        print(f"Error. Could not open fallback video: {VIDEO_PATH}")

    try:
        while not video_stop_event.is_set():
            loop_start = time.time()

            # --- Try Reading Live Stream ---
            frame, metadata = await asyncio.to_thread(video_receiver.read)
            # --- Fallback Logic ---
            if frame is None:
                if fallback_available:
                    ret, file_frame = await asyncio.to_thread(cap.read)

                    # Handle End of File (Loop video)
                    if not ret:
                        await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, file_frame = await asyncio.to_thread(cap.read)

                    if ret:
                        frame = file_frame
                        # Inject dummy metadata
                        metadata = {
                                "last_time": -1,
                                "latitude":-1,
                                "longitude":-1,
                                "rth_altitude":-1,
                                "dlat":-1,
                                "dlon":-1,
                                "dalt":-1,
                                "heading":-1,
                                "roll":-1,
                                "pitch":-1,
                                "yaw":-1,
                                "flight_mode":-1,
                                "battery_remaining":-1,
                                "battery_voltage":-1,
                                "altitude" : -1,
                                "timestamp" : -1,
                                "speed" : -1
                        }

            # --- If live video and mock both fail then sleep and retry ---
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            newest_telemetry = metadata # Update newest telemetry for flight computer task
            telemetry_event.set() # Notify flight_computer_background_task new telemetry is available
            previous_tracking_state = False
            
            # --- D. AI Processing (Common for both sources) ---
            try:
                # Get Cursor/Click data
                cursor = CURSOR_HANDLER.cursor_pos
                click = CURSOR_HANDLER.click_pos

                # Run AI (Wait for result)
                annotated_frame = await asyncio.get_event_loop().run_in_executor(
                        process_frame_executor, 
                        lambda: process_frame(frame, metadata, cursor, click) 
                    )
                
                current_tracking_state = STATE.tracking
                if previous_tracking_state and not current_tracking_state: # Tracking was lost - save recording if previously active
                    save_current_recording()
                previous_tracking_state = current_tracking_state

                if click is not None:
                    CURSOR_HANDLER.clear_click()
                    print("Click cleared")

                # Send to WebRTC
                if annotated_frame is not None:
                    write_frame(annotated_frame)

            except Exception as e:
                print(f"Error processing frame: {e}")
                traceback.print_exc()

            # --- E. Rate Limiting ---
            # Calculates how much time is left in the 1/60th second window
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        print("Video streaming task cancelled.")

    finally:
        # Cleanup both sources
        print("Stopping video sources...")
        video_receiver.stop()
        if cap.isOpened():
            cap.release()
    print("Video streaming task ended.")

async def follows_background_task():
    """Background task that manages following target logic"""
    while True:
        if STATE.tracking:
            mode = newest_telemetry.get("mode") or newest_telemetry.get("flight_mode")
            if mode != "Guided":
                print("Attempting to enter follows mode...")
                await send_to_flight_comp({"command": "set_flight_mode", "mode": "Guided"})
                await asyncio.sleep(2) # Wait for mode change (delay on ArduPilot's side)
                print("Entered Guided mode for follows.")

            follows_altitude = 15.0 # Hard coding the follows altitude to 15 meters (50 ft) for now
            if STATE.last_target_lat is not None and STATE.last_target_lon is not None:
                try:
                    await send_to_flight_comp({"command": "move_to_location", "location": {
                            "lat": STATE.last_target_lat,
                            "lon": STATE.last_target_lon,
                            "alt": follows_altitude
                        }})
                    print(f"Sent follow command to flight computer: lat {STATE.last_target_lat}, lon {STATE.last_target_lon}, alt {follows_altitude}")
                except Exception as e:
                    print(f"Failed to send follow command: {e}")
        await asyncio.sleep(2) # Send follows commands every 2 seconds for now

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks    
    print("[GCS] Starting background tasks...")
    global csv_telemetry_data
    csv_telemetry_data = load_csv_telemetry()

    tasks = [
        asyncio.create_task(csv_telemetry_stream_task()),
        asyncio.create_task(video_streaming_task()),
        asyncio.create_task(follows_background_task()),
    ]
    global process_frame_executor
    process_frame_executor = ThreadPoolExecutor(max_workers=1)
    yield

    print("[GCS] Shutting down...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    video_stop_event.set() # Stop video receiver thread

    if process_frame_executor:
        process_frame_executor.shutdown(wait=True) # Stop video processing thread pool
    
    # Close WebRTC peer connections
    peer_connections = get_peer_connections()
    await asyncio.gather(
        *[pc.close() for pc in list(peer_connections)],
        return_exceptions=True
    )
    peer_connections.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebRTC router
app.include_router(webrtc_router)

# -- Websocket communication --
async def send_data_to_connections(
    message: dict, websockets_list: List[WebSocket] = active_connections
):
    """Send message to all connected WebSocket clients"""
    for websocket in list(websockets_list):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            if websocket in websockets_list:
                websockets_list.remove(websocket)


async def send_to_flight_comp(message: dict):
    """Send a JSON command to the flight computer."""
    global flight_comp_ws
    if flight_comp_ws is None:
        raise RuntimeError("Flight computer not connected")

    try:
        await flight_comp_ws.send(json.dumps(message))
    except Exception as e:
        print(f"Failed to send to flight comp: {e}")
        flight_comp_ws = None
        raise

# -- Database Endpoints --
@app.get("/objects")
def get_all_objects_endpoint():
    """Retrieve a list of all recorded objects with their classifications and timestamps"""
    try:
        return get_all_objects()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve objects: {str(e)}"
        )

@app.delete("/delete/object/{object_id}")
def delete_object_endpoint(object_id: str):
    """Delete a recorded object from the DynamoDB table by its ID"""
    try:
        success = delete_object(object_id)
        if success:
            return {"status": 200}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete object")
    except Exception:
        raise HTTPException(status_code=500, detail=f"Failed to delete object")

@app.post("/recording")
def toggle_recording():
    if TELEMETRY_RECORDER.is_recording:
        try:
            save_current_recording()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to save recording data: {str(e)}"
            )
        return {"is_recording": False}
    TELEMETRY_RECORDER.start()
    return {"is_recording": True}

# -- Flight Computer Communication Endpoints --
@app.post("/setFollowDistance")
async def set_follow_distance(request: dict = Body(...)):
    """Set the follow distance"""
    distance = request.get("distance")
    if distance is None:
        raise HTTPException(status_code=400, detail="Missing 'distance' in body")
    try:
        await send_to_flight_comp({"command": "set_follow_distance", "distance": distance})
        return {"status": 200, "message": f"Follow distance set to {distance} meters"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set follow distance: {str(e)}"
        )


@app.post("/setFlightMode")
async def set_flight_mode(request: dict = Body(...)):
    """Set the flight mode"""
    mode = request.get("mode")
    print(f"Received frontend request to set flight mode: {mode}")
    if not mode:
        raise HTTPException(status_code=400, detail="Missing 'mode' in body")
    try:
        await send_to_flight_comp({"command": "set_flight_mode", "mode": mode})
        return {"status": 200, "message": f"Flight mode set to {mode}"}
    except Exception as e:
        print("Sent request to change mode but failed at the flight computer.")
        raise HTTPException(status_code=500, detail=f"Failed to set flight mode: {str(e)}")

@app.post("/stopFollowing")
async def stop_following():
    """Stop following the target"""
    try:
        save_current_recording()
        STATE.reset_tracking()
        await send_to_flight_comp({"command": "stop_following"})
        return {"status": 200, "message": "Stopped following the target."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop following: {str(e)}"
        )


@app.websocket("/ws/gcs")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for GCS frontend to send commands and receive telemetry"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                command_type = data.get("type")
                # Handle mouse movements and clicks for AI
                if command_type == "mouse_move":
                    CURSOR_HANDLER.update_cursor(data.get("x"), data.get("y"))
                elif command_type == "click":
                    CURSOR_HANDLER.register_click(data.get("x"), data.get("y"))
                    print(f"Registered click at ({data.get('x')}, {data.get('y')})")

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        print(f"Client disconnected.")
    except Exception as e:
        error_response = {"status": 500, "error": str(e)}
        try:
            await websocket.send_text(json.dumps(error_response))
        except:
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

def save_current_recording():
    """Stop recording and save telemetry data to db if present."""
    if not TELEMETRY_RECORDER.is_recording:
        return

    tracked_obj_data = TELEMETRY_RECORDER.stop_and_get_data()
    if not tracked_obj_data:
        return

    classification = "unknown"
    if STATE.tracked_class is not None:
        classification = ENGINE.model.names[STATE.tracked_class]

    record_telemetry_data(tracked_obj_data, classification=classification)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv('GCS_BACKEND_PORT')), reload=False)