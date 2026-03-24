"""Main server for Ground Control Station (GCS) backend."""

import datetime
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional 
import os
import websockets
import cv2
import time
import numpy as np
from database import get_all_objects, delete_object, record_telemetry_data
from ai.AI import ENGINE, STATE, CURSOR_HANDLER, process_frame, TELEMETRY_RECORDER
from dotenv import load_dotenv
from GeoLocate import calculate_horizontal_distance
from webrtc import webrtc_router, write_frame, get_peer_connections
import threading

load_dotenv(dotenv_path="../../.env")


active_connections: List[WebSocket] = []
flight_comp_ws: WebSocket = None

# Video stream configuration
FLIGHT_COMP_URL = f"ws://{os.getenv('FLIGHT_COMP_IP')}:{os.getenv('RPI_BACKEND_PORT', '5555')}/ws/flight-computer"
newest_telemetry = {}

telemetry_event = asyncio.Event()

process_frame_executor: Optional[ThreadPoolExecutor] = None

async def flight_computer_background_task():
    """Background task that connects to flight computer and listens for telemetry"""
    global flight_comp_ws
    while True:
        try:
            data = get_mock_telemetry() 
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
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Flight computer connection error: {e}, retrying in 5s")
            flight_comp_ws = None
            await asyncio.sleep(5)

def get_mock_telemetry():
    """Simulates a drone flying in a circle."""
    return {
        "last_time": datetime.datetime.now().timestamp(),
        "latitude": random.uniform(40.7123, 60.7133),
        "longitude": random.uniform(-74.0065, -60.0055),
        "rth_altitude": random.uniform(145.0, 155.0),
        "dlat": random.uniform(0.1, 5.0), # Ground X speed (Latitude, positive north)
        "dlon": random.uniform(0.1, 5.0), # Ground Y Speed (Longitude, positive east)
        "dalt": random.uniform(0.1, 5.0), # Ground Z speed (Altitude, positive down)
        "heading": random.randint(0, 360),
        "roll": random.uniform(-5.0, 5.0),
        "pitch": random.uniform(-5.0, 5.0),
        "yaw": random.uniform(-5.0, 5.0),
        "flight_mode": -1,
        "battery_remaining": random.uniform(30.0, 100.0), # not receiving from vehicle yet
        "battery_voltage": random.uniform(10.1, 80.6)   # not receiving from vehicle yet
    }

video_stop_event = threading.Event()
camera_executor = ThreadPoolExecutor(max_workers=1)
async def video_streaming_task():
    """Background task that reads video, processes through AI, and streams via WebRTC"""
    print("Starting USB camera video stream...")
    global newest_telemetry
    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(camera_executor, cv2.VideoCapture, 0, cv2.CAP_DSHOW)  
    if not cap.isOpened():
        print("ERROR: Could not open USB camera at index 0. Check connection.")
        return
 
    await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_WIDTH,  1280)
    await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_HEIGHT,  720)
 
    print("USB camera opened successfully.")
 
    target_interval = 1.0 / 60.0
    previous_tracking_state = False

    try:
        while not video_stop_event.is_set():
            loop_start = time.time()

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                print("Warning: failed to read frame from USB camera, retrying...")
                await asyncio.sleep(0.1)
                continue
            
            metadata = get_mock_telemetry()
 
            newest_telemetry = metadata
            telemetry_event.set()

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
        print("Releasing USB camera...")
        await asyncio.to_thread(cap.release)
        print("Video streaming task ended.")


async def follows_background_task():
    """Background task that manages following target logic"""
    while True:
        try:
            if STATE.tracking:
                # Safely check if we have telemetry data yet
                if not newest_telemetry:
                    print("Waiting for telemetry data...")
                    await asyncio.sleep(1)
                    continue

                current_mode = newest_telemetry.get('flight_mode')
                
                # if current_mode != "Guided":
                #     print("Please enter guided mode before attempting to follow.")
                #     continue

                follows_altitude = 15.0 # Hard coding the follows altitude to 15 meters (50 ft) for now
                if STATE.last_target_lat is not None and STATE.last_target_lon is not None:
                    try:
                        await send_to_flight_comp({
                            "command": "move_to_location", 
                            "location": {
                                "lat": STATE.last_target_lat,
                                "lon": STATE.last_target_lon,
                                "alt": follows_altitude
                            }
                        })  
                        print(f"Sent follow command to flight computer: lat {STATE.last_target_lat}, lon {STATE.last_target_lon}, alt {follows_altitude}")
                    except Exception as e:
                        print(f"Failed to send follow command: {e}")
                
        except Exception as e:
            # This prevents the task from dying silently if a bad packet comes through
            print(f"Crash prevented in follows task: {e}")
            traceback.print_exc()

        await asyncio.sleep(2) # Send follows commands every 2 seconds for now

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks    
    print("[GCS] Starting background tasks...")
    tasks = [ asyncio.create_task(video_streaming_task()), asyncio.create_task(follows_background_task())]
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
    for websocket in websockets_list:
        try:
            await websocket.send_text(json.dumps(message))
        except:
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
        await send_data_to_connections({"command": "set_follow_distance", "distance": distance}, flight_comp_ws)
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
        await send_data_to_connections({"command": "set_flight_mode", "mode": mode}, flight_comp_ws)
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
        await send_data_to_connections({"command": "stop_following"}, flight_comp_ws)
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