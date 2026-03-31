"""Flight Computer Server running on the raspberry pi onboard the drone."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
from contextlib import asynccontextmanager
from typing import List
from sendVideoStream import start_streaming_video_and_telemetry
from dotenv import load_dotenv
import threading
import socket
import time
from mavlinkMessages.mode import set_mode
import time
from mavlinkMessages.mode import set_mode
from mavlinkMessages.connect import connect_to_vehicle, verify_connection
from mavlinkMessages.commandToLocation import move_to_location

load_dotenv(dotenv_path="../../.env")

active_connections: List[WebSocket] = []

vehicle_connection = None
aircraft_type = "quadcopter"

vehicle_ip = "udp:127.0.0.1:5006" # Need to run mavproxy module on 5006

basic_telemetry = {
    "last_time": None,
    "latitude": None,
    "longitude": None,
    "altitude": None,
    "dlat": None,
    "dlon": None,
    "dalt": None,
    "heading": None,
    "roll": None,
    "pitch": None,
    "yaw": None,
    "flight_mode": -1,
    "aircraft_type": aircraft_type,
    "battery_remaining": None,
    "battery_voltage": None
}
basic_telemetry_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vehicle_connection
    
    flight_controller_thread = threading.Thread(target=update_vehicle_position_from_flight_controller, daemon=True)
    flight_controller_thread.start()
    time.sleep(0.5) # Give some time for the thread to start
    
    print(f"Attempting to connect to vehicle on: {vehicle_ip}")
    vehicle_connection = connect_to_vehicle(vehicle_ip)
    print("Vehicle connection established.")

    try:
        is_connected = verify_connection(vehicle_connection)
        if is_connected:
            print("Vehicle connection verified.")
        else:
            raise Exception ("Vehicle connection could not be verified.")
    except Exception as e:
        print(f"Error verifying vehicle connection: {e}")
        exit(1)

    if (vehicle_connection is None):
        print("Vehicle connection is None, exiting...")
        # exit(1)

    video_and_telemetry_thread = threading.Thread(
        target=start_streaming_video_and_telemetry, args=(return_telemetry_data,), daemon=True
    )
    video_and_telemetry_thread.start()
    print("Video streaming thread started")
    time.sleep(0.5)  # Give some time for the thread to start
    
    background_task = asyncio.create_task(send_telemetry_data())
    yield
    # Shutdown
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def send_data_to_connections(message: dict):
    """Send message to all connected WebSocket clients"""
    for websocket in active_connections:
        try:
            await websocket.send_text(json.dumps(message))
        except:
            if websocket in active_connections:
                active_connections.remove(websocket)


async def send_telemetry_data():
    global basic_telemetry
    global aircraft_type
    while True:
        with basic_telemetry_lock:
            basic_telemetry["aircraft_type"] = aircraft_type
            await send_data_to_connections(basic_telemetry)
        await asyncio.sleep(1)


def return_telemetry_data():
    global basic_telemetry
    with basic_telemetry_lock:
        return basic_telemetry.copy()


def setFlightMode(mode: str):
    """Set the flight mode of the drone"""
    print(f"Received request to set flight mode: {mode}")
    if not mode:
        raise ValueError("Flight mode cannot be empty.")
    if vehicle_connection is None:
        raise RuntimeError("Vehicle connection is not established.")
    try:
        set_mode(vehicle_connection, mode)
        print(f"Setting flight mode to: {mode}")
    except Exception as e:
        print(f"Failed to set flight mode: {e}")
        raise RuntimeError(f"Failed to set flight mode: {e}")


def setFollowDistance(distance: float):
    # TODO: Deferring the implementation of this until later
    """Set the follow distance of the drone"""
    if not distance or distance <= 0:
        raise ValueError("Follow distance must be a positive number")
    try:
        print(f"Setting follow distance to: {distance} meters")
    except Exception as e:
        raise RuntimeError(f"Failed to set follow distance: {e}")

def stopFollowingTarget():
    # TODO: Deferring the implementation of this until later
    """Stop following the target"""
    try:
        print("Stopping following the target")
    except Exception as e:
        raise RuntimeError(f"Failed to stop following target: {e}")
    
def moveToLocation(location):
    """Move the drone to a specified location"""
    if not location or "lat" not in location or "lon" not in location or "alt" not in location:
        raise ValueError("Invalid location data")
    try:
        print(f"Moving to location - lat: {location['lat']}, lon: {location['lon']}, alt: {location['alt']}")        
        move_to_location(vehicle_connection, location["lat"], location["lon"], location["alt"])
    except Exception as e:
        raise RuntimeError(f"Failed to move to location: {e}")


@app.websocket("/ws/flight-computer")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for GCS frontend to send commands and receive telemetry"""
    global aircraft_type
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command")
            # Handle commands
            if cmd == "move_to_location":
                moveToLocation(msg.get("location"))
            elif cmd == "set_flight_mode":
                setFlightMode(msg.get("mode"))
            elif cmd == "set_follow_distance":
                setFollowDistance(msg.get("distance"))
            elif cmd == "stop_following":
                stopFollowingTarget()
            elif cmd == "set_aircraft_type":
                requested_aircraft_type = msg.get("aircraft_type")
                if requested_aircraft_type not in ["plane", "quadcopter"]:
                    raise HTTPException(status_code=400, detail="Invalid aircraft type")
                aircraft_type = requested_aircraft_type
                with basic_telemetry_lock:
                    basic_telemetry["aircraft_type"] = aircraft_type
                print(f"Received aircraft type: {aircraft_type}")
            else:
                raise HTTPException(status_code=400, detail="Unknown command")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        error_response = {"status": 500, "error": str(e)}
        try:
            await websocket.send_text(json.dumps(error_response))
        except:
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

def update_vehicle_position_from_flight_controller():
    """Update vehicle position from flight controller data"""
    global basic_telemetry

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 5005))

    while True:
        data = sock.recvfrom(1024)
        items = data[0].decode()[1:-1].split(",")
        message_time = float(items[0])

        with basic_telemetry_lock:
            last_time = basic_telemetry.get("last_time")
            if last_time is not None and message_time <= last_time:
                continue
            elif len(items) == len(basic_telemetry) - 2: # Exclude battery fields
                basic_telemetry["last_time"] = message_time
                for i, key in enumerate(list(basic_telemetry.keys())[1:-2], start=1):
                    basic_telemetry[key] = float(items[i])
            else:
                print(f"Received data item does not match expected length...")

if __name__ == "__main__":    
    uvicorn.run("server:app", host="0.0.0.0", port=5555, reload=True)