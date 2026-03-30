from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialect

def move_to_location(vehicle_connection, latitude, longitude, altitude):
    try:
        mavlink_message = dialect.MAVLink_mission_item_int_message(
            target_system=vehicle_connection.target_system,
            target_component=vehicle_connection.target_component,
            seq=0,
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            current=2, # 2 means this is the current waypoint to be executed
            autocontinue=0,
            param1=0, # Hold time in decimal seconds (ignored for fixed-wing)
            param2=0, # Acceptance radius in metres (ignored for fixed-wing)
            param3=0, # Pass through waypoint (0 to pass through, 1 to stop at waypoint) (ignored for fixed-wing)
            param4=0, # Desired yaw angle at waypoint (ignored for fixed-wing)
            x=int(latitude * 10**7), # Latitude in 1E7 degrees
            y=int(longitude * 10**7), # Longitude in 1E7 degrees
            z=altitude # Altitude in metres (relative to
        )
        vehicle_connection.mav.send(mavlink_message)

        received_message = vehicle_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        if received_message:
            print(f"Received MISSION_ACK: {received_message}")
        else:
            print("No MISSION_ACK received within timeout.")
    except Exception as e:
        print(f"Error commanding drone to location: {e}")