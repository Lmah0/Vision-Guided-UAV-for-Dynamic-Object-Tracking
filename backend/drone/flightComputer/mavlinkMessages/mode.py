from pymavlink import mavutil

COPTER_MODES = {
    "Stabilize": 0.0,
    "Acro": 1.0,
    "Alt Hold": 2.0,
    "Auto": 3.0,
    "Guided": 4.0,
    "Loiter": 5.0,
    "RTL": 6.0,
    "Land": 9.0,
}

def set_mode(vehicle_connection, mode_string):
    try:
        mode_id = COPTER_MODES.get(mode_string)

        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode_string}")

        vehicle_connection.mav.command_long_send(
            vehicle_connection.target_system,
            vehicle_connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(mode_id),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        msg = vehicle_connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)

        if msg is None:
            raise RuntimeError("No COMMAND_ACK received")

        if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"Mode change rejected, ACK result={msg.result}")

        print(f"Mode changed to {mode_string}")

    except Exception as e:
        print(f"[ERROR] Failed to set mode: {repr(e)}")