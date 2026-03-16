from pymavlink import mavutil

def verify_connection(connection):
    try:
        connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))
        return True

    except Exception as e:
        print(f"Error: Unable to connect to the vehicle. See error info: {e}")
        return False

def connect_to_vehicle(ip_and_port='udp:127.0.0.1:14550'):
    try:
        connection = mavutil.mavlink_connection(ip_and_port)
        if not verify_connection(connection):
            raise RuntimeError("Heartbeat not received.")
            
    except Exception as e:
        print(f"Error: Unable to connect to the vehicle at {ip_and_port}. See error info: {e}")
        exit(1)
    
    return connection