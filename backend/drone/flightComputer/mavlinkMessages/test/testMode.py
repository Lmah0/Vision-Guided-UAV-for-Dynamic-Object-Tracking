import sys
import os
import time

script_dir = os.path.abspath('./..')
sys.path.append(script_dir)

import mode as mode
import connect as connect

vehicle_connection = connect.connect_to_vehicle('127.0.0.1:14550')

print("Setting mode to Guided...")
mode.set_mode(vehicle_connection, "Guided")

time.sleep(0.5)

print("Setting mode to Stabilize...")
mode.set_mode(vehicle_connection, "Stabilize")

time.sleep(0.5)

print("Setting mode to Alt Hold...")
mode.set_mode(vehicle_connection, "Alt Hold")

time.sleep(0.5)

print("Setting mode to Land...")
mode.set_mode(vehicle_connection, "Land")

time.sleep(0.5)

print("Setting mode to Auto...")
mode.set_mode(vehicle_connection, "Auto")

print("Setting mode to RTL...")
mode.set_mode(vehicle_connection, "RTL")

print("Mode setting test completed.")
