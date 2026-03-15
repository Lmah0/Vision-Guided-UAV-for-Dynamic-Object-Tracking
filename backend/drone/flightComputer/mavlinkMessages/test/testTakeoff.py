import sys
import os
import time

script_dir = os.path.abspath('./..')
sys.path.append(script_dir)

import arm as arm
import mode as mode
import connect as connect
import takeoff as takeoff
import commandToLocation as commandToLocation

vehicle_connection = connect.connect_to_vehicle('127.0.0.1:14550')

arm.arm(vehicle_connection)
time.sleep(1)
mode.set_mode(vehicle_connection, "Guided")
time.sleep(1)
takeoff.takeoff(vehicle_connection, 10)
print("Vehicle should takeoff to 10 m.")