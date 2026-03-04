import math
from geographiclib.geodesic import Geodesic
import numpy as np

'''
AI Helper File:
    - Handles converting bbox information to longitude & latitude

    Parameters:
        uav_longitude - longitude of the center of the image in degrees
        uav_latitude - latitude of the center of the image in degrees
        uav_altitude - altitude of the uav in meters (height above ground)
        bearing - azimuth angle measured from North clockwise in degrees (drone's heading)
        cam_fov - field of view of the camera in degrees (diagonal FOV)
        img_width_px - width of the image in pixels
        img_height_px - height of the image in pixels
        obj_x_px - x coordinate of the target object in pixels where (0,0) is the center of the image
        obj_y_px - y coordinate of the target object in pixels where (0,0) is the center of the image
        
    Returns:
        (obj_longitude, obj_latitude) - the estimated longitude and latitude of the object
        
    How it works:
        1. Calculates ground coverage from camera FOV and altitude
        2. Converts pixel coordinates to real-world meters
        3. Uses geodesic math to account for Earth's curvature
        4. Returns GPS coordinates of the detected object
'''

# Camera specifications
CAM_FOV = 153  # Diagonal field of view in degrees
IMG_WIDTH_PX = 1280  # Image width in pixels
IMG_HEIGHT_PX = 720  # Image height in pixels

K_ESTIMATED = np.array([ # Calculated via intrinsics_from_fov function in this file
    [1.03221724e+03, 0.00000000e+00, 4.65673884e+02],
    [0.00000000e+00, 1.03121094e+03, 3.35705570e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

def locate(uav_latitude: float, uav_longitude: float, uav_altitude:float, bearing:float, obj_x_px:float, obj_y_px:float):
    # Calculate ground coverage area from camera FOV
    cam_fov_rad = math.radians(CAM_FOV)
    # Calculate diagonal distance on ground using FOV and altitude
    diagonal_dist = math.tan(cam_fov_rad/2) * uav_altitude * 2
    
    # Calculate actual image width on ground in meters
    # Using Pythagorean theorem: diagonal² = width² + height²
    img_width = math.sqrt(diagonal_dist**2 / (1 + IMG_HEIGHT_PX / IMG_WIDTH_PX))
    
    # Calculate conversion from pixels to meters
    # Each pixel represents this many meters on the ground
    length_per_px = img_width / IMG_WIDTH_PX
    
    # Convert pixel coordinates to real-world meter offsets
    # obj_x_px and obj_y_px are relative to (0,0)
    obj_x = obj_x_px * length_per_px  # Horizontal offset in meters
    obj_y = obj_y_px * length_per_px  # Vertical offset in meters
    # Calculate straight-line distance from drone to object on ground
    dist = math.sqrt(obj_x**2 + obj_y**2)
    
    # Calculate compass bearing to object
    angle = math.atan2(obj_y_px, obj_x_px)
    # Adjust for drone's heading and coordinate system differences
    true_bearing = (bearing + 90 - math.degrees(angle)) % 360
    
    # Calculate object's GPS coordinates while accounting for Earth's curvature
    geod = Geodesic.WGS84
    azil = true_bearing  # Compass direction from North (clockwise)
    g = geod.Direct(uav_latitude, uav_longitude, azil, dist)
    (obj_latitude, obj_longitude) = (g['lat2'], g['lon2'])
    return (obj_latitude, obj_longitude)

def calculate_horizontal_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the geodesic distance between two GPS coordinates.
    
    Parameters:
        lat1 - latitude of first point in degrees
        lon1 - longitude of first point in degrees
        lat2 - latitude of second point in degrees
        lon2 - longitude of second point in degrees
    
    Returns:
        distance in meters between the two points
    """
    geod = Geodesic.WGS84
    # geodesic inverse calculation, returns several values, we need 's12' for distance
    result = geod.Inverse(lat1, lon1, lat2, lon2) 
    return result['s12']  # distance in meters

