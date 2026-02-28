import math
from geographiclib.geodesic import Geodesic
import numpy as np
import navpy

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


def locate_with_fixed_gimbal(
        pixel_x, pixel_y,
        drone_lat_deg, drone_lon_deg, drone_alt_m,
        drone_roll_rad, drone_pitch_rad, drone_yaw_rad,
        camera_matrix_K=K_ESTIMATED):
    """
    Compute target geolocation from pixel location in a DOWNWARD-facing camera (on a fixed gimbal).
    roll/pitch/yaw in radians. Alt in AGL.
    Make sure that the yaw is the ArduPilot yaw from "ATTITUDE" message since it's bound by [-pi, pi]. NOT to be confused with bearing [0, 360].
    """
    pixel_homogeneous = np.array([pixel_x, pixel_y, 1.0]) # homogeneous pixel coordinate (3d camera ray)

    camera_ray = np.linalg.inv(camera_matrix_K) @ pixel_homogeneous  # Ray in camera coordinates

    camera_ray /= np.linalg.norm(camera_ray) # Normalize (|ray| = 1) - preserve direction but not magnitude (don't care about magnitude)

    R_camera_to_body = navpy.angle2dcm(
        0,                   # roll
        np.deg2rad(90),      # pitch camera down
        0                    # yaw
    )

    R_body_to_ned = navpy.angle2dcm(
        drone_yaw_rad,
        drone_pitch_rad,
        drone_roll_rad
    )

    # Combined transform: camera -> body -> NED
    world_ray = R_body_to_ned @ (R_camera_to_body @ camera_ray)

    if world_ray[2] >= 0:
        raise ValueError("The computed ray does not intersect the ground (ray points upwards). Check the input parameters and camera orientation.")

    # Figure out where the ray-plane intersects with the ground
    t_ground = -drone_alt_m / world_ray[2]

    north_offset_m = t_ground * world_ray[0]
    east_offset_m  = t_ground * world_ray[1]

    # Find the GPS coordinates of the target using the N/E offsets and the drone's current GPS location
    # Note that this should account for the curvature of the Earth? Though it shouldn't matter because we're flying at a low altitude and curvature should be negligible.
    target_lat_deg, target_lon_deg, _ = navpy.ned2lla(
        [north_offset_m, east_offset_m, 0],
        drone_lat_deg,
        drone_lon_deg,
        0  # reference altitude (doesn't matter when using N/E only)
    )

    return target_lat_deg, target_lon_deg


def intrinsics_from_fov(diagonal_fov_deg=CAM_FOV,
                        image_width_px=IMG_WIDTH_PX,
                        image_height_px=IMG_HEIGHT_PX):
    """
    Estimate the camera intrinsic matrix K using only the diagonal field of view (FOV)
    and the resolution. This is an approximation that will function as a sanity check. Hopefully
    we can get around to doing a camera calibration, but this can also allow for us to fly without
    a full calibration complete (Feb. 13, 2026).

    Inputs:
        diagonal_fov_deg - Camera diagonal field of view in degrees (default is 153 degrees for our capstone camera)
        image_width_px - Image width in pixels (default is 1280 for the capstone camera)
        image_height_px - Image height in pixels (default is 720 for the capstone camera)

    Outputs:
        K                 : 3×3 intrinsic matrix - this is what gets outputted from a camera calibration
        horizontal_fov    : Estimated horizontal FOV in radians - I don't think we need this for now???
        vertical_fov      : Estimated vertical FOV in radians - Also not needed for now I think?
    """

    # Convert the diagonal FOV to radians
    diagonal_fov_rad = np.radians(diagonal_fov_deg)

    aspect_ratio = image_width_px / image_height_px  # width / height

    # Solve for horizontal and vertical FOVs using the diagonal FOV and aspect ratio
    # Math/Geometry - tan(Fd/2)^2 = tan(Fh/2)^2 + tan(Fv/2)^2 ... Note that the triangle formed by the projection follows same proportions as the pixel aspect ratio
    tan_half_diagonal = np.tan(diagonal_fov_rad / 2)

    # Normalize contribution of width/height using aspect ratio
    normalization = np.sqrt(aspect_ratio**2 + 1)

    # These give tan(Fh/2) and tan(Fv/2)
    tan_half_horizontal = tan_half_diagonal * (aspect_ratio / normalization)
    tan_half_vertical   = tan_half_diagonal * (1 / normalization)

    # Convert them back to full horizontal/vertical FOVs in radians
    horizontal_fov = 2 * np.arctan(tan_half_horizontal)
    vertical_fov   = 2 * np.arctan(tan_half_vertical)

    # Compute the focal lengths (fx and fy) [convert the angular FOV to focal length in pixel units]
    fx = (image_width_px / 2) / np.tan(horizontal_fov / 2) # fx = (width/2)  / tan(Fh / 2)
    fy = (image_height_px / 2) / np.tan(vertical_fov / 2) # fy = (height/2) / tan(Fv / 2)

    # Principal point (assumption: this is the image centre)
    cx = image_width_px / 2
    cy = image_height_px / 2

    # Intrinsic matrix K (this is what we need)
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])

    return K, horizontal_fov, vertical_fov


if __name__ == "__main__":
    K, horizontal_fov, vertical_fov = intrinsics_from_fov()
    print("Estimated intrinsic matrix K:\n", K)
    print("Estimated horizontal FOV (degrees):", np.degrees(horizontal_fov))
    print("Estimated vertical FOV (degrees):", np.degrees(vertical_fov))