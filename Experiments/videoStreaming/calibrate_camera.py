import numpy as np
import cv2
import glob

# --- CONFIGURATION ---
# Define the number of inner corners (rows, columns). 
# Example: A board with 8x6 squares has 7x5 inner corners.
CHECKERBOARD = (8, 5) 
SQUARE_SIZE = 30  # Size of one square in mm

# Termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0)...
# This represents the "ideal" board coordinates
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Load images
images = glob.glob('calibration_images/*.jpg') # Ensure this path is correct

print(f"Found {len(images)} images. Processing...")

found = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        found += 1

cv2.destroyAllWindows()

if found > 0:
    print(f"Calibration successful with {found} images.")
    
    # --- CALIBRATE CAMERA ---
    # ret: RMS re-projection error (lower is better)
    # mtx: The Camera Matrix (K)
    # dist: Distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\nReprojection Error:", ret)
    print("\nIntrinsic Matrix K:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)

    # Save the K matrix for later use
    np.savez("camera_calibration_data.npz", K=mtx, dist=dist)
else:
    print("Chessboard corners not found. Check pattern size/visibility.")