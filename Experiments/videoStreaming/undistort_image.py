import numpy as np
import cv2
import glob

# Load the data you saved earlier
with np.load('camera_calibration_data.npz') as data:
    mtx = data['K']
    dist = data['dist']

print("Loaded Camera Matrix:\n", mtx)
print("Loaded Distortion Coefficients:\n", dist)

# Open camera
cap = cv2.VideoCapture(1) # Change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    
    # Generate a new optimal camera matrix (refines the view)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop the image (optional, sometimes undistortion creates black edges)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Show side-by-side
    cv2.imshow('Original (Distorted)', frame)
    cv2.imshow('Result (Undistorted)', dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()