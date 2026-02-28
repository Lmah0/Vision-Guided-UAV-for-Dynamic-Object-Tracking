import cv2
import time
import os

# --- CONFIGURATION ---
# IMPORTANT: This must match the INNER corners of your board.
# If your board has 10 squares wide and 7 squares high, use (9, 6).
CHECKERBOARD = (8, 5) 

# Minimum seconds to wait between automatic photos
MIN_TIME_BETWEEN_SHOTS = 2.0 

# Folder to save images
SAVE_DIR = "calibration_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Open the webcam (0 is usually the default laptop/USB camera)
# If you have multiple cameras, try changing 0 to 1 or 2
cap = cv2.VideoCapture(1)

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print(f"Searching for {CHECKERBOARD} checkerboard...")
print(f"Images will be saved to '{SAVE_DIR}'")
print("Press 'q' to quit.")

last_save_time = time.time()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Work on a grayscale copy for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Make a copy to draw on (so we don't save the drawn lines)
    display_frame = frame.copy()

    # Detect corners
    # flags help detection speed and accuracy
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if ret == True:
        # Draw the corners on the screen so you know it's working
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
        
        # Check if enough time has passed to save a new photo
        current_time = time.time()
        if (current_time - last_save_time) > MIN_TIME_BETWEEN_SHOTS:
            # Save the CLEAN frame (not the one with lines drawn)
            filename = os.path.join(SAVE_DIR, f"calib_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"Captured {filename}")
            count += 1
            last_save_time = current_time
            
            # Visual flash effect (white rectangle) to signal a photo was taken
            cv2.rectangle(display_frame, (0,0), (frame.shape[1], frame.shape[0]), (255,255,255), -1)

    # Show the video feed
    cv2.imshow('Calibration Capture', display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")