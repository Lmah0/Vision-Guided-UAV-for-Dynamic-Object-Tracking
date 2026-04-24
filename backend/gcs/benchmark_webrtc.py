import os
import time
import cv2
import requests
import statistics
from webrtc import write_frame  # shares the same buffer as the running server

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai", "video.mp4")
TARGET_FPS_LIST = [15, 30, 45, 60]
TRIALS= 10
DURATION = 240 # seconds
PORT = 8766
STATS_URL = f"http://localhost:{PORT}/webrtc/stats"

def run_trial(target_fps: int, trial_num: int, duration: int):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    interval = 1.0/target_fps
    frame_index = 0
    start_time = time.time()

    while (time.time() - start_time) < duration:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret and frame is not None:
            write_frame(frame)
            frame_index += 1

        elapsed = time.time() - loop_start
        time.sleep(max(0, interval - elapsed))

    cap.release()
    actual_duration = time.time() - start_time

    try:
        resp = requests.get(STATS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error. Could not fetch stats: {e}")
        data = {}

    latencies = data.get("latency_samples_ms", [])
    frames_sent = data.get("frames_sent", 0)
    browser_fps= data.get("browser_fps", float("nan"))
    frame_drop_pct = data.get("frame_drop_rate_pct", float("nan"))
    if actual_duration > 0:
        server_fps = frames_sent / actual_duration
    else:
        server_fps = 0
    mean_lat = statistics.mean(latencies) if latencies else float("nan")

    print(f"Trial #: {trial_num} | Server Latency: {mean_lat:.3f}ms | Browser FPS: {browser_fps:.3f} | Server FPS: {server_fps:.3f} | Frame Drop Rate: {frame_drop_pct:.3f}%")

def main():
    for fps in TARGET_FPS_LIST:
        print(f"Target FPS: {fps}")
        for trial in range(1, TRIALS + 1):
            run_trial(fps, trial, DURATION)
            time.sleep(2)

if __name__ == "__main__":
    main()
