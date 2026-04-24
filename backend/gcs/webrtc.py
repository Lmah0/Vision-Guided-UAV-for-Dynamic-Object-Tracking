"""WebRTC functionality for streaming AI-processed video frames."""
from fastapi import APIRouter
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from pydantic import BaseModel
import cv2
import time
import numpy as np
import traceback
import threading
from collections import deque

# Frame buffer for video streaming
_frame_lock = threading.Lock()
_current_frame = None
_frame_write_time = None # Timestamp when write_frame() was last called

# Benchmark stats
_stats_lock = threading.Lock()
_latency_samples = deque(maxlen=10000) # per-frame server latency (ms)
_frames_sent = 0  # total frames sent

# WebRTC peer connections
_peer_connections = set()


class AIVideoStreamTrack(VideoStreamTrack):
    """
    Custom video track that streams AI-processed frames via WebRTC.
    Optimized for Low Latency using Wall Clock timestamps.
    """
    def __init__(self):
        super().__init__()
        self._start = None

    async def recv(self):
        """Give WebRTC the next frame to send."""
        try:
            if self._start is None:
                self._start = time.time()

            pts, time_base = await self.next_timestamp()

            global _current_frame, _frame_lock, _frame_write_time
            with _frame_lock:
                frame = _current_frame.copy() if _current_frame is not None else None
                write_time = _frame_write_time

            # Measure server-side latency (from when frame recieved from AI Pipeline to WebRTC send)
            if write_time is not None:
                latency_ms = (time.time() - write_time) * 1000 
                global _stats_lock, _latency_samples, _frames_sent
                with _stats_lock:
                    _latency_samples.append(latency_ms)
                    _frames_sent += 1

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
        except Exception as e:
            print(f"WebRTC recv ERROR: {e}")
            traceback.print_exc()
            raise


class RTCOffer(BaseModel):
    """WebRTC offer from client."""
    sdp: str
    type: str


# Create WebRTC router
webrtc_router = APIRouter(prefix="", tags=["webrtc"])

@webrtc_router.post("/offer")
async def handle_offer(offer: RTCOffer):
    """Endpoint that handles WebRTC offer from frontend and return answer"""
    rtc_offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        if state in ("connected", "failed", "closed"):
            print(f"WebRTC connection state: {state}")
        if state == "failed" or state == "closed":
            await pc.close()
            _peer_connections.discard(pc)

    video_track = AIVideoStreamTrack()
    pc.addTrack(video_track)

    # Process the offer from frontend and create an answer
    await pc.setRemoteDescription(rtc_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


@webrtc_router.get("/webrtc/stats")
def get_webrtc_stats():
    """Return accumulated server-side latency samples and frame count, then reset."""
    global _latency_samples, _frames_sent, _stats_lock
    with _stats_lock:
        samples = list(_latency_samples)
        sent = _frames_sent
        _latency_samples.clear()
        _frames_sent = 0
    return {"latency_samples_ms": samples, "frames_sent": sent}


def write_frame(frame):
    """Write a frame to the shared buffer for WebRTC streaming."""
    global _current_frame, _frame_lock, _frame_write_time
    with _frame_lock:
        _current_frame = frame
        _frame_write_time = time.time()


def get_peer_connections():
    """Get the set of active peer connections for cleanup."""
    return _peer_connections