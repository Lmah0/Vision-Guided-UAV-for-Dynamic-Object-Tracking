# 533 - Drone Side Experimentation Steps
1. Increase and decrease FPS in GStreamer
2. `stress-ng --cpu 4 --cpu-load 50` (This pins 4 cores at 50% utilization).
3. `stress-ng --switch 2` (This forces the OS to rapidly switch between tasks, which often increases P95 and P99 latency).