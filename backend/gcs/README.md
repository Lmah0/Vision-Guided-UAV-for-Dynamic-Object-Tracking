# Backend for Ground Control Station (GCS)

The GCS backend serves as the central relay hub between the flight computer and the frontend interface. It handles real-time telemetry data streaming, command processing, and data persistence.

## **File Structure & Details**

### **server.py** - Main Application Server
**Purpose**: FastAPI application that serves as the central communication hub

### **database.py** - Data Persistence Layer
**Purpose**: Centralized database operations for DynamoDB integration

---
## Ports in Use
- `Port 5000 (GCS_VIDEO_PORT)` - Used for receiving video and telemetry from flight computer that's being sent via GStreamer
- `Port 8766 (GCS_BACKEND_PORT)` - Used for communication between frontend and backend
---

### **Data Flow Architecture**

#### **Telemetry Stream** (Real-time Data)
```
Flight Computer → GCS Backend → GCS Frontend
      Sends          Relay       React State
    Telemetry         Hub          Updates
```

---


## videoStreaming
- `receiveVideoStream.py` - Python file used for receiving a video stream over UDP. Also, provides the ability to benchmark video stream.

---

### **Starting the Server**:
```bash
# From project root
./start-backend.sh gcs

# Or directly
cd GCS/backend
python server.py
```

### **Running Video and AI in OpenCV Window**:
```bash
cd GCS
python -m ai.AI
```


---