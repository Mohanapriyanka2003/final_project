import asyncio
import threading
import time
import json
import logging
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data_generator import EcoWaveformGenerator
from ml_classifier import SonarClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SonarApp")

app = FastAPI(title="AI-Enabled Underwater Object Detection System API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve path to frontend files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# Mount static folder assets (css and js)
app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")

# Route root requests directly to index.html
@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


# Global variables for ML Classifier and Sound Generator
classifier = SonarClassifier()
generator = EcoWaveformGenerator()

# Virtual Targets inside the ocean coordinate space
# Format: {"id": str, "name": str, "angle": float, "distance": float, "class": str}
virtual_targets: Dict[str, dict] = {
    "target_1": {"id": "target_1", "name": "Wreckage Rock", "angle": 45.0, "distance": 1.8, "class": "Rock"},
    "target_2": {"id": "target_2", "name": "Deep Sea Fish Shoal", "angle": 90.0, "distance": 3.2, "class": "Fish"},
    "target_3": {"id": "target_3", "name": "Patrol Submarine", "angle": 135.0, "distance": 2.5, "class": "Submarine"},
}

# Sonar State
sonar_config = {
    "sweep_speed_ms": 60,      # Rate of servo step
    "step_degree": 2,          # Angle resolution
    "min_angle": 0,
    "max_angle": 180,
    "current_angle": 0,
    "sweeping_forward": True,
    "sensor_beam_width": 6.0,  # Degrees of sonar dispersion
    "is_running": True
}

# WebSocket Manager to track connected browser clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed, we skip and let disconnect clean it
                pass

manager = ConnectionManager()

# Pydantic models for REST endpoints
class TargetModel(BaseModel):
    id: str
    name: str
    angle: float
    distance: float
    target_class: str # Rock, Fish, Submarine, Void

class SonarConfigModel(BaseModel):
    sweep_speed_ms: Optional[int] = None
    step_degree: Optional[int] = None
    sensor_beam_width: Optional[float] = None
    is_running: Optional[bool] = None

# --- BACKGROUND HARDWARE SIMULATOR ---
def run_sonar_hardware_loop(loop: asyncio.AbstractEventLoop):
    """
    Simulates the physical ESP32 sonar device sweep in a background thread.
    Calculates proximity, synthesizes high-frequency echo waveforms,
    executes Random Forest classification, and broadcasts telemetry packets.
    """
    logger.info("Starting background hardware sonar sweep simulator thread.")
    
    # Pre-load or train ML model
    if not classifier.load_model():
        logger.info("No saved ML model found. Initiating fast model auto-training...")
        classifier.train(samples_per_class=120)
        
    global sonar_config
    
    while True:
        if not sonar_config["is_running"]:
            time.sleep(0.5)
            continue
            
        angle = sonar_config["current_angle"]
        
        # 1. Physical Distance Scan - Find the closest virtual target aligned with the sensor beam
        detected_distance = -1.0
        detected_class = "Void"
        closest_distance = 999.0
        
        for target in virtual_targets.values():
            t_angle = target["angle"]
            t_dist = target["distance"]
            t_class = target["class"]
            
            # Check angular overlap (conical sound beam width)
            angle_diff = abs(angle - t_angle)
            if angle_diff <= (sonar_config["sensor_beam_width"] / 2.0):
                if t_dist < closest_distance:
                    closest_distance = t_dist
                    detected_distance = t_dist
                    detected_class = t_class
                    
        # 2. Physics wave synthesizer (Acoustic Propagation)
        t, wave = generator.generate_waveform(detected_class, detected_distance)
        
        # 3. Machine Learning Classification
        prediction_result = classifier.predict(t, wave)
        
        # 4. Format telemetry stream packet
        # Downsample waveform for smooth network transmission (take every 2nd point)
        downsample_factor = 2
        t_down = t[::downsample_factor].tolist()
        wave_down = wave[::downsample_factor].tolist()
        
        telemetry_packet = {
            "type": "telemetry",
            "angle": angle,
            "distance": float(detected_distance) if detected_distance > 0 else None,
            "waveform": wave_down,
            "time_vector": t_down,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"],
            "features": prediction_result["features"],
            "timestamp": int(time.time() * 1000)
        }
        
        # 5. Broadcast to all active websocket clients
        asyncio.run_coroutine_threadsafe(manager.broadcast(telemetry_packet), loop)
        
        # 6. Increment / Decrement servo motor sweep angle
        step = sonar_config["step_degree"]
        if sonar_config["sweeping_forward"]:
            sonar_config["current_angle"] += step
            if sonar_config["current_angle"] >= sonar_config["max_angle"]:
                sonar_config["current_angle"] = sonar_config["max_angle"]
                sonar_config["sweeping_forward"] = False
        else:
            sonar_config["current_angle"] -= step
            if sonar_config["current_angle"] <= sonar_config["min_angle"]:
                sonar_config["current_angle"] = sonar_config["min_angle"]
                sonar_config["sweeping_forward"] = True
                
        # Emulate servo motor physical stepping delay
        time.sleep(sonar_config["sweep_speed_ms"] / 1000.0)

# --- FASTAPI ENDPOINTS ---

@app.on_event("startup")
def startup_event():
    # Run the hardware simulator thread with access to the FastAPI event loop
    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=run_sonar_hardware_loop, args=(loop,), daemon=True)
    thread.start()

@app.get("/api/targets")
def get_targets():
    """Retrieve all virtual targets in the ocean space."""
    return list(virtual_targets.values())

@app.post("/api/targets")
def add_or_update_target(target: TargetModel):
    """Add a new virtual target or update an existing one."""
    virtual_targets[target.id] = {
        "id": target.id,
        "name": target.name,
        "angle": target.angle,
        "distance": target.distance,
        "class": target.target_class
    }
    logger.info(f"Target added/updated: {target.id} ({target.target_class})")
    return {"status": "success", "message": f"Target '{target.name}' updated."}

@app.delete("/api/targets/{target_id}")
def delete_target(target_id: str):
    """Remove a virtual target from the simulation."""
    if target_id in virtual_targets:
        del virtual_targets[target_id]
        logger.info(f"Target deleted: {target_id}")
        return {"status": "success", "message": "Target removed successfully."}
    return {"status": "error", "message": "Target not found."}, 404

@app.get("/api/config")
def get_config():
    """Retrieve current physical sonar configuration settings."""
    return sonar_config

@app.post("/api/config")
def update_config(config: SonarConfigModel):
    """Dynamically modify ESP32 hardware/sweep configurations."""
    global sonar_config
    if config.sweep_speed_ms is not None:
        sonar_config["sweep_speed_ms"] = config.sweep_speed_ms
    if config.step_degree is not None:
        sonar_config["step_degree"] = config.step_degree
    if config.sensor_beam_width is not None:
        sonar_config["sensor_beam_width"] = config.sensor_beam_width
    if config.is_running is not None:
        sonar_config["is_running"] = config.is_running
    return {"status": "success", "config": sonar_config}

@app.post("/api/model/train")
def train_model():
    """Trigger on-demand training of the Random Forest classifier model."""
    logger.info("On-demand model training requested.")
    results = classifier.train(samples_per_class=150)
    return {
        "status": "success",
        "validation_accuracy": results["validation_accuracy"],
        "confusion_matrix": results["confusion_matrix"],
        "feature_importances": classifier.model.feature_importances_.tolist() if classifier.model else []
    }

@app.get("/api/model/status")
def model_status():
    """Get the current training status of the machine learning classifier."""
    is_trained = classifier.model is not None or classifier.load_model()
    return {
        "is_trained": is_trained,
        "classes": classifier.classes,
        "model_file": classifier.model_path
    }

# --- WEBSOCKET FOR TELEMETRY STREAM ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket route that streams telemetry packets in real-time to the frontend."""
    await manager.connect(websocket)
    try:
        # Keep connection open and listen for any incoming command messages
        while True:
            data = await websocket.receive_text()
            # Handle messages from browser (e.g. configuration changes)
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception as e:
                logger.error(f"Error parsing client WebSocket message: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
