# AI-Enabled Underwater Object Detection System

An ESP32-based underwater sensing and monitoring system designed for real-time sonar mapping, acoustic signature tracking, and machine learning-based object classification using ultrasonic echo waveforms.

This repository implements a complete end-to-end hardware-software pipeline featuring:
1. **ESP32 Arduino Firmware (`esp32_firmware/`):** Controls a rotating servo motor to sweep an ultrasonic sensor (HC-SR04/JSN-SR04T) underwater, reads raw echo returns, filters signal noise, and streams telemetry packets in JSON over USB Serial.
2. **Python Analytics Suite (`backend/`):** Features a high-fidelity synthetic wave generator (simulating acoustic damping, marine noise, and material-specific scattering), feature extraction algorithms, and a scikit-learn Random Forest model.
3. **Command Cockpit Dashboard (`frontend/`):** A breathtaking, dark-themed, glassmorphic cockpit built with HTML5, CSS3, and vanilla JS that works **completely offline** as a standalone browser simulation and bridges to the live Python server over WebSockets.

---

## Cockpit Dashboard Architecture & UI Features

```
underwater_object_detection/
├── README.md                          # Project overview and quick-start instructions
├── esp32_firmware/
│   └── esp32_sonar_system.ino        # ESP32 C++ firmware with servo sonar sweep & serial telemetry
├── backend/
│   ├── requirements.txt               # Python package dependencies
│   ├── app.py                         # FastAPI web server with WebSockets
│   ├── ml_classifier.py               # Feature extraction, scikit-learn training, and inference
│   └── data_generator.py              # Generates synthetic ultrasonic echo datasets (1D waveforms)
└── frontend/
    ├── index.html                     # Breathtaking, glassmorphic dashboard interface
    ├── css/
    │   └── style.css                  # Theme variables, layouts, radar animations, neon effects
    └── js/
        ├── app.js                     # Main application flow, mode selection, and backend state
        ├── environment.js             # Physics-based virtual ocean, target spawning, and current simulation
        ├── sonar.js                   # Canvas radial sweep renderer with decaying blips & vector track history
        ├── oscilloscope.js            # Dual-channel Canvas oscilloscope: Raw signal trace & FFT spectral graph
        └── ml_workbench.js            # JS Neural Network, feature extractor, and confusion matrix renderer
```

* **Radial Sonar Sweep (Canvas):** Implements a sweeping radial line mimicking classic military radars. Detections leave fading phosphor glowing blips that slowly decay and track historical movement vectors.
* **Dual-Channel Signal Analyzer (Canvas):**
  * **Waveform Oscilloscope:** Draws the raw 1D ultrasonic echo voltage wave, showcasing the transmit leak burst, background noise floor, and reflective target signature envelope.
  * **FFT Power Spectrum:** Performs real-time Fourier analysis to display the frequency distribution. Hard specularity (submarines) renders sharp peaks at 40kHz, while diffuse scatterers (rocks) exhibit broad, multi-frequency dispersion.
* **Virtual Ocean Sandbox:** A top-down click-and-drag physics simulator containing active current particles and bubble waves. Users can spawn, drag, and position targets (Submarines, Rocks, Fish shoals) to see the active sonar beam track them.
* **Edge AI Workbench:** Binds a lightweight, 3-layer Multi-Layer Perceptron (MLP) Neural Network written from scratch in vanilla Javascript. Users can click "Train Model" to watch in-browser backpropagation converge (reducing cross-entropy loss, expanding accuracy) and compute a dynamic 4x4 confusion matrix.
* **ESP32 Console Monitor:** Emulates ESP32 microchip status (CPU temperature, Vcc supply, heap memory, RSSI) and logs raw serial data blocks in a retro console window.

---

## Getting Started

You can run this entire application immediately in two different configurations:

### Option A: Standalone Simulator (Zero Setup - Web Browser)
To run the high-fidelity simulator immediately inside your browser without installing any packages or hardware:
1. Locate the `frontend/index.html` file:
   [frontend/index.html](file:///C:/Users/monakarni/.gemini/antigravity/scratch/underwater_object_detection/frontend/index.html)
2. Double-click the file to open it in Google Chrome, Microsoft Edge, or Mozilla Firefox.
3. Choose the **Browser Sim** toggle in the upper header.
4. Interact with the sandbox: drag targets around, spawn new ones, adjust the sweep speed, or click **Train Model** to initiate client-side neural network training!

---

### Option B: Live Python Backend & Mock Hardware Stream (Server Bridge)
To spin up the Python analytics server and run scikit-learn classification models:

#### 1. Setup Python Environment
Ensure Python 3.8+ is installed on your system. Open a terminal and run:
```powershell
# Navigate to the backend directory
cd C:\Users\monakarni\.gemini\antigravity\scratch\underwater_object_detection\backend

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

#### 2. Launch the FastAPI WebSocket Server
To start the mock hardware sweep loop and backend API:
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Once launched, the FastAPI server will spin up a background thread mimicking a physical ESP32 sonar sweep stepping through angles, generating synthetic signals, and broadcasting telemetry.

#### 3. Access and Connect the Dashboard
1. Open your web browser and go to:
   **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
   The FastAPI server will automatically serve the glassmorphic cockpit dashboard, completely eliminating any CORS or local file loading issues!
2. In the upper-right header of the dashboard, click **Live Py-API**.
3. The dashboard will instantly handshake over WebSockets (`ws://127.0.0.1:8000/ws`).
4. The system status badge will change to green `LIVE PY-API STREAM`, and you will see mock serial packets streaming into the ESP32 Hardware Console in real-time, backed by scikit-learn Random Forest model classifications.


---

## Physical Hardware Deployment (ESP32 & Transducer)

To transition from the virtual hardware simulation to a physical underwater rig:
1. **Flashing the Chip:** Open `esp32_firmware/esp32_sonar_system.ino` in the Arduino IDE or VS Code PlatformIO.
2. **Library Installation:** Ensure you have the `ESP32Servo` library installed in your Arduino package manager.
3. **Connections:** Hook up an SG90 servo motor (GPIO 18), and an HC-SR04/JSN-SR04T ultrasonic trigger (GPIO 5) and echo (GPIO 17).
4. **Data Bridge:** Connect the ESP32 to your PC via USB. Set the backend uvicorn server COM port configuration to read from serial, and it will capture the real-time physical telemetry stream and classify physical objects in your water tank!
