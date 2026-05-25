/**
 * AI Underwater Object Detection System - Cockpit Orchestrator
 * 
 * Sets up the coordination between all independent dashboard components:
 * - Standalone Loop: Synthesizes acoustic waves, runs feature extraction, and evaluates predictions.
 * - Live WebSockets: Binds API endpoints, processes telemetry frames, and parses server predictions.
 * - UI Control Panel binds, interactive mock console, and mock ESP32 sensor fluctuations.
 */

// Global variables for standalone waveform synthesis (parallel with python acoustics)
class JS_WaveformSynthesizer {
    constructor() {
        this.fs = 250000;          // 250 kHz sampling rate for browser performance
        this.f0 = 40000;           // 40 kHz pulse
        this.c = 1480;             // 1480 m/s sound speed in water
        this.limit_s = 0.008;      // 8 ms sweep window
        this.t = [];
        for (let i = 0; i < 300; i++) {
            this.t.push(i * (this.limit_s / 300));
        }
    }

    generateWaveform(targetClass, distanceMeters) {
        const len = this.t.length;
        const wave = new Float32Array(len);
        
        // 1. Ambient water noise floor
        const noiseFloor = 0.035;
        for (let i = 0; i < len; i++) {
            wave[i] = (Math.random() - 0.5) * noiseFloor * 2.0;
        }

        // 2. Add transmit leakage pulse (TX burst) at t=0
        const txDur = 0.0003;
        for (let i = 0; i < len; i++) {
            const time = this.t[i];
            const env = Math.exp(-Math.pow((time - 0.00015) / (txDur / 3), 2));
            wave[i] += 0.9 * env * Math.sin(2.0 * Math.PI * this.f0 * time);
        }

        // Return baseline noise if out of range or Void
        if (targetClass === "Void" || distanceMeters < 0.25 || distanceMeters > 5.6) {
            return { t: this.t, wave: wave };
        }

        // 3. Distance attenuation calculation
        const absorptionCoeff = 0.42;
        const attenuation = Math.exp(-absorptionCoeff * distanceMeters);
        
        // 4. Target Echo calculation (Time of Flight)
        const tof = (2.0 * distanceMeters) / this.c;
        const tofIdx = Math.floor((tof / this.limit_s) * len);

        if (tofIdx >= len) return { t: this.t, wave: wave };

        if (targetClass === "Submarine") {
            // Metallic Specular: sharp peak, clean envelope
            const dur = 0.0003;
            const amp = 0.72 * attenuation;
            const sigma = dur / 2.8;
            for (let i = 0; i < len; i++) {
                const time = this.t[i];
                const env = Math.exp(-Math.pow((time - tof) / sigma, 2));
                wave[i] += amp * env * Math.sin(2.0 * Math.PI * this.f0 * (time - tof));
            }
        } else if (targetClass === "Rock") {
            // Rugged diffuse: Wide dispersed double sub-peaks
            const dur = 0.0008;
            const amp = 0.38 * attenuation;
            
            const offsets = [-0.00015, 0.0001, 0.00028];
            const weights = [0.65, 0.95, 0.5];
            
            for (let k = 0; k < offsets.length; k++) {
                const subTof = tof + offsets[k];
                const sigma = (dur / 4.0);
                for (let i = 0; i < len; i++) {
                    const time = this.t[i];
                    if (time < 0 || time >= this.limit_s) continue;
                    const env = Math.exp(-Math.pow((time - subTof) / sigma, 2));
                    wave[i] += amp * weights[k] * env * Math.sin(2.0 * Math.PI * this.f0 * (time - subTof) + k * Math.PI/3);
                }
            }
        } else if (targetClass === "Fish") {
            // Soft scatter: Very weak swim bladder reflection
            const dur = 0.00045;
            const amp = 0.22 * attenuation;
            const sigma = dur / 3.0;
            for (let i = 0; i < len; i++) {
                const time = this.t[i];
                const env = Math.exp(-Math.pow((time - tof) / sigma, 2));
                wave[i] += amp * env * Math.sin(2.0 * Math.PI * this.f0 * (time - tof));
            }
        }

        return { t: this.t, wave: wave };
    }
}

// Global wave generator reference
window.EcoWaveformGenerator = JS_WaveformSynthesizer;

// Initialize components when DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    
    // Core Module Instantiations
    const env = new OceanEnvironment("canvas-env");
    const radar = new SonarRadar("canvas-radar");
    const analyzer = new SignalAnalyzer("canvas-wave-raw", "canvas-wave-fft");
    const edgeAI = new EdgeAIClassifier();
    const jsSynthesizer = new JS_WaveformSynthesizer();
    
    // Application States
    let activeMode = "STANDALONE"; // STANDALONE or LIVE
    let isScanning = true;
    let wsConnection = null;
    
    // Sonar sweep states
    let sweepAngle = 0;
    let sweepingForward = true;
    let lastTime = 0;
    let frameCount = 0;
    let fpsTime = 0;
    
    // UI elements caches
    const modeBadge = document.getElementById("mode-badge");
    const modeText = document.getElementById("mode-text");
    const apiStatus = document.getElementById("api-status");
    const pingVal = document.getElementById("ping-val");
    const fpsVal = document.getElementById("fps-val");
    
    const radarAngleVal = document.getElementById("radar-angle");
    const radarDistVal = document.getElementById("radar-dist");
    
    const rangeSweepSpeed = document.getElementById("range-sweep-speed");
    const rangeBeamWidth = document.getElementById("range-beam-width");
    
    const btnSweepToggle = document.getElementById("btn-sweep-toggle");
    const btnClearRadar = document.getElementById("btn-clear-radar");
    
    const mlPredText = document.getElementById("ml-prediction");
    const mlConfText = document.getElementById("ml-confidence-val");
    const mlConfBar = document.getElementById("ml-confidence-bar");
    
    const probSub = document.getElementById("prob-sub");
    const probRock = document.getElementById("prob-rock");
    const probFish = document.getElementById("prob-fish");
    const probVoid = document.getElementById("prob-void");
    
    const probSubVal = document.getElementById("prob-sub-val");
    const probRockVal = document.getElementById("prob-rock-val");
    const probFishVal = document.getElementById("prob-fish-val");
    const probVoidVal = document.getElementById("prob-void-val");
    
    const serialConsole = document.getElementById("serial-console-feed");
    const btnClearConsole = document.getElementById("btn-clear-console");
    
    // Extracted Features
    const featAmp = document.getElementById("feat-amp");
    const featTof = document.getElementById("feat-tof");
    const featEnergy = document.getElementById("feat-energy");
    const featDuration = document.getElementById("feat-duration");
    const featCentroid = document.getElementById("feat-centroid");
    const btnToggleFeatures = document.getElementById("btn-toggle-features");
    const featuresTable = document.getElementById("features-table-container");

    // --- TAB SWITCHER LOGIC ---
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // --- BTN SPOTS (Spawn targets in simulation environment) ---
    document.querySelectorAll(".btn-spawn").forEach(btn => {
        btn.addEventListener("click", () => {
            const targetType = btn.getAttribute("data-type");
            const id = env.spawnTarget(targetType);
            
            // Log to terminal console
            appendSerialLog(`[SYSTEM]: Created virtual ${targetType.toUpperCase()} target node [${id}] in ocean coordinates.`, "system");
        });
    });

    // --- TELEMETRY LOGGER HELPER ---
    function appendSerialLog(text, type = "input") {
        const line = document.createElement("div");
        line.className = `log-line ${type}`;
        line.textContent = text;
        serialConsole.appendChild(line);
        
        // Cap lines at 80
        while (serialConsole.childNodes.length > 80) {
            serialConsole.removeChild(serialConsole.firstChild);
        }
        // Auto scroll
        serialConsole.scrollTop = serialConsole.scrollHeight;
    }

    btnClearConsole.addEventListener("click", () => {
        serialConsole.innerHTML = "";
    });

    btnToggleFeatures.addEventListener("click", () => {
        featuresTable.classList.toggle("hidden");
    });

    // --- MOCK SYSTEM TELEMETRY FLUCTUATIONS ---
    function updateMockHardwareStats() {
        const heap = 190.0 + Math.random() * 8.0;
        const temp = 40.0 + Math.random() * 4.0;
        const voltage = 3.26 + Math.random() * 0.04;
        
        document.getElementById("esp-heap").textContent = `${heap.toFixed(1)} KB`;
        document.getElementById("esp-temp").textContent = `${temp.toFixed(1)}°C`;
        document.getElementById("esp-voltage").textContent = `${voltage.toFixed(2)} V`;
    }
    setInterval(updateMockHardwareStats, 3000);

    // --- RADAR RENDER LOOPS ---
    function mainBrowserSimulationLoop(timestamp) {
        if (activeMode !== "STANDALONE") return;
        
        // Calculate Frames per Second (FPS) metrics
        frameCount++;
        if (timestamp - fpsTime >= 1000) {
            fpsVal.textContent = frameCount;
            frameCount = 0;
            fpsTime = timestamp;
        }

        const sweepSpeedHz = parseFloat(rangeSweepSpeed.value);
        const beamWidth = parseFloat(rangeBeamWidth.value);
        
        if (isScanning) {
            // Frame-delta calculation
            const dt = timestamp - lastTime;
            lastTime = timestamp;
            
            // Adjust step degree scale factor based on milliseconds speed settings
            const step = (sweepSpeedHz * 0.08) * (dt / 16.6); // normalized to 60fps
            
            if (sweepingForward) {
                sweepAngle += step;
                if (sweepAngle >= 180) {
                    sweepAngle = 180;
                    sweepingForward = false;
                }
            } else {
                sweepAngle -= step;
                if (sweepAngle <= 0) {
                    sweepAngle = 0;
                    sweepingForward = true;
                }
            }

            // 1. Perform intersection check against targets in environment
            const scan = env.getScanResult(sweepAngle, beamWidth);
            
            // 2. Synthesize time-domain acoustic return waveform
            const synth = jsSynthesizer.generateWaveform(scan.class, scan.distance);
            
            // 3. Edge AI prediction model inference
            const prediction = edgeAI.predict(synth.t, synth.wave);
            
            // 4. Dispatch blips & waveforms to canvas render engines
            if (scan.distance > 0) {
                radar.addBlip(sweepAngle, scan.distance, scan.class, prediction.confidence);
                radarAngleVal.textContent = Math.round(sweepAngle);
                radarDistVal.textContent = scan.distance.toFixed(2);
            } else {
                radarAngleVal.textContent = Math.round(sweepAngle);
                radarDistVal.textContent = "0.00";
            }
            
            analyzer.renderWaveform(synth.t, synth.wave, scan.class);
            analyzer.renderFFT(synth.wave, scan.class);
            
            // 5. Update Web Interface layout components
            updateMLDashboardUI(prediction);
            
            // 6. Log dynamic mock hardware serialized outputs
            if (Math.round(sweepAngle) % 8 === 0) {
                const distanceVal = scan.distance > 0 ? `${scan.distance.toFixed(2)}m` : "VOID (OPEN WATER)";
                const tofVal = scan.distance > 0 ? `${Math.round(scan.distance * 1350)}µs` : "0µs";
                const logMsg = `[ESP32]: Angle: ${Math.round(sweepAngle)}° | Dist: ${distanceVal} | TOF: ${tofVal} | Class: ${prediction.class} (${Math.round(prediction.confidence * 100)}%)`;
                appendSerialLog(logMsg, scan.distance > 0 ? "success" : "input");
            }
        } else {
            lastTime = timestamp;
        }

        // Render environments
        env.render(sweepAngle, beamWidth);
        radar.render(sweepAngle, sweepingForward);
        
        requestAnimationFrame(mainBrowserSimulationLoop);
    }

    // --- ML DASHBOARD RENDER DISPATCH ---
    function updateMLDashboardUI(prediction) {
        mlPredText.textContent = prediction.class.toUpperCase();
        mlConfText.textContent = `${Math.round(prediction.confidence * 100)}%`;
        mlConfBar.style.width = `${Math.round(prediction.confidence * 100)}%`;
        
        // Color predict text dynamically based on threats
        mlPredText.style.color = 'var(--neon-cyan)';
        mlPredText.style.textShadow = '0 0 10px rgba(0, 240, 255, 0.4)';
        if (prediction.class === "Submarine") {
            mlPredText.style.color = 'var(--neon-red)';
            mlPredText.style.textShadow = '0 0 10px rgba(255, 51, 85, 0.4)';
        } else if (prediction.class === "Rock") {
            mlPredText.style.color = 'var(--neon-blue)';
            mlPredText.style.textShadow = '0 0 10px rgba(0, 136, 255, 0.4)';
        } else if (prediction.class === "Fish") {
            mlPredText.style.color = 'var(--neon-emerald)';
            mlPredText.style.textShadow = '0 0 10px rgba(0, 230, 118, 0.4)';
        }
        
        // Probability bars
        const probs = prediction.probabilities;
        probSub.style.width = `${Math.round(probs.Submarine * 100)}%`;
        probSubVal.textContent = `${Math.round(probs.Submarine * 100)}%`;
        
        probRock.style.width = `${Math.round(probs.Rock * 100)}%`;
        probRockVal.textContent = `${Math.round(probs.Rock * 100)}%`;
        
        probFish.style.width = `${Math.round(probs.Fish * 100)}%`;
        probFishVal.textContent = `${Math.round(probs.Fish * 100)}%`;
        
        probVoid.style.width = `${Math.round(probs.Void * 100)}%`;
        probVoidVal.textContent = `${Math.round(probs.Void * 100)}%`;
        
        // Extracted features labels
        const feats = prediction.features;
        featAmp.textContent = feats.peakAmp.toFixed(3);
        featTof.textContent = `${(feats.tof * 1000).toFixed(2)}ms`;
        featEnergy.textContent = feats.energy.toFixed(4);
        featDuration.textContent = `${(feats.duration * 1000).toFixed(2)}ms`;
        featCentroid.textContent = `${feats.centroid.toFixed(1)} kHz`;
    }

    // --- WEBSOCKET ENGINE (Python API Mode) ---
    function connectToPythonBackendWS() {
        if (wsConnection) {
            wsConnection.close();
        }
        
        appendSerialLog("[WEBSOCKET]: Initiating handshake connection to ws://127.0.0.1:8000/ws...", "system");
        
        wsConnection = new WebSocket("ws://127.0.0.1:8000/ws");
        let connectTimeout = setTimeout(() => {
            if (wsConnection.readyState !== WebSocket.OPEN) {
                appendSerialLog("[ERROR]: WebSocket handshake connection timed out. Server is offline.", "error");
                wsConnection.close();
                toggleMode("STANDALONE");
            }
        }, 3000);
        
        wsConnection.onopen = () => {
            clearTimeout(connectTimeout);
            appendSerialLog("[WEBSOCKET]: Handshake connection established! Receiving hardware stream.", "success");
            
            // Enable indicators
            apiStatus.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> Connected';
            apiStatus.className = "tel-val api-online";
            pingVal.textContent = "2ms";
            
            // Sync targets over REST
            syncTargetsWithBackend();
        };
        
        wsConnection.onmessage = (event) => {
            if (activeMode !== "LIVE") return;
            
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === "telemetry") {
                    const angle = data.angle;
                    const distance = data.distance;
                    const waveform = data.waveform;
                    const timeVector = data.time_vector;
                    
                    // Add radar sweep blips
                    if (distance) {
                        radar.addBlip(angle, distance, data.prediction, data.confidence);
                        radarAngleVal.textContent = angle;
                        radarDistVal.textContent = distance.toFixed(2);
                    } else {
                        radarAngleVal.textContent = angle;
                        radarDistVal.textContent = "0.00";
                    }
                    
                    // Render waveform scope and FFT
                    analyzer.renderWaveform(timeVector, waveform, data.prediction);
                    analyzer.renderFFT(waveform, data.prediction);
                    
                    // Update prediction panel labels
                    mlPredText.textContent = data.prediction.toUpperCase();
                    mlConfText.textContent = `${Math.round(data.confidence * 100)}%`;
                    mlConfBar.style.width = `${Math.round(data.confidence * 100)}%`;
                    
                    // Color predict text dynamically based on threats
                    mlPredText.style.color = 'var(--neon-cyan)';
                    mlPredText.style.textShadow = '0 0 10px rgba(0, 240, 255, 0.4)';
                    if (data.prediction === "Submarine") {
                        mlPredText.style.color = 'var(--neon-red)';
                        mlPredText.style.textShadow = '0 0 10px rgba(255, 51, 85, 0.4)';
                    } else if (data.prediction === "Rock") {
                        mlPredText.style.color = 'var(--neon-blue)';
                        mlPredText.style.textShadow = '0 0 10px rgba(0, 136, 255, 0.4)';
                    } else if (data.prediction === "Fish") {
                        mlPredText.style.color = 'var(--neon-emerald)';
                        mlPredText.style.textShadow = '0 0 10px rgba(0, 230, 118, 0.4)';
                    }
                    
                    const probs = data.probabilities;
                    probSub.style.width = `${Math.round(probs.Submarine * 100)}%`;
                    probSubVal.textContent = `${Math.round(probs.Submarine * 100)}%`;
                    
                    probRock.style.width = `${Math.round(probs.Rock * 100)}%`;
                    probRockVal.textContent = `${Math.round(probs.Rock * 100)}%`;
                    
                    probFish.style.width = `${Math.round(probs.Fish * 100)}%`;
                    probFishVal.textContent = `${Math.round(probs.Fish * 100)}%`;
                    
                    probVoid.style.width = `${Math.round(probs.Void * 100)}%`;
                    probVoidVal.textContent = `${Math.round(probs.Void * 100)}%`;
                    
                    const feats = data.features;
                    featAmp.textContent = feats.peak_amplitude.toFixed(3);
                    featTof.textContent = `${(feats.time_of_flight * 1000).toFixed(2)}ms`;
                    featEnergy.textContent = feats.energy.toFixed(4);
                    featDuration.textContent = `${(feats.echo_duration * 1000).toFixed(2)}ms`;
                    featCentroid.textContent = `${(feats.spectral_centroid / 1000).toFixed(1)} kHz`;
                    
                    // Draw environment sweep
                    env.render(angle, parseFloat(rangeBeamWidth.value));
                    radar.render(angle, sweepingForward);
                    
                    // Log packet to serial console
                    if (angle % 6 === 0) {
                        const distanceVal = distance ? `${distance.toFixed(2)}m` : "VOID (OPEN WATER)";
                        const tofVal = distance ? `${Math.round(feats.time_of_flight * 2000000.0)}µs` : "0µs";
                        const logMsg = `[ESP32-LIVE]: Angle: ${angle}° | Dist: ${distanceVal} | TOF: ${tofVal} | Prediction: ${data.prediction} (${Math.round(data.confidence * 100)}%)`;
                        appendSerialLog(logMsg, distance ? "success" : "input");
                    }
                }
            } catch (e) {
                console.error("Error parsing websocket message: ", e);
            }
        };
        
        wsConnection.onclose = () => {
            appendSerialLog("[WEBSOCKET]: Handshake connection terminated by remote host.", "error");
            apiStatus.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> Offline';
            apiStatus.className = "tel-val";
            pingVal.textContent = "--";
            if (activeMode === "LIVE") {
                toggleMode("STANDALONE");
            }
        };
        
        wsConnection.onerror = (err) => {
            appendSerialLog("[ERROR]: WebSocket channel experienced an internal protocol error.", "error");
        };
    }

    /**
     * POST virtual targets config down to the Python REST server so backend/frontend simulation match.
     */
    function syncTargetsWithBackend() {
        env.targets.forEach(t => {
            const data = {
                id: t.id,
                name: t.name,
                angle: t.angle,
                distance: t.dist,
                target_class: t.type
            };
            fetch("http://127.0.0.1:8000/api/targets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            }).catch(err => console.error("Error syncing target to backend:", err));
        });
    }

    // --- COCKPIT INTERACTION CONTROLS ---

    btnSweepToggle.addEventListener("click", () => {
        isScanning = !isScanning;
        if (isScanning) {
            btnSweepToggle.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Scan';
            btnSweepToggle.className = "btn btn-primary";
            appendSerialLog("[SYSTEM]: Ultrasonic scanning sweep resumed.", "system");
        } else {
            btnSweepToggle.innerHTML = '<i class="fa-solid fa-play"></i> Resume Scan';
            btnSweepToggle.className = "btn btn-accent";
            appendSerialLog("[SYSTEM]: Ultrasonic scanning sweep paused.", "system");
        }
        
        // Sync configuration settings down to live Python server if online
        if (activeMode === "LIVE") {
            fetch("http://127.0.0.1:8000/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ is_running: isScanning })
            }).catch(e => {});
        }
    });

    btnClearRadar.addEventListener("click", () => {
        radar.clearTrails();
        appendSerialLog("[SYSTEM]: Sonar display phosphor memory flushed.", "system");
    });

    rangeSweepSpeed.addEventListener("input", () => {
        const val = parseFloat(rangeSweepSpeed.value);
        if (activeMode === "LIVE") {
            // Convert Hz to esp32 step delay (e.g. 60Hz sweep frequency maps to ~20ms delay)
            const delay = Math.round(1000 / val);
            fetch("http://127.0.0.1:8000/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sweep_speed_ms: delay })
            }).catch(e => {});
        }
    });

    // --- DYNAMIC MODE TIE TOGGLE ---
    const btnSimMode = document.getElementById("btn-sim-mode");
    const btnLiveMode = document.getElementById("btn-live-mode");

    function toggleMode(mode) {
        if (mode === "STANDALONE") {
            activeMode = "STANDALONE";
            btnSimMode.classList.add("active");
            btnLiveMode.classList.remove("active");
            
            modeBadge.className = "status-indicator";
            modeText.textContent = "STANDALONE SIMULATION";
            document.getElementById("ml-engine-badge").textContent = "JS NN Core";
            document.getElementById("ml-engine-badge").className = "badge badge-accent";
            
            appendSerialLog("[SYSTEM]: Cockpit toggled to STANDALONE BROWSER ENGINE.", "system");
            
            if (wsConnection) {
                wsConnection.close();
            }
            
            // Kickstart animation loop
            lastTime = performance.now();
            requestAnimationFrame(mainBrowserSimulationLoop);
        } else {
            activeMode = "LIVE";
            btnLiveMode.classList.add("active");
            btnSimMode.classList.remove("active");
            
            modeBadge.className = "status-indicator live";
            modeText.textContent = "LIVE PY-API STREAM";
            document.getElementById("ml-engine-badge").textContent = "Py Scikit-RF";
            document.getElementById("ml-engine-badge").className = "badge badge-green";
            
            appendSerialLog("[SYSTEM]: Cockpit toggled to LIVE PYTHON BACKEND BRIDGE.", "system");
            connectToPythonBackendWS();
        }
    }

    btnSimMode.addEventListener("click", () => toggleMode("STANDALONE"));
    btnLiveMode.addEventListener("click", () => toggleMode("LIVE"));

    // --- IN-BROWSER AI NEURAL NETWORK TRAINING ---
    const btnTrainModel = document.getElementById("btn-train-model");
    const trainStatusText = document.getElementById("train-status");
    const trainAccuracyText = document.getElementById("train-accuracy");
    const trainLossText = document.getElementById("train-loss");
    const trainEpochText = document.getElementById("train-epochs");

    btnTrainModel.addEventListener("click", async () => {
        btnTrainModel.disabled = true;
        btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
        appendSerialLog("[AI WORKBENCH]: Training client-side 3-layer MLP Neural Network on 400 simulated waveforms...", "system");
        
        trainStatusText.textContent = "Training...";
        trainStatusText.className = "glow-text-cyan";
        
        try {
            if (activeMode === "STANDALONE") {
                // Train browser neural network
                await edgeAI.trainModelInBrowser((epoch, total, loss, acc) => {
                    trainAccuracyText.textContent = `${(acc * 100).toFixed(1)}%`;
                    trainLossText.textContent = loss.toFixed(4);
                    trainEpochText.textContent = `${epoch}/${total}`;
                });
                
                appendSerialLog("[AI WORKBENCH]: Neural Network training completed! Weights optimized.", "success");
                trainStatusText.textContent = "Trained";
                trainStatusText.className = "glow-text-green";
            } else {
                // Trigger model training over REST
                appendSerialLog("[AI WORKBENCH]: Sending REST command to trigger scikit-learn Random Forest training...", "system");
                const response = await fetch("http://127.0.0.1:8000/api/model/train", { method: "POST" });
                const res = await response.json();
                
                trainAccuracyText.textContent = `${(res.validation_accuracy * 100).toFixed(1)}%`;
                trainLossText.textContent = "N/A (RF)";
                trainEpochText.textContent = "N/A (RF)";
                
                // Redraw confusion matrix using Python results
                edgeAI.renderConfusionMatrixGrid(res.confusion_matrix);
                
                appendSerialLog("[AI WORKBENCH]: Python scikit-learn Random Forest training complete! Accuracy synchronized.", "success");
                trainStatusText.textContent = "Synced";
                trainStatusText.className = "glow-text-green";
            }
        } catch (e) {
            console.error("Training error: ", e);
            appendSerialLog("[ERROR]: ML training aborted due to computation exception.", "error");
            trainStatusText.textContent = "Failed";
            trainStatusText.className = "log-line error";
        }
        
        btnTrainModel.disabled = false;
        btnTrainModel.innerHTML = '<i class="fa-solid fa-graduation-cap"></i> Train Model';
    });

    // --- RUNNING START ---
    // Start by default in Standalone browser simulation mode
    lastTime = performance.now();
    requestAnimationFrame(mainBrowserSimulationLoop);
    appendSerialLog("[SYSTEM]: Cockpit initialized successfully. System status healthy.", "success");
});
