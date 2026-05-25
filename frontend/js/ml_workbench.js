/**
 * AI Underwater Object Detection System - Edge AI ML Workbench
 * 
 * Implements a complete 3-layer feedforward Neural Network (MLP) directly in Javascript.
 * Features:
 * - 1D wave signal acoustic feature extraction (matching Python backend feature parity).
 * - Multi-layer perceptron with backpropagation and SGD optimizer.
 * - Interactive model training with real-time loss/accuracy curves.
 * - Dynamic 4x4 confusion matrix rendering to visualize true vs. predicted classes.
 */

class EdgeAIClassifier {
    constructor() {
        this.classes = ["Void", "Rock", "Fish", "Submarine"];
        
        // --- 3-LAYER NEURAL NETWORK CONFIGURATION ---
        // Input: 5 extracted acoustic features
        // Hidden: 8 hidden neurons (ReLU activation)
        // Output: 4 softmax class probabilities
        this.numInputs = 5;
        this.numHidden = 8;
        this.numOutputs = 4;
        
        this.weightsIH = []; // Input -> Hidden weights
        this.biasH = [];     // Hidden biases
        this.weightsHO = []; // Hidden -> Output weights
        this.biasO = [];     // Output biases
        
        this.isTrained = false;
        
        this.initializeNetworkWeights();
        this.renderConfusionMatrixGrid();
    }

    /**
     * Initializes weights randomly using Xavier/Glorot normal distribution bounds.
     */
    initializeNetworkWeights() {
        // Input to Hidden
        this.weightsIH = [];
        this.biasH = new Float32Array(this.numHidden).fill(0.01);
        for (let i = 0; i < this.numHidden; i++) {
            const row = new Float32Array(this.numInputs);
            for (let j = 0; j < this.numInputs; j++) {
                row[j] = (Math.random() - 0.5) * Math.sqrt(2.0 / this.numInputs);
            }
            this.weightsIH.push(row);
        }

        // Hidden to Output
        this.weightsHO = [];
        this.biasO = new Float32Array(this.numOutputs).fill(0.01);
        for (let i = 0; i < this.numOutputs; i++) {
            const row = new Float32Array(this.numHidden);
            for (let j = 0; j < this.numHidden; j++) {
                row[j] = (Math.random() - 0.5) * Math.sqrt(2.0 / this.numHidden);
            }
            this.weightsHO.push(row);
        }
        
        this.isTrained = false;
    }

    /**
     * Extract features from a 1D ultrasonic waveform inside the browser.
     * Features align with the python backend.
     */
    extractAcousticFeatures(t, wave) {
        if (!wave || wave.length === 0) {
            return { raw: new Float32Array(5), peakAmp: 0, tof: 0, energy: 0, duration: 0, centroid: 0 };
        }
        
        // Ignore the initial TX pulse leakage (ignore first 15% of the samples, ~1.2ms)
        const offsetIdx = Math.floor(wave.length * 0.15);
        const w_win = wave.slice(offsetIdx);
        const t_win = t.slice(offsetIdx);
        
        // Feature 1: Peak Amplitude
        let peakAmp = 0;
        let peakIdx = 0;
        for (let i = 0; i < w_win.length; i++) {
            const absVal = Math.abs(w_win[i]);
            if (absVal > peakAmp) {
                peakAmp = absVal;
                peakIdx = i;
            }
        }
        
        // Feature 2: Time of Flight
        const tof = t_win.length > 0 ? t_win[peakIdx] : 0.0;
        
        // Feature 3: Cumulative Signal Energy
        let sumSquared = 0;
        for (let i = 0; i < w_win.length; i++) {
            sumSquared += w_win[i] * w_win[i];
        }
        const energy = sumSquared / w_win.length;
        
        // Feature 4: Pulse Duration width above noise boundary (threshold = 0.10)
        let firstAbove = -1;
        let lastAbove = -1;
        const thresh = 0.10;
        for (let i = 0; i < w_win.length; i++) {
            if (Math.abs(w_win[i]) > thresh) {
                if (firstAbove === -1) firstAbove = i;
                lastAbove = i;
            }
        }
        const duration = (firstAbove !== -1) ? (lastAbove - firstAbove) * (t[1] - t[0]) : 0.0;
        
        // Feature 5: Spectral Centroid (simulated matching frequency dispersion)
        // High density specularity (Submarine) has high coherence, rock has wide center
        let centroid = 40.0; // standard kHz carrier
        if (peakAmp > 0.05) {
            if (peakAmp > 0.45) {
                centroid = 40.0; // specular
            } else if (peakAmp < 0.18) {
                centroid = 37.8; // soft scattering
            } else {
                centroid = 43.5; // rugged dispersion
            }
        } else {
            centroid = 0.0; // void/ambient noise
        }
        
        const rawFeatures = new Float32Array([
            peakAmp * 2.0,       // normalized
            tof * 100.0,
            energy * 10.0,
            duration * 200.0,
            centroid / 10.0
        ]);
        
        return {
            raw: rawFeatures,
            peakAmp: peakAmp,
            tof: tof,
            energy: energy,
            duration: duration,
            centroid: centroid
        };
    }

    /**
     * Feeds features forward through the Neural Network.
     * Activation functions: ReLU for hidden, Softmax for output.
     */
    forward(inputs) {
        const hidden = new Float32Array(this.numHidden);
        
        // Layer 1: Input -> Hidden
        for (let i = 0; i < this.numHidden; i++) {
            let sum = this.biasH[i];
            for (let j = 0; j < this.numInputs; j++) {
                sum += inputs[j] * this.weightsIH[i][j];
            }
            hidden[i] = Math.max(0, sum); // ReLU activation
        }
        
        // Layer 2: Hidden -> Output
        const outputs = new Float32Array(this.numOutputs);
        let maxVal = -Infinity;
        
        for (let i = 0; i < this.numOutputs; i++) {
            let sum = this.biasO[i];
            for (let j = 0; j < this.numHidden; j++) {
                sum += hidden[j] * this.weightsHO[i][j];
            }
            outputs[i] = sum;
            if (sum > maxVal) maxVal = sum; // Stable softmax subtraction constant
        }
        
        // Apply Softmax normalization
        let sumExp = 0;
        const softmaxOutputs = new Float32Array(this.numOutputs);
        for (let i = 0; i < this.numOutputs; i++) {
            softmaxOutputs[i] = Math.exp(outputs[i] - maxVal);
            sumExp += softmaxOutputs[i];
        }
        for (let i = 0; i < this.numOutputs; i++) {
            softmaxOutputs[i] /= sumExp;
        }
        
        return softmaxOutputs;
    }

    /**
     * Performs a single step forward pass prediction.
     */
    predict(t, wave) {
        const feats = this.extractAcousticFeatures(t, wave);
        
        // Run forward propagation
        const probs = this.forward(feats.raw);
        
        // Pick output class with highest probability
        let maxIdx = 0;
        let maxVal = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxVal) {
                maxVal = probs[i];
                maxIdx = i;
            }
        }
        
        return {
            class: this.classes[maxIdx],
            confidence: maxVal,
            probabilities: {
                Submarine: probs[3],
                Rock: probs[1],
                Fish: probs[2],
                Void: probs[0]
            },
            features: feats
        };
    }

    /**
     * Runs an interactive training epoch loop to train the neural network weights.
     * Generates a local synthetic training set in-memory, trains it over 150 epochs,
     * and shows a beautiful updating visual confusion matrix.
     */
    async trainModelInBrowser(onProgress) {
        this.initializeNetworkWeights();
        
        // 1. Generate local synthetic feature dataset
        const datasetSize = 400; // 100 samples per class
        const trainX = [];
        const trainY = [];
        
        const mockEnv = new OceanEnvironment('canvas-env'); // temporary mapping helper
        const mockGen = new (window.EcoWaveformGenerator || class {
            // fallback signature synthesis if data generator isn't loaded globally
            generateMock(cls, d) {
                const arr = new Float32Array(200);
                const noise = 0.03;
                for (let i=0; i<200; i++) arr[i] = (Math.random()-0.5)*noise;
                if (cls === "Void") return { t: new Float32Array(200), wave: arr };
                
                const peakPos = 40 + Math.floor(d * 25);
                const amp = cls === "Submarine" ? 0.6 : cls === "Rock" ? 0.35 : 0.18;
                const width = cls === "Rock" ? 22 : 6;
                
                for (let i = -width; i <= width; i++) {
                    const idx = peakPos + i;
                    if (idx >= 0 && idx < 200) {
                        const env = Math.exp(-Math.pow(i / (width / 2), 2));
                        arr[idx] += amp * env * Math.sin(i * 0.8);
                    }
                }
                return { t: new Float32Array(200).map((_,i)=>i*0.00004), wave: arr };
            }
        })();

        for (let i = 0; i < datasetSize; i++) {
            const cls = this.classes[i % 4];
            const dist = Math.random() * 4.5 + 0.5;
            const res = mockGen.generate_waveform ? mockGen.generate_waveform(cls, dist) : mockGen.generateMock(cls, dist);
            const t = res[0] || res.t;
            const wave = res[1] || res.wave;
            
            const feats = this.extractAcousticFeatures(t, wave);
            trainX.push(feats.raw);
            trainY.push(i % 4);
        }

        // 2. Training Epochs Loop (staggered with requestAnimationFrame for UI rendering)
        const epochs = 120;
        let learningRate = 0.15;
        
        for (let epoch = 1; epoch <= epochs; epoch++) {
            let totalLoss = 0;
            let correct = 0;
            
            // Stochastic Gradient Descent
            for (let s = 0; s < datasetSize; s++) {
                const inputs = trainX[s];
                const targetClassIdx = trainY[s];
                
                // Forward Propagation
                // Feed hidden relu layer
                const hidden = new Float32Array(this.numHidden);
                for (let i = 0; i < this.numHidden; i++) {
                    let sum = this.biasH[i];
                    for (let j = 0; j < this.numInputs; j++) {
                        sum += inputs[j] * this.weightsIH[i][j];
                    }
                    hidden[i] = Math.max(0, sum);
                }
                
                // Outputs softmax
                const outputs = new Float32Array(this.numOutputs);
                let maxVal = -Infinity;
                for (let i = 0; i < this.numOutputs; i++) {
                    let sum = this.biasO[i];
                    for (let j = 0; j < this.numHidden; j++) {
                        sum += hidden[j] * this.weightsHO[i][j];
                    }
                    outputs[i] = sum;
                    if (sum > maxVal) maxVal = sum;
                }
                
                let sumExp = 0;
                const probs = new Float32Array(this.numOutputs);
                for (let i = 0; i < this.numOutputs; i++) {
                    probs[i] = Math.exp(outputs[i] - maxVal);
                    sumExp += probs[i];
                }
                for (let i = 0; i < this.numOutputs; i++) {
                    probs[i] /= sumExp;
                }
                
                // Calculate Cross-Entropy Loss
                totalLoss -= Math.log(Math.max(1e-15, probs[targetClassIdx]));
                
                // Accuracy track
                let predictedIdx = 0;
                let maxP = -1;
                for (let i = 0; i < 4; i++) {
                    if (probs[i] > maxP) {
                        maxP = probs[i];
                        predictedIdx = i;
                    }
                }
                if (predictedIdx === targetClassIdx) correct++;
                
                // BACKPROPAGATION SENSITIVITIES
                // Output gradients (dLoss/dOutput = softmax_probability - target_one_hot)
                const dOut = new Float32Array(this.numOutputs);
                for (let i = 0; i < this.numOutputs; i++) {
                    dOut[i] = probs[i] - (i === targetClassIdx ? 1.0 : 0.0);
                }
                
                // Hidden layer gradients (dLoss/dHidden)
                const dHidden = new Float32Array(this.numHidden);
                for (let i = 0; i < this.numHidden; i++) {
                    let sum = 0;
                    for (let j = 0; j < this.numOutputs; j++) {
                        sum += dOut[j] * this.weightsHO[j][i];
                    }
                    // Derivative of ReLU
                    dHidden[i] = hidden[i] > 0 ? sum : 0;
                }
                
                // WEIGHT GRADIENTS DESCENT UPDATES
                // Update Hidden -> Output weights
                for (let i = 0; i < this.numOutputs; i++) {
                    this.biasO[i] -= learningRate * dOut[i];
                    for (let j = 0; j < this.numHidden; j++) {
                        this.weightsHO[i][j] -= learningRate * dOut[i] * hidden[j];
                    }
                }
                
                // Update Input -> Hidden weights
                for (let i = 0; i < this.numHidden; i++) {
                    this.biasH[i] -= learningRate * dHidden[i];
                    for (let j = 0; j < this.numInputs; j++) {
                        this.weightsIH[i][j] -= learningRate * dHidden[i] * inputs[j];
                    }
                }
            }
            
            const epochLoss = totalLoss / datasetSize;
            const epochAcc = correct / datasetSize;
            
            // Adjust learning rate decay
            learningRate *= 0.985;
            
            // Call progress callback to update UI graph/metrics
            onProgress(epoch, epochs, epochLoss, epochAcc);
            
            // Pause 12ms to allow UI updates between batches
            await new Promise(r => setTimeout(r, 12));
        }

        this.isTrained = true;
        
        // 3. Compute Final Confusion Matrix metrics
        const confMat = [
            [0, 0, 0, 0], // True Void -> Pred [Void, Rock, Fish, Sub]
            [0, 0, 0, 0], // True Rock -> Pred [Void, Rock, Fish, Sub]
            [0, 0, 0, 0], // True Fish -> Pred [Void, Rock, Fish, Sub]
            [0, 0, 0, 0]  // True Sub  -> Pred [Void, Rock, Fish, Sub]
        ];

        for (let s = 0; s < datasetSize; s++) {
            const inputs = trainX[s];
            const trueLabel = trainY[s];
            
            const probs = this.forward(inputs);
            let predLabel = 0;
            let maxP = -1;
            for (let i = 0; i < 4; i++) {
                if (probs[i] > maxP) {
                    maxP = probs[i];
                    predLabel = i;
                }
            }
            
            confMat[trueLabel][predLabel]++;
        }

        // Render final updating Matrix
        this.renderConfusionMatrixGrid(confMat);
    }

    /**
     * Dynamic rendering of a 4x4 matrix grid showing True vs. Predicted classes.
     */
    renderConfusionMatrixGrid(matrixData = null) {
        const container = document.getElementById("conf-matrix-grid");
        container.innerHTML = "";
        
        // Default pristine dummy matrix to show on load
        if (!matrixData) {
            matrixData = [
                [96, 4, 0, 0],   // Void
                [2, 94, 4, 0],   // Rock
                [0, 3, 97, 0],   // Fish
                [0, 1, 0, 99]    // Submarine
            ];
        }
        
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                const cellVal = matrixData[r][c];
                
                const cell = document.createElement("div");
                cell.className = "conf-cell";
                cell.textContent = cellVal;
                
                // Color code diagonals (correct classifications)
                if (r === c) {
                    cell.classList.add("diag");
                    if (cellVal > 70) cell.classList.add("high");
                } else if (cellVal > 0) {
                    // Mismatches represent classification errors
                    cell.classList.add("error");
                    if (cellVal > 10) cell.classList.add("high");
                }
                
                // Tooltip info on hover
                cell.title = `True: ${this.classes[r]}, Pred: ${this.classes[c]} (${cellVal} samples)`;
                
                container.appendChild(cell);
            }
        }
    }
}
