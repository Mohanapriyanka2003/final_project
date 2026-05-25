import numpy as np
import scipy.stats as stats
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import EcoWaveformGenerator

class SonarClassifier:
    """
    Extracts high-level acoustic features from ultrasonic waveforms 
    and trains a Random Forest Classifier to identify target materials.
    """
    def __init__(self, model_path="sonar_rf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.classes = ["Void", "Rock", "Fish", "Submarine"]
        self.generator = EcoWaveformGenerator()
        
    def extract_features(self, t, wave):
        """
        Extracts 7 descriptive temporal and spectral features from a single 1D waveform.
        Features target the post-transmit pulse window (t > 0.0008s) to ignore direct leakage.
        """
        fs = 500000  # 500 kHz sampling rate
        # 1. Focus on the region after the initial TX pulse leakage (ignore first 0.8ms)
        mask = t > 0.0008
        t_window = t[mask]
        wave_window = wave[mask]
        
        if len(wave_window) == 0:
            return np.zeros(7)
            
        # Feature 1: Peak Amplitude (Absolute max)
        peak_amp = np.max(np.abs(wave_window))
        
        # Feature 2: Time of Flight / Peak Position (Estimates target distance)
        peak_idx = np.argmax(np.abs(wave_window))
        tof_val = t_window[peak_idx]
        
        # Feature 3: Signal Envelope Energy
        # Computes the Hilbert transform analytic signal envelope to get overall reflection energy
        analytic_signal = wave_window + 1j * np.zeros(len(wave_window)) # simplified envelope
        try:
            from scipy.signal import hilbert
            analytic_signal = hilbert(wave_window)
        except Exception:
            pass # fallback if hilbert fails
        envelope = np.abs(analytic_signal)
        total_energy = np.sum(envelope ** 2) / len(envelope)
        
        # Feature 4: Echo Duration (width)
        # Measure number of samples where the envelope exceeds 3x ambient noise floor (0.12)
        threshold = 0.12
        above_threshold_indices = np.where(envelope > threshold)[0]
        if len(above_threshold_indices) > 1:
            echo_width = (above_threshold_indices[-1] - above_threshold_indices[0]) / fs
        else:
            echo_width = 0.0
            
        # Feature 5: Spectral Centroid
        # Renders the balance point of the frequency spectrum (identifies material impedance shifts)
        fft_vals = np.fft.rfft(wave_window)
        fft_freqs = np.fft.rfftfreq(len(wave_window), 1.0 / fs)
        fft_power = np.abs(fft_vals) ** 2
        
        sum_power = np.sum(fft_power)
        if sum_power > 1e-6:
            spectral_centroid = np.sum(fft_freqs * fft_power) / sum_power
        else:
            spectral_centroid = 0.0
            
        # Feature 6: Signal Kurtosis (envelope peak sharpness)
        kurtosis_val = stats.kurtosis(wave_window) if len(wave_window) > 4 else 0.0
        
        # Feature 7: Signal Skewness (envelope asymmetry)
        skewness_val = stats.skew(wave_window) if len(wave_window) > 4 else 0.0
        
        return np.array([
            peak_amp,            # Feature 1
            tof_val,             # Feature 2
            total_energy,        # Feature 3
            echo_width,          # Feature 4
            spectral_centroid,   # Feature 5
            kurtosis_val,        # Feature 6
            skewness_val         # Feature 7
        ])

    def generate_training_data(self, samples_per_class=200):
        """
        Creates a synthetic training dataset by sweeping through random distances
        and target classes to extract feature vectors.
        """
        X = []
        y = []
        
        print("Generating training dataset...")
        for class_label in self.classes:
            for _ in range(samples_per_class):
                # Random target distance between 0.5m and 5.0m
                dist = np.random.uniform(0.5, 5.0)
                
                # Generate waveform
                t, wave = self.generator.generate_waveform(class_label, dist)
                
                # Extract feature vector
                features = self.extract_features(t, wave)
                
                X.append(features)
                y.append(self.classes.index(class_label))
                
        return np.array(X), np.array(y)
        
    def train(self, samples_per_class=200):
        """
        Trains the Random Forest model and saves it to a persistent pickle file.
        """
        X, y = self.generate_training_data(samples_per_class)
        
        # Split into training & validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Train Random Forest Classifier
        print("Training Random Forest Classifier model...")
        self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        print(f"Training Accuracy: {train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {test_acc * 100:.2f}%")
        
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")
        
        return {
            "validation_accuracy": test_acc,
            "confusion_matrix": conf_mat.tolist(),
            "report": report
        }
        
    def load_model(self):
        """
        Loads the pre-trained Random Forest model from disk.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False
        
    def predict(self, t, wave):
        """
        Predicts the class and probability distribution of an input waveform.
        """
        if self.model is None:
            if not self.load_model():
                # Auto-train if model doesn't exist
                self.train(100)
                
        features = self.extract_features(t, wave)
        features_reshaped = features.reshape(1, -1)
        
        class_idx = self.model.predict(features_reshaped)[0]
        probs = self.model.predict_proba(features_reshaped)[0]
        
        return {
            "prediction": self.classes[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "features": {
                "peak_amplitude": float(features[0]),
                "time_of_flight": float(features[1]),
                "energy": float(features[2]),
                "echo_duration": float(features[3]),
                "spectral_centroid": float(features[4]),
                "kurtosis": float(features[5]),
                "skewness": float(features[6])
            }
        }

if __name__ == "__main__":
    classifier = SonarClassifier("test_model.pkl")
    metrics = classifier.train(150)
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    
    # Run test prediction
    test_gen = EcoWaveformGenerator()
    t, wave = test_gen.generate_waveform("Submarine", 3.2)
    res = classifier.predict(t, wave)
    print("\nTest Prediction on Submarine at 3.2m:")
    print(f"Class: {res['prediction']}, Confidence: {res['confidence'] * 100:.1f}%")
