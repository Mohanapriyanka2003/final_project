import numpy as np
import scipy.signal as signal
import math

class EcoWaveformGenerator:
    """
    High-fidelity synthetic ultrasonic echo waveform generator.
    Simulates high-frequency sonar pulses propagating through water,
    attenuated by distance, corrupted by marine ambient noise, and scattered
    by four underwater target classes (Rock, Fish, Submarine, Void/Water).
    """
    def __init__(self, sample_rate_hz=500000, pulse_freq_hz=40000, sound_speed_m_s=1480):
        self.fs = sample_rate_hz          # 500 kHz sampling rate
        self.f0 = pulse_freq_hz          # 40 kHz ultrasonic pulse
        self.c = sound_speed_m_s          # 1480 m/s in water
        self.time_limit_s = 0.008         # 8ms window (~6m max physical range)
        self.t = np.arange(0, self.time_limit_s, 1.0 / self.fs)
        
    def generate_waveform(self, target_class, distance_meters):
        """
        Synthesizes a 1D ultrasonic transducer echo return waveform.
        
        A physical transducer fires a short burst (TX) at t=0, which travels through
        water, attenuates, reflects off a target at distance d, scatters, and returns at t = 2d / c.
        """
        # 1. Initialize empty wave with background ambient marine noise
        # Models thermal noise, wave action, and distant snapping shrimp
        noise_amplitude = 0.04
        wave = np.random.normal(0, noise_amplitude, len(self.t))
        
        # 2. Add the Transmit Leakage Pulse (TX burst at t=0)
        # Represents the initial high-intensity pulse leaking directly to the receiver
        tx_start = 0.0
        tx_duration = 0.0002 # 200 microseconds
        tx_envelope = np.exp(-((self.t - 0.0001) / (tx_duration / 3))**2)
        tx_pulse = 1.0 * tx_envelope * np.sin(2 * np.pi * self.f0 * self.t)
        wave += tx_pulse
        
        # If the target is out of range or class is "Void/Water", return only TX pulse and noise
        if target_class == "Void" or distance_meters < 0.2 or distance_meters > 5.5:
            return self.t, wave
            
        # 3. Calculate Echo Time-of-Flight (TOF)
        # Round trip time: t = 2 * d / c
        tof = (2.0 * distance_meters) / self.c
        
        # Ensure TOF fits within our time window
        if tof >= self.time_limit_s:
            return self.t, wave
            
        # 4. Model Water Attenuation
        # Acoustic absorption in seawater: higher distance leads to exponential attenuation
        # Alpha represents absorption coefficient. Attenuation = e^(-alpha * distance)
        absorption_coeff = 0.45  # Simplified linear dampening for 40kHz in shallow water
        attenuation = np.exp(-absorption_coeff * distance_meters)
        
        # 5. Model Target Echo based on Class
        echo_wave = np.zeros(len(self.t))
        
        if target_class == "Submarine":
            # SPECULAR/METALLIC REFLECTOR:
            # - Extremely high reflectivity (hard metal boundary)
            # - Very sharp, single clean peak
            # - High frequency coherence (mirror-like echo)
            peak_amp = 0.7 * attenuation
            duration = 0.00025  # Short, crisp pulse
            envelope = np.exp(-((self.t - tof) / (duration / 2.5))**2)
            echo_wave = peak_amp * envelope * np.sin(2 * np.pi * self.f0 * (self.t - tof))
            
        elif target_class == "Rock":
            # RUGGED/DIFFUSE REFLECTOR:
            # - Wide, multi-peak echo due to surface roughness
            # - Broad temporal scattering
            # - Frequency scattering (multi-path interference)
            peak_amp = 0.4 * attenuation
            duration = 0.0008  # Long, spread-out pulse
            
            # Create three staggered sub-peaks representing rock contours
            offsets = [-0.0002, 0.0, 0.00025]
            weights = [0.6, 1.0, 0.55]
            phases = [0.0, math.pi/4, -math.pi/3]
            
            for offset, weight, phase in zip(offsets, weights, phases):
                sub_tof = tof + offset
                if 0 <= sub_tof < self.time_limit_s:
                    envelope = np.exp(-((self.t - sub_tof) / (duration / 4))**2)
                    echo_wave += peak_amp * weight * envelope * np.sin(2 * np.pi * self.f0 * (self.t - sub_tof) + phase)
                    
        elif target_class == "Fish":
            # SOFT/DYNAMIC REFLECTOR:
            # - Dual weak reflections (swim bladder = high acoustic impedance, body flesh = low)
            # - Low overall amplitude (absorbs/refracts sound)
            # - Fluctuating phase
            peak_amp = 0.22 * attenuation
            duration = 0.0004
            
            # Primary peak (swim bladder)
            envelope1 = np.exp(-((self.t - tof) / (duration / 3.0))**2)
            echo_wave += peak_amp * 1.0 * envelope1 * np.sin(2 * np.pi * self.f0 * (self.t - tof))
            
            # Secondary weaker peak (body boundary, slightly delayed)
            body_tof = tof + 0.00015
            if body_tof < self.time_limit_s:
                envelope2 = np.exp(-((self.t - body_tof) / (duration / 2.5))**2)
                echo_wave += peak_amp * 0.45 * envelope2 * np.sin(2 * np.pi * self.f0 * (self.t - body_tof) + math.pi/2)
                
        # 6. Add synthesized echo to the acoustic wave channel
        wave += echo_wave
        
        # Apply slight lowpass filter to simulate water acting as a natural high-frequency dampener
        b, a = signal.butter(4, 0.3, 'low')
        wave = signal.filtfilt(b, a, wave)
        
        return self.t, wave

# Simple test function
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    gen = EcoWaveformGenerator()
    classes = ["Submarine", "Rock", "Fish", "Void"]
    
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(classes):
        t, wave = gen.generate_waveform(cls, 2.5) # Target at 2.5 meters
        plt.subplot(4, 1, i+1)
        plt.plot(t * 1000, wave, label=f"Class: {cls}")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("test_waveforms.png")
    print("Waveform generation completed successfully. Test plot saved.")
