"""
generate_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a synthetic but physically realistic training dataset
simulating HC-SR04 ultrasonic sensor readings for 6 underwater
object classes.

Each sample has 8 features derived from sonar echo behaviour:
  1. echo_duration_ms     — time for pulse to return
  2. signal_amplitude     — strength of returned echo (0–1)
  3. distance_cm          — calculated distance
  4. distance_delta       — change vs previous reading (motion)
  5. angular_velocity     — rotation rate (°/s)
  6. echo_count           — number of echoes per pulse
  7. noise_ratio          — ambient noise level
  8. doppler_shift_hz     — frequency shift (motion indicator)

Labels:
  Object class  : 0–5  (Fish, Rock, Submarine, Diver, Debris, Mammal)
  Motion class  : 0=Static, 1=Moving

Output: data/training_data.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Object class definitions ─────────────────────────────────────────────────
# Each object has distinct physical characteristics that affect echo behaviour
OBJECTS = [
    {
        "id": 0, "name": "Fish School",
        "dist_range":   (30, 150),
        "amplitude":    (0.3, 0.6),   # small, scattered → weaker echo
        "echo_count":   (3, 7),        # multiple small reflectors
        "noise":        (0.05, 0.15),
        "move_prob":    0.85,          # usually moving
        "speed_range":  (0.8, 3.5),   # cm/s
    },
    {
        "id": 1, "name": "Rock / Reef",
        "dist_range":   (50, 200),
        "amplitude":    (0.7, 0.95),  # hard surface → strong echo
        "echo_count":   (1, 3),        # clean single reflection
        "noise":        (0.02, 0.08),
        "move_prob":    0.0,           # always static
        "speed_range":  (0.0, 0.0),
    },
    {
        "id": 2, "name": "Submarine",
        "dist_range":   (40, 200),
        "amplitude":    (0.85, 0.99), # large metal → very strong echo
        "echo_count":   (1, 2),
        "noise":        (0.03, 0.10),
        "move_prob":    0.75,
        "speed_range":  (2.0, 8.0),   # faster movement
    },
    {
        "id": 3, "name": "Diver",
        "dist_range":   (15, 130),
        "amplitude":    (0.4, 0.75),  # soft body → medium echo
        "echo_count":   (2, 5),
        "noise":        (0.05, 0.18),
        "move_prob":    0.70,
        "speed_range":  (0.5, 2.5),
    },
    {
        "id": 4, "name": "Debris",
        "dist_range":   (20, 160),
        "amplitude":    (0.2, 0.65),  # irregular shape → noisy echo
        "echo_count":   (2, 8),        # many irregular surfaces
        "noise":        (0.10, 0.28),
        "move_prob":    0.30,          # sometimes drifts
        "speed_range":  (0.1, 1.0),
    },
    {
        "id": 5, "name": "Marine Mammal",
        "dist_range":   (30, 175),
        "amplitude":    (0.45, 0.80),
        "echo_count":   (2, 4),
        "noise":        (0.04, 0.12),
        "move_prob":    0.90,
        "speed_range":  (1.5, 6.0),
    },
]

SAMPLES_PER_CLASS = 500   # 500 × 6 = 3000 total samples
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "training_data.csv")


def generate_sample(obj, prev_dist=None):
    """Generate one sensor reading for a given object class."""
    dist = np.random.uniform(*obj["dist_range"])

    # Echo duration: HC-SR04 measures round-trip time
    # duration_ms = (dist_cm * 2) / (speed_of_sound_cm_ms)
    # speed of sound in air ≈ 34.3 cm/ms
    echo_duration = (dist * 2 / 34.3) + np.random.normal(0, 0.02)

    # Signal amplitude: decreases with distance + object properties
    base_amp = np.random.uniform(*obj["amplitude"])
    amplitude = base_amp * (1 - dist / 400) + np.random.normal(0, 0.03)
    amplitude = np.clip(amplitude, 0.01, 1.0)

    # Is this object moving this reading?
    is_moving = np.random.random() < obj["move_prob"]
    speed = np.random.uniform(*obj["speed_range"]) if is_moving else 0.0

    # Distance delta (change from previous reading)
    if prev_dist is not None:
        dist_delta = dist - prev_dist + np.random.normal(0, 0.05)
    else:
        dist_delta = np.random.normal(0, speed * 0.1) if is_moving else np.random.normal(0, 0.02)

    # Angular velocity (from servo position change + object motion)
    ang_vel = speed * 0.8 + np.random.normal(0, 0.1) if is_moving else np.random.normal(0, 0.02)
    ang_vel = abs(ang_vel)

    # Echo count: more echoes for irregular/multiple objects
    echo_count = np.random.randint(*obj["echo_count"])

    # Noise ratio: depth/environment dependent
    noise = np.random.uniform(*obj["noise"])

    # Doppler shift: frequency change due to motion (Hz)
    # Objects moving toward sensor → positive shift, away → negative
    direction = np.random.choice([-1, 1])
    doppler = direction * speed * 0.85 + np.random.normal(0, 0.2) if is_moving else np.random.normal(0, 0.15)

    return {
        "echo_duration_ms":  round(echo_duration, 4),
        "signal_amplitude":  round(amplitude, 4),
        "distance_cm":       round(dist, 2),
        "distance_delta":    round(dist_delta, 4),
        "angular_velocity":  round(ang_vel, 4),
        "echo_count":        echo_count,
        "noise_ratio":       round(noise, 4),
        "doppler_shift_hz":  round(doppler, 4),
        "object_class":      obj["id"],
        "object_name":       obj["name"],
        "is_moving":         int(is_moving),
    }


def generate_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows = []
    for obj in OBJECTS:
        prev_dist = None
        for _ in range(SAMPLES_PER_CLASS):
            sample = generate_sample(obj, prev_dist)
            prev_dist = sample["distance_cm"]
            rows.append(sample)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved: {OUTPUT_PATH}")
    print(f"Total samples : {len(df)}")
    print(f"Class distribution:\n{df['object_name'].value_counts().to_string()}")
    print(f"Moving vs Static:\n{df['is_moving'].value_counts().rename({1:'Moving',0:'Static'}).to_string()}")
    return df


if __name__ == "__main__":
    generate_dataset()
