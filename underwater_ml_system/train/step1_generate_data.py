"""
step1_generate_data.py
────────────────────────────────────────────────────────────────
Generates realistic HC-SR04 ultrasonic sensor training data.

Each object type has unique physical echo characteristics:
  - Fish School   : scattered, weak echoes, usually moving
  - Rock / Reef   : strong hard echo, always static
  - Submarine     : very strong echo, fast movement
  - Diver         : medium echo, medium speed
  - Debris        : noisy irregular echoes, sometimes drifts
  - Marine Mammal : medium echo, fast movement

Features (what HC-SR04 actually measures):
  1. echo_duration_ms   — round-trip pulse time
  2. signal_amplitude   — echo signal strength (0–1)
  3. distance_cm        — calculated object distance
  4. distance_delta     — change between readings (KEY for motion)
  5. angular_velocity   — servo sweep rate effect
  6. echo_count         — reflections per pulse
  7. noise_ratio        — ambient underwater noise
  8. doppler_shift_hz   — frequency shift (KEY for motion)

Labels:
  object_class  0-5   (what type of object)
  is_moving     0/1   (static or moving)

Run: python step1_generate_data.py
────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

OBJECTS = [
    {
        "id": 0, "name": "Fish School",
        "dist":      (30, 150),
        "amplitude": (0.25, 0.55),
        "echo_count":(3, 8),
        "noise":     (0.08, 0.20),
        "move_prob": 0.85,
        "speed":     (0.8, 4.0),
    },
    {
        "id": 1, "name": "Rock/Reef",
        "dist":      (50, 200),
        "amplitude": (0.72, 0.97),
        "echo_count":(1, 2),
        "noise":     (0.02, 0.07),
        "move_prob": 0.0,
        "speed":     (0.0, 0.0),
    },
    {
        "id": 2, "name": "Submarine",
        "dist":      (40, 200),
        "amplitude": (0.88, 0.99),
        "echo_count":(1, 2),
        "noise":     (0.02, 0.09),
        "move_prob": 0.75,
        "speed":     (2.5, 9.0),
    },
    {
        "id": 3, "name": "Diver",
        "dist":      (15, 130),
        "amplitude": (0.38, 0.72),
        "echo_count":(2, 5),
        "noise":     (0.06, 0.18),
        "move_prob": 0.70,
        "speed":     (0.4, 2.5),
    },
    {
        "id": 4, "name": "Debris",
        "dist":      (20, 160),
        "amplitude": (0.18, 0.60),
        "echo_count":(3, 9),
        "noise":     (0.12, 0.30),
        "move_prob": 0.30,
        "speed":     (0.1, 1.2),
    },
    {
        "id": 5, "name": "Marine Mammal",
        "dist":      (30, 175),
        "amplitude": (0.45, 0.82),
        "echo_count":(2, 4),
        "noise":     (0.04, 0.13),
        "move_prob": 0.90,
        "speed":     (1.5, 7.0),
    },
]

N_PER_CLASS = 600   # 600 x 6 = 3600 samples total

def make_sample(obj, prev_dist=None):
    dist      = np.random.uniform(*obj["dist"])
    echo_dur  = (dist * 2 / 34.3) + np.random.normal(0, 0.025)
    amplitude = np.clip(np.random.uniform(*obj["amplitude"]) * (1 - dist/420) + np.random.normal(0, 0.03), 0.01, 1.0)
    moving    = np.random.random() < obj["move_prob"]
    speed     = np.random.uniform(*obj["speed"]) if moving else 0.0
    delta     = (dist - prev_dist + np.random.normal(0, 0.06)) if prev_dist else (np.random.normal(0, speed*0.15) if moving else np.random.normal(0, 0.03))
    ang_vel   = abs(speed * 0.75 + np.random.normal(0, 0.12)) if moving else abs(np.random.normal(0, 0.025))
    doppler   = np.random.choice([-1,1]) * speed * 0.9 + np.random.normal(0, 0.18) if moving else np.random.normal(0, 0.12)
    return {
        "echo_duration_ms":  round(echo_dur, 4),
        "signal_amplitude":  round(amplitude, 4),
        "distance_cm":       round(dist, 2),
        "distance_delta":    round(delta, 4),
        "angular_velocity":  round(ang_vel, 4),
        "echo_count":        int(np.random.randint(*obj["echo_count"])),
        "noise_ratio":       round(np.random.uniform(*obj["noise"]), 4),
        "doppler_shift_hz":  round(doppler, 4),
        "object_class":      obj["id"],
        "object_name":       obj["name"],
        "is_moving":         int(moving),
    }

rows, prev = [], None
for obj in OBJECTS:
    prev = None
    for _ in range(N_PER_CLASS):
        s = make_sample(obj, prev)
        prev = s["distance_cm"]
        rows.append(s)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
out = os.path.join(os.path.dirname(__file__), "data", "training_data.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)

print("=" * 52)
print("  DATASET GENERATED")
print("=" * 52)
print(f"  Total samples : {len(df)}")
print(f"  Features      : 8")
print(f"  Output        : {out}")
print()
print("  Class distribution:")
for name, cnt in df["object_name"].value_counts().items():
    print(f"    {name:<18} {cnt} samples")
print()
print("  Motion distribution:")
print(f"    Moving : {df['is_moving'].sum()}")
print(f"    Static : {(df['is_moving']==0).sum()}")
print()
print("  Run next: python step2_train_models.py")
