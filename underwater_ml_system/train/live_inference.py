import serial
import numpy as np
import joblib
import json

# 🔹 Load model
model = joblib.load("models/rf_object.pkl")

# 🔹 Serial connection
ser = serial.Serial('COM3', 115200)  # change if needed

print("Live system started...\n")

while True:
    try:
        line = ser.readline().decode().strip()

        if "distance" in line:
            continue

        d, a, v, acc = map(float, line.split(","))

        X = np.array([[d, a, v, acc]])
        prediction = model.predict(X)[0]

        result = {
            "distance": d,
            "angle": a,
            "velocity": v,
            "acceleration": acc,
            "prediction": int(prediction)
        }

        print(result)

        # 🔹 Save for dashboard
        with open("models/results.json", "w") as f:
            json.dump(result, f)

        # 🔹 Send back to ESP32
        if prediction == 1:
            ser.write(b'1\n')
        else:
            ser.write(b'0\n')

    except Exception as e:
        print("Error:", e)