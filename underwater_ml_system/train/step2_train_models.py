"""
step2_train_models.py
────────────────────────────────────────────────────────────────
Trains THREE ML models on the generated HC-SR04 dataset:
  1. Random Forest      (sklearn)
  2. SVM                (sklearn)
  3. XGBoost / Gradient Boosting (sklearn GradientBoostingClassifier)

For EACH model, trains TWO classifiers:
  A. Object classifier  → predicts what type of object (6 classes)
  B. Motion classifier  → predicts STATIC or MOVING (binary)

Outputs real accuracy metrics:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix
  - Per-class performance

Saves:
  models/rf_object.pkl      Random Forest object classifier
  models/rf_motion.pkl      Random Forest motion classifier
  models/svm_object.pkl     SVM object classifier
  models/svm_motion.pkl     SVM motion classifier
  models/gb_object.pkl      Gradient Boosting object classifier
  models/gb_motion.pkl      Gradient Boosting motion classifier
  models/scaler.pkl         Feature scaler (must use at inference time)
  models/results.json       All accuracy metrics for the dashboard

Run: python step2_train_models.py
────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pickle, json, os, time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "training_data.csv")
MDL_DIR   = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

FEATURE_COLS = [
    "echo_duration_ms", "signal_amplitude", "distance_cm",
    "distance_delta", "angular_velocity", "echo_count",
    "noise_ratio", "doppler_shift_hz"
]
OBJECT_NAMES = ["Fish School", "Rock/Reef", "Submarine", "Diver", "Debris", "Marine Mammal"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  UNDERWATER OBJECT DETECTION — ML TRAINING")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print("ERROR: Run step1_generate_data.py first!")
    exit(1)

df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLS].values
y_obj = df["object_class"].values
y_mov = df["is_moving"].values

print(f"\n  Dataset loaded: {len(df)} samples, {len(FEATURE_COLS)} features")

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open(os.path.join(MDL_DIR, "scaler.pkl"), "wb"))
print("  Scaler saved: models/scaler.pkl")

# ── Train/test split ──────────────────────────────────────────────────────────
X_tr, X_te, yo_tr, yo_te, ym_tr, ym_te = train_test_split(
    X_scaled, y_obj, y_mov, test_size=0.2, random_state=42, stratify=y_obj
)
print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "rf": {
        "name":  "Random Forest",
        "short": "RF",
        "obj":   RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1),
        "mov":   RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    },
    "gb": {
        "name":  "Gradient Boosting",
        "short": "GB",
        "obj":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        "mov":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
    },
    "svm": {
        "name":  "Support Vector Machine",
        "short": "SVM",
        "obj":   SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=42),
        "mov":   SVC(C=5,  gamma="scale", kernel="rbf", probability=True, random_state=42),
    },
}

# ── Train + evaluate each model ───────────────────────────────────────────────
results = {}

for key, cfg in MODELS.items():
    print(f"\n{'─'*60}")
    print(f"  Training: {cfg['name']}")
    print(f"{'─'*60}")

    # --- OBJECT CLASSIFIER ---
    print(f"  [1/2] Object classifier (6 classes)...")
    t0 = time.time()
    cfg["obj"].fit(X_tr, yo_tr)
    train_time = round(time.time() - t0, 2)

    yo_pred = cfg["obj"].predict(X_te)
    obj_acc  = round(accuracy_score(yo_te, yo_pred) * 100, 2)
    obj_prec = round(precision_score(yo_te, yo_pred, average="weighted") * 100, 2)
    obj_rec  = round(recall_score(yo_te, yo_pred, average="weighted") * 100, 2)
    obj_f1   = round(f1_score(yo_te, yo_pred, average="weighted") * 100, 2)
    obj_cm   = confusion_matrix(yo_te, yo_pred).tolist()

    pickle.dump(cfg["obj"], open(os.path.join(MDL_DIR, f"{key}_object.pkl"), "wb"))
    print(f"    Accuracy  : {obj_acc}%")
    print(f"    Precision : {obj_prec}%")
    print(f"    Recall    : {obj_rec}%")
    print(f"    F1 Score  : {obj_f1}%")
    print(f"    Train time: {train_time}s")

    # Per-class breakdown
    print(f"  Per-class accuracy:")
    report = classification_report(yo_te, yo_pred, target_names=OBJECT_NAMES, output_dict=True)
    for cls in OBJECT_NAMES:
        print(f"    {cls:<18} {round(report[cls]['f1-score']*100,1)}% F1")

    # --- MOTION CLASSIFIER ---
    print(f"  [2/2] Motion classifier (Static/Moving)...")
    t0 = time.time()
    cfg["mov"].fit(X_tr, ym_tr)
    mov_time = round(time.time() - t0, 2)

    ym_pred  = cfg["mov"].predict(X_te)
    mov_acc  = round(accuracy_score(ym_te, ym_pred) * 100, 2)
    mov_prec = round(precision_score(ym_te, ym_pred) * 100, 2)
    mov_rec  = round(recall_score(ym_te, ym_pred) * 100, 2)
    mov_f1   = round(f1_score(ym_te, ym_pred) * 100, 2)
    mov_cm   = confusion_matrix(ym_te, ym_pred).tolist()

    pickle.dump(cfg["mov"], open(os.path.join(MDL_DIR, f"{key}_motion.pkl"), "wb"))
    print(f"    Accuracy  : {mov_acc}%")
    print(f"    Precision : {mov_prec}%")
    print(f"    Recall    : {mov_rec}%")
    print(f"    F1 Score  : {mov_f1}%")

    # Cross-validation (object classifier)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42) if key == "rf"
        else GradientBoostingClassifier(n_estimators=50, random_state=42) if key == "gb"
        else SVC(kernel="rbf", random_state=42),
        X_scaled, y_obj, cv=5, scoring="accuracy", n_jobs=-1 if key != "svm" else 1
    )
    cv_mean = round(cv_scores.mean() * 100, 2)
    cv_std  = round(cv_scores.std()  * 100, 2)
    print(f"  5-Fold CV  : {cv_mean}% ± {cv_std}%")

    results[key] = {
        "name":  cfg["name"],
        "short": cfg["short"],
        "object": {
            "accuracy":   obj_acc,
            "precision":  obj_prec,
            "recall":     obj_rec,
            "f1":         obj_f1,
            "train_time": train_time,
            "cv_mean":    cv_mean,
            "cv_std":     cv_std,
            "confusion_matrix": obj_cm,
            "per_class_f1": {cls: round(report[cls]["f1-score"]*100,1) for cls in OBJECT_NAMES},
        },
        "motion": {
            "accuracy":  mov_acc,
            "precision": mov_prec,
            "recall":    mov_rec,
            "f1":        mov_f1,
            "confusion_matrix": mov_cm,
        },
    }

# ── Save results JSON ─────────────────────────────────────────────────────────
results_path = os.path.join(MDL_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n{'='*60}")
print(f"  ALL MODELS TRAINED")
print(f"{'='*60}")
print(f"  Results saved: {results_path}")
print()

# ── Final comparison table ────────────────────────────────────────────────────
print(f"  {'Model':<24} {'Obj Acc':>8} {'Obj F1':>8} {'Mot Acc':>8} {'Mot F1':>8}")
print(f"  {'─'*56}")
for key, r in results.items():
    print(f"  {r['name']:<24} {r['object']['accuracy']:>7}% {r['object']['f1']:>7}% {r['motion']['accuracy']:>7}% {r['motion']['f1']:>7}%")
print()
print("  Run next: python step3_export_for_browser.py")
