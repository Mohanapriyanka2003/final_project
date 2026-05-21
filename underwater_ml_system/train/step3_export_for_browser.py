"""
step3_export_for_browser.py
────────────────────────────────────────────────────────────────
Exports trained models into JSON format that the browser
dashboard can load and use for real-time inference.

Random Forest and Gradient Boosting → export all decision trees
as nested JSON so the browser can walk the tree paths in JS.

SVM → export support vectors + coefficients for browser scoring.

Also exports the scaler (mean + std) so the browser normalizes
incoming sensor values the same way training did.

Output:
  dashboard/models/rf_object.json
  dashboard/models/rf_motion.json
  dashboard/models/gb_object.json
  dashboard/models/gb_motion.json
  dashboard/models/svm_object.json
  dashboard/models/svm_motion.json
  dashboard/models/scaler.json
  dashboard/models/results.json  (copied from train/models/)

Run: python step3_export_for_browser.py
────────────────────────────────────────────────────────────────
"""

import pickle, json, os, shutil
import numpy as np

BASE      = os.path.dirname(__file__)
MDL_DIR   = os.path.join(BASE, "models")
DASH_MDL  = os.path.join(BASE, "..", "dashboard", "models")
os.makedirs(DASH_MDL, exist_ok=True)

FEATURE_COLS = [
    "echo_duration_ms", "signal_amplitude", "distance_cm",
    "distance_delta", "angular_velocity", "echo_count",
    "noise_ratio", "doppler_shift_hz"
]
OBJECT_NAMES = ["Fish School", "Rock/Reef", "Submarine", "Diver", "Debris", "Marine Mammal"]

def load(name):
    path = os.path.join(MDL_DIR, name)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run step2 first.")
        exit(1)
    return pickle.load(open(path, "rb"))

# ── Export scaler ─────────────────────────────────────────────────────────────
scaler = load("scaler.pkl")
scaler_json = {
    "mean": scaler.mean_.tolist(),
    "std":  np.sqrt(scaler.var_).tolist(),
    "features": FEATURE_COLS,
}
with open(os.path.join(DASH_MDL, "scaler.json"), "w") as f:
    json.dump(scaler_json, f)
print("Exported: scaler.json")

# ── Export decision tree (RF / GB) ────────────────────────────────────────────
def export_tree(tree, n_classes):
    """Recursively export a single sklearn decision tree to nested dict."""
    t = tree.tree_
    def recurse(node):
        if t.children_left[node] == -1:  # leaf
            vals = t.value[node][0]
            return {"leaf": True, "class": int(np.argmax(vals)), "votes": vals.tolist()}
        return {
            "leaf":    False,
            "feature": int(t.feature[node]),
            "thresh":  float(round(t.threshold[node], 6)),
            "left":    recurse(t.children_left[node]),
            "right":   recurse(t.children_right[node]),
        }
    return recurse(0)

def export_forest(model, path, classes, label_type):
    trees = [export_tree(est, len(classes)) for est in model.estimators_]
    out = {
        "type":       "forest",
        "label_type": label_type,
        "n_trees":    len(trees),
        "n_classes":  len(classes),
        "classes":    classes,
        "trees":      trees,
    }
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    kb = os.path.getsize(path) // 1024
    print(f"Exported: {os.path.basename(path)} ({kb} KB, {len(trees)} trees)")

# ── Export SVM ────────────────────────────────────────────────────────────────
def export_svm(model, path, classes, label_type):
    out = {
        "type":          "svm",
        "label_type":    label_type,
        "n_classes":     len(classes),
        "classes":       classes,
        "support_vectors": model.support_vectors_.tolist(),
        "dual_coef":       model.dual_coef_.tolist(),
        "intercept":       model.intercept_.tolist(),
        "gamma":           float(model._gamma),
        "class_labels":    model.classes_.tolist(),
    }
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    kb = os.path.getsize(path) // 1024
    print(f"Exported: {os.path.basename(path)} ({kb} KB)")

# ── Run exports ───────────────────────────────────────────────────────────────
print("\n=== Exporting models for browser ===\n")

# Random Forest
rf_obj = load("rf_object.pkl")
rf_mov = load("rf_motion.pkl")
export_forest(rf_obj, os.path.join(DASH_MDL, "rf_object.json"),  OBJECT_NAMES, "object")
export_forest(rf_mov, os.path.join(DASH_MDL, "rf_motion.json"),  ["Static","Moving"], "motion")

# Gradient Boosting — estimators_ is list-of-lists per class
# Flatten for export: for binary use estimators_[:,0], for multiclass export per class
def export_gb(model, path, classes, label_type):
    n_cls = len(classes)
    if n_cls == 2:
        # Binary: estimators_ shape = (n_estimators, 1)
        trees = [export_tree(est[0], 2) for est in model.estimators_]
        all_trees = [trees]
    else:
        # Multiclass: estimators_ shape = (n_estimators, n_classes)
        all_trees = []
        for c in range(n_cls):
            all_trees.append([export_tree(model.estimators_[i][c], n_cls) for i in range(model.n_estimators_)])
    out = {
        "type":            "gb",
        "label_type":      label_type,
        "n_classes":       n_cls,
        "classes":         classes,
        "learning_rate":   model.learning_rate,
        "init_prediction": model.init_.class_prior_.tolist() if hasattr(model.init_, "class_prior_") else [1/n_cls]*n_cls,
        "trees_per_class": all_trees,
    }
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    kb = os.path.getsize(path) // 1024
    print(f"Exported: {os.path.basename(path)} ({kb} KB, {model.n_estimators_} estimators)")

gb_obj = load("gb_object.pkl")
gb_mov = load("gb_motion.pkl")
export_gb(gb_obj, os.path.join(DASH_MDL, "gb_object.json"),  OBJECT_NAMES, "object")
export_gb(gb_mov, os.path.join(DASH_MDL, "gb_motion.json"),  ["Static","Moving"], "motion")

# SVM
svm_obj = load("svm_object.pkl")
svm_mov = load("svm_motion.pkl")
export_svm(svm_obj, os.path.join(DASH_MDL, "svm_object.json"), OBJECT_NAMES, "object")
export_svm(svm_mov, os.path.join(DASH_MDL, "svm_motion.json"), ["Static","Moving"], "motion")

# Copy results.json
shutil.copy(os.path.join(MDL_DIR, "results.json"), os.path.join(DASH_MDL, "results.json"))
print(f"Copied:   results.json")

print("\n=== All exports done ===")
print("Dashboard models ready in: dashboard/models/")
print("Open dashboard/index.html in Chrome to run.")
