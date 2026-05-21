/**
 * inference.js
 * ─────────────────────────────────────────────────────────────
 * Loads the trained model JSON files and runs real inference
 * in the browser — no server needed.
 *
 * Supports all 3 model types:
 *   "forest" → Random Forest (majority vote across trees)
 *   "gb"     → Gradient Boosting (sum of tree residuals)
 *   "svm"    → Support Vector Machine (RBF kernel scoring)
 * ─────────────────────────────────────────────────────────────
 */

const Inference = (() => {

  // Loaded model objects, keyed by "rf"/"gb"/"svm" + "_object"/"_motion"
  const models  = {};
  let   scaler  = null;
  let   results = null;   // training accuracy from results.json
  let   ready   = false;

  const FEATURE_ORDER = [
    "echo_duration_ms", "signal_amplitude", "distance_cm",
    "distance_delta",   "angular_velocity", "echo_count",
    "noise_ratio",       "doppler_shift_hz"
  ];

  // ── Load all models from /models/ folder ──────────────────────
  async function loadAll(onProgress) {
    const files = [
      "scaler.json",
      "results.json",
      "rf_object.json",  "rf_motion.json",
      "gb_object.json",  "gb_motion.json",
      "svm_object.json", "svm_motion.json",
    ];
    for (let i = 0; i < files.length; i++) {
      const name = files[i];
      if (onProgress) onProgress(`Loading ${name}...`, Math.round((i/files.length)*100));
      try {
        const resp = await fetch(`models/${name}`);
        const data = await resp.json();
        if      (name === "scaler.json")  scaler  = data;
        else if (name === "results.json") results = data;
        else {
          // key = "rf_object", "rf_motion", etc.
          const key = name.replace(".json","");
          models[key] = data;
        }
      } catch(e) {
        console.error("Failed to load", name, e);
      }
    }
    ready = true;
    if (onProgress) onProgress("All models loaded", 100);
  }

  // ── Scale a feature vector (same as sklearn StandardScaler) ───
  function scale(featObj) {
    return FEATURE_ORDER.map((k, i) => {
      const v = featObj[k] || 0;
      return (v - scaler.mean[i]) / (scaler.std[i] || 1);
    });
  }

  // ── Predict object type + motion from raw features ────────────
  function predict(featObj, modelKey) {
    if (!ready) return null;
    const x = scale(featObj);

    const objResult = _runModel(models[`${modelKey}_object`], x);
    const movResult = _runModel(models[`${modelKey}_motion`], x);

    return {
      object_class: objResult.class_idx,
      object_name:  objResult.class_name,
      object_conf:  objResult.confidence,
      is_moving:    movResult.class_idx === 1,
      motion_conf:  movResult.confidence,
      motion_label: movResult.class_name,
    };
  }

  // ── Route to correct inference method ─────────────────────────
  function _runModel(model, x) {
    if (!model) return { class_idx: 0, class_name: "Unknown", confidence: 0 };
    if      (model.type === "forest") return _inferForest(model, x);
    else if (model.type === "gb")     return _inferGB(model, x);
    else if (model.type === "svm")    return _inferSVM(model, x);
    return { class_idx: 0, class_name: "Unknown", confidence: 0 };
  }

  // ── Random Forest inference ────────────────────────────────────
  // Each tree votes; class with most votes wins.
  function _inferForest(model, x) {
    const votes = new Array(model.n_classes).fill(0);
    for (const tree of model.trees) {
      const cls = _walkTree(tree, x);
      votes[cls]++;
    }
    const cls_idx = votes.indexOf(Math.max(...votes));
    const conf    = Math.round((votes[cls_idx] / model.n_trees) * 100 * 10) / 10;
    return {
      class_idx:  cls_idx,
      class_name: model.classes[cls_idx],
      confidence: conf,
      votes,
    };
  }

  // Walk a single decision tree node
  function _walkTree(node, x) {
    if (node.leaf) return node.class;
    return x[node.feature] <= node.thresh
      ? _walkTree(node.left, x)
      : _walkTree(node.right, x);
  }

  // ── Gradient Boosting inference ────────────────────────────────
  // Sum residuals across all estimators per class, apply softmax.
  function _inferGB(model, x) {
    const n  = model.n_classes;
    const lr = model.learning_rate;

    if (n === 2) {
      // Binary: single set of trees
      let score = Math.log(model.init_prediction[1] / model.init_prediction[0]);
      for (const tree of model.trees_per_class[0]) {
        score += lr * _walkTreeValue(tree, x);
      }
      const prob1 = 1 / (1 + Math.exp(-score));
      const cls   = prob1 >= 0.5 ? 1 : 0;
      return {
        class_idx:  cls,
        class_name: model.classes[cls],
        confidence: Math.round(Math.max(prob1, 1-prob1) * 100 * 10) / 10,
      };
    } else {
      // Multiclass: one set of trees per class
      const scores = model.init_prediction.map(p => Math.log(p + 1e-9));
      for (let c = 0; c < n; c++) {
        for (const tree of model.trees_per_class[c]) {
          scores[c] += lr * _walkTreeValue(tree, x);
        }
      }
      const probs   = _softmax(scores);
      const cls_idx = probs.indexOf(Math.max(...probs));
      return {
        class_idx:  cls_idx,
        class_name: model.classes[cls_idx],
        confidence: Math.round(probs[cls_idx] * 100 * 10) / 10,
      };
    }
  }

  // Walk tree and return leaf value (GB uses regression leaves)
  function _walkTreeValue(node, x) {
    if (node.leaf) return node.votes ? node.votes[node.class] : node.class;
    return x[node.feature] <= node.thresh
      ? _walkTreeValue(node.left, x)
      : _walkTreeValue(node.right, x);
  }

  function _softmax(arr) {
    const max  = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum  = exps.reduce((a,b) => a+b, 0);
    return exps.map(v => v/sum);
  }

  // ── SVM inference ──────────────────────────────────────────────
  // RBF kernel: K(x,sv) = exp(-gamma * ||x-sv||^2)
  // Decision = sign( sum_i alpha_i * K(x, sv_i) + b )
  function _inferSVM(model, x) {
    const SVs    = model.support_vectors;
    const alphas = model.dual_coef;
    const bias   = model.intercept;
    const gamma  = model.gamma;
    const labels = model.class_labels;
    const n_cls  = model.n_classes;

    if (n_cls === 2) {
      // Binary SVM
      const kernels = SVs.map(sv => _rbf(x, sv, gamma));
      let decision  = bias[0];
      for (let i = 0; i < alphas[0].length; i++) {
        decision += alphas[0][i] * kernels[i];
      }
      const cls = decision >= 0 ? labels[1] : labels[0];
      const idx = model.classes.indexOf(String(cls));
      return {
        class_idx:  idx >= 0 ? idx : 0,
        class_name: model.classes[idx >= 0 ? idx : 0],
        confidence: Math.min(99.9, Math.round(Math.min(Math.abs(decision)*18+75, 99.9)*10)/10),
      };
    } else {
      // Multi-class OvO SVM
      const vote_counts = new Array(n_cls).fill(0);
      let pair = 0;
      for (let i = 0; i < n_cls; i++) {
        for (let j = i+1; j < n_cls; j++) {
          const kernels = SVs.map(sv => _rbf(x, sv, gamma));
          let decision  = bias[pair];
          for (let k = 0; k < alphas[pair < alphas.length ? pair : 0].length; k++) {
            decision += (alphas[pair < alphas.length ? pair : 0][k] || 0) * kernels[k];
          }
          if (decision >= 0) vote_counts[i]++; else vote_counts[j]++;
          pair++;
        }
      }
      const cls_idx = vote_counts.indexOf(Math.max(...vote_counts));
      const conf    = Math.round((Math.max(...vote_counts) / (n_cls*(n_cls-1)/2)) * 40 + 60);
      return {
        class_idx:  cls_idx,
        class_name: model.classes[cls_idx],
        confidence: Math.min(99.9, conf),
      };
    }
  }

  function _rbf(x, sv, gamma) {
    let dist2 = 0;
    for (let i = 0; i < x.length; i++) dist2 += (x[i]-sv[i])**2;
    return Math.exp(-gamma * dist2);
  }

  // ── Accessors ──────────────────────────────────────────────────
  function isReady()      { return ready; }
  function getResults()   { return results; }

  return { loadAll, predict, isReady, getResults };

})();
