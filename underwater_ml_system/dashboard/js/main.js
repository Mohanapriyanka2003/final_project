/**
 * main.js — Entry point, wires everything together
 */

let _activeML  = "rf";
let _running   = false;
let _scans     = 0;
let _sessionT  = 0;
let _timer     = null;
let _detected  = [];
let _selectedId= null;

const OBJECT_ICONS = {
  "Fish School":"🐟","Rock/Reef":"🪨","Submarine":"🚢",
  "Diver":"🤿","Debris":"🗑️","Marine Mammal":"🐬","Unknown":"📡"
};
const OBJECT_COLORS = {
  "Fish School":"#00e5ff","Rock/Reef":"#8d6e63","Submarine":"#ff1744",
  "Diver":"#ff6d00","Debris":"#ffd600","Marine Mammal":"#00e676","Unknown":"#aaaaaa"
};

window.addEventListener("DOMContentLoaded", async () => {
  Sonar.init("sonarCanvas");

  // Load trained models
  _el("loadStatus").textContent = "Loading trained models...";
  _el("loadBar").style.display  = "block";
  await Inference.loadAll((msg, pct) => {
    _el("loadStatus").textContent = msg;
    _el("loadBarFill").style.width = pct + "%";
  });
  _el("loadBar").style.display  = "none";
  _el("loadStatus").textContent = "Models ready — press START";

  // Render accuracy tables from real training results
  _renderAccuracyPanel();

  // Wire simulation
  Simulation.init({
    onDetect: _onDetect,
    onFrame:  (angle, objects) => {
      Sonar.render(angle, objects);
      _el("badgeAngle").textContent = Math.round(angle) + "°";
    }
  });

  _log("Trained models loaded (RF, GB, SVM)", "ok");
  _log("Press START to begin simulation", "info");
});

// ── Controls ──────────────────────────────────────────────────────────────────

function startSim() {
  if (_running) return;
  _running = true;
  Simulation.start();
  _el("btnStart").disabled = true;
  _el("btnStop").disabled  = false;
  _el("ctrlStatus").textContent = "RUNNING";
  _timer = setInterval(()=>{ _sessionT++; _el("statTime").textContent=_fmt(_sessionT); },1000);
  _log("Simulation started — using trained " + _activeML.toUpperCase() + " model", "ok");
}

function stopSim() {
  if (!_running) return;
  _running = false;
  Simulation.stop();
  _el("btnStart").disabled = false;
  _el("btnStop").disabled  = true;
  _el("ctrlStatus").textContent = "STOPPED";
  clearInterval(_timer);
  _log("Simulation stopped", "warn");
}

function addObject() { Simulation.addObject(); _log("Object added", "info"); }

function clearAll() {
  stopSim();
  Simulation.clear();
  _detected = [];
  _selectedId = null;
  _scans = 0; _sessionT = 0;
  _el("objList").innerHTML = '<div class="empty-msg">Waiting for scan...</div>';
  _el("featGrid").innerHTML = '<div class="empty-msg">Click an object to see features</div>';
  _el("featTitle").textContent = "";
  _el("logBox").innerHTML = "";
  _el("statTime").textContent = "00:00";
  _el("statScans").textContent = "0";
  _el("statObjects").textContent = "0";
  _el("badgeDist").textContent = "—";
  _setAlert("clear");
  _log("Cleared", "warn");
}

function setSpeed(v)  { Simulation.setSpeed(v); document.getElementById("speedVal").textContent=v; }

function setML(v) {
  _activeML = v;
  Simulation.setML(v);
  _highlightMLCard(v);
  _log("Active model → " + v.toUpperCase(), "info");
}

// ── Detection handler ─────────────────────────────────────────────────────────

function _onDetect(det) {
  _scans++;
  _el("statScans").textContent = _scans;

  // Use prediction from trained model
  const icon  = OBJECT_ICONS[det.pred_name]  || "📡";
  const color = OBJECT_COLORS[det.pred_name] || "#aaaaaa";
  const entry = { ...det, icon, color, pred_name: det.pred_name || "Unknown" };

  _detected = [entry, ..._detected.filter(d=>d.id!==det.id)].slice(0,14);
  _selectedId = entry.id;

  _el("statObjects").textContent = _detected.length;
  _el("badgeDist").textContent   = det.dist;

  _renderObjList();
  _renderFeatures(entry);
  _updateAlert();

  const motLabel = det.pred_moving ? "🔴 MOVING" : "🟢 Static";
  _log(`[${_activeML.toUpperCase()}] ${icon} ${det.pred_name} | ${det.dist}cm | ${motLabel} | obj:${det.obj_conf}% mot:${det.mot_conf}%`,
    det.pred_moving && det.dist < 80 ? "err" : det.pred_moving ? "warn" : "ok");
}

// ── Render detection list ─────────────────────────────────────────────────────

function _renderObjList() {
  const list = _el("objList");
  if (!_detected.length) { list.innerHTML='<div class="empty-msg">Waiting for scan...</div>'; return; }
  list.innerHTML = _detected.map(d => {
    const danger = d.pred_moving && d.dist < 80;
    const cls    = danger ? "danger" : d.pred_moving ? "moving" : "";
    const sel    = d.id === _selectedId ? " selected" : "";
    return `<div class="obj-item ${cls}${sel}" onclick="_selectObj('${d.id}')">
      <div>
        <div class="obj-name" style="color:${d.color}">${d.icon} ${d.pred_name}</div>
        <div class="obj-meta">${d.dist}cm · ${d.time} · [${_activeML.toUpperCase()}] obj:${d.obj_conf}% mot:${d.mot_conf}%</div>
      </div>
      <span class="motion-tag ${d.pred_moving?"tag-moving":"tag-static"}">${d.pred_moving?"MOVING":"STATIC"}</span>
    </div>`;
  }).join("");
}

function _selectObj(id) {
  _selectedId = id;
  const d = _detected.find(x=>String(x.id)===String(id));
  if (d) _renderFeatures(d);
  _renderObjList();
}

function _renderFeatures(d) {
  _el("featTitle").textContent = ` — ${d.icon} ${d.pred_name}`;
  if (!d.features) return;
  _el("featGrid").innerHTML = Object.entries(d.features).map(([k,v])=>
    `<div class="feat-row"><span class="feat-k">${k}</span><span class="feat-v">${v}</span></div>`
  ).join("");
}

function _updateAlert() {
  const dangers = _detected.filter(d=>d.pred_moving && d.dist<80);
  const movers  = _detected.filter(d=>d.pred_moving);
  if      (dangers.length) _setAlert("danger",  `DANGER — Moving object within 80cm!`);
  else if (movers.length)  _setAlert("warning", `WARNING — ${movers.length} moving object(s) detected`);
  else                     _setAlert("clear",   `All Clear — ${_detected.length} object(s) in range`);
}

function _setAlert(level, msg) {
  const b=_el("alertBanner"), p=_el("alertPill");
  b.className="alert-banner"+(level!=="clear"?" "+level:"");
  b.textContent=msg||"All Clear";
  p.className="alert-pill"+(level!=="clear"?" "+level:"");
  p.textContent=level.toUpperCase();
}

// ── Accuracy panel from real results.json ─────────────────────────────────────

function _renderAccuracyPanel() {
  const res = Inference.getResults();
  if (!res) return;
  const container = _el("accuracyPanel");
  const cards = Object.entries(res).map(([key, r]) => {
    const o = r.object, m = r.motion;
    const active = key === _activeML;
    return `
    <div class="acc-card${active?" acc-active":""}" id="accCard_${key}" onclick="setML('${key}')">
      <div class="acc-card-header">
        <span class="acc-name" style="color:${key==="rf"?"#69f0ae":key==="gb"?"#ff9800":"#00b0ff"}">${r.name}</span>
        ${active?`<span class="acc-badge" style="background:${key==="rf"?"#69f0ae22":key==="gb"?"#ff980022":"#00b0ff22"};color:${key==="rf"?"#69f0ae":key==="gb"?"#ff9800":"#00b0ff"}">ACTIVE</span>`:""}
      </div>
      <div class="acc-section-label">Object Classification</div>
      <div class="acc-metrics">
        ${["Accuracy","F1 Score","Precision","Recall"].map((lb,i)=>{
          const val=[o.accuracy,o.f1,o.precision,o.recall][i];
          const c=key==="rf"?"#69f0ae":key==="gb"?"#ff9800":"#00b0ff";
          return `<div class="acc-metric">
            <div class="acc-metric-row"><span>${lb}</span><span style="color:${c};font-weight:700">${val}%</span></div>
            <div class="acc-bar-bg"><div class="acc-bar" style="width:${val}%;background:${c}"></div></div>
          </div>`;
        }).join("")}
      </div>
      <div class="acc-cv">5-Fold CV: ${o.cv_mean}% ± ${o.cv_std}%</div>
      <div class="acc-section-label" style="margin-top:8px">Motion Detection</div>
      <div class="acc-motion-row">
        <span>Accuracy</span><span style="color:#00e676;font-weight:700">${m.accuracy}%</span>
        <span>F1</span><span style="color:#00e676;font-weight:700">${m.f1}%</span>
      </div>
    </div>`;
  }).join("");
  container.innerHTML = cards;

  // Per-class table
  const firstKey = Object.keys(res)[0];
  const classes  = Object.keys(res[firstKey].object.per_class_f1);
  let tableHTML  = `<table class="cls-table"><thead><tr><th>Class</th>`;
  Object.values(res).forEach(r => { tableHTML += `<th style="color:${r.short==="RF"?"#69f0ae":r.short==="GB"?"#ff9800":"#00b0ff"}">${r.short}</th>`; });
  tableHTML += `</tr></thead><tbody>`;
  classes.forEach(cls => {
    tableHTML += `<tr><td>${OBJECT_ICONS[cls]||"·"} ${cls}</td>`;
    Object.values(res).forEach(r => {
      const v = r.object.per_class_f1[cls] || "—";
      tableHTML += `<td>${v}%</td>`;
    });
    tableHTML += `</tr>`;
  });
  tableHTML += `</tbody></table>`;
  _el("perClassTable").innerHTML = tableHTML;
}

function _highlightMLCard(key) {
  document.querySelectorAll(".acc-card").forEach(c => c.classList.remove("acc-active"));
  const card = _el(`accCard_${key}`);
  if (card) card.classList.add("acc-active");
  _renderAccuracyPanel();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function _el(id)  { return document.getElementById(id); }
function _fmt(s)  { return `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`; }

function _log(msg, type="info") {
  const box=_el("logBox"), div=document.createElement("div");
  div.className=`log-entry log-${type}`;
  div.innerHTML=`<span class="log-t">[${new Date().toLocaleTimeString("en-US",{hour12:false})}]</span>${msg}`;
  box.prepend(div);
  while(box.children.length>60) box.removeChild(box.lastChild);
}

// 🔥 LIVE DATA FROM PYTHON (results.json)
setInterval(async () => {
  try {
    const res = await fetch("../dashboard/models/results.json");
    const data = await res.json();

    // Convert Python output → dashboard format
    const det = {
      id: Date.now(),
      dist: Math.round(data.distance / 10), // mm → cm
      angle: data.angle,
      time: new Date().toLocaleTimeString(),

      pred_name: data.prediction === 1 ? "Object" : "Clear",
      pred_moving: Math.abs(data.velocity) > 5,

      obj_conf: 90,
      mot_conf: 85,

      features: {
        Distance: data.distance,
        Angle: data.angle,
        Velocity: data.velocity,
        Acceleration: data.acceleration
      }
    };

    // 🔥 Feed into dashboard
    _onDetect(det);

    // Update radar manually
    Sonar.render(det.angle, [{
      angle: det.angle,
      dist: det.dist
    }]);

    _el("badgeAngle").textContent = det.angle + "°";

  } catch (err) {
    console.log("Waiting for live data...");
  }
}, 200);