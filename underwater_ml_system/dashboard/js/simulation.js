/**
 * simulation.js
 * ─────────────────────────────────────────────────────────────
 * Simulates HC-SR04 sonar sweep and feeds real features
 * into the trained ML inference engine.
 *
 * Each sweep detection:
 *   1. Computes 8 real features from object physics
 *   2. Passes them to Inference.predict()
 *   3. Gets back: object_name, is_moving, confidence
 *   4. Fires onDetect callback with full result
 * ─────────────────────────────────────────────────────────────
 */

const Simulation = (() => {

  const OBJECT_DEFS = [
    { id:0, name:"Fish School",    icon:"🐟", color:"#00e5ff", dist:[30,150],  moveProb:0.85, speed:[0.8,4.0],  amp:[0.25,0.55], echo:[3,8],  noise:[0.08,0.20] },
    { id:1, name:"Rock/Reef",      icon:"🪨", color:"#8d6e63", dist:[50,200],  moveProb:0.0,  speed:[0,0],     amp:[0.72,0.97], echo:[1,2],  noise:[0.02,0.07] },
    { id:2, name:"Submarine",      icon:"🚢", color:"#ff1744", dist:[40,200],  moveProb:0.75, speed:[2.5,9.0],  amp:[0.88,0.99], echo:[1,2],  noise:[0.02,0.09] },
    { id:3, name:"Diver",          icon:"🤿", color:"#ff6d00", dist:[15,130],  moveProb:0.70, speed:[0.4,2.5],  amp:[0.38,0.72], echo:[2,5],  noise:[0.06,0.18] },
    { id:4, name:"Debris",         icon:"🗑️", color:"#ffd600", dist:[20,160],  moveProb:0.30, speed:[0.1,1.2],  amp:[0.18,0.60], echo:[3,9],  noise:[0.12,0.30] },
    { id:5, name:"Marine Mammal",  icon:"🐬", color:"#00e676", dist:[30,175],  moveProb:0.90, speed:[1.5,7.0],  amp:[0.45,0.82], echo:[2,4],  noise:[0.04,0.13] },
  ];

  let objects    = [];
  let angle      = 0;
  let sweepSpeed = 2;
  let running    = false;
  let raf        = null;
  let activeML   = "rf";
  let onDetect   = null;
  let onFrame    = null;
  const ECHO_TTL = 220;
  const TOL      = 0.10;

  function init(cb) { onDetect = cb.onDetect; onFrame = cb.onFrame; }

  function start() {
    if (running) return;
    running = true;
    if (!objects.length) for (let i=0; i<6; i++) _spawn();
    _loop();
  }

  function stop()  { running=false; if(raf){cancelAnimationFrame(raf);raf=null;} }
  function clear() { stop(); objects=[]; angle=0; }
  function addObject() { _spawn(); }
  function setSpeed(v) { sweepSpeed=+v; }
  function setML(k)    { activeML=k; }
  function getAngle()  { return angle; }
  function getObjects(){ return objects; }

  function _loop() {
    if (!running) return;
    objects = objects.map(_tick);
    angle   = (angle + sweepSpeed * 0.4) % 360;
    if (onFrame) onFrame(angle, objects);
    raf = requestAnimationFrame(_loop);
  }

  function _tick(obj) {
    let {a, r, prevR, echoes} = obj;
    const speed = obj.movSpeed || 0;

    if (obj.moving) {
      a += speed * 0.003;
      r  = Math.max(10, Math.min(200, r + (Math.random()-.5)*speed*0.2));
    }

    echoes = echoes.filter(e=>{ e.life--; return e.life>0; });

    const oa = _norm(a), sa = _norm(angle * Math.PI/180);
    const df = Math.abs(sa-oa);
    const hit = df < TOL || Math.abs(df-Math.PI*2) < TOL;

    if (hit && Inference.isReady()) {
      echoes.push({r, a, life: ECHO_TTL});
      const features = _extractFeatures(obj, r, prevR);
      const pred     = Inference.predict(features, activeML);
      if (pred && onDetect) {
        onDetect({
          id:          obj.id,
          true_name:   obj.name,
          true_icon:   obj.icon,
          true_moving: obj.moving,
          pred_name:   pred.object_name,
          pred_moving: pred.is_moving,
          obj_conf:    pred.object_conf,
          mot_conf:    pred.motion_conf,
          dist:        Math.round(r),
          color:       obj.color,
          features,
          time: new Date().toLocaleTimeString("en-US",{hour12:false}),
        });
      }
    }
    return {...obj, a, r, prevR:r, echoes};
  }

  function _extractFeatures(obj, r, prevR) {
    const d      = obj.def;
    const moving = obj.moving;
    const speed  = obj.movSpeed || 0;
    const delta  = prevR != null ? r - prevR + _rnd(0.05) : (moving ? _rnd(speed*0.15) : _rnd(0.03));
    const doppler= moving ? Math.sign(Math.random()-.5)*speed*0.9+_rnd(0.18) : _rnd(0.12);
    return {
      echo_duration_ms:  +(r*2/34.3 + _rnd(0.025)).toFixed(4),
      signal_amplitude:  +Math.max(0.01, Math.min(1, _rng(...d.amp)*(1-r/420)+_rnd(0.03))).toFixed(4),
      distance_cm:       +r.toFixed(2),
      distance_delta:    +delta.toFixed(4),
      angular_velocity:  +Math.abs(moving ? speed*0.75+_rnd(0.12) : _rnd(0.025)).toFixed(4),
      echo_count:        Math.floor(_rng(d.echo[0], d.echo[1])),
      noise_ratio:       +_rng(...d.noise).toFixed(4),
      doppler_shift_hz:  +doppler.toFixed(4),
    };
  }

  function _spawn() {
    const d   = OBJECT_DEFS[Math.floor(Math.random()*OBJECT_DEFS.length)];
    const mov = Math.random() < d.moveProb;
    objects.push({
      id:      Date.now()+Math.random(),
      name:    d.name, icon:  d.icon, color: d.color,
      def:     d, moving: mov,
      movSpeed: mov ? _rng(...d.speed) : 0,
      r:       _rng(...d.dist), a: Math.random()*Math.PI*2,
      prevR:   null, sz: 5+Math.random()*5, echoes: [],
    });
  }

  function _norm(r){ return ((r%(Math.PI*2))+Math.PI*2)%(Math.PI*2); }
  function _rnd(s) { return (Math.random()-.5)*2*s; }
  function _rng(a,b){ return a+Math.random()*(b-a); }

  return { init, start, stop, clear, addObject, setSpeed, setML, getAngle, getObjects };

})();
