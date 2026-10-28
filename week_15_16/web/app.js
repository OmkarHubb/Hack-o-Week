/**
 * app.js - AI Antigravity Levitation Interactive Web Simulator
 * Implements real-time 60 FPS Canvas rendering, pure JS Neural Network inference
 * matching the NumPy MLP model, interactive mouse drag-and-drop, and real-time oscilloscopes.
 */

// Embedded weights from trained Antigravity Neural Network
const MODEL_DATA = {
  architecture: { hidden_sizes: [16, 16], activation: "relu" },
  scaler: {
    mean: [0.46432190220756164, -0.13241647055293873],
    std: [0.2438799750333401, 1.5851991953804]
  },
  layers: [
    {
      in_features: 2,
      out_features: 16,
      activation: "relu",
      weights: [
        [-0.2255, -0.0259, 0.6527, 0.9515, 1.4154, -0.5163, 0.5859, -0.0770, -0.9012, 1.2590, -0.5415, -0.5604, 0.9785, -1.4013, -1.4149, 0.3616],
        [-0.8590, 0.3050, -0.2758, -0.4895, 1.5199, -0.5393, -0.2395, -1.2041, -0.9498, 0.5690, -0.5611, 0.0823, 0.2178, -0.4855, -1.3584, 1.3896]
      ],
      biases: [0.6681, 0.9598, 0.5194, -0.1571, -0.1325, 0.2805, 2.1797, 0.1313, 0.4958, 1.9282, 0.2909, -0.6009, -1.0885, 0.1971, 0.5085, -0.5359]
    },
    {
      in_features: 16,
      out_features: 16,
      activation: "relu",
      weights: [
        [-0.0761, -0.6517, 1.4343, -0.2914, 0.2198, -0.1460, -0.0474, 0.4712, 0.5061, -0.2324, -2.1841, -0.0770, -0.2628, -1.1945, 0.1504, -0.2013],
        [1.2917, -0.3331, 0.3995, 0.2708, 0.1424, -0.1940, 0.5103, 0.5625, 0.1266, 0.4271, -0.0877, 1.2083, -0.6855, 0.8338, -1.2430, 0.7038],
        [0.4564, -0.2622, 0.0935, 0.7415, -0.6084, 0.4947, -0.0502, 0.7017, 0.8832, 0.1100, 0.8904, 0.6053, -0.2710, -0.8398, 0.7750, 0.0851],
        [0.2558, -0.0486, -0.9437, -1.0363, 0.0142, -0.2817, 0.0082, -0.0360, -0.6367, -0.4835, 0.4153, 0.4774, 1.2721, -0.1652, 1.0242, -0.3202],
        [-0.4240, -0.5593, 0.3188, -0.4994, -1.2628, -0.5960, -0.2713, -0.7949, 1.6136, -1.3466, -0.9094, -1.0594, -0.3769, -0.6213, -3.3238, -0.4621],
        [0.4511, -0.2302, -2.0778, 0.1631, 0.5724, -0.6457, 0.4589, -0.3840, -0.4122, -0.0289, 0.0262, 0.1933, 0.2011, 0.0575, -0.0595, -0.0447],
        [0.1584, 0.1788, 1.4478, 1.1112, 1.3002, -0.7631, -0.1955, 0.6650, -0.4777, 0.5196, -1.5938, 0.9937, -3.1685, 1.5016, -0.5766, 0.8609],
        [-0.1960, 0.2291, -0.3830, -0.1336, 0.1434, -0.0634, -0.0019, -0.4915, 0.1112, 0.5278, -8.0277, -0.2341, 0.1750, -0.3800, 0.7812, -0.2667],
        [0.9587, 0.1999, -5.6313, 0.2124, 0.1924, -0.3600, 0.0928, -0.3217, -0.1207, -1.0097, 0.2820, 0.2519, 0.2043, 0.1092, 0.0276, 0.4752],
        [-0.6467, 0.1730, -1.7477, -1.0597, -1.0225, 0.3276, -0.6284, -1.4945, -0.0005, 1.3183, 0.1279, -1.6172, 2.2407, -2.1962, -0.1650, -1.2377],
        [0.2874, -0.5539, -1.2835, 0.3425, -0.0059, -0.2651, -0.1936, 0.2070, 0.8788, 0.5022, 0.6888, -0.0169, 0.2488, -0.3481, 0.2850, 0.4323],
        [1.0743, -0.1390, 0.2559, 0.2426, -0.2319, 0.1969, -0.2345, 0.8542, 0.9712, 0.7021, -0.7105, -0.4353, 0.2424, -0.0386, 2.0684, 0.2303],
        [0.8514, -0.5252, 0.3151, -0.1139, -0.2548, -0.4326, 0.1348, -2.7976, -0.2449, -0.2464, -0.0857, -1.0600, -2.1491, -0.4526, 0.1298, -0.0845],
        [0.1265, -0.0618, 0.0534, 0.3530, 0.9631, -0.0597, -0.3550, -0.5717, -0.4014, 0.0143, 0.6119, 0.0545, 0.0972, -0.2839, -0.0527, 0.1908],
        [-0.7420, 0.1773, -2.4387, 0.2407, 0.0509, 0.0503, -0.4379, 0.5226, 0.2846, 0.0691, 0.5606, -0.0596, -1.2837, 0.4216, 0.5823, 0.0483],
        [0.5472, 0.2129, 0.5744, -0.6348, -0.6479, 0.0322, 0.2950, 0.5498, 0.4584, -0.3678, 0.1060, -0.5685, 0.3120, -0.0115, -0.8744, -0.6726]
      ],
      biases: [1.0821, -0.1993, 0.6912, 0.6650, 0.5836, -0.1395, -0.0210, 0.6004, -0.3527, -0.2291, 0.0735, 0.8740, -0.3632, 0.8637, -0.5188, 0.6577]
    },
    {
      in_features: 16,
      out_features: 1,
      activation: "linear",
      weights: [
        [0.3181], [-0.0196], [-0.7710], [0.4912], [0.3974], [-0.2054], [0.0016], [0.1661],
        [-0.0957], [-0.1777], [0.1947], [0.2888], [-0.0073], [0.3612], [-0.2315], [0.3285]
      ],
      biases: [0.5931]
    }
  ],
  physics: {
    H: 1.0,
    y_target: 0.5,
    m: 0.05,
    g: 9.81,
    k_mag: 0.030656,
    c_drag: 0.015,
    eps: 0.02,
    max_current: 10.0,
    equilibrium_current: 2.08,
    dt: 0.005
  }
};

// =====================================================================
// Neural Network Forward Inference Engine (Pure JS)
// =====================================================================
class AntigravityJSModel {
  constructor(modelData) {
    this.data = modelData;
    this.mean = modelData.scaler.mean;
    this.std = modelData.scaler.std;
    this.layers = modelData.layers;
  }

  predict(y, v) {
    // 1. Standardize features
    const x0 = (y - this.mean[0]) / this.std[0];
    const x1 = (v - this.mean[1]) / this.std[1];
    let act = [x0, x1];

    // 2. Layer 1 (Dense 2 -> 16, ReLU)
    const l1 = this.layers[0];
    let next1 = new Array(l1.out_features);
    for (let j = 0; j < l1.out_features; j++) {
      let sum = l1.biases[j];
      for (let i = 0; i < l1.in_features; i++) {
        sum += act[i] * l1.weights[i][j];
      }
      next1[j] = Math.max(0, sum); // ReLU
    }

    // 3. Layer 2 (Dense 16 -> 16, ReLU)
    const l2 = this.layers[1];
    let next2 = new Array(l2.out_features);
    for (let j = 0; j < l2.out_features; j++) {
      let sum = l2.biases[j];
      for (let i = 0; i < l2.in_features; i++) {
        sum += next1[i] * l2.weights[i][j];
      }
      next2[j] = Math.max(0, sum); // ReLU
    }

    // 4. Output Layer (Dense 16 -> 1, Linear)
    const l3 = this.layers[2];
    let out = l3.biases[0];
    for (let i = 0; i < l3.in_features; i++) {
      out += next2[i] * l3.weights[i][0];
    }

    // Current cannot be negative, and clamped to max_current
    return Math.min(Math.max(0.0, out), this.data.physics.max_current);
  }
}

// =====================================================================
// Web Audio API Synthesizer (Magnetic Hum Sound Effect)
// =====================================================================
class SoundSynth {
  constructor() {
    this.ctx = null;
    this.osc = null;
    this.gain = null;
    this.filter = null;
    this.enabled = false;
  }

  init() {
    if (this.ctx) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioContext();

    this.osc = this.ctx.createOscillator();
    this.osc.type = "sawtooth";

    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 180;

    this.gain = this.ctx.createGain();
    this.gain.gain.value = 0.0;

    this.osc.connect(this.filter);
    this.filter.connect(this.gain);
    this.gain.connect(this.ctx.destination);
    this.osc.start();
  }

  update(current, maxCurrent) {
    if (!this.enabled || !this.ctx) return;
    const norm = Math.min(1.0, current / maxCurrent);
    // Pitch rises with current (100 Hz to 350 Hz)
    const targetFreq = 90 + norm * 260;
    this.osc.frequency.setTargetAtTime(targetFreq, this.ctx.currentTime, 0.05);
    // Volume rises with current
    const targetGain = 0.02 + norm * 0.08;
    this.gain.gain.setTargetAtTime(targetGain, this.ctx.currentTime, 0.05);
  }

  toggle() {
    this.init();
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    this.enabled = !this.enabled;
    if (this.gain) {
      this.gain.gain.value = this.enabled ? 0.03 : 0.0;
    }
    return this.enabled;
  }
}

// =====================================================================
// Main Simulation Engine
// =====================================================================
const aiModel = new AntigravityJSModel(MODEL_DATA);
const audio = new SoundSynth();

// Physical state
const phys = {
  H: 1.0,
  y_target: 0.50,
  m: 0.050,
  g: 9.81,
  k_mag: 0.030656,
  c_drag: 0.015,
  eps: 0.02,
  max_current: 10.0,
  restitution: 0.15,
  y: 0.50,
  v: 0.0,
  current: 2.08,
  f_mag: 0.49,
  f_grav: 0.49,
  isDragging: false,
  dragStartY: 0,
  lastMouseY: 0,
  lastMouseTime: 0,
  modelType: "crash_penalty", // "crash_penalty", "mse", "analytic"
  particles: []
};

// Pre-populate magnetic field particles
for (let i = 0; i < 40; i++) {
  phys.particles.push({
    x: (Math.random() - 0.5) * 0.8,
    y: Math.random() * 0.9,
    speed: 0.003 + Math.random() * 0.007,
    alpha: Math.random() * 0.6
  });
}

// Oscilloscope History Buffers
const HISTORY_LEN = 180;
const historyY = new Array(HISTORY_LEN).fill(0.5);
const historyI = new Array(HISTORY_LEN).fill(2.08);
const historyV = new Array(HISTORY_LEN).fill(0.0);

// Canvas contexts
const chamberCanvas = document.getElementById("chamberCanvas");
const ctxChamber = chamberCanvas.getContext("2d");

const scopeY = document.getElementById("scopeY");
const ctxScopeY = scopeY.getContext("2d");

const scopeI = document.getElementById("scopeI");
const ctxScopeI = scopeI.getContext("2d");

const scopeV = document.getElementById("scopeV");
const ctxScopeV = scopeV.getContext("2d");

// DOM Readouts
const valY = document.getElementById("valY");
const valI = document.getElementById("valI");
const valV = document.getElementById("valV");
const statusBadge = document.getElementById("statusBadge");
const targetReadout = document.getElementById("targetReadout");
const errorReadout = document.getElementById("errorReadout");
const fmagReadout = document.getElementById("fmagReadout");
const fgravReadout = document.getElementById("fgravReadout");
const modelActiveBadge = document.getElementById("modelActiveBadge");

// Controls
const modelSelect = document.getElementById("modelSelect");
const targetSlider = document.getElementById("targetSlider");
const gravitySlider = document.getElementById("gravitySlider");
const massSlider = document.getElementById("massSlider");
const soundToggleBtn = document.getElementById("soundToggleBtn");
const dropHazardBtn = document.getElementById("dropHazardBtn");
const dropHighBtn = document.getElementById("dropHighBtn");
const gustBtn = document.getElementById("gustBtn");
const clearScopeBtn = document.getElementById("clearScopeBtn");

// =====================================================================
// Physics Calculations
// =====================================================================

function calcMagneticForce(y, current) {
  const i = Math.max(0, Math.min(current, phys.max_current));
  const dist = Math.max(phys.eps, phys.H - y + phys.eps);
  return (phys.k_mag * (i * i)) / (dist * dist);
}

function calcIdealCurrent(y, v) {
  const omega_n = 12.0;
  const zeta = 1.0;
  const e = y - phys.y_target;
  const a_desired = -2.0 * zeta * omega_n * v - (omega_n * omega_n) * e;
  const f_mag_needed = phys.m * (a_desired + phys.g) + phys.c_drag * v;
  if (f_mag_needed <= 0.0) return 0.0;
  const dist = Math.max(phys.eps, phys.H - y + phys.eps);
  const i_needed = dist * Math.sqrt(f_mag_needed / phys.k_mag);
  return Math.min(phys.max_current, Math.max(0, i_needed));
}

function stepPhysics(dt) {
  if (phys.isDragging) return;

  // 1. Controller Inference
  let cmdCurrent = 0;
  if (phys.modelType === "analytic") {
    cmdCurrent = calcIdealCurrent(phys.y, phys.v);
  } else {
    cmdCurrent = aiModel.predict(phys.y, phys.v);
  }
  phys.current = cmdCurrent;

  // 2. Compute Forces
  phys.f_mag = calcMagneticForce(phys.y, cmdCurrent);
  phys.f_grav = phys.m * phys.g;
  const f_drag = phys.c_drag * phys.v;
  const a_net = (phys.f_mag - phys.f_grav - f_drag) / phys.m;

  // 3. RK4 / Semi-implicit Integration
  phys.v += a_net * dt;
  phys.y += phys.v * dt;

  // 4. Boundary Collision Constraints
  if (phys.y <= 0.015) {
    phys.y = 0.015;
    if (phys.v < 0) {
      phys.v = -phys.v * phys.restitution;
      if (Math.abs(phys.v) < 0.05) phys.v = 0.0;
    }
  } else if (phys.y >= phys.H - 0.02) {
    phys.y = phys.H - 0.02;
    if (phys.v > 0) {
      phys.v = -phys.v * phys.restitution;
      if (Math.abs(phys.v) < 0.05) phys.v = 0.0;
    }
  }

  audio.update(phys.current, phys.max_current);
}

// =====================================================================
// Canvas Rendering
// =====================================================================

function worldToScreenY(y) {
  const floorY = chamberCanvas.height - 40;
  const ceilingY = 70;
  const heightSpan = floorY - ceilingY;
  return floorY - (y / phys.H) * heightSpan;
}

function screenToWorldY(pixelY) {
  const floorY = chamberCanvas.height - 40;
  const ceilingY = 70;
  const heightSpan = floorY - ceilingY;
  const norm = (floorY - pixelY) / heightSpan;
  return Math.min(phys.H, Math.max(0.02, norm * phys.H));
}

function renderChamber() {
  const W = chamberCanvas.width;
  const H = chamberCanvas.height;
  ctxChamber.clearRect(0, 0, W, H);

  const centerX = W / 2;
  const floorY = chamberCanvas.height - 40;
  const ceilingY = 70;

  // 1. Chamber Background Grid
  ctxChamber.fillStyle = "#090d16";
  ctxChamber.fillRect(0, 0, W, H);

  // Background subtle grid lines
  ctxChamber.strokeStyle = "rgba(56, 189, 248, 0.08)";
  ctxChamber.lineWidth = 1;
  for (let y = 0.1; y < 1.0; y += 0.1) {
    const py = worldToScreenY(y);
    ctxChamber.beginPath();
    ctxChamber.moveTo(40, py);
    ctxChamber.lineTo(W - 40, py);
    ctxChamber.stroke();

    // Height Labels
    ctxChamber.fillStyle = "rgba(148, 163, 184, 0.4)";
    ctxChamber.font = "10px JetBrains Mono";
    ctxChamber.fillText(`${y.toFixed(1)}m`, 10, py + 3);
  }

  // 2. Glass Tube Walls
  ctxChamber.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctxChamber.lineWidth = 2;
  ctxChamber.strokeRect(centerX - 130, ceilingY, 260, floorY - ceilingY);

  // 3. Dynamic Magnetic Field Lines & Particles
  const fluxAlpha = Math.min(0.85, 0.1 + (phys.current / phys.max_current) * 0.75);
  ctxChamber.save();
  ctxChamber.strokeStyle = `rgba(192, 132, 252, ${fluxAlpha * 0.4})`;
  ctxChamber.lineWidth = 1.5;
  for (let r = 40; r <= 140; r += 25) {
    ctxChamber.beginPath();
    ctxChamber.ellipse(centerX, ceilingY, r, r * 1.5, 0, 0, Math.PI);
    ctxChamber.stroke();
  }
  ctxChamber.restore();

  // Floating Magnetic Particles pulled up
  for (let p of phys.particles) {
    p.y += p.speed * (1.0 + phys.current * 0.5);
    if (p.y > 1.0) p.y = 0.0;
    const px = centerX + p.x * 200;
    const py = worldToScreenY(p.y);
    ctxChamber.fillStyle = `rgba(192, 132, 252, ${p.alpha * fluxAlpha})`;
    ctxChamber.beginPath();
    ctxChamber.arc(px, py, 1.8, 0, Math.PI * 2);
    ctxChamber.fill();
  }

  // 4. Target Hover Plane (0.50m)
  const targetPy = worldToScreenY(phys.y_target);
  // Translucent green tolerance corridor
  ctxChamber.fillStyle = "rgba(16, 185, 129, 0.12)";
  ctxChamber.fillRect(centerX - 128, targetPy - 12, 256, 24);

  // Dashed glowing green target line
  ctxChamber.save();
  ctxChamber.strokeStyle = "#10b981";
  ctxChamber.lineWidth = 2;
  ctxChamber.setLineDash([8, 6]);
  ctxChamber.shadowColor = "#10b981";
  ctxChamber.shadowBlur = 10;
  ctxChamber.beginPath();
  ctxChamber.moveTo(centerX - 128, targetPy);
  ctxChamber.lineTo(centerX + 128, targetPy);
  ctxChamber.stroke();
  ctxChamber.restore();

  // Target label
  ctxChamber.fillStyle = "#34d399";
  ctxChamber.font = "bold 11px Rajdhani";
  ctxChamber.fillText(`✦ TARGET HOVER: ${phys.y_target.toFixed(2)}m`, centerX - 60, targetPy - 16);

  // 5. Ceiling Electromagnet Solenoid
  const coilGlow = Math.min(1.0, phys.current / phys.max_current);
  ctxChamber.fillStyle = "#1e1136";
  ctxChamber.strokeStyle = "#8b5cf6";
  ctxChamber.lineWidth = 2;
  ctxChamber.fillRect(centerX - 90, ceilingY - 35, 180, 35);
  ctxChamber.strokeRect(centerX - 90, ceilingY - 35, 180, 35);

  // Coils with dynamic brightness
  for (let cx = centerX - 75; cx <= centerX + 75; cx += 22) {
    ctxChamber.fillStyle = `rgba(216, 180, 254, ${0.4 + coilGlow * 0.6})`;
    ctxChamber.beginPath();
    ctxChamber.ellipse(cx, ceilingY - 18, 8, 14, 0, 0, Math.PI * 2);
    ctxChamber.fill();
  }

  ctxChamber.fillStyle = "#e9d5ff";
  ctxChamber.font = "bold 10px Orbitron";
  ctxChamber.textAlign = "center";
  ctxChamber.fillText("ELECTROMAGNET", centerX, ceilingY - 40);

  // 6. Hazard Floor Stripes (Crash Boundary)
  ctxChamber.fillStyle = "#7f1d1d";
  ctxChamber.fillRect(centerX - 130, floorY, 260, 25);
  ctxChamber.strokeStyle = "#ef4444";
  ctxChamber.lineWidth = 2;
  ctxChamber.strokeRect(centerX - 130, floorY, 260, 25);

  // Hazard diagonal lines
  ctxChamber.save();
  ctxChamber.strokeStyle = "#fca5a5";
  ctxChamber.lineWidth = 3;
  for (let x = centerX - 120; x < centerX + 130; x += 20) {
    ctxChamber.beginPath();
    ctxChamber.moveTo(x, floorY + 23);
    ctxChamber.lineTo(x + 12, floorY + 2);
    ctxChamber.stroke();
  }
  ctxChamber.restore();

  ctxChamber.fillStyle = "#fef2f2";
  ctxChamber.font = "bold 9px Orbitron";
  ctxChamber.fillText("▲ CRASH FLOOR y = 0.0 m ▲", centerX, floorY + 36);

  // 7. Ball Position & Laser Beam
  const ballPy = worldToScreenY(phys.y);
  const ballRadius = 18;

  // Red Laser Distance Tracker Beam
  ctxChamber.strokeStyle = "rgba(239, 68, 68, 0.75)";
  ctxChamber.lineWidth = 1.5;
  ctxChamber.setLineDash([4, 4]);
  ctxChamber.beginPath();
  ctxChamber.moveTo(centerX, ceilingY);
  ctxChamber.lineTo(centerX, ballPy - ballRadius);
  ctxChamber.stroke();
  ctxChamber.setLineDash([]);

  // Laser target dot on top of ball
  ctxChamber.fillStyle = "#f87171";
  ctxChamber.beginPath();
  ctxChamber.arc(centerX, ballPy - ballRadius, 3.5, 0, Math.PI * 2);
  ctxChamber.fill();

  // Floor Shadow (reacts to distance from floor)
  const distFromFloor = Math.max(0.01, phys.y);
  const shadowAlpha = Math.min(0.75, Math.max(0.1, 0.85 * Math.exp(-distFromFloor / 0.18)));
  const shadowScale = Math.min(30, Math.max(8, 28 - distFromFloor * 20));
  ctxChamber.fillStyle = `rgba(0, 0, 0, ${shadowAlpha})`;
  ctxChamber.beginPath();
  ctxChamber.ellipse(centerX, floorY, shadowScale, 6, 0, 0, Math.PI * 2);
  ctxChamber.fill();

  // 8. Metallic Levitating Sphere
  const grad = ctxChamber.createRadialGradient(
    centerX - 5, ballPy - 6, 2,
    centerX, ballPy, ballRadius
  );
  grad.addColorStop(0, "#ffffff");
  grad.addColorStop(0.35, "#cbd5e1");
  grad.addColorStop(0.85, "#475569");
  grad.addColorStop(1, "#1e293b");

  ctxChamber.save();
  ctxChamber.shadowColor = "rgba(0, 242, 254, 0.3)";
  ctxChamber.shadowBlur = 15;
  ctxChamber.fillStyle = grad;
  ctxChamber.beginPath();
  ctxChamber.arc(centerX, ballPy, ballRadius, 0, Math.PI * 2);
  ctxChamber.fill();
  ctxChamber.strokeStyle = "#94a3b8";
  ctxChamber.lineWidth = 1.5;
  ctxChamber.stroke();
  ctxChamber.restore();

  // 9. Force Vector Arrows
  // Upward Magnetic Pull (Purple)
  const magArrowLen = Math.min(70, phys.f_mag * 50);
  if (magArrowLen > 5) {
    drawArrow(ctxChamber, centerX + 30, ballPy, centerX + 30, ballPy - magArrowLen, "#c084fc", "F_mag");
  }

  // Downward Gravity Pull (Red)
  const gravArrowLen = Math.min(70, phys.f_grav * 50);
  drawArrow(ctxChamber, centerX - 30, ballPy, centerX - 30, ballPy + gravArrowLen, "#f87171", "F_g");
}

function drawArrow(ctx, fromX, fromY, toX, toY, color, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrowhead
  const headLen = 8;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();

  ctx.font = "bold 9px JetBrains Mono";
  ctx.fillText(label, toX + (fromX < toX ? 6 : -6), toY);
  ctx.restore();
}

function renderScope(canvas, ctx, data, minVal, maxVal, lineColor, targetVal = null) {
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background Grid
  ctx.fillStyle = "#060913";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H * 0.25); ctx.lineTo(W, H * 0.25);
  ctx.moveTo(0, H * 0.5); ctx.lineTo(W, H * 0.5);
  ctx.moveTo(0, H * 0.75); ctx.lineTo(W, H * 0.75);
  ctx.stroke();

  // Target Reference line
  if (targetVal !== null) {
    const normTarget = (targetVal - minVal) / (maxVal - minVal);
    const py = H - normTarget * H;
    ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(W, py);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Live Curve
  ctx.save();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2.2;
  ctx.shadowColor = lineColor;
  ctx.shadowBlur = 8;

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const px = (i / (data.length - 1)) * W;
    const norm = (data[i] - minVal) / (maxVal - minVal);
    const py = Math.min(H - 2, Math.max(2, H - norm * H));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Rightmost Cursor Dot
  const lastVal = data[data.length - 1];
  const lastNorm = (lastVal - minVal) / (maxVal - minVal);
  const lastPy = Math.min(H - 2, Math.max(2, H - lastNorm * H));
  ctx.fillStyle = lineColor;
  ctx.beginPath();
  ctx.arc(W - 4, lastPy, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

// =====================================================================
// Interactive Mouse & Touch Dragging
// =====================================================================
function setupInteractions() {
  const getEventY = (e) => {
    const rect = chamberCanvas.getBoundingClientRect();
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return clientY - rect.top;
  };

  const getEventX = (e) => {
    const rect = chamberCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return clientX - rect.left;
  };

  const onStart = (e) => {
    const px = getEventX(e);
    const py = getEventY(e);
    const ballPy = worldToScreenY(phys.y);
    const centerX = chamberCanvas.width / 2;

    const dist = Math.hypot(px - centerX, py - ballPy);
    if (dist < 45) {
      phys.isDragging = true;
      phys.lastMouseY = py;
      phys.lastMouseTime = performance.now();
      phys.v = 0.0;
    }
  };

  const onMove = (e) => {
    if (!phys.isDragging) return;
    const py = getEventY(e);
    const now = performance.now();
    const dt = (now - phys.lastMouseTime) / 1000;

    const newY = screenToWorldY(py);
    if (dt > 0.01) {
      phys.v = (newY - phys.y) / dt;
      phys.lastMouseY = py;
      phys.lastMouseTime = now;
    }
    phys.y = newY;
  };

  const onEnd = () => {
    if (phys.isDragging) {
      phys.isDragging = false;
      // Cap throw velocity for stability
      phys.v = Math.max(-4.0, Math.min(4.0, phys.v));
    }
  };

  chamberCanvas.addEventListener("mousedown", onStart);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onEnd);

  chamberCanvas.addEventListener("touchstart", onStart, { passive: true });
  window.addEventListener("touchmove", onMove, { passive: true });
  window.addEventListener("touchend", onEnd);

  // Preset Buttons
  dropHazardBtn.addEventListener("click", () => {
    phys.y = 0.12;
    phys.v = -0.5;
  });

  dropHighBtn.addEventListener("click", () => {
    phys.y = 0.88;
    phys.v = 0.0;
  });

  gustBtn.addEventListener("click", () => {
    // Inject sudden downward or upward wind burst
    const gust = (Math.random() > 0.5 ? 1 : -1) * (1.8 + Math.random() * 1.5);
    phys.v += gust;
  });

  soundToggleBtn.addEventListener("click", () => {
    const on = audio.toggle();
    soundToggleBtn.textContent = on ? "🔊 Audio Hum: ON" : "🔇 Audio Hum: OFF";
    soundToggleBtn.style.color = on ? "#10b981" : "#e9d5ff";
  });

  clearScopeBtn.addEventListener("click", () => {
    historyY.fill(phys.y);
    historyI.fill(phys.current);
    historyV.fill(phys.v);
  });

  // Parameter sliders
  targetSlider.addEventListener("input", (e) => {
    phys.y_target = parseFloat(e.target.value);
    document.getElementById("targetSliderVal").textContent = `${phys.y_target.toFixed(2)} m`;
    targetReadout.textContent = `${phys.y_target.toFixed(2)} m`;
  });

  gravitySlider.addEventListener("input", (e) => {
    phys.g = parseFloat(e.target.value);
    document.getElementById("gravitySliderVal").textContent = `${phys.g.toFixed(2)} m/s²`;
  });

  massSlider.addEventListener("input", (e) => {
    phys.m = parseFloat(e.target.value);
    document.getElementById("massSliderVal").textContent = `${phys.m.toFixed(3)} kg`;
  });

  modelSelect.addEventListener("change", (e) => {
    phys.modelType = e.target.value;
    const map = {
      crash_penalty: "Crash-Loss Model",
      mse: "Standard MSE Model",
      analytic: "Feedback Linearization"
    };
    modelActiveBadge.textContent = map[phys.modelType];
  });
}

// =====================================================================
// Main Animation & Simulation Loop (60 FPS)
// =====================================================================
let lastTime = performance.now();

function loop(currentTime) {
  const dtSec = Math.min(0.04, (currentTime - lastTime) / 1000);
  lastTime = currentTime;

  // Run multiple smaller sub-steps (200 Hz internal physics clock)
  const subSteps = 4;
  const dtSub = dtSec / subSteps;
  for (let s = 0; s < subSteps; s++) {
    stepPhysics(dtSub);
  }

  // Update Telemetry Buffers
  historyY.shift();
  historyY.push(phys.y);

  historyI.shift();
  historyI.push(phys.current);

  historyV.shift();
  historyV.push(phys.v);

  // Render Chamber and Oscilloscopes
  renderChamber();
  renderScope(scopeY, ctxScopeY, historyY, 0.0, 1.0, "#38bdf8", phys.y_target);
  renderScope(scopeI, ctxScopeI, historyI, 0.0, 10.0, "#f43f5e", 2.08);
  renderScope(scopeV, ctxScopeV, historyV, -3.5, 3.5, "#22c55e", 0.0);

  // Update UI Telemetry Values
  valY.textContent = `${phys.y.toFixed(3)} m`;
  valI.textContent = `${phys.current.toFixed(2)} A`;
  valV.textContent = `${(phys.v >= 0 ? "+" : "") + phys.v.toFixed(2)} m/s`;

  const err = Math.abs(phys.y - phys.y_target);
  errorReadout.textContent = `${(err * 1000).toFixed(1)} mm`;
  fmagReadout.textContent = `${phys.f_mag.toFixed(2)} N`;
  fgravReadout.textContent = `${phys.f_grav.toFixed(2)} N`;

  // Status Badge Logic
  if (phys.isDragging) {
    statusBadge.textContent = "👆 USER GRAB & DRAG ACTIVE";
    statusBadge.style.color = "#c084fc";
  } else if (err < 0.012 && Math.abs(phys.v) < 0.05) {
    statusBadge.textContent = "✦ HOVER LOCKED (STABLE)";
    statusBadge.style.color = "#10b981";
  } else if (phys.y <= 0.02) {
    statusBadge.textContent = "▲ CRASH ON FLOOR ▲";
    statusBadge.style.color = "#ef4444";
  } else if (phys.v < -0.3 && phys.y < phys.y_target) {
    statusBadge.textContent = "⚡ AI EMERGENCY BRAKING BOOST";
    statusBadge.style.color = "#f43f5e";
  } else if (phys.y > phys.y_target + 0.08) {
    statusBadge.textContent = "▼ GRAVITATIONAL RESTORE";
    statusBadge.style.color = "#fbbf24";
  } else {
    statusBadge.textContent = "✦ CATCHING IN MID-AIR";
    statusBadge.style.color = "#38bdf8";
  }

  requestAnimationFrame(loop);
}

setupInteractions();
requestAnimationFrame(loop);
