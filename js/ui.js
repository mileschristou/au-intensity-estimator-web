/**
 * ui.js — AU Intensity Estimator
 *
 * Controls all three modes (Webcam / Image / Video), the canvas overlay,
 * the AU panel, GPU/CPU toggle, and frame recording.
 */

import {
  initFaceAligner, alignFace, setRunningMode,
} from './face_align.js';

import {
  initModel, predictNamed, getAUNames, isReady, setDeviceLostHandler,
} from './inference.js';

import { downloadCSV, downloadXLSX } from './export.js';

// ─── Config ───────────────────────────────────────────────────────────────────

const HF_BASE = 'https://huggingface.co/mileschristou/au-intensity-estimator/resolve/main/';

// Shared fallback config for all DINOv3 models (same preprocessing + AU set)
const _DINO_CONFIG = {
  au_names: ['AU01','AU02','AU04','AU05','AU06','AU09','AU12','AU15','AU17','AU20','AU25','AU26'],
  num_aus: 12, resize: 256, crop: 224,
  norm_mean: [0.485, 0.456, 0.406], norm_std: [0.229, 0.224, 0.225],
  align: { output_size: 224, eye_dist_ratio: 0.30, eye_center_x: 0.50, eye_center_y: 0.40,
           left_eye_landmark: 33, right_eye_landmark: 263 },
};

const MODELS = {
  resnet18: {
    label:     'ResNet-18',
    size:      '49 MB',
    modelUrl:  HF_BASE + 'resnet18_model_fold1_single.onnx',
    configUrl: HF_BASE + 'resnet18_model_fold1_config.json',
  },
  vitb: {
    label:        'DINOv3 ViT-B',
    size:         '354 MB',
    modelUrl:     HF_BASE + 'model_fold1_single.onnx',
    configUrl:    HF_BASE + 'model_fold1_single_config.json',
    configInline: { ..._DINO_CONFIG, backbone_type: 'dinov3_vitb16' },
  },
  vitl: {
    label:        'DINOv3 ViT-L',
    size:         '1.2 GB',
    modelUrl:     HF_BASE + 'dinov3_vitl_model_fold1.onnx',
    configUrl:    HF_BASE + 'dinov3_vitl_model_fold1_config.json',
    configInline: { ..._DINO_CONFIG, backbone_type: 'dinov3_vitl14' },
  },
  vith: {
    label:        'DINOv3 ViT-H',
    size:         '3.4 GB',
    modelUrl:     HF_BASE + 'dinov3_vith_model_fold1_single.onnx',
    configUrl:    HF_BASE + 'dinov3_vith_model_fold1_config.json',
    configInline: { ..._DINO_CONFIG, backbone_type: 'dinov3_vith14' },
    warn:         'Very large (3.4 GB) — needs merging before use',
    unavailable:  true,
  },
};

// Per-model status: 'idle' | 'loading' | 'ready' | 'active' | 'error'
const modelStatus = {};
Object.keys(MODELS).forEach(k => modelStatus[k] = 'idle');

let activeModel = 'resnet18';
function modelUrls() { return MODELS[activeModel]; }

const CANVAS_W = 640;
const CANVAS_H = 480;

const TARGET_FPS      = 30;
const FRAME_INTERVAL  = 1000 / TARGET_FPS;

const ACTIVE_THRESHOLD = 0.3;

const AU_LABELS = {
  AU01: 'Inner Brow Raiser',    AU02: 'Outer Brow Raiser',
  AU04: 'Brow Lowerer',         AU05: 'Upper Lid Raiser',
  AU06: 'Cheek Raiser',         AU09: 'Nose Wrinkler',
  AU12: 'Lip Corner Puller',    AU15: 'Lip Corner Depressor',
  AU17: 'Chin Raiser',          AU20: 'Lip Stretcher',
  AU25: 'Lips Part',            AU26: 'Jaw Drop',
};

// ─── DOM ──────────────────────────────────────────────────────────────────────

const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const loadingOverlay  = $('loading-overlay');
const loadingText     = $('loading-text');
const headerSubtitle  = $('header-subtitle');
const modelList       = $('model-list');
const displayCanvas   = $('display-canvas');
const ctx             = displayCanvas.getContext('2d');
const fpsEl           = $('fps-counter');
const noFaceEl        = $('no-face-msg');
const detectedEl      = $('detected-badge');
const frameCountEl    = $('frame-count');
const statusEl        = $('status-text');
const webcamVideo     = $('webcam-video');
const uploadVideo     = $('upload-video');
const imageInput      = $('image-input');
const videoInput      = $('video-input');
const uploadArea      = $('upload-area');
const uploadPrompt    = $('upload-prompt');
const classifyList    = $('classification-list');
const regressionList  = $('regression-list');
const btnCpu          = $('btn-cpu');
const btnGpu          = $('btn-gpu');
const btnRecord       = $('btn-record');
const btnPause        = $('btn-pause');
const btnStop         = $('btn-stop');
const btnCsv          = $('btn-csv');
const btnXlsx         = $('btn-xlsx');
const btnClear        = $('btn-clear');
const tabBtns         = $$('.tab');

// ─── State ────────────────────────────────────────────────────────────────────

let activeTab    = 'webcam';
let provider     = 'wasm';
let running      = false;
let webcamStream = null;
let frameData    = [];
let frameIndex   = 0;
let recordStart  = 0;
let lastT        = 0;
let fps          = 0;
let recording    = false;   // true while actively capturing frames to frameData
let paused       = false;   // true while recording is paused

// ─── Bootstrap ────────────────────────────────────────────────────────────────

displayCanvas.width  = CANVAS_W;
displayCanvas.height = CANVAS_H;

async function init() {
  try {
    buildModelPanel();

    const webGpuAvailable = await checkWebGPU();
    if (!webGpuAvailable) {
      // WebGPU API absent entirely — keep button but warn
      btnGpu.title = 'WebGPU not detected — click to try anyway (Chrome/Edge 113+ required)';
      btnGpu.style.opacity = '0.5';
    }

    setStatus('Loading MediaPipe…');
    await initFaceAligner('VIDEO');

    // Auto-fallback to CPU if the WebGPU device is lost mid-session
    setDeviceLostHandler(() => {
      console.warn('[ui] GPU device lost — falling back to CPU');
      provider = 'wasm';
      setProviderButtons('wasm');
      btnGpu.disabled = true;
      btnGpu.title    = 'GPU device was lost — reload the page to retry WebGPU';
      setStatus('GPU device lost — switched to CPU');
      const m = modelUrls();
      initModel(m.modelUrl, m.configUrl, 'wasm').catch(console.error);
    });

    setStatus('Loading ResNet-18 (49 MB)…');
    const m = MODELS['resnet18'];
    await initModel(m.modelUrl, m.configUrl, provider);
    modelStatus['resnet18'] = 'active';
    buildModelPanel();
    buildRegressionPanel();

    loadingOverlay.classList.add('hidden');
    setStatus('Ready');
    await startWebcam();
  } catch (err) {
    loadingText.textContent = `Load failed: ${err.message}`;
    console.error(err);
  }
}

async function checkWebGPU() {
  if (!('gpu' in navigator)) return false;
  try {
    // Try high-performance adapter first (discrete GPU), fall back to default
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                 ?? await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

// ─── Model panel ──────────────────────────────────────────────────────────────

function buildModelPanel() {
  if (!modelList) return;
  modelList.innerHTML = '';

  for (const [key, m] of Object.entries(MODELS)) {
    const status = modelStatus[key];
    const isActive = status === 'active';
    const isLoading = status === 'loading';

    const row = document.createElement('div');
    row.className = 'model-row' + (isActive ? ' model-active' : '');
    row.id = `model-row-${key}`;

    // Radio dot
    const radio = document.createElement('div');
    radio.className = 'model-radio' + (isActive ? ' model-radio-on' : '');

    // Label + size
    const info = document.createElement('div');
    info.className = 'model-info';
    info.innerHTML =
      `<span class="model-label">${m.label}</span>` +
      `<span class="model-size">${m.size}${m.warn ? ' ⚠' : ''}</span>`;

    // Action button / status badge
    const action = document.createElement('div');
    action.className = 'model-action';

    if (isActive) {
      action.innerHTML = '<span class="model-badge-active">Active</span>';
    } else if (isLoading) {
      action.innerHTML = '<span class="model-badge-loading">Loading…</span>';
    } else if (m.unavailable) {
      action.innerHTML = `<span class="model-badge-warn" title="${m.warn}">Unavailable</span>`;
    } else {
      const btn = document.createElement('button');
      btn.className = 'model-load-btn';
      btn.textContent = status === 'ready' ? 'Switch' : 'Load';
      btn.title = m.warn ?? '';
      btn.addEventListener('click', () => loadModel(key));
      action.appendChild(btn);
    }

    row.appendChild(radio);
    row.appendChild(info);
    row.appendChild(action);
    modelList.appendChild(row);
  }
}

async function loadModel(key) {
  if (key === activeModel && modelStatus[key] === 'active') return;
  const m = MODELS[key];
  if (!m || m.unavailable) return;

  // Mark loading
  modelStatus[key] = 'loading';
  buildModelPanel();
  setStatus(`Loading ${m.label} (${m.size})…`);
  stopAll();

  try {
    await initModel(m.modelUrl, m.configUrl, provider, m.configInline);

    // Deactivate old model
    if (modelStatus[activeModel] === 'active') modelStatus[activeModel] = 'ready';
    activeModel = key;
    modelStatus[key] = 'active';
    if (headerSubtitle) headerSubtitle.textContent = `/ ${m.label} · DISFA`;

    setProviderButtons(provider);   // keep provider buttons in sync after model switch
    buildModelPanel();
    buildRegressionPanel();
    setStatus(`Model: ${m.label}`);
    if (activeTab === 'webcam') await startWebcam();
  } catch (err) {
    modelStatus[key] = 'error';
    buildModelPanel();
    setStatus(`Failed to load ${m.label}: ${err.message}`);
    console.error(err);
    // Restart previous model if needed
    if (!isReady()) {
      const prev = MODELS[activeModel];
      await initModel(prev.modelUrl, prev.configUrl, provider).catch(() => {});
    }
  }
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

tabBtns.forEach(btn =>
  btn.addEventListener('click', () => switchTab(btn.dataset.tab))
);

async function switchTab(tab) {
  if (tab === activeTab) return;
  stopAll();
  activeTab = tab;

  tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  uploadArea.classList.toggle('hidden', tab === 'webcam');
  uploadVideo.classList.add('hidden');

  if (tab === 'webcam') {
    await setRunningMode('VIDEO');
    await startWebcam();
  } else {
    await setRunningMode(tab === 'image' ? 'IMAGE' : 'VIDEO');
    drawPlaceholder('Drop or click to upload a ' + (tab === 'image' ? 'photo' : 'video'));
  }
}

// ─── Webcam ───────────────────────────────────────────────────────────────────

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();
    running     = true;
    recordStart = performance.now();
    requestAnimationFrame(webcamLoop);
    setStatus('Webcam active');
  } catch (err) {
    setStatus(`Webcam unavailable: ${err.message}`);
    drawPlaceholder('Webcam unavailable — check browser permissions');
  }
}

function stopWebcam() {
  webcamStream?.getTracks().forEach(t => t.stop());
  webcamStream = null;
}

async function webcamLoop(now) {
  if (!running || activeTab !== 'webcam') return;

  if (now - lastT < FRAME_INTERVAL) {
    requestAnimationFrame(webcamLoop);
    return;
  }

  updateFPS(now);

  if (webcamVideo.readyState >= 2) {
    ctx.drawImage(webcamVideo, 0, 0, CANVAS_W, CANVAS_H);
    try {
      await processFrame(webcamVideo, webcamVideo.videoWidth, webcamVideo.videoHeight, now);
    } catch (err) {
      console.warn('[ui] processFrame error:', err);
      // If GPU session crashes mid-run, trigger fallback and keep the loop alive
      if (provider === 'webgpu') {
        provider = 'wasm';
        setProviderButtons('wasm');
        btnGpu.disabled = true;
        btnGpu.title    = 'WebGPU session crashed — reload the page to retry';
        setStatus('WebGPU crashed — switched to CPU');
        const m = modelUrls();
        await initModel(m.modelUrl, m.configUrl, 'wasm').catch(console.error);
      }
    }
  }

  requestAnimationFrame(webcamLoop);
}

// ─── Image upload ─────────────────────────────────────────────────────────────

uploadPrompt.addEventListener('click', () => {
  if (activeTab === 'image') imageInput.click();
  if (activeTab === 'video') videoInput.click();
});

uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (activeTab === 'image' && file.type.startsWith('image/')) handleImageFile(file);
  if (activeTab === 'video' && file.type.startsWith('video/')) handleVideoFile(file);
});

imageInput.addEventListener('change', e => {
  if (e.target.files[0]) handleImageFile(e.target.files[0]);
  e.target.value = '';
});

async function handleImageFile(file) {
  const img = new Image();
  img.onload = async () => {
    uploadArea.classList.add('hidden');
    setStatus('Analysing…');
    ctx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);
    await processFrame(img, img.naturalWidth, img.naturalHeight, performance.now(), 0);
    uploadArea.classList.remove('hidden');
    setStatus('Done');
  };
  img.src = URL.createObjectURL(file);
}

// ─── Video upload ─────────────────────────────────────────────────────────────

videoInput.addEventListener('change', e => {
  if (e.target.files[0]) handleVideoFile(e.target.files[0]);
  e.target.value = '';
});

function handleVideoFile(file) {
  frameData  = [];
  frameIndex = 0;
  updateExportButtons();
  frameCountEl.textContent = 'Frames: 0';

  uploadArea.classList.add('hidden');
  uploadVideo.src = URL.createObjectURL(file);
  uploadVideo.classList.remove('hidden');

  uploadVideo.oncanplay = async () => {
    running     = true;
    recordStart = 0;
    lastT       = 0;
    setStatus('Processing video…');
    requestAnimationFrame(videoLoop);
  };

  uploadVideo.play().catch(() => {});
}

async function videoLoop(now) {
  if (!running || activeTab !== 'video') return;
  if (uploadVideo.paused || uploadVideo.ended) {
    if (uploadVideo.ended) setStatus(`Done — ${frameData.length} frames recorded`);
    return;
  }

  updateFPS(now);

  ctx.drawImage(uploadVideo, 0, 0, CANVAS_W, CANVAS_H);
  const ts = uploadVideo.currentTime * 1000;
  await processFrame(uploadVideo, uploadVideo.videoWidth, uploadVideo.videoHeight, now, ts);

  requestAnimationFrame(videoLoop);
}

// ─── Core frame processor ─────────────────────────────────────────────────────

async function processFrame(source, srcW, srcH, mpTs, recordTs) {
  if (!isReady() || srcW === 0) return;

  if (recordTs === undefined) recordTs = mpTs - recordStart;

  const result = alignFace(source, srcW, srcH, mpTs);

  if (!result) {
    noFaceEl.classList.remove('hidden');
    detectedEl.classList.add('hidden');
    clearAUBars();
    return;
  }

  noFaceEl.classList.add('hidden');
  detectedEl.classList.remove('hidden');

  const { alignedCanvas, landmarks } = result;
  const predictions = await predictNamed(alignedCanvas);

  drawOverlay(landmarks);
  updateAUPanels(predictions);

  if (recording && !paused) {
    const row = { frame: frameIndex++, timestamp_ms: Math.round(recordTs) };
    for (const [k, v] of Object.entries(predictions)) row[k] = parseFloat(v.toFixed(4));
    frameData.push(row);
    frameCountEl.textContent = `Frames: ${frameData.length}`;
    updateExportButtons();
  }
}

// ─── Overlay ──────────────────────────────────────────────────────────────────

function drawOverlay(landmarks) {
  ctx.fillStyle = 'rgba(255, 210, 40, 0.72)';
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * CANVAS_W, lm.y * CANVAS_H, 1.3, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = '#ff3355';
  for (const idx of [33, 263]) {
    const lm = landmarks[idx];
    ctx.beginPath();
    ctx.arc(lm.x * CANVAS_W, lm.y * CANVAS_H, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const padX = 0.04 * (maxX - minX);
  const padY = 0.04 * (maxY - minY);
  const bx = (minX - padX) * CANVAS_W;
  const by = (minY - padY) * CANVAS_H;
  const bw = (maxX - minX + 2 * padX) * CANVAS_W;
  const bh = (maxY - minY + 2 * padY) * CANVAS_H;

  ctx.strokeStyle = '#00ff66';
  ctx.lineWidth   = 2;
  ctx.strokeRect(bx, by, bw, bh);

  const cl = 14;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(bx, by + cl); ctx.lineTo(bx, by); ctx.lineTo(bx + cl, by); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx + bw - cl, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cl); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx, by + bh - cl); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cl, by + bh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx + bw - cl, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cl); ctx.stroke();
}

// ─── AU Panel ─────────────────────────────────────────────────────────────────

function buildRegressionPanel() {
  const names = getAUNames();
  regressionList.innerHTML = '';
  for (const name of names) {
    const row = document.createElement('div');
    row.className = 'au-row';
    row.id = `reg-${name}`;
    row.innerHTML =
      `<span class="au-name">${name} <span class="au-desc">${AU_LABELS[name] ?? ''}</span></span>` +
      `<div class="au-track"><div class="au-fill" id="fill-${name}"></div></div>` +
      `<span class="au-val" id="val-${name}">0.00</span>`;
    regressionList.appendChild(row);
  }
}

function updateAUPanels(predictions) {
  const names = getAUNames();

  for (const name of names) {
    const val  = predictions[name] ?? 0;
    const pct  = (val * 100).toFixed(1);
    const fill = $(`fill-${name}`);
    const valEl = $(`val-${name}`);
    if (fill)  fill.style.width  = `${pct}%`;
    if (valEl) valEl.textContent = val.toFixed(2);
    $(`reg-${name}`)?.classList.toggle('au-active', val >= ACTIVE_THRESHOLD);
  }

  const actives = names.filter(n => (predictions[n] ?? 0) >= ACTIVE_THRESHOLD);
  classifyList.innerHTML = '';
  if (actives.length === 0) {
    classifyList.innerHTML = '<div class="au-empty">—</div>';
    return;
  }
  for (const name of actives) {
    const val = predictions[name];
    const row = document.createElement('div');
    row.className = 'au-row au-active';
    row.innerHTML =
      `<span class="au-name">${name} <span class="au-desc">${AU_LABELS[name] ?? ''}</span></span>` +
      `<div class="au-track"><div class="au-fill" style="width:${(val * 100).toFixed(1)}%"></div></div>` +
      `<span class="au-val">${val.toFixed(2)}</span>`;
    classifyList.appendChild(row);
  }
}

function clearAUBars() {
  for (const name of getAUNames()) {
    const fill  = $(`fill-${name}`);
    const valEl = $(`val-${name}`);
    if (fill)  fill.style.width  = '0%';
    if (valEl) valEl.textContent = '0.00';
    $(`reg-${name}`)?.classList.remove('au-active');
  }
  classifyList.innerHTML = '<div class="au-empty">—</div>';
}

// ─── Provider toggle ──────────────────────────────────────────────────────────

btnCpu.addEventListener('click', () => switchProvider('wasm'));
btnGpu.addEventListener('click', () => switchProvider('webgpu'));

const PROVIDER_LABELS = { wasm: 'CPU', webgpu: 'WebGPU' };

function setProviderButtons(active) {
  btnCpu.classList.toggle('active', active === 'wasm');
  btnGpu.classList.toggle('active', active === 'webgpu');
}

async function switchProvider(p) {
  // Allow re-triggering same provider if session is dead (e.g. after GPU crash fallback)
  if (p === provider && isReady()) return;
  const label = PROVIDER_LABELS[p] ?? p;
  setStatus(`Switching to ${label}…`);
  setProviderButtons(p);
  try {
    const m = modelUrls();
    await initModel(m.modelUrl, m.configUrl, p);
    provider = p;
    setStatus(`Running on ${label}`);
  } catch (err) {
    console.warn(`[ui] ${label} failed:`, err);
    const errMsg = err?.message ?? String(err);

    let hint = errMsg.slice(0, 140);
    if (p === 'webgpu' && errMsg.includes('backend not found')) {
      hint = 'WebGPU not available on this device — requires Chrome 113+ with DX12/Vulkan GPU support. '
           + 'Try chrome://flags/#enable-webgpu-developer-features';
    }

    btnGpu.disabled = true;
    btnGpu.title    = hint;
    setProviderButtons(provider);
    setStatus(hint);
    const m = modelUrls();
    await initModel(m.modelUrl, m.configUrl, provider);
  }
}

// ─── Recording controls ───────────────────────────────────────────────────────

btnRecord.addEventListener('click', () => {
  recording = true;
  paused    = false;
  recordStart = performance.now();
  frameData   = [];
  frameIndex  = 0;
  frameCountEl.textContent = 'Frames: 0';
  btnRecord.classList.add('recording');
  btnRecord.disabled = true;
  btnPause.disabled  = false;
  btnStop.disabled   = false;
  updateExportButtons();
  setStatus('Recording…');
});

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? '▶ Resume' : '⏸ Pause';
  setStatus(paused ? 'Recording paused' : 'Recording…');
});

btnStop.addEventListener('click', () => {
  recording = false;
  paused    = false;
  btnRecord.classList.remove('recording');
  btnRecord.disabled = false;
  btnPause.disabled  = true;
  btnPause.textContent = '⏸ Pause';
  btnStop.disabled   = true;
  setStatus(`Stopped — ${frameData.length} frames captured`);
});

// ─── Export ───────────────────────────────────────────────────────────────────

btnCsv.addEventListener('click',  () => downloadCSV(frameData));
btnXlsx.addEventListener('click', () => downloadXLSX(frameData));
btnClear.addEventListener('click', () => {
  frameData  = [];
  frameIndex = 0;
  frameCountEl.textContent = 'Frames: 0';
  updateExportButtons();
  setStatus('Data cleared');
});

function updateExportButtons() {
  const hasData = frameData.length > 0;
  btnCsv.disabled   = !hasData;
  btnXlsx.disabled  = !hasData;
  btnClear.disabled = !hasData;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function stopAll() {
  running = false;
  stopWebcam();
  uploadVideo.pause();
  uploadVideo.src = '';
  lastT = 0;
  fps   = 0;
  fpsEl.textContent = 'FPS: --';
}

function updateFPS(now) {
  if (lastT) fps = Math.round(1000 / (now - lastT));
  lastT = now;
  fpsEl.textContent = `FPS: ${fps || '--'}`;
}

function setStatus(msg) {
  statusEl.textContent = msg;
  console.log(`[ui] ${msg}`);
}

function drawPlaceholder(msg = '') {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  if (msg) {
    ctx.fillStyle  = '#444';
    ctx.font       = '15px system-ui';
    ctx.textAlign  = 'center';
    ctx.fillText(msg, CANVAS_W / 2, CANVAS_H / 2);
    ctx.textAlign  = 'left';
  }
}

// ─── Start ────────────────────────────────────────────────────────────────────
init();
