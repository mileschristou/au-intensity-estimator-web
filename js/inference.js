/**
 * inference.js
 *
 * ONNX Runtime Web wrapper for the AU Intensity Estimator.
 *
 * Replicates the full preprocessing pipeline from inference.py get_transform():
 *   1. Resize aligned 224×224 → 256×256
 *   2. Centre-crop 256×256 → 224×224  (removes 16 px from each side)
 *   3. Convert to float32 CHW tensor
 *   4. Normalise with ImageNet mean/std
 *
 * GPU/CPU is toggled by passing 'webgpu' or 'wasm' to initModel().
 * WebGPU is faster but requires Chrome/Edge 113+; WASM works everywhere.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.min.mjs';

// Configure ONNX Runtime Web WASM paths (must point to CDN or local copy)
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ─── WebGPU device setup ──────────────────────────────────────────────────────

let _deviceLostHandler = null;

/** Register a callback for GPU device loss (e.g. to auto-switch to CPU). */
export function setDeviceLostHandler(fn) { _deviceLostHandler = fn; }

async function _prepareWebGPUDevice() {
  if (!('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                 ?? await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    const adapterMax = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
    console.log(`[inference] WebGPU adapter — maxStorageBuffers: ${adapterMax}`);

    if (adapterMax < 12) {
      console.warn(`[inference] GPU only supports ${adapterMax} storage buffers (need ≥12 for ORT). WebGPU may fail.`);
    }

    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: adapterMax,
      },
    });

    device.lost.then(info => {
      console.warn('[inference] WebGPU device lost:', info.message);
      _session  = null;
      _provider = null;
      ort.env.webgpu.device = undefined;
      _deviceLostHandler?.();
    });

    ort.env.webgpu.device = device;
    console.log(`[inference] WebGPU device ready`);
    return true;
  } catch (e) {
    console.warn('[inference] Could not create WebGPU device:', e);
    return false;
  }
}

// ─── Module state ─────────────────────────────────────────────────────────────

let _session   = null;   // ort.InferenceSession
let _config    = null;   // parsed model_config.json
let _provider  = null;   // 'webgpu' | 'wasm'
let _modelUrl  = null;   // URL of currently loaded model
let _switching = false;   // true while a new session is being created
let _diagCount = 0;

// ─── Preprocessing constants (overridden by config after load) ────────────────

let RESIZE     = 256;
let CROP       = 224;
let NORM_MEAN  = [0.485, 0.456, 0.406];
let NORM_STD   = [0.229, 0.224, 0.225];

const _resizeCanvas = document.createElement('canvas');
const _resizeCtx    = _resizeCanvas.getContext('2d', { willReadFrequently: true });

// ─── Init ─────────────────────────────────────────────────────────────────────

/**
 * Load the ONNX model and config JSON.
 * Safe to call multiple times — only reloads if provider or model changes.
 */
export async function initModel(
  modelUrl,
  configUrl,
  provider = 'wasm',
  configInline = null
) {
  if (_session && _provider === provider && _modelUrl === modelUrl) return _config;

  _switching = true;

  // Keep the old session alive until the new one is confirmed working.
  const prevSession  = _session;
  const prevProvider = _provider;
  const prevConfig   = _config;

  try {
    // Load config
    let newConfig;
    try {
      const res = await fetch(configUrl);
      if (res.ok) {
        newConfig = await res.json();
      } else if (configInline) {
        console.warn(`[inference] Config fetch failed (${res.status}); using inline fallback`);
        newConfig = configInline;
      } else {
        throw new Error(`Failed to fetch config: ${configUrl} (${res.status})`);
      }
    } catch (fetchErr) {
      if (configInline) {
        console.warn(`[inference] Config fetch error; using inline fallback:`, fetchErr.message);
        newConfig = configInline;
      } else {
        throw fetchErr;
      }
    }

    // Apply preprocessing constants
    RESIZE    = newConfig.resize ?? 256;
    CROP      = newConfig.crop   ?? 224;
    NORM_MEAN = newConfig.norm_mean ?? [0.485, 0.456, 0.406];
    NORM_STD  = newConfig.norm_std  ?? [0.229, 0.224, 0.225];

    // Clear dead GPU device when switching away from WebGPU
    if (provider !== 'webgpu' && ort.env.webgpu?.device) {
      ort.env.webgpu.device = undefined;
    }

    // Pre-create WebGPU device with maximal limits
    if (provider === 'webgpu' && !ort.env.webgpu?.device) {
      const ok = await _prepareWebGPUDevice();
      if (!ok) throw new Error('WebGPU device creation failed — no compatible GPU adapter found');
    }

    // Session options — preferredOutputLocation is session-level (per ORT docs)
    const sessionOptions = {
      executionProviders: [provider === 'webgpu' ? { name: 'webgpu' } : provider],
      graphOptimizationLevel: 'all',
      ...(provider === 'webgpu' && { preferredOutputLocation: 'cpu' }),
    };

    console.log(`[inference] Creating ${provider} session for ${modelUrl.split('/').pop()}…`);
    const newSession = await ort.InferenceSession.create(modelUrl, sessionOptions);

    // Verification run — confirm the session works end-to-end
    const dummy = new ort.Tensor('float32', new Float32Array(3 * CROP * CROP), [1, 3, CROP, CROP]);
    const testResults = await newSession.run({ input: dummy });
    const testOutput = testResults['au_intensities'] ?? testResults[Object.keys(testResults)[0]];

    if (testOutput) {
      const testData = typeof testOutput.getData === 'function'
        ? await testOutput.getData()
        : testOutput.data;
      const loc = testOutput.location ?? 'cpu';
      console.log(`[inference] Verification OK — provider: ${provider}, location: ${loc}, type: ${testOutput.type}, values: ${testData?.length}`);

      // Dispose the test output tensor to free GPU memory
      if (typeof testOutput.dispose === 'function') testOutput.dispose();
    }

    // Release the old session now that the new one is confirmed
    if (prevSession && prevSession !== newSession) {
      try { await prevSession.release(); } catch (_) { /* ok */ }
    }

    _session  = newSession;
    _provider = provider;
    _modelUrl = modelUrl;
    _config   = newConfig;

    console.log(`[inference] Model loaded — provider: ${provider}, AUs: ${_config.au_names}`);
    return _config;

  } catch (err) {
    // Restore previous state so inference can continue
    _session  = prevSession;
    _provider = prevProvider;
    _config   = prevConfig;
    throw err;
  } finally {
    _switching = false;
  }
}

/** True once initModel() has succeeded and no switch is in progress. */
export function isReady() {
  return _session !== null && _config !== null && !_switching;
}

/** Returns the AU name list from the loaded config. */
export function getAUNames() {
  return _config?.au_names ?? [];
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

export function preprocessCanvas(alignedCanvas) {
  _resizeCanvas.width  = RESIZE;
  _resizeCanvas.height = RESIZE;
  _resizeCtx.drawImage(alignedCanvas, 0, 0, RESIZE, RESIZE);

  const offset = Math.floor((RESIZE - CROP) / 2);
  const imageData = _resizeCtx.getImageData(offset, offset, CROP, CROP);
  const pixels    = imageData.data;

  const nPixels = CROP * CROP;
  const tensor  = new Float32Array(3 * nPixels);

  for (let i = 0; i < nPixels; i++) {
    const r = pixels[i * 4]     / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    tensor[i]               = (r - NORM_MEAN[0]) / NORM_STD[0];
    tensor[nPixels + i]     = (g - NORM_MEAN[1]) / NORM_STD[1];
    tensor[nPixels * 2 + i] = (b - NORM_MEAN[2]) / NORM_STD[2];
  }

  return new ort.Tensor('float32', tensor, [1, 3, CROP, CROP]);
}

// ─── Inference ────────────────────────────────────────────────────────────────

/**
 * Run AU intensity prediction on one aligned face canvas.
 * @returns {Promise<Float32Array>}  AU intensities in [0, 1]
 */
export async function predict(alignedCanvas) {
  if (!_session) throw new Error('Call initModel() before predict()');

  const inputTensor = preprocessCanvas(alignedCanvas);
  const results     = await _session.run({ input: inputTensor });

  const output = results['au_intensities'];
  if (!output) {
    const keys = Object.keys(results);
    throw new Error(`Output 'au_intensities' not found. Available: ${keys.join(', ')}`);
  }

  // Read output — getData() handles GPU→CPU transfer; .data is CPU-only
  let data;
  try {
    if (typeof output.getData === 'function') {
      const raw = await output.getData();
      data = raw instanceof Float32Array ? raw : new Float32Array(raw);
    } else {
      data = output.data instanceof Float32Array ? output.data : new Float32Array(output.data);
    }
  } catch (readErr) {
    console.error('[inference] Failed to read output:', readErr,
      `location=${output.location}, type=${output.type}, dims=${output.dims}`);
    throw readErr;
  }

  // CRITICAL: dispose output tensors to free GPU memory — without this,
  // GPU buffers accumulate every frame and eventually crash the device.
  if (typeof output.dispose === 'function') {
    try { output.dispose(); } catch (_) { /* already disposed */ }
  }

  // Periodic diagnostic
  if (++_diagCount % 60 === 1) {
    const max = data.length > 0 ? Math.max(...data).toFixed(3) : 'EMPTY';
    const sample = Array.from(data.slice(0, 4)).map(x => x.toFixed(3)).join(', ');
    console.log(`[inference] ${_provider} | len=${data.length} max=${max} sample=[${sample}]`);
  }

  return data.slice();
}

/**
 * Convenience wrapper — runs predict() and returns { AU01: 0.12, … }
 */
export async function predictNamed(alignedCanvas) {
  const values  = await predict(alignedCanvas);
  const auNames = getAUNames();
  const result  = {};
  for (let i = 0; i < auNames.length; i++) {
    result[auNames[i]] = values[i];
  }
  return result;
}
