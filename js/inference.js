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

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.all.min.mjs';

// Configure ONNX Runtime Web WASM paths
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

// ─── WebGPU ──────────────────────────────────────────────────────────────────

let _deviceLostHandler = null;

/** Register a callback for GPU device loss (e.g. to auto-switch to CPU). */
export function setDeviceLostHandler(fn) { _deviceLostHandler = fn; }

/** Quick check: does this browser/GPU support WebGPU at all? */
async function _checkWebGPUSupport() {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU API not available — requires Chrome/Edge 113+');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
               ?? await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No WebGPU adapter found — GPU may not support WebGPU');
  }
  console.log(`[inference] WebGPU adapter OK — maxStorageBuffers: ${adapter.limits.maxStorageBuffersPerShaderStage}`);
}

// ─── Module state ─────────────────────────────────────────────────────────────

let _session   = null;   // ort.InferenceSession
let _config    = null;   // parsed model_config.json
let _provider  = null;   // 'webgpu' | 'wasm'
let _modelUrl  = null;   // URL of currently loaded model
let _switching = false;   // true while a new session is being created
let _seqNo     = 0;       // monotonic counter — detects superseded initModel calls
let _diagCount = 0;
let _onProgress = null;   // (pct: number) => void — download progress callback

/** Register a download progress callback: fn(percent 0–100). */
export function setProgressCallback(fn) { _onProgress = fn; }

/**
 * Fetch a URL as an ArrayBuffer with progress reporting.
 * Falls back to plain fetch if Content-Length is unavailable.
 */
async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${url} (${res.status})`);

  const total = parseInt(res.headers.get('Content-Length') ?? '0', 10);
  if (!total || !res.body) {
    // No Content-Length or no ReadableStream — fall back to plain fetch
    _onProgress?.(0);
    const buf = await res.arrayBuffer();
    _onProgress?.(100);
    return buf;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    _onProgress?.(Math.round((loaded / total) * 100));
  }

  // Combine chunks into single ArrayBuffer
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf.buffer;
}

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
 * Uses a sequence counter so concurrent calls don't corrupt state.
 */
export async function initModel(
  modelUrl,
  configUrl,
  provider = 'wasm',
  configInline = null
) {
  if (_session && _provider === provider && _modelUrl === modelUrl) return _config;

  const mySeq = ++_seqNo;   // capture our sequence number
  _switching = true;

  const prevSession  = _session;
  const prevProvider = _provider;
  const prevConfig   = _config;

  try {
    // ── Load config ──
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

    // If another initModel call arrived while we were fetching, bail out
    if (mySeq !== _seqNo) {
      console.log('[inference] Superseded by newer initModel call — aborting');
      return _config;
    }

    // Check WebGPU support before attempting a GPU session
    if (provider === 'webgpu') {
      await _checkWebGPUSupport();
    }

    if (mySeq !== _seqNo) {
      console.log('[inference] Superseded by newer initModel call — aborting');
      return _config;
    }

    // ── Create session ──
    const sessionOptions = {
      executionProviders: [provider === 'webgpu' ? { name: 'webgpu' } : provider],
      graphOptimizationLevel: 'all',
      ...(provider === 'webgpu' && { preferredOutputLocation: 'cpu' }),
    };

    const cropSize = newConfig.crop ?? 224;
    const fileName = modelUrl.split('/').pop();
    console.log(`[inference] Downloading ${fileName}…`);
    _onProgress?.(0);
    const modelBuffer = await fetchWithProgress(modelUrl);

    if (mySeq !== _seqNo) {
      console.log('[inference] Superseded during download — aborting');
      return _config;
    }

    console.log(`[inference] Creating ${provider} session for ${fileName}…`);
    const newSession = await ort.InferenceSession.create(modelBuffer, sessionOptions);

    if (mySeq !== _seqNo) {
      console.log('[inference] Superseded — releasing newly created session');
      try { await newSession.release(); } catch (_) {}
      return _config;
    }

    // ── Verification run ──
    const dummy = new ort.Tensor('float32', new Float32Array(3 * cropSize * cropSize), [1, 3, cropSize, cropSize]);
    const testResults = await newSession.run({ input: dummy });
    dummy.dispose();

    const testOutput = testResults['au_intensities'] ?? testResults[Object.keys(testResults)[0]];
    if (testOutput) {
      const testData = typeof testOutput.getData === 'function'
        ? await testOutput.getData()
        : testOutput.data;
      console.log(`[inference] Verification OK — provider: ${provider}, location: ${testOutput.location ?? 'cpu'}, type: ${testOutput.type}, values: ${testData?.length}`);
      if (typeof testOutput.dispose === 'function') testOutput.dispose();
    }

    if (mySeq !== _seqNo) {
      console.log('[inference] Superseded — releasing newly created session');
      try { await newSession.release(); } catch (_) {}
      return _config;
    }

    // ── Commit: only now update all state atomically ──
    // Release the old session
    if (prevSession && prevSession !== newSession) {
      try { await prevSession.release(); } catch (_) {}
    }

    _session  = newSession;
    _provider = provider;
    _modelUrl = modelUrl;
    _config   = newConfig;

    // Update preprocessing constants only after session is confirmed
    RESIZE    = newConfig.resize ?? 256;
    CROP      = newConfig.crop   ?? 224;
    NORM_MEAN = newConfig.norm_mean ?? [0.485, 0.456, 0.406];
    NORM_STD  = newConfig.norm_std  ?? [0.229, 0.224, 0.225];

    console.log(`[inference] Model loaded — provider: ${provider}, AUs: ${_config.au_names}`);
    return _config;

  } catch (err) {
    // Restore previous state only if we haven't been superseded
    if (mySeq === _seqNo) {
      _session  = prevSession;
      _provider = prevProvider;
      _config   = prevConfig;
    }
    throw err;
  } finally {
    // Only clear the switching flag if we are still the active call
    if (mySeq === _seqNo) _switching = false;
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
  let results;
  try {
    results = await _session.run({ input: inputTensor });
  } finally {
    // Always dispose the input tensor to free GPU memory
    inputTensor.dispose();
  }

  const output = results['au_intensities'];
  if (!output) {
    const keys = Object.keys(results);
    throw new Error(`Output 'au_intensities' not found. Available: ${keys.join(', ')}`);
  }

  // Read output — getData() handles GPU→CPU transfer
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

  // Copy data before disposing the output tensor
  const result = data.slice();

  // Dispose output tensor to free GPU memory
  if (typeof output.dispose === 'function') {
    try { output.dispose(); } catch (_) {}
  }

  // Periodic diagnostic
  if (++_diagCount % 60 === 1) {
    const max = result.length > 0 ? Math.max(...result).toFixed(3) : 'EMPTY';
    const sample = Array.from(result.slice(0, 4)).map(x => x.toFixed(3)).join(', ');
    console.log(`[inference] ${_provider} | len=${result.length} max=${max} sample=[${sample}]`);
  }

  return result;
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
