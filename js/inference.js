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

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.mjs';

// Configure ONNX Runtime Web WASM paths (must point to CDN or local copy)
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ─── Module state ─────────────────────────────────────────────────────────────

let _session   = null;   // ort.InferenceSession
let _config    = null;   // parsed model_config.json
let _provider  = null;   // 'webgpu' | 'wasm'

// ─── Preprocessing constants (overridden by config after load) ────────────────

let RESIZE     = 256;
let CROP       = 224;
let NORM_MEAN  = [0.485, 0.456, 0.406];
let NORM_STD   = [0.229, 0.224, 0.225];

// Reusable off-screen canvas for resize step
const _resizeCanvas = document.createElement('canvas');
const _resizeCtx    = _resizeCanvas.getContext('2d', { willReadFrequently: true });

// ─── Init ─────────────────────────────────────────────────────────────────────

/**
 * Load the ONNX model and config JSON.
 * Safe to call multiple times — only reloads if provider changes.
 *
 * @param {string} modelUrl   Path/URL to .onnx file
 * @param {string} configUrl  Path/URL to _config.json
 * @param {'webgpu'|'wasm'} provider  Execution provider
 * @returns {Promise<object>}  The parsed config (au_names etc.)
 */
export async function initModel(
  modelUrl,
  configUrl,
  provider = 'wasm'
) {
  // Reload if provider changed
  if (_session && _provider === provider) return _config;

  // Keep the old session alive until the new one is confirmed working.
  // If the new session fails, the old one is still intact and usable.
  const prevSession  = _session;
  const prevProvider = _provider;
  const prevConfig   = _config;

  // Load config first (small JSON, fast)
  const res = await fetch(configUrl);
  if (!res.ok) throw new Error(`Failed to fetch config: ${configUrl}`);
  _config = await res.json();

  // Apply preprocessing constants from config
  RESIZE    = _config.resize ?? 256;
  CROP      = _config.crop   ?? 224;
  NORM_MEAN = _config.norm_mean ?? [0.485, 0.456, 0.406];
  NORM_STD  = _config.norm_std  ?? [0.229, 0.224, 0.225];

  // Build session options
  const sessionOptions = {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
  };

  let newSession;
  try {
    newSession = await ort.InferenceSession.create(modelUrl, sessionOptions);
    // Verify the session actually works end-to-end before committing.
    // This catches WebGPU failures that don't surface until the first run().
    const dummy = new ort.Tensor('float32', new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);
    await newSession.run({ input: dummy });
  } catch (err) {
    // New session failed — restore previous state so the site keeps working
    _session  = prevSession;
    _provider = prevProvider;
    _config   = prevConfig;
    throw err;   // re-throw so switchProvider() can handle the UI
  }

  _session  = newSession;
  _provider = provider;

  console.log(`[inference] Model loaded — provider: ${provider}, AUs: ${_config.au_names}`);
  return _config;
}

/**
 * Switch execution provider (GPU ↔ CPU) without reloading the model file.
 * Triggers a session rebuild with the same URLs — store them externally.
 */
export async function setProvider(modelUrl, configUrl, provider) {
  if (_provider === provider) return _config;
  return initModel(modelUrl, configUrl, provider);
}

/** True once initModel() has succeeded. */
export function isReady() {
  return _session !== null && _config !== null;
}

/** Returns the AU name list from the loaded config, e.g. ['AU01', 'AU02', ...] */
export function getAUNames() {
  return _config?.au_names ?? [];
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

/**
 * Convert an aligned 224×224 canvas to a normalised float32 NCHW tensor.
 *
 * Replicates inference.py get_transform():
 *   transforms.Resize(256) → transforms.CenterCrop(224) → ToTensor → Normalize
 *
 * @param {HTMLCanvasElement|OffscreenCanvas} alignedCanvas  Output of drawAlignedFace()
 * @returns {ort.Tensor}  Shape [1, 3, 224, 224], dtype float32
 */
export function preprocessCanvas(alignedCanvas) {
  // ── Step 1: Resize to RESIZE × RESIZE ──────────────────────────────────────
  _resizeCanvas.width  = RESIZE;
  _resizeCanvas.height = RESIZE;
  _resizeCtx.drawImage(alignedCanvas, 0, 0, RESIZE, RESIZE);

  // ── Step 2: Centre crop to CROP × CROP ─────────────────────────────────────
  const offset = Math.floor((RESIZE - CROP) / 2);   // = 16 for 256→224
  const imageData = _resizeCtx.getImageData(offset, offset, CROP, CROP);
  const pixels    = imageData.data;   // Uint8ClampedArray, RGBA interleaved, HWC

  // ── Step 3 & 4: ToTensor (÷255) + Normalize (subtract mean, divide std) ───
  // Output layout: NCHW float32  →  [1, 3, 224, 224]
  const nPixels = CROP * CROP;
  const tensor  = new Float32Array(3 * nPixels);

  for (let i = 0; i < nPixels; i++) {
    const r = pixels[i * 4]     / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    // Channel-first layout: R plane → G plane → B plane
    tensor[i]               = (r - NORM_MEAN[0]) / NORM_STD[0];
    tensor[nPixels + i]     = (g - NORM_MEAN[1]) / NORM_STD[1];
    tensor[nPixels * 2 + i] = (b - NORM_MEAN[2]) / NORM_STD[2];
  }

  return new ort.Tensor('float32', tensor, [1, 3, CROP, CROP]);
}

// ─── Inference ────────────────────────────────────────────────────────────────

/**
 * Run AU intensity prediction on one aligned face canvas.
 *
 * @param {HTMLCanvasElement|OffscreenCanvas} alignedCanvas
 * @returns {Promise<Float32Array>}  AU intensities in [0, 1], length = num_aus
 */
export async function predict(alignedCanvas) {
  if (!_session) throw new Error('Call initModel() before predict()');

  const inputTensor = preprocessCanvas(alignedCanvas);
  const feeds       = { input: inputTensor };
  const results     = await _session.run(feeds);

  // Output tensor is named 'au_intensities' (set in export_onnx.py)
  const output = results['au_intensities'];
  return output.data.slice();   // Float32Array copy
}

/**
 * Convenience wrapper — runs predict() and returns a plain object.
 *
 * @param {HTMLCanvasElement|OffscreenCanvas} alignedCanvas
 * @returns {Promise<object>}  e.g. { AU01: 0.12, AU02: 0.04, ... }
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
