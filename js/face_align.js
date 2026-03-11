/**
 * face_align.js
 *
 * JavaScript port of face_align.py — MediaPipe-based face alignment
 * matching the exact training pipeline used for the AU model.
 *
 * Alignment spec (must not change — model is sensitive to these values):
 *   output size  : 224 × 224 px
 *   eye distance : 30% of output size  (= 67.2 px)
 *   eye center   : (50%, 40%) of output size  (= 112, 89.6 px)
 *   landmarks    : MediaPipe indices 33 (left eye) and 263 (right eye)
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

// ─── Constants (mirror face_align.py) ────────────────────────────────────────

const OUTPUT_SIZE    = 224;
const LEFT_EYE_IDX  = 33;
const RIGHT_EYE_IDX = 263;
const EYE_DIST_RATIO = 0.30;  // desired eye_dist / output_size
const EYE_CENTER_X   = 0.50;  // fraction of output_size
const EYE_CENTER_Y   = 0.40;

// ─── Singleton state ──────────────────────────────────────────────────────────

let _landmarker = null;
let _currentMode = null;   // 'IMAGE' | 'VIDEO'

// ─── Init ─────────────────────────────────────────────────────────────────────

/**
 * Initialise MediaPipe FaceLandmarker. Call once at startup before anything
 * else. Safe to call multiple times (no-op after first success).
 *
 * @param {'IMAGE'|'VIDEO'} mode  Use 'IMAGE' for static images,
 *                                'VIDEO' for webcam / video frame loops.
 */
export async function initFaceAligner(mode = 'IMAGE') {
  if (_landmarker && _currentMode === mode) return;

  // Close existing instance before creating a new one (avoids memory leak)
  if (_landmarker) {
    _landmarker.close();
    _landmarker = null;
  }

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  _landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
        'face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode:                 mode,
    numFaces:                    1,
    outputFaceBlendshapes:       false,
    outputFacialTransformationMatrixes: false,
    minFaceDetectionConfidence:  0.5,
    minFacePresenceConfidence:   0.5,
    minTrackingConfidence:       0.5,
  });

  _currentMode = mode;
}

/**
 * Switch between IMAGE and VIDEO running mode (e.g. when the user switches
 * from image upload to webcam). FaceLandmarker must already be initialised.
 */
export async function setRunningMode(mode) {
  if (!_landmarker) throw new Error('Call initFaceAligner() first');
  if (_currentMode === mode) return;
  await _landmarker.setOptions({ runningMode: mode });
  _currentMode = mode;
}

// ─── Core detection ───────────────────────────────────────────────────────────

/**
 * Run MediaPipe landmark detection on one frame.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @param {number} timestampMs  Required only for VIDEO mode.
 * @returns {Array|null}  Array of 478 normalised landmarks, or null if no face.
 */
export function detectLandmarks(source, timestampMs = 0) {
  if (!_landmarker) throw new Error('Call initFaceAligner() first');

  const result = _currentMode === 'VIDEO'
    ? _landmarker.detectForVideo(source, timestampMs)
    : _landmarker.detect(source);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) return null;
  return result.faceLandmarks[0];
}

// ─── Alignment ────────────────────────────────────────────────────────────────

/**
 * Compute the 2×3 affine matrix that maps source-image pixels to the
 * 224×224 aligned output, replicating the Python cv2.getRotationMatrix2D
 * + translation adjustment in face_align.py.
 *
 * Matrix layout (matching OpenCV convention):
 *   dst_x = α·src_x + β·src_y + tx
 *   dst_y = −β·src_x + α·src_y + ty
 *
 * where  α = scale·cos(angle),  β = scale·sin(angle)
 *
 * @param {Array} landmarks   Output of detectLandmarks()
 * @param {number} W          Source width  in pixels
 * @param {number} H          Source height in pixels
 * @returns {{ a, b, c, d, tx, ty, leftEye, rightEye, eyeDist }|null}
 */
export function computeAlignMatrix(landmarks, W, H) {
  if (!landmarks) return null;

  // Eye positions in pixel coordinates
  const lx = landmarks[LEFT_EYE_IDX].x  * W;
  const ly = landmarks[LEFT_EYE_IDX].y  * H;
  const rx = landmarks[RIGHT_EYE_IDX].x * W;
  const ry = landmarks[RIGHT_EYE_IDX].y * H;

  // Eye centre (midpoint)
  const cx = (lx + rx) / 2;
  const cy = (ly + ry) / 2;

  // Angle of the eye line (Python: np.arctan2(dy, dx) then degrees)
  const dx      = rx - lx;
  const dy      = ry - ly;
  const angle   = Math.atan2(dy, dx);   // radians; converted to degrees in Python
                                         // but the trig is identical

  // Scale so eye distance equals EYE_DIST_RATIO * OUTPUT_SIZE
  const eyeDist = Math.hypot(dx, dy);
  const scale   = (OUTPUT_SIZE * EYE_DIST_RATIO) / eyeDist;

  // Destination eye-centre position
  const dstCx = OUTPUT_SIZE * EYE_CENTER_X;
  const dstCy = OUTPUT_SIZE * EYE_CENTER_Y;

  // Affine components — derived from getRotationMatrix2D + Python's M[0,2]/M[1,2] nudge:
  //   tx = dst_cx − α·cx − β·cy
  //   ty = dst_cy + β·cx − α·cy
  const α  = scale * Math.cos(angle);
  const β  = scale * Math.sin(angle);
  const tx = dstCx - α * cx - β * cy;
  const ty = dstCy + β * cx - α * cy;

  return {
    // Canvas setTransform(a, b, c, d, e, f) args:
    //   dst_x = a·src_x + c·src_y + e
    //   dst_y = b·src_x + d·src_y + f
    // → a=α, b=−β, c=β, d=α, e=tx, f=ty
    a: α, b: -β, c: β, d: α, tx, ty,
    leftEye:  { x: lx, y: ly },
    rightEye: { x: rx, y: ry },
    eyeDist,
    confidence: 1.0,   // MediaPipe doesn't expose per-detection confidence here
  };
}

/**
 * Draw an aligned 224×224 face onto a canvas.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @param {{ a,b,c,d,tx,ty }} matrix  Output of computeAlignMatrix()
 * @param {HTMLCanvasElement|OffscreenCanvas} [outCanvas]  Optional: supply your
 *   own canvas; otherwise a new OffscreenCanvas is created.
 * @returns {HTMLCanvasElement|OffscreenCanvas}
 */
export function drawAlignedFace(source, matrix, outCanvas) {
  const canvas = outCanvas ?? new OffscreenCanvas(OUTPUT_SIZE, OUTPUT_SIZE);
  const ctx    = canvas.getContext('2d');

  ctx.clearRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.tx, matrix.ty);
  ctx.drawImage(source, 0, 0);
  ctx.resetTransform();

  return canvas;
}

/**
 * Full pipeline: detect → compute matrix → draw.
 * Returns null if no face is detected.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @param {number} W   Source width  in pixels
 * @param {number} H   Source height in pixels
 * @param {number} [timestampMs]   Needed for VIDEO mode
 * @returns {{ alignedCanvas, matrix, landmarks }|null}
 */
export function alignFace(source, W, H, timestampMs = 0) {
  const landmarks = detectLandmarks(source, timestampMs);
  if (!landmarks) return null;

  const matrix        = computeAlignMatrix(landmarks, W, H);
  const alignedCanvas = drawAlignedFace(source, matrix);

  return { alignedCanvas, matrix, landmarks };
}

// ─── Helpers for visualisation ────────────────────────────────────────────────

/**
 * Compute an axis-aligned bounding box around all 478 landmarks in the
 * source frame (for drawing the green box in the GUI).
 *
 * @param {Array} landmarks   Output of detectLandmarks()
 * @param {number} W          Source width
 * @param {number} H          Source height
 * @param {number} [pad=0.05] Fractional padding added to each side
 * @returns {{ x, y, w, h }} Pixel coordinates in the source frame
 */
export function faceBoundingBox(landmarks, W, H, pad = 0.05) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const px = pad * (maxX - minX);
  const py = pad * (maxY - minY);
  return {
    x: (minX - px) * W,
    y: (minY - py) * H,
    w: (maxX - minX + 2 * px) * W,
    h: (maxY - minY + 2 * py) * H,
  };
}
