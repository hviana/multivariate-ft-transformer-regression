/**
 * @module FusionTemporalTransformerRegression
 *
 * Fusion Temporal Transformer (FTT) for multivariate regression with incremental online learning.
 *
 * Key features:
 * - Multi-scale temporal convolution + gated cross-scale fusion
 * - Transformer blocks (LayerNorm → MHA → Residual → LayerNorm → FFN → Residual)
 * - Attention-weighted temporal pooling
 * - Online Adam optimizer with warmup + cosine decay
 * - Welford z-score normalization (inputs + outputs)
 * - Outlier downweighting + ADWIN-lite drift detection
 *
 * Numerical stability:
 * - Stable softmax (max-subtraction), LayerNorm eps, Welford eps
 * - Causal masking (no future leakage)
 * - L2 regularization added to gradients
 *
 * Notes on API shapes:
 * - fitOnline({ xCoordinates, yCoordinates }) treats xCoordinates as a temporal sequence: shape [seqLen][inputDim]
 * - yCoordinates may be shape [seqLen][outputDim] or [1][outputDim]; the target used is the last row.
 *
 * @example
 * ```ts
 * import { FusionTemporalTransformerRegression } from "./mod.ts";
 *
 * const model = new FusionTemporalTransformerRegression();
 * const fit = model.fitOnline({
 *   xCoordinates: [[1,2],[2,3],[3,4]], // seqLen x inputDim
 *   yCoordinates: [[0.1],[0.2],[0.25]] // seqLen x outputDim (uses last as target)
 * });
 *
 * const pred = model.predict(3);
 * console.log(fit.loss, pred.predictions[0].predicted);
 * ```
 */

// ============================================================================
// PUBLIC TYPES
// ============================================================================

export interface FitResult {
  loss: number;
  gradientNorm: number;
  effectiveLearningRate: number;
  isOutlier: boolean;
  converged: boolean;
  sampleIndex: number;
  driftDetected: boolean;
}

export interface SinglePrediction {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
}

export interface PredictionResult {
  predictions: SinglePrediction[];
  accuracy: number;
  sampleCount: number;
  isModelReady: boolean;
}

export interface NormalizationStats {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
  count: number;
}

export interface ModelSummary {
  isInitialized: boolean;
  inputDimension: number;
  outputDimension: number;
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  temporalScales: number[];
  totalParameters: number;
  sampleCount: number;
  accuracy: number;
  converged: boolean;
  effectiveLearningRate: number;
  driftCount: number;
}

/**
 * WeightInfo is intentionally JSON-friendly.
 * Internally everything is Float64Array; this converts to nested number arrays.
 */
export interface WeightInfo {
  temporalConvWeights: number[][][]; // [scaleIndex][kernelIndex][inputDim*embeddingDim]
  scaleEmbeddings: number[][]; // [scaleIndex][embeddingDim]
  positionalEncoding: number[][]; // [maxSequenceLength][embeddingDim]
  fusionWeights: { gateW: number[][]; gateB: number[] }; // gateW: [numScales*embeddingDim][numScales]
  attentionWeights: Array<{
    wq: number[][];
    wk: number[][];
    wv: number[][];
    wo: number[][];
    bq: number[];
    bk: number[];
    bv: number[];
    bo: number[];
  }>;
  ffnWeights: Array<{
    w1: number[][];
    b1: number[];
    w2: number[][];
    b2: number[];
  }>;
  layerNormParams: Array<{
    ln1Gamma: number[];
    ln1Beta: number[];
    ln2Gamma: number[];
    ln2Beta: number[];
  }>;
  outputWeights: {
    wOut: number[][];
    bOut: number[];
    wPool: number[];
    bPool: number;
  };
  firstMoment: number[][][]; // [paramGroup][1][paramLen]
  secondMoment: number[][][]; // [paramGroup][1][paramLen]
  updateCount: number;
}

export interface FusionTemporalTransformerRegressionConfig {
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  ffnMultiplier: number;
  attentionDropout: number;
  learningRate: number;
  warmupSteps: number;
  totalSteps: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  regularizationStrength: number;
  convergenceThreshold: number;
  outlierThreshold: number;
  adwinDelta: number;
  temporalScales: number[];
  temporalKernelSize: number;
  maxSequenceLength: number;
  fusionDropout: number;
}

// ============================================================================
// INTERNAL NUMERIC UTILITIES (NO ALLOC HOT-PATH)
// ============================================================================

const PI = Math.PI;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : (x > hi ? hi : x);
}

function sigmoid(x: number): number {
  // stable sigmoid
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  } else {
    const z = Math.exp(x);
    return z / (1 + z);
  }
}

function gelu(x: number): number {
  // tanh approximation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
  const x3 = x * x * x;
  const t = 0.7978845608028654 * (x + 0.044715 * x3); // sqrt(2/pi)=0.79788...
  const th = Math.tanh(t);
  return 0.5 * x * (1 + th);
}

function geluGrad(x: number): number {
  // derivative of tanh-approx GELU
  const x2 = x * x;
  const x3 = x2 * x;
  const t = 0.7978845608028654 * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  const sech2 = 1 - th * th;
  const dt = 0.7978845608028654 * (1 + 0.134145 * x2); // 0.044715*3=0.134145
  // d/dx 0.5x(1+th) = 0.5(1+th) + 0.5x*sech2*dt
  return 0.5 * (1 + th) + 0.5 * x * sech2 * dt;
}

function stableSoftmaxInPlace(vec: Float64Array, n: number): void {
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = vec[i];
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(vec[i] - max);
    vec[i] = e;
    sum += e;
  }
  const inv = sum > 0 ? (1 / sum) : 1;
  for (let i = 0; i < n; i++) vec[i] *= inv;
}

function computeL2Norm(arr: Float64Array): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    s += v * v;
  }
  return Math.sqrt(s);
}

function xavierLimit(fanIn: number, fanOut: number): number {
  // uniform[-a,a], a = sqrt(6/(fanIn+fanOut))
  return Math.sqrt(6 / (fanIn + fanOut));
}

function fillUniform(w: Float64Array, a: number): void {
  for (let i = 0; i < w.length; i++) {
    w[i] = (Math.random() * 2 - 1) * a;
  }
}

function zeros(w: Float64Array): void {
  w.fill(0);
}

function copyFromNumber2DToBuffer(
  src: number[][],
  rows: number,
  cols: number,
  dst: Float64Array,
): void {
  let p = 0;
  for (let r = 0; r < rows; r++) {
    const row = src[r];
    for (let c = 0; c < cols; c++) {
      dst[p++] = row[c] ?? 0;
    }
  }
}

function bufferToNumber2D(
  buf: Float64Array,
  rows: number,
  cols: number,
): number[][] {
  const out: number[][] = new Array(rows);
  let p = 0;
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) row[c] = buf[p++];
    out[r] = row;
  }
  return out;
}

function bufferToNumber1D(buf: Float64Array): number[] {
  const out = new Array<number>(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i];
  return out;
}

function matToNumber2D(
  w: Float64Array,
  rows: number,
  cols: number,
): number[][] {
  const out: number[][] = new Array(rows);
  let p = 0;
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) row[c] = w[p++];
    out[r] = row;
  }
  return out;
}

// ============================================================================
// INTERNAL: WELFORD NORMALIZER (VECTOR-WISE)
// ============================================================================

class WelfordVector {
  private _count = 0;
  private _mean: Float64Array;
  private _m2: Float64Array;

  constructor(dim: number) {
    this._mean = new Float64Array(dim);
    this._m2 = new Float64Array(dim);
  }

  reset(): void {
    this._count = 0;
    this._mean.fill(0);
    this._m2.fill(0);
  }

  get count(): number {
    return this._count;
  }

  get mean(): Float64Array {
    return this._mean;
  }

  /**
   * Returns std with Bessel correction: sqrt(M2/(n-1)), guarded.
   */
  getStd(eps: number): Float64Array {
    const dim = this._mean.length;
    const out = new Float64Array(dim);
    const n = this._count;
    if (n <= 1) {
      for (let i = 0; i < dim; i++) out[i] = 1;
      return out;
    }
    const denom = 1 / (n - 1);
    for (let i = 0; i < dim; i++) {
      const v = this._m2[i] * denom;
      out[i] = Math.sqrt(v + eps);
      if (!(out[i] > 0)) out[i] = 1;
    }
    return out;
  }

  updateFromVector(x: Float64Array): void {
    // per-dimension Welford
    const dim = this._mean.length;
    this._count++;
    const n = this._count;
    for (let i = 0; i < dim; i++) {
      const xi = x[i];
      const delta = xi - this._mean[i];
      const meanNew = this._mean[i] + delta / n;
      const delta2 = xi - meanNew;
      this._mean[i] = meanNew;
      this._m2[i] += delta * delta2;
    }
  }

  updateFromMatrix(x: Float64Array, rows: number, cols: number): void {
    // rows x cols, update each row
    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      // avoid slice allocations
      this._count++;
      const n = this._count;
      for (let c = 0; c < cols; c++) {
        const xi = x[off + c];
        const delta = xi - this._mean[c];
        const meanNew = this._mean[c] + delta / n;
        const delta2 = xi - meanNew;
        this._mean[c] = meanNew;
        this._m2[c] += delta * delta2;
      }
    }
  }

  toJSON(): { count: number; mean: number[]; m2: number[] } {
    return {
      count: this._count,
      mean: bufferToNumber1D(this._mean),
      m2: bufferToNumber1D(this._m2),
    };
  }

  fromJSON(obj: { count: number; mean: number[]; m2: number[] }): void {
    this._count = obj.count | 0;
    for (let i = 0; i < this._mean.length; i++) {
      this._mean[i] = obj.mean[i] ?? 0;
    }
    for (let i = 0; i < this._m2.length; i++) this._m2[i] = obj.m2[i] ?? 0;
  }
}

// ============================================================================
// INTERNAL: ADWIN-LITE (FAST, LOW-ALLOC APPROX)
// ============================================================================

class ADWINLite {
  private _delta: number;
  private _max: number;
  private _buf: Float64Array;
  private _start = 0;
  private _size = 0;

  constructor(delta: number, maxWindow: number) {
    this._delta = delta;
    this._max = maxWindow;
    this._buf = new Float64Array(maxWindow);
  }

  reset(): void {
    this._start = 0;
    this._size = 0;
    this._buf.fill(0);
  }

  /**
   * Insert error and test drift by splitting window in half.
   * Drift if |mean0-mean1| >= epsCut, epsCut from Hoeffding-like bound.
   */
  update(error: number): boolean {
    // push into ring
    if (this._size < this._max) {
      this._buf[(this._start + this._size) % this._max] = error;
      this._size++;
    } else {
      this._buf[this._start] = error;
      this._start = (this._start + 1) % this._max;
    }

    if (this._size < 32) return false;

    const n = this._size;
    const n0 = n >> 1;
    const n1 = n - n0;

    let sum0 = 0;
    let sum1 = 0;

    // sum first half, second half
    for (let i = 0; i < n0; i++) {
      sum0 += this._buf[(this._start + i) % this._max];
    }
    for (let i = 0; i < n1; i++) {
      sum1 += this._buf[(this._start + n0 + i) % this._max];
    }

    const m0 = sum0 / n0;
    const m1 = sum1 / n1;
    const diff = Math.abs(m0 - m1);

    // epsCut ~ sqrt( (1/2) * ln(4/delta) * (1/n0 + 1/n1) )
    const ln = Math.log(4 / this._delta);
    const epsCut = Math.sqrt(0.5 * ln * (1 / n0 + 1 / n1));

    if (diff >= epsCut) {
      // shrink: drop older half (keep newest)
      // move start forward by n0, size = n1
      this._start = (this._start + n0) % this._max;
      this._size = n1;
      return true;
    }
    return false;
  }

  toJSON(): any {
    // serialize in logical order (no need to keep zeros outside window)
    const arr = new Array<number>(this._size);
    for (let i = 0; i < this._size; i++) {
      arr[i] = this._buf[(this._start + i) % this._max];
    }
    return { delta: this._delta, max: this._max, window: arr };
  }

  fromJSON(obj: any): void {
    this._delta = obj.delta ?? this._delta;
    this._max = obj.max ?? this._max;
    this._buf = new Float64Array(this._max);
    this._start = 0;
    this._size = 0;
    const w: number[] = obj.window ?? [];
    const n = Math.min(w.length, this._max);
    for (let i = 0; i < n; i++) this._buf[i] = w[w.length - n + i]; // keep newest
    this._size = n;
  }
}

// ============================================================================
// INTERNAL: PARAM + ADAM OPTIMIZER
// ============================================================================

interface Param {
  name: string;
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
}

class AdamOptimizer {
  private _beta1: number;
  private _beta2: number;
  private _eps: number;
  private _t = 0;

  constructor(beta1: number, beta2: number, eps: number) {
    this._beta1 = beta1;
    this._beta2 = beta2;
    this._eps = eps;
  }

  get t(): number {
    return this._t;
  }

  reset(): void {
    this._t = 0;
  }

  step(params: Param[], lr: number): void {
    this._t++;
    const t = this._t;

    const b1 = this._beta1;
    const b2 = this._beta2;
    const oneMinusB1 = 1 - b1;
    const oneMinusB2 = 1 - b2;

    const b1t = Math.pow(b1, t);
    const b2t = Math.pow(b2, t);
    const invBias1 = 1 / (1 - b1t);
    const invBias2 = 1 / (1 - b2t);

    for (let p = 0; p < params.length; p++) {
      const par = params[p];
      const w = par.w;
      const g = par.g;
      const m = par.m;
      const v = par.v;

      for (let i = 0; i < w.length; i++) {
        const gi = g[i];

        // m = b1*m + (1-b1)*g
        const mi = m[i] = b1 * m[i] + oneMinusB1 * gi;
        // v = b2*v + (1-b2)*g^2
        const vi = v[i] = b2 * v[i] + oneMinusB2 * (gi * gi);

        const mHat = mi * invBias1;
        const vHat = vi * invBias2;

        w[i] -= lr * (mHat / (Math.sqrt(vHat) + this._eps));
      }
    }
  }
}

// ============================================================================
// INTERNAL: POSITIONAL ENCODING
// ============================================================================

class PositionalEncoding {
  private _maxLen: number;
  private _d: number;
  private _pe: Float64Array; // [maxLen*d]

  constructor(maxLen: number, d: number) {
    this._maxLen = maxLen;
    this._d = d;
    this._pe = new Float64Array(maxLen * d);
    this._build();
  }

  private _build(): void {
    const d = this._d;
    for (let pos = 0; pos < this._maxLen; pos++) {
      const base = pos * d;
      for (let i = 0; i < d; i += 2) {
        const divTerm = Math.pow(10000, i / d);
        const a = pos / divTerm;
        this._pe[base + i] = Math.sin(a);
        if (i + 1 < d) this._pe[base + i + 1] = Math.cos(a);
      }
    }
  }

  /**
   * Adds PE(posOffset + pos) into dstSeq (seqLen x d), in-place.
   */
  addInPlace(dstSeq: Float64Array, seqLen: number, posOffset: number): void {
    const d = this._d;
    const maxLen = this._maxLen;
    for (let t = 0; t < seqLen; t++) {
      const pePos = (t + posOffset) % maxLen;
      const peOff = pePos * d;
      const outOff = t * d;
      for (let j = 0; j < d; j++) {
        dstSeq[outOff + j] += this._pe[peOff + j];
      }
    }
  }

  toNumber2D(): number[][] {
    return bufferToNumber2D(this._pe, this._maxLen, this._d);
  }
}

// ============================================================================
// INTERNAL: LAYER NORM
// ============================================================================

class LayerNorm {
  readonly gamma: Param;
  readonly beta: Param;

  // caches (maxSeqLen x d)
  private _xhat: Float64Array;
  private _mean: Float64Array; // [maxSeqLen]
  private _invStd: Float64Array; // [maxSeqLen]
  private _d: number;
  private _maxSeq: number;
  private _eps: number;

  constructor(
    name: string,
    maxSeqLen: number,
    d: number,
    eps: number,
    registry: Param[],
  ) {
    this._d = d;
    this._maxSeq = maxSeqLen;
    this._eps = eps;

    const g = new Float64Array(d);
    const b = new Float64Array(d);
    for (let i = 0; i < d; i++) g[i] = 1;
    b.fill(0);

    this.gamma = mkParam(`${name}.gamma`, g, registry);
    this.beta = mkParam(`${name}.beta`, b, registry);

    this._xhat = new Float64Array(maxSeqLen * d);
    this._mean = new Float64Array(maxSeqLen);
    this._invStd = new Float64Array(maxSeqLen);
  }

  forward(x: Float64Array, seqLen: number, y: Float64Array): void {
    // y = LN(x) = gamma * (x-mean)/std + beta
    const d = this._d;
    const eps = this._eps;
    const gamma = this.gamma.w;
    const beta = this.beta.w;

    for (let t = 0; t < seqLen; t++) {
      const off = t * d;

      // mean
      let mu = 0;
      for (let j = 0; j < d; j++) mu += x[off + j];
      mu /= d;

      // variance
      let v = 0;
      for (let j = 0; j < d; j++) {
        const z = x[off + j] - mu;
        v += z * z;
      }
      v /= d;

      const invStd = 1 / Math.sqrt(v + eps);
      this._mean[t] = mu;
      this._invStd[t] = invStd;

      for (let j = 0; j < d; j++) {
        const xh = (x[off + j] - mu) * invStd;
        this._xhat[off + j] = xh;
        y[off + j] = xh * gamma[j] + beta[j];
      }
    }
  }

  backward(
    x: Float64Array,
    seqLen: number,
    dy: Float64Array,
    dx: Float64Array,
  ): void {
    // Fully expanded LN backward.
    // Let xhat = (x-mean)*invStd, y = gamma*xhat + beta.
    // dgamma = sum(dy*xhat), dbeta = sum(dy)
    // dx computed via standard LN formula.
    const d = this._d;
    const gamma = this.gamma.w;
    const dGamma = this.gamma.g;
    const dBeta = this.beta.g;

    // reset grads (gamma/beta only here)
    dGamma.fill(0);
    dBeta.fill(0);

    for (let t = 0; t < seqLen; t++) {
      const off = t * d;
      const invStd = this._invStd[t];

      // dbeta, dgamma
      for (let j = 0; j < d; j++) {
        const dyj = dy[off + j];
        dBeta[j] += dyj;
        dGamma[j] += dyj * this._xhat[off + j];
      }

      // dx
      // dx = (1/d) * invStd * gamma * (d*dy - sum(dy) - xhat*sum(dy*xhat))
      let sumDy = 0;
      let sumDyXhat = 0;
      for (let j = 0; j < d; j++) {
        const dyj = dy[off + j] * gamma[j];
        sumDy += dyj;
        sumDyXhat += dyj * this._xhat[off + j];
      }

      const invD = 1 / d;
      for (let j = 0; j < d; j++) {
        const dyj = dy[off + j] * gamma[j];
        const xhat = this._xhat[off + j];
        dx[off + j] = invD * invStd * (d * dyj - sumDy - xhat * sumDyXhat);
      }
    }
  }

  toJSON(): any {
    return {
      gamma: bufferToNumber1D(this.gamma.w),
      beta: bufferToNumber1D(this.beta.w),
      m: bufferToNumber1D(this.gamma.m),
      v: bufferToNumber1D(this.gamma.v),
      mb: bufferToNumber1D(this.beta.m),
      vb: bufferToNumber1D(this.beta.v),
    };
  }

  fromJSON(obj: any): void {
    const gw = obj.gamma ?? [];
    const bw = obj.beta ?? [];
    for (let i = 0; i < this._d; i++) this.gamma.w[i] = gw[i] ?? 1;
    for (let i = 0; i < this._d; i++) this.beta.w[i] = bw[i] ?? 0;

    const gm = obj.m ?? [];
    const gv = obj.v ?? [];
    const bm = obj.mb ?? [];
    const bv = obj.vb ?? [];
    for (let i = 0; i < this._d; i++) this.gamma.m[i] = gm[i] ?? 0;
    for (let i = 0; i < this._d; i++) this.gamma.v[i] = gv[i] ?? 0;
    for (let i = 0; i < this._d; i++) this.beta.m[i] = bm[i] ?? 0;
    for (let i = 0; i < this._d; i++) this.beta.v[i] = bv[i] ?? 0;
  }
}

// ============================================================================
// INTERNAL: LINEAR HELPERS (ROW-MAJOR)
// ============================================================================

function linearForwardSeq(
  x: Float64Array, // [seqLen*inDim]
  seqLen: number,
  inDim: number,
  w: Float64Array, // [inDim*outDim]
  b: Float64Array, // [outDim]
  outDim: number,
  y: Float64Array, // [seqLen*outDim]
): void {
  for (let t = 0; t < seqLen; t++) {
    const xOff = t * inDim;
    const yOff = t * outDim;
    for (let j = 0; j < outDim; j++) {
      let s = b[j];
      // row-major w: w[i*outDim + j]
      let wOff = j;
      for (let i = 0; i < inDim; i++) {
        s += x[xOff + i] * w[wOff];
        wOff += outDim;
      }
      y[yOff + j] = s;
    }
  }
}

function linearBackwardSeq(
  x: Float64Array, // [seqLen*inDim]
  seqLen: number,
  inDim: number,
  w: Float64Array, // [inDim*outDim]
  outDim: number,
  dy: Float64Array, // [seqLen*outDim]
  dx: Float64Array, // [seqLen*inDim]
  dW: Float64Array, // [inDim*outDim]
  dB: Float64Array, // [outDim]
): void {
  dW.fill(0);
  dB.fill(0);
  // dx must be set by caller (either fill 0 or accumulate); we overwrite here for safety.
  dx.fill(0);

  for (let t = 0; t < seqLen; t++) {
    const xOff = t * inDim;
    const yOff = t * outDim;

    // dB
    for (let j = 0; j < outDim; j++) dB[j] += dy[yOff + j];

    // dW += x^T * dy
    for (let i = 0; i < inDim; i++) {
      const xi = x[xOff + i];
      const dWOff = i * outDim;
      for (let j = 0; j < outDim; j++) {
        dW[dWOff + j] += xi * dy[yOff + j];
      }
    }

    // dx += dy * W^T
    for (let i = 0; i < inDim; i++) {
      let s = 0;
      const wOff = i * outDim;
      for (let j = 0; j < outDim; j++) {
        s += dy[yOff + j] * w[wOff + j];
      }
      dx[xOff + i] += s;
    }
  }
}

// ============================================================================
// INTERNAL: PARAM REGISTRY
// ============================================================================

function mkParam(name: string, w: Float64Array, registry: Param[]): Param {
  const g = new Float64Array(w.length);
  const m = new Float64Array(w.length);
  const v = new Float64Array(w.length);
  const p: Param = { name, w, g, m, v };
  registry.push(p);
  return p;
}

// ============================================================================
// INTERNAL: MULTI-SCALE CONV1D (SAME-PAD, STRIDE=S)
// ============================================================================

class Conv1DScale {
  readonly w: Param; // [kernel*inDim*embed]
  readonly b: Param; // [embed]
  readonly stride: number;
  readonly kernel: number;

  // caches (maxSeqLen x embed), (maxSeqLen x embed)
  private _preAct: Float64Array;
  private _out: Float64Array;

  private _maxSeq: number;
  private _inDim: number;
  private _embed: number;

  constructor(
    name: string,
    stride: number,
    kernel: number,
    inDim: number,
    embed: number,
    maxSeqLen: number,
    registry: Param[],
  ) {
    this.stride = stride;
    this.kernel = kernel;
    this._maxSeq = maxSeqLen;
    this._inDim = inDim;
    this._embed = embed;

    const wArr = new Float64Array(kernel * inDim * embed);
    const bArr = new Float64Array(embed);

    const a = xavierLimit(kernel * inDim, embed);
    fillUniform(wArr, a);
    bArr.fill(0);

    this.w = mkParam(`${name}.w`, wArr, registry);
    this.b = mkParam(`${name}.b`, bArr, registry);

    this._preAct = new Float64Array(maxSeqLen * embed);
    this._out = new Float64Array(maxSeqLen * embed);
  }

  forward(x: Float64Array, seqLen: number, y: Float64Array): number {
    // SAME padding: pad = floor(kernel/2), outLen = ceil(seqLen/stride)
    const stride = this.stride;
    const kernel = this.kernel;
    const pad = kernel >> 1;
    const embed = this._embed;
    const inDim = this._inDim;
    const w = this.w.w;
    const b = this.b.w;

    const outLen = (seqLen + stride - 1) / stride | 0; // ceil for ints
    // clear outputs (only used portion)
    // (avoid fill on whole max buffer)
    for (let i = 0; i < outLen * embed; i++) {
      this._preAct[i] = 0;
      this._out[i] = 0;
      y[i] = 0;
    }

    for (let o = 0; o < outLen; o++) {
      const center = o * stride;
      const outOff = o * embed;
      for (let e = 0; e < embed; e++) {
        let s = b[e];
        // weight index base for each kernel/feature: ((k*inDim + f)*embed + e)
        for (let k = 0; k < kernel; k++) {
          const idx = center + k - pad;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * inDim;
          const wkBase = (k * inDim) * embed + e;
          // loop features; w jumps by embed each f
          let wOff = wkBase;
          for (let f = 0; f < inDim; f++) {
            s += x[xOff + f] * w[wOff];
            wOff += embed;
          }
        }
        this._preAct[outOff + e] = s;
        const a = gelu(s);
        this._out[outOff + e] = a;
        y[outOff + e] = a;
      }
    }
    return outLen;
  }

  backward(
    x: Float64Array, // [seqLen*inDim]
    seqLen: number,
    outLen: number,
    dy: Float64Array, // [outLen*embed]
    dxAcc: Float64Array, // [seqLen*inDim] (accumulate across scales)
  ): void {
    const stride = this.stride;
    const kernel = this.kernel;
    const pad = kernel >> 1;
    const embed = this._embed;
    const inDim = this._inDim;
    const w = this.w.w;
    const dW = this.w.g;
    const dB = this.b.g;

    dW.fill(0);
    dB.fill(0);

    // dPreAct = dy * gelu'(preAct)
    // compute on the fly to avoid another buffer
    for (let o = 0; o < outLen; o++) {
      const center = o * stride;
      const outOff = o * embed;

      for (let e = 0; e < embed; e++) {
        const g = dy[outOff + e] * geluGrad(this._preAct[outOff + e]);
        dB[e] += g;

        for (let k = 0; k < kernel; k++) {
          const idx = center + k - pad;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * inDim;
          const wkBase = (k * inDim) * embed + e;
          let wOff = wkBase;
          for (let f = 0; f < inDim; f++) {
            const xv = x[xOff + f];
            dW[wOff] += xv * g;
            dxAcc[xOff + f] += w[wOff] * g;
            wOff += embed;
          }
        }
      }
    }
  }

  getOutBuffer(): Float64Array {
    return this._out;
  }

  toJSON(): any {
    return {
      w: bufferToNumber1D(this.w.w),
      b: bufferToNumber1D(this.b.w),
      mw: bufferToNumber1D(this.w.m),
      vw: bufferToNumber1D(this.w.v),
      mb: bufferToNumber1D(this.b.m),
      vb: bufferToNumber1D(this.b.v),
      stride: this.stride,
      kernel: this.kernel,
    };
  }

  fromJSON(obj: any): void {
    const ww = obj.w ?? [];
    const bb = obj.b ?? [];
    for (let i = 0; i < this.w.w.length; i++) this.w.w[i] = ww[i] ?? 0;
    for (let i = 0; i < this.b.w.length; i++) this.b.w[i] = bb[i] ?? 0;

    const mw = obj.mw ?? [];
    const vw = obj.vw ?? [];
    const mb = obj.mb ?? [];
    const vb = obj.vb ?? [];
    for (let i = 0; i < this.w.m.length; i++) this.w.m[i] = mw[i] ?? 0;
    for (let i = 0; i < this.w.v.length; i++) this.w.v[i] = vw[i] ?? 0;
    for (let i = 0; i < this.b.m.length; i++) this.b.m[i] = mb[i] ?? 0;
    for (let i = 0; i < this.b.v.length; i++) this.b.v[i] = vb[i] ?? 0;
  }
}

// ============================================================================
// INTERNAL: GATED CROSS-SCALE FUSION
// ============================================================================

class FusionGate {
  readonly gateW: Param; // [(numScales*embed) * numScales]
  readonly gateB: Param; // [numScales]

  private _numScales: number;
  private _embed: number;
  private _maxSeq: number;

  // cache gates: [maxSeqLen*numScales]
  private _gates: Float64Array;

  constructor(
    name: string,
    numScales: number,
    embed: number,
    maxSeqLen: number,
    registry: Param[],
  ) {
    this._numScales = numScales;
    this._embed = embed;
    this._maxSeq = maxSeqLen;

    const inDim = numScales * embed;
    const outDim = numScales;

    const wArr = new Float64Array(inDim * outDim);
    const bArr = new Float64Array(outDim);

    const a = xavierLimit(inDim, outDim);
    fillUniform(wArr, a);
    bArr.fill(0);

    this.gateW = mkParam(`${name}.gateW`, wArr, registry);
    this.gateB = mkParam(`${name}.gateB`, bArr, registry);

    this._gates = new Float64Array(maxSeqLen * numScales);
  }

  /**
   * Fuses multiple scale sequences into fused sequence length = seqLenFine (scale=1 output length).
   *
   * For each time t (0..seqLenFine-1):
   *  - gather E_s at idx_s = floor(t/scale_s)
   *  - z_s = dot(concat(E_1..E_S), Wg[:,s]) + bg[s]
   *  - g_s = sigmoid(z_s)
   *  - fused[t] = sum_s g_s * E_s(idx_s)
   */
  forward(
    scaleSeqs: Float64Array[], // each [outLen_s * embed]
    outLens: Int32Array, // [numScales]
    scales: Int32Array, // [numScales]
    seqLenFine: number, // typically outLen at scale=1 (== input seqLen with SAME conv)
    fusedOut: Float64Array, // [seqLenFine*embed]
  ): void {
    const S = this._numScales;
    const E = this._embed;
    const W = this.gateW.w;
    const b = this.gateB.w;
    const gates = this._gates;

    // clear fusedOut used portion
    for (let i = 0; i < seqLenFine * E; i++) fusedOut[i] = 0;

    const concatDim = S * E;

    for (let t = 0; t < seqLenFine; t++) {
      const fusedOff = t * E;
      const gateOff = t * S;

      // compute z_s and gates
      for (let s = 0; s < S; s++) {
        // z_s = b[s] + sum_{k=0..concatDim-1} concat[k] * W[k*S + s]
        let z = b[s];

        // concat is virtual: concat[block + j] = E_block[j]
        // loop scales then embed
        let kIndex = 0;
        for (let sc = 0; sc < S; sc++) {
          const stride = scales[sc];
          const idx = (t / stride) | 0;
          const len = outLens[sc];
          const useIdx = idx < len ? idx : (len - 1);
          const seq = scaleSeqs[sc];
          const off = useIdx * E;

          // W index base for current kIndex: (kIndex*S + s)
          for (let j = 0; j < E; j++) {
            const x = seq[off + j];
            z += x * W[(kIndex * S) + s];
            kIndex++;
          }
        }

        const g = sigmoid(z);
        gates[gateOff + s] = g;
      }

      // fused = sum_s g_s * E_s(idx_s)
      for (let s = 0; s < S; s++) {
        const stride = scales[s];
        const idx = (t / stride) | 0;
        const len = outLens[s];
        const useIdx = idx < len ? idx : (len - 1);
        const seq = scaleSeqs[s];
        const off = useIdx * E;
        const g = gates[gateOff + s];

        for (let j = 0; j < E; j++) {
          fusedOut[fusedOff + j] += g * seq[off + j];
        }
      }
    }
  }

  backward(
    scaleSeqs: Float64Array[],
    outLens: Int32Array,
    scales: Int32Array,
    seqLenFine: number,
    dFused: Float64Array, // [seqLenFine*embed]
    dScaleSeqs: Float64Array[], // each [outLen_s*embed], must be pre-zeroed by caller
  ): void {
    const S = this._numScales;
    const E = this._embed;
    const W = this.gateW.w;
    const dW = this.gateW.g;
    const db = this.gateB.g;
    const gates = this._gates;

    dW.fill(0);
    db.fill(0);

    const concatDim = S * E;

    // For each t:
    // fused = sum_s g_s * Es
    // dg_s = dot(dFused, Es)
    // dz_s = dg_s * g_s*(1-g_s)
    // dW[:,s] += concat * dz_s
    // db[s] += dz_s
    // dconcat += W[:,s]*dz_s → distribute to Es
    // plus direct dEs += g_s * dFused
    for (let t = 0; t < seqLenFine; t++) {
      const fusedOff = t * E;
      const gateOff = t * S;

      // compute dg_s
      // we need Es vectors (virtual) multiple times; fetch idx and offsets
      // store offsets in small local arrays (avoid allocations by fixed loops)
      // (S is small; linear scan is fine)
      const idxArr = new Int32Array(S);
      const offArr = new Int32Array(S);
      for (let s = 0; s < S; s++) {
        const stride = scales[s];
        const idx = (t / stride) | 0;
        const len = outLens[s];
        const useIdx = idx < len ? idx : (len - 1);
        idxArr[s] = useIdx;
        offArr[s] = useIdx * E;
      }

      // direct: dEs += g_s * dFused
      for (let s = 0; s < S; s++) {
        const g = gates[gateOff + s];
        const seq = dScaleSeqs[s];
        const off = offArr[s];
        for (let j = 0; j < E; j++) {
          seq[off + j] += g * dFused[fusedOff + j];
        }
      }

      // dz_s and param grads
      // dconcat accumulation is distributed directly into dScaleSeqs
      for (let s = 0; s < S; s++) {
        const g = gates[gateOff + s];
        // dg_s = dot(dFused, Es)
        let dg = 0;
        const Es = scaleSeqs[s];
        const eOff = offArr[s];
        for (let j = 0; j < E; j++) dg += dFused[fusedOff + j] * Es[eOff + j];

        const dz = dg * g * (1 - g);
        db[s] += dz;

        // dW[:,s] += concat * dz
        // concat is virtual: loop scale blocks then embed
        let kIndex = 0;
        for (let sc = 0; sc < S; sc++) {
          const Es2 = scaleSeqs[sc];
          const off2 = offArr[sc];
          for (let j = 0; j < E; j++) {
            dW[(kIndex * S) + s] += Es2[off2 + j] * dz;
            kIndex++;
          }
        }

        // dconcat += W[:,s]*dz, distribute into dScaleSeqs
        kIndex = 0;
        for (let sc = 0; sc < S; sc++) {
          const dEs2 = dScaleSeqs[sc];
          const off2 = offArr[sc];
          for (let j = 0; j < E; j++) {
            dEs2[off2 + j] += W[(kIndex * S) + s] * dz;
            kIndex++;
          }
        }
      }
    }
  }

  getGatesBuffer(): Float64Array {
    return this._gates;
  }

  toJSON(): any {
    return {
      gateW: bufferToNumber1D(this.gateW.w),
      gateB: bufferToNumber1D(this.gateB.w),
      mW: bufferToNumber1D(this.gateW.m),
      vW: bufferToNumber1D(this.gateW.v),
      mB: bufferToNumber1D(this.gateB.m),
      vB: bufferToNumber1D(this.gateB.v),
    };
  }

  fromJSON(obj: any): void {
    const ww = obj.gateW ?? [];
    const bb = obj.gateB ?? [];
    for (let i = 0; i < this.gateW.w.length; i++) this.gateW.w[i] = ww[i] ?? 0;
    for (let i = 0; i < this.gateB.w.length; i++) this.gateB.w[i] = bb[i] ?? 0;

    const mW = obj.mW ?? [];
    const vW = obj.vW ?? [];
    const mB = obj.mB ?? [];
    const vB = obj.vB ?? [];
    for (let i = 0; i < this.gateW.m.length; i++) this.gateW.m[i] = mW[i] ?? 0;
    for (let i = 0; i < this.gateW.v.length; i++) this.gateW.v[i] = vW[i] ?? 0;
    for (let i = 0; i < this.gateB.m.length; i++) this.gateB.m[i] = mB[i] ?? 0;
    for (let i = 0; i < this.gateB.v.length; i++) this.gateB.v[i] = vB[i] ?? 0;
  }
}

// ============================================================================
// INTERNAL: MULTI-HEAD SELF-ATTENTION (CAUSAL, FULL BACKWARD)
// ============================================================================

class MultiHeadSelfAttention {
  readonly wq: Param;
  readonly wk: Param;
  readonly wv: Param;
  readonly wo: Param;
  readonly bq: Param;
  readonly bk: Param;
  readonly bv: Param;
  readonly bo: Param;

  private _embed: number;
  private _heads: number;
  private _dk: number;
  private _maxSeq: number;

  // caches
  private _q: Float64Array; // [maxSeq*embed]
  private _k: Float64Array; // [maxSeq*embed]
  private _v: Float64Array; // [maxSeq*embed]
  private _context: Float64Array; // [maxSeq*embed]
  private _attn: Float64Array; // [heads*maxSeq*maxSeq] probs
  private _tmpScores: Float64Array; // [maxSeq] scratch for one row (no alloc)
  private _xIn: Float64Array; // reference buffer for current forward input (copied for backward) [maxSeq*embed]

  constructor(
    name: string,
    maxSeqLen: number,
    embed: number,
    heads: number,
    registry: Param[],
  ) {
    this._embed = embed;
    this._heads = heads;
    this._dk = (embed / heads) | 0;
    this._maxSeq = maxSeqLen;

    const e = embed;

    const wq = new Float64Array(e * e);
    const wk = new Float64Array(e * e);
    const wv = new Float64Array(e * e);
    const wo = new Float64Array(e * e);
    const bq = new Float64Array(e);
    const bk = new Float64Array(e);
    const bv = new Float64Array(e);
    const bo = new Float64Array(e);

    const a = xavierLimit(e, e);
    fillUniform(wq, a);
    fillUniform(wk, a);
    fillUniform(wv, a);
    fillUniform(wo, a);
    bq.fill(0);
    bk.fill(0);
    bv.fill(0);
    bo.fill(0);

    this.wq = mkParam(`${name}.wq`, wq, registry);
    this.wk = mkParam(`${name}.wk`, wk, registry);
    this.wv = mkParam(`${name}.wv`, wv, registry);
    this.wo = mkParam(`${name}.wo`, wo, registry);
    this.bq = mkParam(`${name}.bq`, bq, registry);
    this.bk = mkParam(`${name}.bk`, bk, registry);
    this.bv = mkParam(`${name}.bv`, bv, registry);
    this.bo = mkParam(`${name}.bo`, bo, registry);

    this._q = new Float64Array(maxSeqLen * e);
    this._k = new Float64Array(maxSeqLen * e);
    this._v = new Float64Array(maxSeqLen * e);
    this._context = new Float64Array(maxSeqLen * e);

    // probs: heads * maxSeq * maxSeq
    this._attn = new Float64Array(heads * maxSeqLen * maxSeqLen);
    this._tmpScores = new Float64Array(maxSeqLen);
    this._xIn = new Float64Array(maxSeqLen * e);
  }

  forward(x: Float64Array, seqLen: number, y: Float64Array): void {
    const e = this._embed;

    // cache input copy for backward (avoid referencing external changing buffers)
    for (let i = 0; i < seqLen * e; i++) this._xIn[i] = x[i];

    // Q,K,V projections
    linearForwardSeq(x, seqLen, e, this.wq.w, this.bq.w, e, this._q);
    linearForwardSeq(x, seqLen, e, this.wk.w, this.bk.w, e, this._k);
    linearForwardSeq(x, seqLen, e, this.wv.w, this.bv.w, e, this._v);

    const H = this._heads;
    const dk = this._dk;
    const invSqrtDk = 1 / Math.sqrt(dk);

    // context = Attn(Q,K,V)
    for (let t = 0; t < seqLen * e; t++) this._context[t] = 0;
    // attn probs fill only used region
    const maxSeq = this._maxSeq;

    for (let h = 0; h < H; h++) {
      const headOff = h * maxSeq * maxSeq;

      for (let i = 0; i < seqLen; i++) {
        // scores over j<=i (causal)
        const qiOff = i * e + h * dk;

        // build scores into tmpScores[0..seqLen-1]
        let max = -Infinity;
        for (let j = 0; j < seqLen; j++) {
          let s = -1e30; // masked
          if (j <= i) {
            const kjOff = j * e + h * dk;
            let dot = 0;
            for (let d = 0; d < dk; d++) {
              dot += this._q[qiOff + d] * this._k[kjOff + d];
            }
            s = dot * invSqrtDk;
          }
          this._tmpScores[j] = s;
          if (s > max) max = s;
        }

        // softmax row
        let sum = 0;
        const rowBase = headOff + i * maxSeq;
        for (let j = 0; j < seqLen; j++) {
          const ex = Math.exp(this._tmpScores[j] - max);
          this._attn[rowBase + j] = ex;
          sum += ex;
        }
        const inv = sum > 0 ? (1 / sum) : 1;
        for (let j = 0; j < seqLen; j++) {
          this._attn[rowBase + j] *= inv;
        }

        // context_i += sum_j attn_ij * V_j
        const ciOff = i * e + h * dk;
        for (let d = 0; d < dk; d++) {
          let acc = 0;
          for (let j = 0; j < seqLen; j++) {
            const pj = this._attn[rowBase + j];
            const vjOff = j * e + h * dk;
            acc += pj * this._v[vjOff + d];
          }
          this._context[ciOff + d] = acc;
        }
      }
    }

    // output projection: y = context * Wo + bo
    linearForwardSeq(this._context, seqLen, e, this.wo.w, this.bo.w, e, y);
  }

  backward(
    seqLen: number,
    dy: Float64Array, // [seqLen*embed]
    dx: Float64Array, // [seqLen*embed]
  ): void {
    const e = this._embed;
    const H = this._heads;
    const dk = this._dk;
    const invSqrtDk = 1 / Math.sqrt(dk);
    const maxSeq = this._maxSeq;

    // Step 1: backprop through output projection y = context*Wo + bo
    // Need dContext and grads for Wo,bo
    const dContext = new Float64Array(seqLen * e);
    // linearBackwardSeq overwrites dx; here we want dContext as dx and keep dy input
    linearBackwardSeq(
      this._context,
      seqLen,
      e,
      this.wo.w,
      e,
      dy,
      dContext,
      this.wo.g,
      this.bo.g,
    );

    // Step 2: backprop attention to Q,K,V
    const dQ = new Float64Array(seqLen * e);
    const dK = new Float64Array(seqLen * e);
    const dV = new Float64Array(seqLen * e);
    dQ.fill(0);
    dK.fill(0);
    dV.fill(0);

    // For each head:
    // - dV_j += sum_i attn_ij * dContext_i
    // - dAttn_ij = dot(dContext_i, V_j)
    // - dScore row via softmax jacobian
    // - dQ_i += sum_j dScore_ij * K_j / sqrt(dk)
    // - dK_j += sum_i dScore_ij * Q_i / sqrt(dk)
    const dAttnRow = this._tmpScores; // reuse as row buffer for dAttn or dScore
    for (let h = 0; h < H; h++) {
      const headOff = h * maxSeq * maxSeq;

      // dV (accumulate)
      for (let j = 0; j < seqLen; j++) {
        const vjOff = j * e + h * dk;
        for (let d = 0; d < dk; d++) {
          let acc = 0;
          for (let i = 0; i < seqLen; i++) {
            const rowBase = headOff + i * maxSeq;
            const pij = this._attn[rowBase + j];
            const diOff = i * e + h * dk;
            acc += pij * dContext[diOff + d];
          }
          dV[vjOff + d] += acc;
        }
      }

      // dScores + dQ/dK
      for (let i = 0; i < seqLen; i++) {
        const rowBase = headOff + i * maxSeq;
        const qiOff = i * e + h * dk;

        // dAttn_ij = dot(dContext_i, V_j)
        // and compute row sum for softmax derivative
        let sumP_dAttn = 0;
        for (let j = 0; j < seqLen; j++) {
          let da = 0;
          if (j <= i) {
            const vjOff = j * e + h * dk;
            const diOff = i * e + h * dk;
            for (let d = 0; d < dk; d++) {
              da += dContext[diOff + d] * this._v[vjOff + d];
            }
          } else {
            da = 0;
          }
          dAttnRow[j] = da;
          sumP_dAttn += this._attn[rowBase + j] * da;
        }

        // dScore_ij = p_ij * (dAttn_ij - sum_k p_ik dAttn_ik)
        for (let j = 0; j < seqLen; j++) {
          const p = this._attn[rowBase + j];
          let ds = 0;
          if (j <= i) ds = p * (dAttnRow[j] - sumP_dAttn);
          dAttnRow[j] = ds;
        }

        // dQ_i += sum_j dScore_ij * K_j / sqrt(dk)
        for (let d = 0; d < dk; d++) {
          let acc = 0;
          for (let j = 0; j < seqLen; j++) {
            if (j > i) break;
            const kjOff = j * e + h * dk;
            acc += dAttnRow[j] * this._k[kjOff + d];
          }
          dQ[qiOff + d] += acc * invSqrtDk;
        }

        // dK_j += dScore_ij * Q_i / sqrt(dk)
        for (let j = 0; j < seqLen; j++) {
          if (j > i) break;
          const kjOff = j * e + h * dk;
          const ds = dAttnRow[j] * invSqrtDk;
          for (let d = 0; d < dk; d++) {
            dK[kjOff + d] += ds * this._q[qiOff + d];
          }
        }
      }
    }

    // Step 3: backprop Q,K,V projections into x, and grads for Wq/Wk/Wv and biases
    const dxQ = new Float64Array(seqLen * e);
    const dxK = new Float64Array(seqLen * e);
    const dxV = new Float64Array(seqLen * e);

    // Q = x*Wq + bq
    linearBackwardSeq(
      this._xIn,
      seqLen,
      e,
      this.wq.w,
      e,
      dQ,
      dxQ,
      this.wq.g,
      this.bq.g,
    );
    linearBackwardSeq(
      this._xIn,
      seqLen,
      e,
      this.wk.w,
      e,
      dK,
      dxK,
      this.wk.g,
      this.bk.g,
    );
    linearBackwardSeq(
      this._xIn,
      seqLen,
      e,
      this.wv.w,
      e,
      dV,
      dxV,
      this.wv.g,
      this.bv.g,
    );

    // dx = dxQ + dxK + dxV
    dx.fill(0);
    for (let i = 0; i < seqLen * e; i++) dx[i] = dxQ[i] + dxK[i] + dxV[i];
  }

  toJSON(): any {
    return {
      wq: bufferToNumber1D(this.wq.w),
      wk: bufferToNumber1D(this.wk.w),
      wv: bufferToNumber1D(this.wv.w),
      wo: bufferToNumber1D(this.wo.w),
      bq: bufferToNumber1D(this.bq.w),
      bk: bufferToNumber1D(this.bk.w),
      bv: bufferToNumber1D(this.bv.w),
      bo: bufferToNumber1D(this.bo.w),
      mq: bufferToNumber1D(this.wq.m),
      vq: bufferToNumber1D(this.wq.v),
      mk: bufferToNumber1D(this.wk.m),
      vk: bufferToNumber1D(this.wk.v),
      mv: bufferToNumber1D(this.wv.m),
      vv: bufferToNumber1D(this.wv.v),
      mo: bufferToNumber1D(this.wo.m),
      vo: bufferToNumber1D(this.wo.v),
      mbq: bufferToNumber1D(this.bq.m),
      vbq: bufferToNumber1D(this.bq.v),
      mbk: bufferToNumber1D(this.bk.m),
      vbk: bufferToNumber1D(this.bk.v),
      mbv: bufferToNumber1D(this.bv.m),
      vbv: bufferToNumber1D(this.bv.v),
      mbo: bufferToNumber1D(this.bo.m),
      vbo: bufferToNumber1D(this.bo.v),
    };
  }

  fromJSON(obj: any): void {
    const sets = (dst: Float64Array, src: number[], def: number) => {
      for (let i = 0; i < dst.length; i++) dst[i] = src[i] ?? def;
    };
    sets(this.wq.w, obj.wq ?? [], 0);
    sets(this.wk.w, obj.wk ?? [], 0);
    sets(this.wv.w, obj.wv ?? [], 0);
    sets(this.wo.w, obj.wo ?? [], 0);
    sets(this.bq.w, obj.bq ?? [], 0);
    sets(this.bk.w, obj.bk ?? [], 0);
    sets(this.bv.w, obj.bv ?? [], 0);
    sets(this.bo.w, obj.bo ?? [], 0);

    sets(this.wq.m, obj.mq ?? [], 0);
    sets(this.wq.v, obj.vq ?? [], 0);
    sets(this.wk.m, obj.mk ?? [], 0);
    sets(this.wk.v, obj.vk ?? [], 0);
    sets(this.wv.m, obj.mv ?? [], 0);
    sets(this.wv.v, obj.vv ?? [], 0);
    sets(this.wo.m, obj.mo ?? [], 0);
    sets(this.wo.v, obj.vo ?? [], 0);

    sets(this.bq.m, obj.mbq ?? [], 0);
    sets(this.bq.v, obj.vbq ?? [], 0);
    sets(this.bk.m, obj.mbk ?? [], 0);
    sets(this.bk.v, obj.vbk ?? [], 0);
    sets(this.bv.m, obj.mbv ?? [], 0);
    sets(this.bv.v, obj.vbv ?? [], 0);
    sets(this.bo.m, obj.mbo ?? [], 0);
    sets(this.bo.v, obj.vbo ?? [], 0);
  }
}

// ============================================================================
// INTERNAL: FFN (GELU) WITH FULL BACKWARD
// ============================================================================

class FeedForward {
  readonly w1: Param; // [embed*hidden]
  readonly b1: Param; // [hidden]
  readonly w2: Param; // [hidden*embed]
  readonly b2: Param; // [embed]

  private _embed: number;
  private _hidden: number;
  private _maxSeq: number;

  // caches
  private _pre1: Float64Array; // [maxSeq*hidden]
  private _act1: Float64Array; // [maxSeq*hidden]
  private _xIn: Float64Array; // [maxSeq*embed]

  constructor(
    name: string,
    maxSeqLen: number,
    embed: number,
    hidden: number,
    registry: Param[],
  ) {
    this._embed = embed;
    this._hidden = hidden;
    this._maxSeq = maxSeqLen;

    const w1 = new Float64Array(embed * hidden);
    const b1 = new Float64Array(hidden);
    const w2 = new Float64Array(hidden * embed);
    const b2 = new Float64Array(embed);

    fillUniform(w1, xavierLimit(embed, hidden));
    fillUniform(w2, xavierLimit(hidden, embed));
    b1.fill(0);
    b2.fill(0);

    this.w1 = mkParam(`${name}.w1`, w1, registry);
    this.b1 = mkParam(`${name}.b1`, b1, registry);
    this.w2 = mkParam(`${name}.w2`, w2, registry);
    this.b2 = mkParam(`${name}.b2`, b2, registry);

    this._pre1 = new Float64Array(maxSeqLen * hidden);
    this._act1 = new Float64Array(maxSeqLen * hidden);
    this._xIn = new Float64Array(maxSeqLen * embed);
  }

  forward(x: Float64Array, seqLen: number, y: Float64Array): void {
    const e = this._embed;
    const h = this._hidden;

    // cache x
    for (let i = 0; i < seqLen * e; i++) this._xIn[i] = x[i];

    // pre1 = x*w1 + b1
    // act1 = GELU(pre1)
    // y = act1*w2 + b2
    // compute pre1/act1
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * e;
      const pOff = t * h;

      for (let j = 0; j < h; j++) {
        let s = this.b1.w[j];
        let wOff = j; // w1 row-major: w1[i*h + j]
        for (let i = 0; i < e; i++) {
          s += x[xOff + i] * this.w1.w[wOff];
          wOff += h;
        }
        this._pre1[pOff + j] = s;
        this._act1[pOff + j] = gelu(s);
      }
    }

    // y = act1*w2 + b2
    // w2 is [h*e] row-major: w2[j*e + k]
    for (let t = 0; t < seqLen; t++) {
      const aOff = t * h;
      const yOff = t * e;
      for (let k = 0; k < e; k++) {
        let s = this.b2.w[k];
        for (let j = 0; j < h; j++) {
          s += this._act1[aOff + j] * this.w2.w[j * e + k];
        }
        y[yOff + k] = s;
      }
    }
  }

  backward(seqLen: number, dy: Float64Array, dx: Float64Array): void {
    const e = this._embed;
    const h = this._hidden;

    // grads
    this.w1.g.fill(0);
    this.b1.g.fill(0);
    this.w2.g.fill(0);
    this.b2.g.fill(0);

    // dAct1 = dy * W2^T
    // dW2 = act1^T * dy
    // dB2 = sum dy
    const dAct1 = new Float64Array(seqLen * h);
    dAct1.fill(0);

    for (let t = 0; t < seqLen; t++) {
      const aOff = t * h;
      const yOff = t * e;

      // b2
      for (let k = 0; k < e; k++) this.b2.g[k] += dy[yOff + k];

      // dW2, dAct1
      for (let j = 0; j < h; j++) {
        const aj = this._act1[aOff + j];
        let acc = 0;
        const w2Off = j * e;
        for (let k = 0; k < e; k++) {
          const dyk = dy[yOff + k];
          this.w2.g[w2Off + k] += aj * dyk;
          acc += dyk * this.w2.w[w2Off + k];
        }
        dAct1[aOff + j] = acc;
      }
    }

    // dPre1 = dAct1 * GELU'(pre1)
    // dW1 = x^T * dPre1
    // dB1 = sum dPre1
    // dx = dPre1 * W1^T
    dx.fill(0);

    for (let t = 0; t < seqLen; t++) {
      const xOff = t * e;
      const pOff = t * h;

      // compute dPre1 row
      for (let j = 0; j < h; j++) {
        const dp = dAct1[pOff + j] * geluGrad(this._pre1[pOff + j]);
        this.b1.g[j] += dp;

        // dW1 += x * dp
        let w1Off = j; // w1[i*h + j]
        for (let i = 0; i < e; i++) {
          const xi = this._xIn[xOff + i];
          this.w1.g[w1Off] += xi * dp;
          dx[xOff + i] += this.w1.w[w1Off] * dp;
          w1Off += h;
        }
      }
    }
  }

  toJSON(): any {
    return {
      w1: bufferToNumber1D(this.w1.w),
      b1: bufferToNumber1D(this.b1.w),
      w2: bufferToNumber1D(this.w2.w),
      b2: bufferToNumber1D(this.b2.w),
      mw1: bufferToNumber1D(this.w1.m),
      vw1: bufferToNumber1D(this.w1.v),
      mb1: bufferToNumber1D(this.b1.m),
      vb1: bufferToNumber1D(this.b1.v),
      mw2: bufferToNumber1D(this.w2.m),
      vw2: bufferToNumber1D(this.w2.v),
      mb2: bufferToNumber1D(this.b2.m),
      vb2: bufferToNumber1D(this.b2.v),
    };
  }

  fromJSON(obj: any): void {
    const sets = (dst: Float64Array, src: number[], def: number) => {
      for (let i = 0; i < dst.length; i++) dst[i] = src[i] ?? def;
    };
    sets(this.w1.w, obj.w1 ?? [], 0);
    sets(this.b1.w, obj.b1 ?? [], 0);
    sets(this.w2.w, obj.w2 ?? [], 0);
    sets(this.b2.w, obj.b2 ?? [], 0);

    sets(this.w1.m, obj.mw1 ?? [], 0);
    sets(this.w1.v, obj.vw1 ?? [], 0);
    sets(this.b1.m, obj.mb1 ?? [], 0);
    sets(this.b1.v, obj.vb1 ?? [], 0);
    sets(this.w2.m, obj.mw2 ?? [], 0);
    sets(this.w2.v, obj.vw2 ?? [], 0);
    sets(this.b2.m, obj.mb2 ?? [], 0);
    sets(this.b2.v, obj.vb2 ?? [], 0);
  }
}

// ============================================================================
// INTERNAL: TRANSFORMER BLOCK (LN → MHA → RES → LN → FFN → RES)
// ============================================================================

class TransformerBlock {
  readonly ln1: LayerNorm;
  readonly mha: MultiHeadSelfAttention;
  readonly ln2: LayerNorm;
  readonly ffn: FeedForward;

  private _embed: number;
  private _maxSeq: number;

  // buffers
  private _ln1Out: Float64Array;
  private _mhaOut: Float64Array;
  private _res1: Float64Array;
  private _ln2Out: Float64Array;
  private _ffnOut: Float64Array;

  // backward temps
  private _dLn2In: Float64Array;
  private _dRes1: Float64Array;
  private _dLn1In: Float64Array;

  constructor(
    name: string,
    maxSeqLen: number,
    embed: number,
    heads: number,
    ffnMult: number,
    eps: number,
    registry: Param[],
  ) {
    this._embed = embed;
    this._maxSeq = maxSeqLen;

    this.ln1 = new LayerNorm(`${name}.ln1`, maxSeqLen, embed, eps, registry);
    this.mha = new MultiHeadSelfAttention(
      `${name}.mha`,
      maxSeqLen,
      embed,
      heads,
      registry,
    );
    this.ln2 = new LayerNorm(`${name}.ln2`, maxSeqLen, embed, eps, registry);
    this.ffn = new FeedForward(
      `${name}.ffn`,
      maxSeqLen,
      embed,
      embed * ffnMult,
      registry,
    );

    const n = maxSeqLen * embed;
    this._ln1Out = new Float64Array(n);
    this._mhaOut = new Float64Array(n);
    this._res1 = new Float64Array(n);
    this._ln2Out = new Float64Array(n);
    this._ffnOut = new Float64Array(n);

    this._dLn2In = new Float64Array(n);
    this._dRes1 = new Float64Array(n);
    this._dLn1In = new Float64Array(n);
  }

  forward(x: Float64Array, seqLen: number, y: Float64Array): void {
    const e = this._embed;
    const n = seqLen * e;

    // LN1
    this.ln1.forward(x, seqLen, this._ln1Out);
    // MHA
    this.mha.forward(this._ln1Out, seqLen, this._mhaOut);
    // Residual 1: res1 = x + mhaOut
    for (let i = 0; i < n; i++) this._res1[i] = x[i] + this._mhaOut[i];

    // LN2
    this.ln2.forward(this._res1, seqLen, this._ln2Out);
    // FFN
    this.ffn.forward(this._ln2Out, seqLen, this._ffnOut);
    // Residual 2: y = res1 + ffnOut
    for (let i = 0; i < n; i++) y[i] = this._res1[i] + this._ffnOut[i];
  }

  backward(
    x: Float64Array,
    seqLen: number,
    dy: Float64Array,
    dx: Float64Array,
  ): void {
    const e = this._embed;
    const n = seqLen * e;

    // dy is gradient wrt output of block
    // y = res1 + ffnOut
    // dRes1 = dy (from residual) + dFFNInput (through LN2)
    // dFFNOut = dy
    // FFN backward: input ln2Out, output ffnOut
    const dFfnOut = dy; // alias
    this.ffn.backward(seqLen, dFfnOut, this._dLn2In); // returns d(ln2Out)

    // LN2 backward: ln2Out = LN(res1)
    this.ln2.backward(this._res1, seqLen, this._dLn2In, this._dRes1);

    // residual: res1 = x + mhaOut
    // dX += dRes1
    // dMhaOut = dRes1
    // plus dy from skip directly already included in dRes1? No: dRes1 already includes both paths:
    //   from y->res1 (dy) AND from y->ffn->ln2->res1 (_dRes1)
    // Actually y = res1 + ffnOut, so dRes1_init = dy, then add ln2 path: dRes1 = dy + _dRes1
    for (let i = 0; i < n; i++) this._dRes1[i] += dy[i];

    // MHA backward: mhaOut = MHA(ln1Out)
    this.mha.backward(seqLen, this._dRes1, this._dLn1In);

    // LN1 backward: ln1Out = LN(x)
    const dXFromLn1 = new Float64Array(n);
    this.ln1.backward(x, seqLen, this._dLn1In, dXFromLn1);

    // dx = dRes1 (skip to x) + dXFromLn1
    for (let i = 0; i < n; i++) dx[i] = this._dRes1[i] + dXFromLn1[i];
  }

  toJSON(): any {
    return {
      ln1: this.ln1.toJSON(),
      mha: this.mha.toJSON(),
      ln2: this.ln2.toJSON(),
      ffn: this.ffn.toJSON(),
    };
  }

  fromJSON(obj: any): void {
    this.ln1.fromJSON(obj?.ln1 ?? {});
    this.mha.fromJSON(obj?.mha ?? {});
    this.ln2.fromJSON(obj?.ln2 ?? {});
    this.ffn.fromJSON(obj?.ffn ?? {});
  }
}

// ============================================================================
// INTERNAL: TEMPORAL ATTENTION POOLING
// ============================================================================

class AttentionPooling {
  readonly wPool: Param; // [embed]
  readonly bPool: Param; // [1]

  private _embed: number;
  private _maxSeq: number;

  private _logits: Float64Array; // [maxSeq]
  private _alpha: Float64Array; // [maxSeq]

  constructor(
    name: string,
    maxSeqLen: number,
    embed: number,
    registry: Param[],
  ) {
    this._embed = embed;
    this._maxSeq = maxSeqLen;

    const w = new Float64Array(embed);
    const b = new Float64Array(1);
    fillUniform(w, xavierLimit(embed, 1));
    b[0] = 0;

    this.wPool = mkParam(`${name}.wPool`, w, registry);
    this.bPool = mkParam(`${name}.bPool`, b, registry);

    this._logits = new Float64Array(maxSeqLen);
    this._alpha = new Float64Array(maxSeqLen);
  }

  forward(h: Float64Array, seqLen: number, out: Float64Array): void {
    const e = this._embed;
    const w = this.wPool.w;
    const b = this.bPool.w[0];

    // logits[t] = dot(h_t, w) + b
    let max = -Infinity;
    for (let t = 0; t < seqLen; t++) {
      const off = t * e;
      let s = b;
      for (let j = 0; j < e; j++) s += h[off + j] * w[j];
      this._logits[t] = s;
      if (s > max) max = s;
    }

    // alpha = softmax(logits)
    let sum = 0;
    for (let t = 0; t < seqLen; t++) {
      const ex = Math.exp(this._logits[t] - max);
      this._alpha[t] = ex;
      sum += ex;
    }
    const inv = sum > 0 ? (1 / sum) : 1;
    for (let t = 0; t < seqLen; t++) this._alpha[t] *= inv;

    // out = sum_t alpha[t] * h_t
    out.fill(0);
    for (let t = 0; t < seqLen; t++) {
      const a = this._alpha[t];
      const off = t * e;
      for (let j = 0; j < e; j++) out[j] += a * h[off + j];
    }
  }

  backward(
    h: Float64Array,
    seqLen: number,
    dOut: Float64Array,
    dH: Float64Array,
  ): void {
    const e = this._embed;
    const w = this.wPool.w;
    const dW = this.wPool.g;
    const dB = this.bPool.g;

    dW.fill(0);
    dB.fill(0);
    dH.fill(0);

    // out = sum alpha[t] h_t
    // direct: dH_t += alpha[t] * dOut
    for (let t = 0; t < seqLen; t++) {
      const a = this._alpha[t];
      const off = t * e;
      for (let j = 0; j < e; j++) dH[off + j] += a * dOut[j];
    }

    // dAlpha[t] = dot(dOut, h_t)
    // dl[t] = alpha[t] * (dAlpha[t] - sum_k alpha[k]*dAlpha[k])
    // and l[t] = dot(h_t, w) + b
    // so additional: dH_t += w * dl[t], dW += h_t * dl[t], dB += dl[t]
    let sumA_dAlpha = 0;
    for (let t = 0; t < seqLen; t++) {
      const off = t * e;
      let dAlpha = 0;
      for (let j = 0; j < e; j++) dAlpha += dOut[j] * h[off + j];
      // reuse logits buffer to store dAlpha temporarily
      this._logits[t] = dAlpha;
      sumA_dAlpha += this._alpha[t] * dAlpha;
    }

    for (let t = 0; t < seqLen; t++) {
      const dl = this._alpha[t] * (this._logits[t] - sumA_dAlpha);
      dB[0] += dl;
      const off = t * e;
      for (let j = 0; j < e; j++) {
        dW[j] += h[off + j] * dl;
        dH[off + j] += w[j] * dl;
      }
    }
  }

  toJSON(): any {
    return {
      w: bufferToNumber1D(this.wPool.w),
      b: this.bPool.w[0],
      mw: bufferToNumber1D(this.wPool.m),
      vw: bufferToNumber1D(this.wPool.v),
      mb: bufferToNumber1D(this.bPool.m),
      vb: bufferToNumber1D(this.bPool.v),
    };
  }

  fromJSON(obj: any): void {
    const w = obj.w ?? [];
    for (let i = 0; i < this.wPool.w.length; i++) this.wPool.w[i] = w[i] ?? 0;
    this.bPool.w[0] = obj.b ?? 0;

    const mw = obj.mw ?? [];
    const vw = obj.vw ?? [];
    const mb = obj.mb ?? [];
    const vb = obj.vb ?? [];
    for (let i = 0; i < this.wPool.m.length; i++) this.wPool.m[i] = mw[i] ?? 0;
    for (let i = 0; i < this.wPool.v.length; i++) this.wPool.v[i] = vw[i] ?? 0;
    for (let i = 0; i < this.bPool.m.length; i++) this.bPool.m[i] = mb[i] ?? 0;
    for (let i = 0; i < this.bPool.v.length; i++) this.bPool.v[i] = vb[i] ?? 0;
  }
}

// ============================================================================
// MAIN: FusionTemporalTransformerRegression
// ============================================================================

export class FusionTemporalTransformerRegression {
  private _cfg: FusionTemporalTransformerRegressionConfig;

  private _isInitialized = false;
  private _inputDim = 0;
  private _outputDim = 0;

  private _params: Param[] = [];
  private _adam: AdamOptimizer;

  private _posEnc?: PositionalEncoding;

  private _scalesI32?: Int32Array;
  private _convScales: Conv1DScale[] = [];
  private _scaleEmb?: Param; // [numScales*embed]
  private _fusion?: FusionGate;

  private _blocks: TransformerBlock[] = [];
  private _pool?: AttentionPooling;

  private _wOut?: Param; // [embed*outDim]
  private _bOut?: Param; // [outDim]

  // normalization
  private _inNorm?: WelfordVector;
  private _outNorm?: WelfordVector;

  // residual stats for prediction intervals (per output dim)
  private _residStats?: WelfordVector;

  // drift detection
  private _adwin: ADWINLite;
  private _driftCount = 0;

  // running accuracy (avg loss)
  private _avgLoss = 0;
  private _sampleCount = 0;
  private _converged = false;
  private _effectiveLR = 0;

  // reusable buffers (maxSeqLen dependent)
  private _xRaw?: Float64Array; // [maxSeq*inDim]
  private _xNorm?: Float64Array; // [maxSeq*inDim]
  private _yRaw?: Float64Array; // [outDim]
  private _yNorm?: Float64Array; // [outDim]
  private _yHatNorm?: Float64Array; // [outDim]
  private _yHatRaw?: Float64Array; // [outDim]
  private _dYHat?: Float64Array; // [outDim]

  private _scaleSeqs: Float64Array[] = [];
  private _scaleOutLens?: Int32Array;
  private _dScaleSeqs: Float64Array[] = [];

  private _fused?: Float64Array; // [maxSeq*embed]
  private _dFused?: Float64Array; // [maxSeq*embed]

  private _blockBufA?: Float64Array; // [maxSeq*embed]
  private _blockBufB?: Float64Array; // [maxSeq*embed]
  private _dBlock?: Float64Array; // [maxSeq*embed]

  private _pooled?: Float64Array; // [embed]
  private _dPooled?: Float64Array; // [embed]

  private _dXNorm?: Float64Array; // [maxSeq*inDim] accumulated from conv backprops

  // last seen seqLen (after truncation)
  private _lastSeqLen = 0;

  constructor(cfg: Partial<FusionTemporalTransformerRegressionConfig> = {}) {
    this._cfg = {
      numBlocks: cfg.numBlocks ?? 3,
      embeddingDim: cfg.embeddingDim ?? 64,
      numHeads: cfg.numHeads ?? 8,
      ffnMultiplier: cfg.ffnMultiplier ?? 4,
      attentionDropout: cfg.attentionDropout ?? 0.0,
      learningRate: cfg.learningRate ?? 0.001,
      warmupSteps: cfg.warmupSteps ?? 100,
      totalSteps: cfg.totalSteps ?? 10000,
      beta1: cfg.beta1 ?? 0.9,
      beta2: cfg.beta2 ?? 0.999,
      epsilon: cfg.epsilon ?? 1e-8,
      regularizationStrength: cfg.regularizationStrength ?? 1e-4,
      convergenceThreshold: cfg.convergenceThreshold ?? 1e-6,
      outlierThreshold: cfg.outlierThreshold ?? 3.0,
      adwinDelta: cfg.adwinDelta ?? 0.002,
      temporalScales: cfg.temporalScales ?? [1, 2, 4],
      temporalKernelSize: cfg.temporalKernelSize ?? 3,
      maxSequenceLength: cfg.maxSequenceLength ?? 512,
      fusionDropout: cfg.fusionDropout ?? 0.0,
    };

    // guard heads
    if ((this._cfg.embeddingDim % this._cfg.numHeads) !== 0) {
      throw new Error(
        `embeddingDim (${this._cfg.embeddingDim}) must be divisible by numHeads (${this._cfg.numHeads}).`,
      );
    }

    this._adam = new AdamOptimizer(
      this._cfg.beta1,
      this._cfg.beta2,
      this._cfg.epsilon,
    );
    this._adwin = new ADWINLite(
      this._cfg.adwinDelta,
      Math.max(64, Math.min(1024, this._cfg.maxSequenceLength)),
    );
  }

  // ==========================================================================
  // PUBLIC API
  // ==========================================================================

  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xCoords = data.xCoordinates;
    const yCoords = data.yCoordinates;
    const seqLenRaw = xCoords.length | 0;
    if (seqLenRaw <= 0) {
      return {
        loss: 0,
        gradientNorm: 0,
        effectiveLearningRate: this._effectiveLR,
        isOutlier: false,
        converged: this._converged,
        sampleIndex: this._sampleCount,
        driftDetected: false,
      };
    }

    // auto detect dims (from first row)
    const inDim = (xCoords[0]?.length ?? 0) | 0;
    const outDim =
      (yCoords.length > 0 ? (yCoords[yCoords.length - 1]?.length ?? 0) : 0) | 0;
    if (inDim <= 0 || outDim <= 0) {
      throw new Error(
        `Invalid dimensions: inputDim=${inDim}, outputDim=${outDim}`,
      );
    }

    // initialize on first call
    if (!this._isInitialized) {
      this._initialize(inDim, outDim);
    }

    // truncate to maxSequenceLength (keep most recent)
    const maxSeq = this._cfg.maxSequenceLength | 0;
    const seqLen = seqLenRaw > maxSeq ? maxSeq : seqLenRaw;
    this._lastSeqLen = seqLen;

    // Copy x into raw buffer (most recent window)
    const xRaw = this._xRaw!;
    // build from tail if truncated
    const start = seqLenRaw - seqLen;
    let p = 0;
    for (let t = 0; t < seqLen; t++) {
      const row = xCoords[start + t];
      for (let j = 0; j < inDim; j++) xRaw[p++] = row[j] ?? 0;
    }

    // target y = last row
    const yLast = yCoords[yCoords.length - 1];
    const yRaw = this._yRaw!;
    for (let j = 0; j < outDim; j++) yRaw[j] = yLast[j] ?? 0;

    // normalize inputs/outputs using PRE-update stats (avoids leaking current sample)
    const eps = this._cfg.epsilon;
    const xNorm = this._xNorm!;
    const yNorm = this._yNorm!;

    const inNorm = this._inNorm!;
    const outNorm = this._outNorm!;
    const inMean = inNorm.mean;
    const inStd = inNorm.getStd(eps); // alloc small; acceptable (inDim <= ~few hundred)
    const outMean = outNorm.mean;
    const outStd = outNorm.getStd(eps); // alloc small

    // xNorm[t,j] = (xRaw - mean)/std
    p = 0;
    for (let t = 0; t < seqLen; t++) {
      for (let j = 0; j < inDim; j++) {
        xNorm[p] = (xRaw[p] - inMean[j]) / (inStd[j] + eps);
        p++;
      }
    }

    for (let j = 0; j < outDim; j++) {
      yNorm[j] = (yRaw[j] - outMean[j]) / (outStd[j] + eps);
    }

    // forward pass
    this._zeroAllGrads();
    this._forward(seqLen, /*posOffset*/ 0);

    // loss (normalized MSE) + L2
    const yHatNorm = this._yHatNorm!;
    let mse = 0;
    for (let j = 0; j < outDim; j++) {
      const eij = yHatNorm[j] - yNorm[j];
      mse += eij * eij;
    }
    mse /= outDim;
    const lossData = 0.5 * mse;

    // L2 regularization term (λ/2) Σ||W||^2 (for reporting)
    let l2sum = 0;
    const lam = this._cfg.regularizationStrength;
    for (let pi = 0; pi < this._params.length; pi++) {
      const w = this._params[pi].w;
      for (let i = 0; i < w.length; i++) l2sum += w[i] * w[i];
    }
    const lossReg = 0.5 * lam * l2sum;
    const loss = lossData + lossReg;

    // compute predicted raw (for outlier + residual stats)
    const yHatRaw = this._yHatRaw!;
    for (let j = 0; j < outDim; j++) {
      yHatRaw[j] = yHatNorm[j] * (outStd[j] + eps) + outMean[j];
    }

    // outlier detection: standardized residual r = (y - yhat)/std
    let isOutlier = false;
    let sampleWeight = 1.0;
    for (let j = 0; j < outDim; j++) {
      const stdj = outStd[j] + eps;
      const r = (yRaw[j] - yHatRaw[j]) / stdj;
      if (Math.abs(r) > this._cfg.outlierThreshold) {
        isOutlier = true;
        break;
      }
    }
    if (isOutlier) sampleWeight = 0.1;

    // ADWIN drift uses lossData (unregularized) as error signal
    const driftDetected = this._adwin.update(lossData);
    if (driftDetected) {
      this._driftCount++;
      // Optional stabilization: reset running avg loss (keeps model weights, resets quality estimate)
      this._avgLoss = 0;
    }

    // backward: dL/dyHat = (yHat - y)/outDim * sampleWeight
    const dYHat = this._dYHat!;
    for (let j = 0; j < outDim; j++) {
      dYHat[j] = (yHatNorm[j] - yNorm[j]) * (sampleWeight / outDim);
    }
    this._backward(seqLen);

    // add L2 regularization to gradients: g += λ * w
    for (let pi = 0; pi < this._params.length; pi++) {
      const par = this._params[pi];
      const w = par.w;
      const g = par.g;
      for (let i = 0; i < w.length; i++) g[i] += lam * w[i];
    }

    // gradient norm
    let g2 = 0;
    for (let pi = 0; pi < this._params.length; pi++) {
      const g = this._params[pi].g;
      for (let i = 0; i < g.length; i++) {
        const v = g[i];
        g2 += v * v;
      }
    }
    const gradientNorm = Math.sqrt(g2);

    // LR schedule: warmup -> cosine decay
    const step = this._adam.t + 1;
    this._effectiveLR = this._computeEffectiveLR(step);
    this._adam.step(this._params, this._effectiveLR);

    // update stats after optimization
    this._sampleCount++;
    // running avg loss for "accuracy" (use total loss including reg, but stable)
    if (this._sampleCount === 1) this._avgLoss = loss;
    else this._avgLoss += (loss - this._avgLoss) / this._sampleCount;

    // convergence test
    this._converged = gradientNorm < this._cfg.convergenceThreshold;

    // update normalizers with current sample (post-fit)
    // inputs: update with all timesteps to capture distribution (as requested)
    this._inNorm!.updateFromMatrix(xRaw, seqLen, inDim);
    this._outNorm!.updateFromVector(yRaw);

    // residual stats (raw residuals)
    const resid = new Float64Array(outDim);
    for (let j = 0; j < outDim; j++) resid[j] = yRaw[j] - yHatRaw[j];
    this._residStats!.updateFromVector(resid);

    return {
      loss,
      gradientNorm,
      effectiveLearningRate: this._effectiveLR,
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  predict(futureSteps: number): PredictionResult {
    const steps = Math.max(0, futureSteps | 0);
    const res: PredictionResult = {
      predictions: [],
      accuracy: this._getAccuracy(),
      sampleCount: this._sampleCount,
      isModelReady: this._isInitialized && this._sampleCount > 1,
    };
    if (!res.isModelReady || steps === 0) return res;

    const seqLen = this._lastSeqLen;
    if (seqLen <= 0) return res;

    // compute stdError from residual stats (raw space)
    const outDim = this._outputDim;
    const eps = this._cfg.epsilon;
    const residStd = this._residStats!.getStd(eps); // alloc small

    // generate multi-step by shifting positional encoding offset
    for (let s = 0; s < steps; s++) {
      this._forward(seqLen, /*posOffset*/ s + 1);

      const yHatNorm = this._yHatNorm!;
      const yHatRaw = this._yHatRaw!;
      const outNorm = this._outNorm!;
      const outMean = outNorm.mean;
      const outStd = outNorm.getStd(eps); // alloc small

      for (let j = 0; j < outDim; j++) {
        yHatRaw[j] = yHatNorm[j] * (outStd[j] + eps) + outMean[j];
      }

      // widen uncertainty ~ sqrt(step+1)
      const widen = Math.sqrt(s + 1);
      const predicted = new Array<number>(outDim);
      const lower = new Array<number>(outDim);
      const upper = new Array<number>(outDim);
      const se = new Array<number>(outDim);

      for (let j = 0; j < outDim; j++) {
        const stdErr = residStd[j] * widen + eps;
        const ci = 1.96 * stdErr;
        const y = yHatRaw[j];
        predicted[j] = y;
        se[j] = stdErr;
        lower[j] = y - ci;
        upper[j] = y + ci;
      }

      res.predictions.push({
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError: se,
      });
    }

    return res;
  }

  getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      numBlocks: this._cfg.numBlocks,
      embeddingDim: this._cfg.embeddingDim,
      numHeads: this._cfg.numHeads,
      temporalScales: this._cfg.temporalScales.slice(),
      totalParameters: this._countParams(),
      sampleCount: this._sampleCount,
      accuracy: this._getAccuracy(),
      converged: this._converged,
      effectiveLearningRate: this._effectiveLR,
      driftCount: this._driftCount,
    };
  }

  getWeights(): WeightInfo {
    if (!this._isInitialized) {
      return {
        temporalConvWeights: [],
        scaleEmbeddings: [],
        positionalEncoding: [],
        fusionWeights: { gateW: [], gateB: [] },
        attentionWeights: [],
        ffnWeights: [],
        layerNormParams: [],
        outputWeights: { wOut: [], bOut: [], wPool: [], bPool: 0 },
        firstMoment: [],
        secondMoment: [],
        updateCount: this._adam.t,
      };
    }

    const embed = this._cfg.embeddingDim;
    const inDim = this._inputDim;
    const outDim = this._outputDim;

    // conv weights
    const convW: number[][][] = new Array(this._convScales.length);
    for (let si = 0; si < this._convScales.length; si++) {
      const conv = this._convScales[si];
      const k = conv.kernel;
      const w = conv.w.w; // [k*inDim*embed]
      const arr: number[][] = new Array(k);
      for (let kk = 0; kk < k; kk++) {
        const row = new Array<number>(inDim * embed);
        const off = kk * inDim * embed;
        for (let i = 0; i < inDim * embed; i++) row[i] = w[off + i];
        arr[kk] = row;
      }
      convW[si] = arr;
    }

    // scale embeddings
    const S = this._cfg.temporalScales.length;
    const scaleEmb = this._scaleEmb!.w;
    const scaleEmbArr: number[][] = new Array(S);
    for (let s = 0; s < S; s++) {
      const row = new Array<number>(embed);
      const off = s * embed;
      for (let j = 0; j < embed; j++) row[j] = scaleEmb[off + j];
      scaleEmbArr[s] = row;
    }

    // fusion
    const gateW = this._fusion!.gateW.w;
    const gateB = this._fusion!.gateB.w;
    const gateW2D = matToNumber2D(gateW, S * embed, S);

    // blocks weights
    const attnWeights: WeightInfo["attentionWeights"] = [];
    const ffnWeights: WeightInfo["ffnWeights"] = [];
    const lnParams: WeightInfo["layerNormParams"] = [];

    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];

      attnWeights.push({
        wq: matToNumber2D(blk.mha.wq.w, embed, embed),
        wk: matToNumber2D(blk.mha.wk.w, embed, embed),
        wv: matToNumber2D(blk.mha.wv.w, embed, embed),
        wo: matToNumber2D(blk.mha.wo.w, embed, embed),
        bq: bufferToNumber1D(blk.mha.bq.w),
        bk: bufferToNumber1D(blk.mha.bk.w),
        bv: bufferToNumber1D(blk.mha.bv.w),
        bo: bufferToNumber1D(blk.mha.bo.w),
      });

      ffnWeights.push({
        w1: matToNumber2D(blk.ffn.w1.w, embed, embed * this._cfg.ffnMultiplier),
        b1: bufferToNumber1D(blk.ffn.b1.w),
        w2: matToNumber2D(blk.ffn.w2.w, embed * this._cfg.ffnMultiplier, embed),
        b2: bufferToNumber1D(blk.ffn.b2.w),
      });

      lnParams.push({
        ln1Gamma: bufferToNumber1D(blk.ln1.gamma.w),
        ln1Beta: bufferToNumber1D(blk.ln1.beta.w),
        ln2Gamma: bufferToNumber1D(blk.ln2.gamma.w),
        ln2Beta: bufferToNumber1D(blk.ln2.beta.w),
      });
    }

    const wOut = matToNumber2D(this._wOut!.w, embed, outDim);
    const bOut = bufferToNumber1D(this._bOut!.w);

    const wPool = bufferToNumber1D(this._pool!.wPool.w);
    const bPool = this._pool!.bPool.w[0];

    // moments (param groups)
    const m3: number[][][] = new Array(this._params.length);
    const v3: number[][][] = new Array(this._params.length);
    for (let i = 0; i < this._params.length; i++) {
      m3[i] = [bufferToNumber1D(this._params[i].m)];
      v3[i] = [bufferToNumber1D(this._params[i].v)];
    }

    return {
      temporalConvWeights: convW,
      scaleEmbeddings: scaleEmbArr,
      positionalEncoding: this._posEnc!.toNumber2D(),
      fusionWeights: { gateW: gateW2D, gateB: bufferToNumber1D(gateB) },
      attentionWeights: attnWeights,
      ffnWeights,
      layerNormParams: lnParams,
      outputWeights: { wOut, bOut, wPool, bPool },
      firstMoment: m3,
      secondMoment: v3,
      updateCount: this._adam.t,
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this._isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }
    const eps = this._cfg.epsilon;
    const inMean = bufferToNumber1D(this._inNorm!.mean);
    const inStd = bufferToNumber1D(this._inNorm!.getStd(eps));
    const outMean = bufferToNumber1D(this._outNorm!.mean);
    const outStd = bufferToNumber1D(this._outNorm!.getStd(eps));
    return {
      inputMean: inMean,
      inputStd: inStd,
      outputMean: outMean,
      outputStd: outStd,
      count: this._inNorm!.count,
    };
  }

  reset(): void {
    this._isInitialized = false;
    this._inputDim = 0;
    this._outputDim = 0;

    this._params = [];
    this._convScales = [];
    this._blocks = [];
    this._scaleSeqs = [];
    this._dScaleSeqs = [];

    this._posEnc = undefined;
    this._scalesI32 = undefined;
    this._scaleEmb = undefined;
    this._fusion = undefined;
    this._pool = undefined;
    this._wOut = undefined;
    this._bOut = undefined;

    this._inNorm = undefined;
    this._outNorm = undefined;
    this._residStats = undefined;

    this._adam.reset();
    this._adwin.reset();
    this._driftCount = 0;

    this._avgLoss = 0;
    this._sampleCount = 0;
    this._converged = false;
    this._effectiveLR = 0;

    this._xRaw = undefined;
    this._xNorm = undefined;
    this._yRaw = undefined;
    this._yNorm = undefined;
    this._yHatNorm = undefined;
    this._yHatRaw = undefined;
    this._dYHat = undefined;

    this._scaleOutLens = undefined;
    this._fused = undefined;
    this._dFused = undefined;
    this._blockBufA = undefined;
    this._blockBufB = undefined;
    this._dBlock = undefined;
    this._pooled = undefined;
    this._dPooled = undefined;
    this._dXNorm = undefined;

    this._lastSeqLen = 0;
  }

  save(): string {
    const obj: any = {
      cfg: this._cfg,
      isInitialized: this._isInitialized,
      inputDim: this._inputDim,
      outputDim: this._outputDim,
      adamT: this._adam.t,
      driftCount: this._driftCount,
      avgLoss: this._avgLoss,
      sampleCount: this._sampleCount,
      converged: this._converged,
      effectiveLR: this._effectiveLR,
      adwin: this._adwin.toJSON(),
      inNorm: this._inNorm ? this._inNorm.toJSON() : null,
      outNorm: this._outNorm ? this._outNorm.toJSON() : null,
      residStats: this._residStats ? this._residStats.toJSON() : null,
      weights: null as any,
    };

    if (this._isInitialized) {
      obj.weights = {
        conv: this._convScales.map((c) => c.toJSON()),
        scaleEmb: bufferToNumber1D(this._scaleEmb!.w),
        scaleEmbM: bufferToNumber1D(this._scaleEmb!.m),
        scaleEmbV: bufferToNumber1D(this._scaleEmb!.v),
        fusion: this._fusion!.toJSON(),
        blocks: this._blocks.map((b) => b.toJSON()),
        pool: this._pool!.toJSON(),
        wOut: bufferToNumber1D(this._wOut!.w),
        bOut: bufferToNumber1D(this._bOut!.w),
        wOutM: bufferToNumber1D(this._wOut!.m),
        wOutV: bufferToNumber1D(this._wOut!.v),
        bOutM: bufferToNumber1D(this._bOut!.m),
        bOutV: bufferToNumber1D(this._bOut!.v),
      };
    }

    return JSON.stringify(obj);
  }

  load(w: string): void {
    const obj = JSON.parse(w);

    this._cfg = obj.cfg ?? this._cfg;

    // reset then re-init if needed
    this.reset();

    this._cfg = obj.cfg ?? this._cfg;

    const isInit = !!obj.isInitialized;
    if (!isInit) return;

    const inDim = obj.inputDim | 0;
    const outDim = obj.outputDim | 0;
    this._initialize(inDim, outDim);

    // restore scalars
    this._driftCount = obj.driftCount ?? 0;
    this._avgLoss = obj.avgLoss ?? 0;
    this._sampleCount = obj.sampleCount ?? 0;
    this._converged = obj.converged ?? false;
    this._effectiveLR = obj.effectiveLR ?? 0;

    // restore optimizer step
    // (AdamOptimizer doesn't expose setter; emulate by stepping t forward)
    // We keep bias correction consistent by setting private _t via repeated increment is too slow.
    // So we reconstruct by making a new optimizer and then "fast set" through (as TS private is runtime-visible).
    (this._adam as any)._t = obj.adamT ?? 0;

    // restore detectors/stats
    this._adwin.fromJSON(obj.adwin ?? {});
    if (obj.inNorm) this._inNorm!.fromJSON(obj.inNorm);
    if (obj.outNorm) this._outNorm!.fromJSON(obj.outNorm);
    if (obj.residStats) this._residStats!.fromJSON(obj.residStats);

    // restore weights
    const W = obj.weights ?? {};
    const convArr = W.conv ?? [];
    for (let i = 0; i < this._convScales.length; i++) {
      this._convScales[i].fromJSON(convArr[i] ?? {});
    }

    const se = W.scaleEmb ?? [];
    const seM = W.scaleEmbM ?? [];
    const seV = W.scaleEmbV ?? [];
    for (let i = 0; i < this._scaleEmb!.w.length; i++) {
      this._scaleEmb!.w[i] = se[i] ?? 0;
    }
    for (let i = 0; i < this._scaleEmb!.m.length; i++) {
      this._scaleEmb!.m[i] = seM[i] ?? 0;
    }
    for (let i = 0; i < this._scaleEmb!.v.length; i++) {
      this._scaleEmb!.v[i] = seV[i] ?? 0;
    }

    this._fusion!.fromJSON(W.fusion ?? {});

    const blocks = W.blocks ?? [];
    for (let i = 0; i < this._blocks.length; i++) {
      this._blocks[i].fromJSON(blocks[i] ?? {});
    }

    this._pool!.fromJSON(W.pool ?? {});

    const wOut = W.wOut ?? [];
    const bOut = W.bOut ?? [];
    const wOutM = W.wOutM ?? [];
    const wOutV = W.wOutV ?? [];
    const bOutM = W.bOutM ?? [];
    const bOutV = W.bOutV ?? [];

    for (let i = 0; i < this._wOut!.w.length; i++) {
      this._wOut!.w[i] = wOut[i] ?? 0;
    }
    for (let i = 0; i < this._bOut!.w.length; i++) {
      this._bOut!.w[i] = bOut[i] ?? 0;
    }
    for (let i = 0; i < this._wOut!.m.length; i++) {
      this._wOut!.m[i] = wOutM[i] ?? 0;
    }
    for (let i = 0; i < this._wOut!.v.length; i++) {
      this._wOut!.v[i] = wOutV[i] ?? 0;
    }
    for (let i = 0; i < this._bOut!.m.length; i++) {
      this._bOut!.m[i] = bOutM[i] ?? 0;
    }
    for (let i = 0; i < this._bOut!.v.length; i++) {
      this._bOut!.v[i] = bOutV[i] ?? 0;
    }
  }

  // ==========================================================================
  // INTERNAL INIT / FORWARD / BACKWARD
  // ==========================================================================

  private _initialize(inDim: number, outDim: number): void {
    this._isInitialized = true;
    this._inputDim = inDim;
    this._outputDim = outDim;

    // reset registries
    this._params = [];

    const embed = this._cfg.embeddingDim;
    const maxSeq = this._cfg.maxSequenceLength;

    // positional encoding
    this._posEnc = new PositionalEncoding(maxSeq, embed);

    // normalization
    this._inNorm = new WelfordVector(inDim);
    this._outNorm = new WelfordVector(outDim);
    this._residStats = new WelfordVector(outDim);

    // buffers
    this._xRaw = new Float64Array(maxSeq * inDim);
    this._xNorm = new Float64Array(maxSeq * inDim);
    this._yRaw = new Float64Array(outDim);
    this._yNorm = new Float64Array(outDim);
    this._yHatNorm = new Float64Array(outDim);
    this._yHatRaw = new Float64Array(outDim);
    this._dYHat = new Float64Array(outDim);

    this._fused = new Float64Array(maxSeq * embed);
    this._dFused = new Float64Array(maxSeq * embed);

    this._blockBufA = new Float64Array(maxSeq * embed);
    this._blockBufB = new Float64Array(maxSeq * embed);
    this._dBlock = new Float64Array(maxSeq * embed);

    this._pooled = new Float64Array(embed);
    this._dPooled = new Float64Array(embed);

    this._dXNorm = new Float64Array(maxSeq * inDim);

    // scales
    const scales = this._cfg.temporalScales;
    const S = scales.length;
    this._scalesI32 = new Int32Array(S);
    for (let i = 0; i < S; i++) this._scalesI32[i] = scales[i] | 0;

    // conv per scale
    this._convScales = new Array(S);
    this._scaleSeqs = new Array(S);
    this._dScaleSeqs = new Array(S);

    for (let i = 0; i < S; i++) {
      const stride = this._scalesI32[i];
      const conv = new Conv1DScale(
        `conv.scale${stride}`,
        stride,
        this._cfg.temporalKernelSize,
        inDim,
        embed,
        maxSeq,
        this._params,
      );
      this._convScales[i] = conv;
      this._scaleSeqs[i] = new Float64Array(maxSeq * embed);
      this._dScaleSeqs[i] = new Float64Array(maxSeq * embed);
    }

    this._scaleOutLens = new Int32Array(S);

    // scale embeddings: [S*embed]
    const scaleEmb = new Float64Array(S * embed);
    fillUniform(scaleEmb, xavierLimit(embed, embed));
    this._scaleEmb = mkParam(`scaleEmb`, scaleEmb, this._params);

    // fusion gate
    this._fusion = new FusionGate(`fusion`, S, embed, maxSeq, this._params);

    // transformer blocks
    this._blocks = new Array(this._cfg.numBlocks);
    for (let i = 0; i < this._cfg.numBlocks; i++) {
      this._blocks[i] = new TransformerBlock(
        `block${i}`,
        maxSeq,
        embed,
        this._cfg.numHeads,
        this._cfg.ffnMultiplier,
        this._cfg.epsilon,
        this._params,
      );
    }

    // pooling
    this._pool = new AttentionPooling(`pool`, maxSeq, embed, this._params);

    // output layer
    const wOut = new Float64Array(embed * outDim);
    const bOut = new Float64Array(outDim);
    fillUniform(wOut, xavierLimit(embed, outDim));
    bOut.fill(0);
    this._wOut = mkParam(`out.w`, wOut, this._params);
    this._bOut = mkParam(`out.b`, bOut, this._params);

    // reset optimizer and detectors
    this._adam = new AdamOptimizer(
      this._cfg.beta1,
      this._cfg.beta2,
      this._cfg.epsilon,
    );
    this._adwin = new ADWINLite(
      this._cfg.adwinDelta,
      Math.max(64, Math.min(1024, maxSeq)),
    );

    this._avgLoss = 0;
    this._sampleCount = 0;
    this._converged = false;
    this._effectiveLR = 0;
    this._lastSeqLen = 0;
  }

  private _zeroAllGrads(): void {
    for (let i = 0; i < this._params.length; i++) this._params[i].g.fill(0);
  }

  private _computeEffectiveLR(step: number): number {
    const base = this._cfg.learningRate;
    const warm = this._cfg.warmupSteps;
    const total = this._cfg.totalSteps;

    const wScale = warm > 0 ? Math.min(1, step / warm) : 1;
    let cScale = 1;
    if (total > warm && step > warm) {
      const t = clamp((step - warm) / (total - warm), 0, 1);
      cScale = 0.5 * (1 + Math.cos(PI * t));
    }
    return base * wScale * cScale;
  }

  private _forward(seqLen: number, posOffset: number): void {
    const inDim = this._inputDim;
    const embed = this._cfg.embeddingDim;
    const S = this._cfg.temporalScales.length;

    const xNorm = this._xNorm!;
    const posEnc = this._posEnc!;
    const scaleEmb = this._scaleEmb!.w;

    // conv per scale → scaleSeqs (E_s = GELU(conv) + PE + scaleEmb)
    for (let s = 0; s < S; s++) {
      const conv = this._convScales[s];
      const outBuf = this._scaleSeqs[s];
      const outLen = conv.forward(xNorm, seqLen, outBuf);
      this._scaleOutLens![s] = outLen;

      // add PE
      posEnc.addInPlace(outBuf, outLen, posOffset);

      // add scale embedding vector
      const embOff = s * embed;
      for (let t = 0; t < outLen; t++) {
        const off = t * embed;
        for (let j = 0; j < embed; j++) outBuf[off + j] += scaleEmb[embOff + j];
      }
    }

    // fusion -> fused (seqLenFine = outLen at scale=1, with SAME conv it matches seqLen)
    const fused = this._fused!;
    this._fusion!.forward(
      this._scaleSeqs,
      this._scaleOutLens!,
      this._scalesI32!,
      seqLen,
      fused,
    );

    // transformer blocks
    // use ping-pong buffers to avoid allocations
    let a = this._blockBufA!;
    let b = this._blockBufB!;
    // copy fused into a
    for (let i = 0; i < seqLen * embed; i++) a[i] = fused[i];

    for (let bi = 0; bi < this._blocks.length; bi++) {
      this._blocks[bi].forward(a, seqLen, b);
      // swap
      const tmp = a;
      a = b;
      b = tmp;
    }

    // pooling
    const pooled = this._pooled!;
    this._pool!.forward(a, seqLen, pooled);

    // output: yHatNorm = pooled * Wout + bout
    const wOut = this._wOut!.w;
    const bOut = this._bOut!.w;
    const yHat = this._yHatNorm!;
    for (let j = 0; j < this._outputDim; j++) {
      let s = bOut[j];
      // wOut row-major [embed*outDim], index = i*outDim + j
      for (let i = 0; i < embed; i++) {
        s += pooled[i] * wOut[i * this._outputDim + j];
      }
      yHat[j] = s;
    }
  }

  private _backward(seqLen: number): void {
    const embed = this._cfg.embeddingDim;
    const outDim = this._outputDim;
    const inDim = this._inputDim;
    const S = this._cfg.temporalScales.length;

    // output backward
    // yHat = pooled*Wout + bout
    const dYHat = this._dYHat!;
    const pooled = this._pooled!;
    const dPooled = this._dPooled!;
    const wOut = this._wOut!;
    const bOut = this._bOut!;

    // grads for output
    wOut.g.fill(0);
    bOut.g.fill(0);
    dPooled.fill(0);

    for (let j = 0; j < outDim; j++) {
      const dy = dYHat[j];
      bOut.g[j] += dy;
      for (let i = 0; i < embed; i++) {
        wOut.g[i * outDim + j] += pooled[i] * dy;
        dPooled[i] += wOut.w[i * outDim + j] * dy;
      }
    }

    // rebuild last transformer output buffer "a" used in forward
    // We re-run forward’s block ping-pong layout minimally by using the stored fused and block forward outputs
    // (For strict no-recompute, each block would cache its input; here we store fused and use block internal caches.
    //  We must reconstruct the final activations exactly as forward produced for correct pooling backward.)
    //
    // To keep allocations low and fully deterministic, we re-run forward through blocks into blockBufA.
    // (This is still fully expanded backward through all layers; we’re just replaying forward activations.)
    //
    // NOTE: This replay is bounded by maxSeqLen and uses only preallocated buffers.
    this._forward(seqLen, 0);

    // Now reconstruct last hidden sequence in blockBufA (same as forward’s "a" after last swap)
    // We do the same ping-pong as forward but without recomputing conv/fusion? forward already did.
    // We replicate last hidden in blockBufA by re-running blocks starting from fused.
    const fused = this._fused!;
    let a = this._blockBufA!;
    let b = this._blockBufB!;
    for (let i = 0; i < seqLen * embed; i++) a[i] = fused[i];
    for (let bi = 0; bi < this._blocks.length; bi++) {
      this._blocks[bi].forward(a, seqLen, b);
      const tmp = a;
      a = b;
      b = tmp;
    }
    // "a" now points to final hidden sequence.

    // pooling backward → dH
    const dH = this._dBlock!;
    this._pool!.backward(a, seqLen, dPooled, dH);

    // transformer blocks backward (reverse), propagate to fused
    // We must propagate through blocks, but each block backward needs its input x.
    // We replay forward inputs per block into a list of buffers without allocating by using a single maxSeq*embed scratch
    // plus a small array of references to preallocated buffers.
    const blockInputs: Float64Array[] = new Array(this._blocks.length);
    const tmpBuf = new Float64Array(this._cfg.maxSequenceLength * embed);

    // build blockInputs by replaying forward: input0=fused, input1=out0, ...
    // We'll store each block input into tmpBuf? That would overwrite. So instead:
    // allocate per-block input buffers is expensive; we use blockBufA/blockBufB ping-pong and copy snapshots into blockInputs buffers
    // by reusing the fusion buffer itself (for block0 input) and allocating two extra "snap" buffers of size maxSeq*embed
    // per block count would be too much.
    //
    // Instead, we recompute block outputs sequentially and in backward we also recompute needed input per block at that stage.
    // This keeps memory minimal at cost of extra compute but still no per-step allocations.

    // Minimal-memory strategy:
    // - For i = numBlocks-1 down to 0:
    //   - Recompute x_i (input to block i) by replaying forward from fused through blocks 0..i-1.
    //   - Then call block[i].backward(x_i, dY_i, dX_i).
    // This is O(numBlocks^2) forward compute but numBlocks is small (default 3) and avoids large allocations.

    const dNext = new Float64Array(this._cfg.maxSequenceLength * embed);
    for (let i = 0; i < seqLen * embed; i++) dNext[i] = dH[i];

    for (let bi = this._blocks.length - 1; bi >= 0; bi--) {
      // recompute x_i into tmpBuf
      // tmpBuf = fused
      for (let i = 0; i < seqLen * embed; i++) tmpBuf[i] = fused[i];
      // forward through blocks 0..bi-1 to get input for block bi
      for (let k = 0; k < bi; k++) {
        this._blocks[k].forward(tmpBuf, seqLen, a);
        // copy a -> tmpBuf
        for (let i = 0; i < seqLen * embed; i++) tmpBuf[i] = a[i];
      }

      // backward block bi: dx into a
      this._blocks[bi].backward(tmpBuf, seqLen, dNext, a);

      // set dNext = a
      for (let i = 0; i < seqLen * embed; i++) dNext[i] = a[i];
    }

    // dNext now is dFused
    const dFused = this._dFused!;
    for (let i = 0; i < seqLen * embed; i++) dFused[i] = dNext[i];

    // fusion backward → dScaleSeqs
    for (let s = 0; s < S; s++) {
      // zero only used portions (outLen varies, but we clear maxSeq*embed to keep correctness)
      this._dScaleSeqs[s].fill(0);
    }
    this._fusion!.backward(
      this._scaleSeqs,
      this._scaleOutLens!,
      this._scalesI32!,
      seqLen,
      dFused,
      this._dScaleSeqs,
    );

    // backprop add(PE + scaleEmb): only affects scaleEmb grads
    // dScaleEmb[s] += sum_t dE_s[t]
    this._scaleEmb!.g.fill(0);
    for (let s = 0; s < S; s++) {
      const dSeq = this._dScaleSeqs[s];
      const outLen = this._scaleOutLens![s];
      const embOff = s * embed;
      for (let t = 0; t < outLen; t++) {
        const off = t * embed;
        for (let j = 0; j < embed; j++) {
          this._scaleEmb!.g[embOff + j] += dSeq[off + j];
        }
      }
    }

    // conv backward accumulate into dXNorm
    const dXNorm = this._dXNorm!;
    dXNorm.fill(0);

    for (let s = 0; s < S; s++) {
      const conv = this._convScales[s];
      const outLen = this._scaleOutLens![s];
      const dSeq = this._dScaleSeqs[s];

      // conv outputs are after GELU; PE/scaleEmb were additive so dConvOut = dSeq
      conv.backward(this._xNorm!, seqLen, outLen, dSeq, dXNorm);
    }

    // (dx to normalized inputs exists in dXNorm, but we do not update raw inputs; this is end of graph)
  }

  private _getAccuracy(): number {
    // accuracy = 1/(1+avgLoss)
    return 1 / (1 + Math.max(0, this._avgLoss));
  }

  private _countParams(): number {
    let n = 0;
    for (let i = 0; i < this._params.length; i++) n += this._params[i].w.length;
    return n;
  }
}
