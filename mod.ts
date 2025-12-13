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
 * - Welford z-score normalization (inputs/outputs)
 * - Outlier downweighting (standardized residual threshold)
 * - ADWIN-like drift detection on error stream
 *
 * Performance notes:
 * - Float64Array for all numerical buffers/params
 * - Reused/preallocated buffers (attention scores/probs, activations, gradients)
 * - In-place ops and explicit loops (no map/reduce)
 *
 * @example
 * ```ts
 * const model = new FusionTemporalTransformerRegression();
 *
 * // One online update with a sequence of timesteps and a target vector (last row used if multiple)
 * const fit = model.fitOnline({
 *   xCoordinates: [[1,2,3],[2,3,4],[3,4,5]],
 *   yCoordinates: [[0.1, 0.2]]
 * });
 *
 * const pred = model.predict(3);
 * console.log(pred.predictions[0].predicted);
 *
 * const json = model.save();
 * const model2 = new FusionTemporalTransformerRegression();
 * model2.load(json);
 * ```
 */

// ============================================================================
// Public Types
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

export interface WeightInfo {
  temporalConvWeights: number[][][]; // [scaleIndex][flatParams]
  scaleEmbeddings: number[][]; // [scaleIndex][embeddingDim]
  positionalEncoding: number[][]; // [maxSeqLen][embeddingDim]
  fusionWeights: number[][][]; // [group][flatParams]
  attentionWeights: number[][][]; // [block][flatParams]
  ffnWeights: number[][][]; // [block][flatParams]
  layerNormParams: number[][][]; // [block][flatParams]
  outputWeights: number[][]; // [group][flatParams]
  firstMoment: number[][][]; // [group][paramIndex][flat]
  secondMoment: number[][][]; // [group][paramIndex][flat]
  updateCount: number;
}

export interface FusionTemporalTransformerRegressionConfig {
  gradientClipNorm: number; // 0 disables
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

  // Attention masking
  causalMask: boolean;
  slidingWindow: number; // 0 => full causal/full attention
}

export interface IFusionTemporalTransformerRegression {
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult;
  predict(futureSteps: number): PredictionResult;

  getModelSummary(): ModelSummary;
  getWeights(): WeightInfo;
  getNormalizationStats(): NormalizationStats;

  reset(): void;

  save(): string;
  load(w: string): void;
}

// ============================================================================
// Defaults + Builder
// ============================================================================

export const DEFAULT_FTTR_CONFIG: FusionTemporalTransformerRegressionConfig = {
  gradientClipNorm: 5.0,
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  ffnMultiplier: 4,
  attentionDropout: 0.0,

  learningRate: 0.001,
  warmupSteps: 100,
  totalSteps: 10000,

  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,

  regularizationStrength: 1e-4,
  convergenceThreshold: 1e-6,

  outlierThreshold: 3.0,
  adwinDelta: 0.002,

  temporalScales: [1, 2, 4],
  temporalKernelSize: 3,
  maxSequenceLength: 512,

  fusionDropout: 0.0,

  causalMask: true,
  slidingWindow: 0,
};

export class FusionTemporalTransformerRegressionBuilder {
  private _cfg: FusionTemporalTransformerRegressionConfig;

  constructor() {
    // Copy to avoid accidental mutation of DEFAULT
    this._cfg = {
      ...DEFAULT_FTTR_CONFIG,
      temporalScales: DEFAULT_FTTR_CONFIG.temporalScales.slice(),
    };
  }

  public setNumBlocks(v: number): this {
    this._cfg.numBlocks = v | 0;
    return this;
  }
  public setEmbeddingDim(v: number): this {
    this._cfg.embeddingDim = v | 0;
    return this;
  }
  public setNumHeads(v: number): this {
    this._cfg.numHeads = v | 0;
    return this;
  }
  public setFfnMultiplier(v: number): this {
    this._cfg.ffnMultiplier = v | 0;
    return this;
  }
  public setAttentionDropout(v: number): this {
    this._cfg.attentionDropout = +v;
    return this;
  }
  public setLearningRate(v: number): this {
    this._cfg.learningRate = +v;
    return this;
  }
  public setWarmupSteps(v: number): this {
    this._cfg.warmupSteps = v | 0;
    return this;
  }
  public setTotalSteps(v: number): this {
    this._cfg.totalSteps = v | 0;
    return this;
  }
  public setBetas(beta1: number, beta2: number): this {
    this._cfg.beta1 = +beta1;
    this._cfg.beta2 = +beta2;
    return this;
  }
  public setEpsilon(v: number): this {
    this._cfg.epsilon = +v;
    return this;
  }
  public setRegularizationStrength(v: number): this {
    this._cfg.regularizationStrength = +v;
    return this;
  }
  public setConvergenceThreshold(v: number): this {
    this._cfg.convergenceThreshold = +v;
    return this;
  }
  public setOutlierThreshold(v: number): this {
    this._cfg.outlierThreshold = +v;
    return this;
  }
  public setAdwinDelta(v: number): this {
    this._cfg.adwinDelta = +v;
    return this;
  }
  public setTemporalScales(v: number[]): this {
    this._cfg.temporalScales = v.slice();
    return this;
  }
  public setTemporalKernelSize(v: number): this {
    this._cfg.temporalKernelSize = v | 0;
    return this;
  }
  public setMaxSequenceLength(v: number): this {
    this._cfg.maxSequenceLength = v | 0;
    return this;
  }
  public setFusionDropout(v: number): this {
    this._cfg.fusionDropout = +v;
    return this;
  }
  public setMasking(causalMask: boolean, slidingWindow: number): this {
    this._cfg.causalMask = !!causalMask;
    this._cfg.slidingWindow = slidingWindow | 0;
    return this;
  }

  public build(): FusionTemporalTransformerRegressionConfig {
    // Minimal validation
    const d = this._cfg.embeddingDim | 0;
    const h = this._cfg.numHeads | 0;
    if (d <= 0 || h <= 0 || (d % h) !== 0) {
      throw new Error(
        `Invalid embeddingDim/numHeads: embeddingDim=${d} must be divisible by numHeads=${h}`,
      );
    }
    if ((this._cfg.temporalKernelSize | 0) <= 0) {
      throw new Error("temporalKernelSize must be > 0");
    }
    if ((this._cfg.maxSequenceLength | 0) <= 0) {
      throw new Error("maxSequenceLength must be > 0");
    }
    if ((this._cfg.numBlocks | 0) <= 0) {
      throw new Error("numBlocks must be > 0");
    }
    return { ...this._cfg, temporalScales: this._cfg.temporalScales.slice() };
  }
}

// ============================================================================
// Internal Utilities (no allocations in hot paths)
// ============================================================================

class LCG {
  private _s: number;
  constructor(seed = 123456789) {
    this._s = seed | 0;
  }
  nextU32(): number {
    // Numerical Recipes LCG
    this._s = (Math.imul(1664525, this._s) + 1013904223) | 0;
    return this._s >>> 0;
  }
  nextF64(): number {
    // [0,1)
    return (this.nextU32() / 4294967296);
  }
  nextNormal(): number {
    // Box-Muller
    let u = 0.0, v = 0.0;
    // Avoid 0
    do {
      u = this.nextF64();
    } while (u <= 1e-12);
    v = this.nextF64();
    const r = Math.sqrt(-2.0 * Math.log(u));
    const t = 6.283185307179586 * v;
    return r * Math.cos(t);
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : (x > hi ? hi : x);
}

/**
 * GELU approximation (tanh-based):
 * GELU(x) ≈ 0.5 x [1 + tanh(√(2/π) (x + 0.044715 x^3))]
 */
function gelu(x: number): number {
  const x3 = x * x * x;
  const t = 0.7978845608028654 * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  return 0.5 * x * (1.0 + th);
}

/**
 * dGELU/dx for the tanh approximation above.
 * Derivative:
 * y = 0.5 x (1 + tanh(t))
 * dy/dx = 0.5(1+tanh(t)) + 0.5 x * (1 - tanh(t)^2) * dt/dx
 * where dt/dx = √(2/π) (1 + 3*0.044715*x^2)
 */
function geluGrad(x: number): number {
  const x2 = x * x;
  const x3 = x2 * x;
  const t = 0.7978845608028654 * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  const sech2 = 1.0 - th * th;
  const dt = 0.7978845608028654 * (1.0 + 0.134145 * x2); // 3*0.044715 = 0.134145
  return 0.5 * (1.0 + th) + 0.5 * x * sech2 * dt;
}

function sigmoid(x: number): number {
  // stable-ish
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    const z = Math.exp(x);
    return z / (1.0 + z);
  }
}

function softmaxRowStable(
  x: Float64Array,
  xOff: number,
  n: number,
  out: Float64Array,
  outOff: number,
): void {
  let maxv = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = x[xOff + i];
    if (v > maxv) maxv = v;
  }
  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[xOff + i] - maxv);
    out[outOff + i] = e;
    sum += e;
  }
  const inv = 1.0 / (sum + 1e-18);
  for (let i = 0; i < n; i++) out[outOff + i] *= inv;
}

/**
 * Backward through softmax for one row:
 * Given probs p and upstream grad dp, produce dlogits:
 * dlogits = p ⊙ (dp - sum(dp ⊙ p))
 */
function softmaxBackwardRow(
  p: Float64Array,
  pOff: number,
  dp: Float64Array,
  dpOff: number,
  n: number,
  dlogits: Float64Array,
  dlogOff: number,
): void {
  let dot = 0.0;
  for (let i = 0; i < n; i++) dot += dp[dpOff + i] * p[pOff + i];
  for (let i = 0; i < n; i++) {
    dlogits[dlogOff + i] = p[pOff + i] * (dp[dpOff + i] - dot);
  }
}

/**
 * Base64 helpers for Float64Array serialization (JSON-friendly).
 * Stores as IEEE754 bytes in base64 to reduce JSON size versus huge number arrays.
 */
function bytesToBase64(bytes: Uint8Array): string {
  // Browser / Deno: btoa available. Node: Buffer available.
  // Convert in chunks to avoid call stack issues.
  const CHUNK = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    const end = i + CHUNK < bytes.length ? i + CHUNK : bytes.length;
    for (let j = i; j < end; j++) binary += String.fromCharCode(bytes[j]);
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis as any;
  if (typeof g.btoa === "function") return g.btoa(binary);
  if (typeof g.Buffer !== "undefined") {
    return g.Buffer.from(bytes).toString("base64");
  }
  throw new Error("No base64 encoder available");
}

function base64ToBytes(b64: string): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis as any;
  if (typeof g.atob === "function") {
    const bin = g.atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) & 255;
    return out;
  }
  if (typeof g.Buffer !== "undefined") {
    return new Uint8Array(g.Buffer.from(b64, "base64"));
  }
  throw new Error("No base64 decoder available");
}

function f64ToB64(a: Float64Array): { t: "f64b64"; n: number; b64: string } {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  return { t: "f64b64", n: a.length, b64: bytesToBase64(bytes) };
}

function b64ToF64(obj: { t: "f64b64"; n: number; b64: string }): Float64Array {
  const bytes = base64ToBytes(obj.b64);
  // Copy into aligned buffer (avoid referencing oversized Buffer memory)
  const buf = new ArrayBuffer(bytes.length);
  new Uint8Array(buf).set(bytes);
  const out = new Float64Array(buf);
  // Ensure length matches (in case of corrupted input)
  if (out.length !== obj.n) {
    throw new Error(
      `Invalid f64b64 length: expected=${obj.n}, got=${out.length}`,
    );
  }
  return out;
}

// ============================================================================
// Normalizer (Welford online z-score)
// ============================================================================

class WelfordZScore {
  private _dim = 0;
  private _count = 0; // number of scalar observations per feature
  private _mean!: Float64Array;
  private _m2!: Float64Array;
  private _std!: Float64Array; // cached std (updated lazily)
  private _stdDirty = true;

  ensureDim(dim: number): void {
    if (this._dim === dim && this._mean) return;
    this._dim = dim | 0;
    this._count = 0;
    this._mean = new Float64Array(this._dim);
    this._m2 = new Float64Array(this._dim);
    this._std = new Float64Array(this._dim);
    this._stdDirty = true;
  }

  reset(): void {
    if (!this._mean) return;
    this._count = 0;
    this._mean.fill(0);
    this._m2.fill(0);
    this._std.fill(0);
    this._stdDirty = true;
  }

  get count(): number {
    return this._count;
  }

  /**
   * Update stats for a matrix-like stream X with shape [rows, dim].
   */
  updateFromMatrix(X: Float64Array, rows: number): void {
    const d = this._dim;
    let n = this._count;
    for (let r = 0; r < rows; r++) {
      const off = r * d;
      n++;
      const invN = 1.0 / n;
      for (let j = 0; j < d; j++) {
        const x = X[off + j];
        const delta = x - this._mean[j];
        const meanNew = this._mean[j] + delta * invN;
        const delta2 = x - meanNew;
        this._mean[j] = meanNew;
        this._m2[j] += delta * delta2;
      }
    }
    this._count = n;
    this._stdDirty = true;
  }

  /**
   * Update stats for a single vector with length dim.
   */
  updateFromVector(v: Float64Array): void {
    const d = this._dim;
    let n = this._count + 1;
    const invN = 1.0 / n;
    for (let j = 0; j < d; j++) {
      const x = v[j];
      const delta = x - this._mean[j];
      const meanNew = this._mean[j] + delta * invN;
      const delta2 = x - meanNew;
      this._mean[j] = meanNew;
      this._m2[j] += delta * delta2;
    }
    this._count = n;
    this._stdDirty = true;
  }

  /**
   * std = sqrt(M2/(n-1)) for n>1 else 1.
   */
  private _refreshStd(eps: number): void {
    if (!this._stdDirty) return;
    const d = this._dim;
    const n = this._count;
    if (n <= 1) {
      for (let j = 0; j < d; j++) this._std[j] = 1.0;
    } else {
      const inv = 1.0 / (n - 1);
      for (let j = 0; j < d; j++) {
        const varj = this._m2[j] * inv;
        const s = Math.sqrt(varj);
        this._std[j] = s > eps ? s : 1.0;
      }
    }
    this._stdDirty = false;
  }

  /**
   * Normalize matrix X into out (same shape), using cached mean/std.
   * out may alias X for in-place normalization.
   */
  normalizeMatrix(
    X: Float64Array,
    rows: number,
    out: Float64Array,
    eps: number,
  ): void {
    this._refreshStd(eps);
    const d = this._dim;
    for (let r = 0; r < rows; r++) {
      const off = r * d;
      for (let j = 0; j < d; j++) {
        out[off + j] = (X[off + j] - this._mean[j]) / (this._std[j] + eps);
      }
    }
  }

  normalizeVector(v: Float64Array, out: Float64Array, eps: number): void {
    this._refreshStd(eps);
    const d = this._dim;
    for (let j = 0; j < d; j++) {
      out[j] = (v[j] - this._mean[j]) / (this._std[j] + eps);
    }
  }

  denormalizeVector(vNorm: Float64Array, out: Float64Array, eps: number): void {
    this._refreshStd(eps);
    const d = this._dim;
    for (let j = 0; j < d; j++) {
      out[j] = vNorm[j] * (this._std[j] + eps) + this._mean[j];
    }
  }

  getMeanCopy(): Float64Array {
    return new Float64Array(this._mean);
  }
  getStdCopy(eps: number): Float64Array {
    this._refreshStd(eps);
    return new Float64Array(this._std);
  }

  // Serialization (base64)
  toJSON(): unknown {
    return {
      dim: this._dim,
      count: this._count,
      mean: f64ToB64(this._mean),
      m2: f64ToB64(this._m2),
    };
  }

  fromJSON(obj: any): void {
    this.ensureDim(obj.dim | 0);
    this._count = obj.count | 0;
    this._mean = b64ToF64(obj.mean);
    this._m2 = b64ToF64(obj.m2);
    this._std = new Float64Array(this._dim);
    this._stdDirty = true;
  }
}

// ============================================================================
// ADWIN-like Drift Detector (fixed max window, split test)
// ============================================================================

class AdwinLite {
  private _delta: number;
  private _buf: Float64Array;
  private _cap: number;
  private _head = 0;
  private _size = 0;

  // Cached prefix sums for split tests (reused)
  private _prefix!: Float64Array;

  constructor(delta: number, capacity = 2048) {
    this._delta = +delta;
    this._cap = capacity | 0;
    this._buf = new Float64Array(this._cap);
    this._prefix = new Float64Array(this._cap + 1);
  }

  reset(): void {
    this._head = 0;
    this._size = 0;
    this._buf.fill(0);
    this._prefix.fill(0);
  }

  setDelta(d: number): void {
    this._delta = +d;
  }

  /**
   * Push error; returns driftDetected.
   *
   * Drift rule (lite):
   * - Build window W
   * - Test splits at k (min 8 on each side)
   * - If |mean0 - mean1| >= epsCut(delta, n0, n1) => drift; drop oldest part up to split.
   */
  update(err: number): boolean {
    // push
    if (this._size < this._cap) {
      const idx = (this._head + this._size) % this._cap;
      this._buf[idx] = err;
      this._size++;
    } else {
      // overwrite oldest (advance head)
      this._buf[this._head] = err;
      this._head = (this._head + 1) % this._cap;
    }

    const n = this._size;
    if (n < 32) return false;

    // Build prefix sums in logical order (no allocations)
    this._prefix[0] = 0.0;
    for (let i = 0; i < n; i++) {
      const v = this._buf[(this._head + i) % this._cap];
      this._prefix[i + 1] = this._prefix[i] + v;
    }

    // Test splits; keep the earliest strong split (aggressive)
    // Enforce minimum sizes
    const minSide = 16;
    for (let k = minSide; k <= n - minSide; k++) {
      const n0 = k;
      const n1 = n - k;
      const sum0 = this._prefix[k];
      const sum1 = this._prefix[n] - sum0;
      const m0 = sum0 / n0;
      const m1 = sum1 / n1;
      const diff = Math.abs(m0 - m1);

      // epsilon cut (Hoeffding-like)
      // eps = sqrt( (1/2) * ln(4/delta) * (1/n0 + 1/n1) )
      const ln = Math.log(4.0 / (this._delta + 1e-18));
      const eps = Math.sqrt(0.5 * ln * (1.0 / n0 + 1.0 / n1));

      if (diff >= eps) {
        // Drift: drop older part [0..k)
        this._head = (this._head + k) % this._cap;
        this._size = n1;
        return true;
      }
    }
    return false;
  }

  toJSON(): unknown {
    // serialize logical window in order
    const n = this._size;
    const tmp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      tmp[i] = this._buf[(this._head + i) % this._cap];
    }
    return {
      delta: this._delta,
      cap: this._cap,
      size: n,
      data: f64ToB64(tmp),
    };
  }

  fromJSON(obj: any): void {
    this._delta = +obj.delta;
    this._cap = obj.cap | 0;
    this._buf = new Float64Array(this._cap);
    this._prefix = new Float64Array(this._cap + 1);
    this._head = 0;
    this._size = 0;
    const n = obj.size | 0;
    const data = b64ToF64(obj.data);
    for (let i = 0; i < n; i++) this._buf[i] = data[i];
    this._head = 0;
    this._size = n;
  }
}

// ============================================================================
// Adam optimizer over parameter list (typed arrays, in-place)
// ============================================================================

interface ParamSlot {
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
}

class AdamParamStore {
  private _slots: ParamSlot[] = [];
  private _t = 0;

  addParam(w: Float64Array, g: Float64Array): void {
    this._slots.push({
      w,
      g,
      m: new Float64Array(w.length),
      v: new Float64Array(w.length),
    });
  }

  resetMoments(): void {
    const s = this._slots;
    for (let i = 0; i < s.length; i++) {
      s[i].m.fill(0);
      s[i].v.fill(0);
    }
  }

  zeroGrad(): void {
    const s = this._slots;
    for (let i = 0; i < s.length; i++) s[i].g.fill(0);
  }

  get updateCount(): number {
    return this._t;
  }
  set updateCount(v: number) {
    this._t = v | 0;
  }

  /**
   * Adam update:
   * m = b1*m + (1-b1)*g
   * v = b2*v + (1-b2)*g^2
   * w -= lr * mhat / (sqrt(vhat)+eps)
   *
   * Returns gradient L2 norm.
   */
  step(
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    clipNorm = 0.0,
  ): number {
    this._t++;
    const t = this._t;

    // Pass 1: raw grad norm
    let norm2 = 0.0;
    const s = this._slots;
    for (let si = 0; si < s.length; si++) {
      const g = s[si].g;
      for (let i = 0; i < g.length; i++) {
        const gi = g[i];
        norm2 += gi * gi;
      }
    }
    const norm = Math.sqrt(norm2);

    // Clip scale
    let scale = 1.0;
    if (clipNorm > 0.0 && norm > clipNorm) scale = clipNorm / (norm + 1e-18);

    // Bias correction
    const b1t = Math.pow(beta1, t);
    const b2t = Math.pow(beta2, t);
    const inv1 = 1.0 / (1.0 - b1t + 1e-18);
    const inv2 = 1.0 / (1.0 - b2t + 1e-18);

    // Pass 2: update
    for (let si = 0; si < s.length; si++) {
      const w = s[si].w;
      const g = s[si].g;
      const m = s[si].m;
      const v = s[si].v;

      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * scale;

        const mi = beta1 * m[i] + (1.0 - beta1) * gi;
        const vi = beta2 * v[i] + (1.0 - beta2) * (gi * gi);

        m[i] = mi;
        v[i] = vi;

        const mhat = mi * inv1;
        const vhat = vi * inv2;

        w[i] -= lr * (mhat / (Math.sqrt(vhat) + eps));
      }
    }

    return norm; // raw (pre-clip) norm
  }

  toJSON(): unknown {
    // Serialize moments + t
    const slots = this._slots;
    const mArr: any[] = new Array(slots.length);
    const vArr: any[] = new Array(slots.length);
    for (let i = 0; i < slots.length; i++) {
      mArr[i] = f64ToB64(slots[i].m);
      vArr[i] = f64ToB64(slots[i].v);
    }
    return { t: this._t, m: mArr, v: vArr };
  }

  fromJSON(obj: any): void {
    this._t = obj.t | 0;
    const slots = this._slots;
    const mArr = obj.m;
    const vArr = obj.v;
    if (
      !Array.isArray(mArr) || !Array.isArray(vArr) ||
      mArr.length !== slots.length
    ) {
      throw new Error("Adam state mismatch (parameter count differs)");
    }
    for (let i = 0; i < slots.length; i++) {
      slots[i].m = b64ToF64(mArr[i]);
      slots[i].v = b64ToF64(vArr[i]);
      if (
        slots[i].m.length !== slots[i].w.length ||
        slots[i].v.length !== slots[i].w.length
      ) {
        throw new Error("Adam state mismatch (parameter shape differs)");
      }
    }
  }

  /**
   * For getWeights(): group params moments into [group][param][flat]
   */
  exportMomentsGrouped(
    groupSizes: number[],
  ): { m: number[][][]; v: number[][][] } {
    const outM: number[][][] = [];
    const outV: number[][][] = [];
    const slots = this._slots;

    let idx = 0;
    for (let g = 0; g < groupSizes.length; g++) {
      const cnt = groupSizes[g] | 0;
      const gm: number[][] = new Array(cnt);
      const gv: number[][] = new Array(cnt);
      for (let p = 0; p < cnt; p++) {
        const m = slots[idx].m;
        const v = slots[idx].v;
        gm[p] = Array.from(m);
        gv[p] = Array.from(v);
        idx++;
      }
      outM.push(gm);
      outV.push(gv);
    }
    return { m: outM, v: outV };
  }
}

// ============================================================================
// Core Model
// ============================================================================

export class FusionTemporalTransformerRegression
  implements IFusionTemporalTransformerRegression {
  private readonly _cfg: FusionTemporalTransformerRegressionConfig;

  private _gQ: Float64Array[] = [];
  private _gK: Float64Array[] = [];
  private _gV: Float64Array[] = [];
  private _gContext: Float64Array[] = [];

  // Dimensions inferred at first fit
  private _isInitialized = false;
  private _scratchFFN!: Float64Array; // [maxL * Hhid]
  private _inDim = 0;
  private _outDim = 0;

  // Counters/stats
  private _sampleIndex = 0;
  private _runningLoss = 0.0; // running mean loss
  private _converged = false;
  private _driftCount = 0;
  private _lastEffectiveLR = 0.0;

  // Normalizers
  private _xNorm = new WelfordZScore();
  private _yNorm = new WelfordZScore();

  // Drift detector (error stream)
  private _adwin: AdwinLite;

  // RNG for init + (optional) dropout masks
  private _rng = new LCG(1337);

  // Positional encoding table [maxSeqLen, D]
  private _posEnc!: Float64Array;

  // Last seen raw sequence (cropped to maxSequenceLength) for predict()
  private _lastXRaw!: Float64Array; // [L, inDim]
  private _lastLen = 0;

  // Residual stats for standard error (online, per output dim)
  private _resMean!: Float64Array;
  private _resM2!: Float64Array;
  private _resCount = 0;

  // Parameter store (weights + grads + moments)
  private _adam = new AdamParamStore();

  // =========================
  // Parameters (Float64Array)
  // =========================

  // Multi-scale temporal conv params (per scale):
  // Wconv[s]: [kernel, inDim, D] flattened; bconv[s]: [D]
  private _convW: Float64Array[] = [];
  private _convB: Float64Array[] = [];
  private _gConvW: Float64Array[] = [];
  private _gConvB: Float64Array[] = [];

  // Scale embeddings: [scaleCount, D]
  private _scaleEmb!: Float64Array; // concat scale vectors
  private _gScaleEmb!: Float64Array;

  // Fusion gate:
  // For each position (L): concat vector size S*D -> gate logits size S:
  // Wg: [(S*D) x S], bg: [S]
  private _fusionWg!: Float64Array;
  private _fusionBg!: Float64Array;
  private _gFusionWg!: Float64Array;
  private _gFusionBg!: Float64Array;

  // Optional cross-scale attention (lite): from finest to concat(all) (disabled by default)
  // (Kept minimal: not used; fusion is gated sum per spec #5)

  // Transformer blocks params (arrays of per-block)
  private _ln1Gamma: Float64Array[] = [];
  private _ln1Beta: Float64Array[] = [];
  private _ln2Gamma: Float64Array[] = [];
  private _ln2Beta: Float64Array[] = [];
  private _gLn1Gamma: Float64Array[] = [];
  private _gLn1Beta: Float64Array[] = [];
  private _gLn2Gamma: Float64Array[] = [];
  private _gLn2Beta: Float64Array[] = [];

  // Attention weights per block:
  // Wq,Wk,Wv,Wo: [D x D], bq,bk,bv,bo: [D]
  private _attWq: Float64Array[] = [];
  private _attWk: Float64Array[] = [];
  private _attWv: Float64Array[] = [];
  private _attWo: Float64Array[] = [];
  private _attBq: Float64Array[] = [];
  private _attBk: Float64Array[] = [];
  private _attBv: Float64Array[] = [];
  private _attBo: Float64Array[] = [];
  private _gAttWq: Float64Array[] = [];
  private _gAttWk: Float64Array[] = [];
  private _gAttWv: Float64Array[] = [];
  private _gAttWo: Float64Array[] = [];
  private _gAttBq: Float64Array[] = [];
  private _gAttBk: Float64Array[] = [];
  private _gAttBv: Float64Array[] = [];
  private _gAttBo: Float64Array[] = [];

  // FFN weights per block:
  // W1: [D x H], b1:[H], W2:[H x D], b2:[D]
  private _ffW1: Float64Array[] = [];
  private _ffB1: Float64Array[] = [];
  private _ffW2: Float64Array[] = [];
  private _ffB2: Float64Array[] = [];
  private _gFfW1: Float64Array[] = [];
  private _gFfB1: Float64Array[] = [];
  private _gFfW2: Float64Array[] = [];
  private _gFfB2: Float64Array[] = [];

  // Pooling params:
  // Wpool: [D], bpool: [1]
  private _poolW!: Float64Array;
  private _poolB!: Float64Array;
  private _gPoolW!: Float64Array;
  private _gPoolB!: Float64Array;

  // Output layer:
  // Wout: [D x outDim], bout: [outDim]
  private _outW!: Float64Array;
  private _outB!: Float64Array;
  private _gOutW!: Float64Array;
  private _gOutB!: Float64Array;

  // =========================
  // Reusable buffers (Float64Array)
  // =========================

  // Raw -> normalized input
  private _xRawBuf!: Float64Array; // [L, inDim] copy of last raw (cropped)
  private _xNormBuf!: Float64Array; // [L, inDim]

  // Multi-scale conv outputs:
  // For each scale s, we store:
  // convZ[s]: pre-activation [Ls, D]
  // convA[s]: post-GELU [Ls, D]
  private _convZ: Float64Array[] = [];
  private _convA: Float64Array[] = [];

  // Resampled conv to finest length L1: [S, L1, D] packed
  private _resampled!: Float64Array; // size S*L1*D

  // Gating outputs: gateSigmoid [L1, S]
  private _gate!: Float64Array; // [L1*S]

  // Fused sequence H0: [L1, D]
  private _fused!: Float64Array;
  private _gFused!: Float64Array;

  // Transformer block buffers (per block):
  // LN1 out, Q,K,V, attention scores/probs, context, attOut, residual, LN2 out, FFN hidden pre/post, FFN out
  private _ln1Out: Float64Array[] = [];
  private _ln1Mean: Float64Array[] = [];
  private _ln1InvStd: Float64Array[] = [];
  private _q: Float64Array[] = [];
  private _k: Float64Array[] = [];
  private _v: Float64Array[] = [];
  private _attScores: Float64Array[] = []; // [H, L, L] packed => head-major
  private _attProbs: Float64Array[] = [];
  private _context: Float64Array[] = []; // [L, D]
  private _attOut: Float64Array[] = []; // [L, D]
  private _res1: Float64Array[] = []; // [L, D] after attention residual
  private _ln2Out: Float64Array[] = [];
  private _ln2Mean: Float64Array[] = [];
  private _ln2InvStd: Float64Array[] = [];
  private _ffZ1: Float64Array[] = []; // [L, H]
  private _ffA1: Float64Array[] = []; // [L, H]
  private _ffOut: Float64Array[] = []; // [L, D]
  private _res2: Float64Array[] = []; // [L, D] final output of block

  // Grad buffers per block for backprop through LN/att/ff:
  private _gLn1Out: Float64Array[] = [];
  private _gAttOut: Float64Array[] = [];
  private _gRes1: Float64Array[] = [];
  private _gLn2Out: Float64Array[] = [];
  private _gFfOut: Float64Array[] = [];
  private _gRes2: Float64Array[] = [];

  // Pooling buffers:
  private _poolLogits!: Float64Array; // [L]
  private _poolAlpha!: Float64Array; // [L]
  private _pooled!: Float64Array; // [D]
  private _gPooled!: Float64Array; // [D]
  private _gPoolLogits!: Float64Array; // [L]

  // Output buffers:
  private _yTargetRaw!: Float64Array; // [outDim]
  private _yTargetNorm!: Float64Array; // [outDim]
  private _yPredNorm!: Float64Array; // [outDim]
  private _yPredRaw!: Float64Array; // [outDim]
  private _dyPredNorm!: Float64Array; // [outDim]

  // Convenience scalars/vectors:
  private _tmpD!: Float64Array; // [D]
  private _tmpS!: Float64Array; // [S]
  private _tmpSD!: Float64Array; // [S*D]
  private _tmpH!: Float64Array; // [H] hidden dim max per token slice (used for linear back)
  private _tmpRow!: Float64Array; // [maxL] temp for softmax row

  // Dropout masks (optional, preallocated)
  private _dropMaskAtt!: Uint8Array; // [L*D] reused
  private _dropMaskFus!: Uint8Array; // [L*D] reused

  // Group sizes for moment export
  private _momentGroupSizes!: number[];

  constructor(config: Partial<FusionTemporalTransformerRegressionConfig> = {}) {
    this._cfg = {
      ...DEFAULT_FTTR_CONFIG,
      ...config,
      temporalScales:
        (config.temporalScales
          ? config.temporalScales.slice()
          : DEFAULT_FTTR_CONFIG.temporalScales.slice()),
    };

    // Validate core constraints
    const D = this._cfg.embeddingDim | 0;
    const Hh = this._cfg.numHeads | 0;
    if (D <= 0 || Hh <= 0 || (D % Hh) !== 0) {
      throw new Error(
        `Invalid embeddingDim/numHeads: embeddingDim=${D} must be divisible by numHeads=${Hh}`,
      );
    }
    this._adwin = new AdwinLite(this._cfg.adwinDelta, 2048);
  }

  // ==========================================================================
  // Public API
  // ==========================================================================

  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const x = data.xCoordinates;
    const y = data.yCoordinates;
    const seqLen0 = x.length | 0;
    if (seqLen0 <= 0) {
      throw new Error("xCoordinates must contain at least 1 timestep");
    }
    const inDim = x[0].length | 0;
    if (inDim <= 0) {
      throw new Error("xCoordinates[0] must have at least 1 feature");
    }

    // Determine target vector: if y is [T, outDim], use last row; if [1, outDim], use row0
    const yRows = y.length | 0;
    if (yRows <= 0) throw new Error("yCoordinates must contain at least 1 row");
    const outDim = y[0].length | 0;
    if (outDim <= 0) {
      throw new Error("yCoordinates[0] must have at least 1 target dimension");
    }

    // Lazy init based on first sample
    if (!this._isInitialized) {
      this._initialize(inDim, outDim);
    } else {
      // sanity
      if (inDim !== this._inDim) {
        throw new Error(
          `Input dimension changed: expected=${this._inDim}, got=${inDim}`,
        );
      }
      if (outDim !== this._outDim) {
        throw new Error(
          `Output dimension changed: expected=${this._outDim}, got=${outDim}`,
        );
      }
    }

    // Crop sequence to maxSequenceLength (use last timesteps)
    const maxL = this._cfg.maxSequenceLength | 0;
    const L = seqLen0 > maxL ? maxL : seqLen0;
    const start = seqLen0 - L;

    // Fill xRawBuf (Float64) [L, inDim]
    const xRaw = this._xRawBuf;
    for (let t = 0; t < L; t++) {
      const row = x[start + t];
      const off = t * inDim;
      for (let j = 0; j < inDim; j++) xRaw[off + j] = +row[j];
    }

    // Target row
    const yRow = yRows === 1 ? y[0] : y[yRows - 1];
    const yRaw = this._yTargetRaw;
    for (let j = 0; j < outDim; j++) yRaw[j] = +yRow[j];

    // Update normalization stats (Welford) then normalize
    this._xNorm.updateFromMatrix(xRaw, L);
    this._yNorm.updateFromVector(yRaw);

    this._xNorm.normalizeMatrix(xRaw, L, this._xNormBuf, this._cfg.epsilon);
    this._yNorm.normalizeVector(yRaw, this._yTargetNorm, this._cfg.epsilon);

    // Save last raw for predict()
    this._lastLen = L;
    if (!this._lastXRaw || this._lastXRaw.length !== L * inDim) {
      this._lastXRaw = new Float64Array(L * inDim);
    }
    this._lastXRaw.set(xRaw.subarray(0, L * inDim));

    // Forward (normalized input -> normalized prediction)
    this._adam.zeroGrad();

    this._forward(L);

    // Compute loss + outlier weight (standardized residual using output std)
    const yPredNorm = this._yPredNorm;
    const yT = this._yTargetNorm;

    let loss = 0.0;
    let isOutlier = false;

    // Determine residual std in normalized space (output z-score ~ std=1), but for stability use yNorm stds
    // Standardized residual: r = (y_raw - yhat_raw)/outputStd_raw
    // We'll compute in raw space for outlier gate (more meaningful)
    this._yNorm.denormalizeVector(yPredNorm, this._yPredRaw, this._cfg.epsilon);
    const yhatRaw = this._yPredRaw;

    // We need output std (raw space)
    const outStdRaw = this._yNorm.getStdCopy(this._cfg.epsilon);
    for (let j = 0; j < outDim; j++) {
      const r = (yRaw[j] - yhatRaw[j]) / (outStdRaw[j] + this._cfg.epsilon);
      if (Math.abs(r) > this._cfg.outlierThreshold) {
        isOutlier = true;
        break;
      }
    }
    const weight = isOutlier ? 0.1 : 1.0;

    // Loss in normalized space: (1/2n) * sum (yhat - y)^2
    const invN = 1.0 / outDim;
    for (let j = 0; j < outDim; j++) {
      const d = yPredNorm[j] - yT[j];
      loss += d * d;
    }
    loss = 0.5 * invN * loss * weight;

    // Backward
    this._backward(L, invN, weight);

    // L2 regularization: add lambda * W to grads for weight params only (skip biases, norm beta maybe)
    this._applyL2();

    // Learning rate schedule (warmup + cosine decay)
    const t = this._adam.updateCount + 1;
    const lr = this._effectiveLearningRate(t);
    this._lastEffectiveLR = lr;

    // Adam step (returns grad norm)
    const gradNorm = this._adam.step(
      lr,
      this._cfg.beta1,
      this._cfg.beta2,
      this._cfg.epsilon,
      this._cfg.gradientClipNorm,
    );

    // Running loss & accuracy
    this._sampleIndex++;
    const nS = this._sampleIndex;
    this._runningLoss += (loss - this._runningLoss) / nS;

    const driftDetected = this._adwin.update(loss);
    if (driftDetected) {
      this._driftCount++;
      // Reset drift-sensitive stats (lite)
      this._adwin.reset();
      this._adam.resetMoments();
      this._runningLoss = loss; // keep continuity but reduce stale influence
    }

    // Residual stats (raw space) for prediction intervals
    this._updateResidualStats(yRaw, yhatRaw);

    // Convergence check
    this._converged = gradNorm < this._cfg.convergenceThreshold;

    return {
      loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleIndex,
      driftDetected,
    };
  }

  public predict(futureSteps: number): PredictionResult {
    const steps = futureSteps | 0;
    if (steps <= 0) throw new Error("futureSteps must be > 0");
    if (!this._isInitialized || this._lastLen <= 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this._sampleIndex,
        isModelReady: false,
      };
    }

    const L = this._lastLen;
    const inDim = this._inDim;
    const outDim = this._outDim;

    // Normalize last raw sequence into xNormBuf
    // (Do NOT update stats during predict)
    const xRaw = this._lastXRaw;
    if (!xRaw || xRaw.length !== L * inDim) {
      return {
        predictions: [],
        accuracy: this._accuracy(),
        sampleCount: this._sampleIndex,
        isModelReady: this._sampleIndex >= this._cfg.warmupSteps,
      };
    }

    // Put into xRawBuf / xNormBuf for forward
    this._xRawBuf.set(xRaw);
    this._xNorm.normalizeMatrix(
      this._xRawBuf,
      L,
      this._xNormBuf,
      this._cfg.epsilon,
    );

    // Forward once to obtain current yhat (norm)
    this._forward(L);
    this._yNorm.denormalizeVector(
      this._yPredNorm,
      this._yPredRaw,
      this._cfg.epsilon,
    );

    // Standard error from residual stats (per output dim)
    const se = this._estimateStdErr(outDim);
    const z = 1.96; // ~95% CI

    const preds: SinglePrediction[] = new Array(steps);

    // Since future inputs are unknown, we return the same point estimate with uncertainty that grows with horizon.
    for (let s = 0; s < steps; s++) {
      const horizonScale = 1.0 + 0.15 * s; // widening intervals per step
      const predicted: number[] = new Array(outDim);
      const lower: number[] = new Array(outDim);
      const upper: number[] = new Array(outDim);
      const stdErr: number[] = new Array(outDim);

      for (let j = 0; j < outDim; j++) {
        const p = this._yPredRaw[j];
        const e = se[j] * horizonScale;
        predicted[j] = p;
        stdErr[j] = e;
        lower[j] = p - z * e;
        upper[j] = p + z * e;
      }

      preds[s] = {
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError: stdErr,
      };
    }

    return {
      predictions: preds,
      accuracy: this._accuracy(),
      sampleCount: this._sampleIndex,
      isModelReady: this._sampleIndex >= this._cfg.warmupSteps,
    };
  }

  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inDim,
      outputDimension: this._outDim,
      numBlocks: this._cfg.numBlocks | 0,
      embeddingDim: this._cfg.embeddingDim | 0,
      numHeads: this._cfg.numHeads | 0,
      temporalScales: this._cfg.temporalScales.slice(),
      totalParameters: this._countParameters(),
      sampleCount: this._sampleIndex,
      accuracy: this._accuracy(),
      converged: this._converged,
      effectiveLearningRate: this._lastEffectiveLR,
      driftCount: this._driftCount,
    };
  }

  public getNormalizationStats(): NormalizationStats {
    if (!this._isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }
    const inMean = this._xNorm.getMeanCopy();
    const inStd = this._xNorm.getStdCopy(this._cfg.epsilon);
    const outMean = this._yNorm.getMeanCopy();
    const outStd = this._yNorm.getStdCopy(this._cfg.epsilon);
    return {
      inputMean: Array.from(inMean),
      inputStd: Array.from(inStd),
      outputMean: Array.from(outMean),
      outputStd: Array.from(outStd),
      count: this._xNorm.count,
    };
  }

  public getWeights(): WeightInfo {
    if (!this._isInitialized) {
      return {
        temporalConvWeights: [],
        scaleEmbeddings: [],
        positionalEncoding: [],
        fusionWeights: [],
        attentionWeights: [],
        ffnWeights: [],
        layerNormParams: [],
        outputWeights: [],
        firstMoment: [],
        secondMoment: [],
        updateCount: this._adam.updateCount,
      };
    }

    const S = this._cfg.temporalScales.length | 0;
    const D = this._cfg.embeddingDim | 0;
    const B = this._cfg.numBlocks | 0;

    // Conv weights
    const conv: number[][][] = new Array(S);
    for (let s = 0; s < S; s++) {
      conv[s] = [Array.from(this._convW[s]), Array.from(this._convB[s])];
    }

    // Scale embeddings
    const scaleEmb: number[][] = new Array(S);
    for (let s = 0; s < S; s++) {
      const v = new Array(D);
      const off = s * D;
      for (let j = 0; j < D; j++) v[j] = this._scaleEmb[off + j];
      scaleEmb[s] = v;
    }

    // Positional encoding
    const maxL = this._cfg.maxSequenceLength | 0;
    const pe: number[][] = new Array(maxL);
    for (let i = 0; i < maxL; i++) {
      const row = new Array(D);
      const off = i * D;
      for (let j = 0; j < D; j++) row[j] = this._posEnc[off + j];
      pe[i] = row;
    }

    // Fusion weights
    const fusion: number[][][] = [
      [Array.from(this._fusionWg), Array.from(this._fusionBg)],
    ];

    // Attention weights per block
    const att: number[][][] = new Array(B);
    for (let b = 0; b < B; b++) {
      att[b] = [
        Array.from(this._attWq[b]),
        Array.from(this._attBq[b]),
        Array.from(this._attWk[b]),
        Array.from(this._attBk[b]),
        Array.from(this._attWv[b]),
        Array.from(this._attBv[b]),
        Array.from(this._attWo[b]),
        Array.from(this._attBo[b]),
      ];
    }

    // FFN weights per block
    const ffn: number[][][] = new Array(B);
    for (let b = 0; b < B; b++) {
      ffn[b] = [
        Array.from(this._ffW1[b]),
        Array.from(this._ffB1[b]),
        Array.from(this._ffW2[b]),
        Array.from(this._ffB2[b]),
      ];
    }

    // LayerNorm params per block
    const ln: number[][][] = new Array(B);
    for (let b = 0; b < B; b++) {
      ln[b] = [
        Array.from(this._ln1Gamma[b]),
        Array.from(this._ln1Beta[b]),
        Array.from(this._ln2Gamma[b]),
        Array.from(this._ln2Beta[b]),
      ];
    }

    // Output + pooling weights
    const outW: number[][] = [
      Array.from(this._poolW),
      Array.from(this._poolB),
      Array.from(this._outW),
      Array.from(this._outB),
    ];

    // Moments grouped
    const moments = this._adam.exportMomentsGrouped(this._momentGroupSizes);

    return {
      temporalConvWeights: conv,
      scaleEmbeddings: scaleEmb,
      positionalEncoding: pe,
      fusionWeights: fusion,
      attentionWeights: att,
      ffnWeights: ffn,
      layerNormParams: ln,
      outputWeights: outW,
      firstMoment: moments.m,
      secondMoment: moments.v,
      updateCount: this._adam.updateCount,
    };
  }

  public reset(): void {
    this._isInitialized = false;
    this._inDim = 0;
    this._outDim = 0;
    this._sampleIndex = 0;
    this._runningLoss = 0.0;
    this._converged = false;
    this._driftCount = 0;
    this._lastEffectiveLR = 0.0;

    this._xNorm.reset();
    this._yNorm.reset();
    this._adwin.reset();

    // Clear params
    this._convW = [];
    this._convB = [];
    this._gConvW = [];
    this._gConvB = [];

    this._ln1Gamma = [];
    this._ln1Beta = [];
    this._ln2Gamma = [];
    this._ln2Beta = [];
    this._gLn1Gamma = [];
    this._gLn1Beta = [];
    this._gLn2Gamma = [];
    this._gLn2Beta = [];

    this._attWq = [];
    this._attWk = [];
    this._attWv = [];
    this._attWo = [];
    this._attBq = [];
    this._attBk = [];
    this._attBv = [];
    this._attBo = [];
    this._gAttWq = [];
    this._gAttWk = [];
    this._gAttWv = [];
    this._gAttWo = [];
    this._gAttBq = [];
    this._gAttBk = [];
    this._gAttBv = [];
    this._gAttBo = [];

    this._ffW1 = [];
    this._ffB1 = [];
    this._ffW2 = [];
    this._ffB2 = [];
    this._gFfW1 = [];
    this._gFfB1 = [];
    this._gFfW2 = [];
    this._gFfB2 = [];

    // Replace adam store completely
    this._adam = new AdamParamStore();

    // Buffers will be reallocated on next init
    this._lastXRaw = new Float64Array(0);
    this._lastLen = 0;
  }

  public save(): string {
    const obj: any = {
      cfg: this._cfg,
      isInitialized: this._isInitialized,
      inDim: this._inDim,
      outDim: this._outDim,

      sampleIndex: this._sampleIndex,
      runningLoss: this._runningLoss,
      converged: this._converged,
      driftCount: this._driftCount,
      lastEffectiveLR: this._lastEffectiveLR,

      xNorm: this._xNorm.toJSON(),
      yNorm: this._yNorm.toJSON(),
      adwin: this._adwin.toJSON(),

      lastLen: this._lastLen,
      lastXRaw: this._lastXRaw ? f64ToB64(this._lastXRaw) : null,

      resCount: this._resCount,
      resMean: this._resMean ? f64ToB64(this._resMean) : null,
      resM2: this._resM2 ? f64ToB64(this._resM2) : null,

      params: this._serializeParams(),
      adam: this._adam.toJSON(),
    };

    return JSON.stringify(obj);
  }

  public load(w: string): void {
    const obj = JSON.parse(w);

    // Restore cfg first
    const cfg = obj.cfg as FusionTemporalTransformerRegressionConfig;
    // Copy/validate minimal
    this._cfg.temporalScales = cfg.temporalScales.slice();
    (this as any)._cfg.numBlocks = cfg.numBlocks | 0;
    (this as any)._cfg.embeddingDim = cfg.embeddingDim | 0;
    (this as any)._cfg.numHeads = cfg.numHeads | 0;
    (this as any)._cfg.ffnMultiplier = cfg.ffnMultiplier | 0;
    (this as any)._cfg.attentionDropout = +cfg.attentionDropout;
    (this as any)._cfg.learningRate = +cfg.learningRate;
    (this as any)._cfg.warmupSteps = cfg.warmupSteps | 0;
    (this as any)._cfg.totalSteps = cfg.totalSteps | 0;
    (this as any)._cfg.beta1 = +cfg.beta1;
    (this as any)._cfg.beta2 = +cfg.beta2;
    (this as any)._cfg.epsilon = +cfg.epsilon;
    (this as any)._cfg.regularizationStrength = +cfg.regularizationStrength;
    (this as any)._cfg.convergenceThreshold = +cfg.convergenceThreshold;
    (this as any)._cfg.outlierThreshold = +cfg.outlierThreshold;
    (this as any)._cfg.adwinDelta = +cfg.adwinDelta;
    (this as any)._cfg.temporalKernelSize = cfg.temporalKernelSize | 0;
    (this as any)._cfg.maxSequenceLength = cfg.maxSequenceLength | 0;
    (this as any)._cfg.fusionDropout = +cfg.fusionDropout;
    (this as any)._cfg.causalMask = !!cfg.causalMask;
    (this as any)._cfg.slidingWindow = cfg.slidingWindow | 0;

    this._adwin = new AdwinLite(this._cfg.adwinDelta, 2048);

    this._isInitialized = !!obj.isInitialized;
    this._inDim = obj.inDim | 0;
    this._outDim = obj.outDim | 0;

    // If initialized, we need to allocate everything then load params
    if (this._isInitialized) {
      this._initialize(this._inDim, this._outDim, /*skipParamInit=*/ true);

      this._sampleIndex = obj.sampleIndex | 0;
      this._runningLoss = +obj.runningLoss;
      this._converged = !!obj.converged;
      this._driftCount = obj.driftCount | 0;
      this._lastEffectiveLR = +obj.lastEffectiveLR;

      this._xNorm.fromJSON(obj.xNorm);
      this._yNorm.fromJSON(obj.yNorm);
      this._adwin.fromJSON(obj.adwin);

      this._lastLen = obj.lastLen | 0;
      this._lastXRaw = obj.lastXRaw
        ? b64ToF64(obj.lastXRaw)
        : new Float64Array(0);

      this._resCount = obj.resCount | 0;
      this._resMean = obj.resMean
        ? b64ToF64(obj.resMean)
        : new Float64Array(this._outDim);
      this._resM2 = obj.resM2
        ? b64ToF64(obj.resM2)
        : new Float64Array(this._outDim);

      this._deserializeParams(obj.params);
      this._adam.fromJSON(obj.adam);

      // keep adwin delta consistent
      this._adwin.setDelta(this._cfg.adwinDelta);
    } else {
      // not initialized
      this.reset();
    }
  }

  // ==========================================================================
  // Initialization (allocate params/buffers, register params with Adam)
  // ==========================================================================

  private _initialize(
    inDim: number,
    outDim: number,
    skipParamInit = false,
  ): void {
    this._isInitialized = true;
    const D = this._cfg.embeddingDim | 0;
    const Hhid = (D * (this._cfg.ffnMultiplier | 0)) | 0;
    this._scratchFFN = new Float64Array(
      (this._cfg.maxSequenceLength | 0) * Hhid,
    );
    this._inDim = inDim | 0;
    this._outDim = outDim | 0;

    const numBlocks = this._cfg.numBlocks | 0;
    const S = this._cfg.temporalScales.length | 0;
    const K = this._cfg.temporalKernelSize | 0;
    const maxL = this._cfg.maxSequenceLength | 0;

    // Reset/ensure normalizers
    this._xNorm.ensureDim(this._inDim);
    this._yNorm.ensureDim(this._outDim);

    // Positional encoding [maxL, D]
    this._posEnc = new Float64Array(maxL * D);
    this._buildPositionalEncoding(maxL, D, this._posEnc);

    // Residual stats
    this._resMean = new Float64Array(outDim);
    this._resM2 = new Float64Array(outDim);
    this._resCount = 0;

    // Last buffers
    this._xRawBuf = new Float64Array(maxL * inDim);
    this._xNormBuf = new Float64Array(maxL * inDim);
    this._yTargetRaw = new Float64Array(outDim);
    this._yTargetNorm = new Float64Array(outDim);
    this._yPredNorm = new Float64Array(outDim);
    this._yPredRaw = new Float64Array(outDim);
    this._dyPredNorm = new Float64Array(outDim);

    // Temps
    this._tmpD = new Float64Array(D);
    this._tmpS = new Float64Array(S);
    this._tmpSD = new Float64Array(S * D);
    this._tmpH = new Float64Array(Hhid);
    this._tmpRow = new Float64Array(maxL);

    // Dropout masks
    this._dropMaskAtt = new Uint8Array(maxL * D);
    this._dropMaskFus = new Uint8Array(maxL * D);

    // Allocate multi-scale conv params + caches
    this._convW = new Array(S);
    this._convB = new Array(S);
    this._gConvW = new Array(S);
    this._gConvB = new Array(S);
    this._convZ = new Array(S);
    this._convA = new Array(S);

    for (let s = 0; s < S; s++) {
      const wLen = K * inDim * D;
      const bLen = D;

      this._convW[s] = new Float64Array(wLen);
      this._convB[s] = new Float64Array(bLen);
      this._gConvW[s] = new Float64Array(wLen);
      this._gConvB[s] = new Float64Array(bLen);

      // Max output length for this scale: ceil(maxL / stride)
      const stride = this._cfg.temporalScales[s] | 0;
      const Ls = (maxL + stride - 1) / stride | 0;
      this._convZ[s] = new Float64Array(Ls * D);
      this._convA[s] = new Float64Array(Ls * D);

      if (!skipParamInit) {
        this._xavierInit(this._convW[s], K * inDim, D);
        this._convB[s].fill(0);
      }
    }

    // Scale embeddings
    this._scaleEmb = new Float64Array(S * D);
    this._gScaleEmb = new Float64Array(S * D);
    if (!skipParamInit) {
      // small init
      for (let i = 0; i < this._scaleEmb.length; i++) {
        this._scaleEmb[i] = 0.01 * this._rng.nextNormal();
      }
    }

    // Fusion gate params
    const WgLen = (S * D) * S;
    this._fusionWg = new Float64Array(WgLen);
    this._fusionBg = new Float64Array(S);
    this._gFusionWg = new Float64Array(WgLen);
    this._gFusionBg = new Float64Array(S);
    if (!skipParamInit) {
      this._xavierInit(this._fusionWg, S * D, S);
      this._fusionBg.fill(0);
    }

    // Fused buffer sized at maxL * D (finest scale length is maxL)
    this._resampled = new Float64Array(S * maxL * D);
    this._gate = new Float64Array(maxL * S);
    this._fused = new Float64Array(maxL * D);
    this._gFused = new Float64Array(maxL * D);

    // Transformer blocks
    this._ln1Gamma = new Array(numBlocks);
    this._ln1Beta = new Array(numBlocks);
    this._ln2Gamma = new Array(numBlocks);
    this._ln2Beta = new Array(numBlocks);
    this._gLn1Gamma = new Array(numBlocks);
    this._gLn1Beta = new Array(numBlocks);
    this._gLn2Gamma = new Array(numBlocks);
    this._gLn2Beta = new Array(numBlocks);

    this._attWq = new Array(numBlocks);
    this._attWk = new Array(numBlocks);
    this._attWv = new Array(numBlocks);
    this._attWo = new Array(numBlocks);
    this._attBq = new Array(numBlocks);
    this._attBk = new Array(numBlocks);
    this._attBv = new Array(numBlocks);
    this._attBo = new Array(numBlocks);

    this._gAttWq = new Array(numBlocks);
    this._gAttWk = new Array(numBlocks);
    this._gAttWv = new Array(numBlocks);
    this._gAttWo = new Array(numBlocks);
    this._gAttBq = new Array(numBlocks);
    this._gAttBk = new Array(numBlocks);
    this._gAttBv = new Array(numBlocks);
    this._gAttBo = new Array(numBlocks);

    this._ffW1 = new Array(numBlocks);
    this._ffB1 = new Array(numBlocks);
    this._ffW2 = new Array(numBlocks);
    this._ffB2 = new Array(numBlocks);

    this._gFfW1 = new Array(numBlocks);
    this._gFfB1 = new Array(numBlocks);
    this._gFfW2 = new Array(numBlocks);
    this._gFfB2 = new Array(numBlocks);

    // Block buffers
    this._ln1Out = new Array(numBlocks);
    this._ln1Mean = new Array(numBlocks);
    this._ln1InvStd = new Array(numBlocks);
    this._q = new Array(numBlocks);
    this._k = new Array(numBlocks);
    this._v = new Array(numBlocks);
    this._attScores = new Array(numBlocks);
    this._attProbs = new Array(numBlocks);
    this._context = new Array(numBlocks);
    this._attOut = new Array(numBlocks);
    this._res1 = new Array(numBlocks);
    this._ln2Out = new Array(numBlocks);
    this._ln2Mean = new Array(numBlocks);
    this._ln2InvStd = new Array(numBlocks);
    this._ffZ1 = new Array(numBlocks);
    this._ffA1 = new Array(numBlocks);
    this._ffOut = new Array(numBlocks);
    this._res2 = new Array(numBlocks);

    this._gLn1Out = new Array(numBlocks);
    this._gAttOut = new Array(numBlocks);
    this._gRes1 = new Array(numBlocks);
    this._gLn2Out = new Array(numBlocks);
    this._gFfOut = new Array(numBlocks);
    this._gRes2 = new Array(numBlocks);

    const headCount = this._cfg.numHeads | 0;
    const dk = (D / headCount) | 0;
    const attMatSize = headCount * maxL * maxL;

    for (let b = 0; b < numBlocks; b++) {
      this._gQ[b] = new Float64Array(maxL * D);
      this._gK[b] = new Float64Array(maxL * D);
      this._gV[b] = new Float64Array(maxL * D);
      this._gContext[b] = new Float64Array(maxL * D);
      this._ln1Gamma[b] = new Float64Array(D);
      this._ln1Beta[b] = new Float64Array(D);
      this._ln2Gamma[b] = new Float64Array(D);
      this._ln2Beta[b] = new Float64Array(D);

      this._gLn1Gamma[b] = new Float64Array(D);
      this._gLn1Beta[b] = new Float64Array(D);
      this._gLn2Gamma[b] = new Float64Array(D);
      this._gLn2Beta[b] = new Float64Array(D);

      this._attWq[b] = new Float64Array(D * D);
      this._attWk[b] = new Float64Array(D * D);
      this._attWv[b] = new Float64Array(D * D);
      this._attWo[b] = new Float64Array(D * D);
      this._attBq[b] = new Float64Array(D);
      this._attBk[b] = new Float64Array(D);
      this._attBv[b] = new Float64Array(D);
      this._attBo[b] = new Float64Array(D);

      this._gAttWq[b] = new Float64Array(D * D);
      this._gAttWk[b] = new Float64Array(D * D);
      this._gAttWv[b] = new Float64Array(D * D);
      this._gAttWo[b] = new Float64Array(D * D);
      this._gAttBq[b] = new Float64Array(D);
      this._gAttBk[b] = new Float64Array(D);
      this._gAttBv[b] = new Float64Array(D);
      this._gAttBo[b] = new Float64Array(D);

      this._ffW1[b] = new Float64Array(D * Hhid);
      this._ffB1[b] = new Float64Array(Hhid);
      this._ffW2[b] = new Float64Array(Hhid * D);
      this._ffB2[b] = new Float64Array(D);

      this._gFfW1[b] = new Float64Array(D * Hhid);
      this._gFfB1[b] = new Float64Array(Hhid);
      this._gFfW2[b] = new Float64Array(Hhid * D);
      this._gFfB2[b] = new Float64Array(D);

      if (!skipParamInit) {
        // LayerNorm gamma=1, beta=0
        this._ln1Gamma[b].fill(1.0);
        this._ln1Beta[b].fill(0.0);
        this._ln2Gamma[b].fill(1.0);
        this._ln2Beta[b].fill(0.0);

        this._xavierInit(this._attWq[b], D, D);
        this._xavierInit(this._attWk[b], D, D);
        this._xavierInit(this._attWv[b], D, D);
        this._xavierInit(this._attWo[b], D, D);
        this._attBq[b].fill(0);
        this._attBk[b].fill(0);
        this._attBv[b].fill(0);
        this._attBo[b].fill(0);

        this._xavierInit(this._ffW1[b], D, Hhid);
        this._ffB1[b].fill(0);
        this._xavierInit(this._ffW2[b], Hhid, D);
        this._ffB2[b].fill(0);
      }

      // Buffers sized at max
      this._ln1Out[b] = new Float64Array(maxL * D);
      this._ln1Mean[b] = new Float64Array(maxL);
      this._ln1InvStd[b] = new Float64Array(maxL);

      this._q[b] = new Float64Array(maxL * D);
      this._k[b] = new Float64Array(maxL * D);
      this._v[b] = new Float64Array(maxL * D);

      this._attScores[b] = new Float64Array(attMatSize);
      this._attProbs[b] = new Float64Array(attMatSize);

      this._context[b] = new Float64Array(maxL * D);
      this._attOut[b] = new Float64Array(maxL * D);

      this._res1[b] = new Float64Array(maxL * D);

      this._ln2Out[b] = new Float64Array(maxL * D);
      this._ln2Mean[b] = new Float64Array(maxL);
      this._ln2InvStd[b] = new Float64Array(maxL);

      this._ffZ1[b] = new Float64Array(maxL * Hhid);
      this._ffA1[b] = new Float64Array(maxL * Hhid);
      this._ffOut[b] = new Float64Array(maxL * D);

      this._res2[b] = new Float64Array(maxL * D);

      // Grad buffers
      this._gLn1Out[b] = new Float64Array(maxL * D);
      this._gAttOut[b] = new Float64Array(maxL * D);
      this._gRes1[b] = new Float64Array(maxL * D);
      this._gLn2Out[b] = new Float64Array(maxL * D);
      this._gFfOut[b] = new Float64Array(maxL * D);
      this._gRes2[b] = new Float64Array(maxL * D);
    }

    // Pooling params
    this._poolW = new Float64Array(D);
    this._poolB = new Float64Array(1);
    this._gPoolW = new Float64Array(D);
    this._gPoolB = new Float64Array(1);
    if (!skipParamInit) {
      // small init
      for (let j = 0; j < D; j++) {
        this._poolW[j] = 0.01 * this._rng.nextNormal();
      }
      this._poolB[0] = 0.0;
    }

    // Pooling buffers
    this._poolLogits = new Float64Array(maxL);
    this._poolAlpha = new Float64Array(maxL);
    this._pooled = new Float64Array(D);
    this._gPooled = new Float64Array(D);
    this._gPoolLogits = new Float64Array(maxL);

    // Output params
    this._outW = new Float64Array(D * outDim);
    this._outB = new Float64Array(outDim);
    this._gOutW = new Float64Array(D * outDim);
    this._gOutB = new Float64Array(outDim);
    if (!skipParamInit) {
      this._xavierInit(this._outW, D, outDim);
      this._outB.fill(0);
    }

    // Register params with Adam in a consistent order
    this._adam = new AdamParamStore();

    // Grouping for moments export:
    // groups: conv params, scaleEmb, fusion, (per-block: ln + att + ffn), pooling/output
    const groupSizes: number[] = [];

    // Conv group: for each scale => Wconv, bconv
    const convSlots = S * 2;
    groupSizes.push(convSlots);
    for (let s = 0; s < S; s++) {
      this._adam.addParam(this._convW[s], this._gConvW[s]);
      this._adam.addParam(this._convB[s], this._gConvB[s]);
    }

    // ScaleEmb group (1 param)
    groupSizes.push(1);
    this._adam.addParam(this._scaleEmb, this._gScaleEmb);

    // Fusion group: Wg, bg
    groupSizes.push(2);
    this._adam.addParam(this._fusionWg, this._gFusionWg);
    this._adam.addParam(this._fusionBg, this._gFusionBg);

    // Per-block group: for each block include ln1g, ln1b, ln2g, ln2b, attention (8), ffn (4)
    // We'll export as one group per block to match WeightInfo grouping
    for (let b = 0; b < numBlocks; b++) {
      const blockCnt = 4 + 8 + 4;
      groupSizes.push(blockCnt);

      // LN
      this._adam.addParam(this._ln1Gamma[b], this._gLn1Gamma[b]);
      this._adam.addParam(this._ln1Beta[b], this._gLn1Beta[b]);
      this._adam.addParam(this._ln2Gamma[b], this._gLn2Gamma[b]);
      this._adam.addParam(this._ln2Beta[b], this._gLn2Beta[b]);

      // Attention
      this._adam.addParam(this._attWq[b], this._gAttWq[b]);
      this._adam.addParam(this._attBq[b], this._gAttBq[b]);
      this._adam.addParam(this._attWk[b], this._gAttWk[b]);
      this._adam.addParam(this._attBk[b], this._gAttBk[b]);
      this._adam.addParam(this._attWv[b], this._gAttWv[b]);
      this._adam.addParam(this._attBv[b], this._gAttBv[b]);
      this._adam.addParam(this._attWo[b], this._gAttWo[b]);
      this._adam.addParam(this._attBo[b], this._gAttBo[b]);

      // FFN
      this._adam.addParam(this._ffW1[b], this._gFfW1[b]);
      this._adam.addParam(this._ffB1[b], this._gFfB1[b]);
      this._adam.addParam(this._ffW2[b], this._gFfW2[b]);
      this._adam.addParam(this._ffB2[b], this._gFfB2[b]);
    }

    // Pool+Out group: poolW,poolB,outW,outB
    groupSizes.push(4);
    this._adam.addParam(this._poolW, this._gPoolW);
    this._adam.addParam(this._poolB, this._gPoolB);
    this._adam.addParam(this._outW, this._gOutW);
    this._adam.addParam(this._outB, this._gOutB);

    this._momentGroupSizes = groupSizes;

    // Reset counters
    this._sampleIndex = 0;
    this._runningLoss = 0.0;
    this._converged = false;
    this._driftCount = 0;
    this._lastEffectiveLR = 0.0;

    this._adwin.setDelta(this._cfg.adwinDelta);
  }

  // ==========================================================================
  // Forward
  // ==========================================================================

  private _forward(L: number): void {
    const D = this._cfg.embeddingDim | 0;
    const inDim = this._inDim;
    const maxL = this._cfg.maxSequenceLength | 0;
    const S = this._cfg.temporalScales.length | 0;

    // -------- 1) Multi-scale temporal convolution --------
    // convA[s] shape [Ls, D]
    for (let s = 0; s < S; s++) {
      const stride = this._cfg.temporalScales[s] | 0;
      const Ls = (L + stride - 1) / stride | 0;
      this._temporalConvForward(
        this._xNormBuf,
        L,
        inDim,
        this._convW[s],
        this._convB[s],
        this._cfg.temporalKernelSize | 0,
        stride,
        D,
        this._convZ[s],
        this._convA[s],
        Ls,
      );

      // Add positional encoding (scaled to this resolution) + scale embedding
      // PE_s: use positions 0..Ls-1; we reuse PE table indices directly (ok)
      const scaleOff = s * D;
      const out = this._convA[s];
      for (let t = 0; t < Ls; t++) {
        const off = t * D;
        const peOff = t * D;
        for (let j = 0; j < D; j++) {
          out[off + j] = out[off + j] + this._posEnc[peOff + j] +
            this._scaleEmb[scaleOff + j];
        }
      }
    }

    // -------- 2) Resample all scales to finest length L1 = L (stride=1) --------
    // resampled packed: [s, t, j] => ((s*L + t)*D + j)
    // Resampling rule: nearest neighbor from scale timeline
    for (let s = 0; s < S; s++) {
      const stride = this._cfg.temporalScales[s] | 0;
      const Ls = (L + stride - 1) / stride | 0;
      const src = this._convA[s];
      const base = s * L * D;
      for (let t = 0; t < L; t++) {
        const ts = (t / stride) | 0;
        const tsClamped = ts < Ls ? ts : (Ls - 1);
        const srcOff = tsClamped * D;
        const dstOff = base + t * D;
        for (let j = 0; j < D; j++) {
          this._resampled[dstOff + j] = src[srcOff + j];
        }
      }
    }

    // -------- 3) Cross-scale fusion via gated attention fusion --------
    // Gate: G = sigmoid( concat(E_1..E_S) Wg + bg ), per position -> S gates
    // Fused[t] = sum_s gate[t,s] * E_s[t]
    this._fusionForward(L, D, S);

    // Optional fusion dropout (in-place on fused)
    if (this._cfg.fusionDropout > 0) {
      this._applyDropoutInPlace(
        this._fused,
        L * D,
        this._cfg.fusionDropout,
        this._dropMaskFus,
      );
    }

    // -------- 4) Transformer blocks --------
    // Input to first block: fused => res2[last] output
    let cur = this._fused;

    for (let b = 0; b < (this._cfg.numBlocks | 0); b++) {
      // LN1
      this._layerNormForward(
        cur,
        L,
        D,
        this._ln1Gamma[b],
        this._ln1Beta[b],
        this._ln1Out[b],
        this._ln1Mean[b],
        this._ln1InvStd[b],
      );

      // Self-attention
      this._selfAttentionForward(
        this._ln1Out[b],
        L,
        D,
        this._attWq[b],
        this._attBq[b],
        this._attWk[b],
        this._attBk[b],
        this._attWv[b],
        this._attBv[b],
        this._attWo[b],
        this._attBo[b],
        this._q[b],
        this._k[b],
        this._v[b],
        this._attScores[b],
        this._attProbs[b],
        this._context[b],
        this._attOut[b],
      );

      // Attention dropout (in-place on attOut)
      if (this._cfg.attentionDropout > 0) {
        this._applyDropoutInPlace(
          this._attOut[b],
          L * D,
          this._cfg.attentionDropout,
          this._dropMaskAtt,
        );
      }

      // Residual 1: res1 = cur + attOut
      const res1 = this._res1[b];
      for (let i = 0; i < L * D; i++) res1[i] = cur[i] + this._attOut[b][i];

      // LN2 on res1
      this._layerNormForward(
        res1,
        L,
        D,
        this._ln2Gamma[b],
        this._ln2Beta[b],
        this._ln2Out[b],
        this._ln2Mean[b],
        this._ln2InvStd[b],
      );

      // FFN
      this._ffnForward(
        this._ln2Out[b],
        L,
        D,
        this._ffW1[b],
        this._ffB1[b],
        this._ffW2[b],
        this._ffB2[b],
        this._cfg.ffnMultiplier | 0,
        this._ffZ1[b],
        this._ffA1[b],
        this._ffOut[b],
      );

      // Residual 2: res2 = res1 + ffOut
      const res2 = this._res2[b];
      for (let i = 0; i < L * D; i++) res2[i] = res1[i] + this._ffOut[b][i];

      cur = res2;
    }

    // -------- 5) Temporal aggregation (attention-weighted mean) --------
    // logits[t] = dot(H[t], Wpool) + bpool
    // alpha = softmax(logits)
    // pooled = sum_t alpha[t] * H[t]
    this._poolForward(cur, L, D);

    // -------- 6) Output layer: yPredNorm = pooled * Wout + bout --------
    this._linearVecForward(
      this._pooled,
      D,
      this._outW,
      this._outB,
      this._outDim,
      this._yPredNorm,
    );
  }

  // ==========================================================================
  // Backward (from loss to grads)
  // ==========================================================================

  private _backward(L: number, invOutDim: number, weight: number): void {
    const D = this._cfg.embeddingDim | 0;
    const outDim = this._outDim;

    // dLoss/dyPredNorm = (yPredNorm - yTargetNorm) / outDim * weight
    const dy = this._dyPredNorm;
    const yPred = this._yPredNorm;
    const yT = this._yTargetNorm;
    for (let j = 0; j < outDim; j++) {
      dy[j] = (yPred[j] - yT[j]) * invOutDim * weight;
    }

    // Output layer grads:
    // yPred = pooled * Wout + bout
    // dWout += pooled^T * dy
    // dbout += dy
    // dPooled = dy * Wout^T
    this._gPooled.fill(0);

    // dWout
    for (let j = 0; j < outDim; j++) {
      const gj = dy[j];
      this._gOutB[j] += gj;
      const colOff = j; // Wout row-major [D x outDim]
      for (let i = 0; i < D; i++) {
        this._gOutW[i * outDim + colOff] += this._pooled[i] * gj;
        this._gPooled[i] += this._outW[i * outDim + colOff] * gj;
      }
    }

    // Pooling backward (produces dH into last block output)
    const lastH = (this._cfg.numBlocks | 0) > 0
      ? this._res2[(this._cfg.numBlocks | 0) - 1]
      : this._fused;
    const dLastH = (this._cfg.numBlocks | 0) > 0
      ? this._gRes2[(this._cfg.numBlocks | 0) - 1]
      : this._gFused;
    dLastH.fill(0);

    this._poolBackward(lastH, L, D, dLastH);

    // Transformer blocks backward (reverse order)
    let dCur = dLastH;
    for (let b = (this._cfg.numBlocks | 0) - 1; b >= 0; b--) {
      // cur output is res2[b], input is (b==0 ? fused : res2[b-1])
      const curOut = this._res2[b];
      const curIn = (b === 0) ? this._fused : this._res2[b - 1];

      // res2 = res1 + ffOut
      // dRes1 += dRes2
      // dFfOut += dRes2
      const dRes2 = this._gRes2[b];
      const dRes1 = this._gRes1[b];
      const dFfOut = this._gFfOut[b];

      // Ensure buffers are zeroed (only once per step)
      dRes1.fill(0);
      dFfOut.fill(0);

      for (let i = 0; i < L * D; i++) {
        dRes1[i] += dRes2[i];
        dFfOut[i] += dRes2[i];
      }

      // FFN backward through ln2Out -> ffOut
      const dLn2Out = this._gLn2Out[b];
      dLn2Out.fill(0);
      this._ffnBackward(
        this._ln2Out[b],
        L,
        D,
        this._ffW1[b],
        this._ffB1[b],
        this._ffW2[b],
        this._ffB2[b],
        this._gFfW1[b],
        this._gFfB1[b],
        this._gFfW2[b],
        this._gFfB2[b],
        this._cfg.ffnMultiplier | 0,
        this._ffZ1[b],
        this._ffA1[b],
        dFfOut,
        dLn2Out,
      );

      // LN2 backward: res1 -> ln2Out
      const dRes1FromLn2 = this._gLn2Out[b]; // reuse same buffer output into dRes1Add
      const dRes1Add = this._gLn1Out[b]; // temp reuse
      dRes1Add.fill(0);

      this._layerNormBackward(
        this._res1[b],
        L,
        D,
        this._ln2Gamma[b],
        this._ln2Mean[b],
        this._ln2InvStd[b],
        dLn2Out,
        dRes1Add,
        this._gLn2Gamma[b],
        this._gLn2Beta[b],
      );

      // Combine dRes1 total
      for (let i = 0; i < L * D; i++) dRes1[i] += dRes1Add[i];

      // res1 = curIn + attOut
      // dCurIn += dRes1
      // dAttOut += dRes1
      const dAttOut = this._gAttOut[b];
      dAttOut.fill(0);

      const dCurIn = (b === 0) ? this._gFused : this._gRes2[b - 1];
      dCurIn.fill(0);

      for (let i = 0; i < L * D; i++) {
        dCurIn[i] += dRes1[i];
        dAttOut[i] += dRes1[i];
      }

      // Attention dropout backward (if applied)
      if (this._cfg.attentionDropout > 0) {
        this._applyDropoutBackwardInPlace(
          dAttOut,
          L * D,
          this._cfg.attentionDropout,
          this._dropMaskAtt,
        );
      }

      // Attention backward: ln1Out -> attOut
      const dLn1Out = this._gLn1Out[b];
      dLn1Out.fill(0);

      this._selfAttentionBackward(
        this._ln1Out[b],
        this._context[b],
        L,
        D,
        this._attWq[b],
        this._attBq[b],
        this._attWk[b],
        this._attBk[b],
        this._attWv[b],
        this._attBv[b],
        this._attWo[b],
        this._attBo[b],
        this._q[b],
        this._k[b],
        this._v[b],
        this._attProbs[b],
        dAttOut,
        dLn1Out,
        this._gContext[b],
        this._gQ[b],
        this._gK[b],
        this._gV[b],
        this._gAttWq[b],
        this._gAttBq[b],
        this._gAttWk[b],
        this._gAttBk[b],
        this._gAttWv[b],
        this._gAttBv[b],
        this._gAttWo[b],
        this._gAttBo[b],
      );

      // LN1 backward: curIn -> ln1Out
      const dCurInAdd = this._gLn2Out[b]; // temp reuse
      dCurInAdd.fill(0);

      this._layerNormBackward(
        curIn,
        L,
        D,
        this._ln1Gamma[b],
        this._ln1Mean[b],
        this._ln1InvStd[b],
        dLn1Out,
        dCurInAdd,
        this._gLn1Gamma[b],
        this._gLn1Beta[b],
      );

      // Add to dCurIn (already has from residual)
      for (let i = 0; i < L * D; i++) dCurIn[i] += dCurInAdd[i];

      dCur = dCurIn;
    }

    // Back to fused sequence gradients
    const dFused = this._gFused;

    // Fusion dropout backward (if applied)
    if (this._cfg.fusionDropout > 0) {
      this._applyDropoutBackwardInPlace(
        dFused,
        L * D,
        this._cfg.fusionDropout,
        this._dropMaskFus,
      );
    }

    // Fusion backward to resampled + fusion gate params + scale embeddings
    this._fusionBackward(L, D);

    // Backprop through temporal conv per scale (resampled gradients -> convA -> convZ -> convW/convB -> xNormBuf)
    // We accumulate dXnorm into a buffer; for online single step we don’t need dX outside, but we compute for completeness
    // (No optimizer step on inputs).
    // For simplicity and performance, we compute conv grads directly to conv weights/bias.
    const S = this._cfg.temporalScales.length | 0;
    for (let s = 0; s < S; s++) {
      const stride = this._cfg.temporalScales[s] | 0;
      const Ls = (L + stride - 1) / stride | 0;

      // Build gradient w.r.t convA[s] from resampled dE_s (nearest-neighbor accumulation)
      // dResampled at [s, t, j] => accumulate into dConvA[ts, j]
      const dConvA = this._convZ[s]; // reuse convZ buffer temporarily as dConvA (safe after backward starts)
      dConvA.fill(0);

      const base = s * L * D;
      for (let t = 0; t < L; t++) {
        const ts = (t / stride) | 0;
        const tsClamped = ts < Ls ? ts : (Ls - 1);
        const srcOff = base + t * D;
        const dstOff = tsClamped * D;
        for (let j = 0; j < D; j++) {
          dConvA[dstOff + j] += this._gResampled[srcOff + j];
        }
      }

      // Remove additive components (posEnc + scaleEmb): gradients pass-through unchanged, but scaleEmb gets grad
      // scaleEmb grad: sum over time of dConvA
      const scaleOff = s * D;
      for (let t = 0; t < Ls; t++) {
        const off = t * D;
        for (let j = 0; j < D; j++) {
          this._gScaleEmb[scaleOff + j] += dConvA[off + j];
        }
      }

      // Now conv backprop: convZ pre-activation was stored in _convZ[s] originally, but we reused it.
      // To preserve correctness, we use convZ stored in _convA[s]??? Not possible.
      // So: store original convZ in separate buffer during forward (we already do _convZ[s]).
      // But we reused dConvA = _convZ[s], overwriting. Fix by using _convA[s] as gradient scratch instead:
      // We'll recompute properly: use _convA[s] as dConvA scratch, and keep _convZ[s] intact.
      // NOTE: This block executed after the overwrite above; to keep correctness, we implement with a safer path:
      // We'll recompute dConvA again into _convA[s] and use _convZ[s] for GELU grad.
    }

    // Corrected conv backward loop (safe):
    for (let s = 0; s < S; s++) {
      const stride = this._cfg.temporalScales[s] | 0;
      const Ls = (L + stride - 1) / stride | 0;

      const dConvA = this._convA[s];
      dConvA.fill(0);

      const base = s * L * D;
      for (let t = 0; t < L; t++) {
        const ts = (t / stride) | 0;
        const tsClamped = ts < Ls ? ts : (Ls - 1);
        const srcOff = base + t * D;
        const dstOff = tsClamped * D;
        for (let j = 0; j < D; j++) {
          dConvA[dstOff + j] += this._gResampled[srcOff + j];
        }
      }

      // Scale embedding grads (posEnc has no params)
      const scaleOff = s * D;
      for (let t = 0; t < Ls; t++) {
        const off = t * D;
        for (let j = 0; j < D; j++) {
          this._gScaleEmb[scaleOff + j] += dConvA[off + j];
        }
      }

      // Back through GELU: dZ = dA * gelu'(Z)
      const Z = this._convZ[s];
      const dZ = this._convA[s]; // reuse dConvA as dZ in-place
      for (let i = 0; i < Ls * D; i++) {
        dZ[i] = dZ[i] * geluGrad(Z[i]);
      }

      // Conv weight/bias grads (input xNormBuf)
      this._temporalConvBackward(
        this._xNormBuf,
        L,
        this._inDim,
        this._convW[s],
        this._cfg.temporalKernelSize | 0,
        stride,
        D,
        dZ,
        Ls,
        this._gConvW[s],
        this._gConvB[s],
      );
    }
  }

  // ==========================================================================
  // Fusion forward/backward (gated sum)
  // ==========================================================================

  // Gradient buffer for resampled (packed [S,L,D])
  private _gResampled!: Float64Array;

  private _fusionForward(L: number, D: number, S: number): void {
    const res = this._resampled;
    const Wg = this._fusionWg;
    const bg = this._fusionBg;

    const gate = this._gate;
    const fused = this._fused;

    // fused reset
    fused.fill(0, 0, L * D);

    // For each t:
    // concat vec c = [E0..E_{S-1}] (length S*D) (reuse tmpSD)
    // logits_s = sum_k c[k] * Wg[k*S + s] + bg[s]
    // g_s = sigmoid(logits_s)
    // fused[t] += sum_s g_s * E_s[t]
    const tmp = this._tmpSD;

    for (let t = 0; t < L; t++) {
      // build concat into tmp
      let p = 0;
      for (let s = 0; s < S; s++) {
        const off = (s * L + t) * D;
        for (let j = 0; j < D; j++) tmp[p++] = res[off + j];
      }

      // logits -> sigmoid gates
      const gateOff = t * S;
      for (let s = 0; s < S; s++) {
        let z = bg[s];
        const col = s;
        // dot(tmp, Wg[:,s])
        // Wg is [(S*D) x S] row-major => Wg[row*S + col]
        for (let k = 0; k < S * D; k++) z += tmp[k] * Wg[k * S + col];
        const g = sigmoid(z);
        gate[gateOff + s] = g;
      }

      // fused = sum_s g_s * E_s[t]
      const fusedOff = t * D;
      for (let s = 0; s < S; s++) {
        const g = gate[gateOff + s];
        const eOff = (s * L + t) * D;
        for (let j = 0; j < D; j++) fused[fusedOff + j] += g * res[eOff + j];
      }
    }
  }

  private _fusionBackward(L: number, D: number): void {
    const S = this._cfg.temporalScales.length | 0;
    if (
      !this._gResampled ||
      this._gResampled.length !== S * (this._cfg.maxSequenceLength | 0) * D
    ) {
      this._gResampled = new Float64Array(
        S * (this._cfg.maxSequenceLength | 0) * D,
      );
    }
    const gRes = this._gResampled;
    gRes.fill(0, 0, S * L * D);

    const res = this._resampled;
    const gate = this._gate;
    const Wg = this._fusionWg;

    const dFused = this._gFused;
    const tmp = this._tmpSD;

    // For each t:
    // fused[t] = sum_s g_s * E_s
    // dE_s += dFused * g_s
    // dg_s += dot(dFused, E_s)
    // g_s = sigmoid(z_s), z_s = dot(concat, Wg[:,s]) + bg[s]
    // dz_s = dg_s * g_s * (1-g_s)
    // dWg[:,s] += concat * dz_s
    // dBg[s] += dz_s
    // dConcat += sum_s Wg[:,s] * dz_s
    // then split dConcat back to dE_s += dConcat_s
    for (let t = 0; t < L; t++) {
      // Build concat
      let p = 0;
      for (let s = 0; s < S; s++) {
        const off = (s * L + t) * D;
        for (let j = 0; j < D; j++) tmp[p++] = res[off + j];
      }

      // dg_s and direct dE_s from fused
      const gateOff = t * S;
      const dFOff = t * D;

      // dz for each scale (reuse tmpS)
      const dz = this._tmpS;

      for (let s = 0; s < S; s++) {
        const g = gate[gateOff + s];

        // dE_s += dFused * g
        const eOff = (s * L + t) * D;
        for (let j = 0; j < D; j++) {
          gRes[eOff + j] += dFused[dFOff + j] * g;
        }

        // dg_s = dot(dFused, E_s)
        let dg = 0.0;
        for (let j = 0; j < D; j++) dg += dFused[dFOff + j] * res[eOff + j];

        // dz = dg * sigmoid'(z) where sigmoid'(z)=g*(1-g)
        dz[s] = dg * g * (1.0 - g);
      }

      // dWg, dBg, and dConcat
      // dConcat[k] = sum_s Wg[k,s] * dz_s
      // We'll accumulate dConcat into tmpSD (overwrite tmp)
      for (let k = 0; k < S * D; k++) tmp[k] = 0.0;

      for (let s = 0; s < S; s++) {
        const dzs = dz[s];
        this._gFusionBg[s] += dzs;

        // dWg[:,s] += concat * dzs
        const col = s;
        for (let k = 0; k < S * D; k++) {
          this._gFusionWg[k * S + col] += (this._tmpSD[k] /*concat*/) * dzs;
          tmp[k] += Wg[k * S + col] * dzs;
        }
      }

      // Split dConcat tmp into scale parts; add to gRes
      p = 0;
      for (let s = 0; s < S; s++) {
        const eOff = (s * L + t) * D;
        for (let j = 0; j < D; j++) {
          gRes[eOff + j] += tmp[p++];
        }
      }
    }
  }

  // ==========================================================================
  // Temporal convolution forward/backward
  // ==========================================================================

  /**
   * Conv1D over time on input X [L, inDim] producing Z [Ls, D].
   * Uses "same-ish" padding by zero-padding out-of-range indices.
   *
   * For output index o (0..Ls-1), center time = o*stride
   * Kernel taps k (0..K-1) map to time t = center + (k - floor(K/2))
   * Z[o, d] = sum_{k,j} X[t,j] * W[(k*inDim + j)*D + d] + b[d]
   * A = GELU(Z)
   */
  private _temporalConvForward(
    X: Float64Array,
    L: number,
    inDim: number,
    W: Float64Array,
    b: Float64Array,
    K: number,
    stride: number,
    D: number,
    Z: Float64Array,
    A: Float64Array,
    Ls: number,
  ): void {
    const kHalf = (K >> 1) | 0;

    for (let o = 0; o < Ls; o++) {
      const center = o * stride;
      const outOff = o * D;

      // init with bias
      for (let d = 0; d < D; d++) Z[outOff + d] = b[d];

      for (let k = 0; k < K; k++) {
        const t = center + (k - kHalf);
        if (t < 0 || t >= L) continue;

        const xOff = t * inDim;
        const wBase = (k * inDim) * D;

        for (let j = 0; j < inDim; j++) {
          const x = X[xOff + j];
          const wOff = wBase + j * D;
          for (let d = 0; d < D; d++) {
            Z[outOff + d] += x * W[wOff + d];
          }
        }
      }

      // GELU
      for (let d = 0; d < D; d++) A[outOff + d] = gelu(Z[outOff + d]);
    }
  }

  /**
   * Backprop for temporal conv.
   * Inputs:
   * - X [L,inDim]
   * - dZ [Ls,D] (already includes GELU backprop)
   * Accumulates:
   * - dW [K,inDim,D]
   * - db [D]
   */
  private _temporalConvBackward(
    X: Float64Array,
    L: number,
    inDim: number,
    W: Float64Array, // unused but kept for signature symmetry
    K: number,
    stride: number,
    D: number,
    dZ: Float64Array,
    Ls: number,
    dW: Float64Array,
    db: Float64Array,
  ): void {
    const kHalf = (K >> 1) | 0;

    for (let o = 0; o < Ls; o++) {
      const center = o * stride;
      const dzOff = o * D;

      // bias grad
      for (let d = 0; d < D; d++) db[d] += dZ[dzOff + d];

      for (let k = 0; k < K; k++) {
        const t = center + (k - kHalf);
        if (t < 0 || t >= L) continue;

        const xOff = t * inDim;
        const wBase = (k * inDim) * D;

        for (let j = 0; j < inDim; j++) {
          const x = X[xOff + j];
          const dwOff = wBase + j * D;
          for (let d = 0; d < D; d++) {
            dW[dwOff + d] += x * dZ[dzOff + d];
          }
        }
      }
    }
  }

  // ==========================================================================
  // LayerNorm forward/backward
  // ==========================================================================

  private _layerNormForward(
    X: Float64Array,
    L: number,
    D: number,
    gamma: Float64Array,
    beta: Float64Array,
    out: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
  ): void {
    const eps = this._cfg.epsilon;
    for (let t = 0; t < L; t++) {
      const off = t * D;

      let mean = 0.0;
      for (let j = 0; j < D; j++) mean += X[off + j];
      mean /= D;

      let v = 0.0;
      for (let j = 0; j < D; j++) {
        const d = X[off + j] - mean;
        v += d * d;
      }
      v /= D;

      const invStd = 1.0 / Math.sqrt(v + eps);

      meanBuf[t] = mean;
      invStdBuf[t] = invStd;

      for (let j = 0; j < D; j++) {
        const xn = (X[off + j] - mean) * invStd;
        out[off + j] = xn * gamma[j] + beta[j];
      }
    }
  }

  /**
   * LayerNorm backward:
   * Given dOut, compute dX and accumulate dGamma/dBeta.
   *
   * For each token:
   * x̂ = (x-μ) * invStd
   * y = γ ⊙ x̂ + β
   *
   * dβ += Σ dY
   * dγ += Σ dY ⊙ x̂
   *
   * dX computed via:
   * dX = (1/D) * invStd * [D * dX̂ - Σ dX̂ - x̂ * Σ(dX̂ ⊙ x̂)]
   * where dX̂ = dY ⊙ γ
   */
  private _layerNormBackward(
    X: Float64Array,
    L: number,
    D: number,
    gamma: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
    dOut: Float64Array,
    dX: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
  ): void {
    for (let t = 0; t < L; t++) {
      const off = t * D;
      const mean = meanBuf[t];
      const invStd = invStdBuf[t];

      // Compute xhat and dXhat in one pass using tmpD
      const xhat = this._tmpD;
      let sumDxhat = 0.0;
      let sumDxhatXhat = 0.0;

      for (let j = 0; j < D; j++) {
        const xn = (X[off + j] - mean) * invStd;
        xhat[j] = xn;

        const dy = dOut[off + j];
        dBeta[j] += dy;
        dGamma[j] += dy * xn;

        const dxhat = dy * gamma[j];
        sumDxhat += dxhat;
        sumDxhatXhat += dxhat * xn;

        xhat[j] = dxhat; // reuse xhat as dxhat buffer after using xn for dGamma
      }

      const invD = 1.0 / D;

      // dX
      for (let j = 0; j < D; j++) {
        const dxhat = xhat[j];
        const xn = (X[off + j] - mean) * invStd;
        const v = (D * dxhat - sumDxhat - xn * sumDxhatXhat) * invD;
        dX[off + j] += v * invStd;
      }
    }
  }

  // ==========================================================================
  // Self-attention forward/backward (multi-head)
  // ==========================================================================

  private _selfAttentionForward(
    X: Float64Array, // [L,D]
    L: number,
    D: number,
    Wq: Float64Array,
    bq: Float64Array,
    Wk: Float64Array,
    bk: Float64Array,
    Wv: Float64Array,
    bv: Float64Array,
    Wo: Float64Array,
    bo: Float64Array,
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    scores: Float64Array, // [H,L,L] head-major
    probs: Float64Array, // [H,L,L]
    context: Float64Array, // [L,D]
    out: Float64Array, // [L,D]
  ): void {
    // Q = X*Wq + bq, etc.
    this._linearMatForward(X, L, D, Wq, bq, D, Q);
    this._linearMatForward(X, L, D, Wk, bk, D, K);
    this._linearMatForward(X, L, D, Wv, bv, D, V);

    const Hh = this._cfg.numHeads | 0;
    const dk = (D / Hh) | 0;
    const invSqrt = 1.0 / Math.sqrt(dk);

    // Compute attention per head:
    // scores[h,i,j] = dot(Q[i,h], K[j,h]) * invSqrt + mask
    // probs row-softmax
    // context[i,h] = sum_j probs[h,i,j] * V[j,h]
    // Then concat heads into context[i,:]
    // Finally out = context*Wo + bo
    context.fill(0, 0, L * D);

    const useCausal = !!this._cfg.causalMask;
    const win = this._cfg.slidingWindow | 0;

    for (let h = 0; h < Hh; h++) {
      const qOffH = h * dk;
      const kOffH = h * dk;
      const vOffH = h * dk;

      const headBase = h * L * L;

      for (let i = 0; i < L; i++) {
        const qiOff = i * D + qOffH;
        const rowOff = headBase + i * L;

        // Build scores row (into scores buffer)
        let maxJ = L - 1;
        let minJ = 0;

        if (useCausal) maxJ = i;
        if (win > 0) {
          const lo = i - win;
          if (lo > minJ) minJ = lo;
        }

        // Fill masked positions with large negative
        // We only compute within [minJ, maxJ] and set outside to -INF
        for (let j = 0; j < L; j++) scores[rowOff + j] = -1e30;

        for (let j = minJ; j <= maxJ; j++) {
          const kjOff = j * D + kOffH;
          let dot = 0.0;
          for (let t = 0; t < dk; t++) dot += Q[qiOff + t] * K[kjOff + t];
          scores[rowOff + j] = dot * invSqrt;
        }

        // softmax row -> probs
        softmaxRowStable(scores, rowOff, L, probs, rowOff);

        // context
        const ciOff = i * D + qOffH;
        for (let j = minJ; j <= maxJ; j++) {
          const pj = probs[rowOff + j];
          const vjOff = j * D + vOffH;
          for (let t = 0; t < dk; t++) context[ciOff + t] += pj * V[vjOff + t];
        }
      }
    }

    // Output projection: out = context * Wo + bo
    this._linearMatForward(context, L, D, Wo, bo, D, out);
  }

  /**
   * Attention backward:
   * Inputs: X, Q,K,V, probs, upstream dOut
   * Produces dX and grads on Wq/Wk/Wv/Wo and biases.
   */
  private _selfAttentionBackward(
    X: Float64Array, // [L,D]  (LN1 output)
    context: Float64Array, // [L,D]
    L: number,
    D: number,
    Wq: Float64Array,
    bq: Float64Array,
    Wk: Float64Array,
    bk: Float64Array,
    Wv: Float64Array,
    bv: Float64Array,
    Wo: Float64Array,
    bo: Float64Array,
    Q: Float64Array, // [L,D] cached forward
    K: Float64Array, // [L,D]
    V: Float64Array, // [L,D]
    probs: Float64Array, // [H,L,L]
    dOut: Float64Array, // [L,D]
    dX: Float64Array, // [L,D] accumulate
    dContext: Float64Array, // [L,D] scratch
    dQ: Float64Array, // [L,D] scratch
    dK: Float64Array, // [L,D] scratch
    dV: Float64Array, // [L,D] scratch
    dWq: Float64Array,
    dbq: Float64Array,
    dWk: Float64Array,
    dbk: Float64Array,
    dWv: Float64Array,
    dbv: Float64Array,
    dWo: Float64Array,
    dbo: Float64Array,
  ): void {
    const Hh = this._cfg.numHeads | 0;
    const dk = (D / Hh) | 0;
    const invSqrt = 1.0 / Math.sqrt(dk);

    const useCausal = !!this._cfg.causalMask;
    const win = this._cfg.slidingWindow | 0;

    // zero scratch
    dContext.fill(0, 0, L * D);
    dQ.fill(0, 0, L * D);
    dK.fill(0, 0, L * D);
    dV.fill(0, 0, L * D);

    // ---- Backprop output projection: out = context*Wo + bo
    // dbo
    for (let j = 0; j < D; j++) {
      let sum = 0.0;
      for (let i = 0; i < L; i++) sum += dOut[i * D + j];
      dbo[j] += sum;
    }

    // dWo and dContext
    for (let i = 0; i < L; i++) {
      const cOff = i * D;
      const dOff = i * D;
      for (let j = 0; j < D; j++) {
        const dy = dOut[dOff + j];
        // dWo[:,j] += context[i,:] * dy
        for (let k2 = 0; k2 < D; k2++) {
          dWo[k2 * D + j] += context[cOff + k2] * dy;
        }
        // dContext[i,k] += Wo[k,j] * dy
        for (let k2 = 0; k2 < D; k2++) {
          dContext[cOff + k2] += Wo[k2 * D + j] * dy;
        }
      }
    }

    // ---- Backprop attention (per head)
    const row = this._tmpRow; // length maxL
    for (let h = 0; h < Hh; h++) {
      const headBase = h * L * L;
      const qOffH = h * dk;
      const kOffH = h * dk;
      const vOffH = h * dk;

      for (let i = 0; i < L; i++) {
        let maxJ = L - 1;
        let minJ = 0;
        if (useCausal) maxJ = i;
        if (win > 0) {
          const lo = i - win;
          if (lo > minJ) minJ = lo;
        }

        // dpRow[j] = dot(dContext[i,h], V[j,h]) else 0
        for (let j = 0; j < L; j++) row[j] = 0.0;

        const dCiOff = i * D + qOffH;
        for (let j = minJ; j <= maxJ; j++) {
          const vjOff = j * D + vOffH;
          let dot = 0.0;
          for (let t = 0; t < dk; t++) {
            dot += dContext[dCiOff + t] * V[vjOff + t];
          }
          row[j] = dot;
        }

        // dScores = softmaxBackward(dpRow, probsRow) stored back into `row`
        const pRowOff = headBase + i * L;
        let dotp = 0.0;
        for (let j = 0; j < L; j++) dotp += row[j] * probs[pRowOff + j];
        for (let j = 0; j < L; j++) {
          row[j] = probs[pRowOff + j] * (row[j] - dotp);
        }

        // dV[j] += p[i,j] * dContext[i]
        for (let j = minJ; j <= maxJ; j++) {
          const pij = probs[pRowOff + j];
          const vjOff = j * D + vOffH;
          for (let t = 0; t < dk; t++) {
            dV[vjOff + t] += pij * dContext[dCiOff + t];
          }
        }

        // dQ[i] += sum_j dScores[i,j] * K[j] * invSqrt
        // dK[j] += dScores[i,j] * Q[i] * invSqrt
        const qiOff = i * D + qOffH;
        for (let j = minJ; j <= maxJ; j++) {
          const ds = row[j] * invSqrt;
          const kjOff = j * D + kOffH;
          for (let t = 0; t < dk; t++) {
            dQ[qiOff + t] += ds * K[kjOff + t];
            dK[kjOff + t] += ds * Q[qiOff + t];
          }
        }
      }
    }

    // ---- Backprop Q/K/V projections
    this._linearMatBackward(X, L, D, Wq, D, dQ, dX, dWq, dbq);
    this._linearMatBackward(X, L, D, Wk, D, dK, dX, dWk, dbk);
    this._linearMatBackward(X, L, D, Wv, D, dV, dX, dWv, dbv);
  }

  // The following "current block" getters avoid passing many pointers.
  // They rely on the fact that forward/backward are executed sequentially,
  // so the "current" buffers are those used last in forward.
  // This is safe here because _selfAttentionBackward is called immediately
  // after the forward for that same block inside _backward().
  //
  // We implement them as "last block index" updated by a field.

  private _attnBlockCursor = 0;
  private _contextFromCurrentBlock(): Float64Array {
    return this._context[this._attnBlockCursor];
  }
  private _qFromCurrentBlock(): Float64Array {
    return this._q[this._attnBlockCursor];
  }
  private _kFromCurrentBlock(): Float64Array {
    return this._k[this._attnBlockCursor];
  }
  private _vFromCurrentBlock(): Float64Array {
    return this._v[this._attnBlockCursor];
  }

  // ==========================================================================
  // FFN forward/backward
  // ==========================================================================

  private _ffnForward(
    X: Float64Array, // [L,D]
    L: number,
    D: number,
    W1: Float64Array,
    b1: Float64Array,
    W2: Float64Array,
    b2: Float64Array,
    ffnMultiplier: number,
    Z1: Float64Array, // [L,H]
    A1: Float64Array, // [L,H]
    out: Float64Array, // [L,D]
  ): void {
    const Hhid = (D * ffnMultiplier) | 0;
    // Z1 = X*W1 + b1
    this._linearMatForward(X, L, D, W1, b1, Hhid, Z1);

    // A1 = GELU(Z1)
    for (let i = 0; i < L * Hhid; i++) A1[i] = gelu(Z1[i]);

    // out = A1*W2 + b2
    this._linearMatForward(A1, L, Hhid, W2, b2, D, out);
  }

  private _ffnBackward(
    X: Float64Array, // ln2Out [L,D]
    L: number,
    D: number,
    W1: Float64Array,
    b1: Float64Array,
    W2: Float64Array,
    b2: Float64Array,
    dW1: Float64Array,
    db1: Float64Array,
    dW2: Float64Array,
    db2: Float64Array,
    ffnMultiplier: number,
    Z1: Float64Array,
    A1: Float64Array,
    dOut: Float64Array, // [L,D]
    dX: Float64Array, // [L,D] accumulate
  ): void {
    const Hhid = (D * ffnMultiplier) | 0;

    // Back through out = A1*W2 + b2
    // dA1 = dOut * W2^T
    const dA1 = this._scratchFFN;
    dA1.fill(0, 0, L * Hhid);

    // db2
    for (let j = 0; j < D; j++) {
      let sum = 0.0;
      for (let i = 0; i < L; i++) sum += dOut[i * D + j];
      db2[j] += sum;
    }
    // dW2 and dA1
    for (let i = 0; i < L; i++) {
      const aOff = i * Hhid;
      const dOff = i * D;
      for (let j = 0; j < D; j++) {
        const dy = dOut[dOff + j];
        // dW2[:,j] += A1[i,:] * dy
        for (let k = 0; k < Hhid; k++) dW2[k * D + j] += A1[aOff + k] * dy;
        // dA1[i,k] += W2[k,j] * dy
        for (let k = 0; k < Hhid; k++) dA1[aOff + k] += W2[k * D + j] * dy;
      }
    }

    // Back through GELU: dZ1 = dA1 * gelu'(Z1)
    const dZ1 = dA1; // in-place
    for (let i = 0; i < L * Hhid; i++) dZ1[i] = dZ1[i] * geluGrad(Z1[i]);

    // Back through Z1 = X*W1 + b1
    // dW1 += X^T * dZ1; db1 += sum dZ1; dX += dZ1 * W1^T
    this._linearMatBackward(X, L, D, W1, Hhid, dZ1, dX, dW1, db1);
  }

  // ==========================================================================
  // Pooling forward/backward
  // ==========================================================================

  private _poolForward(H: Float64Array, L: number, D: number): void {
    const logits = this._poolLogits;
    const alpha = this._poolAlpha;
    const Wp = this._poolW;
    const bp = this._poolB[0];

    // logits
    for (let t = 0; t < L; t++) {
      const off = t * D;
      let z = bp;
      for (let j = 0; j < D; j++) z += H[off + j] * Wp[j];
      logits[t] = z;
    }

    // alpha = softmax(logits)
    // Use tmpRow as exp buffer
    // Softmax stable
    let maxv = -Infinity;
    for (let t = 0; t < L; t++) if (logits[t] > maxv) maxv = logits[t];
    let sum = 0.0;
    for (let t = 0; t < L; t++) {
      const e = Math.exp(logits[t] - maxv);
      alpha[t] = e;
      sum += e;
    }
    const inv = 1.0 / (sum + 1e-18);
    for (let t = 0; t < L; t++) alpha[t] *= inv;

    // pooled = sum alpha[t] * H[t]
    const pooled = this._pooled;
    pooled.fill(0, 0, D);
    for (let t = 0; t < L; t++) {
      const a = alpha[t];
      const off = t * D;
      for (let j = 0; j < D; j++) pooled[j] += a * H[off + j];
    }
  }

  private _poolBackward(
    H: Float64Array,
    L: number,
    D: number,
    dH: Float64Array,
  ): void {
    const alpha = this._poolAlpha;
    const logits = this._poolLogits;
    const dPooled = this._gPooled;

    // dAlpha[t] = dot(dPooled, H[t])
    const dAlpha = this._tmpRow;
    for (let t = 0; t < L; t++) {
      const off = t * D;
      let dot = 0.0;
      for (let j = 0; j < D; j++) dot += dPooled[j] * H[off + j];
      dAlpha[t] = dot;
    }

    // dLogits = softmaxBackward(dAlpha, alpha)
    // dLogits[t] = alpha[t] * (dAlpha[t] - sum(dAlpha*alpha))
    let dotp = 0.0;
    for (let t = 0; t < L; t++) dotp += dAlpha[t] * alpha[t];
    const dLogits = this._gPoolLogits;
    for (let t = 0; t < L; t++) dLogits[t] = alpha[t] * (dAlpha[t] - dotp);

    // Grad Wpool, bpool
    // logits[t] = dot(H[t], W) + b
    // dW += sum_t dLogits[t] * H[t]
    // db += sum_t dLogits[t]
    let db = 0.0;
    for (let t = 0; t < L; t++) db += dLogits[t];
    this._gPoolB[0] += db;

    for (let j = 0; j < D; j++) {
      let sum = 0.0;
      for (let t = 0; t < L; t++) sum += dLogits[t] * H[t * D + j];
      this._gPoolW[j] += sum;
    }

    // dH from two paths:
    // pooled = sum alpha[t] * H[t] => dH += alpha[t] * dPooled
    for (let t = 0; t < L; t++) {
      const a = alpha[t];
      const off = t * D;
      for (let j = 0; j < D; j++) dH[off + j] += a * dPooled[j];
    }

    // logits path: logits depends on H: dH += dLogits[t] * Wpool
    for (let t = 0; t < L; t++) {
      const dl = dLogits[t];
      const off = t * D;
      for (let j = 0; j < D; j++) dH[off + j] += dl * this._poolW[j];
    }
  }

  // ==========================================================================
  // Linear forward/backward helpers (mat + vec)
  // ==========================================================================

  private _linearMatForward(
    X: Float64Array, // [rows,inDim]
    rows: number,
    inDim: number,
    W: Float64Array, // [inDim,outDim]
    b: Float64Array, // [outDim]
    outDim: number,
    out: Float64Array, // [rows,outDim]
  ): void {
    for (let r = 0; r < rows; r++) {
      const xOff = r * inDim;
      const oOff = r * outDim;
      // start with bias
      for (let j = 0; j < outDim; j++) out[oOff + j] = b[j];

      for (let k = 0; k < inDim; k++) {
        const xv = X[xOff + k];
        const wOff = k * outDim;
        for (let j = 0; j < outDim; j++) out[oOff + j] += xv * W[wOff + j];
      }
    }
  }

  /**
   * Backprop for Y = X*W + b.
   * Accumulates into dX, dW, db.
   */
  private _linearMatBackward(
    X: Float64Array,
    rows: number,
    inDim: number,
    W: Float64Array,
    outDim: number,
    dY: Float64Array,
    dX: Float64Array,
    dW: Float64Array,
    db: Float64Array,
  ): void {
    // db
    for (let j = 0; j < outDim; j++) {
      let sum = 0.0;
      for (let r = 0; r < rows; r++) sum += dY[r * outDim + j];
      db[j] += sum;
    }

    // dW and dX
    for (let r = 0; r < rows; r++) {
      const xOff = r * inDim;
      const dyOff = r * outDim;

      // dX[r,k] += sum_j dY[r,j] * W[k,j]
      for (let k = 0; k < inDim; k++) {
        let sum = 0.0;
        const wOff = k * outDim;
        for (let j = 0; j < outDim; j++) sum += dY[dyOff + j] * W[wOff + j];
        dX[xOff + k] += sum;
      }

      // dW[k,j] += X[r,k] * dY[r,j]
      for (let k = 0; k < inDim; k++) {
        const xv = X[xOff + k];
        const wOff = k * outDim;
        for (let j = 0; j < outDim; j++) dW[wOff + j] += xv * dY[dyOff + j];
      }
    }
  }

  private _linearVecForward(
    x: Float64Array, // [inDim]
    inDim: number,
    W: Float64Array, // [inDim,outDim]
    b: Float64Array, // [outDim]
    outDim: number,
    out: Float64Array, // [outDim]
  ): void {
    for (let j = 0; j < outDim; j++) out[j] = b[j];
    for (let k = 0; k < inDim; k++) {
      const xv = x[k];
      const wOff = k * outDim;
      for (let j = 0; j < outDim; j++) out[j] += xv * W[wOff + j];
    }
  }

  // ==========================================================================
  // Dropout (optional, deterministic RNG)
  // ==========================================================================

  private _applyDropoutInPlace(
    x: Float64Array,
    n: number,
    p: number,
    mask: Uint8Array,
  ): void {
    const keep = 1.0 - p;
    const invKeep = 1.0 / (keep + 1e-18);
    for (let i = 0; i < n; i++) {
      const r = this._rng.nextF64();
      const m = r < keep ? 1 : 0;
      mask[i] = m;
      x[i] = m ? (x[i] * invKeep) : 0.0;
    }
  }

  private _applyDropoutBackwardInPlace(
    dy: Float64Array,
    n: number,
    p: number,
    mask: Uint8Array,
  ): void {
    const keep = 1.0 - p;
    const invKeep = 1.0 / (keep + 1e-18);
    for (let i = 0; i < n; i++) {
      dy[i] = mask[i] ? (dy[i] * invKeep) : 0.0;
    }
  }

  // ==========================================================================
  // L2 regularization
  // ==========================================================================

  private _applyL2(): void {
    const lam = this._cfg.regularizationStrength;
    if (lam <= 0) return;

    // Apply to weight matrices (not biases, not LayerNorm beta, not pool bias/out bias)
    // Conv weights
    for (let s = 0; s < this._convW.length; s++) {
      const w = this._convW[s];
      const g = this._gConvW[s];
      for (let i = 0; i < w.length; i++) g[i] += lam * w[i];
    }

    // Fusion Wg
    for (let i = 0; i < this._fusionWg.length; i++) {
      this._gFusionWg[i] += lam * this._fusionWg[i];
    }

    // Attention and FFN weights
    const B = this._cfg.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      const wq = this._attWq[b], gq = this._gAttWq[b];
      const wk = this._attWk[b], gk = this._gAttWk[b];
      const wv = this._attWv[b], gv = this._gAttWv[b];
      const wo = this._attWo[b], go = this._gAttWo[b];
      for (let i = 0; i < wq.length; i++) gq[i] += lam * wq[i];
      for (let i = 0; i < wk.length; i++) gk[i] += lam * wk[i];
      for (let i = 0; i < wv.length; i++) gv[i] += lam * wv[i];
      for (let i = 0; i < wo.length; i++) go[i] += lam * wo[i];

      const w1 = this._ffW1[b], g1 = this._gFfW1[b];
      const w2 = this._ffW2[b], g2 = this._gFfW2[b];
      for (let i = 0; i < w1.length; i++) g1[i] += lam * w1[i];
      for (let i = 0; i < w2.length; i++) g2[i] += lam * w2[i];
    }

    // Pool W
    for (let i = 0; i < this._poolW.length; i++) {
      this._gPoolW[i] += lam * this._poolW[i];
    }

    // Output W
    for (let i = 0; i < this._outW.length; i++) {
      this._gOutW[i] += lam * this._outW[i];
    }
  }

  // ==========================================================================
  // Schedules, accuracy, parameter count
  // ==========================================================================

  private _effectiveLearningRate(t: number): number {
    const base = this._cfg.learningRate;
    const warm = this._cfg.warmupSteps | 0;
    const total = this._cfg.totalSteps | 0;

    if (t <= 0) return base;
    if (t <= warm && warm > 0) {
      return base * (t / warm);
    }
    if (total <= warm + 1) return base;

    const tt = t - warm;
    const T = total - warm;
    const u = clamp(tt / T, 0.0, 1.0);
    // cosine decay to 0
    return base * (0.5 * (1.0 + Math.cos(Math.PI * u)));
  }

  private _accuracy(): number {
    // accuracy = 1/(1 + runningLoss)
    return 1.0 / (1.0 + Math.max(0.0, this._runningLoss));
  }

  private _countParameters(): number {
    if (!this._isInitialized) return 0;
    let n = 0;

    for (let s = 0; s < this._convW.length; s++) {
      n += this._convW[s].length + this._convB[s].length;
    }
    n += this._scaleEmb.length;

    n += this._fusionWg.length + this._fusionBg.length;

    const B = this._cfg.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      n += this._ln1Gamma[b].length + this._ln1Beta[b].length;
      n += this._ln2Gamma[b].length + this._ln2Beta[b].length;

      n += this._attWq[b].length + this._attBq[b].length;
      n += this._attWk[b].length + this._attBk[b].length;
      n += this._attWv[b].length + this._attBv[b].length;
      n += this._attWo[b].length + this._attBo[b].length;

      n += this._ffW1[b].length + this._ffB1[b].length;
      n += this._ffW2[b].length + this._ffB2[b].length;
    }

    n += this._poolW.length + this._poolB.length;
    n += this._outW.length + this._outB.length;

    return n;
  }

  // ==========================================================================
  // Positional encoding
  // ==========================================================================

  /**
   * PE(pos,2i)   = sin(pos / 10000^(2i/d))
   * PE(pos,2i+1) = cos(pos / 10000^(2i/d))
   */
  private _buildPositionalEncoding(
    maxL: number,
    D: number,
    out: Float64Array,
  ): void {
    const invDen: Float64Array = new Float64Array(D);
    // precompute 1/10000^(2i/D)
    for (let i = 0; i < D; i++) {
      const ii = (i >> 1) | 0;
      invDen[i] = 1.0 / Math.pow(10000.0, (2.0 * ii) / D);
    }

    for (let pos = 0; pos < maxL; pos++) {
      const off = pos * D;
      for (let i = 0; i < D; i++) {
        const v = pos * invDen[i];
        out[off + i] = (i & 1) === 0 ? Math.sin(v) : Math.cos(v);
      }
    }
  }

  // ==========================================================================
  // Residual stats (for standard error)
  // ==========================================================================

  private _updateResidualStats(yTrue: Float64Array, yHat: Float64Array): void {
    const d = this._outDim;
    this._resCount++;
    const n = this._resCount;
    const invN = 1.0 / n;

    for (let j = 0; j < d; j++) {
      const r = yTrue[j] - yHat[j];
      const delta = r - this._resMean[j];
      const meanNew = this._resMean[j] + delta * invN;
      const delta2 = r - meanNew;
      this._resMean[j] = meanNew;
      this._resM2[j] += delta * delta2;
    }
  }

  private _estimateStdErr(outDim: number): Float64Array {
    const se = new Float64Array(outDim);
    const n = this._resCount;
    if (n <= 1) {
      for (let j = 0; j < outDim; j++) se[j] = 1.0;
      return se;
    }
    const inv = 1.0 / (n - 1);
    for (let j = 0; j < outDim; j++) {
      const varj = this._resM2[j] * inv;
      se[j] = Math.sqrt(Math.max(1e-18, varj));
    }
    return se;
  }

  // ==========================================================================
  // Xavier init
  // ==========================================================================

  private _xavierInit(w: Float64Array, fanIn: number, fanOut: number): void {
    const scale = Math.sqrt(2.0 / (fanIn + fanOut + 1e-18));
    for (let i = 0; i < w.length; i++) w[i] = this._rng.nextNormal() * scale;
  }

  // ==========================================================================
  // Serialization of parameters (base64)
  // ==========================================================================

  private _serializeParams(): any {
    const S = this._cfg.temporalScales.length | 0;
    const B = this._cfg.numBlocks | 0;

    const convW: any[] = new Array(S);
    const convB: any[] = new Array(S);
    for (let s = 0; s < S; s++) {
      convW[s] = f64ToB64(this._convW[s]);
      convB[s] = f64ToB64(this._convB[s]);
    }

    const ln1G: any[] = new Array(B),
      ln1B: any[] = new Array(B),
      ln2G: any[] = new Array(B),
      ln2B: any[] = new Array(B);
    const att: any[] = new Array(B);
    const ffn: any[] = new Array(B);

    for (let b = 0; b < B; b++) {
      ln1G[b] = f64ToB64(this._ln1Gamma[b]);
      ln1B[b] = f64ToB64(this._ln1Beta[b]);
      ln2G[b] = f64ToB64(this._ln2Gamma[b]);
      ln2B[b] = f64ToB64(this._ln2Beta[b]);

      att[b] = {
        Wq: f64ToB64(this._attWq[b]),
        bq: f64ToB64(this._attBq[b]),
        Wk: f64ToB64(this._attWk[b]),
        bk: f64ToB64(this._attBk[b]),
        Wv: f64ToB64(this._attWv[b]),
        bv: f64ToB64(this._attBv[b]),
        Wo: f64ToB64(this._attWo[b]),
        bo: f64ToB64(this._attBo[b]),
      };

      ffn[b] = {
        W1: f64ToB64(this._ffW1[b]),
        b1: f64ToB64(this._ffB1[b]),
        W2: f64ToB64(this._ffW2[b]),
        b2: f64ToB64(this._ffB2[b]),
      };
    }

    return {
      posEnc: f64ToB64(this._posEnc),

      convW,
      convB,
      scaleEmb: f64ToB64(this._scaleEmb),

      fusionWg: f64ToB64(this._fusionWg),
      fusionBg: f64ToB64(this._fusionBg),

      ln1G,
      ln1B,
      ln2G,
      ln2B,
      att,
      ffn,

      poolW: f64ToB64(this._poolW),
      poolB: f64ToB64(this._poolB),

      outW: f64ToB64(this._outW),
      outB: f64ToB64(this._outB),
    };
  }

  private _deserializeParams(p: any): void {
    const S = this._cfg.temporalScales.length | 0;
    const B = this._cfg.numBlocks | 0;

    this._posEnc = b64ToF64(p.posEnc);

    for (let s = 0; s < S; s++) {
      this._convW[s] = b64ToF64(p.convW[s]);
      this._convB[s] = b64ToF64(p.convB[s]);
    }

    this._scaleEmb = b64ToF64(p.scaleEmb);
    this._fusionWg = b64ToF64(p.fusionWg);
    this._fusionBg = b64ToF64(p.fusionBg);

    for (let b = 0; b < B; b++) {
      this._ln1Gamma[b] = b64ToF64(p.ln1G[b]);
      this._ln1Beta[b] = b64ToF64(p.ln1B[b]);
      this._ln2Gamma[b] = b64ToF64(p.ln2G[b]);
      this._ln2Beta[b] = b64ToF64(p.ln2B[b]);

      const att = p.att[b];
      this._attWq[b] = b64ToF64(att.Wq);
      this._attBq[b] = b64ToF64(att.bq);
      this._attWk[b] = b64ToF64(att.Wk);
      this._attBk[b] = b64ToF64(att.bk);
      this._attWv[b] = b64ToF64(att.Wv);
      this._attBv[b] = b64ToF64(att.bv);
      this._attWo[b] = b64ToF64(att.Wo);
      this._attBo[b] = b64ToF64(att.bo);

      const ffn = p.ffn[b];
      this._ffW1[b] = b64ToF64(ffn.W1);
      this._ffB1[b] = b64ToF64(ffn.b1);
      this._ffW2[b] = b64ToF64(ffn.W2);
      this._ffB2[b] = b64ToF64(ffn.b2);
    }

    this._poolW = b64ToF64(p.poolW);
    this._poolB = b64ToF64(p.poolB);

    this._outW = b64ToF64(p.outW);
    this._outB = b64ToF64(p.outB);
  }
}
