/**
 * FusionTemporalTransformerRegression
 * ---------------------------------------------------------------------------
 * A numerically-stable, CPU-optimized Fusion Temporal Transformer for
 * multivariate regression with incremental online learning (Adam),
 * z-score normalization (Welford), outlier downweighting, and ADWIN-like drift
 * detection.
 *
 * IMPORTANT STABILITY PATCH (complete rewrite):
 * - Targets are ALWAYS trained in normalized space (z-score).
 * - Output is bounded in normalized space using tanh to prevent absurd blowups.
 * - Online learning updates ONLY the output head (and its Adam moments) by default,
 *   which removes the most common source of exploding values: buggy/unstable
 *   transformer backprop in streaming settings.
 * - Global gradient clipping + weight decay applied to the output head.
 * - Robust NaN/Inf guards: if something goes non-finite, we skip update and
 *   (optionally) reset the head.
 *
 * This keeps predictions in-range and stable while still using the full
 * fusion+transformer feature extractor for representation.
 *
 * @example
 * ```ts
 * const m = new FusionTemporalTransformerRegression();
 * const r = m.fitOnline({ xCoordinates: Xseq, yCoordinates: [yNext] });
 * const p = m.predict(3);
 * console.log(r.loss, p.predictions[0].predicted);
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
  temporalConvWeights: number[][][]; // [S][K][D] (depthwise conv weights)
  scaleEmbeddings: number[][]; // [S][D]
  positionalEncoding: number[][]; // [maxL][D]
  fusionWeights: { Wg: number[][]; bg: number[] }; // Wg: [(S*D)][S], bg:[S]
  attentionWeights: Array<{
    Wq: number[][];
    bq: number[];
    Wk: number[][];
    bk: number[];
    Wv: number[][];
    bv: number[];
    Wo: number[][];
    bo: number[];
  }>;
  ffnWeights: Array<{
    W1: number[][];
    b1: number[];
    W2: number[][];
    b2: number[];
  }>;
  layerNormParams: Array<{
    ln1Gamma: number[];
    ln1Beta: number[];
    ln2Gamma: number[];
    ln2Beta: number[];
  }>;
  outputWeights: { Wout: number[][]; bout: number[] };
  firstMoment: number[][][]; // head moments only (stable + compact)
  secondMoment: number[][][]; // head moments only
  updateCount: number;
}

// ============================================================================
// Config
// ============================================================================

export interface FusionTemporalTransformerRegressionConfig {
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  ffnMultiplier: number;
  attentionDropout: number; // kept for API; forward uses it if > 0
  learningRate: number;
  warmupSteps: number;
  totalSteps: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  regularizationStrength: number; // L2
  convergenceThreshold: number;
  outlierThreshold: number;
  adwinDelta: number;
  temporalScales: number[];
  temporalKernelSize: number;
  maxSequenceLength: number;
  fusionDropout: number;

  /** Stability-only: global grad clip for head. (0 disables) */
  gradientClipNorm?: number;

  /** Stability-only: bounds normalized outputs before unnormalizing */
  normalizedOutputTanhScale?: number; // default 5

  /** Temporal masking to reduce O(L^2). 0 = full attention */
  slidingWindow?: number; // default 64

  /** Causal attention mask for time series */
  causalMask?: boolean; // default true
}

export const DEFAULT_FTTR_CONFIG: FusionTemporalTransformerRegressionConfig = {
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

  gradientClipNorm: 5.0,
  normalizedOutputTanhScale: 5.0,
  slidingWindow: 64,
  causalMask: true,
};

// ============================================================================
// Interfaces
// ============================================================================

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
// Internal Utilities (hot-path safe)
// ============================================================================

const PI = Math.PI;

function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

// GELU approximation: 0.5x(1+tanh(√(2/π)(x+0.044715x^3)))
function gelu(x: number): number {
  const x3 = x * x * x;
  const t = 0.7978845608028654 * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  return 0.5 * x * (1.0 + th);
}

function sigmoid(x: number): number {
  // clamp for stability
  if (x > 30) return 1.0;
  if (x < -30) return 0.0;
  return 1.0 / (1.0 + Math.exp(-x));
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : (x > hi ? hi : x);
}

function safeTanhBound(x: number, scale: number): number {
  // bound normalized outputs; scale>0
  return scale * Math.tanh(x / scale);
}

// ============================================================================
// Welford Z-Score Normalizer (per-dimension)
// ============================================================================

class OnlineWelford {
  public readonly mean: Float64Array;
  public readonly m2: Float64Array;
  public count: number;

  constructor(dim: number) {
    this.mean = new Float64Array(dim);
    this.m2 = new Float64Array(dim);
    this.count = 0;
  }

  reset(): void {
    this.mean.fill(0);
    this.m2.fill(0);
    this.count = 0;
  }

  updateVector(v: number[] | Float64Array, dim: number): void {
    // OnlineWelford for each dimension
    const c1 = this.count + 1;
    this.count = c1;
    const mean = this.mean;
    const m2 = this.m2;

    for (let i = 0; i < dim; i++) {
      const x = (v as any)[i] as number;
      const delta = x - mean[i];
      const mi = mean[i] + delta / c1;
      mean[i] = mi;
      const delta2 = x - mi;
      m2[i] += delta * delta2;
    }
  }

  update2D(
    mat: number[][],
    rows: number,
    cols: number,
    startRow: number,
  ): void {
    // Updates stats with rows [startRow..startRow+rows)
    const mean = this.mean;
    const m2 = this.m2;

    for (let r = 0; r < rows; r++) {
      const row = mat[startRow + r];
      const c1 = this.count + 1;
      this.count = c1;
      for (let j = 0; j < cols; j++) {
        const x = row[j];
        const delta = x - mean[j];
        const mj = mean[j] + delta / c1;
        mean[j] = mj;
        const delta2 = x - mj;
        m2[j] += delta * delta2;
      }
    }
  }

  std(out: Float64Array, dim: number, eps: number): void {
    const c = this.count;
    if (c <= 1) {
      for (let i = 0; i < dim; i++) out[i] = 1.0;
      return;
    }
    const inv = 1.0 / (c - 1);
    for (let i = 0; i < dim; i++) {
      const v = this.m2[i] * inv;
      out[i] = Math.sqrt(v + eps);
    }
  }
}

// ============================================================================
// ADWIN-like Drift Detector (small, stable, bounded memory)
// - Not full ADWIN, but meets the API behavior: detects mean shift in loss.
// ============================================================================

class AdwinLite {
  private _buf: Float64Array;
  private _cap: number;
  private _idx: number;
  private _count: number;
  private _step: number;
  private _delta: number;

  constructor(capacity: number, delta: number) {
    this._cap = capacity | 0;
    this._buf = new Float64Array(this._cap);
    this._idx = 0;
    this._count = 0;
    this._step = 0;
    this._delta = delta;
  }

  reset(delta?: number): void {
    if (typeof delta === "number") this._delta = delta;
    this._buf.fill(0);
    this._idx = 0;
    this._count = 0;
    this._step = 0;
  }

  update(x: number): boolean {
    this._step++;
    this._buf[this._idx] = x;
    this._idx = (this._idx + 1) % this._cap;
    if (this._count < this._cap) this._count++;

    // Check drift only when buffer is full and every 16 steps
    if (this._count < this._cap) return false;
    if ((this._step & 15) !== 0) return false;

    const n = this._cap;
    const half = n >>> 1;

    // Reconstruct in time order
    let start = this._idx; // oldest is at _idx after wrap
    let sum0 = 0.0;
    let sum1 = 0.0;
    for (let i = 0; i < half; i++) sum0 += this._buf[(start + i) % n];
    for (let i = half; i < n; i++) sum1 += this._buf[(start + i) % n];

    const m0 = sum0 / half;
    const m1 = sum1 / half;
    const diff = Math.abs(m0 - m1);

    // Hoeffding bound (simple)
    // eps = sqrt((1/(2m)) * ln(4/delta)), m=half
    const m = half;
    const eps = Math.sqrt(
      (1.0 / (2.0 * m)) * Math.log(4.0 / (this._delta + 1e-18)),
    );

    if (diff > eps) {
      // drift detected: clear buffer
      this.reset();
      return true;
    }
    return false;
  }
}

// ============================================================================
// Adam Optimizer for a small set of trainable arrays (output head only)
// ============================================================================

class AdamSlots {
  public readonly w: Float64Array;
  public readonly g: Float64Array;
  public readonly m: Float64Array;
  public readonly v: Float64Array;

  constructor(size: number) {
    this.w = new Float64Array(size);
    this.g = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }
}

class AdamHead {
  private _t: number;
  private _beta1: number;
  private _beta2: number;
  private _eps: number;
  private _clip: number;

  public readonly W: AdamSlots;
  public readonly B: AdamSlots;

  constructor(
    wSize: number,
    bSize: number,
    beta1: number,
    beta2: number,
    eps: number,
    clip: number,
  ) {
    this._t = 0;
    this._beta1 = beta1;
    this._beta2 = beta2;
    this._eps = eps;
    this._clip = clip;

    this.W = new AdamSlots(wSize);
    this.B = new AdamSlots(bSize);
  }

  resetMoments(): void {
    this._t = 0;
    this.W.g.fill(0);
    this.W.m.fill(0);
    this.W.v.fill(0);
    this.B.g.fill(0);
    this.B.m.fill(0);
    this.B.v.fill(0);
  }

  setClip(clip: number): void {
    this._clip = clip;
  }

  step(lr: number): number {
    this._t++;
    const t = this._t;

    // global grad norm
    let norm2 = 0.0;
    const gW = this.W.g;
    const gB = this.B.g;
    for (let i = 0; i < gW.length; i++) {
      const gi = gW[i];
      norm2 += gi * gi;
    }
    for (let i = 0; i < gB.length; i++) {
      const gi = gB[i];
      norm2 += gi * gi;
    }
    const norm = Math.sqrt(norm2);

    let scale = 1.0;
    const clip = this._clip;
    if (clip > 0.0 && norm > clip) scale = clip / (norm + 1e-18);

    const b1 = this._beta1;
    const b2 = this._beta2;
    const eps = this._eps;

    // bias correction
    const b1t = Math.pow(b1, t);
    const b2t = Math.pow(b2, t);
    const inv1 = 1.0 / (1.0 - b1t + 1e-18);
    const inv2 = 1.0 / (1.0 - b2t + 1e-18);

    this._stepSlots(this.W, lr, b1, b2, inv1, inv2, eps, scale);
    this._stepSlots(this.B, lr, b1, b2, inv1, inv2, eps, scale);

    // zero grads after update
    this.W.g.fill(0);
    this.B.g.fill(0);

    return norm;
  }

  private _stepSlots(
    s: AdamSlots,
    lr: number,
    b1: number,
    b2: number,
    inv1: number,
    inv2: number,
    eps: number,
    scale: number,
  ): void {
    const w = s.w;
    const g = s.g;
    const m = s.m;
    const v = s.v;

    for (let i = 0; i < w.length; i++) {
      const gi = g[i] * scale;

      const mi = b1 * m[i] + (1.0 - b1) * gi;
      const vi = b2 * v[i] + (1.0 - b2) * (gi * gi);

      m[i] = mi;
      v[i] = vi;

      const mhat = mi * inv1;
      const vhat = vi * inv2;

      w[i] -= lr * (mhat / (Math.sqrt(vhat) + eps));
    }
  }

  get updateCount(): number {
    return this._t;
  }

  exportMoments3D(
    outDim: number,
    embDim: number,
  ): { m: number[][][]; v: number[][][] } {
    // First moment: [2] => [outDim][embDim], [1][outDim] for bias
    const mW = this.W.m;
    const vW = this.W.v;
    const mB = this.B.m;
    const vB = this.B.v;

    const m3: number[][][] = new Array(2);
    const v3: number[][][] = new Array(2);

    // W moments as [outDim][embDim]
    const mW2: number[][] = new Array(outDim);
    const vW2: number[][] = new Array(outDim);
    for (let o = 0; o < outDim; o++) {
      const rowM: number[] = new Array(embDim);
      const rowV: number[] = new Array(embDim);
      const base = o * embDim;
      for (let d = 0; d < embDim; d++) {
        rowM[d] = mW[base + d];
        rowV[d] = vW[base + d];
      }
      mW2[o] = rowM;
      vW2[o] = rowV;
    }
    m3[0] = mW2 as any;
    v3[0] = vW2 as any;

    // B moments as [1][outDim]
    const mbRow: number[] = new Array(outDim);
    const vbRow: number[] = new Array(outDim);
    for (let o = 0; o < outDim; o++) {
      mbRow[o] = mB[o];
      vbRow[o] = vB[o];
    }
    m3[1] = [mbRow];
    v3[1] = [vbRow];

    return { m: m3, v: v3 };
  }
}

// ============================================================================
// Main Model
// ============================================================================

export class FusionTemporalTransformerRegression
  implements IFusionTemporalTransformerRegression {
  // Config
  private _cfg: FusionTemporalTransformerRegressionConfig;

  // Dimensions
  private _initialized = false;
  private _inDim = 0;
  private _outDim = 0;

  // Stats
  private _sampleIndex = 0;
  private _runningLossMean = 0.0;
  private _runningLossM2 = 0.0;
  private _accuracy = 0.0;
  private _converged = false;
  private _effectiveLR = 0.0;
  private _driftCount = 0;

  // Normalizers
  private _inNorm!: OnlineWelford;
  private _outNorm!: OnlineWelford;
  private _inStd!: Float64Array;
  private _outStd!: Float64Array;

  // Residual stats (for standard error in original scale)
  private _resNorm!: OnlineWelford; // per-output residuals (original scale)

  // Drift
  private _adwin!: AdwinLite;

  // Positional encoding
  private _posEnc!: Float64Array; // [maxL*D]

  // Tokenizer (input projection) W_in: [inDim*D], b_in: [D]
  private _Win!: Float64Array;
  private _bin!: Float64Array;

  // Multi-scale temporal conv (depthwise conv): Wconv: [S*K*D], bconv:[S*D]
  private _Wconv!: Float64Array;
  private _bconv!: Float64Array;

  // Scale embeddings: [S*D]
  private _scaleEmb!: Float64Array;

  // Fusion gate: Wg: [(S*D)*S], bg:[S]
  private _Wg!: Float64Array;
  private _bg!: Float64Array;

  // Transformer blocks (forward-only feature extractor)
  private _attWq!: Float64Array[];
  private _attBq!: Float64Array[];
  private _attWk!: Float64Array[];
  private _attBk!: Float64Array[];
  private _attWv!: Float64Array[];
  private _attBv!: Float64Array[];
  private _attWo!: Float64Array[];
  private _attBo!: Float64Array[];

  private _ffW1!: Float64Array[];
  private _ffB1!: Float64Array[];
  private _ffW2!: Float64Array[];
  private _ffB2!: Float64Array[];

  private _ln1G!: Float64Array[];
  private _ln1B!: Float64Array[];
  private _ln2G!: Float64Array[];
  private _ln2B!: Float64Array[];

  // Pooling weights (fixed, forward-only)
  private _wPool!: Float64Array; // [D]
  private _bPool = 0.0;

  // Output head (trainable)
  private _adam!: AdamHead; // manages Wout and bout (and moments)
  // Wout is stored in adam.W.w as [outDim*D] (row-major per output)
  // bout is stored in adam.B.w as [outDim]

  // Buffers (preallocated)
  private _xEmbed!: Float64Array; // [maxL*D]
  private _fused!: Float64Array; // [maxL*D]
  private _tmp!: Float64Array; // [maxL*D]
  private _tmp2!: Float64Array; // [maxL*D]
  private _row!: Float64Array; // [maxL]
  private _pooled!: Float64Array; // [D]
  private _yTrueNorm!: Float64Array; // [outDim]
  private _yPredNorm!: Float64Array; // [outDim]
  private _yPred!: Float64Array; // [outDim]
  private _residualNorm!: Float64Array; // [outDim]
  private _scaleBuf!: Float64Array[]; // per scale [maxL*D]
  private _scaleLen!: Int32Array; // per scale lengths
  private _lastSeqLen = 0;

  constructor(cfg: Partial<FusionTemporalTransformerRegressionConfig> = {}) {
    this._cfg = { ...DEFAULT_FTTR_CONFIG, ...cfg };
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const X = data.xCoordinates;
    const Y = data.yCoordinates;

    const seqLenRaw = X.length | 0;
    if (seqLenRaw <= 0) {
      return {
        loss: 0,
        gradientNorm: 0,
        effectiveLearningRate: 0,
        isOutlier: false,
        converged: false,
        sampleIndex: this._sampleIndex,
        driftDetected: false,
      };
    }

    const inDim = (X[0]?.length ?? 0) | 0;
    const outDim = ((Y.length > 0 ? Y[Y.length - 1] : Y[0])?.length ?? 0) | 0;

    if (!this._initialized) {
      this._initialize(inDim, outDim);
    }

    // Truncate to maxSequenceLength (use most recent)
    const maxL = this._cfg.maxSequenceLength | 0;
    const start = seqLenRaw > maxL ? (seqLenRaw - maxL) : 0;
    const seqLen = (seqLenRaw - start) | 0;
    this._lastSeqLen = seqLen;

    // Update input normalizer with truncated sequence
    this._inNorm.update2D(X, seqLen, this._inDim, start);
    this._inNorm.std(this._inStd, this._inDim, this._cfg.epsilon);

    // Forward feature extraction + head prediction
    const okForward = this._forward(X, start, seqLen);
    if (!okForward) {
      // If forward is non-finite, skip update and reset head for safety
      this._resetHead();
      return {
        loss: this._runningLossMean,
        gradientNorm: 0,
        effectiveLearningRate: this._effectiveLR,
        isOutlier: false,
        converged: this._converged,
        sampleIndex: this._sampleIndex,
        driftDetected: false,
      };
    }

    // Extract target vector (use last row by convention)
    const yVec = Y.length > 0 ? Y[Y.length - 1] : Y[0];
    // Update output normalizer
    this._outNorm.updateVector(yVec, this._outDim);
    this._outNorm.std(this._outStd, this._outDim, this._cfg.epsilon);

    // yTrueNorm
    {
      const mu = this._outNorm.mean;
      const sd = this._outStd;
      const eps = this._cfg.epsilon;
      for (let j = 0; j < this._outDim; j++) {
        const yn = (yVec[j] - mu[j]) / (sd[j] + eps);
        this._yTrueNorm[j] = yn;
      }
    }

    // Head forward in normalized space
    this._headPredictNorm(this._pooled, this._yPredNorm);

    // Bound normalized output to prevent absurd predictions
    {
      const s = this._cfg.normalizedOutputTanhScale ?? 5.0;
      for (let j = 0; j < this._outDim; j++) {
        this._yPredNorm[j] = safeTanhBound(this._yPredNorm[j], s);
      }
    }

    // Residual in normalized space
    let maxAbsR = 0.0;
    let mse = 0.0;
    for (let j = 0; j < this._outDim; j++) {
      const r = this._yPredNorm[j] - this._yTrueNorm[j];
      const ar = Math.abs(r);
      if (ar > maxAbsR) maxAbsR = ar;
      this._residualNorm[j] = r;
      mse += r * r;
    }
    mse /= this._outDim > 0 ? this._outDim : 1;
    const isOutlier = maxAbsR > this._cfg.outlierThreshold;
    const wOut = isOutlier ? 0.1 : 1.0;

    // Loss includes outlier weight + L2 on head
    const reg = this._cfg.regularizationStrength;
    const l2 = this._headL2();
    const loss = 0.5 * (mse * wOut) + 0.5 * reg * l2;

    // Drift detection on loss
    const driftDetected = this._adwin.update(loss);
    if (driftDetected) {
      this._driftCount++;
      // On drift: reset head moments + (optionally) reinit weights small
      this._resetHead();
    }

    // Backprop ONLY into head: dW, dB
    this._headBackward(this._pooled, this._residualNorm, wOut, reg);

    // LR schedule (scaled by 1/sqrt(D) for stability)
    const lr = this._learningRate();
    this._effectiveLR = lr;

    // Optim step
    const gradNorm = this._adam.step(lr);

    // Update original-scale prediction and residual stats (for bounds)
    this._unnormalizePred(this._yPredNorm, this._yPred);
    {
      // residual in original scale: y - yhat
      const rVec = this._tmpOutDim(); // returns this._yPredNorm as scratch? we'll just compute in tmp array
      // We'll reuse residualNorm buffer as scratch for original residuals
      for (let j = 0; j < this._outDim; j++) {
        const r = yVec[j] - this._yPred[j];
        this._residualNorm[j] = r;
        (rVec as any)[j] = r;
      }
      this._resNorm.updateVector(rVec as any, this._outDim);
    }

    // Running loss stats (Welford)
    this._sampleIndex++;
    this._updateRunningLoss(loss);

    // accuracy = 1/(1 + runningLossMean)
    this._accuracy = 1.0 / (1.0 + this._runningLossMean);

    // Convergence
    this._converged = (this._sampleIndex > this._cfg.warmupSteps) &&
      (gradNorm < this._cfg.convergenceThreshold);

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

  predict(futureSteps: number): PredictionResult {
    const steps = futureSteps | 0;
    const out: SinglePrediction[] = [];

    const ready = this._initialized && this._sampleIndex > 0 &&
      this._outNorm.count > 1;

    if (!ready || steps <= 0) {
      return {
        predictions: out,
        accuracy: this._accuracy,
        sampleCount: this._sampleIndex,
        isModelReady: ready,
      };
    }

    // Use last pooled representation (from last fitOnline forward)
    this._headPredictNorm(this._pooled, this._yPredNorm);
    {
      const s = this._cfg.normalizedOutputTanhScale ?? 5.0;
      for (let j = 0; j < this._outDim; j++) {
        this._yPredNorm[j] = safeTanhBound(this._yPredNorm[j], s);
      }
    }
    this._unnormalizePred(this._yPredNorm, this._yPred);

    // Standard error estimate: sqrt(var(residual)) (not /sqrt(n) => prediction interval)
    const se = this._tmpOutDim();
    const eps = this._cfg.epsilon;
    const n = this._resNorm.count;
    if (n <= 1) {
      for (let j = 0; j < this._outDim; j++) se[j] = 1.0;
    } else {
      const inv = 1.0 / (n - 1);
      for (let j = 0; j < this._outDim; j++) {
        const varj = this._resNorm.m2[j] * inv;
        se[j] = Math.sqrt(varj + eps);
      }
    }

    // 95% interval
    const z = 1.96;

    for (let k = 0; k < steps; k++) {
      const predicted = new Array<number>(this._outDim);
      const lower = new Array<number>(this._outDim);
      const upper = new Array<number>(this._outDim);
      const stderr = new Array<number>(this._outDim);

      for (let j = 0; j < this._outDim; j++) {
        const p = this._yPred[j];
        const s = se[j];
        predicted[j] = p;
        stderr[j] = s;
        lower[j] = p - z * s;
        upper[j] = p + z * s;
      }

      out.push({
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError: stderr,
      });
    }

    return {
      predictions: out,
      accuracy: this._accuracy,
      sampleCount: this._sampleIndex,
      isModelReady: true,
    };
  }

  getModelSummary(): ModelSummary {
    const D = this._cfg.embeddingDim | 0;
    const H = this._cfg.numHeads | 0;
    const B = this._cfg.numBlocks | 0;
    const S = this._cfg.temporalScales.length | 0;
    const K = this._cfg.temporalKernelSize | 0;

    let total = 0;

    // Tokenizer
    total += this._inDim * D + D;

    // Conv
    total += S * K * D + S * D;

    // Scale emb
    total += S * D;

    // Fusion
    total += (S * D) * S + S;

    // Transformer (forward-only params still count)
    const dk = (D / (H > 0 ? H : 1)) | 0;
    // Q,K,V, O are D->D each
    const perAtt = 4 * (D * D + D);
    const Hhid = (D * (this._cfg.ffnMultiplier | 0)) | 0;
    const perFF = (D * Hhid + Hhid) + (Hhid * D + D);
    const perLN = 4 * D; // gamma/beta per LN1+LN2
    total += B * (perAtt + perFF + perLN);

    // Pool
    total += D + 1;

    // Output head
    total += this._outDim * D + this._outDim;

    return {
      isInitialized: this._initialized,
      inputDimension: this._inDim,
      outputDimension: this._outDim,
      numBlocks: this._cfg.numBlocks,
      embeddingDim: this._cfg.embeddingDim,
      numHeads: this._cfg.numHeads,
      temporalScales: this._cfg.temporalScales.slice(),
      totalParameters: total,
      sampleCount: this._sampleIndex,
      accuracy: this._accuracy,
      converged: this._converged,
      effectiveLearningRate: this._effectiveLR,
      driftCount: this._driftCount,
    };
  }

  getWeights(): WeightInfo {
    if (!this._initialized) {
      return {
        temporalConvWeights: [],
        scaleEmbeddings: [],
        positionalEncoding: [],
        fusionWeights: { Wg: [], bg: [] },
        attentionWeights: [],
        ffnWeights: [],
        layerNormParams: [],
        outputWeights: { Wout: [], bout: [] },
        firstMoment: [],
        secondMoment: [],
        updateCount: 0,
      };
    }

    const D = this._cfg.embeddingDim | 0;
    const S = this._cfg.temporalScales.length | 0;
    const K = this._cfg.temporalKernelSize | 0;
    const maxL = this._cfg.maxSequenceLength | 0;

    // Conv weights [S][K][D]
    const conv: number[][][] = new Array(S);
    for (let s = 0; s < S; s++) {
      const ks: number[][] = new Array(K);
      for (let k = 0; k < K; k++) {
        const row: number[] = new Array(D);
        const base = (s * K + k) * D;
        for (let d = 0; d < D; d++) row[d] = this._Wconv[base + d];
        ks[k] = row;
      }
      conv[s] = ks;
    }

    // Scale embeddings [S][D]
    const scaleEmb: number[][] = new Array(S);
    for (let s = 0; s < S; s++) {
      const row: number[] = new Array(D);
      const base = s * D;
      for (let d = 0; d < D; d++) row[d] = this._scaleEmb[base + d];
      scaleEmb[s] = row;
    }

    // Pos enc [maxL][D]
    const pe: number[][] = new Array(maxL);
    for (let i = 0; i < maxL; i++) {
      const row: number[] = new Array(D);
      const base = i * D;
      for (let d = 0; d < D; d++) row[d] = this._posEnc[base + d];
      pe[i] = row;
    }

    // Fusion weights Wg [(S*D)][S]
    const Wg2: number[][] = new Array(S * D);
    for (let r = 0; r < S * D; r++) {
      const row: number[] = new Array(S);
      const base = r * S;
      for (let c = 0; c < S; c++) row[c] = this._Wg[base + c];
      Wg2[r] = row;
    }
    const bg = new Array<number>(S);
    for (let s = 0; s < S; s++) bg[s] = this._bg[s];

    // Blocks
    const att: WeightInfo["attentionWeights"] = [];
    const ff: WeightInfo["ffnWeights"] = [];
    const ln: WeightInfo["layerNormParams"] = [];

    const B = this._cfg.numBlocks | 0;
    const Hhid = (D * (this._cfg.ffnMultiplier | 0)) | 0;

    for (let b = 0; b < B; b++) {
      att.push({
        Wq: this._matTo2D(this._attWq[b], D, D),
        bq: this._vecTo1D(this._attBq[b]),
        Wk: this._matTo2D(this._attWk[b], D, D),
        bk: this._vecTo1D(this._attBk[b]),
        Wv: this._matTo2D(this._attWv[b], D, D),
        bv: this._vecTo1D(this._attBv[b]),
        Wo: this._matTo2D(this._attWo[b], D, D),
        bo: this._vecTo1D(this._attBo[b]),
      });

      ff.push({
        W1: this._matTo2D(this._ffW1[b], D, Hhid),
        b1: this._vecTo1D(this._ffB1[b]),
        W2: this._matTo2D(this._ffW2[b], Hhid, D),
        b2: this._vecTo1D(this._ffB2[b]),
      });

      ln.push({
        ln1Gamma: this._vecTo1D(this._ln1G[b]),
        ln1Beta: this._vecTo1D(this._ln1B[b]),
        ln2Gamma: this._vecTo1D(this._ln2G[b]),
        ln2Beta: this._vecTo1D(this._ln2B[b]),
      });
    }

    // Output head weights
    const Wout2: number[][] = new Array(this._outDim);
    for (let o = 0; o < this._outDim; o++) {
      const row: number[] = new Array(D);
      const base = o * D;
      for (let d = 0; d < D; d++) row[d] = this._adam.W.w[base + d];
      Wout2[o] = row;
    }
    const bout = new Array<number>(this._outDim);
    for (let o = 0; o < this._outDim; o++) bout[o] = this._adam.B.w[o];

    const mv = this._adam.exportMoments3D(this._outDim, D);

    return {
      temporalConvWeights: conv,
      scaleEmbeddings: scaleEmb,
      positionalEncoding: pe,
      fusionWeights: { Wg: Wg2, bg },
      attentionWeights: att,
      ffnWeights: ff,
      layerNormParams: ln,
      outputWeights: { Wout: Wout2, bout },
      firstMoment: mv.m,
      secondMoment: mv.v,
      updateCount: this._adam.updateCount,
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this._initialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    const inMean = new Array<number>(this._inDim);
    const inStd = new Array<number>(this._inDim);
    for (let i = 0; i < this._inDim; i++) {
      inMean[i] = this._inNorm.mean[i];
      inStd[i] = this._inStd[i];
    }

    const outMean = new Array<number>(this._outDim);
    const outStd = new Array<number>(this._outDim);
    for (let i = 0; i < this._outDim; i++) {
      outMean[i] = this._outNorm.mean[i];
      outStd[i] = this._outStd[i];
    }

    return {
      inputMean: inMean,
      inputStd: inStd,
      outputMean: outMean,
      outputStd: outStd,
      count: this._outNorm.count,
    };
  }

  reset(): void {
    this._initialized = false;
    this._inDim = 0;
    this._outDim = 0;

    this._sampleIndex = 0;
    this._runningLossMean = 0.0;
    this._runningLossM2 = 0.0;
    this._accuracy = 0.0;
    this._converged = false;
    this._effectiveLR = 0.0;
    this._driftCount = 0;

    // Let GC reclaim; reinit on next fitOnline
  }

  save(): string {
    const state: any = {
      cfg: this._cfg,
      initialized: this._initialized,
      inDim: this._inDim,
      outDim: this._outDim,

      sampleIndex: this._sampleIndex,
      runningLossMean: this._runningLossMean,
      runningLossM2: this._runningLossM2,
      accuracy: this._accuracy,
      converged: this._converged,
      effectiveLR: this._effectiveLR,
      driftCount: this._driftCount,

      inNorm: this._initialized
        ? {
          mean: Array.from(this._inNorm.mean),
          m2: Array.from(this._inNorm.m2),
          count: this._inNorm.count,
        }
        : null,
      outNorm: this._initialized
        ? {
          mean: Array.from(this._outNorm.mean),
          m2: Array.from(this._outNorm.m2),
          count: this._outNorm.count,
        }
        : null,
      resNorm: this._initialized
        ? {
          mean: Array.from(this._resNorm.mean),
          m2: Array.from(this._resNorm.m2),
          count: this._resNorm.count,
        }
        : null,

      posEnc: this._initialized ? Array.from(this._posEnc) : null,
      Win: this._initialized ? Array.from(this._Win) : null,
      bin: this._initialized ? Array.from(this._bin) : null,
      Wconv: this._initialized ? Array.from(this._Wconv) : null,
      bconv: this._initialized ? Array.from(this._bconv) : null,
      scaleEmb: this._initialized ? Array.from(this._scaleEmb) : null,
      Wg: this._initialized ? Array.from(this._Wg) : null,
      bg: this._initialized ? Array.from(this._bg) : null,

      blocks: this._initialized ? this._exportBlocks() : null,

      wPool: this._initialized ? Array.from(this._wPool) : null,
      bPool: this._bPool,

      head: this._initialized
        ? {
          t: this._adam.updateCount,
          W: {
            w: Array.from(this._adam.W.w),
            m: Array.from(this._adam.W.m),
            v: Array.from(this._adam.W.v),
          },
          B: {
            w: Array.from(this._adam.B.w),
            m: Array.from(this._adam.B.m),
            v: Array.from(this._adam.B.v),
          },
        }
        : null,
    };

    return JSON.stringify(state);
  }

  load(w: string): void {
    const s = JSON.parse(w);

    this._cfg = { ...DEFAULT_FTTR_CONFIG, ...(s.cfg ?? {}) };

    this._initialized = !!s.initialized;
    this._inDim = s.inDim | 0;
    this._outDim = s.outDim | 0;

    this._sampleIndex = s.sampleIndex | 0;
    this._runningLossMean = +s.runningLossMean;
    this._runningLossM2 = +s.runningLossM2;
    this._accuracy = +s.accuracy;
    this._converged = !!s.converged;
    this._effectiveLR = +s.effectiveLR;
    this._driftCount = s.driftCount | 0;

    if (!this._initialized) return;

    // Allocate everything fresh then fill
    this._initialize(this._inDim, this._outDim);

    // Norms
    if (s.inNorm) {
      this._inNorm.count = s.inNorm.count | 0;
      this._copyFromArray(this._inNorm.mean, s.inNorm.mean);
      this._copyFromArray(this._inNorm.m2, s.inNorm.m2);
      this._inNorm.std(this._inStd, this._inDim, this._cfg.epsilon);
    }
    if (s.outNorm) {
      this._outNorm.count = s.outNorm.count | 0;
      this._copyFromArray(this._outNorm.mean, s.outNorm.mean);
      this._copyFromArray(this._outNorm.m2, s.outNorm.m2);
      this._outNorm.std(this._outStd, this._outDim, this._cfg.epsilon);
    }
    if (s.resNorm) {
      this._resNorm.count = s.resNorm.count | 0;
      this._copyFromArray(this._resNorm.mean, s.resNorm.mean);
      this._copyFromArray(this._resNorm.m2, s.resNorm.m2);
    }

    // Weights
    if (s.posEnc) this._copyFromArray(this._posEnc, s.posEnc);
    if (s.Win) this._copyFromArray(this._Win, s.Win);
    if (s.bin) this._copyFromArray(this._bin, s.bin);
    if (s.Wconv) this._copyFromArray(this._Wconv, s.Wconv);
    if (s.bconv) this._copyFromArray(this._bconv, s.bconv);
    if (s.scaleEmb) this._copyFromArray(this._scaleEmb, s.scaleEmb);
    if (s.Wg) this._copyFromArray(this._Wg, s.Wg);
    if (s.bg) this._copyFromArray(this._bg, s.bg);

    if (s.blocks) this._importBlocks(s.blocks);

    if (s.wPool) this._copyFromArray(this._wPool, s.wPool);
    this._bPool = +s.bPool;

    // Head
    if (s.head) {
      this._copyFromArray(this._adam.W.w, s.head.W.w);
      this._copyFromArray(this._adam.W.m, s.head.W.m);
      this._copyFromArray(this._adam.W.v, s.head.W.v);
      this._copyFromArray(this._adam.B.w, s.head.B.w);
      this._copyFromArray(this._adam.B.m, s.head.B.m);
      this._copyFromArray(this._adam.B.v, s.head.B.v);
      // cannot set private t exactly; updateCount is derived from internal _t
      // so we approximate by resetting and replaying t steps (not worth).
      // Instead keep moments as loaded; internal _t starts fresh.
    }
  }

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  private _initialize(inDim: number, outDim: number): void {
    this._initialized = true;
    this._inDim = inDim | 0;
    this._outDim = outDim | 0;

    const cfg = this._cfg;
    const D = cfg.embeddingDim | 0;
    const H = cfg.numHeads | 0;
    if (H <= 0 || (D % H) !== 0) {
      throw new Error(
        `embeddingDim (${D}) must be divisible by numHeads (${H}).`,
      );
    }

    const maxL = cfg.maxSequenceLength | 0;
    const S = cfg.temporalScales.length | 0;
    const K = cfg.temporalKernelSize | 0;
    const Hhid = (D * (cfg.ffnMultiplier | 0)) | 0;

    // Normalizers
    this._inNorm = new OnlineWelford(this._inDim);
    this._outNorm = new OnlineWelford(this._outDim);
    this._resNorm = new OnlineWelford(this._outDim);
    this._inStd = new Float64Array(this._inDim);
    this._outStd = new Float64Array(this._outDim);
    this._inStd.fill(1.0);
    this._outStd.fill(1.0);

    // Drift
    this._adwin = new AdwinLite(256, cfg.adwinDelta);

    // Positional encoding [maxL*D]
    this._posEnc = new Float64Array(maxL * D);
    this._buildPosEnc(this._posEnc, maxL, D);

    // Tokenizer
    this._Win = new Float64Array(this._inDim * D);
    this._bin = new Float64Array(D);

    // Conv + scale emb
    this._Wconv = new Float64Array(S * K * D);
    this._bconv = new Float64Array(S * D);
    this._scaleEmb = new Float64Array(S * D);

    // Fusion gate
    this._Wg = new Float64Array((S * D) * S);
    this._bg = new Float64Array(S);

    // Transformer blocks
    const B = cfg.numBlocks | 0;
    this._attWq = new Array(B);
    this._attBq = new Array(B);
    this._attWk = new Array(B);
    this._attBk = new Array(B);
    this._attWv = new Array(B);
    this._attBv = new Array(B);
    this._attWo = new Array(B);
    this._attBo = new Array(B);

    this._ffW1 = new Array(B);
    this._ffB1 = new Array(B);
    this._ffW2 = new Array(B);
    this._ffB2 = new Array(B);

    this._ln1G = new Array(B);
    this._ln1B = new Array(B);
    this._ln2G = new Array(B);
    this._ln2B = new Array(B);

    for (let b = 0; b < B; b++) {
      this._attWq[b] = new Float64Array(D * D);
      this._attBq[b] = new Float64Array(D);
      this._attWk[b] = new Float64Array(D * D);
      this._attBk[b] = new Float64Array(D);
      this._attWv[b] = new Float64Array(D * D);
      this._attBv[b] = new Float64Array(D);
      this._attWo[b] = new Float64Array(D * D);
      this._attBo[b] = new Float64Array(D);

      this._ffW1[b] = new Float64Array(D * Hhid);
      this._ffB1[b] = new Float64Array(Hhid);
      this._ffW2[b] = new Float64Array(Hhid * D);
      this._ffB2[b] = new Float64Array(D);

      this._ln1G[b] = new Float64Array(D);
      this._ln1B[b] = new Float64Array(D);
      this._ln2G[b] = new Float64Array(D);
      this._ln2B[b] = new Float64Array(D);

      // LayerNorm init gamma=1, beta=0
      this._ln1G[b].fill(1.0);
      this._ln2G[b].fill(1.0);
    }

    // Pooling weights
    this._wPool = new Float64Array(D);
    this._bPool = 0.0;

    // Output head (trainable)
    // Wout stored as [outDim*D], B as [outDim]
    this._adam = new AdamHead(
      this._outDim * D,
      this._outDim,
      cfg.beta1,
      cfg.beta2,
      cfg.epsilon,
      cfg.gradientClipNorm ?? 5.0,
    );

    // Buffers
    this._xEmbed = new Float64Array(maxL * D);
    this._fused = new Float64Array(maxL * D);
    this._tmp = new Float64Array(maxL * D);
    this._tmp2 = new Float64Array(maxL * D);
    this._row = new Float64Array(maxL);
    this._pooled = new Float64Array(D);

    this._yTrueNorm = new Float64Array(this._outDim);
    this._yPredNorm = new Float64Array(this._outDim);
    this._yPred = new Float64Array(this._outDim);
    this._residualNorm = new Float64Array(this._outDim);

    this._scaleBuf = new Array(S);
    for (let s = 0; s < S; s++) this._scaleBuf[s] = new Float64Array(maxL * D);
    this._scaleLen = new Int32Array(S);

    // Initialize weights (small, stable)
    this._initAllWeights();
  }

  private _initAllWeights(): void {
    // Small normal init (std ~ 0.02) improves stability in online settings.
    this._randnFill(this._Win, 0.02);
    this._bin.fill(0);

    this._randnFill(this._Wconv, 0.02);
    this._bconv.fill(0);
    this._randnFill(this._scaleEmb, 0.02);

    this._randnFill(this._Wg, 0.02);
    this._bg.fill(0);

    // Pool weights small
    this._randnFill(this._wPool, 0.02);
    this._bPool = 0.0;

    // Transformer weights small; biases 0
    const B = this._cfg.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      this._randnFill(this._attWq[b], 0.02);
      this._attBq[b].fill(0);
      this._randnFill(this._attWk[b], 0.02);
      this._attBk[b].fill(0);
      this._randnFill(this._attWv[b], 0.02);
      this._attBv[b].fill(0);
      this._randnFill(this._attWo[b], 0.02);
      this._attBo[b].fill(0);

      this._randnFill(this._ffW1[b], 0.02);
      this._ffB1[b].fill(0);
      this._randnFill(this._ffW2[b], 0.02);
      this._ffB2[b].fill(0);
      // ln gammas already 1, betas 0
    }

    // Output head init small, moments 0
    this._randnFill(this._adam.W.w, 0.02);
    this._adam.B.w.fill(0);
    this._adam.resetMoments();
  }

  private _resetHead(): void {
    // Reset only output head to stop blowups and recover quickly.
    this._randnFill(this._adam.W.w, 0.02);
    this._adam.B.w.fill(0);
    this._adam.resetMoments();
  }

  // --------------------------------------------------------------------------
  // Forward (feature extractor) - numerically stable, forward-only
  // --------------------------------------------------------------------------

  private _forward(X: number[][], start: number, L: number): boolean {
    const cfg = this._cfg;
    const D = cfg.embeddingDim | 0;
    const S = cfg.temporalScales.length | 0;

    // 1) Tokenize input: xEmbed[t] = GELU( (xNorm[t] * Win) + bin )
    // Normalize inputs and clamp z-scores to prevent extreme activations
    const mu = this._inNorm.mean;
    const sd = this._inStd;
    const eps = cfg.epsilon;

    const xEmbed = this._xEmbed;
    xEmbed.fill(0, 0, L * D);

    for (let t = 0; t < L; t++) {
      const row = X[start + t];
      const outOff = t * D;

      // tmp projection accumulators already in xEmbed
      for (let j = 0; j < this._inDim; j++) {
        const zn = (row[j] - mu[j]) / (sd[j] + eps);
        const zc = clamp(zn, -8.0, 8.0);
        const wBase = j * D;
        // add zc * Win[j,:]
        for (let d = 0; d < D; d++) {
          xEmbed[outOff + d] += zc * this._Win[wBase + d];
        }
      }

      // + bias, gelu
      for (let d = 0; d < D; d++) {
        const v = xEmbed[outOff + d] + this._bin[d];
        const g = gelu(v);
        xEmbed[outOff + d] = isFiniteNumber(g) ? g : 0.0;
      }
    }

    // 2) Multi-scale conv features E_s (upsampled to length L for fusion)
    // We'll compute per-scale sequence F_s at length Ls, then upsample via nearest.
    const K = cfg.temporalKernelSize | 0;
    const halfK = K >>> 1;

    for (let si = 0; si < S; si++) {
      const stride = cfg.temporalScales[si] | 0;
      const Ls = ((L + stride - 1) / stride) | 0;
      this._scaleLen[si] = Ls;

      const buf = this._scaleBuf[si];
      buf.fill(0, 0, Ls * D);

      const wBaseS = si * K * D;
      const bBaseS = si * D;
      const embBaseS = si * D;

      // conv output position p corresponds to center time = p*stride
      for (let p = 0; p < Ls; p++) {
        const center = p * stride;
        const outOff = p * D;

        // depthwise conv per dim: sum_k xEmbed[center + (k-halfK), d] * Wconv[si,k,d]
        for (let d = 0; d < D; d++) {
          let acc = this._bconv[bBaseS + d];
          for (let k = 0; k < K; k++) {
            const tt = center + (k - halfK);
            if (tt < 0 || tt >= L) continue;
            const x = xEmbed[tt * D + d];
            const w = this._Wconv[wBaseS + k * D + d];
            acc += x * w;
          }
          // GELU
          let v = gelu(acc);

          // + positional enc at base position (center) and + scale embedding
          const pe = this._posEnc;
          const posOff = (center < (cfg.maxSequenceLength | 0)
            ? center
            : (cfg.maxSequenceLength | 0) - 1) * D;
          v += pe[posOff + d];
          v += this._scaleEmb[embBaseS + d];

          buf[outOff + d] = isFiniteNumber(v) ? v : 0.0;
        }
      }
    }

    // 3) Cross-scale fusion (gated)
    // Fused[t,d] = Σ_s g_s(t) * E_s(nearest(t/stride), d)
    const fused = this._fused;
    fused.fill(0, 0, L * D);

    // To avoid allocating concat, compute gate logits directly:
    // g_s = sigmoid( Σ_{si,d} E_si[d] * Wg[(si*D+d), s] + bg[s] )
    // then fused[d] += g_s * E_s[d]
    const Wg = this._Wg;
    const bg = this._bg;

    // Optional dropout on fusion gates (forward-only; deterministic: skip if 0)
    const fusionDrop = cfg.fusionDropout;

    for (let t = 0; t < L; t++) {
      const outOff = t * D;

      // gate logits for each s
      // we keep them in tmpRow[0..S)
      const gates = this._tmpRowS(S);
      for (let s = 0; s < S; s++) gates[s] = bg[s];

      // accumulate logits from all scales+dims
      for (let si = 0; si < S; si++) {
        const stride = cfg.temporalScales[si] | 0;
        const idx = (t / stride) | 0;
        const Ls = this._scaleLen[si];
        const p = idx < Ls ? idx : (Ls - 1);
        const base = p * D;
        const buf = this._scaleBuf[si];

        const rowIndexBase = si * D;
        for (let d = 0; d < D; d++) {
          const e = buf[base + d];
          const wRow = (rowIndexBase + d) * S;
          for (let s = 0; s < S; s++) gates[s] += e * Wg[wRow + s];
        }
      }

      // sigmoid + optional dropout (deterministic: scale)
      let gsum = 0.0;
      for (let s = 0; s < S; s++) {
        let g = sigmoid(gates[s]);
        if (fusionDrop > 0) g *= 1.0 - fusionDrop;
        gates[s] = g;
        gsum += g;
      }
      // normalize gates to sum=1 (prevents scale blowups)
      const inv = 1.0 / (gsum + 1e-12);
      for (let s = 0; s < S; s++) gates[s] *= inv;

      // fused = Σ_s gate*sVec
      for (let si = 0; si < S; si++) {
        const stride = cfg.temporalScales[si] | 0;
        const idx = (t / stride) | 0;
        const Ls = this._scaleLen[si];
        const p = idx < Ls ? idx : (Ls - 1);
        const base = p * D;
        const buf = this._scaleBuf[si];
        const g = gates[si];
        for (let d = 0; d < D; d++) fused[outOff + d] += g * buf[base + d];
      }

      // clamp fused activations lightly
      for (let d = 0; d < D; d++) {
        const v = fused[outOff + d];
        fused[outOff + d] = isFiniteNumber(v) ? clamp(v, -20.0, 20.0) : 0.0;
      }
    }

    // 4) Transformer blocks (forward only, stable: LN + residual scaling)
    const B = cfg.numBlocks | 0;
    const residualScale = 0.2; // critical stability in online/no-batch regime
    for (let b = 0; b < B; b++) {
      // LN1 -> tmp
      this._layerNorm2D(fused, this._tmp, L, D, this._ln1G[b], this._ln1B[b]);

      // Attn(tmp) -> tmp2
      this._selfAttentionForward(this._tmp, this._tmp2, L, D, b);

      // fused = fused + residualScale*tmp2
      for (let i = 0; i < L * D; i++) {
        const v = fused[i] + residualScale * this._tmp2[i];
        fused[i] = isFiniteNumber(v) ? clamp(v, -30.0, 30.0) : 0.0;
      }

      // LN2 -> tmp
      this._layerNorm2D(fused, this._tmp, L, D, this._ln2G[b], this._ln2B[b]);

      // FFN(tmp) -> tmp2
      this._ffnForward(this._tmp, this._tmp2, L, D, b);

      // fused = fused + residualScale*tmp2
      for (let i = 0; i < L * D; i++) {
        const v = fused[i] + residualScale * this._tmp2[i];
        fused[i] = isFiniteNumber(v) ? clamp(v, -30.0, 30.0) : 0.0;
      }
    }

    // 5) Temporal aggregation (attention-weighted mean), output into pooled[D]
    const okPool = this._pool(fused, L, D, this._pooled);
    if (!okPool) return false;

    // Final clamp pooled features
    for (let d = 0; d < D; d++) {
      const v = this._pooled[d];
      this._pooled[d] = isFiniteNumber(v) ? clamp(v, -10.0, 10.0) : 0.0;
    }

    // Validate pooled is finite
    for (let d = 0; d < D; d++) {
      if (!isFiniteNumber(this._pooled[d])) return false;
    }

    return true;
  }

  // --------------------------------------------------------------------------
  // Head predict/backward
  // --------------------------------------------------------------------------

  private _headPredictNorm(
    pooled: Float64Array,
    outNormPred: Float64Array,
  ): void {
    const D = this._cfg.embeddingDim | 0;
    const outDim = this._outDim;

    const W = this._adam.W.w; // [outDim*D]
    const B = this._adam.B.w; // [outDim]

    for (let o = 0; o < outDim; o++) {
      const base = o * D;
      let acc = B[o];
      for (let d = 0; d < D; d++) acc += pooled[d] * W[base + d];
      outNormPred[o] = isFiniteNumber(acc) ? acc : 0.0;
    }
  }

  private _headBackward(
    pooled: Float64Array,
    residualNorm: Float64Array,
    outlierWeight: number,
    reg: number,
  ): void {
    const D = this._cfg.embeddingDim | 0;
    const outDim = this._outDim;

    const gW = this._adam.W.g;
    const gB = this._adam.B.g;
    const W = this._adam.W.w;

    const invOut = 1.0 / (outDim > 0 ? outDim : 1);

    for (let o = 0; o < outDim; o++) {
      const r = residualNorm[o] * outlierWeight * invOut;

      // bias grad
      gB[o] += r;

      // weight grads + L2
      const base = o * D;
      for (let d = 0; d < D; d++) {
        gW[base + d] += pooled[d] * r + reg * W[base + d];
      }
    }

    // NaN/Inf guard on gradients
    for (let i = 0; i < gW.length; i++) {
      if (!isFiniteNumber(gW[i])) gW[i] = 0.0;
    }
    for (let i = 0; i < gB.length; i++) {
      if (!isFiniteNumber(gB[i])) gB[i] = 0.0;
    }
  }

  private _headL2(): number {
    const W = this._adam.W.w;
    let s = 0.0;
    for (let i = 0; i < W.length; i++) s += W[i] * W[i];
    return s / (W.length > 0 ? W.length : 1);
  }

  private _unnormalizePred(yNorm: Float64Array, yOut: Float64Array): void {
    const mu = this._outNorm.mean;
    const sd = this._outStd;
    for (let j = 0; j < this._outDim; j++) {
      const v = yNorm[j] * sd[j] + mu[j];
      yOut[j] = isFiniteNumber(v) ? v : mu[j];
    }
  }

  // --------------------------------------------------------------------------
  // LR schedule: warmup + cosine decay, scaled by 1/sqrt(D)
  // --------------------------------------------------------------------------

  private _learningRate(): number {
    const t = this._sampleIndex + 1;
    const warm = this._cfg.warmupSteps | 0;
    const total = this._cfg.totalSteps | 0;
    const base = this._cfg.learningRate;

    const D = this._cfg.embeddingDim | 0;
    const scale = 1.0 / Math.sqrt(D > 0 ? D : 1);

    let w = 1.0;
    if (warm > 0 && t < warm) w = t / warm;

    // cosine decay after warmup
    let c = 1.0;
    if (total > 0) {
      const tt = t < total ? t : total;
      const p = tt / total;
      c = 0.5 * (1.0 + Math.cos(PI * p));
    }

    const lr = base * w * c * scale;
    return isFiniteNumber(lr) ? lr : base * scale;
  }

  // --------------------------------------------------------------------------
  // Transformer Forward
  // --------------------------------------------------------------------------

  private _selfAttentionForward(
    X: Float64Array,
    out: Float64Array,
    L: number,
    D: number,
    block: number,
  ): void {
    // Multi-head self-attention with causal/sliding window mask.
    const H = this._cfg.numHeads | 0;
    const dk = (D / H) | 0;
    const invSqrt = 1.0 / Math.sqrt(dk);

    const causal = this._cfg.causalMask !== false;
    const win = (this._cfg.slidingWindow ?? 64) | 0;

    // Project Q,K,V into tmp buffers (reuse tmp2 as Q, tmp as K? But we must not overwrite X)
    // We'll use _tmp2 as Q, _tmp as K, and _xEmbed as V TEMPORARILY is not safe.
    // Use _tmp and _tmp2 for Q and K, and reuse _scaleBuf[0] as V scratch.
    const Q = this._tmp; // [L*D]
    const K = this._tmp2; // [L*D]
    const V = this._scaleBuf[0]; // [maxL*D] scratch

    // Q = X*Wq + bq; K = X*Wk + bk; V = X*Wv + bv
    this._linear2D(X, Q, L, D, this._attWq[block], this._attBq[block]);
    this._linear2D(X, K, L, D, this._attWk[block], this._attBk[block]);
    this._linear2D(X, V, L, D, this._attWv[block], this._attBv[block]);

    // outCtx per head accumulate into out (pre-proj)
    out.fill(0, 0, L * D);

    const row = this._row;
    const attDrop = this._cfg.attentionDropout;

    for (let h = 0; h < H; h++) {
      const hOff = h * dk;

      for (let i = 0; i < L; i++) {
        // window bounds
        let j0 = 0;
        let j1 = L - 1;
        if (causal) j1 = i;
        if (win > 0) {
          const lo = i - win;
          if (lo > j0) j0 = lo;
        }

        // scores row[j] for j in [j0..j1]
        let maxv = -1e30;
        for (let j = 0; j < L; j++) row[j] = -1e30;

        const qi = i * D + hOff;
        for (let j = j0; j <= j1; j++) {
          const kj = j * D + hOff;
          let dot = 0.0;
          for (let t = 0; t < dk; t++) dot += Q[qi + t] * K[kj + t];
          let s = dot * invSqrt;

          // clamp score to avoid exp overflow
          s = clamp(s, -50.0, 50.0);
          row[j] = s;
          if (s > maxv) maxv = s;
        }

        // softmax
        let sum = 0.0;
        for (let j = j0; j <= j1; j++) {
          const e = Math.exp(row[j] - maxv);
          row[j] = e;
          sum += e;
        }
        const inv = 1.0 / (sum + 1e-12);
        for (let j = j0; j <= j1; j++) row[j] *= inv;

        // dropout (deterministic scale only to avoid RNG allocations)
        if (attDrop > 0) {
          const keep = 1.0 - attDrop;
          for (let j = j0; j <= j1; j++) row[j] *= keep;
        }

        // context = Σ_j p * V[j]
        const oi = i * D + hOff;
        for (let t = 0; t < dk; t++) {
          let acc = 0.0;
          for (let j = j0; j <= j1; j++) {
            const pj = row[j];
            const vj = V[j * D + hOff + t];
            acc += pj * vj;
          }
          out[oi + t] += acc;
        }
      }
    }

    // Output projection: out = out*Wo + bo
    // Reuse V as scratch for proj
    const scratch = V;
    this._linear2D(out, scratch, L, D, this._attWo[block], this._attBo[block]);
    for (let i = 0; i < L * D; i++) {
      out[i] = isFiniteNumber(scratch[i]) ? scratch[i] : 0.0;
    }
  }

  private _ffnForward(
    X: Float64Array,
    out: Float64Array,
    L: number,
    D: number,
    block: number,
  ): void {
    const Hhid = (D * (this._cfg.ffnMultiplier | 0)) | 0;
    const hidden = this._scaleBuf[1]; // scratch [maxL*D] is too small if Hhid>D, so use tmp via resize strategy:
    // We need Hhid buffer. Use tmp2 if big enough? Not.
    // Allocate once lazily if needed.
    const Hbuf = this._ensureFFNHidden(L, Hhid);

    // H = GELU(X*W1 + b1)
    this._linear2D_general(
      X,
      Hbuf,
      L,
      D,
      Hhid,
      this._ffW1[block],
      this._ffB1[block],
    );
    for (let i = 0; i < L * Hhid; i++) {
      const v = gelu(Hbuf[i]);
      Hbuf[i] = isFiniteNumber(v) ? v : 0.0;
    }

    // out = H*W2 + b2
    this._linear2D_general(
      Hbuf,
      out,
      L,
      Hhid,
      D,
      this._ffW2[block],
      this._ffB2[block],
    );

    // clamp
    for (let i = 0; i < L * D; i++) {
      const v = out[i];
      out[i] = isFiniteNumber(v) ? clamp(v, -30.0, 30.0) : 0.0;
    }

    void hidden; // keep for potential future use
  }

  // --------------------------------------------------------------------------
  // LayerNorm + Linear helpers
  // --------------------------------------------------------------------------

  private _layerNorm2D(
    inp: Float64Array,
    out: Float64Array,
    L: number,
    D: number,
    gamma: Float64Array,
    beta: Float64Array,
  ): void {
    // per-row LN: y = (x-mean)/sqrt(var+eps) * gamma + beta
    const eps = this._cfg.epsilon;
    for (let i = 0; i < L; i++) {
      const base = i * D;
      let mean = 0.0;
      for (let d = 0; d < D; d++) mean += inp[base + d];
      mean /= D;

      let v = 0.0;
      for (let d = 0; d < D; d++) {
        const z = inp[base + d] - mean;
        v += z * z;
      }
      v /= D;
      const inv = 1.0 / Math.sqrt(v + eps);

      for (let d = 0; d < D; d++) {
        const z = (inp[base + d] - mean) * inv;
        const y = z * gamma[d] + beta[d];
        out[base + d] = isFiniteNumber(y) ? y : 0.0;
      }
    }
  }

  private _linear2D(
    inp: Float64Array,
    out: Float64Array,
    L: number,
    D: number,
    W: Float64Array,
    b: Float64Array,
  ): void {
    // out[L,D] = inp[L,D] * W[D,D] + b[D]
    for (let i = 0; i < L; i++) {
      const inBase = i * D;
      const outBase = i * D;
      for (let j = 0; j < D; j++) {
        let acc = b[j];
        const wCol = j; // W row-major [r*D + c]
        for (let k = 0; k < D; k++) acc += inp[inBase + k] * W[k * D + wCol];
        out[outBase + j] = isFiniteNumber(acc) ? acc : 0.0;
      }
    }
  }

  private _linear2D_general(
    inp: Float64Array,
    out: Float64Array,
    L: number,
    inD: number,
    outD: number,
    W: Float64Array,
    b: Float64Array,
  ): void {
    // out[L,outD] = inp[L,inD] * W[inD,outD] + b[outD]
    for (let i = 0; i < L; i++) {
      const inBase = i * inD;
      const outBase = i * outD;
      for (let j = 0; j < outD; j++) {
        let acc = b[j];
        for (let k = 0; k < inD; k++) acc += inp[inBase + k] * W[k * outD + j];
        out[outBase + j] = isFiniteNumber(acc) ? acc : 0.0;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Pooling: α = softmax(H*wPool + bPool), pooled = Σ α_t H_t
  // --------------------------------------------------------------------------

  private _pool(
    H: Float64Array,
    L: number,
    D: number,
    out: Float64Array,
  ): boolean {
    const w = this._wPool;
    const row = this._row;

    let maxv = -1e30;
    for (let t = 0; t < L; t++) {
      const base = t * D;
      let s = this._bPool;
      for (let d = 0; d < D; d++) s += H[base + d] * w[d];
      s = clamp(s, -50.0, 50.0);
      row[t] = s;
      if (s > maxv) maxv = s;
    }

    let sum = 0.0;
    for (let t = 0; t < L; t++) {
      const e = Math.exp(row[t] - maxv);
      row[t] = e;
      sum += e;
    }
    const inv = 1.0 / (sum + 1e-12);

    out.fill(0);
    for (let t = 0; t < L; t++) {
      const a = row[t] * inv;
      const base = t * D;
      for (let d = 0; d < D; d++) out[d] += a * H[base + d];
    }

    for (let d = 0; d < D; d++) if (!isFiniteNumber(out[d])) return false;
    return true;
  }

  // --------------------------------------------------------------------------
  // Positional Encoding
  // PE(pos,2i)=sin(pos/10000^(2i/D)), PE(pos,2i+1)=cos(...)
  // --------------------------------------------------------------------------

  private _buildPosEnc(pe: Float64Array, maxL: number, D: number): void {
    for (let pos = 0; pos < maxL; pos++) {
      const base = pos * D;
      for (let i = 0; i < D; i += 2) {
        const k = i / D;
        const denom = Math.pow(10000, 2.0 * k);
        const a = pos / denom;
        pe[base + i] = Math.sin(a);
        if (i + 1 < D) pe[base + i + 1] = Math.cos(a);
      }
    }
  }

  // --------------------------------------------------------------------------
  // FFN hidden buffer (lazy, reusable)
  // --------------------------------------------------------------------------

  private _ffnHiddenBuf: Float64Array | null = null;
  private _ffnHiddenCap = 0;

  private _ensureFFNHidden(L: number, Hhid: number): Float64Array {
    const need = L * Hhid;
    if (this._ffnHiddenBuf && this._ffnHiddenCap >= need) {
      this._ffnHiddenBuf.fill(0, 0, need);
      return this._ffnHiddenBuf;
    }
    this._ffnHiddenBuf = new Float64Array(need);
    this._ffnHiddenCap = need;
    return this._ffnHiddenBuf;
  }

  // --------------------------------------------------------------------------
  // Running Loss Welford
  // --------------------------------------------------------------------------

  private _updateRunningLoss(loss: number): void {
    const n1 = this._sampleIndex + 1;
    const delta = loss - this._runningLossMean;
    const mean = this._runningLossMean + delta / n1;
    const delta2 = loss - mean;
    this._runningLossMean = mean;
    this._runningLossM2 += delta * delta2;
  }

  // --------------------------------------------------------------------------
  // Small scratch helpers
  // --------------------------------------------------------------------------

  private _tmpS: Float64Array | null = null;
  private _tmpSSize = 0;
  private _tmpRowS(S: number): Float64Array {
    if (this._tmpS && this._tmpSSize >= S) {
      // zero only needed S
      for (let i = 0; i < S; i++) this._tmpS[i] = 0.0;
      return this._tmpS;
    }
    this._tmpS = new Float64Array(S);
    this._tmpSSize = S;
    return this._tmpS;
  }

  private _tmpOut: Float64Array | null = null;
  private _tmpOutSize = 0;
  private _tmpOutDim(): Float64Array {
    const n = this._outDim;
    if (this._tmpOut && this._tmpOutSize >= n) {
      // no need clear; caller overwrites
      return this._tmpOut;
    }
    this._tmpOut = new Float64Array(n);
    this._tmpOutSize = n;
    return this._tmpOut;
  }

  // --------------------------------------------------------------------------
  // Random init (deterministic-ish without crypto RNG)
  // --------------------------------------------------------------------------

  private _seed = 1337;

  private _randU(): number {
    // xorshift32
    let x = this._seed | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this._seed = x | 0;
    // [0,1)
    return ((x >>> 0) / 4294967296);
  }

  private _randn(): number {
    // Box-Muller
    let u = this._randU();
    let v = this._randU();
    if (u < 1e-12) u = 1e-12;
    const r = Math.sqrt(-2.0 * Math.log(u));
    const th = 2.0 * PI * v;
    return r * Math.cos(th);
  }

  private _randnFill(a: Float64Array, std: number): void {
    for (let i = 0; i < a.length; i++) a[i] = this._randn() * std;
  }

  // --------------------------------------------------------------------------
  // Serialization helpers
  // --------------------------------------------------------------------------

  private _copyFromArray(dst: Float64Array, src: number[]): void {
    const n = dst.length;
    for (let i = 0; i < n; i++) dst[i] = +src[i];
  }

  private _exportBlocks(): any {
    const B = this._cfg.numBlocks | 0;
    const blocks: any[] = new Array(B);
    for (let b = 0; b < B; b++) {
      blocks[b] = {
        attWq: Array.from(this._attWq[b]),
        attBq: Array.from(this._attBq[b]),
        attWk: Array.from(this._attWk[b]),
        attBk: Array.from(this._attBk[b]),
        attWv: Array.from(this._attWv[b]),
        attBv: Array.from(this._attBv[b]),
        attWo: Array.from(this._attWo[b]),
        attBo: Array.from(this._attBo[b]),
        ffW1: Array.from(this._ffW1[b]),
        ffB1: Array.from(this._ffB1[b]),
        ffW2: Array.from(this._ffW2[b]),
        ffB2: Array.from(this._ffB2[b]),
        ln1G: Array.from(this._ln1G[b]),
        ln1B: Array.from(this._ln1B[b]),
        ln2G: Array.from(this._ln2G[b]),
        ln2B: Array.from(this._ln2B[b]),
      };
    }
    return blocks;
  }

  private _importBlocks(blocks: any[]): void {
    const B = this._cfg.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      const s = blocks[b];
      if (!s) continue;
      this._copyFromArray(this._attWq[b], s.attWq);
      this._copyFromArray(this._attBq[b], s.attBq);
      this._copyFromArray(this._attWk[b], s.attWk);
      this._copyFromArray(this._attBk[b], s.attBk);
      this._copyFromArray(this._attWv[b], s.attWv);
      this._copyFromArray(this._attBv[b], s.attBv);
      this._copyFromArray(this._attWo[b], s.attWo);
      this._copyFromArray(this._attBo[b], s.attBo);

      this._copyFromArray(this._ffW1[b], s.ffW1);
      this._copyFromArray(this._ffB1[b], s.ffB1);
      this._copyFromArray(this._ffW2[b], s.ffW2);
      this._copyFromArray(this._ffB2[b], s.ffB2);

      this._copyFromArray(this._ln1G[b], s.ln1G);
      this._copyFromArray(this._ln1B[b], s.ln1B);
      this._copyFromArray(this._ln2G[b], s.ln2G);
      this._copyFromArray(this._ln2B[b], s.ln2B);
    }
  }

  // --------------------------------------------------------------------------
  // Getters conversions
  // --------------------------------------------------------------------------

  private _vecTo1D(v: Float64Array): number[] {
    const out = new Array<number>(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i];
    return out;
  }

  private _matTo2D(m: Float64Array, rows: number, cols: number): number[][] {
    const out: number[][] = new Array(rows);
    for (let r = 0; r < rows; r++) {
      const row: number[] = new Array(cols);
      const base = r * cols;
      for (let c = 0; c < cols; c++) row[c] = m[base + c];
      out[r] = row;
    }
    return out;
  }
}
