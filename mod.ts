/**
 * @module FusionTemporalTransformerRegression
 *
 * Fusion Temporal Transformer (FTT) for multivariate regression with incremental online learning.
 *
 * Core features:
 * - Multi-scale temporal depthwise convolution (stride = temporalScales) + scale embeddings + sinusoidal positional encoding
 * - Gated cross-scale fusion (learned sigmoid gates from concatenated multi-scale features)
 * - Transformer blocks: LayerNorm → Multi-Head Causal Self-Attention → Residual → LayerNorm → FFN(GELU) → Residual
 * - Attention-weighted temporal pooling → output dense
 * - Online Adam optimizer with warmup + cosine decay
 * - Welford online z-score normalization (inputs + outputs)
 * - L2 regularization (weight decay on selected parameters)
 * - Outlier downweighting (based on normalized residual)
 * - ADWIN drift detection on error stream
 *
 * Numerical notes:
 * - Positional encoding: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))
 * - Adam update:
 *   m = β1 m + (1-β1) g
 *   v = β2 v + (1-β2) g^2
 *   W -= lr * (m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ε)
 *
 * Performance notes:
 * - All compute tensors are Float64Array (row-major 2D packed into 1D).
 * - Hot paths use for-loops; buffers/caches are preallocated and reused.
 * - Lazy init on first fitOnline() to auto-detect dimensions.
 */

export type FitResult = {
  loss: number;
  gradientNorm: number;
  effectiveLearningRate: number;
  isOutlier: boolean;
  converged: boolean;
  sampleIndex: number;
  driftDetected: boolean;
};

export type SinglePrediction = {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
};

export type PredictionResult = {
  predictions: SinglePrediction[];
  accuracy: number;
  sampleCount: number;
  isModelReady: boolean;
};

export type WeightInfo = {
  temporalConvWeights: number[][][]; // [scale][kernel][embeddingDim]
  scaleEmbeddings: number[][]; // [scale][embeddingDim]
  positionalEncoding: number[][]; // [maxSequenceLength][embeddingDim]
  fusionWeights: number[][]; // [scaleCount*embeddingDim][scaleCount]
  attentionWeights: number[][][]; // [numBlocks][4][embeddingDim*embeddingDim + embeddingDim] (W+b packed)
  ffnWeights: number[][][]; // [numBlocks][2][...packed]
  layerNormParams: number[][][]; // [numBlocks][2][(gamma+beta)+(gamma+beta)]
  outputWeights: number[][]; // [embeddingDim][outputDim] (bias stored separately in export pack)
  firstMoment: number[][][]; // grouped moments (same packing as weights in export groups)
  secondMoment: number[][][]; // grouped moments
  updateCount: number;
};

export type NormalizationStats = {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
  count: number;
};

export type ModelSummary = {
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
};

export type Config = {
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  ffnMultiplier: number;

  attentionDropout: number; // currently unused (default 0.0)
  fusionDropout: number; // currently unused (default 0.0)

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

  // Optional locality control (causal + sliding window). 0 means full causal.
  attentionWindow: number;

  // Optional seed for deterministic init (LCG).
  seed: number;
};

type _ParamTensor = {
  name: string;
  shape: number[];
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
  applyL2: boolean;
};

type _Block = {
  // LN1
  ln1Gamma: Float64Array;
  ln1Beta: Float64Array;

  // Attn projections
  Wq: Float64Array;
  bq: Float64Array;
  Wk: Float64Array;
  bk: Float64Array;
  Wv: Float64Array;
  bv: Float64Array;
  Wo: Float64Array;
  bo: Float64Array;

  // LN2
  ln2Gamma: Float64Array;
  ln2Beta: Float64Array;

  // FFN
  W1: Float64Array;
  b1: Float64Array;
  W2: Float64Array;
  b2: Float64Array;

  // Forward caches (reused each fit; sized maxSeqLen)
  // Block inputs/outputs
  xIn: Float64Array; // [maxSeqLen*d] input to block
  xRes1: Float64Array; // [maxSeqLen*d] after attn residual
  xOut: Float64Array; // [maxSeqLen*d] output of block

  // LN caches
  ln1Mean: Float64Array; // [maxSeqLen]
  ln1InvStd: Float64Array; // [maxSeqLen]
  ln1Xhat: Float64Array; // [maxSeqLen*d]
  ln1Out: Float64Array; // [maxSeqLen*d]

  ln2Mean: Float64Array; // [maxSeqLen]
  ln2InvStd: Float64Array; // [maxSeqLen]
  ln2Xhat: Float64Array; // [maxSeqLen*d]
  ln2Out: Float64Array; // [maxSeqLen*d]

  // Attention caches
  Q: Float64Array; // [maxSeqLen*d]
  K: Float64Array; // [maxSeqLen*d]
  V: Float64Array; // [maxSeqLen*d]
  attnCtx: Float64Array; // [maxSeqLen*d]

  // FFN caches
  ffnPre: Float64Array; // [maxSeqLen*hidden]
  ffnAct: Float64Array; // [maxSeqLen*hidden]
  ffnOut: Float64Array; // [maxSeqLen*d]
};

class _Welford {
  readonly dim: number;
  count: number;
  mean: Float64Array;
  m2: Float64Array;

  constructor(dim: number) {
    this.dim = dim;
    this.count = 0;
    this.mean = new Float64Array(dim);
    this.m2 = new Float64Array(dim);
  }

  reset(): void {
    this.count = 0;
    this.mean.fill(0);
    this.m2.fill(0);
  }

  /**
   * Update per observation vector x[dim].
   * Welford:
   *   n += 1
   *   δ = x - μ
   *   μ += δ/n
   *   M2 += δ*(x - μ)
   */
  update(x: Float64Array, xOffset: number): void {
    const d = this.dim;
    const n1 = this.count + 1;
    this.count = n1;
    const invN = 1.0 / n1;
    const mean = this.mean;
    const m2 = this.m2;
    for (let i = 0; i < d; i++) {
      const xi = x[xOffset + i];
      const delta = xi - mean[i];
      const newMean = mean[i] + delta * invN;
      mean[i] = newMean;
      const delta2 = xi - newMean;
      m2[i] += delta * delta2;
    }
  }

  /**
   * Return std (sqrt(variance)) into outStd[dim], with unbiased variance M2/(n-1).
   * If n < 2, std = 1.
   */
  computeStd(outStd: Float64Array, epsilon: number): void {
    const d = this.dim;
    const n = this.count;
    if (n < 2) {
      for (let i = 0; i < d; i++) outStd[i] = 1.0;
      return;
    }
    const inv = 1.0 / (n - 1);
    const m2 = this.m2;
    for (let i = 0; i < d; i++) {
      const v = m2[i] * inv;
      outStd[i] = Math.sqrt(v + epsilon);
    }
  }

  toJSON(): any {
    return {
      dim: this.dim,
      count: this.count,
      mean: Array.from(this.mean),
      m2: Array.from(this.m2),
    };
  }

  static fromJSON(obj: any): _Welford {
    const d = (obj && typeof obj.dim === "number") ? obj.dim : 0;
    const w = new _Welford(d);
    w.count = (obj && typeof obj.count === "number") ? obj.count : 0;
    const meanArr: number[] = (obj && Array.isArray(obj.mean)) ? obj.mean : [];
    const m2Arr: number[] = (obj && Array.isArray(obj.m2)) ? obj.m2 : [];
    for (let i = 0; i < d; i++) {
      w.mean[i] = i < meanArr.length ? +meanArr[i] : 0;
      w.m2[i] = i < m2Arr.length ? +m2Arr[i] : 0;
    }
    return w;
  }
}

class _ADWIN {
  // Lightweight ADWIN-like drift detector on scalar error stream.
  // Keeps a bounded window; checks mean change with Hoeffding bound.
  readonly delta: number;
  readonly maxWindow: number;
  private _buf: Float64Array;
  private _len: number;
  private _head: number;
  driftCount: number;

  constructor(delta: number, maxWindow: number) {
    this.delta = delta;
    this.maxWindow = maxWindow;
    this._buf = new Float64Array(maxWindow);
    this._len = 0;
    this._head = 0;
    this.driftCount = 0;
  }

  reset(): void {
    this._len = 0;
    this._head = 0;
    this._buf.fill(0);
    this.driftCount = 0;
  }

  update(value: number): boolean {
    // push circular
    if (this._len < this.maxWindow) {
      this._buf[this._len++] = value;
    } else {
      this._buf[this._head] = value;
      this._head++;
      if (this._head >= this.maxWindow) this._head = 0;
    }

    // need enough data
    const n = this._len;
    if (n < 32) return false;

    // materialize window into linear order into scratch sums on-the-fly
    // Scan possible cut points (coarse stride for speed)
    const stride = n > 128 ? 2 : 1;
    let best = 0.0;
    let bestCut = -1;

    // prefix sums for faster scan (O(n))
    // We avoid allocating arrays by computing sums twice with incremental accumulators.
    // First compute total sum:
    let total = 0.0;
    for (let i = 0; i < n; i++) total += this._at(i);
    // Scan cut:
    let leftSum = 0.0;
    for (let cut = 8; cut <= n - 8; cut += stride) {
      leftSum += this._at(cut - 1);
      const n0 = cut;
      const n1 = n - cut;
      if (n0 < 8 || n1 < 8) continue;
      const mean0 = leftSum / n0;
      const mean1 = (total - leftSum) / n1;
      const diff = Math.abs(mean0 - mean1);
      if (diff > best) {
        best = diff;
        bestCut = cut;
      }
    }
    if (bestCut < 0) return false;

    // Hoeffding bound (simplified):
    // eps = sqrt( (1/(2)) * ln(2/delta) * (1/n0 + 1/n1) )
    const n0 = bestCut;
    const n1 = n - bestCut;
    const eps = Math.sqrt(
      0.5 * Math.log(2.0 / this.delta) * (1.0 / n0 + 1.0 / n1),
    );

    if (best >= eps) {
      // drift: shrink to second half (keep recent)
      this._shrink(bestCut);
      this.driftCount++;
      return true;
    }
    return false;
  }

  private _at(i: number): number {
    // i in [0, len)
    if (this._len < this.maxWindow) return this._buf[i];
    const idx = this._head + i;
    return this._buf[idx < this.maxWindow ? idx : (idx - this.maxWindow)];
  }

  private _shrink(cut: number): void {
    // keep [cut..len)
    const n = this._len;
    const keep = n - cut;
    if (keep <= 0) {
      this._len = 0;
      this._head = 0;
      return;
    }
    const tmp = new Float64Array(keep);
    for (let i = 0; i < keep; i++) tmp[i] = this._at(cut + i);
    // reset to linear
    this._buf.fill(0);
    for (let i = 0; i < keep; i++) this._buf[i] = tmp[i];
    this._len = keep;
    this._head = 0;
  }

  toJSON(): any {
    const n = this._len;
    const arr: number[] = new Array(n);
    for (let i = 0; i < n; i++) arr[i] = this._at(i);
    return {
      delta: this.delta,
      maxWindow: this.maxWindow,
      buf: arr,
      driftCount: this.driftCount,
    };
  }

  static fromJSON(obj: any): _ADWIN {
    const delta = (obj && typeof obj.delta === "number") ? obj.delta : 0.002;
    const maxWindow = (obj && typeof obj.maxWindow === "number")
      ? obj.maxWindow
      : 200;
    const a = new _ADWIN(delta, maxWindow);
    const buf: any[] = (obj && Array.isArray(obj.buf)) ? obj.buf : [];
    a._len = 0;
    a._head = 0;
    a._buf.fill(0);
    const n = buf.length;
    for (let i = 0; i < n; i++) a.update(+buf[i]);
    a.driftCount = (obj && typeof obj.driftCount === "number")
      ? obj.driftCount
      : 0;
    return a;
  }
}

export class FusionTemporalTransformerRegression {
  private _cfg: Config;

  private _isInitialized: boolean = false;
  private _inputDim: number = 0;
  private _outputDim: number = 0;
  private _maxSeqLen: number = 512;
  private _seqLen: number = 0;

  private _scaleCount: number = 0;
  private _scales: Int32Array = new Int32Array(0);
  private _kernelSize: number = 3;
  private _kernelHalf: number = 1;

  private _d: number = 64;
  private _h: number = 8;
  private _dh: number = 8;
  private _hidden: number = 256;

  // RNG (LCG) for deterministic init
  private _rngState: number = 1;

  // Normalization
  private _inWelford: _Welford | null = null;
  private _outWelford: _Welford | null = null;
  private _inStd: Float64Array = new Float64Array(0);
  private _outStd: Float64Array = new Float64Array(0);

  // Drift
  private _adwin: _ADWIN | null = null;
  private _driftCount: number = 0;

  // Training tracking
  private _updateCount: number = 0;
  private _sampleCount: number = 0;
  private _lossSum: number = 0.0;
  private _lossAvg: number = 0.0;
  private _prevLossAvg: number = Number.POSITIVE_INFINITY;
  private _converged: boolean = false;
  private _effectiveLR: number = 0.0;

  // Buffers for input/output
  private _xRaw: Float64Array = new Float64Array(0); // [maxSeqLen*inDim]
  private _xNorm: Float64Array = new Float64Array(0); // [maxSeqLen*inDim]
  private _xEmb: Float64Array = new Float64Array(0); // [maxSeqLen*d]
  private _dxEmb: Float64Array = new Float64Array(0); // [maxSeqLen*d]

  private _yRaw: Float64Array = new Float64Array(0); // [outputDim] (target last step)
  private _yNorm: Float64Array = new Float64Array(0); // [outputDim]
  private _yHatNorm: Float64Array = new Float64Array(0); // [outputDim]
  private _dYHat: Float64Array = new Float64Array(0); // [outputDim]

  // Positional encoding (maxSeqLen x d)
  private _posEnc: Float64Array = new Float64Array(0);

  // Input projection params
  private _Win: Float64Array = new Float64Array(0); // [inDim*d]
  private _bin: Float64Array = new Float64Array(0); // [d]

  // Multi-scale conv params (depthwise): [scaleCount*kernelSize*d]
  private _Wconv: Float64Array = new Float64Array(0);
  private _bconv: Float64Array = new Float64Array(0); // [scaleCount*d]
  private _scaleEmb: Float64Array = new Float64Array(0); // [scaleCount*d]

  // Per-scale buffers
  private _scaleLen: Int32Array = new Int32Array(0); // [scaleCount]
  private _convPre: Float64Array[] = [];
  private _convAct: Float64Array[] = [];
  private _Escale: Float64Array[] = []; // after +PE +scaleEmb

  // Fusion gate params: concatDim = scaleCount*d, gates = scaleCount
  private _Wg: Float64Array = new Float64Array(0); // [concatDim*scaleCount]
  private _bg: Float64Array = new Float64Array(0); // [scaleCount]
  private _gates: Float64Array = new Float64Array(0); // [maxSeqLen*scaleCount]
  private _fused: Float64Array = new Float64Array(0); // [maxSeqLen*d]
  private _dFused: Float64Array = new Float64Array(0); // [maxSeqLen*d]

  // Transformer blocks
  private _blocks: _Block[] = [];

  // Pooling + Output
  private _Wpool: Float64Array = new Float64Array(0); // [d]
  private _bpool: Float64Array = new Float64Array(1); // [1]
  private _poolScores: Float64Array = new Float64Array(0); // [maxSeqLen]
  private _poolAlpha: Float64Array = new Float64Array(0); // [maxSeqLen]
  private _pooled: Float64Array = new Float64Array(0); // [d]
  private _dPooled: Float64Array = new Float64Array(0); // [d]

  private _Wout: Float64Array = new Float64Array(0); // [d*outDim]
  private _bout: Float64Array = new Float64Array(0); // [outDim]

  // Attention scratch (global)
  private _attnScores: Float64Array = new Float64Array(0); // [h*maxSeqLen*maxSeqLen]
  private _attnProbs: Float64Array = new Float64Array(0); // [h*maxSeqLen*maxSeqLen]
  private _tmpD: Float64Array = new Float64Array(0); // [maxSeqLen*d] scratch
  private _tmpHidden: Float64Array = new Float64Array(0); // [maxSeqLen*hidden] scratch
  private _concat: Float64Array = new Float64Array(0); // [scaleCount*d] scratch per time

  // Parameter registry for optimizer
  private _params: _ParamTensor[] = [];

  constructor(config?: Partial<Config>) {
    this._cfg = this._withDefaults(config);
    this._rngState = (this._cfg.seed | 0) ^ 0x9e3779b9;
  }

  /**
   * Incremental online fit on a single sequence sample.
   * Auto-detects dimensions on first call.
   *
   * Data contract:
   * - xCoordinates: number[seqLen][inputDim]
   * - yCoordinates: number[seqLen][outputDim] (target = last timestep)
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xCoords = data.xCoordinates;
    const yCoords = data.yCoordinates;

    const seqLen = xCoords.length | 0;
    if (seqLen <= 0) {
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

    const inputDim = (xCoords[0] ? xCoords[0].length : 0) | 0;
    const outputDim = (yCoords[0] ? yCoords[0].length : 0) | 0;

    if (!this._isInitialized) {
      this._initialize(inputDim, outputDim, seqLen);
    } else {
      // handle varying seqLen (<= maxSequenceLength)
      this._seqLen = seqLen <= this._maxSeqLen ? seqLen : this._maxSeqLen;
    }

    this._seqLen = seqLen <= this._maxSeqLen ? seqLen : this._maxSeqLen;
    const L = this._seqLen;
    const inDim = this._inputDim;
    const outDim = this._outputDim;

    // Copy raw inputs into _xRaw
    {
      const xRaw = this._xRaw;
      let idx = 0;
      for (let t = 0; t < L; t++) {
        const row = xCoords[t];
        for (let j = 0; j < inDim; j++) {
          xRaw[idx++] = +row[j];
        }
      }
    }

    // Target: last timestep of yCoordinates
    {
      const yRaw = this._yRaw;
      const last = yCoords[(yCoords.length - 1) | 0] || yCoords[0];
      for (let j = 0; j < outDim; j++) yRaw[j] = +last[j];
    }

    // Update Welford stats and compute stds
    const eps = this._cfg.epsilon;
    const inW = this._inWelford!;
    const outW = this._outWelford!;
    for (let t = 0; t < L; t++) inW.update(this._xRaw, t * inDim);
    outW.update(this._yRaw, 0);

    inW.computeStd(this._inStd, eps);
    outW.computeStd(this._outStd, eps);

    // Normalize x and y into _xNorm, _yNorm
    {
      const xRaw = this._xRaw;
      const xNorm = this._xNorm;
      const mean = inW.mean;
      const std = this._inStd;
      let idx = 0;
      for (let t = 0; t < L; t++) {
        const base = t * inDim;
        for (let j = 0; j < inDim; j++) {
          const v = xRaw[base + j];
          xNorm[idx++] = (v - mean[j]) / (std[j] + eps);
        }
      }
    }
    {
      const yRaw = this._yRaw;
      const yNorm = this._yNorm;
      const mean = outW.mean;
      const std = this._outStd;
      for (let j = 0; j < outDim; j++) {
        yNorm[j] = (yRaw[j] - mean[j]) / (std[j] + eps);
      }
    }

    // Zero grads
    this._zeroGrads();

    // Forward pass
    this._forward(L);

    // Loss + outlier detection (normalized residual)
    let isOutlier = false;
    let weight = 1.0;
    let mse = 0.0;
    {
      const yHat = this._yHatNorm;
      const y = this._yNorm;
      const outStd = this._outStd;
      let maxAbsR = 0.0;
      for (let j = 0; j < outDim; j++) {
        const r = yHat[j] - y[j]; // already normalized
        const ar = Math.abs(r);
        if (ar > maxAbsR) maxAbsR = ar;
        mse += r * r;
      }
      mse /= outDim > 0 ? outDim : 1;
      if (maxAbsR > this._cfg.outlierThreshold) {
        isOutlier = true;
        weight = 0.1;
      }
      // Convert loss to original scale for reporting: use avg std^2 (approx)
      // loss_norm = 0.5*mse*weight, loss_orig ≈ 0.5*mean((r*std)^2) = 0.5*mse*mean(std^2)
      let std2 = 0.0;
      for (let j = 0; j < outDim; j++) std2 += outStd[j] * outStd[j];
      std2 /= outDim > 0 ? outDim : 1;
      const lossOrig = 0.5 * mse * std2 * weight;

      // Backprop seed gradient on normalized outputs: dL/dyHat = (yHat - y)/outDim * weight
      const dY = this._dYHat;
      const invN = 1.0 / (outDim > 0 ? outDim : 1);
      for (let j = 0; j < outDim; j++) {
        dY[j] = (this._yHatNorm[j] - this._yNorm[j]) * invN * weight;
      }

      // Backward pass
      this._backward(L);

      // L2 regularization
      this._applyL2();

      // Adam update
      const lr = this._computeLearningRate();
      this._effectiveLR = lr;
      const gradNorm = this._adamStep(lr);

      // Update trackers
      this._sampleCount++;
      this._updateCount++;

      // running average loss (normalized, weighted)
      const lossNorm = 0.5 * mse * weight;
      this._lossSum += lossNorm;
      this._lossAvg = this._lossSum /
        (this._sampleCount > 0 ? this._sampleCount : 1);

      // accuracy = 1/(1+L̄)
      const driftDetected = this._adwin ? this._adwin.update(lossNorm) : false;
      if (driftDetected) {
        this._driftCount = this._adwin
          ? this._adwin.driftCount
          : (this._driftCount + 1);
        // reset running loss on drift (keep model weights)
        this._lossSum = 0.0;
        this._lossAvg = 0.0;
        this._prevLossAvg = Number.POSITIVE_INFINITY;
        this._converged = false;
      }

      // Convergence check
      const diff = Math.abs(this._lossAvg - this._prevLossAvg);
      this._converged = diff <= this._cfg.convergenceThreshold;
      this._prevLossAvg = this._lossAvg;

      return {
        loss: lossOrig,
        gradientNorm: gradNorm,
        effectiveLearningRate: lr,
        isOutlier,
        converged: this._converged,
        sampleIndex: this._sampleCount,
        driftDetected,
      };
    }
  }

  /**
   * Predict futureSteps outputs using the last seen sequence context.
   * Since future x is not provided, this uses a simple persistence strategy:
   * - Reuses the last observed x vector to extend the sequence, autoregressively shifting the window.
   */
  predict(futureSteps: number): PredictionResult {
    const steps = (futureSteps | 0) > 0 ? (futureSteps | 0) : 0;
    const preds: SinglePrediction[] = [];

    const isReady = this._isInitialized && this._sampleCount > 0;
    if (!isReady || steps === 0) {
      return {
        predictions: preds,
        accuracy: this._accuracy(),
        sampleCount: this._sampleCount,
        isModelReady: isReady,
      };
    }

    const L = this._seqLen;
    const inDim = this._inputDim;
    const outDim = this._outputDim;

    // Working copy of last normalized x sequence into tmp buffer (row-major)
    const xWork = this._xNorm; // already has last input normalized
    // A scratch buffer for rolling if needed
    const xRoll = this._xRaw; // reuse raw buffer as scratch (safe: not in hot training now)

    // Standard error estimate from running average loss in original scale
    const outStd = this._outStd;
    let lossAvg = this._lossAvg;
    if (!(lossAvg >= 0)) lossAvg = 0;
    // lossAvg is normalized loss ~ 0.5 * mse ; mse ~ 2*loss
    const mseNorm = 2.0 * lossAvg;
    // per-dim SE in original space: sqrt(mseNorm) * std_j
    const z = 1.96;

    for (let s = 0; s < steps; s++) {
      // Forward with current xWork (in _xNorm)
      this._forward(L);

      // Denormalize prediction
      const yHat = this._yHatNorm;
      const mean = this._outWelford
        ? this._outWelford.mean
        : new Float64Array(outDim);
      const eps = this._cfg.epsilon;

      const predicted = new Array<number>(outDim);
      const se = new Array<number>(outDim);
      const lower = new Array<number>(outDim);
      const upper = new Array<number>(outDim);

      for (let j = 0; j < outDim; j++) {
        const y = yHat[j] * (outStd[j] + eps) + mean[j];
        predicted[j] = y;
        const sErr = Math.sqrt(Math.max(0.0, mseNorm)) * (outStd[j] + eps);
        se[j] = sErr;
        lower[j] = y - z * sErr;
        upper[j] = y + z * sErr;
      }

      preds.push({
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError: se,
      });

      // Roll x sequence by 1 step: shift left and append last vector (persistence)
      // Use xRoll scratch: first copy (L-1) rows
      if (L > 1) {
        const rowSize = inDim;
        // shift
        let dst = 0;
        let src = rowSize;
        const end = (L - 1) * rowSize;
        while (src < L * rowSize) {
          xRoll[dst++] = xWork[src++];
          if (dst >= end) break;
        }
        // append last row (previous last row)
        const lastBase = (L - 1) * rowSize;
        const prevLastBase = (L - 1) * rowSize;
        for (let j = 0; j < rowSize; j++) {
          xRoll[lastBase + j] = xWork[prevLastBase + j];
        }
        // copy back into xWork
        for (let i = 0; i < L * rowSize; i++) xWork[i] = xRoll[i];
      }
    }

    return {
      predictions: preds,
      accuracy: this._accuracy(),
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      numBlocks: this._cfg.numBlocks,
      embeddingDim: this._d,
      numHeads: this._h,
      temporalScales: Array.from(this._scales as any),
      totalParameters: this._totalParams(),
      sampleCount: this._sampleCount,
      accuracy: this._accuracy(),
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
        fusionWeights: [],
        attentionWeights: [],
        ffnWeights: [],
        layerNormParams: [],
        outputWeights: [],
        firstMoment: [],
        secondMoment: [],
        updateCount: this._updateCount,
      };
    }

    const d = this._d;
    const k = this._kernelSize;
    const sCount = this._scaleCount;

    // temporalConvWeights: [scale][kernel][d]
    const temporalConvWeights: number[][][] = new Array(sCount);
    for (let s = 0; s < sCount; s++) {
      const arrK: number[][] = new Array(k);
      for (let kk = 0; kk < k; kk++) {
        const row = new Array<number>(d);
        const base = (s * k + kk) * d;
        for (let j = 0; j < d; j++) row[j] = this._Wconv[base + j];
        arrK[kk] = row;
      }
      temporalConvWeights[s] = arrK;
    }

    // scaleEmbeddings: [scale][d]
    const scaleEmbeddings: number[][] = new Array(sCount);
    for (let s = 0; s < sCount; s++) {
      const row = new Array<number>(d);
      const base = s * d;
      for (let j = 0; j < d; j++) row[j] = this._scaleEmb[base + j];
      scaleEmbeddings[s] = row;
    }

    // positionalEncoding: [maxSeqLen][d]
    const positionalEncoding: number[][] = new Array(this._maxSeqLen);
    for (let t = 0; t < this._maxSeqLen; t++) {
      const row = new Array<number>(d);
      const base = t * d;
      for (let j = 0; j < d; j++) row[j] = this._posEnc[base + j];
      positionalEncoding[t] = row;
    }

    // fusionWeights: [concatDim][scaleCount]
    const concatDim = sCount * d;
    const fusionWeights: number[][] = new Array(concatDim);
    for (let i = 0; i < concatDim; i++) {
      const row = new Array<number>(sCount);
      const base = i * sCount;
      for (let j = 0; j < sCount; j++) row[j] = this._Wg[base + j];
      fusionWeights[i] = row;
    }

    // attentionWeights: [numBlocks][4][packed(W+b)]
    const attentionWeights: number[][][] = new Array(this._blocks.length);
    const ffnWeights: number[][][] = new Array(this._blocks.length);
    const layerNormParams: number[][][] = new Array(this._blocks.length);

    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];

      attentionWeights[b] = [
        this._packWB(blk.Wq, blk.bq),
        this._packWB(blk.Wk, blk.bk),
        this._packWB(blk.Wv, blk.bv),
        this._packWB(blk.Wo, blk.bo),
      ];

      ffnWeights[b] = [
        this._packWB(blk.W1, blk.b1),
        this._packWB(blk.W2, blk.b2),
      ];

      // layerNormParams: [2][(gamma+beta) packed], store ln1 and ln2
      layerNormParams[b] = [
        this._packVec2(blk.ln1Gamma, blk.ln1Beta),
        this._packVec2(blk.ln2Gamma, blk.ln2Beta),
      ];
    }

    // outputWeights: [d][outDim]
    const outDim = this._outputDim;
    const outputWeights: number[][] = new Array(d);
    for (let i = 0; i < d; i++) {
      const row = new Array<number>(outDim);
      const base = i * outDim;
      for (let j = 0; j < outDim; j++) row[j] = this._Wout[base + j];
      outputWeights[i] = row;
    }

    // Moments grouped (best-effort packing for export)
    // Group 0: conv + scaleEmb + fusion + input + pool + output
    // Group 1+: per block (attn + ffn + ln)
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    // Group 0 packing
    const g0m: number[][] = [];
    const g0v: number[][] = [];
    {
      // helper to push tensor
      const pushTensor = (
        w: Float64Array,
        m: Float64Array,
        v: Float64Array,
      ) => {
        g0m.push(Array.from(m));
        g0v.push(Array.from(v));
      };
      // Find in registry by reference
      for (let i = 0; i < this._params.length; i++) {
        const p = this._params[i];
        if (
          p.name === "Win" ||
          p.name === "bin" ||
          p.name === "Wconv" ||
          p.name === "bconv" ||
          p.name === "scaleEmb" ||
          p.name === "Wg" ||
          p.name === "bg" ||
          p.name === "Wpool" ||
          p.name === "bpool" ||
          p.name === "Wout" ||
          p.name === "bout"
        ) {
          pushTensor(p.w, p.m, p.v);
        }
      }
    }
    firstMoment.push(g0m as any);
    secondMoment.push(g0v as any);

    // Per-block groups
    for (let b = 0; b < this._blocks.length; b++) {
      const gm: number[][] = [];
      const gv: number[][] = [];
      const prefix = `block${b}.`;
      for (let i = 0; i < this._params.length; i++) {
        const p = this._params[i];
        if (p.name.indexOf(prefix) === 0) {
          gm.push(Array.from(p.m));
          gv.push(Array.from(p.v));
        }
      }
      firstMoment.push(gm as any);
      secondMoment.push(gv as any);
    }

    return {
      temporalConvWeights,
      scaleEmbeddings,
      positionalEncoding,
      fusionWeights,
      attentionWeights,
      ffnWeights,
      layerNormParams,
      outputWeights,
      firstMoment,
      secondMoment,
      updateCount: this._updateCount,
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this._isInitialized || !this._inWelford || !this._outWelford) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }
    return {
      inputMean: Array.from(this._inWelford.mean),
      inputStd: Array.from(this._inStd),
      outputMean: Array.from(this._outWelford.mean),
      outputStd: Array.from(this._outStd),
      count: this._inWelford.count,
    };
  }

  reset(): void {
    this._isInitialized = false;
    this._inputDim = 0;
    this._outputDim = 0;
    this._seqLen = 0;

    this._updateCount = 0;
    this._sampleCount = 0;
    this._lossSum = 0.0;
    this._lossAvg = 0.0;
    this._prevLossAvg = Number.POSITIVE_INFINITY;
    this._converged = false;
    this._effectiveLR = 0.0;

    this._driftCount = 0;

    // Clear all arrays
    this._params = [];
    this._blocks = [];

    this._inWelford = null;
    this._outWelford = null;
    this._adwin = null;

    this._xRaw = new Float64Array(0);
    this._xNorm = new Float64Array(0);
    this._xEmb = new Float64Array(0);
    this._dxEmb = new Float64Array(0);

    this._yRaw = new Float64Array(0);
    this._yNorm = new Float64Array(0);
    this._yHatNorm = new Float64Array(0);
    this._dYHat = new Float64Array(0);

    this._posEnc = new Float64Array(0);

    this._Win = new Float64Array(0);
    this._bin = new Float64Array(0);

    this._Wconv = new Float64Array(0);
    this._bconv = new Float64Array(0);
    this._scaleEmb = new Float64Array(0);

    this._scaleLen = new Int32Array(0);
    this._convPre = [];
    this._convAct = [];
    this._Escale = [];

    this._Wg = new Float64Array(0);
    this._bg = new Float64Array(0);
    this._gates = new Float64Array(0);
    this._fused = new Float64Array(0);
    this._dFused = new Float64Array(0);

    this._Wpool = new Float64Array(0);
    this._bpool = new Float64Array(1);
    this._poolScores = new Float64Array(0);
    this._poolAlpha = new Float64Array(0);
    this._pooled = new Float64Array(0);
    this._dPooled = new Float64Array(0);

    this._Wout = new Float64Array(0);
    this._bout = new Float64Array(0);

    this._attnScores = new Float64Array(0);
    this._attnProbs = new Float64Array(0);
    this._tmpD = new Float64Array(0);
    this._tmpHidden = new Float64Array(0);
    this._concat = new Float64Array(0);

    // keep config + rng
    this._rngState = (this._cfg.seed | 0) ^ 0x9e3779b9;
  }

  save(): string {
    const obj: any = {
      cfg: this._cfg,
      isInitialized: this._isInitialized,
      inputDim: this._inputDim,
      outputDim: this._outputDim,
      maxSeqLen: this._maxSeqLen,
      seqLen: this._seqLen,
      updateCount: this._updateCount,
      sampleCount: this._sampleCount,
      lossSum: this._lossSum,
      lossAvg: this._lossAvg,
      prevLossAvg: this._prevLossAvg,
      converged: this._converged,
      effectiveLR: this._effectiveLR,
      driftCount: this._driftCount,
      rngState: this._rngState,

      inWelford: this._inWelford ? this._inWelford.toJSON() : null,
      outWelford: this._outWelford ? this._outWelford.toJSON() : null,
      adwin: this._adwin ? this._adwin.toJSON() : null,

      weights: {},
    };

    if (this._isInitialized) {
      const w: any = obj.weights;

      w.posEnc = Array.from(this._posEnc);

      w.Win = Array.from(this._Win);
      w.bin = Array.from(this._bin);

      w.Wconv = Array.from(this._Wconv);
      w.bconv = Array.from(this._bconv);
      w.scaleEmb = Array.from(this._scaleEmb);

      w.Wg = Array.from(this._Wg);
      w.bg = Array.from(this._bg);

      w.Wpool = Array.from(this._Wpool);
      w.bpool = Array.from(this._bpool);

      w.Wout = Array.from(this._Wout);
      w.bout = Array.from(this._bout);

      // Blocks
      const blocks: any[] = new Array(this._blocks.length);
      for (let i = 0; i < this._blocks.length; i++) {
        const b = this._blocks[i];
        blocks[i] = {
          ln1Gamma: Array.from(b.ln1Gamma),
          ln1Beta: Array.from(b.ln1Beta),
          Wq: Array.from(b.Wq),
          bq: Array.from(b.bq),
          Wk: Array.from(b.Wk),
          bk: Array.from(b.bk),
          Wv: Array.from(b.Wv),
          bv: Array.from(b.bv),
          Wo: Array.from(b.Wo),
          bo: Array.from(b.bo),
          ln2Gamma: Array.from(b.ln2Gamma),
          ln2Beta: Array.from(b.ln2Beta),
          W1: Array.from(b.W1),
          b1: Array.from(b.b1),
          W2: Array.from(b.W2),
          b2: Array.from(b.b2),
        };
      }
      w.blocks = blocks;

      // Optimizer moments registry
      const params: any[] = new Array(this._params.length);
      for (let i = 0; i < this._params.length; i++) {
        const p = this._params[i];
        params[i] = {
          name: p.name,
          shape: p.shape.slice(0),
          w: Array.from(p.w),
          m: Array.from(p.m),
          v: Array.from(p.v),
          applyL2: p.applyL2,
        };
      }
      w.params = params;
    }

    return JSON.stringify(obj);
  }

  load(w: string): void {
    const obj = JSON.parse(w);

    const cfg = obj && obj.cfg ? obj.cfg : null;
    this._cfg = this._withDefaults(cfg || {});
    this._rngState = (obj && typeof obj.rngState === "number")
      ? (obj.rngState | 0)
      : ((this._cfg.seed | 0) ^ 0x9e3779b9);

    this._isInitialized = !!(obj && obj.isInitialized);
    this._inputDim = (obj && typeof obj.inputDim === "number")
      ? obj.inputDim | 0
      : 0;
    this._outputDim = (obj && typeof obj.outputDim === "number")
      ? obj.outputDim | 0
      : 0;
    this._maxSeqLen = (obj && typeof obj.maxSeqLen === "number")
      ? obj.maxSeqLen | 0
      : this._cfg.maxSequenceLength;
    this._seqLen = (obj && typeof obj.seqLen === "number") ? obj.seqLen | 0 : 0;

    this._updateCount = (obj && typeof obj.updateCount === "number")
      ? obj.updateCount | 0
      : 0;
    this._sampleCount = (obj && typeof obj.sampleCount === "number")
      ? obj.sampleCount | 0
      : 0;
    this._lossSum = (obj && typeof obj.lossSum === "number")
      ? +obj.lossSum
      : 0.0;
    this._lossAvg = (obj && typeof obj.lossAvg === "number")
      ? +obj.lossAvg
      : 0.0;
    this._prevLossAvg = (obj && typeof obj.prevLossAvg === "number")
      ? +obj.prevLossAvg
      : Number.POSITIVE_INFINITY;
    this._converged = !!(obj && obj.converged);
    this._effectiveLR = (obj && typeof obj.effectiveLR === "number")
      ? +obj.effectiveLR
      : 0.0;
    this._driftCount = (obj && typeof obj.driftCount === "number")
      ? obj.driftCount | 0
      : 0;

    this._inWelford = (obj && obj.inWelford)
      ? _Welford.fromJSON(obj.inWelford)
      : null;
    this._outWelford = (obj && obj.outWelford)
      ? _Welford.fromJSON(obj.outWelford)
      : null;
    this._adwin = (obj && obj.adwin) ? _ADWIN.fromJSON(obj.adwin) : null;

    if (!this._isInitialized) return;

    // Ensure init with loaded dims
    this._initialize(
      this._inputDim,
      this._outputDim,
      this._seqLen > 0 ? this._seqLen : 1,
    );

    const weights = obj.weights || {};
    this._copyFromArray(this._posEnc, weights.posEnc);

    this._copyFromArray(this._Win, weights.Win);
    this._copyFromArray(this._bin, weights.bin);

    this._copyFromArray(this._Wconv, weights.Wconv);
    this._copyFromArray(this._bconv, weights.bconv);
    this._copyFromArray(this._scaleEmb, weights.scaleEmb);

    this._copyFromArray(this._Wg, weights.Wg);
    this._copyFromArray(this._bg, weights.bg);

    this._copyFromArray(this._Wpool, weights.Wpool);
    this._copyFromArray(this._bpool, weights.bpool);

    this._copyFromArray(this._Wout, weights.Wout);
    this._copyFromArray(this._bout, weights.bout);

    // Blocks
    const blocks = Array.isArray(weights.blocks) ? weights.blocks : [];
    for (let i = 0; i < this._blocks.length && i < blocks.length; i++) {
      const src = blocks[i];
      const dst = this._blocks[i];
      this._copyFromArray(dst.ln1Gamma, src.ln1Gamma);
      this._copyFromArray(dst.ln1Beta, src.ln1Beta);

      this._copyFromArray(dst.Wq, src.Wq);
      this._copyFromArray(dst.bq, src.bq);
      this._copyFromArray(dst.Wk, src.Wk);
      this._copyFromArray(dst.bk, src.bk);
      this._copyFromArray(dst.Wv, src.Wv);
      this._copyFromArray(dst.bv, src.bv);
      this._copyFromArray(dst.Wo, src.Wo);
      this._copyFromArray(dst.bo, src.bo);

      this._copyFromArray(dst.ln2Gamma, src.ln2Gamma);
      this._copyFromArray(dst.ln2Beta, src.ln2Beta);

      this._copyFromArray(dst.W1, src.W1);
      this._copyFromArray(dst.b1, src.b1);
      this._copyFromArray(dst.W2, src.W2);
      this._copyFromArray(dst.b2, src.b2);
    }

    // Params registry moments (rebuild)
    const params = Array.isArray(weights.params) ? weights.params : [];
    // Replace existing registry weights/moments if present
    for (let i = 0; i < params.length; i++) {
      const src = params[i];
      if (!src || typeof src.name !== "string") continue;
      const p = this._findParamByName(src.name);
      if (!p) continue;
      this._copyFromArray(p.w, src.w);
      this._copyFromArray(p.m, src.m);
      this._copyFromArray(p.v, src.v);
      p.applyL2 = !!src.applyL2;
    }

    // Recompute std arrays
    if (this._inWelford) {
      this._inWelford.computeStd(this._inStd, this._cfg.epsilon);
    }
    if (this._outWelford) {
      this._outWelford.computeStd(this._outStd, this._cfg.epsilon);
    }
  }

  // ------------------------- Internal: Initialization -------------------------

  private _withDefaults(config?: Partial<Config> | null): Config {
    const c = (config || {}) as any;

    const temporalScales = Array.isArray(c.temporalScales)
      ? c.temporalScales.slice(0)
      : [1, 2, 4];

    return {
      numBlocks: this._num(c.numBlocks, 3),
      embeddingDim: this._num(c.embeddingDim, 64),
      numHeads: this._num(c.numHeads, 8),
      ffnMultiplier: this._num(c.ffnMultiplier, 4),

      attentionDropout: this._num(c.attentionDropout, 0.0),
      fusionDropout: this._num(c.fusionDropout, 0.0),

      learningRate: this._num(c.learningRate, 0.001),
      warmupSteps: this._num(c.warmupSteps, 100),
      totalSteps: this._num(c.totalSteps, 10000),
      beta1: this._num(c.beta1, 0.9),
      beta2: this._num(c.beta2, 0.999),
      epsilon: this._num(c.epsilon, 1e-8),

      regularizationStrength: this._num(c.regularizationStrength, 1e-4),
      convergenceThreshold: this._num(c.convergenceThreshold, 1e-6),
      outlierThreshold: this._num(c.outlierThreshold, 3.0),

      adwinDelta: this._num(c.adwinDelta, 0.002),

      temporalScales,
      temporalKernelSize: this._num(c.temporalKernelSize, 3),
      maxSequenceLength: this._num(c.maxSequenceLength, 512),

      attentionWindow: this._num(c.attentionWindow, 0),

      seed: this._num(c.seed, 1337),
    };
  }

  private _initialize(
    inputDim: number,
    outputDim: number,
    seqLen: number,
  ): void {
    this._inputDim = inputDim | 0;
    this._outputDim = outputDim | 0;

    this._d = this._cfg.embeddingDim | 0;
    this._h = this._cfg.numHeads | 0;
    if (this._h <= 0) this._h = 1;
    if ((this._d % this._h) !== 0) {
      // force divisibility by reducing heads
      let hh = this._h;
      while (hh > 1 && (this._d % hh) !== 0) hh--;
      this._h = hh;
    }
    this._dh = (this._d / this._h) | 0;
    this._hidden = (this._d * (this._cfg.ffnMultiplier | 0)) | 0;

    this._maxSeqLen = this._cfg.maxSequenceLength | 0;
    if (this._maxSeqLen <= 0) this._maxSeqLen = 512;
    this._seqLen = seqLen <= this._maxSeqLen ? seqLen : this._maxSeqLen;

    // scales
    const scales = this._cfg.temporalScales;
    this._scaleCount = scales.length | 0;
    const sArr = new Int32Array(this._scaleCount);
    for (let i = 0; i < this._scaleCount; i++) {
      let v = scales[i] | 0;
      if (v <= 0) v = 1;
      sArr[i] = v;
    }
    this._scales = sArr;

    // kernel
    this._kernelSize = this._cfg.temporalKernelSize | 0;
    if (this._kernelSize <= 0) this._kernelSize = 3;
    if ((this._kernelSize & 1) === 0) this._kernelSize += 1; // force odd
    this._kernelHalf = (this._kernelSize / 2) | 0;

    // Normalization
    this._inWelford = new _Welford(this._inputDim);
    this._outWelford = new _Welford(this._outputDim);
    this._inStd = new Float64Array(this._inputDim);
    this._outStd = new Float64Array(this._outputDim);
    this._inWelford.computeStd(this._inStd, this._cfg.epsilon);
    this._outWelford.computeStd(this._outStd, this._cfg.epsilon);

    // Drift detector
    this._adwin = new _ADWIN(this._cfg.adwinDelta, 200);

    // Buffers
    this._xRaw = new Float64Array(this._maxSeqLen * this._inputDim);
    this._xNorm = new Float64Array(this._maxSeqLen * this._inputDim);
    this._xEmb = new Float64Array(this._maxSeqLen * this._d);
    this._dxEmb = new Float64Array(this._maxSeqLen * this._d);

    this._yRaw = new Float64Array(this._outputDim);
    this._yNorm = new Float64Array(this._outputDim);
    this._yHatNorm = new Float64Array(this._outputDim);
    this._dYHat = new Float64Array(this._outputDim);

    // Positional encoding
    this._posEnc = new Float64Array(this._maxSeqLen * this._d);
    this._buildPositionalEncoding(this._posEnc, this._maxSeqLen, this._d);

    // Input projection params
    this._Win = new Float64Array(this._inputDim * this._d);
    this._bin = new Float64Array(this._d);

    // Conv params
    this._Wconv = new Float64Array(
      this._scaleCount * this._kernelSize * this._d,
    );
    this._bconv = new Float64Array(this._scaleCount * this._d);
    this._scaleEmb = new Float64Array(this._scaleCount * this._d);

    // Scale lengths and per-scale buffers
    this._scaleLen = new Int32Array(this._scaleCount);
    this._convPre = new Array(this._scaleCount);
    this._convAct = new Array(this._scaleCount);
    this._Escale = new Array(this._scaleCount);
    for (let s = 0; s < this._scaleCount; s++) {
      const stride = this._scales[s];
      let outLen = (this._maxSeqLen / stride) | 0;
      if (outLen < 1) outLen = 1;
      this._scaleLen[s] = outLen;
      this._convPre[s] = new Float64Array(outLen * this._d);
      this._convAct[s] = new Float64Array(outLen * this._d);
      this._Escale[s] = new Float64Array(outLen * this._d);
    }

    // Fusion
    const concatDim = this._scaleCount * this._d;
    this._Wg = new Float64Array(concatDim * this._scaleCount);
    this._bg = new Float64Array(this._scaleCount);
    this._gates = new Float64Array(this._maxSeqLen * this._scaleCount);
    this._fused = new Float64Array(this._maxSeqLen * this._d);
    this._dFused = new Float64Array(this._maxSeqLen * this._d);
    this._concat = new Float64Array(concatDim);

    // Transformer blocks
    this._blocks = [];
    for (let b = 0; b < (this._cfg.numBlocks | 0); b++) {
      this._blocks.push(this._createBlock(b));
    }

    // Pool + output
    this._Wpool = new Float64Array(this._d);
    this._bpool = new Float64Array(1);
    this._poolScores = new Float64Array(this._maxSeqLen);
    this._poolAlpha = new Float64Array(this._maxSeqLen);
    this._pooled = new Float64Array(this._d);
    this._dPooled = new Float64Array(this._d);

    this._Wout = new Float64Array(this._d * this._outputDim);
    this._bout = new Float64Array(this._outputDim);

    // Attention scratch
    this._attnScores = new Float64Array(
      this._h * this._maxSeqLen * this._maxSeqLen,
    );
    this._attnProbs = new Float64Array(
      this._h * this._maxSeqLen * this._maxSeqLen,
    );
    this._tmpD = new Float64Array(this._maxSeqLen * this._d);
    this._tmpHidden = new Float64Array(this._maxSeqLen * this._hidden);

    // Init weights
    this._initAllParams();

    // Build optimizer registry
    this._buildParamRegistry();

    this._isInitialized = true;
  }

  private _createBlock(blockIndex: number): _Block {
    const d = this._d;
    const hidden = this._hidden;
    const maxSeq = this._maxSeqLen;

    const blk: _Block = {
      ln1Gamma: new Float64Array(d),
      ln1Beta: new Float64Array(d),

      Wq: new Float64Array(d * d),
      bq: new Float64Array(d),
      Wk: new Float64Array(d * d),
      bk: new Float64Array(d),
      Wv: new Float64Array(d * d),
      bv: new Float64Array(d),
      Wo: new Float64Array(d * d),
      bo: new Float64Array(d),

      ln2Gamma: new Float64Array(d),
      ln2Beta: new Float64Array(d),

      W1: new Float64Array(d * hidden),
      b1: new Float64Array(hidden),
      W2: new Float64Array(hidden * d),
      b2: new Float64Array(d),

      xIn: new Float64Array(maxSeq * d),
      xRes1: new Float64Array(maxSeq * d),
      xOut: new Float64Array(maxSeq * d),

      ln1Mean: new Float64Array(maxSeq),
      ln1InvStd: new Float64Array(maxSeq),
      ln1Xhat: new Float64Array(maxSeq * d),
      ln1Out: new Float64Array(maxSeq * d),

      ln2Mean: new Float64Array(maxSeq),
      ln2InvStd: new Float64Array(maxSeq),
      ln2Xhat: new Float64Array(maxSeq * d),
      ln2Out: new Float64Array(maxSeq * d),

      Q: new Float64Array(maxSeq * d),
      K: new Float64Array(maxSeq * d),
      V: new Float64Array(maxSeq * d),
      attnCtx: new Float64Array(maxSeq * d),

      ffnPre: new Float64Array(maxSeq * hidden),
      ffnAct: new Float64Array(maxSeq * hidden),
      ffnOut: new Float64Array(maxSeq * d),
    };

    // LN init: gamma=1, beta=0
    for (let i = 0; i < d; i++) {
      blk.ln1Gamma[i] = 1.0;
      blk.ln2Gamma[i] = 1.0;
      blk.ln1Beta[i] = 0.0;
      blk.ln2Beta[i] = 0.0;
    }

    return blk;
  }

  private _initAllParams(): void {
    // Xavier init for matrices, zeros for biases, gamma=1 for LN.
    this._xavierUniform(this._Win, this._inputDim, this._d);
    this._bin.fill(0);

    // Depthwise conv weights: fanIn ~ kernelSize, fanOut ~ kernelSize
    // Use small init
    this._smallUniform(this._Wconv, 0.02);
    this._bconv.fill(0);

    // Scale embeddings small init
    this._smallUniform(this._scaleEmb, 0.02);

    // Fusion gate
    const concatDim = this._scaleCount * this._d;
    this._xavierUniform(this._Wg, concatDim, this._scaleCount);
    this._bg.fill(0);

    // Pooling
    this._smallUniform(this._Wpool, 0.02);
    this._bpool[0] = 0;

    // Output
    this._xavierUniform(this._Wout, this._d, this._outputDim);
    this._bout.fill(0);

    // Blocks
    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];
      this._xavierUniform(blk.Wq, this._d, this._d);
      this._xavierUniform(blk.Wk, this._d, this._d);
      this._xavierUniform(blk.Wv, this._d, this._d);
      this._xavierUniform(blk.Wo, this._d, this._d);
      blk.bq.fill(0);
      blk.bk.fill(0);
      blk.bv.fill(0);
      blk.bo.fill(0);

      this._xavierUniform(blk.W1, this._d, this._hidden);
      this._xavierUniform(blk.W2, this._hidden, this._d);
      blk.b1.fill(0);
      blk.b2.fill(0);

      // LN gammas already 1; betas 0 in _createBlock
    }
  }

  private _buildParamRegistry(): void {
    this._params = [];

    const push = (
      name: string,
      shape: number[],
      w: Float64Array,
      applyL2: boolean,
    ) => {
      const g = new Float64Array(w.length);
      const m = new Float64Array(w.length);
      const v = new Float64Array(w.length);
      this._params.push({ name, shape, w, g, m, v, applyL2 });
    };

    push("Win", [this._inputDim, this._d], this._Win, true);
    push("bin", [this._d], this._bin, false);

    push(
      "Wconv",
      [this._scaleCount, this._kernelSize, this._d],
      this._Wconv,
      true,
    );
    push("bconv", [this._scaleCount, this._d], this._bconv, false);
    push("scaleEmb", [this._scaleCount, this._d], this._scaleEmb, false);

    const concatDim = this._scaleCount * this._d;
    push("Wg", [concatDim, this._scaleCount], this._Wg, true);
    push("bg", [this._scaleCount], this._bg, false);

    push("Wpool", [this._d], this._Wpool, true);
    push("bpool", [1], this._bpool, false);

    push("Wout", [this._d, this._outputDim], this._Wout, true);
    push("bout", [this._outputDim], this._bout, false);

    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];
      const prefix = `block${b}.`;

      push(prefix + "ln1Gamma", [this._d], blk.ln1Gamma, false);
      push(prefix + "ln1Beta", [this._d], blk.ln1Beta, false);

      push(prefix + "Wq", [this._d, this._d], blk.Wq, true);
      push(prefix + "bq", [this._d], blk.bq, false);
      push(prefix + "Wk", [this._d, this._d], blk.Wk, true);
      push(prefix + "bk", [this._d], blk.bk, false);
      push(prefix + "Wv", [this._d, this._d], blk.Wv, true);
      push(prefix + "bv", [this._d], blk.bv, false);
      push(prefix + "Wo", [this._d, this._d], blk.Wo, true);
      push(prefix + "bo", [this._d], blk.bo, false);

      push(prefix + "ln2Gamma", [this._d], blk.ln2Gamma, false);
      push(prefix + "ln2Beta", [this._d], blk.ln2Beta, false);

      push(prefix + "W1", [this._d, this._hidden], blk.W1, true);
      push(prefix + "b1", [this._hidden], blk.b1, false);
      push(prefix + "W2", [this._hidden, this._d], blk.W2, true);
      push(prefix + "b2", [this._d], blk.b2, false);
    }
  }

  private _findParamByName(name: string): _ParamTensor | null {
    for (let i = 0; i < this._params.length; i++) {
      if (this._params[i].name === name) return this._params[i];
    }
    return null;
  }

  // ------------------------- Internal: Forward/Backward -------------------------

  private _forward(L: number): void {
    const d = this._d;
    const inDim = this._inputDim;

    // 1) Input projection: Xemb = Xnorm * Win + bin  => [L x d]
    // Xnorm is stored row-major in _xNorm [L*inDim]
    {
      const X = this._xNorm;
      const W = this._Win;
      const b = this._bin;
      const out = this._xEmb;
      // out[t, j] = sum_k X[t,k] * W[k,j] + b[j]
      for (let t = 0; t < L; t++) {
        const xBase = t * inDim;
        const oBase = t * d;
        for (let j = 0; j < d; j++) out[oBase + j] = b[j];
        for (let k = 0; k < inDim; k++) {
          const xv = X[xBase + k];
          const wBase = k * d;
          for (let j = 0; j < d; j++) out[oBase + j] += xv * W[wBase + j];
        }
      }
    }

    // 2) Multi-scale temporal depthwise conv + GELU + (PE + scaleEmb) => Es
    for (let s = 0; s < this._scaleCount; s++) {
      const stride = this._scales[s];
      const outLen = (L / stride) | 0;
      const convPre = this._convPre[s];
      const convAct = this._convAct[s];
      const E = this._Escale[s];

      // convPre[p,c] = Σ_k Wconv[s,k,c] * xEmb[idx,c] + bconv[s,c], idx = clamp(p*stride + (k-half))
      const wOffBase = s * this._kernelSize * d;
      const bOff = s * d;
      for (let p = 0; p < outLen; p++) {
        const baseIn = (p * stride) | 0;
        const oBase = p * d;
        // init with bias
        for (let c = 0; c < d; c++) convPre[oBase + c] = this._bconv[bOff + c];
        for (let kk = 0; kk < this._kernelSize; kk++) {
          let idx = baseIn + kk - this._kernelHalf;
          if (idx < 0) idx = 0;
          else if (idx >= L) idx = L - 1;
          const xBase = idx * d;
          const wBase = wOffBase + kk * d;
          for (let c = 0; c < d; c++) {
            convPre[oBase + c] += this._Wconv[wBase + c] *
              this._xEmb[xBase + c];
          }
        }
      }

      // GELU
      for (let i = 0; i < outLen * d; i++) convAct[i] = this._gelu(convPre[i]);

      // Add positional encoding (use pos = p, shared) and scale embedding vector
      const seBase = s * d;
      for (let p = 0; p < outLen; p++) {
        const eBase = p * d;
        const peBase = p * d;
        for (let c = 0; c < d; c++) {
          E[eBase + c] = convAct[eBase + c] + this._posEnc[peBase + c] +
            this._scaleEmb[seBase + c];
        }
      }
    }

    // 3) Cross-scale gated fusion to length L: fused[t] = Σ_s sigmoid(concat*Wg+bg)[s] * E_s[floor(t/stride)]
    //    concat is [E0..Es] flattened (scaleCount*d)
    {
      const sCount = this._scaleCount;
      const concatDim = sCount * d;
      const Wg = this._Wg;
      const bg = this._bg;
      const gates = this._gates;
      const fused = this._fused;
      const concat = this._concat;

      for (let t = 0; t < L; t++) {
        // build concat
        let off = 0;
        for (let s = 0; s < sCount; s++) {
          const stride = this._scales[s];
          let p = (t / stride) | 0;
          const outLen = (L / stride) | 0;
          if (p < 0) p = 0;
          else if (p >= outLen) p = outLen - 1;
          const E = this._Escale[s];
          const eBase = p * d;
          for (let c = 0; c < d; c++) concat[off++] = E[eBase + c];
        }

        // compute gates
        const gBase = t * sCount;
        for (let s = 0; s < sCount; s++) {
          let z = bg[s];
          // z += Σ_j concat[j] * Wg[j,s]
          let wBase = s; // column-major access via row-major Wg[j*sCount + s]
          for (let j = 0; j < concatDim; j++) {
            z += concat[j] * Wg[wBase];
            wBase += sCount;
          }
          const g = this._sigmoid(z);
          gates[gBase + s] = g;
        }

        // fused = Σ_s g_s * E_s
        const outBase = t * d;
        for (let c = 0; c < d; c++) fused[outBase + c] = 0.0;

        for (let s = 0; s < sCount; s++) {
          const g = gates[gBase + s];
          const stride = this._scales[s];
          let p = (t / stride) | 0;
          const outLen = (L / stride) | 0;
          if (p < 0) p = 0;
          else if (p >= outLen) p = outLen - 1;
          const E = this._Escale[s];
          const eBase = p * d;
          for (let c = 0; c < d; c++) fused[outBase + c] += g * E[eBase + c];
        }
      }
    }

    // 4) Transformer blocks
    // input to first block = fused
    let cur = this._fused;
    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];

      // store xIn
      this._copyBlockSeq(blk.xIn, cur, L * d);

      // LN1
      this._layerNormForward(
        blk.xIn,
        blk.ln1Gamma,
        blk.ln1Beta,
        blk.ln1Out,
        blk.ln1Mean,
        blk.ln1InvStd,
        blk.ln1Xhat,
        L,
        d,
      );

      // QKV
      this._linearForward2D(blk.ln1Out, blk.Wq, blk.bq, blk.Q, L, d, d);
      this._linearForward2D(blk.ln1Out, blk.Wk, blk.bk, blk.K, L, d, d);
      this._linearForward2D(blk.ln1Out, blk.Wv, blk.bv, blk.V, L, d, d);

      // Attention ctx
      this._mhaCausalForward(blk.Q, blk.K, blk.V, blk.attnCtx, L);

      // Attn output: attnCtx * Wo + bo
      this._linearForward2D(blk.attnCtx, blk.Wo, blk.bo, this._tmpD, L, d, d);

      // Residual1
      for (let i = 0; i < L * d; i++) blk.xRes1[i] = blk.xIn[i] + this._tmpD[i];

      // LN2
      this._layerNormForward(
        blk.xRes1,
        blk.ln2Gamma,
        blk.ln2Beta,
        blk.ln2Out,
        blk.ln2Mean,
        blk.ln2InvStd,
        blk.ln2Xhat,
        L,
        d,
      );

      // FFN: pre = ln2Out * W1 + b1  => [L x hidden]
      this._linearForward2D(
        blk.ln2Out,
        blk.W1,
        blk.b1,
        blk.ffnPre,
        L,
        d,
        this._hidden,
      );

      // act = gelu(pre)
      for (let i = 0; i < L * this._hidden; i++) {
        blk.ffnAct[i] = this._gelu(blk.ffnPre[i]);
      }

      // out = act * W2 + b2 => [L x d]
      this._linearForward2D(
        blk.ffnAct,
        blk.W2,
        blk.b2,
        blk.ffnOut,
        L,
        this._hidden,
        d,
      );

      // Residual2 output
      for (let i = 0; i < L * d; i++) {
        blk.xOut[i] = blk.xRes1[i] + blk.ffnOut[i];
      }

      cur = blk.xOut;
    }

    // 5) Temporal aggregation: alpha = softmax(H*Wpool + bpool); pooled = Σ alpha_i * H_i
    {
      const H = cur;
      const Wp = this._Wpool;
      const bp = this._bpool[0];
      const scores = this._poolScores;
      const alpha = this._poolAlpha;

      // scores[t] = dot(H[t], Wp) + bp
      for (let t = 0; t < L; t++) {
        const base = t * d;
        let s = bp;
        for (let j = 0; j < d; j++) s += H[base + j] * Wp[j];
        scores[t] = s;
      }

      // softmax(scores) -> alpha
      this._softmax1D(scores, alpha, L);

      // pooled = Σ alpha[t] * H[t]
      const pooled = this._pooled;
      for (let j = 0; j < d; j++) pooled[j] = 0.0;
      for (let t = 0; t < L; t++) {
        const a = alpha[t];
        const base = t * d;
        for (let j = 0; j < d; j++) pooled[j] += a * H[base + j];
      }
    }

    // 6) Output: yHatNorm = pooled * Wout + bout
    {
      const outDim = this._outputDim;
      const yHat = this._yHatNorm;
      const Wout = this._Wout;
      const bout = this._bout;
      for (let j = 0; j < outDim; j++) yHat[j] = bout[j];
      for (let i = 0; i < d; i++) {
        const pv = this._pooled[i];
        const wBase = i * outDim;
        for (let j = 0; j < outDim; j++) yHat[j] += pv * Wout[wBase + j];
      }
    }
  }

  private _backward(L: number): void {
    const d = this._d;
    const outDim = this._outputDim;

    // Backprop output layer
    // yHat = pooled * Wout + bout
    // dWout += pooled^T * dY, dbout += dY, dPooled = dY * Wout^T
    {
      const pW = this._findParamByName("Wout")!;
      const pb = this._findParamByName("bout")!;
      const dY = this._dYHat;
      const pooled = this._pooled;

      // dbout
      for (let j = 0; j < outDim; j++) pb.g[j] += dY[j];

      // dWout
      for (let i = 0; i < d; i++) {
        const pv = pooled[i];
        const wBase = i * outDim;
        for (let j = 0; j < outDim; j++) pW.g[wBase + j] += pv * dY[j];
      }

      // dPooled = dY * Wout^T
      const dP = this._dPooled;
      for (let i = 0; i < d; i++) dP[i] = 0.0;
      for (let i = 0; i < d; i++) {
        const wBase = i * outDim;
        let sum = 0.0;
        for (let j = 0; j < outDim; j++) sum += dY[j] * this._Wout[wBase + j];
        dP[i] = sum;
      }
    }

    // Backprop pooling
    // pooled = Σ alpha[t]*H[t]
    // alpha = softmax(scores), scores[t]=dot(H[t],Wpool)+bpool
    // dH[t] += alpha[t]*dPooled + dScore[t]*Wpool
    // dWpool += H[t]*dScore[t], dbpool += dScore[t]
    // where dAlpha[t] = dot(dPooled, H[t])
    // and dScore = softmaxBackward(alpha, dAlpha)
    const dH_last = this._tmpD; // reuse as dH for final transformer output: [L*d]
    dH_last.fill(0, 0, L * d);

    {
      // Identify final H pointer
      const H = (this._blocks.length > 0)
        ? this._blocks[this._blocks.length - 1].xOut
        : this._fused;
      const alpha = this._poolAlpha;
      const dP = this._dPooled;

      // base contribution dH += alpha*dP
      for (let t = 0; t < L; t++) {
        const a = alpha[t];
        const base = t * d;
        for (let j = 0; j < d; j++) dH_last[base + j] += a * dP[j];
      }

      // dAlpha[t] = dot(dP, H[t])
      const dAlpha = this._poolScores; // reuse scores buffer to store dAlpha
      for (let t = 0; t < L; t++) {
        const base = t * d;
        let s = 0.0;
        for (let j = 0; j < d; j++) s += dP[j] * H[base + j];
        dAlpha[t] = s;
      }

      // softmax backward: dScore[t] = alpha[t]*(dAlpha[t] - Σ alpha*u dAlpha[u])
      let sum = 0.0;
      for (let t = 0; t < L; t++) sum += alpha[t] * dAlpha[t];
      const dScore = dAlpha; // in-place overwrite ok
      for (let t = 0; t < L; t++) dScore[t] = alpha[t] * (dScore[t] - sum);

      // grads for Wpool, bpool and add to dH
      const pWp = this._findParamByName("Wpool")!;
      const pBp = this._findParamByName("bpool")!;
      const Wp = this._Wpool;

      // dbpool
      let db = 0.0;
      for (let t = 0; t < L; t++) db += dScore[t];
      pBp.g[0] += db;

      // dWpool += Σ H[t]*dScore[t]
      for (let j = 0; j < d; j++) {
        let acc = 0.0;
        for (let t = 0; t < L; t++) acc += H[t * d + j] * dScore[t];
        pWp.g[j] += acc;
      }

      // dH += dScore[t] * Wpool
      for (let t = 0; t < L; t++) {
        const ds = dScore[t];
        const base = t * d;
        for (let j = 0; j < d; j++) dH_last[base + j] += ds * Wp[j];
      }
    }

    // Backprop through transformer blocks (reverse)
    for (let bi = this._blocks.length - 1; bi >= 0; bi--) {
      const blk = this._blocks[bi];
      const prefix = `block${bi}.`;

      // dXout is currently in dH_last (for last block) or in tmpD carried across
      // We'll use:
      // dXout (L*d) = dH_last
      // Residual2: xOut = xRes1 + ffnOut => dffnOut += dXout, dxRes1 += dXout
      const dXout = dH_last;
      const dFfnOut = this._tmpD; // reuse tmpD, but we need dXout still. Copy dXout into tmpD2? use _dFused as scratch.
      // We'll use _dFused as scratch for dRes1 to avoid overwrite.
      const dRes1 = this._dFused;
      // dRes1 = dXout
      this._copyBlockSeq(dRes1, dXout, L * d);

      // dFfnOut = dXout
      this._copyBlockSeq(dFfnOut, dXout, L * d);

      // FFN backward
      // ffnOut = act*W2 + b2
      // act = gelu(ffnPre)
      // ffnPre = ln2Out*W1 + b1
      // ln2Out = LN2(xRes1)
      {
        const pW2 = this._findParamByName(prefix + "W2")!;
        const pb2 = this._findParamByName(prefix + "b2")!;
        const pW1 = this._findParamByName(prefix + "W1")!;
        const pb1 = this._findParamByName(prefix + "b1")!;
        const pG2 = this._findParamByName(prefix + "ln2Gamma")!;
        const pB2 = this._findParamByName(prefix + "ln2Beta")!;

        // db2 += sum dFfnOut
        for (let j = 0; j < d; j++) {
          let acc = 0.0;
          for (let t = 0; t < L; t++) acc += dFfnOut[t * d + j];
          pb2.g[j] += acc;
        }

        // Need act for dW2 and dAct: recompute act = gelu(ffnPre) into blk.ffnAct (already there)
        // dW2 += act^T * dFfnOut  where act shape [L x hidden], dFfnOut [L x d], W2 [hidden x d]
        const hidden = this._hidden;
        // dW2
        for (let i = 0; i < hidden; i++) {
          const wBase = i * d;
          for (let j = 0; j < d; j++) {
            let acc = 0.0;
            for (let t = 0; t < L; t++) {
              acc += blk.ffnAct[t * hidden + i] * dFfnOut[t * d + j];
            }
            pW2.g[wBase + j] += acc;
          }
        }

        // dAct = dFfnOut * W2^T  => [L x hidden]
        const dAct = this._tmpHidden; // [L*hidden]
        for (let t = 0; t < L; t++) {
          const dBase = t * hidden;
          const dyBase = t * d;
          for (let i = 0; i < hidden; i++) dAct[dBase + i] = 0.0;
          for (let j = 0; j < d; j++) {
            const dy = dFfnOut[dyBase + j];
            // W2 row-major [hidden x d], so W2[i*d+j]
            for (let i = 0; i < hidden; i++) {
              dAct[dBase + i] += dy * blk.W2[i * d + j];
            }
          }
        }

        // dPre = dAct * gelu'(pre)
        const dPre = dAct; // in-place
        for (let i = 0; i < L * hidden; i++) {
          dPre[i] = dPre[i] * this._geluDeriv(blk.ffnPre[i]);
        }

        // db1 += sum dPre
        for (let i = 0; i < hidden; i++) {
          let acc = 0.0;
          for (let t = 0; t < L; t++) acc += dPre[t * hidden + i];
          pb1.g[i] += acc;
        }

        // dW1 += ln2Out^T * dPre  where ln2Out [L x d], dPre [L x hidden], W1 [d x hidden]
        for (let j = 0; j < d; j++) {
          const wBase = j * hidden;
          for (let i = 0; i < hidden; i++) {
            let acc = 0.0;
            for (let t = 0; t < L; t++) {
              acc += blk.ln2Out[t * d + j] * dPre[t * hidden + i];
            }
            pW1.g[wBase + i] += acc;
          }
        }

        // dLn2Out = dPre * W1^T => [L x d]
        const dLn2Out = this._tmpD; // [L*d]
        for (let t = 0; t < L; t++) {
          const base = t * d;
          const dpBase = t * hidden;
          for (let j = 0; j < d; j++) dLn2Out[base + j] = 0.0;
          for (let i = 0; i < hidden; i++) {
            const dp = dPre[dpBase + i];
            // W1 row-major [d x hidden], so W1[j*hidden+i]
            for (let j = 0; j < d; j++) {
              dLn2Out[base + j] += dp * blk.W1[j * hidden + i];
            }
          }
        }

        // LayerNorm2 backward: inputs are xRes1, cache ln2Xhat, mean, invStd
        // Outputs gradient into dRes1 (add)
        this._layerNormBackward(
          blk.ln2Xhat,
          blk.ln2InvStd,
          blk.ln2Gamma,
          dLn2Out,
          pG2.g,
          pB2.g,
          dRes1,
          L,
          d,
        );
      }

      // Residual1: xRes1 = xIn + attnOut => dxIn += dRes1, dAttnOut += dRes1
      const dAttnOut = this._tmpD; // reuse [L*d]
      this._copyBlockSeq(dAttnOut, dRes1, L * d);

      // accumulate into dXin (will be used after attn backward)
      const dXin = this._dFused; // reuse as dxIn [L*d]
      this._copyBlockSeq(dXin, dRes1, L * d);

      // Attention backward:
      // attnOut = attnCtx*Wo + bo
      // attnCtx = MHA(Q,K,V)
      // Q = ln1Out*Wq + bq, etc
      // ln1Out = LN1(xIn)
      {
        const pWo = this._findParamByName(prefix + "Wo")!;
        const pbo = this._findParamByName(prefix + "bo")!;
        const pWq = this._findParamByName(prefix + "Wq")!;
        const pbq = this._findParamByName(prefix + "bq")!;
        const pWk = this._findParamByName(prefix + "Wk")!;
        const pbk = this._findParamByName(prefix + "bk")!;
        const pWv = this._findParamByName(prefix + "Wv")!;
        const pbv = this._findParamByName(prefix + "bv")!;
        const pG1 = this._findParamByName(prefix + "ln1Gamma")!;
        const pB1 = this._findParamByName(prefix + "ln1Beta")!;

        // Recompute attnCtx from stored Q,K,V (already from forward)
        this._mhaCausalForward(blk.Q, blk.K, blk.V, blk.attnCtx, L);

        // bo grad
        for (let j = 0; j < d; j++) {
          let acc = 0.0;
          for (let t = 0; t < L; t++) acc += dAttnOut[t * d + j];
          pbo.g[j] += acc;
        }

        // Wo grad: attnCtx^T * dAttnOut
        for (let i = 0; i < d; i++) {
          const wBase = i * d;
          for (let j = 0; j < d; j++) {
            let acc = 0.0;
            for (let t = 0; t < L; t++) {
              acc += blk.attnCtx[t * d + i] * dAttnOut[t * d + j];
            }
            pWo.g[wBase + j] += acc;
          }
        }

        // dAttnCtx = dAttnOut * Wo^T  => [L x d]
        const dCtx = this._tmpD; // [L*d]
        for (let t = 0; t < L; t++) {
          const base = t * d;
          for (let i = 0; i < d; i++) dCtx[base + i] = 0.0;
          for (let j = 0; j < d; j++) {
            const dy = dAttnOut[base + j];
            for (let i = 0; i < d; i++) {
              dCtx[base + i] += dy * blk.Wo[i * d + j];
            }
          }
        }

        // MHA backward to get dQ,dK,dV (into tmp buffers)
        const dQ = this._tmpD; // reuse [L*d] (will overwrite dCtx, but okay after)
        const dK = this._dFused; // [L*d]
        const dV = this._dxEmb; // [L*d] (scratch)
        this._mhaCausalBackward(blk.Q, blk.K, blk.V, dCtx, dQ, dK, dV, L);

        // Projection grads:
        // dWq += ln1Out^T * dQ, dbq += sum dQ, dLn1Out += dQ*Wq^T
        // similarly for K,V; accumulate into dLn1Out
        const dLn1Out = this._tmpHidden; // use hidden scratch as [L*d] by viewing prefix (safe size)
        // Ensure size
        // (tmpHidden length is maxSeqLen*hidden >= maxSeqLen*d)
        for (let i = 0; i < L * d; i++) dLn1Out[i] = 0.0;

        // dbq, dbk, dbv
        for (let j = 0; j < d; j++) {
          let aq = 0.0, ak = 0.0, av = 0.0;
          for (let t = 0; t < L; t++) {
            const base = t * d + j;
            aq += dQ[base];
            ak += dK[base];
            av += dV[base];
          }
          pbq.g[j] += aq;
          pbk.g[j] += ak;
          pbv.g[j] += av;
        }

        // dWq,dWk,dWv and dLn1Out
        for (let i = 0; i < d; i++) {
          const wBase = i * d;
          for (let j = 0; j < d; j++) {
            let aq = 0.0, ak = 0.0, av = 0.0;
            for (let t = 0; t < L; t++) {
              const x = blk.ln1Out[t * d + i];
              aq += x * dQ[t * d + j];
              ak += x * dK[t * d + j];
              av += x * dV[t * d + j];
            }
            pWq.g[wBase + j] += aq;
            pWk.g[wBase + j] += ak;
            pWv.g[wBase + j] += av;
          }
        }

        // dLn1Out += dQ*Wq^T + dK*Wk^T + dV*Wv^T
        for (let t = 0; t < L; t++) {
          const base = t * d;
          for (let i = 0; i < d; i++) {
            let sum = 0.0;
            // Wq^T: Wq[j*d + i] corresponds to (i,j) in transpose, but Wq is row-major [i*d+j].
            // Compute sum_j dQ[j]*Wq[i,j]? Actually dX = dY * W^T => dX[i] = Σ_j dY[j]*W[i,j]
            // So use W row-major as is: W[i*d + j]
            const rowQ = i * d;
            for (let j = 0; j < d; j++) sum += dQ[base + j] * blk.Wq[rowQ + j];
            const rowK = i * d;
            for (let j = 0; j < d; j++) sum += dK[base + j] * blk.Wk[rowK + j];
            const rowV = i * d;
            for (let j = 0; j < d; j++) sum += dV[base + j] * blk.Wv[rowV + j];
            dLn1Out[base + i] += sum;
          }
        }

        // LayerNorm1 backward: input xIn, cache ln1Xhat, invStd
        // Output gradient into dXin (add)
        this._layerNormBackward(
          blk.ln1Xhat,
          blk.ln1InvStd,
          blk.ln1Gamma,
          dLn1Out as any as Float64Array,
          pG1.g,
          pB1.g,
          dXin,
          L,
          d,
        );
      }

      // Now dXin contains gradients for blk.xIn. If this is first block, it maps to fused.
      // Prepare dH_last for previous block: dH_prev = dXin (since blk.xIn was previous output)
      this._copyBlockSeq(dH_last, dXin, L * d);
    }

    // Backprop into fused input if there are no blocks
    if (this._blocks.length === 0) {
      // dFused = dH_last already (which currently holds gradient after pooling)
      this._copyBlockSeq(this._dFused, dH_last, L * d);
    } else {
      // After loop, dH_last holds grad for fused (input to block0)
      this._copyBlockSeq(this._dFused, dH_last, L * d);
    }

    // Backprop through fusion gates into per-scale E and Wg/bg
    {
      const dFused = this._dFused;
      const sCount = this._scaleCount;
      const concatDim = sCount * d;
      const gates = this._gates;
      const Wg = this._Wg;

      const pWg = this._findParamByName("Wg")!;
      const pbg = this._findParamByName("bg")!;

      // dE accumulators per scale buffers in convAct (reuse convAct as dE) to avoid new alloc
      // We'll use convAct[s] as dE buffer (zero then accumulate).
      for (let s = 0; s < sCount; s++) {
        const stride = this._scales[s];
        const outLen = (L / stride) | 0;
        const dE = this._convAct[s];
        dE.fill(0, 0, outLen * d);
      }

      // For each time t
      for (let t = 0; t < L; t++) {
        // reconstruct concat from Es
        let off = 0;
        const idxP: Int32Array = new Int32Array(sCount);
        for (let s = 0; s < sCount; s++) {
          const stride = this._scales[s];
          let p = (t / stride) | 0;
          const outLen = (L / stride) | 0;
          if (p < 0) p = 0;
          else if (p >= outLen) p = outLen - 1;
          idxP[s] = p;
          const E = this._Escale[s];
          const base = p * d;
          for (let c = 0; c < d; c++) this._concat[off++] = E[base + c];
        }

        // 1) contribution to dE from fused weighted sum: dE += gate * dFused
        const gBase = t * sCount;
        const dfBase = t * d;
        for (let s = 0; s < sCount; s++) {
          const g = gates[gBase + s];
          const p = idxP[s];
          const dE = this._convAct[s]; // used as dE buffer
          const dEBase = p * d;
          for (let c = 0; c < d; c++) dE[dEBase + c] += g * dFused[dfBase + c];
        }

        // 2) dGates[s] = dot(dFused[t], E_s[p])
        const dGate = this._poolScores; // reuse [maxSeqLen] as dGate per s by using first sCount elements
        for (let s = 0; s < sCount; s++) {
          const p = idxP[s];
          const E = this._Escale[s];
          const eBase = p * d;
          let dot = 0.0;
          for (let c = 0; c < d; c++) dot += dFused[dfBase + c] * E[eBase + c];
          dGate[s] = dot;
        }

        // 3) sigmoid backward: dLogit = dGate * g*(1-g)
        const dLogit = this._poolAlpha; // reuse [maxSeqLen] as dLogit for sCount
        for (let s = 0; s < sCount; s++) {
          const g = gates[gBase + s];
          dLogit[s] = dGate[s] * g * (1.0 - g);
        }

        // 4) dWg += concat * dLogit, dbg += dLogit
        for (let s = 0; s < sCount; s++) pbg.g[s] += dLogit[s];

        for (let j = 0; j < concatDim; j++) {
          const x = this._concat[j];
          const base = j * sCount;
          for (let s = 0; s < sCount; s++) pWg.g[base + s] += x * dLogit[s];
        }

        // 5) dConcat = Wg * dLogit (since concat feeds gate linear)
        // dConcat[j] = Σ_s Wg[j,s] * dLogit[s]
        // Then add to corresponding dE buffers (since concat is concatenation of Es)
        off = 0;
        for (let s = 0; s < sCount; s++) {
          const p = idxP[s];
          const dE = this._convAct[s];
          const dEBase = p * d;
          for (let c = 0; c < d; c++) {
            // j = off + c
            const j = off + c;
            const base = j * sCount;
            let acc = 0.0;
            for (let ss = 0; ss < sCount; ss++) {
              acc += Wg[base + ss] * dLogit[ss];
            }
            dE[dEBase + c] += acc;
          }
          off += d;
        }
      }
    }

    // Backprop per-scale: Es = convAct + PE + scaleEmb; convAct=gelu(convPre)
    // convPre = depthwise conv over xEmb
    // Need grads for scaleEmb, Wconv, bconv, and dXemb accumulation
    this._dxEmb.fill(0, 0, L * d);
    {
      const pWconv = this._findParamByName("Wconv")!;
      const pbconv = this._findParamByName("bconv")!;
      const pSE = this._findParamByName("scaleEmb")!;

      for (let s = 0; s < this._scaleCount; s++) {
        const stride = this._scales[s];
        const outLen = (L / stride) | 0;
        const dE = this._convAct[s]; // dE buffer
        const convPre = this._convPre[s];
        const convAct = this._convAct[s]; // same buffer, but currently holds dE; we must be careful
        // We'll compute dAct into tmp buffer (reuse Escale[s] as tmp dAct), then overwrite dE? No.
        const dAct = this._Escale[s]; // reuse as dAct buffer
        // dAct = dE (since Es = act + PE + scaleEmb)
        for (let i = 0; i < outLen * d; i++) dAct[i] = dE[i];

        // scaleEmb grads: sum over positions
        const seBase = s * d;
        for (let c = 0; c < d; c++) {
          let acc = 0.0;
          for (let p = 0; p < outLen; p++) acc += dAct[p * d + c];
          pSE.g[seBase + c] += acc;
        }

        // dPre = dAct * gelu'(pre)
        const dPre = dAct; // in-place
        for (let i = 0; i < outLen * d; i++) {
          dPre[i] = dPre[i] * this._geluDeriv(convPre[i]);
        }

        // bconv grads: sum over p
        const bBase = s * d;
        for (let c = 0; c < d; c++) {
          let acc = 0.0;
          for (let p = 0; p < outLen; p++) acc += dPre[p * d + c];
          pbconv.g[bBase + c] += acc;
        }

        // Wconv grads and dXemb accumulation
        const wBaseScale = s * this._kernelSize * d;
        for (let p = 0; p < outLen; p++) {
          const baseIn = (p * stride) | 0;
          const dpBase = p * d;
          for (let kk = 0; kk < this._kernelSize; kk++) {
            let idx = baseIn + kk - this._kernelHalf;
            if (idx < 0) idx = 0;
            else if (idx >= L) idx = L - 1;
            const xBase = idx * d;
            const wBase = wBaseScale + kk * d;
            for (let c = 0; c < d; c++) {
              const grad = dPre[dpBase + c];
              pWconv.g[wBase + c] += grad * this._xEmb[xBase + c];
              this._dxEmb[xBase + c] += grad * this._Wconv[wBase + c];
            }
          }
        }
      }
    }

    // Backprop input projection: xEmb = xNorm*Win + bin
    {
      const inDim = this._inputDim;
      const pWin = this._findParamByName("Win")!;
      const pbin = this._findParamByName("bin")!;
      const Xn = this._xNorm;
      const dXemb = this._dxEmb;

      // dbin += sum dXemb over time
      for (let j = 0; j < d; j++) {
        let acc = 0.0;
        for (let t = 0; t < L; t++) acc += dXemb[t * d + j];
        pbin.g[j] += acc;
      }

      // dWin += Xn^T * dXemb
      for (let k = 0; k < inDim; k++) {
        const wBase = k * d;
        for (let j = 0; j < d; j++) {
          let acc = 0.0;
          for (let t = 0; t < L; t++) {
            acc += Xn[t * inDim + k] * dXemb[t * d + j];
          }
          pWin.g[wBase + j] += acc;
        }
      }

      // We do not backprop into Xn -> raw x (not needed for training parameters).
    }
  }

  // ------------------------- Internal: Optimizer/Regularization -------------------------

  private _zeroGrads(): void {
    for (let i = 0; i < this._params.length; i++) {
      this._params[i].g.fill(0);
    }
  }

  private _applyL2(): void {
    const lambda = this._cfg.regularizationStrength;
    if (!(lambda > 0)) return;
    for (let i = 0; i < this._params.length; i++) {
      const p = this._params[i];
      if (!p.applyL2) continue;
      const w = p.w;
      const g = p.g;
      for (let j = 0; j < w.length; j++) g[j] += lambda * w[j];
    }
  }

  private _computeLearningRate(): number {
    const base = this._cfg.learningRate;
    const t = (this._updateCount + 1) | 0;
    const warm = this._cfg.warmupSteps | 0;
    const total = this._cfg.totalSteps | 0;

    let lr = base;

    // warmup
    if (warm > 0 && t < warm) {
      lr *= t / warm;
    }

    // cosine decay after warmup
    if (total > warm && t > warm) {
      const prog = Math.min(1.0, (t - warm) / (total - warm));
      const cos = 0.5 * (1.0 + Math.cos(Math.PI * prog));
      lr *= cos;
    }

    return lr;
  }

  private _adamStep(lr: number): number {
    const b1 = this._cfg.beta1;
    const b2 = this._cfg.beta2;
    const eps = this._cfg.epsilon;
    const t = (this._updateCount + 1) | 0;

    const b1t = Math.pow(b1, t);
    const b2t = Math.pow(b2, t);
    const inv1 = 1.0 / (1.0 - b1t);
    const inv2 = 1.0 / (1.0 - b2t);

    let g2sum = 0.0;

    for (let i = 0; i < this._params.length; i++) {
      const p = this._params[i];
      const w = p.w;
      const g = p.g;
      const m = p.m;
      const v = p.v;

      for (let j = 0; j < w.length; j++) {
        const gj = g[j];
        g2sum += gj * gj;

        const mj = m[j] = b1 * m[j] + (1.0 - b1) * gj;
        const vj = v[j] = b2 * v[j] + (1.0 - b2) * (gj * gj);

        const mHat = mj * inv1;
        const vHat = vj * inv2;

        w[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }

    return Math.sqrt(g2sum);
  }

  // ------------------------- Internal: Layers -------------------------

  private _linearForward2D(
    X: Float64Array,
    W: Float64Array,
    b: Float64Array,
    Y: Float64Array,
    rows: number,
    inCols: number,
    outCols: number,
  ): void {
    // X [rows x inCols], W [inCols x outCols], Y [rows x outCols]
    for (let r = 0; r < rows; r++) {
      const yBase = r * outCols;
      for (let j = 0; j < outCols; j++) Y[yBase + j] = b[j];
      const xBase = r * inCols;
      for (let k = 0; k < inCols; k++) {
        const xv = X[xBase + k];
        const wBase = k * outCols;
        for (let j = 0; j < outCols; j++) Y[yBase + j] += xv * W[wBase + j];
      }
    }
  }

  private _layerNormForward(
    X: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    Y: Float64Array,
    mean: Float64Array,
    invStd: Float64Array,
    xhat: Float64Array,
    rows: number,
    cols: number,
  ): void {
    const eps = 1e-5;
    for (let r = 0; r < rows; r++) {
      const base = r * cols;

      // mean
      let m = 0.0;
      for (let c = 0; c < cols; c++) m += X[base + c];
      m /= cols;

      // var
      let v = 0.0;
      for (let c = 0; c < cols; c++) {
        const d = X[base + c] - m;
        v += d * d;
      }
      v /= cols;

      const inv = 1.0 / Math.sqrt(v + eps);
      mean[r] = m;
      invStd[r] = inv;

      for (let c = 0; c < cols; c++) {
        const nh = (X[base + c] - m) * inv;
        xhat[base + c] = nh;
        Y[base + c] = nh * gamma[c] + beta[c];
      }
    }
  }

  /**
   * LayerNorm backward (per-row):
   * Given xhat, invStd, gamma and dY:
   * dBeta += Σ dY
   * dGamma += Σ dY * xhat
   * dX = (1/cols)*invStd * (cols*dYg - Σ dYg - xhat*Σ(dYg*xhat))
   * where dYg = dY * gamma
   *
   * Adds gradient to dXout (accumulate).
   */
  private _layerNormBackward(
    xhat: Float64Array,
    invStd: Float64Array,
    gamma: Float64Array,
    dY: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
    dXout: Float64Array,
    rows: number,
    cols: number,
  ): void {
    const invCols = 1.0 / cols;

    // dBeta, dGamma
    for (let c = 0; c < cols; c++) {
      let db = 0.0;
      let dg = 0.0;
      for (let r = 0; r < rows; r++) {
        const idx = r * cols + c;
        const dy = dY[idx];
        db += dy;
        dg += dy * xhat[idx];
      }
      dBeta[c] += db;
      dGamma[c] += dg;
    }

    // dX
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      const inv = invStd[r];

      // compute sums over cols for this row
      let sum1 = 0.0; // Σ dYg
      let sum2 = 0.0; // Σ dYg * xhat
      for (let c = 0; c < cols; c++) {
        const idx = base + c;
        const dyg = dY[idx] * gamma[c];
        sum1 += dyg;
        sum2 += dyg * xhat[idx];
      }

      for (let c = 0; c < cols; c++) {
        const idx = base + c;
        const dyg = dY[idx] * gamma[c];
        const dx = inv * invCols * (cols * dyg - sum1 - xhat[idx] * sum2);
        dXout[idx] += dx;
      }
    }
  }

  private _mhaCausalForward(
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    out: Float64Array,
    L: number,
  ): void {
    const d = this._d;
    const h = this._h;
    const dh = this._dh;
    const invSqrt = 1.0 / Math.sqrt(dh);

    // out init
    for (let i = 0; i < L * d; i++) out[i] = 0.0;

    const win = this._cfg.attentionWindow | 0;

    for (let head = 0; head < h; head++) {
      const headOff = head * dh;

      for (let i = 0; i < L; i++) {
        const rowBase = (head * this._maxSeqLen + i) * this._maxSeqLen;
        const jEnd = i;
        const jStart = (win > 0) ? Math.max(0, i - win) : 0;

        // compute scores for j in [jStart..jEnd]
        let max = -1e300;
        for (let j = jStart; j <= jEnd; j++) {
          let dot = 0.0;
          const qi = i * d + headOff;
          const kj = j * d + headOff;
          for (let k = 0; k < dh; k++) dot += Q[qi + k] * K[kj + k];
          const s = dot * invSqrt;
          this._attnScores[rowBase + j] = s;
          if (s > max) max = s;
        }

        // softmax
        let sum = 0.0;
        for (let j = jStart; j <= jEnd; j++) {
          const e = Math.exp(this._attnScores[rowBase + j] - max);
          this._attnProbs[rowBase + j] = e;
          sum += e;
        }
        const invSum = 1.0 / (sum + 1e-12);
        for (let j = jStart; j <= jEnd; j++) {
          this._attnProbs[rowBase + j] *= invSum;
        }

        // context = Σ prob * V[j]
        const outBase = i * d + headOff;
        for (let k = 0; k < dh; k++) {
          let acc = 0.0;
          for (let j = jStart; j <= jEnd; j++) {
            const p = this._attnProbs[rowBase + j];
            acc += p * V[j * d + headOff + k];
          }
          out[outBase + k] = acc;
        }
      }
    }
  }

  /**
   * MHA causal backward:
   * Inputs: Q,K,V, dOut (gradient wrt context output, shape [L x d])
   * Outputs: dQ,dK,dV (shape [L x d])
   *
   * Notes:
   * - Recomputes probs from stored _attnProbs as populated by a prior _mhaCausalForward call.
   * - Uses standard softmax backward.
   */
  private _mhaCausalBackward(
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    dOut: Float64Array,
    dQ: Float64Array,
    dK: Float64Array,
    dV: Float64Array,
    L: number,
  ): void {
    const d = this._d;
    const h = this._h;
    const dh = this._dh;
    const invSqrt = 1.0 / Math.sqrt(dh);
    const win = this._cfg.attentionWindow | 0;

    dQ.fill(0, 0, L * d);
    dK.fill(0, 0, L * d);
    dV.fill(0, 0, L * d);

    for (let head = 0; head < h; head++) {
      const headOff = head * dh;

      for (let i = 0; i < L; i++) {
        const rowBase = (head * this._maxSeqLen + i) * this._maxSeqLen;
        const jEnd = i;
        const jStart = (win > 0) ? Math.max(0, i - win) : 0;

        // dV accum: dV[j] += prob_ij * dOut[i]
        // dP_ij = dot(dOut[i], V[j])
        // Then softmax backward: dS_ij = P_ij * (dP_ij - Σ_k P_ik dP_ik)
        // dQ[i] += Σ_j dS_ij * K[j] / sqrt(dh)
        // dK[j] += dS_ij * Q[i] / sqrt(dh)

        // compute dP for each j (store in scores buffer row as scratch)
        let sumPdP = 0.0;
        for (let j = jStart; j <= jEnd; j++) {
          const p = this._attnProbs[rowBase + j];
          // dV
          const dvBase = j * d + headOff;
          const doBase = i * d + headOff;
          for (let k = 0; k < dh; k++) dV[dvBase + k] += p * dOut[doBase + k];

          // dP = dot(dOut[i], V[j])
          let dot = 0.0;
          const vBase = j * d + headOff;
          for (let k = 0; k < dh; k++) dot += dOut[doBase + k] * V[vBase + k];
          this._attnScores[rowBase + j] = dot; // reuse as dP
          sumPdP += p * dot;
        }

        // dS and accumulate dQ/dK
        const qiBase = i * d + headOff;
        for (let j = jStart; j <= jEnd; j++) {
          const p = this._attnProbs[rowBase + j];
          const dP = this._attnScores[rowBase + j];
          const dS = p * (dP - sumPdP);

          // dQ[i] += dS * K[j] / sqrt
          const kjBase = j * d + headOff;
          for (let k = 0; k < dh; k++) {
            dQ[qiBase + k] += dS * K[kjBase + k] * invSqrt;
          }

          // dK[j] += dS * Q[i] / sqrt
          for (let k = 0; k < dh; k++) {
            dK[kjBase + k] += dS * Q[qiBase + k] * invSqrt;
          }
        }
      }
    }
  }

  // ------------------------- Internal: Math helpers -------------------------

  private _buildPositionalEncoding(
    pe: Float64Array,
    maxLen: number,
    d: number,
  ): void {
    // PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
    const div = 10000.0;
    for (let pos = 0; pos < maxLen; pos++) {
      const base = pos * d;
      for (let i = 0; i < d; i += 2) {
        const k = i / d;
        const denom = Math.pow(div, k);
        const v = pos / denom;
        pe[base + i] = Math.sin(v);
        if (i + 1 < d) pe[base + i + 1] = Math.cos(v);
      }
    }
  }

  private _softmax1D(x: Float64Array, out: Float64Array, n: number): void {
    let max = -1e300;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      if (v > max) max = v;
    }
    let sum = 0.0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(x[i] - max);
      out[i] = e;
      sum += e;
    }
    const inv = 1.0 / (sum + 1e-12);
    for (let i = 0; i < n; i++) out[i] *= inv;
  }

  private _sigmoid(x: number): number {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1.0 / (1.0 + z);
    } else {
      const z = Math.exp(x);
      return z / (1.0 + z);
    }
  }

  private _gelu(x: number): number {
    // tanh approximation
    // gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const c = 0.7978845608028654; // sqrt(2/pi)
    const x3 = x * x * x;
    const u = c * (x + 0.044715 * x3);
    const t = Math.tanh(u);
    return 0.5 * x * (1.0 + t);
  }

  private _geluDeriv(x: number): number {
    // derivative of tanh-approx GELU
    const c = 0.7978845608028654; // sqrt(2/pi)
    const x2 = x * x;
    const x3 = x2 * x;
    const u = c * (x + 0.044715 * x3);
    const t = Math.tanh(u);
    const sech2 = 1.0 - t * t;
    const du = c * (1.0 + 3.0 * 0.044715 * x2);
    // d/dx 0.5*x*(1+t) = 0.5*(1+t) + 0.5*x*sech2*du
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
  }

  // ------------------------- Internal: Utility -------------------------

  private _num(v: any, d: number): number {
    const x = +v;
    return Number.isFinite(x) ? x : d;
  }

  private _rand(): number {
    // LCG: x = (a*x + c) mod 2^32
    this._rngState = (this._rngState * 1664525 + 1013904223) | 0;
    // [0,1)
    return ((this._rngState >>> 0) / 4294967296.0);
  }

  private _xavierUniform(w: Float64Array, fanIn: number, fanOut: number): void {
    const limit = Math.sqrt(6.0 / (fanIn + fanOut));
    for (let i = 0; i < w.length; i++) {
      w[i] = (this._rand() * 2.0 - 1.0) * limit;
    }
  }

  private _smallUniform(w: Float64Array, scale: number): void {
    for (let i = 0; i < w.length; i++) {
      w[i] = (this._rand() * 2.0 - 1.0) * scale;
    }
  }

  private _copyFromArray(dst: Float64Array, src: any): void {
    if (!src || !Array.isArray(src)) return;
    const n = Math.min(dst.length, src.length);
    for (let i = 0; i < n; i++) dst[i] = +src[i];
  }

  private _copyBlockSeq(dst: Float64Array, src: Float64Array, n: number): void {
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }

  private _accuracy(): number {
    const Lbar = this._lossAvg;
    return 1.0 / (1.0 + (Lbar >= 0 ? Lbar : 0.0));
  }

  private _totalParams(): number {
    let sum = 0;
    for (let i = 0; i < this._params.length; i++) {
      sum += this._params[i].w.length;
    }
    return sum;
  }

  private _packWB(W: Float64Array, b: Float64Array): number[] {
    const out = new Array<number>(W.length + b.length);
    let k = 0;
    for (let i = 0; i < W.length; i++) out[k++] = W[i];
    for (let i = 0; i < b.length; i++) out[k++] = b[i];
    return out;
  }

  private _packVec2(a: Float64Array, b: Float64Array): number[] {
    const out = new Array<number>(a.length + b.length);
    let k = 0;
    for (let i = 0; i < a.length; i++) out[k++] = a[i];
    for (let i = 0; i < b.length; i++) out[k++] = b[i];
    return out;
  }
}
