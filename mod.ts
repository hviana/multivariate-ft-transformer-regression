/* FusionTemporalTransformerRegression - pure TypeScript, no external deps.
   Implements: multi-scale temporal conv embedding + gated fusion + transformer blocks (MHA + FFN) + attention pooling + dense output
   Training: online Adam w/ cosine warmup schedule, Welford z-score normalization, L2 regularization, outlier downweighting, ADWIN drift detection

   Notes on numerical stability:
   - LayerNorm uses epsilon in variance
   - Softmax uses max-subtraction
   - Sigmoid is stable for large |x|
*/

export interface Config {
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  ffnMultiplier: number;

  attentionDropout: number;
  fusionDropout: number;

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
}

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
  temporalConvWeights: number[][][];
  scaleEmbeddings: number[][];
  positionalEncoding: number[][];
  fusionWeights: number[][][];
  attentionWeights: number[][][];
  ffnWeights: number[][][];
  layerNormParams: number[][][];
  outputWeights: number[][][];
  firstMoment: number[][][];
  secondMoment: number[][][];
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

type _Mat = {
  name: string;
  rows: number;
  cols: number;
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
  l2: boolean;
};

type _Block = {
  // LayerNorm1
  ln1Gamma: _Mat; // 1 x D
  ln1Beta: _Mat; // 1 x D
  // Attention projections
  wq: _Mat; // D x D
  wk: _Mat; // D x D
  wv: _Mat; // D x D
  wo: _Mat; // D x D
  bo: _Mat; // 1 x D
  // temporal bias slope per head
  attnSlope: _Mat; // 1 x H
  // LayerNorm2
  ln2Gamma: _Mat;
  ln2Beta: _Mat;
  // FFN
  w1: _Mat; // D x (D*ffnMult)
  b1: _Mat; // 1 x hidden
  w2: _Mat; // hidden x D
  b2: _Mat; // 1 x D
};

type _Fusion = {
  convK: _Mat[]; // per scale: D x (inDim*K)
  convB: _Mat[]; // per scale: 1 x D
  scaleEmb: _Mat; // S x D
  gateW: _Mat; // (S*D) x S
  gateB: _Mat; // 1 x S
};

type _PoolOut = {
  poolW: _Mat; // 1 x D
  poolB: _Mat; // 1 x 1
  outW: _Mat; // D x outDim
  outB: _Mat; // 1 x outDim
};

export class FusionTemporalTransformerRegression {
  private _cfg: Config;

  private _isInitialized = false;
  private _inDim = 0;
  private _outDim = 0;

  // Adam
  private _step = 0;
  private _effectiveLR = 0;

  // Running metrics
  private _sampleCount = 0;
  private _lossSum = 0;
  private _accuracy = 0;
  private _prevLoss = Number.POSITIVE_INFINITY;
  private _converged = false;

  // Drift
  private _driftCount = 0;

  // Welford stats (separate counts for X and Y)
  private _xCount = 0;
  private _yCount = 0;
  private _xMean!: Float64Array;
  private _xM2!: Float64Array;
  private _xStd!: Float64Array;

  private _yMean!: Float64Array;
  private _yM2!: Float64Array;
  private _yStd!: Float64Array;

  // Positional encoding (maxSeq x D)
  private _posEnc!: Float64Array;

  // Parameters
  private _fusion!: _Fusion;
  private _blocks!: _Block[];
  private _po!: _PoolOut;
  private _params!: _Mat[];

  // Buffers
  private _seqCap = 0;
  private _lastSeqLen = 0;

  private _xRaw!: Float64Array; // seqCap*inDim
  private _xNorm!: Float64Array; // seqCap*inDim
  private _yRaw!: Float64Array; // outDim
  private _yNorm!: Float64Array; // outDim
  private _lastXNormSeq!: Float64Array;

  // Fusion caches
  private _scaleOutLen!: Int32Array;
  private _convOff!: Int32Array;
  private _convPre!: Float64Array; // packed (sum outLen*D)
  private _convAct!: Float64Array; // packed
  private _eUp!: Float64Array; // S*seqCap*D
  private _gateZ!: Float64Array; // seqCap*S
  private _gateSig!: Float64Array; // seqCap*S
  private _fused!: Float64Array; // seqCap*D
  private _fusionDropMask!: Uint8Array;

  // Transformer caches
  private _blockIn!: Float64Array; // B*seqCap*D
  private _afterAttn!: Float64Array; // B*seqCap*D

  private _ln1Xhat!: Float64Array; // B*seqCap*D
  private _ln1Mean!: Float64Array; // B*seqCap
  private _ln1InvStd!: Float64Array; // B*seqCap

  private _ln2Xhat!: Float64Array;
  private _ln2Mean!: Float64Array;
  private _ln2InvStd!: Float64Array;

  private _q!: Float64Array; // B*seqCap*D
  private _k!: Float64Array;
  private _v!: Float64Array;
  private _ctx!: Float64Array; // B*seqCap*D
  private _attnOut!: Float64Array; // B*seqCap*D

  private _dQ!: Float64Array; // B*seqCap*D
  private _dK!: Float64Array;
  private _dV!: Float64Array;

  private _attnDropMask!: Uint8Array;

  private _ffnHidden!: Float64Array; // B*seqCap*H
  private _ffnAct!: Float64Array; // B*seqCap*H
  private _ffnOut!: Float64Array; // B*seqCap*D

  // Pool/output caches
  private _poolScore!: Float64Array; // seqCap
  private _poolProb!: Float64Array; // seqCap
  private _aggregated!: Float64Array; // D
  private _yPred!: Float64Array; // outDim
  private _yDenorm!: Float64Array; // outDim (denormalized base prediction)

  // Backprop scratch
  private _dFused!: Float64Array; // seqCap*D
  private _dTmpSeqD!: Float64Array; // seqCap*D
  private _dTmpSeqH!: Float64Array; // seqCap*H
  private _softRowScore!: Float64Array; // seqCap
  private _softRowProb!: Float64Array; // seqCap
  private _dY!: Float64Array; // outDim
  private _dZScale!: Float64Array; // S (for fusion backprop)

  // ADWIN
  private _adwinWin!: Float64Array;
  private _adwinPref!: Float64Array;
  private _adwinLen = 0;
  private _adwinCap = 1024;

  // RNG
  private _rngState = 0x9e3779b9 | 0;

  /**
   * @param config Optional partial config overrides.
   * @example
   * const m = new FusionTemporalTransformerRegression({ embeddingDim: 64, numHeads: 8 });
   */
  constructor(config?: Partial<Config>) {
    const d: Config = {
      numBlocks: 3,
      embeddingDim: 64,
      numHeads: 8,
      ffnMultiplier: 4,

      attentionDropout: 0.0,
      fusionDropout: 0.0,

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
    };

    const c = config || {};
    this._cfg = this._validateCfg({
      numBlocks: c.numBlocks ?? d.numBlocks,
      embeddingDim: c.embeddingDim ?? d.embeddingDim,
      numHeads: c.numHeads ?? d.numHeads,
      ffnMultiplier: c.ffnMultiplier ?? d.ffnMultiplier,

      attentionDropout: c.attentionDropout ?? d.attentionDropout,
      fusionDropout: c.fusionDropout ?? d.fusionDropout,

      learningRate: c.learningRate ?? d.learningRate,
      warmupSteps: c.warmupSteps ?? d.warmupSteps,
      totalSteps: c.totalSteps ?? d.totalSteps,

      beta1: c.beta1 ?? d.beta1,
      beta2: c.beta2 ?? d.beta2,
      epsilon: c.epsilon ?? d.epsilon,

      regularizationStrength: c.regularizationStrength ??
        d.regularizationStrength,
      convergenceThreshold: c.convergenceThreshold ?? d.convergenceThreshold,

      outlierThreshold: c.outlierThreshold ?? d.outlierThreshold,
      adwinDelta: c.adwinDelta ?? d.adwinDelta,

      temporalScales: c.temporalScales ?? d.temporalScales,
      temporalKernelSize: c.temporalKernelSize ?? d.temporalKernelSize,

      maxSequenceLength: c.maxSequenceLength ?? d.maxSequenceLength,
    });

    if ((this._cfg.embeddingDim % this._cfg.numHeads) !== 0) {
      throw new Error(
        `embeddingDim (${this._cfg.embeddingDim}) must be divisible by numHeads (${this._cfg.numHeads}).`,
      );
    }

    this._adwinWin = new Float64Array(this._adwinCap);
    this._adwinPref = new Float64Array(this._adwinCap);
  }

  /**
   * Incremental online fit.
   * Implements:
   * - Welford z-score normalization (online)
   * - Full forward + full backprop
   * - Online Adam + cosine warmup
   * - L2 regularization
   * - Outlier downweighting
   * - ADWIN drift detection (bounded cost)
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xSeq = data.xCoordinates;
    const ySeq = data.yCoordinates;

    const seqLen0 = xSeq.length | 0;
    if (seqLen0 <= 0) {
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

    const inDim = (xSeq[0]?.length ?? 0) | 0;
    const outDim =
      (ySeq.length > 0 ? (ySeq[ySeq.length - 1]?.length ?? 0) : 0) | 0;

    if (inDim <= 0 || outDim <= 0) {
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

    if (!this._isInitialized) {
      this._initialize(inDim, outDim);
    } else if (inDim !== this._inDim || outDim !== this._outDim) {
      this.reset();
      this._initialize(inDim, outDim);
    }

    // Enforce maxSequenceLength by taking tail window
    let seqLen = seqLen0;
    let start = 0;
    if (seqLen > this._cfg.maxSequenceLength) {
      start = seqLen - this._cfg.maxSequenceLength;
      seqLen = this._cfg.maxSequenceLength;
    }

    this._copySequenceWindow(xSeq, start, seqLen, inDim);
    this._lastSeqLen = seqLen;

    // Target = last row in yCoordinates
    {
      const yLast = ySeq[ySeq.length - 1];
      for (let j = 0; j < this._outDim; j++) this._yRaw[j] = +yLast[j];
    }

    // Update Welford stats, compute std arrays
    this._updateWelfordX(this._xRaw, this._lastSeqLen);
    this._updateWelfordY(this._yRaw);

    // Normalize
    this._normalizeX(this._xRaw, this._xNorm, this._lastSeqLen);
    this._normalizeY(this._yRaw, this._yNorm);

    // Save last normalized sequence for predict()
    this._storeLastXNorm(this._xNorm, this._lastSeqLen);

    // Zero grads
    this._zeroGrads();

    // Forward
    this._forward(this._lastSeqLen);

    // Residual norm for outlier detection (in normalized space, scaled by output std)
    let residualNorm = 0;
    const eps = this._cfg.epsilon;
    for (let j = 0; j < this._outDim; j++) {
      const denom = this._yStd[j] + eps;
      const r = (this._yNorm[j] - this._yPred[j]) / denom;
      residualNorm += r * r;
    }
    residualNorm = Math.sqrt(residualNorm);

    const isOutlier = residualNorm > this._cfg.outlierThreshold;
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // MSE loss: sum((pred - y)^2) / (2*outDim)
    let mse = 0;
    for (let j = 0; j < this._outDim; j++) {
      const d = this._yPred[j] - this._yNorm[j];
      mse += d * d;
    }
    mse *= 0.5 / Math.max(1, this._outDim);

    // L2 loss
    let l2 = 0;
    const reg = this._cfg.regularizationStrength;
    if (reg > 0) {
      for (let p = 0; p < this._params.length; p++) {
        const pm = this._params[p];
        if (!pm.l2) continue;
        const w = pm.w;
        let s = 0;
        for (let i = 0; i < w.length; i++) {
          const wi = w[i];
          s += wi * wi;
        }
        l2 += s;
      }
      l2 *= 0.5 * reg;
    }

    const loss = mse * sampleWeight + l2;

    // Backward
    const gradNorm = this._backward(this._lastSeqLen, sampleWeight);

    // Update
    this._step++;
    this._effectiveLR = this._lrSchedule(this._step);
    this._adamUpdate(this._effectiveLR);

    // Running accuracy
    this._sampleCount++;
    this._lossSum += loss;
    const avgLoss = this._lossSum / Math.max(1, this._sampleCount);
    this._accuracy = 1 / (1 + avgLoss);

    // Convergence
    this._converged =
      Math.abs(this._prevLoss - loss) < this._cfg.convergenceThreshold;
    this._prevLoss = loss;

    // ADWIN drift detection using mse (unweighted) as error stream
    const driftDetected = this._adwinUpdateAndCheck(mse);
    if (driftDetected) {
      this._driftCount++;
      this._resetNormalizationOnly();
    }

    return {
      loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: this._effectiveLR,
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict future steps based on last observed (fitOnline) sequence.
   * Uses last normalized input and current weights; returns repeated base prediction for steps with widening uncertainty.
   */
  public predict(futureSteps: number): PredictionResult {
    const steps = _clampInt(futureSteps | 0, 0, 4096);
    if (
      !this._isInitialized || this._sampleCount <= 0 || this._lastSeqLen <= 0
    ) {
      return {
        predictions: [],
        accuracy: this._accuracy,
        sampleCount: this._sampleCount,
        isModelReady: false,
      };
    }

    this._loadLastXNormIntoWorking(this._lastSeqLen);
    this._forward(this._lastSeqLen);

    // Denormalize base prediction
    const eps = this._cfg.epsilon;
    for (let j = 0; j < this._outDim; j++) {
      this._yDenorm[j] = this._yPred[j] * (this._yStd[j] + eps) +
        this._yMean[j];
    }

    const n = Math.max(1, this._sampleCount);
    const invSqrtN = 1 / Math.sqrt(n);
    const z = 1.96;

    const preds: SinglePrediction[] = [];
    for (let s = 0; s < steps; s++) {
      const mult = Math.sqrt(s + 1);
      const pr: number[] = new Array(this._outDim);
      const lo: number[] = new Array(this._outDim);
      const hi: number[] = new Array(this._outDim);
      const se: number[] = new Array(this._outDim);

      for (let j = 0; j < this._outDim; j++) {
        const stderr = (this._yStd[j] * invSqrtN) * mult;
        const pj = this._yDenorm[j];
        pr[j] = pj;
        se[j] = stderr;
        lo[j] = pj - z * stderr;
        hi[j] = pj + z * stderr;
      }
      preds.push({
        predicted: pr,
        lowerBound: lo,
        upperBound: hi,
        standardError: se,
      });
    }

    return {
      predictions: preds,
      accuracy: this._accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inDim,
      outputDimension: this._outDim,
      numBlocks: this._cfg.numBlocks,
      embeddingDim: this._cfg.embeddingDim,
      numHeads: this._cfg.numHeads,
      temporalScales: this._cfg.temporalScales.slice(),
      totalParameters: this._countParams(),
      sampleCount: this._sampleCount,
      accuracy: this._accuracy,
      converged: this._converged,
      effectiveLearningRate: this._effectiveLR,
      driftCount: this._driftCount,
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
        updateCount: this._step,
      };
    }

    const S = this._cfg.temporalScales.length;
    const D = this._cfg.embeddingDim;
    const maxSeq = this._cfg.maxSequenceLength;

    // Temporal conv weights
    const temporalConvWeights: number[][][] = new Array(S);
    for (let si = 0; si < S; si++) {
      const k = this._fusion.convK[si];
      const b = this._fusion.convB[si];
      const rows: number[][] = new Array(D + 1);
      for (let r = 0; r < D; r++) {
        const row: number[] = new Array(k.cols);
        const off = r * k.cols;
        for (let c = 0; c < k.cols; c++) row[c] = k.w[off + c];
        rows[r] = row;
      }
      const brow: number[] = new Array(D);
      for (let j = 0; j < D; j++) brow[j] = b.w[j];
      rows[D] = brow;
      temporalConvWeights[si] = rows;
    }

    // Scale embeddings
    const scaleEmbeddings: number[][] = new Array(S);
    for (let si = 0; si < S; si++) {
      const row: number[] = new Array(D);
      const off = si * D;
      for (let j = 0; j < D; j++) row[j] = this._fusion.scaleEmb.w[off + j];
      scaleEmbeddings[si] = row;
    }

    // Positional encoding
    const positionalEncoding: number[][] = new Array(maxSeq);
    for (let t = 0; t < maxSeq; t++) {
      const row: number[] = new Array(D);
      const off = t * D;
      for (let j = 0; j < D; j++) row[j] = this._posEnc[off + j];
      positionalEncoding[t] = row;
    }

    const fusionWeights: number[][][] = [];
    fusionWeights.push(_matTo2D(this._fusion.gateW));
    fusionWeights.push(_matTo2D(this._fusion.gateB));
    fusionWeights.push(_matTo2D(this._po.poolW));
    fusionWeights.push(_matTo2D(this._po.poolB));

    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];
      attentionWeights.push(_matTo2D(blk.wq));
      attentionWeights.push(_matTo2D(blk.wk));
      attentionWeights.push(_matTo2D(blk.wv));
      attentionWeights.push(_matTo2D(blk.wo));
      attentionWeights.push(_matTo2D(blk.bo));
      attentionWeights.push(_matTo2D(blk.attnSlope));
    }

    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];
      ffnWeights.push(_matTo2D(blk.w1));
      ffnWeights.push(_matTo2D(blk.b1));
      ffnWeights.push(_matTo2D(blk.w2));
      ffnWeights.push(_matTo2D(blk.b2));
    }

    const layerNormParams: number[][][] = [];
    for (let b = 0; b < this._blocks.length; b++) {
      const blk = this._blocks[b];
      layerNormParams.push(_matTo2D(blk.ln1Gamma));
      layerNormParams.push(_matTo2D(blk.ln1Beta));
      layerNormParams.push(_matTo2D(blk.ln2Gamma));
      layerNormParams.push(_matTo2D(blk.ln2Beta));
    }

    const outputWeights: number[][][] = [];
    outputWeights.push(_matTo2D(this._po.outW));
    outputWeights.push(_matTo2D(this._po.outB));

    const firstMoment: number[][][] = new Array(this._params.length);
    const secondMoment: number[][][] = new Array(this._params.length);
    for (let i = 0; i < this._params.length; i++) {
      firstMoment[i] = _arrTo2D(
        this._params[i].m,
        this._params[i].rows,
        this._params[i].cols,
      );
      secondMoment[i] = _arrTo2D(
        this._params[i].v,
        this._params[i].rows,
        this._params[i].cols,
      );
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
      updateCount: this._step,
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
    const inputMean: number[] = new Array(this._inDim);
    const inputStd: number[] = new Array(this._inDim);
    const outputMean: number[] = new Array(this._outDim);
    const outputStd: number[] = new Array(this._outDim);

    for (let i = 0; i < this._inDim; i++) {
      inputMean[i] = this._xMean[i];
      inputStd[i] = this._xStd[i];
    }
    for (let j = 0; j < this._outDim; j++) {
      outputMean[j] = this._yMean[j];
      outputStd[j] = this._yStd[j];
    }

    // count: min(xCount, yCount) as "samples" proxy
    const count = Math.min(this._xCount, this._yCount);
    return { inputMean, inputStd, outputMean, outputStd, count };
  }

  public reset(): void {
    this._isInitialized = false;

    this._inDim = 0;
    this._outDim = 0;

    this._step = 0;
    this._effectiveLR = 0;

    this._sampleCount = 0;
    this._lossSum = 0;
    this._accuracy = 0;
    this._prevLoss = Number.POSITIVE_INFINITY;
    this._converged = false;

    this._driftCount = 0;

    this._xCount = 0;
    this._yCount = 0;

    this._seqCap = 0;
    this._lastSeqLen = 0;

    this._adwinLen = 0;
  }

  public save(): string {
    const obj: any = {
      cfg: this._cfg,
      isInitialized: this._isInitialized,
      inDim: this._inDim,
      outDim: this._outDim,
      step: this._step,
      effectiveLR: this._effectiveLR,
      sampleCount: this._sampleCount,
      lossSum: this._lossSum,
      accuracy: this._accuracy,
      prevLoss: this._prevLoss,
      converged: this._converged,
      driftCount: this._driftCount,
      xCount: this._xCount,
      yCount: this._yCount,
      xMean: this._isInitialized ? Array.from(this._xMean) : [],
      xM2: this._isInitialized ? Array.from(this._xM2) : [],
      xStd: this._isInitialized ? Array.from(this._xStd) : [],
      yMean: this._isInitialized ? Array.from(this._yMean) : [],
      yM2: this._isInitialized ? Array.from(this._yM2) : [],
      yStd: this._isInitialized ? Array.from(this._yStd) : [],
      lastSeqLen: this._lastSeqLen,
      lastXNormSeq: this._isInitialized && this._lastSeqLen > 0
        ? Array.from(
          this._lastXNormSeq.subarray(0, this._lastSeqLen * this._inDim),
        )
        : [],
      adwinLen: this._adwinLen,
      adwinWin: Array.from(this._adwinWin.subarray(0, this._adwinLen)),
      params: this._isInitialized
        ? this._params.map((p) => ({
          name: p.name,
          rows: p.rows,
          cols: p.cols,
          l2: p.l2,
          w: Array.from(p.w),
          m: Array.from(p.m),
          v: Array.from(p.v),
        }))
        : [],
      rngState: this._rngState | 0,
    };
    return JSON.stringify(obj);
  }

  public load(w: string): void {
    const obj = JSON.parse(w);

    // Config override
    if (obj.cfg) this._cfg = this._validateCfg(obj.cfg as Config);

    // Hard reset internal allocations
    this.reset();

    this._isInitialized = !!obj.isInitialized;
    this._inDim = obj.inDim | 0;
    this._outDim = obj.outDim | 0;

    this._step = obj.step | 0;
    this._effectiveLR = +obj.effectiveLR;

    this._sampleCount = obj.sampleCount | 0;
    this._lossSum = +obj.lossSum;
    this._accuracy = +obj.accuracy;
    this._prevLoss = +obj.prevLoss;
    this._converged = !!obj.converged;

    this._driftCount = obj.driftCount | 0;

    this._xCount = obj.xCount | 0;
    this._yCount = obj.yCount | 0;

    this._rngState = (obj.rngState | 0) || (0x9e3779b9 | 0);

    // ADWIN restore
    this._adwinLen = obj.adwinLen | 0;
    if (this._adwinLen > this._adwinCap) {
      this._adwinCap = _nextPow2(this._adwinLen);
      this._adwinWin = new Float64Array(this._adwinCap);
      this._adwinPref = new Float64Array(this._adwinCap);
    } else {
      this._adwinWin = new Float64Array(this._adwinCap);
      this._adwinPref = new Float64Array(this._adwinCap);
    }
    const aw: number[] = obj.adwinWin || [];
    for (let i = 0; i < this._adwinLen; i++) this._adwinWin[i] = +aw[i];
    this._rebuildAdwinPrefix();

    if (!this._isInitialized) return;

    this._initialize(this._inDim, this._outDim);

    // Restore normalization
    const xMean: number[] = obj.xMean || [];
    const xM2: number[] = obj.xM2 || [];
    const xStd: number[] = obj.xStd || [];
    const yMean: number[] = obj.yMean || [];
    const yM2: number[] = obj.yM2 || [];
    const yStd: number[] = obj.yStd || [];

    for (let i = 0; i < this._inDim; i++) {
      this._xMean[i] = +xMean[i];
      this._xM2[i] = +xM2[i];
      this._xStd[i] = +xStd[i];
    }
    for (let j = 0; j < this._outDim; j++) {
      this._yMean[j] = +yMean[j];
      this._yM2[j] = +yM2[j];
      this._yStd[j] = +yStd[j];
    }

    // Restore params by name
    const savedParams: any[] = obj.params || [];
    const map = new Map<string, any>();
    for (let i = 0; i < savedParams.length; i++) {
      map.set(savedParams[i].name, savedParams[i]);
    }

    for (let i = 0; i < this._params.length; i++) {
      const p = this._params[i];
      const sp = map.get(p.name);
      if (!sp) continue;
      const sw: number[] = sp.w || [];
      const sm: number[] = sp.m || [];
      const sv: number[] = sp.v || [];
      const n = p.w.length;
      for (let k = 0; k < n; k++) {
        p.w[k] = +sw[k];
        p.m[k] = +sm[k];
        p.v[k] = +sv[k];
      }
    }

    // Restore last sequence
    this._lastSeqLen = obj.lastSeqLen | 0;
    if (this._lastSeqLen > 0) {
      const arr: number[] = obj.lastXNormSeq || [];
      const need = this._lastSeqLen * this._inDim;
      for (let i = 0; i < need; i++) this._lastXNormSeq[i] = +arr[i];
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Initialization
  // ─────────────────────────────────────────────────────────────────────────────

  private _validateCfg(cfg: Config): Config {
    const c = cfg;
    const out: Config = {
      numBlocks: _clampInt(c.numBlocks, 1, 64),
      embeddingDim: _clampInt(c.embeddingDim, 2, 2048),
      numHeads: _clampInt(c.numHeads, 1, 128),
      ffnMultiplier: _clampInt(c.ffnMultiplier, 1, 16),

      attentionDropout: _clamp01(c.attentionDropout),
      fusionDropout: _clamp01(c.fusionDropout),

      learningRate: _pos(c.learningRate, 0.001),
      warmupSteps: _clampInt(c.warmupSteps, 0, 1_000_000),
      totalSteps: _clampInt(c.totalSteps, 1, 10_000_000),

      beta1: _clamp01(c.beta1),
      beta2: _clamp01(c.beta2),
      epsilon: _pos(c.epsilon, 1e-8),

      regularizationStrength: Math.max(0, c.regularizationStrength),
      convergenceThreshold: Math.max(0, c.convergenceThreshold),

      outlierThreshold: Math.max(0, c.outlierThreshold),
      adwinDelta: Math.max(1e-12, c.adwinDelta),

      temporalScales: _sanitizeScales(c.temporalScales),
      temporalKernelSize: _clampInt(c.temporalKernelSize, 1, 63),

      maxSequenceLength: _clampInt(c.maxSequenceLength, 2, 8192),
    };
    if ((out.embeddingDim % out.numHeads) !== 0) {
      // Keep as-is; caller will error
    }
    return out;
  }

  private _initialize(inDim: number, outDim: number): void {
    this._inDim = inDim | 0;
    this._outDim = outDim | 0;
    this._isInitialized = true;

    // Welford stats
    this._xMean = new Float64Array(this._inDim);
    this._xM2 = new Float64Array(this._inDim);
    this._xStd = new Float64Array(this._inDim);
    this._xStd.fill(1);

    this._yMean = new Float64Array(this._outDim);
    this._yM2 = new Float64Array(this._outDim);
    this._yStd = new Float64Array(this._outDim);
    this._yStd.fill(1);

    // Positional encoding
    this._posEnc = new Float64Array(
      this._cfg.maxSequenceLength * this._cfg.embeddingDim,
    );
    this._buildPosEnc();

    // Buffers
    this._ensureSeqCapacity(this._cfg.maxSequenceLength);

    // Build model parameters
    this._buildParams();

    // ADWIN
    this._adwinLen = 0;
    this._rebuildAdwinPrefix();
  }

  private _ensureSeqCapacity(seqCap: number): void {
    const cap = Math.max(2, seqCap | 0);
    if (cap <= this._seqCap && this._inDim > 0 && this._outDim > 0) return;

    this._seqCap = cap;

    const inN = this._seqCap * this._inDim;
    const outD = this._outDim;

    this._xRaw = new Float64Array(inN);
    this._xNorm = new Float64Array(inN);
    this._yRaw = new Float64Array(outD);
    this._yNorm = new Float64Array(outD);
    this._lastXNormSeq = new Float64Array(inN);

    const S = this._cfg.temporalScales.length;
    const D = this._cfg.embeddingDim;
    const B = this._cfg.numBlocks;
    const H = D * this._cfg.ffnMultiplier;

    this._scaleOutLen = new Int32Array(S);
    this._convOff = new Int32Array(S);

    // Packed conv buffers size
    let pack = 0;
    for (let si = 0; si < S; si++) {
      const stride = this._cfg.temporalScales[si] | 0;
      const outLen = ((this._seqCap + stride - 1) / stride) | 0;
      this._scaleOutLen[si] = outLen;
      this._convOff[si] = pack;
      pack += outLen * D;
    }

    this._convPre = new Float64Array(pack);
    this._convAct = new Float64Array(pack);

    this._eUp = new Float64Array(S * this._seqCap * D);
    this._gateZ = new Float64Array(this._seqCap * S);
    this._gateSig = new Float64Array(this._seqCap * S);
    this._fused = new Float64Array(this._seqCap * D);
    this._fusionDropMask = new Uint8Array(this._seqCap * D);

    this._blockIn = new Float64Array(B * this._seqCap * D);
    this._afterAttn = new Float64Array(B * this._seqCap * D);

    this._ln1Xhat = new Float64Array(B * this._seqCap * D);
    this._ln1Mean = new Float64Array(B * this._seqCap);
    this._ln1InvStd = new Float64Array(B * this._seqCap);

    this._ln2Xhat = new Float64Array(B * this._seqCap * D);
    this._ln2Mean = new Float64Array(B * this._seqCap);
    this._ln2InvStd = new Float64Array(B * this._seqCap);

    this._q = new Float64Array(B * this._seqCap * D);
    this._k = new Float64Array(B * this._seqCap * D);
    this._v = new Float64Array(B * this._seqCap * D);
    this._ctx = new Float64Array(B * this._seqCap * D);
    this._attnOut = new Float64Array(B * this._seqCap * D);

    this._dQ = new Float64Array(B * this._seqCap * D);
    this._dK = new Float64Array(B * this._seqCap * D);
    this._dV = new Float64Array(B * this._seqCap * D);

    this._attnDropMask = new Uint8Array(B * this._seqCap * D);

    this._ffnHidden = new Float64Array(B * this._seqCap * H);
    this._ffnAct = new Float64Array(B * this._seqCap * H);
    this._ffnOut = new Float64Array(B * this._seqCap * D);

    this._poolScore = new Float64Array(this._seqCap);
    this._poolProb = new Float64Array(this._seqCap);
    this._aggregated = new Float64Array(D);
    this._yPred = new Float64Array(this._outDim);
    this._yDenorm = new Float64Array(this._outDim);

    this._dFused = new Float64Array(this._seqCap * D);
    this._dTmpSeqD = new Float64Array(this._seqCap * D);
    this._dTmpSeqH = new Float64Array(this._seqCap * H);

    this._softRowScore = new Float64Array(this._seqCap);
    this._softRowProb = new Float64Array(this._seqCap);

    this._dY = new Float64Array(this._outDim);
    this._dZScale = new Float64Array(S);
  }

  private _buildPosEnc(): void {
    const maxSeq = this._cfg.maxSequenceLength | 0;
    const D = this._cfg.embeddingDim | 0;
    for (let pos = 0; pos < maxSeq; pos++) {
      const base = pos * D;
      for (let i = 0; i < (D >> 1); i++) {
        const denom = Math.pow(10000, (2 * i) / D);
        const angle = pos / denom;
        const j = i << 1;
        this._posEnc[base + j] = Math.sin(angle);
        this._posEnc[base + j + 1] = Math.cos(angle);
      }
    }
  }

  private _buildParams(): void {
    const S = this._cfg.temporalScales.length;
    const D = this._cfg.embeddingDim;
    const Hh = this._cfg.numHeads;
    const K = this._cfg.temporalKernelSize;
    const H = D * this._cfg.ffnMultiplier;
    const B = this._cfg.numBlocks;

    const params: _Mat[] = [];

    // Fusion
    const convK: _Mat[] = new Array(S);
    const convB: _Mat[] = new Array(S);

    for (let si = 0; si < S; si++) {
      convK[si] = this._makeMat(`convK_s${si}`, D, this._inDim * K, true);
      convB[si] = this._makeMat(`convB_s${si}`, 1, D, false);
      this._xavierInit(convK[si].w, convK[si].cols, convK[si].rows);
      convB[si].w.fill(0);
      params.push(convK[si], convB[si]);
    }

    const scaleEmb = this._makeMat("scaleEmb", S, D, true);
    this._xavierInit(scaleEmb.w, D, S);
    params.push(scaleEmb);

    const gateW = this._makeMat("gateW", S * D, S, true);
    const gateB = this._makeMat("gateB", 1, S, false);
    this._xavierInit(gateW.w, gateW.cols, gateW.rows);
    gateB.w.fill(0);
    params.push(gateW, gateB);

    this._fusion = { convK, convB, scaleEmb, gateW, gateB };

    // Blocks
    const blocks: _Block[] = new Array(B);
    for (let bi = 0; bi < B; bi++) {
      const ln1Gamma = this._makeMat(`b${bi}_ln1G`, 1, D, false);
      const ln1Beta = this._makeMat(`b${bi}_ln1B`, 1, D, false);
      ln1Gamma.w.fill(1);
      ln1Beta.w.fill(0);

      const wq = this._makeMat(`b${bi}_wq`, D, D, true);
      const wk = this._makeMat(`b${bi}_wk`, D, D, true);
      const wv = this._makeMat(`b${bi}_wv`, D, D, true);
      const wo = this._makeMat(`b${bi}_wo`, D, D, true);
      const bo = this._makeMat(`b${bi}_bo`, 1, D, false);

      this._xavierInit(wq.w, D, D);
      this._xavierInit(wk.w, D, D);
      this._xavierInit(wv.w, D, D);
      this._xavierInit(wo.w, D, D);
      bo.w.fill(0);

      const attnSlope = this._makeMat(`b${bi}_slope`, 1, Hh, false);
      attnSlope.w.fill(0);

      const ln2Gamma = this._makeMat(`b${bi}_ln2G`, 1, D, false);
      const ln2Beta = this._makeMat(`b${bi}_ln2B`, 1, D, false);
      ln2Gamma.w.fill(1);
      ln2Beta.w.fill(0);

      const w1 = this._makeMat(`b${bi}_w1`, D, H, true);
      const b1 = this._makeMat(`b${bi}_b1`, 1, H, false);
      const w2 = this._makeMat(`b${bi}_w2`, H, D, true);
      const b2 = this._makeMat(`b${bi}_b2`, 1, D, false);

      this._xavierInit(w1.w, H, D);
      b1.w.fill(0);
      this._xavierInit(w2.w, D, H);
      b2.w.fill(0);

      const blk: _Block = {
        ln1Gamma,
        ln1Beta,
        wq,
        wk,
        wv,
        wo,
        bo,
        attnSlope,
        ln2Gamma,
        ln2Beta,
        w1,
        b1,
        w2,
        b2,
      };
      blocks[bi] = blk;

      params.push(
        ln1Gamma,
        ln1Beta,
        wq,
        wk,
        wv,
        wo,
        bo,
        attnSlope,
        ln2Gamma,
        ln2Beta,
        w1,
        b1,
        w2,
        b2,
      );
    }
    this._blocks = blocks;

    // Pool + output
    const poolW = this._makeMat("poolW", 1, D, true);
    const poolB = this._makeMat("poolB", 1, 1, false);
    const outW = this._makeMat("outW", D, this._outDim, true);
    const outB = this._makeMat("outB", 1, this._outDim, false);

    this._xavierInit(poolW.w, D, 1);
    poolB.w[0] = 0;
    this._xavierInit(outW.w, this._outDim, D);
    outB.w.fill(0);

    params.push(poolW, poolB, outW, outB);

    this._po = { poolW, poolB, outW, outB };
    this._params = params;
  }

  private _makeMat(
    name: string,
    rows: number,
    cols: number,
    l2: boolean,
  ): _Mat {
    const n = (rows * cols) | 0;
    return {
      name,
      rows: rows | 0,
      cols: cols | 0,
      w: new Float64Array(n),
      g: new Float64Array(n),
      m: new Float64Array(n),
      v: new Float64Array(n),
      l2: !!l2,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Forward
  // ─────────────────────────────────────────────────────────────────────────────

  private _forward(seqLen: number): void {
    const S = this._cfg.temporalScales.length;
    const D = this._cfg.embeddingDim;
    const K = this._cfg.temporalKernelSize | 0;
    const inDim = this._inDim;
    const eps = this._cfg.epsilon;

    // 1) Multi-scale conv embeddings -> packed convPre/convAct and upsample to E_up
    for (let si = 0; si < S; si++) {
      const stride = this._cfg.temporalScales[si] | 0;
      const outLen = ((seqLen + stride - 1) / stride) | 0;
      this._scaleOutLen[si] = outLen;

      const off = this._convOff[si] | 0;
      const pre = this._convPre;
      const act = this._convAct;

      const kMat = this._fusion.convK[si];
      const bMat = this._fusion.convB[si];

      const pad = (K >> 1) | 0;

      // conv
      for (let o = 0; o < outLen; o++) {
        const center = (o * stride) | 0;
        const rowBase = (off + o * D) | 0;

        for (let d = 0; d < D; d++) {
          let sum = bMat.w[d];
          const kRow = d * (inDim * K);

          for (let kk = 0; kk < K; kk++) {
            let t = center + (kk - pad);
            if (t < 0) t = 0;
            else if (t >= seqLen) t = seqLen - 1;

            const xOff = t * inDim;
            const kOff = kRow + kk * inDim;

            for (let c = 0; c < inDim; c++) {
              sum += kMat.w[kOff + c] * this._xNorm[xOff + c];
            }
          }

          pre[rowBase + d] = sum;
        }
      }

      // GELU + add positional + scale embedding
      const scaleEmbOff = si * D;
      for (let o = 0; o < outLen; o++) {
        const rowBase = (off + o * D) | 0;
        const peBase = (o * D) | 0;
        for (let d = 0; d < D; d++) {
          const x = pre[rowBase + d];
          const g = _gelu(x);
          act[rowBase + d] = g + this._posEnc[peBase + d] +
            this._fusion.scaleEmb.w[scaleEmbOff + d];
        }
      }

      // Upsample to E_up
      const eUpBase = si * (this._seqCap * D);
      for (let t = 0; t < seqLen; t++) {
        const o = ((t / stride) | 0) < outLen
          ? ((t / stride) | 0)
          : (outLen - 1);
        const src = (off + o * D) | 0;
        const dst = (eUpBase + t * D) | 0;
        for (let d = 0; d < D; d++) this._eUp[dst + d] = act[src + d];
      }
    }

    // 2) Gated fusion:
    // z[t,s] = sum_{j=0..S*D-1} concat[j] * gateW[j,s] + gateB[s]
    // gate = sigmoid(z)
    // fused[t,d] = sum_s gate[t,s] * E_up[s,t,d]
    {
      const gateW = this._fusion.gateW.w;
      const gateB = this._fusion.gateB.w;
      const SxD = S * D;
      const seqStride = this._seqCap * D;

      // clear fused
      for (let i = 0; i < seqLen * D; i++) this._fused[i] = 0;

      for (let t = 0; t < seqLen; t++) {
        const zOff = (t * S) | 0;

        // compute z/gate per scale
        for (let s = 0; s < S; s++) {
          let z = gateB[s];
          for (let j = 0; j < SxD; j++) {
            const si = (j / D) | 0;
            const dj = (j - si * D) | 0;
            const val = this._eUp[si * seqStride + t * D + dj];
            z += val * gateW[j * S + s];
          }
          this._gateZ[zOff + s] = z;
          this._gateSig[zOff + s] = _sigmoid(z);
        }

        // fused row
        const fOff = t * D;
        for (let s = 0; s < S; s++) {
          const g = this._gateSig[zOff + s];
          const src = s * seqStride + t * D;
          for (let d = 0; d < D; d++) {
            this._fused[fOff + d] += g * this._eUp[src + d];
          }
        }
      }

      // Fusion dropout (inverted)
      const p = this._cfg.fusionDropout;
      if (p > 0) {
        const inv = 1 / (1 - p);
        const n = seqLen * D;
        for (let i = 0; i < n; i++) {
          const keep = (this._rand01() >= p) ? 1 : 0;
          this._fusionDropMask[i] = keep as any;
          this._fused[i] *= keep * inv;
        }
      }
    }

    // 3) Transformer blocks
    const B = this._cfg.numBlocks;
    const Hh = this._cfg.numHeads;
    const Hd = D / Hh;
    const invSqrtHd = 1 / Math.sqrt(Hd);

    let cur = this._fused;

    for (let bi = 0; bi < B; bi++) {
      const blk = this._blocks[bi];
      const baseBD = (bi * this._seqCap * D) | 0;
      const baseB = (bi * this._seqCap) | 0;

      // store block input
      for (let i = 0; i < seqLen * D; i++) this._blockIn[baseBD + i] = cur[i];

      // LN1 -> out into _dTmpSeqD
      this._layerNormForward(
        cur,
        this._dTmpSeqD,
        blk.ln1Gamma.w,
        blk.ln1Beta.w,
        this._ln1Mean,
        this._ln1InvStd,
        this._ln1Xhat,
        baseB,
        baseBD,
        seqLen,
        D,
        eps,
      );

      // Q/K/V
      _matMulSeq(this._dTmpSeqD, this._q, blk.wq.w, seqLen, D, D, 0, baseBD);
      _matMulSeq(this._dTmpSeqD, this._k, blk.wk.w, seqLen, D, D, 0, baseBD);
      _matMulSeq(this._dTmpSeqD, this._v, blk.wv.w, seqLen, D, D, 0, baseBD);

      // ctx = MHA
      const slope = blk.attnSlope.w;
      const invSeq = 1 / Math.max(1, seqLen);

      for (let i = 0; i < seqLen * D; i++) this._ctx[baseBD + i] = 0;

      for (let h = 0; h < Hh; h++) {
        const hOff = h * Hd;

        for (let i = 0; i < seqLen; i++) {
          let max = -1e300;
          const qiOff = baseBD + i * D + hOff;

          for (let j = 0; j < seqLen; j++) {
            const kjOff = baseBD + j * D + hOff;
            let dot = 0;
            for (let u = 0; u < Hd; u++) {
              dot += this._q[qiOff + u] * this._k[kjOff + u];
            }
            dot *= invSqrtHd;
            dot += slope[h] * (Math.abs(i - j) * invSeq);
            this._softRowScore[j] = dot;
            if (dot > max) max = dot;
          }

          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            const e = Math.exp(this._softRowScore[j] - max);
            this._softRowProb[j] = e;
            sum += e;
          }
          const invSum = 1 / (sum + 1e-30);
          for (let j = 0; j < seqLen; j++) this._softRowProb[j] *= invSum;

          const ctxOff = baseBD + i * D + hOff;
          for (let u = 0; u < Hd; u++) this._ctx[ctxOff + u] = 0;
          for (let j = 0; j < seqLen; j++) {
            const pj = this._softRowProb[j];
            const vjOff = baseBD + j * D + hOff;
            for (let u = 0; u < Hd; u++) {
              this._ctx[ctxOff + u] += pj * this._v[vjOff + u];
            }
          }
        }
      }

      // attnOut = ctx*Wo + bo
      _matMulSeq(
        this._ctx.subarray(baseBD, baseBD + seqLen * D),
        this._attnOut,
        blk.wo.w,
        seqLen,
        D,
        D,
        blk.bo.w,
        baseBD,
      );

      // Attention dropout (inverted)
      const pAttn = this._cfg.attentionDropout;
      if (pAttn > 0) {
        const inv = 1 / (1 - pAttn);
        const n = seqLen * D;
        for (let i = 0; i < n; i++) {
          const keep = (this._rand01() >= pAttn) ? 1 : 0;
          this._attnDropMask[baseBD + i] = keep as any;
          this._attnOut[baseBD + i] *= keep * inv;
        }
      }

      // afterAttn = cur + attnOut
      for (let i = 0; i < seqLen * D; i++) {
        this._afterAttn[baseBD + i] = cur[i] + this._attnOut[baseBD + i];
      }

      // LN2 on afterAttn -> _dTmpSeqD
      this._layerNormForward(
        this._afterAttn.subarray(baseBD, baseBD + seqLen * D),
        this._dTmpSeqD,
        blk.ln2Gamma.w,
        blk.ln2Beta.w,
        this._ln2Mean,
        this._ln2InvStd,
        this._ln2Xhat,
        baseB,
        baseBD,
        seqLen,
        D,
        eps,
      );

      // FFN
      const Hdim = D * this._cfg.ffnMultiplier;
      const hiddenBase = (bi * this._seqCap * Hdim) | 0;

      _matMulSeq(
        this._dTmpSeqD,
        this._ffnHidden,
        blk.w1.w,
        seqLen,
        D,
        Hdim,
        blk.b1.w,
        hiddenBase,
      );

      for (let i = 0; i < seqLen * Hdim; i++) {
        const x = this._ffnHidden[hiddenBase + i];
        this._ffnAct[hiddenBase + i] = _gelu(x);
      }

      _matMulSeq(
        this._ffnAct.subarray(hiddenBase, hiddenBase + seqLen * Hdim),
        this._ffnOut,
        blk.w2.w,
        seqLen,
        Hdim,
        D,
        blk.b2.w,
        baseBD,
      );

      // Residual
      for (let i = 0; i < seqLen * D; i++) {
        cur[i] = this._afterAttn[baseBD + i] + this._ffnOut[baseBD + i];
      }
    }

    // 4) Attention pooling + output
    {
      const w = this._po.poolW.w;
      const b = this._po.poolB.w[0];

      let max = -1e300;
      for (let t = 0; t < seqLen; t++) {
        let s = b;
        const off = t * D;
        for (let d = 0; d < D; d++) s += cur[off + d] * w[d];
        this._poolScore[t] = s;
        if (s > max) max = s;
      }

      let sum = 0;
      for (let t = 0; t < seqLen; t++) {
        const e = Math.exp(this._poolScore[t] - max);
        this._poolProb[t] = e;
        sum += e;
      }
      const invSum = 1 / (sum + 1e-30);
      for (let t = 0; t < seqLen; t++) this._poolProb[t] *= invSum;

      for (let d = 0; d < D; d++) this._aggregated[d] = 0;
      for (let t = 0; t < seqLen; t++) {
        const p = this._poolProb[t];
        const off = t * D;
        for (let d = 0; d < D; d++) this._aggregated[d] += p * cur[off + d];
      }

      const outW = this._po.outW.w;
      const outB = this._po.outB.w;
      for (let j = 0; j < this._outDim; j++) {
        let s = outB[j];
        for (let d = 0; d < D; d++) {
          s += this._aggregated[d] * outW[d * this._outDim + j];
        }
        this._yPred[j] = s;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Backward
  // ─────────────────────────────────────────────────────────────────────────────

  private _backward(seqLen: number, sampleWeight: number): number {
    const D = this._cfg.embeddingDim;
    const S = this._cfg.temporalScales.length;
    const K = this._cfg.temporalKernelSize | 0;
    const B = this._cfg.numBlocks;
    const Hh = this._cfg.numHeads;
    const Hd = D / Hh;
    const invSqrtHd = 1 / Math.sqrt(Hd);
    const eps = this._cfg.epsilon;

    // dY = (yPred - yTrue) / outDim * sampleWeight
    const invN = sampleWeight / Math.max(1, this._outDim);
    for (let j = 0; j < this._outDim; j++) {
      this._dY[j] = (this._yPred[j] - this._yNorm[j]) * invN;
    }

    // Output grads
    {
      const outW = this._po.outW;
      const outB = this._po.outB;

      for (let j = 0; j < this._outDim; j++) outB.g[j] += this._dY[j];

      for (let d = 0; d < D; d++) {
        const a = this._aggregated[d];
        const row = d * this._outDim;
        for (let j = 0; j < this._outDim; j++) {
          outW.g[row + j] += a * this._dY[j];
        }
      }
    }

    // dAggregated = outW * dY
    const dAgg = this._dTmpSeqD; // first D entries
    {
      const outW = this._po.outW.w;
      for (let d = 0; d < D; d++) {
        let s = 0;
        const row = d * this._outDim;
        for (let j = 0; j < this._outDim; j++) s += outW[row + j] * this._dY[j];
        dAgg[d] = s;
      }
    }

    // Pooling backward -> dFused (gradient wrt final fused sequence)
    const cur = this._fused;
    {
      for (let i = 0; i < seqLen * D; i++) this._dFused[i] = 0;

      // dP[t] = dot(dAgg, cur[t])
      let sumDPp = 0;
      for (let t = 0; t < seqLen; t++) {
        let dp = 0;
        const off = t * D;
        for (let d = 0; d < D; d++) dp += dAgg[d] * cur[off + d];
        this._poolScore[t] = dp; // reuse buffer
        sumDPp += dp * this._poolProb[t];
      }

      const poolW = this._po.poolW;
      const poolB = this._po.poolB;

      for (let t = 0; t < seqLen; t++) {
        const p = this._poolProb[t];
        const off = t * D;

        // aggregated path
        for (let d = 0; d < D; d++) this._dFused[off + d] += p * dAgg[d];

        // scores path
        const dScore = p * (this._poolScore[t] - sumDPp);
        poolB.g[0] += dScore;

        for (let d = 0; d < D; d++) {
          poolW.g[d] += cur[off + d] * dScore;
          this._dFused[off + d] += poolW.w[d] * dScore;
        }
      }
    }

    // Transformer backward (reverse blocks)
    for (let bi = B - 1; bi >= 0; bi--) {
      const blk = this._blocks[bi];
      const baseBD = (bi * this._seqCap * D) | 0;
      const baseB = (bi * this._seqCap) | 0;

      const Hdim = D * this._cfg.ffnMultiplier;
      const hiddenBase = (bi * this._seqCap * Hdim) | 0;

      // afterAttn + ffnOut -> output
      // dAfter = dOut, dFfnOut = dOut
      const dAfter = this._dTmpSeqD;
      for (let i = 0; i < seqLen * D; i++) dAfter[i] = this._dFused[i];

      // ---- FFN backward ----
      // Linear2 grads + dAct
      {
        const w2 = blk.w2;
        const b2 = blk.b2;

        // b2
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let i = 0; i < seqLen; i++) s += this._dFused[i * D + d];
          b2.g[d] += s;
        }

        // w2
        for (let h = 0; h < Hdim; h++) {
          const row = h * D;
          for (let d = 0; d < D; d++) {
            let s = 0;
            for (let i = 0; i < seqLen; i++) {
              s += this._ffnAct[hiddenBase + i * Hdim + h] *
                this._dFused[i * D + d];
            }
            w2.g[row + d] += s;
          }
        }

        // dAct = dFfnOut * W2^T (store in _dTmpSeqH)
        for (let i = 0; i < seqLen; i++) {
          const dOff = i * D;
          const aOff = i * Hdim;
          for (let h = 0; h < Hdim; h++) {
            let s = 0;
            const row = h * D;
            for (let d = 0; d < D; d++) {
              s += this._dFused[dOff + d] * w2.w[row + d];
            }
            this._dTmpSeqH[aOff + h] = s;
          }
        }
      }

      // GELU backward
      for (let i = 0; i < seqLen * Hdim; i++) {
        const x = this._ffnHidden[hiddenBase + i];
        this._dTmpSeqH[i] *= _geluDeriv(x);
      }

      // Linear1 grads + dLn2Out
      const dLn2Out = this._dFused; // reuse dFused as dLn2Out buffer
      for (let i = 0; i < seqLen * D; i++) dLn2Out[i] = 0;

      {
        const w1 = blk.w1;
        const b1 = blk.b1;

        // b1
        for (let h = 0; h < Hdim; h++) {
          let s = 0;
          for (let i = 0; i < seqLen; i++) s += this._dTmpSeqH[i * Hdim + h];
          b1.g[h] += s;
        }

        // reconstruct ln2Out from xhat
        const g2 = blk.ln2Gamma.w;
        const b2 = blk.ln2Beta.w;

        for (let i = 0; i < seqLen; i++) {
          const xhOff = baseBD + i * D;
          const hOff = i * Hdim;

          for (let d = 0; d < D; d++) {
            const ln2Out_d = g2[d] * this._ln2Xhat[xhOff + d] + b2[d];
            const w1Row = d * Hdim;
            let acc = 0;

            for (let h = 0; h < Hdim; h++) {
              const dh = this._dTmpSeqH[hOff + h];
              w1.g[w1Row + h] += ln2Out_d * dh;
              acc += dh * w1.w[w1Row + h];
            }
            dLn2Out[i * D + d] += acc;
          }
        }
      }

      // LN2 backward into dAfter (accumulate)
      this._layerNormBackward(
        this._afterAttn.subarray(baseBD, baseBD + seqLen * D),
        dLn2Out,
        blk.ln2Gamma.w,
        this._ln2Mean,
        this._ln2InvStd,
        this._ln2Xhat,
        blk.ln2Gamma.g,
        blk.ln2Beta.g,
        dAfter,
        baseB,
        baseBD,
        seqLen,
        D,
        eps,
      );

      // afterAttn = blockIn + attnOut
      // dBlockIn += dAfter, dAttnOut += dAfter
      const dAttnOut = dLn2Out; // reuse buffer
      for (let i = 0; i < seqLen * D; i++) dAttnOut[i] = dAfter[i];

      const dBlockIn = this._dFused; // reuse
      for (let i = 0; i < seqLen * D; i++) dBlockIn[i] = dAfter[i];

      // attention dropout backward (approx)
      const pAttn = this._cfg.attentionDropout;
      if (pAttn > 0) {
        const inv = 1 / (1 - pAttn);
        for (let i = 0; i < seqLen * D; i++) {
          const keep = this._attnDropMask[baseBD + i] | 0;
          dAttnOut[i] *= keep * inv;
        }
      }

      // ---- Attention backward ----
      // attnOut = ctx*Wo + bo
      // 1) dWo/db + dCtx
      const dCtx = this._dTmpSeqD;
      for (let i = 0; i < seqLen * D; i++) dCtx[i] = 0;

      {
        const wo = blk.wo;
        const bo = blk.bo;

        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let i = 0; i < seqLen; i++) s += dAttnOut[i * D + d];
          bo.g[d] += s;
        }

        // dWo += ctx^T * dAttnOut
        for (let r = 0; r < D; r++) {
          const row = r * D;
          for (let c = 0; c < D; c++) {
            let s = 0;
            for (let i = 0; i < seqLen; i++) {
              s += this._ctx[baseBD + i * D + r] * dAttnOut[i * D + c];
            }
            wo.g[row + c] += s;
          }
        }

        // dCtx = dAttnOut * Wo^T
        for (let i = 0; i < seqLen; i++) {
          const off = i * D;
          for (let r = 0; r < D; r++) {
            let s = 0;
            const row = r * D;
            for (let c = 0; c < D; c++) s += dAttnOut[off + c] * wo.w[row + c];
            dCtx[off + r] = s;
          }
        }
      }

      // 2) Backprop MHA to dQ/dK/dV and slope
      // Clear dQ/dK/dV regions
      for (let i = 0; i < seqLen * D; i++) {
        this._dQ[baseBD + i] = 0;
        this._dK[baseBD + i] = 0;
        this._dV[baseBD + i] = 0;
      }
      const slope = blk.attnSlope.w;
      const slopeG = blk.attnSlope.g;
      const invSeq = 1 / Math.max(1, seqLen);

      for (let h = 0; h < Hh; h++) {
        const hOff = h * Hd;

        for (let i = 0; i < seqLen; i++) {
          // scores/probs for row i
          let max = -1e300;
          const qiOff = baseBD + i * D + hOff;

          for (let j = 0; j < seqLen; j++) {
            const kjOff = baseBD + j * D + hOff;
            let dot = 0;
            for (let u = 0; u < Hd; u++) {
              dot += this._q[qiOff + u] * this._k[kjOff + u];
            }
            dot *= invSqrtHd;
            dot += slope[h] * (Math.abs(i - j) * invSeq);
            this._softRowScore[j] = dot;
            if (dot > max) max = dot;
          }

          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            const e = Math.exp(this._softRowScore[j] - max);
            this._softRowProb[j] = e;
            sum += e;
          }
          const invSum = 1 / (sum + 1e-30);
          for (let j = 0; j < seqLen; j++) this._softRowProb[j] *= invSum;

          // dP[j] = dot(dCtx_i, V_j)
          const dCiOff = i * D + hOff;
          let sumDPp = 0;

          for (let j = 0; j < seqLen; j++) {
            const vjOff = baseBD + j * D + hOff;
            let dp = 0;
            for (let u = 0; u < Hd; u++) {
              dp += dCtx[dCiOff + u] * this._v[vjOff + u];
            }
            this._softRowScore[j] = dp; // reuse as dP
            sumDPp += dp * this._softRowProb[j];
          }

          for (let j = 0; j < seqLen; j++) {
            const pj = this._softRowProb[j];
            const dScore = pj * (this._softRowScore[j] - sumDPp);

            // slope grad
            slopeG[h] += dScore * (Math.abs(i - j) * invSeq);

            // dQ_i += dScore * K_j / sqrt(Hd)
            const kjOff = baseBD + j * D + hOff;
            const dQiOff = baseBD + i * D + hOff;
            for (let u = 0; u < Hd; u++) {
              this._dQ[dQiOff + u] += dScore * this._k[kjOff + u] * invSqrtHd;
            }

            // dK_j += dScore * Q_i / sqrt(Hd)
            const dKjOff = baseBD + j * D + hOff;
            for (let u = 0; u < Hd; u++) {
              this._dK[dKjOff + u] += dScore * this._q[qiOff + u] * invSqrtHd;
            }

            // dV_j += p[j] * dCtx_i
            for (let u = 0; u < Hd; u++) {
              this._dV[dKjOff + u] += pj * dCtx[dCiOff + u];
            }
          }
        }
      }

      // 3) Backprop Q/K/V projections to LN1 output
      // ln1Out reconstructed from xhat1
      const ln1Out = this._dTmpSeqD; // seqLen*D
      {
        const g1 = blk.ln1Gamma.w;
        const b1 = blk.ln1Beta.w;
        for (let i = 0; i < seqLen; i++) {
          const xhOff = baseBD + i * D;
          const oOff = i * D;
          for (let d = 0; d < D; d++) {
            ln1Out[oOff + d] = g1[d] * this._ln1Xhat[xhOff + d] + b1[d];
          }
        }
      }

      // dLn1Out = dQ*Wq^T + dK*Wk^T + dV*Wv^T
      const dLn1Out = dCtx; // reuse; overwrite
      for (let i = 0; i < seqLen * D; i++) dLn1Out[i] = 0;

      _projBackwardOffset(ln1Out, this._dQ, baseBD, blk.wq, dLn1Out, seqLen, D);
      _projBackwardOffset(ln1Out, this._dK, baseBD, blk.wk, dLn1Out, seqLen, D);
      _projBackwardOffset(ln1Out, this._dV, baseBD, blk.wv, dLn1Out, seqLen, D);

      // LN1 backward into dBlockIn (accumulate)
      this._layerNormBackward(
        this._blockIn.subarray(baseBD, baseBD + seqLen * D),
        dLn1Out,
        blk.ln1Gamma.w,
        this._ln1Mean,
        this._ln1InvStd,
        this._ln1Xhat,
        blk.ln1Gamma.g,
        blk.ln1Beta.g,
        dBlockIn,
        baseB,
        baseBD,
        seqLen,
        D,
        eps,
      );

      // dFused becomes gradient wrt previous block output
      for (let i = 0; i < seqLen * D; i++) this._dFused[i] = dBlockIn[i];
    }

    // Fusion dropout backward
    if (this._cfg.fusionDropout > 0) {
      const p = this._cfg.fusionDropout;
      const inv = 1 / (1 - p);
      const n = seqLen * D;
      for (let i = 0; i < n; i++) {
        this._dFused[i] *= (this._fusionDropMask[i] | 0) * inv;
      }
    }

    // Fusion backward (streaming, no extra allocations)
    {
      const gateW = this._fusion.gateW;
      const gateB = this._fusion.gateB;
      const scaleEmb = this._fusion.scaleEmb;

      const seqStride = this._seqCap * D;
      const SxD = S * D;

      const inDim = this._inDim;
      const pad = (K >> 1) | 0;

      for (let t = 0; t < seqLen; t++) {
        const zOff = t * S;

        // dZ[s] = dGate[s] * sigmoid'(z)
        for (let s = 0; s < S; s++) {
          let dGate = 0;
          const src = s * seqStride + t * D;
          const dF = t * D;
          for (let d = 0; d < D; d++) {
            dGate += this._dFused[dF + d] * this._eUp[src + d];
          }

          const sig = this._gateSig[zOff + s];
          const dz = dGate * (sig * (1 - sig));
          this._dZScale[s] = dz;
          gateB.g[s] += dz;
        }

        // For each scale and dim, compute dE_up = gate[s]*dFused + dConcat
        // dConcat[j] = sum_col gateW[j,col] * dZ[col], where j = s*D + d
        for (let s = 0; s < S; s++) {
          const sig = this._gateSig[zOff + s];
          const stride = this._cfg.temporalScales[s] | 0;
          const outLen = this._scaleOutLen[s] | 0;
          const o = ((t / stride) | 0) < outLen
            ? ((t / stride) | 0)
            : (outLen - 1);

          const convBase = (this._convOff[s] + o * D) | 0;
          const src = s * seqStride + t * D;
          const dF = t * D;

          for (let d = 0; d < D; d++) {
            const j = s * D + d;
            const row = j * S;
            const concatVal = this._eUp[src + d];

            // gateW grads: dW[j,col] += concatVal * dZ[col]
            for (let col = 0; col < S; col++) {
              gateW.g[row + col] += concatVal * this._dZScale[col];
            }

            // dConcat
            let dConcat = 0;
            for (let col = 0; col < S; col++) {
              dConcat += gateW.w[row + col] * this._dZScale[col];
            }

            const dE = sig * this._dFused[dF + d] + dConcat;

            // scale embedding grad
            scaleEmb.g[s * D + d] += dE;

            // dConvPre = dE * gelu'(convPre)
            const dConvPre = dE * _geluDeriv(this._convPre[convBase + d]);

            // Conv grads
            const kMat = this._fusion.convK[s];
            const bMat = this._fusion.convB[s];

            bMat.g[d] += dConvPre;

            const center = (o * stride) | 0;
            const kRow = d * (inDim * K);
            for (let kk = 0; kk < K; kk++) {
              let tIn = center + (kk - pad);
              if (tIn < 0) tIn = 0;
              else if (tIn >= seqLen) tIn = seqLen - 1;

              const xOff = tIn * inDim;
              const kOff = kRow + kk * inDim;
              for (let c = 0; c < inDim; c++) {
                kMat.g[kOff + c] += dConvPre * this._xNorm[xOff + c];
              }
            }
          }
        }
      }
    }

    // L2 to grads
    const reg = this._cfg.regularizationStrength;
    if (reg > 0) {
      for (let p = 0; p < this._params.length; p++) {
        const pm = this._params[p];
        if (!pm.l2) continue;
        const w = pm.w;
        const g = pm.g;
        for (let i = 0; i < w.length; i++) g[i] += reg * w[i];
      }
    }

    // grad norm
    let gn = 0;
    for (let p = 0; p < this._params.length; p++) {
      const g = this._params[p].g;
      for (let i = 0; i < g.length; i++) gn += g[i] * g[i];
    }
    return Math.sqrt(gn);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Optimizer
  // ─────────────────────────────────────────────────────────────────────────────

  private _lrSchedule(step: number): number {
    const lr0 = this._cfg.learningRate;
    const warm = this._cfg.warmupSteps;
    const total = Math.max(warm + 1, this._cfg.totalSteps);

    if (warm > 0 && step < warm) return lr0 * (step / warm);

    const prog = (step - warm) / Math.max(1, total - warm);
    const c = 0.5 * (1 + Math.cos(Math.PI * _clamp01(prog)));
    return lr0 * c;
  }

  private _adamUpdate(lr: number): void {
    const b1 = this._cfg.beta1;
    const b2 = this._cfg.beta2;
    const eps = this._cfg.epsilon;

    const t = this._step;
    const b1t = 1 - Math.pow(b1, t);
    const b2t = 1 - Math.pow(b2, t);

    for (let p = 0; p < this._params.length; p++) {
      const pm = this._params[p];
      const w = pm.w;
      const g = pm.g;
      const m = pm.m;
      const v = pm.v;

      for (let i = 0; i < w.length; i++) {
        const gi = g[i];
        const mi = (m[i] = b1 * m[i] + (1 - b1) * gi);
        const vi = (v[i] = b2 * v[i] + (1 - b2) * (gi * gi));

        const mHat = mi / (b1t + 1e-30);
        const vHat = vi / (b2t + 1e-30);

        w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
  }

  private _zeroGrads(): void {
    for (let p = 0; p < this._params.length; p++) this._params[p].g.fill(0);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // LayerNorm
  // ─────────────────────────────────────────────────────────────────────────────

  private _layerNormForward(
    x: Float64Array,
    out: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
    xhatBuf: Float64Array,
    meanBase: number,
    xhatBase: number,
    seqLen: number,
    D: number,
    eps: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const off = t * D;
      let mean = 0;
      for (let d = 0; d < D; d++) mean += x[off + d];
      mean /= D;

      let varSum = 0;
      for (let d = 0; d < D; d++) {
        const z = x[off + d] - mean;
        varSum += z * z;
      }
      const invStd = 1 / Math.sqrt(varSum / D + eps);

      meanBuf[meanBase + t] = mean;
      invStdBuf[meanBase + t] = invStd;

      const xhOff = xhatBase + off;
      for (let d = 0; d < D; d++) {
        const xhat = (x[off + d] - mean) * invStd;
        xhatBuf[xhOff + d] = xhat;
        out[off + d] = gamma[d] * xhat + beta[d];
      }
    }
  }

  private _layerNormBackward(
    x: Float64Array,
    dY: Float64Array,
    gamma: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
    xhatBuf: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
    dX_accum: Float64Array,
    meanBase: number,
    xhatBase: number,
    seqLen: number,
    D: number,
    eps: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const off = t * D;
      const xhOff = xhatBase + off;

      let sum1 = 0;
      let sum2 = 0;

      for (let d = 0; d < D; d++) {
        const dy = dY[off + d];
        dBeta[d] += dy;
        dGamma[d] += dy * xhatBuf[xhOff + d];
        const dxhat = dy * gamma[d];
        sum1 += dxhat;
        sum2 += dxhat * xhatBuf[xhOff + d];
      }

      const invStd = invStdBuf[meanBase + t];
      const invD = 1 / D;

      for (let d = 0; d < D; d++) {
        const dy = dY[off + d];
        const dxhat = dy * gamma[d];
        const xhat = xhatBuf[xhOff + d];
        const dx = invStd * (dxhat - sum1 * invD - xhat * sum2 * invD);
        dX_accum[off + d] += dx;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Welford normalization
  // ─────────────────────────────────────────────────────────────────────────────

  private _updateWelfordX(xSeqPacked: Float64Array, seqLen: number): void {
    const inDim = this._inDim;
    for (let t = 0; t < seqLen; t++) {
      const off = t * inDim;
      this._xCount++;
      const n = this._xCount;

      for (let d = 0; d < inDim; d++) {
        const x = xSeqPacked[off + d];
        const mean = this._xMean[d];
        const delta = x - mean;
        const mean2 = mean + delta / n;
        const delta2 = x - mean2;
        this._xMean[d] = mean2;
        this._xM2[d] += delta * delta2;
      }
    }

    // update std
    if (this._xCount > 1) {
      const inv = 1 / (this._xCount - 1);
      for (let d = 0; d < inDim; d++) {
        this._xStd[d] = Math.max(1e-6, Math.sqrt(this._xM2[d] * inv));
      }
    } else {
      this._xStd.fill(1);
    }
  }

  private _updateWelfordY(y: Float64Array): void {
    this._yCount++;
    const n = this._yCount;

    for (let d = 0; d < this._outDim; d++) {
      const x = y[d];
      const mean = this._yMean[d];
      const delta = x - mean;
      const mean2 = mean + delta / n;
      const delta2 = x - mean2;
      this._yMean[d] = mean2;
      this._yM2[d] += delta * delta2;
    }

    if (this._yCount > 1) {
      const inv = 1 / (this._yCount - 1);
      for (let d = 0; d < this._outDim; d++) {
        this._yStd[d] = Math.max(1e-6, Math.sqrt(this._yM2[d] * inv));
      }
    } else {
      this._yStd.fill(1);
    }
  }

  private _normalizeX(
    x: Float64Array,
    out: Float64Array,
    seqLen: number,
  ): void {
    const inDim = this._inDim;
    const eps = this._cfg.epsilon;
    for (let t = 0; t < seqLen; t++) {
      const off = t * inDim;
      for (let d = 0; d < inDim; d++) {
        out[off + d] = (x[off + d] - this._xMean[d]) / (this._xStd[d] + eps);
      }
    }
  }

  private _normalizeY(y: Float64Array, out: Float64Array): void {
    const eps = this._cfg.epsilon;
    for (let d = 0; d < this._outDim; d++) {
      out[d] = (y[d] - this._yMean[d]) / (this._yStd[d] + eps);
    }
  }

  private _resetNormalizationOnly(): void {
    this._xCount = 0;
    this._yCount = 0;
    if (this._isInitialized) {
      this._xMean.fill(0);
      this._xM2.fill(0);
      this._xStd.fill(1);
      this._yMean.fill(0);
      this._yM2.fill(0);
      this._yStd.fill(1);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // ADWIN (bounded cost)
  // ─────────────────────────────────────────────────────────────────────────────

  private _adwinUpdateAndCheck(err: number): boolean {
    if (this._adwinLen >= this._adwinCap) {
      // drop oldest half
      const half = (this._adwinLen >> 1) | 0;
      for (let i = 0; i < this._adwinLen - half; i++) {
        this._adwinWin[i] = this._adwinWin[i + half];
      }
      this._adwinLen -= half;
    }

    this._adwinWin[this._adwinLen++] = err;
    this._rebuildAdwinPrefix();

    const n = this._adwinLen;
    if (n < 32) return false;

    const delta = this._cfg.adwinDelta;
    const lnTerm = Math.log(4 / delta);

    for (let cut = 16; cut <= n - 16; cut += 8) {
      const n0 = cut;
      const n1 = n - cut;
      const sum0 = this._adwinPref[cut - 1];
      const sum1 = this._adwinPref[n - 1] - sum0;

      const m0 = sum0 / n0;
      const m1 = sum1 / n1;

      const hm = (1 / n0) + (1 / n1);
      const epsCut = Math.sqrt(hm * lnTerm / 2);

      if (Math.abs(m0 - m1) >= epsCut) {
        // remove older portion
        for (let i = 0; i < n1; i++) {
          this._adwinWin[i] = this._adwinWin[cut + i];
        }
        this._adwinLen = n1;
        this._rebuildAdwinPrefix();
        return true;
      }
    }

    return false;
  }

  private _rebuildAdwinPrefix(): void {
    let sum = 0;
    for (let i = 0; i < this._adwinLen; i++) {
      sum += this._adwinWin[i];
      this._adwinPref[i] = sum;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Utilities
  // ─────────────────────────────────────────────────────────────────────────────

  private _copySequenceWindow(
    xSeq: number[][],
    start: number,
    len: number,
    inDim: number,
  ): void {
    this._ensureSeqCapacity(this._cfg.maxSequenceLength);
    for (let t = 0; t < len; t++) {
      const row = xSeq[start + t];
      const off = t * inDim;
      for (let d = 0; d < inDim; d++) this._xRaw[off + d] = +row[d];
    }
  }

  private _storeLastXNorm(xNorm: Float64Array, seqLen: number): void {
    const n = seqLen * this._inDim;
    for (let i = 0; i < n; i++) this._lastXNormSeq[i] = xNorm[i];
  }

  private _loadLastXNormIntoWorking(seqLen: number): void {
    const n = seqLen * this._inDim;
    for (let i = 0; i < n; i++) this._xNorm[i] = this._lastXNormSeq[i];
  }

  private _xavierInit(w: Float64Array, fanIn: number, fanOut: number): void {
    const a = Math.sqrt(6 / Math.max(1, fanIn + fanOut));
    for (let i = 0; i < w.length; i++) w[i] = (this._rand01() * 2 - 1) * a;
  }

  private _rand01(): number {
    let x = this._rngState | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this._rngState = x | 0;
    return ((x >>> 0) / 4294967296);
  }

  private _countParams(): number {
    if (!this._isInitialized) return 0;
    let n = 0;
    for (let i = 0; i < this._params.length; i++) n += this._params[i].w.length;
    return n;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

function _sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  } else {
    const z = Math.exp(x);
    return z / (1 + z);
  }
}

/**
 * GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
 */
function _gelu(x: number): number {
  const c = 0.7978845608028654;
  const x3 = x * x * x;
  const t = c * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  return 0.5 * x * (1 + th);
}

function _geluDeriv(x: number): number {
  const c = 0.7978845608028654;
  const x2 = x * x;
  const x3 = x2 * x;
  const t = c * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  const sech2 = 1 - th * th;
  const dt = c * (1 + 3 * 0.044715 * x2);
  return 0.5 * (1 + th) + 0.5 * x * sech2 * dt;
}

function _clamp01(x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x;
}

function _clampInt(x: number, lo: number, hi: number): number {
  const v = x | 0;
  if (v < lo) return lo | 0;
  if (v > hi) return hi | 0;
  return v;
}

function _pos(x: number, fallback: number): number {
  return x > 0 ? x : fallback;
}

function _sanitizeScales(scales: number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < scales.length; i++) {
    const s = scales[i] | 0;
    if (s > 0) out.push(s);
  }
  if (out.length === 0) out.push(1);
  out.sort((a, b) => a - b);
  const uniq: number[] = [];
  let prev = 0;
  for (let i = 0; i < out.length; i++) {
    if (i === 0 || out[i] !== prev) uniq.push(out[i]);
    prev = out[i];
  }
  return uniq;
}

function _nextPow2(x: number): number {
  let v = x | 0;
  if (v <= 1) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return (v + 1) | 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers (row-major)
// ─────────────────────────────────────────────────────────────────────────────

function _matMulSeq(
  X: Float64Array,
  out: Float64Array,
  W: Float64Array,
  seqLen: number,
  inCols: number,
  outCols: number,
  bias: Float64Array | 0,
  dstBase: number,
): void {
  for (let i = 0; i < seqLen; i++) {
    const xOff = i * inCols;
    const oOff = dstBase + i * outCols;

    for (let j = 0; j < outCols; j++) {
      let s = bias ? bias[j] : 0;
      for (let k = 0; k < inCols; k++) s += X[xOff + k] * W[k * outCols + j];
      out[oOff + j] = s;
    }
  }
}

/**
 * Backprop for projection with offset in dOut:
 * - mat.g += X^T * dOut
 * - dX_accum += dOut * mat.w^T
 */
function _projBackwardOffset(
  X: Float64Array,
  dOutBig: Float64Array,
  dOutBase: number,
  mat: _Mat,
  dX_accum: Float64Array,
  seqLen: number,
  D: number,
): void {
  const W = mat.w;
  const G = mat.g;

  // G += X^T * dOut
  for (let r = 0; r < D; r++) {
    const row = r * D;
    for (let c = 0; c < D; c++) {
      let s = 0;
      for (let i = 0; i < seqLen; i++) {
        s += X[i * D + r] * dOutBig[dOutBase + i * D + c];
      }
      G[row + c] += s;
    }
  }

  // dX += dOut * W^T
  for (let i = 0; i < seqLen; i++) {
    const off = i * D;
    const doff = dOutBase + off;
    for (let r = 0; r < D; r++) {
      let s = 0;
      const row = r * D;
      for (let c = 0; c < D; c++) s += dOutBig[doff + c] * W[row + c];
      dX_accum[off + r] += s;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export helpers (conversion)
// ─────────────────────────────────────────────────────────────────────────────

function _arrTo2D(arr: Float64Array, rows: number, cols: number): number[][] {
  const out: number[][] = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row: number[] = new Array(cols);
    const off = r * cols;
    for (let c = 0; c < cols; c++) row[c] = arr[off + c];
    out[r] = row;
  }
  return out;
}

function _matTo2D(m: _Mat): number[][] {
  return _arrTo2D(m.w, m.rows, m.cols);
}
