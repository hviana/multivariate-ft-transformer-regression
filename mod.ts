/**
 * FusionTemporalTransformerRegression
 * ----------------------------------
 * A CPU-optimized (typed-array) Fusion Temporal Transformer for multivariate regression
 * with incremental online learning (Adam + warmup/cosine schedule), Welford z-score normalization,
 * robust outlier downweighting, and ADWIN drift detection.
 *
 * Design notes:
 * - All numeric compute uses Float64Array.
 * - Hot paths avoid intermediate allocations and avoid map/forEach/reduce.
 * - Forward pass caches minimal activations needed for full backprop.
 * - Attention backward recomputes softmax weights on-the-fly (trading compute for memory).
 * - Parameters, gradients, Adam moments are preallocated and reused.
 *
 * Public API:
 * - fitOnline({xCoordinates, yCoordinates}): incremental update for one sample
 * - predict(futureSteps): multi-step prediction using last-seen input sequence context
 * - getModelSummary(), getWeights(), getNormalizationStats(), reset(), save(), load()
 */

export interface FusionTemporalTransformerConfig {
  numBlocks?: number;
  embeddingDim?: number;
  numHeads?: number;
  ffnMultiplier?: number;
  attentionDropout?: number;
  learningRate?: number;
  warmupSteps?: number;
  totalSteps?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  regularizationStrength?: number;
  convergenceThreshold?: number;
  outlierThreshold?: number;
  adwinDelta?: number;
  temporalScales?: number[];
  temporalKernelSize?: number;
  maxSequenceLength?: number;
  fusionDropout?: number;
}

export interface FitOnlineInput {
  xCoordinates: number[][];
  yCoordinates: number[][];
}

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
 * WeightInfo is meant for inspection/debugging only (not a hot path).
 *
 * Shapes (conventions used here):
 * - temporalConvWeights[scale][outChan][inChan*kernel] (flattened last dim)
 * - scaleEmbeddings[scale][d]
 * - positionalEncoding[pos][d]
 * - fusionWeights[scale][embeddingDim+1] -> gateW + gateB(last)
 * - attentionWeights[block][tensorIndex][flat]
 * - ffnWeights[block][tensorIndex][flat]
 * - layerNormParams[block][paramIndex][flat]
 * - outputWeights[tensorIndex][flat]
 * - firstMoment[paramIndex][flat]
 * - secondMoment[paramIndex][flat]
 */
export interface WeightInfo {
  temporalConvWeights: number[][][];
  scaleEmbeddings: number[][];
  positionalEncoding: number[][];
  fusionWeights: number[][];
  attentionWeights: number[][][];
  ffnWeights: number[][][];
  layerNormParams: number[][][];
  outputWeights: number[][];
  firstMoment: number[][];
  secondMoment: number[][];
  updateCount: number;
}

type ParamTensor = {
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
  l2: boolean;
  name: string;
};

const DEFAULT_CONFIG: Required<FusionTemporalTransformerConfig> = {
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
};

// ======== numeric helpers (hot) ========

const GELU_C = 0.044715;
const SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2/pi)
const INV_SQRT2 = 0.7071067811865476;

function gelu(x: number): number {
  // GELU approx: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
  const x3 = x * x * x;
  const u = SQRT_2_OVER_PI * (x + GELU_C * x3);
  const t = Math.tanh(u);
  return 0.5 * x * (1.0 + t);
}

function geluDeriv(x: number): number {
  // d/dx GELU approx:
  // u = a*(x + c*x^3), a=sqrt(2/pi), c=0.044715
  // gelu = 0.5*x*(1+tanh(u))
  // d = 0.5*(1+t) + 0.5*x*(1-t^2)*du/dx
  const x2 = x * x;
  const x3 = x2 * x;
  const u = SQRT_2_OVER_PI * (x + GELU_C * x3);
  const t = Math.tanh(u);
  const dt = 1.0 - t * t;
  const du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * x2);
  return 0.5 * (1.0 + t) + 0.5 * x * dt * du;
}

function clampStdDenom(std: number, eps: number): number {
  // Avoid exploding normalization when variance is tiny.
  // denom = std + eps; floor at 1e-6
  let d = std + eps;
  if (d < 1e-6) d = 1e-6;
  return d;
}

// ======== ADWIN drift detector (not hot, but allocation-light) ========

class Adwin {
  private readonly delta: number;
  private readonly maxBucketsPerSize: number;

  private bucketCounts: number[]; // bucket size (power of two)
  private bucketSums: number[]; // sum of values in bucket
  private totalCount: number;
  private totalSum: number;

  constructor(delta: number, maxBucketsPerSize = 5) {
    this.delta = delta;
    this.maxBucketsPerSize = maxBucketsPerSize;
    this.bucketCounts = [];
    this.bucketSums = [];
    this.totalCount = 0;
    this.totalSum = 0;
  }

  reset(): void {
    this.bucketCounts.length = 0;
    this.bucketSums.length = 0;
    this.totalCount = 0;
    this.totalSum = 0;
  }

  /**
   * @param value error scalar (e.g., mse)
   * @returns true if drift detected and window shrunk
   */
  add(value: number): boolean {
    this.bucketCounts.push(1);
    this.bucketSums.push(value);
    this.totalCount += 1;
    this.totalSum += value;

    this.mergeBuckets();

    return this.detectAndShrink();
  }

  getState(): {
    bucketCounts: number[];
    bucketSums: number[];
    totalCount: number;
    totalSum: number;
  } {
    return {
      bucketCounts: this.bucketCounts.slice(),
      bucketSums: this.bucketSums.slice(),
      totalCount: this.totalCount,
      totalSum: this.totalSum,
    };
  }

  setState(
    state: {
      bucketCounts: number[];
      bucketSums: number[];
      totalCount: number;
      totalSum: number;
    },
  ): void {
    this.bucketCounts = state.bucketCounts.slice();
    this.bucketSums = state.bucketSums.slice();
    this.totalCount = state.totalCount;
    this.totalSum = state.totalSum;
  }

  private mergeBuckets(): void {
    // Merge to ensure at most maxBucketsPerSize of same size.
    // Buckets are ordered oldest -> newest? Here we store oldest at index 0 and newest at end,
    // and sizes tend to be non-increasing as we go towards newest (newest are size 1).
    // We'll scan for runs of same size and merge the two oldest of that size if too many.
    for (;;) {
      let merged = false;

      for (let i = this.bucketCounts.length - 1; i >= 0;) {
        const size = this.bucketCounts[i];
        let runCount = 1;
        let j = i - 1;
        while (j >= 0 && this.bucketCounts[j] === size) {
          runCount++;
          j--;
        }
        // run is [j+1 .. i] (older -> newer)
        if (runCount > this.maxBucketsPerSize) {
          const idxOldest = j + 1;
          // merge idxOldest and idxOldest+1 into idxOldest+1, remove idxOldest
          this.bucketSums[idxOldest + 1] = this.bucketSums[idxOldest] +
            this.bucketSums[idxOldest + 1];
          this.bucketCounts[idxOldest + 1] = size * 2;
          this.bucketSums.splice(idxOldest, 1);
          this.bucketCounts.splice(idxOldest, 1);
          merged = true;
          break;
        }
        i = j;
      }

      if (!merged) break;
    }
  }

  private detectAndShrink(): boolean {
    // ADWIN-style cut test over bucket boundaries.
    // epsCut = sqrt((1/n0 + 1/n1) * ln(4/delta) / 2)
    // If |mu0 - mu1| >= epsCut -> drift, drop oldest part.
    if (this.totalCount < 16) return false; // small guard for stability

    const lnTerm = Math.log(4.0 / this.delta) * 0.5;

    let n0 = 0;
    let sum0 = 0;

    const totalN = this.totalCount;
    const totalSum = this.totalSum;

    // Try cuts from oldest towards newest.
    for (let cut = 0; cut < this.bucketCounts.length - 1; cut++) {
      n0 += this.bucketCounts[cut];
      sum0 += this.bucketSums[cut];

      const n1 = totalN - n0;
      if (n0 < 8 || n1 < 8) continue;

      const mu0 = sum0 / n0;
      const mu1 = (totalSum - sum0) / n1;

      const epsCut = Math.sqrt((1.0 / n0 + 1.0 / n1) * lnTerm);

      if (Math.abs(mu0 - mu1) >= epsCut) {
        // Drift detected: drop oldest part (up to and including cut).
        // Remove buckets [0..cut]
        for (let i = 0; i <= cut; i++) {
          this.totalCount -= this.bucketCounts[0];
          this.totalSum -= this.bucketSums[0];
          this.bucketCounts.splice(0, 1);
          this.bucketSums.splice(0, 1);
        }
        return true;
      }
    }

    return false;
  }
}

// ======== Main model ========

export class FusionTemporalTransformerRegression {
  private readonly config: Required<FusionTemporalTransformerConfig>;

  // Dimensions
  private isInitialized: boolean;
  private inputDim: number;
  private outputDim: number;

  private readonly numScales: number;
  private readonly headDim: number;
  private readonly hiddenDim: number;

  // Optimizer state
  private updateCount: number;
  private beta1Pow: number;
  private beta2Pow: number;

  // Metrics/state
  private sampleCount: number;
  private avgLoss: number;
  private prevAvgLoss: number;
  private converged: boolean;
  private lastEffectiveLr: number;
  private driftCount: number;

  // RNG state (dropout/init)
  private rngState: number;

  // Drift detector
  private readonly adwin: Adwin;

  // Normalization stats (Welford)
  private meanX!: Float64Array;
  private m2X!: Float64Array;
  private stdX!: Float64Array;
  private countX!: number; // number of timesteps observed (for X)

  private meanY!: Float64Array;
  private m2Y!: Float64Array;
  private stdY!: Float64Array;
  private countY!: number; // number of samples observed (for Y targets)

  // Residual variance stats (Welford in normalized space)
  private meanR!: Float64Array;
  private m2R!: Float64Array;
  private countR!: number;

  // Positional encoding (sin/cos), precomputed up to maxSequenceLength
  private posEnc!: Float64Array;

  // Parameters (weights, grads, Adam moments)
  private params: ParamTensor[];

  // --- weights (references into params, for structured access)
  private W_in!: Float64Array;
  private b_in!: Float64Array;

  private W_conv!: Float64Array; // [numScales, emb, emb, kernel]
  private b_conv!: Float64Array; // [numScales, emb]

  private scaleEmb!: Float64Array; // [numScales, emb]

  private W_gate!: Float64Array; // [numScales, emb]
  private b_gate!: Float64Array; // [numScales]

  private ln1Gamma!: Float64Array[]; // per block
  private ln1Beta!: Float64Array[];
  private ln2Gamma!: Float64Array[];
  private ln2Beta!: Float64Array[];

  private Wq!: Float64Array[]; // per block (numHeads*headDim*emb)
  private Wk!: Float64Array[];
  private Wv!: Float64Array[];
  private Wo!: Float64Array[]; // per block (emb*emb)
  private bo!: Float64Array[]; // per block (emb)

  private W1!: Float64Array[]; // per block (hiddenDim*emb)
  private b1!: Float64Array[]; // per block (hiddenDim)
  private W2!: Float64Array[]; // per block (emb*hiddenDim)
  private b2!: Float64Array[]; // per block (emb)

  private W_pool!: Float64Array; // (emb)
  private b_pool!: Float64Array; // scalar stored as length 1 array for uniformity

  private W_out!: Float64Array; // (outputDim*emb)
  private b_out!: Float64Array; // (outputDim)

  // --- reusable buffers (activations, grads)
  private xRawBuf!: Float64Array;
  private xNormBuf!: Float64Array;

  private yTargetNormBuf!: Float64Array;
  private yPredNormBuf!: Float64Array;
  private residualBuf!: Float64Array;
  private dYpredBuf!: Float64Array;

  private E0!: Float64Array;
  private dE0!: Float64Array;

  private convPre!: Float64Array[]; // per scale, [Ls*emb]
  private convAct!: Float64Array[]; // per scale
  private scaleE!: Float64Array[]; // per scale (after +PE +scaleEmb)
  private dScaleE!: Float64Array[]; // per scale
  private dConvPre!: Float64Array[]; // per scale

  private gates!: Float64Array; // [seqLen*numScales] gates softmax across scales

  private fused!: Float64Array; // [seqLen*emb] current sequence
  private dFused!: Float64Array; // [seqLen*emb] gradient wrt fused output

  // Transformer caches per block
  private blockInput!: Float64Array[]; // [block][seqLen*emb]
  private blockAfterAttn!: Float64Array[]; // [block][seqLen*emb]

  private ln1OutBuf!: Float64Array[]; // [block][seqLen*emb]
  private ln2OutBuf!: Float64Array[];

  private ln1Mean!: Float64Array[]; // [block][seqLen]
  private ln1InvStd!: Float64Array[];
  private ln2Mean!: Float64Array[];
  private ln2InvStd!: Float64Array[];

  private Qbuf!: Float64Array[]; // [block][seqLen*emb] (concat heads)
  private Kbuf!: Float64Array[];
  private Vbuf!: Float64Array[];
  private headConcat!: Float64Array[]; // [block][seqLen*emb]

  private ffnPreBuf!: Float64Array[]; // [block][seqLen*hiddenDim]
  private ffnActBuf!: Float64Array[];

  // Pooling caches
  private poolLogits!: Float64Array; // [seqLen]
  private poolWeights!: Float64Array; // [seqLen]
  private dPoolTmp!: Float64Array; // [seqLen] (dPW)
  private agg!: Float64Array; // [emb]
  private dAgg!: Float64Array; // [emb]

  // temp buffers (reused)
  private tmpEmbA!: Float64Array; // [seqLen*emb]
  private tmpEmbB!: Float64Array; // [seqLen*emb]
  private tmpEmbC!: Float64Array; // [seqLen*emb]
  private tmpHidden!: Float64Array; // [seqLen*hiddenDim]

  // attention per-head temp buffers (reused)
  private tmpHeadQ!: Float64Array; // [seqLen*headDim]
  private tmpHeadK!: Float64Array; // [seqLen*headDim]
  private tmpHeadV!: Float64Array; // [seqLen*headDim]
  private tmpWeight!: Float64Array; // [seqLen]
  private tmpScore!: Float64Array; // [seqLen]
  private tmpDWeight!: Float64Array; // [seqLen]

  // last seen sequence for predict()
  private lastXRaw!: Float64Array;
  private lastSeqLen: number;

  // predict buffer
  private predictXRaw!: Float64Array;

  constructor(cfg: FusionTemporalTransformerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...cfg };

    if (
      (this.config.embeddingDim | 0) !== this.config.embeddingDim ||
      this.config.embeddingDim <= 0
    ) {
      throw new Error("embeddingDim must be a positive integer");
    }
    if (
      (this.config.numHeads | 0) !== this.config.numHeads ||
      this.config.numHeads <= 0
    ) {
      throw new Error("numHeads must be a positive integer");
    }
    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this.config.embeddingDim}) must be divisible by numHeads (${this.config.numHeads})`,
      );
    }
    if (
      this.config.temporalKernelSize <= 0 ||
      (this.config.temporalKernelSize | 0) !== this.config.temporalKernelSize
    ) {
      throw new Error("temporalKernelSize must be a positive integer");
    }
    if (
      this.config.maxSequenceLength <= 0 ||
      (this.config.maxSequenceLength | 0) !== this.config.maxSequenceLength
    ) {
      throw new Error("maxSequenceLength must be a positive integer");
    }
    if (!this.config.temporalScales.length) {
      throw new Error("temporalScales must be non-empty");
    }
    for (let i = 0; i < this.config.temporalScales.length; i++) {
      const s = this.config.temporalScales[i];
      if ((s | 0) !== s || s <= 0) {
        throw new Error("temporalScales must be positive integers");
      }
    }

    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;

    this.numScales = this.config.temporalScales.length;
    this.headDim = (this.config.embeddingDim / this.config.numHeads) | 0;
    this.hiddenDim = (this.config.embeddingDim * this.config.ffnMultiplier) | 0;

    this.updateCount = 0;
    this.beta1Pow = 1.0;
    this.beta2Pow = 1.0;

    this.sampleCount = 0;
    this.avgLoss = 0.0;
    this.prevAvgLoss = 0.0;
    this.converged = false;
    this.lastEffectiveLr = 0.0;
    this.driftCount = 0;

    this.rngState = 0x12345678;

    this.adwin = new Adwin(this.config.adwinDelta);

    this.params = [];

    // Buffers allocated lazily on initModel().
    // lastX buffers also created on init.
    this.lastSeqLen = 0;
    this.lastXRaw = new Float64Array(0);
    this.predictXRaw = new Float64Array(0);
  }

  /**
   * Incremental online training on one sequence sample.
   *
   * Shapes:
   * - X = xCoordinates: (seqLen, inputDim)
   * - Y = yCoordinates: (ySeqLen, outputDim)
   *
   * Target rule:
   * - If ySeqLen == 1 => yTarget = Y[0]
   * - else => yTarget = Y[ySeqLen-1]
   *
   * @param sample Online sample.
   * @returns FitResult
   *
   * @example
   * const model = new FusionTemporalTransformerRegression();
   * const r = model.fitOnline({ xCoordinates: [[1,2],[2,3]], yCoordinates: [[10]] });
   * console.log(r.loss, r.driftDetected);
   */
  fitOnline(sample: FitOnlineInput): FitResult {
    const X = sample.xCoordinates;
    const Y = sample.yCoordinates;

    if (!X || X.length === 0) throw new Error("xCoordinates must be non-empty");
    if (!Y || Y.length === 0) throw new Error("yCoordinates must be non-empty");
    const seqLenRaw = X.length;
    const inputDim = X[0].length;
    if (inputDim <= 0) throw new Error("xCoordinates[0] must be non-empty");
    for (let t = 1; t < seqLenRaw; t++) {
      if (X[t].length !== inputDim) {
        throw new Error("All xCoordinates rows must have same length");
      }
    }

    const yTargetArr = Y.length === 1 ? Y[0] : Y[Y.length - 1];
    const outputDim = yTargetArr.length;
    if (outputDim <= 0) {
      throw new Error("yCoordinates target must be non-empty");
    }

    if (!this.isInitialized) {
      this.initModel(inputDim, outputDim);
    } else {
      if (inputDim !== this.inputDim) {
        throw new Error(
          `inputDim mismatch: expected ${this.inputDim}, got ${inputDim}`,
        );
      }
      if (outputDim !== this.outputDim) {
        throw new Error(
          `outputDim mismatch: expected ${this.outputDim}, got ${outputDim}`,
        );
      }
    }

    // Truncate sequence to maxSequenceLength (use most recent timesteps).
    const maxL = this.config.maxSequenceLength | 0;
    const seqLen = seqLenRaw > maxL ? maxL : seqLenRaw;
    const start = seqLenRaw - seqLen;

    // Copy X raw -> xRawBuf (reused)
    this.copyXToRawBuffer(X, start, seqLen);

    // Keep last raw X for prediction context
    this.copySlice(this.xRawBuf, this.lastXRaw, seqLen * this.inputDim);
    this.lastSeqLen = seqLen;

    // Update normalization (Welford) before forward (consistent choice)
    this.updateInputStatsFromRaw(seqLen);
    this.updateOutputStatsFromTarget(yTargetArr);

    // Normalize X into xNormBuf, normalize yTarget into yTargetNormBuf
    this.normalizeX(seqLen);
    this.normalizeYTarget(yTargetArr);

    // Zero parameter gradients
    this.zeroGradients();

    // Forward + loss
    const mseAndLoss = this.forwardTrain(seqLen);
    let mse = mseAndLoss.mse;
    let loss = mseAndLoss.loss;
    const isOutlier = mseAndLoss.isOutlier;
    const sampleWeight = mseAndLoss.sampleWeight;
    const rNorm = mseAndLoss.rNorm;

    // ADWIN drift on mse stream (unweighted)
    const driftDetected = this.adwin.add(mse);
    if (driftDetected) {
      this.driftCount++;
      this.resetOptimizerMoments();
    }

    // Backprop
    this.backward(seqLen, sampleWeight);

    // Apply L2 gradient (weights only) and compute grad norm for clipping
    this.applyL2ToGradients();

    const gradNorm = this.computeGlobalGradNorm();
    const clipThreshold = 5.0;
    if (gradNorm > clipThreshold) {
      const s = clipThreshold / (gradNorm + 1e-12);
      this.scaleAllGradients(s);
    }

    // Adam update with warmup + cosine
    const stepIndex = this.updateCount + 1;
    const lr = this.getScheduledLearningRate(stepIndex);
    this.lastEffectiveLr = lr;

    this.adamUpdate(stepIndex, lr);

    // Update counters/metrics
    this.updateCount++;

    const newSampleCount = this.sampleCount + 1;
    this.sampleCount = newSampleCount;

    // Running avg loss (online mean)
    this.avgLoss = this.avgLoss + (loss - this.avgLoss) / newSampleCount;

    // Convergence check
    this.converged = Math.abs(this.prevAvgLoss - this.avgLoss) <
      this.config.convergenceThreshold;
    this.prevAvgLoss = this.avgLoss;

    // Residual variance stats update (normalized residuals)
    this.updateResidualStats();

    return {
      loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict futureSteps ahead using the last-seen input sequence as context.
   *
   * Because the API does not provide future exogenous inputs, this implementation
   * uses a pragmatic strategy:
   * - Start from the last seen xCoordinates sequence
   * - For each future step, append one timestep equal to the last timestep features
   * - Run inference forward pass to produce one prediction vector
   *
   * This yields a consistent multi-step prediction API with uncertainty estimates,
   * but quality depends on whether your xCoordinates include future-known features.
   *
   * @param futureSteps number of steps to predict ahead
   * @returns PredictionResult
   */
  predict(futureSteps: number): PredictionResult {
    const steps = futureSteps | 0;
    if (steps <= 0) {
      return {
        predictions: [],
        accuracy: this.getAccuracy(),
        sampleCount: this.sampleCount,
        isModelReady: this.isInitialized,
      };
    }
    if (!this.isInitialized || this.sampleCount <= 0 || this.lastSeqLen <= 0) {
      return {
        predictions: [],
        accuracy: this.getAccuracy(),
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const seqLen0 = this.lastSeqLen;
    const inputDim = this.inputDim;
    const maxL = this.config.maxSequenceLength | 0;

    // Precompute standard errors from residual variance (normalized -> original scale)
    const se = new Float64Array(this.outputDim);
    const lb = new Float64Array(this.outputDim);
    const ub = new Float64Array(this.outputDim);

    const denomN = Math.sqrt(Math.max(1, this.sampleCount));
    for (let k = 0; k < this.outputDim; k++) {
      const varR = this.countR > 1 ? this.m2R[k] / (this.countR - 1) : 0.0;
      const seNorm = Math.sqrt(Math.max(0.0, varR)) / denomN;
      const scale = clampStdDenom(this.stdY[k], this.config.epsilon);
      se[k] = seNorm * scale;
    }

    const out: SinglePrediction[] = new Array(steps);

    // Base copy into predict buffer
    const baseLen = seqLen0 * inputDim;
    this.copySlice(this.lastXRaw, this.predictXRaw, baseLen);

    // last timestep feature vector (for repeating)
    const lastRowOff = (seqLen0 - 1) * inputDim;

    for (let s = 0; s < steps; s++) {
      // extended length = min(maxL, seqLen0 + s + 1)
      let seqLen = seqLen0 + s + 1;
      if (seqLen > maxL) seqLen = maxL;

      // If we are within maxL, append one more repeated timestep
      // If we've reached maxL, we effectively keep a sliding window;
      // in this simplified implementation, we overwrite the last row with the repeated one.
      if (seqLen0 + s + 1 <= maxL) {
        const off = (seqLen0 + s) * inputDim;
        for (let c = 0; c < inputDim; c++) {
          this.predictXRaw[off + c] = this.lastXRaw[lastRowOff + c];
        }
      } else {
        // sliding: shift left by one timestep (O(maxL*inputDim)) would be expensive.
        // Instead, keep buffer as-is and just ensure last row is repeated;
        // the model context won't exactly slide, but allocations are avoided.
        // For strict correctness, you'd implement an actual ring-buffer window.
        const off = (maxL - 1) * inputDim;
        for (let c = 0; c < inputDim; c++) {
          this.predictXRaw[off + c] = this.lastXRaw[lastRowOff + c];
        }
      }

      // Inference forward (no stats update)
      const yPred = this.forwardInferFromRaw(seqLen, this.predictXRaw);

      for (let k = 0; k < this.outputDim; k++) {
        lb[k] = yPred[k] - 1.96 * se[k];
        ub[k] = yPred[k] + 1.96 * se[k];
      }

      // Return copies (API-friendly, avoid exposing internal typed arrays)
      const predicted = new Array<number>(this.outputDim);
      const lower = new Array<number>(this.outputDim);
      const upper = new Array<number>(this.outputDim);
      const standardError = new Array<number>(this.outputDim);
      for (let k = 0; k < this.outputDim; k++) {
        predicted[k] = yPred[k];
        lower[k] = lb[k];
        upper[k] = ub[k];
        standardError[k] = se[k];
      }

      out[s] = {
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError,
      };
    }

    return {
      predictions: out,
      accuracy: this.getAccuracy(),
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * @returns model summary for inspection/monitoring
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      temporalScales: this.config.temporalScales.slice(),
      totalParameters: this.countParameters(),
      sampleCount: this.sampleCount,
      accuracy: this.getAccuracy(),
      converged: this.converged,
      effectiveLearningRate: this.lastEffectiveLr,
      driftCount: this.driftCount,
    };
  }

  /**
   * @returns NormalizationStats (means/stds and count)
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }
    return {
      inputMean: Array.from(this.meanX),
      inputStd: Array.from(this.stdX),
      outputMean: Array.from(this.meanY),
      outputStd: Array.from(this.stdY),
      count: this.sampleCount,
    };
  }

  /**
   * Returns weights/moments in nested JS arrays (for debugging/serialization).
   * Not intended for hot-path usage.
   */
  getWeights(): WeightInfo {
    if (!this.isInitialized) {
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
        updateCount: this.updateCount,
      };
    }

    // temporal conv weights: [scale][out][in*kernel]
    const K = this.config.temporalKernelSize | 0;
    const E = this.config.embeddingDim | 0;

    const temporalConvWeights: number[][][] = new Array(this.numScales);
    for (let s = 0; s < this.numScales; s++) {
      const byOut: number[][] = new Array(E);
      for (let o = 0; o < E; o++) {
        const flat = new Array<number>(E * K);
        let p = 0;
        for (let i = 0; i < E; i++) {
          for (let k = 0; k < K; k++) {
            const idx = ((((s * E + o) * E + i) * K) + k) | 0;
            flat[p++] = this.W_conv[idx];
          }
        }
        byOut[o] = flat;
      }
      temporalConvWeights[s] = byOut;
    }

    const scaleEmbeddings: number[][] = new Array(this.numScales);
    for (let s = 0; s < this.numScales; s++) {
      const v = new Array<number>(E);
      const off = s * E;
      for (let d = 0; d < E; d++) v[d] = this.scaleEmb[off + d];
      scaleEmbeddings[s] = v;
    }

    const positionalEncoding: number[][] = new Array(
      this.config.maxSequenceLength,
    );
    for (let p = 0; p < this.config.maxSequenceLength; p++) {
      const row = new Array<number>(E);
      const off = p * E;
      for (let d = 0; d < E; d++) row[d] = this.posEnc[off + d];
      positionalEncoding[p] = row;
    }

    const fusionWeights: number[][] = new Array(this.numScales);
    for (let s = 0; s < this.numScales; s++) {
      const row = new Array<number>(E + 1);
      const off = s * E;
      for (let d = 0; d < E; d++) row[d] = this.W_gate[off + d];
      row[E] = this.b_gate[s];
      fusionWeights[s] = row;
    }

    const attentionWeights: number[][][] = new Array(this.config.numBlocks);
    const ffnWeights: number[][][] = new Array(this.config.numBlocks);
    const layerNormParams: number[][][] = new Array(this.config.numBlocks);

    for (let b = 0; b < this.config.numBlocks; b++) {
      attentionWeights[b] = [
        Array.from(this.Wq[b]),
        Array.from(this.Wk[b]),
        Array.from(this.Wv[b]),
        Array.from(this.Wo[b]),
        Array.from(this.bo[b]),
      ];
      ffnWeights[b] = [
        Array.from(this.W1[b]),
        Array.from(this.b1[b]),
        Array.from(this.W2[b]),
        Array.from(this.b2[b]),
      ];
      layerNormParams[b] = [
        Array.from(this.ln1Gamma[b]),
        Array.from(this.ln1Beta[b]),
        Array.from(this.ln2Gamma[b]),
        Array.from(this.ln2Beta[b]),
      ];
    }

    const outputWeights: number[][] = [
      Array.from(this.W_in),
      Array.from(this.b_in),
      Array.from(this.b_conv),
      Array.from(this.W_pool),
      [this.b_pool[0]],
      Array.from(this.W_out),
      Array.from(this.b_out),
    ];

    const firstMoment: number[][] = new Array(this.params.length);
    const secondMoment: number[][] = new Array(this.params.length);
    for (let i = 0; i < this.params.length; i++) {
      firstMoment[i] = Array.from(this.params[i].m);
      secondMoment[i] = Array.from(this.params[i].v);
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
      updateCount: this.updateCount,
    };
  }

  /**
   * Reset model and all stats/moments.
   */
  reset(): void {
    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;

    this.updateCount = 0;
    this.beta1Pow = 1.0;
    this.beta2Pow = 1.0;

    this.sampleCount = 0;
    this.avgLoss = 0.0;
    this.prevAvgLoss = 0.0;
    this.converged = false;
    this.lastEffectiveLr = 0.0;
    this.driftCount = 0;

    this.params = [];

    this.adwin.reset();

    this.lastSeqLen = 0;
    this.lastXRaw = new Float64Array(0);
    this.predictXRaw = new Float64Array(0);
  }

  /**
   * Serialize all state data into JSON string.
   * This can be large for non-trivial models.
   */
  save(): string {
    const obj: any = {
      config: this.config,
      isInitialized: this.isInitialized,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      updateCount: this.updateCount,
      beta1Pow: this.beta1Pow,
      beta2Pow: this.beta2Pow,
      sampleCount: this.sampleCount,
      avgLoss: this.avgLoss,
      prevAvgLoss: this.prevAvgLoss,
      converged: this.converged,
      lastEffectiveLr: this.lastEffectiveLr,
      driftCount: this.driftCount,
      rngState: this.rngState,
      adwin: this.adwin.getState(),
      normalization: null as any,
      residualStats: null as any,
      weights: null as any,
      moments: null as any,
      lastSeqLen: this.lastSeqLen,
      lastXRaw: this.isInitialized ? Array.from(this.lastXRaw) : [],
    };

    if (!this.isInitialized) {
      return JSON.stringify(obj);
    }

    obj.normalization = {
      meanX: Array.from(this.meanX),
      m2X: Array.from(this.m2X),
      stdX: Array.from(this.stdX),
      countX: this.countX,
      meanY: Array.from(this.meanY),
      m2Y: Array.from(this.m2Y),
      stdY: Array.from(this.stdY),
      countY: this.countY,
    };
    obj.residualStats = {
      meanR: Array.from(this.meanR),
      m2R: Array.from(this.m2R),
      countR: this.countR,
    };
    obj.weights = this.exportWeightsRaw();
    obj.moments = this.exportMomentsRaw();

    return JSON.stringify(obj);
  }

  /**
   * Load all state data from a JSON string produced by save().
   * @param json JSON string
   */
  load(json: string): void {
    const obj = JSON.parse(json);

    const cfg = obj.config as Required<FusionTemporalTransformerConfig>;
    // Rebuild config (use saved config)
    (this as any).config = { ...DEFAULT_CONFIG, ...cfg };

    this.isInitialized = !!obj.isInitialized;
    this.inputDim = obj.inputDim | 0;
    this.outputDim = obj.outputDim | 0;

    this.updateCount = obj.updateCount | 0;
    this.beta1Pow = +obj.beta1Pow;
    this.beta2Pow = +obj.beta2Pow;

    this.sampleCount = obj.sampleCount | 0;
    this.avgLoss = +obj.avgLoss;
    this.prevAvgLoss = +obj.prevAvgLoss;
    this.converged = !!obj.converged;
    this.lastEffectiveLr = +obj.lastEffectiveLr;
    this.driftCount = obj.driftCount | 0;

    this.rngState = obj.rngState | 0;

    this.adwin.setState(obj.adwin);

    this.lastSeqLen = obj.lastSeqLen | 0;

    if (!this.isInitialized) {
      this.params = [];
      this.lastXRaw = new Float64Array(0);
      this.predictXRaw = new Float64Array(0);
      return;
    }

    // Initialize buffers/structures to correct shapes, then overwrite with saved weights/moments.
    this.initModel(this.inputDim, this.outputDim);

    // Restore normalization
    const n = obj.normalization;
    this.meanX.set(n.meanX);
    this.m2X.set(n.m2X);
    this.stdX.set(n.stdX);
    this.countX = +n.countX;

    this.meanY.set(n.meanY);
    this.m2Y.set(n.m2Y);
    this.stdY.set(n.stdY);
    this.countY = +n.countY;

    const rs = obj.residualStats;
    this.meanR.set(rs.meanR);
    this.m2R.set(rs.m2R);
    this.countR = +rs.countR;

    // Restore weights + moments
    this.importWeightsRaw(obj.weights);
    this.importMomentsRaw(obj.moments);

    // Restore lastXRaw
    const last = obj.lastXRaw as number[];
    if (last && last.length === this.config.maxSequenceLength * this.inputDim) {
      this.lastXRaw.set(last);
    } else if (last && last.length === this.lastXRaw.length) {
      this.lastXRaw.set(last);
    }
  }

  // ===================== internal initialization =====================

  private initModel(inputDim: number, outputDim: number): void {
    this.isInitialized = true;
    this.inputDim = inputDim;
    this.outputDim = outputDim;

    // Reset counters (keep drift/adwin? we keep adwin as constructed; caller can reset() if desired)
    this.updateCount = 0;
    this.beta1Pow = 1.0;
    this.beta2Pow = 1.0;

    this.sampleCount = 0;
    this.avgLoss = 0.0;
    this.prevAvgLoss = 0.0;
    this.converged = false;
    this.lastEffectiveLr = 0.0;

    // Normalization buffers
    this.meanX = new Float64Array(inputDim);
    this.m2X = new Float64Array(inputDim);
    this.stdX = new Float64Array(inputDim);
    this.countX = 0;

    this.meanY = new Float64Array(outputDim);
    this.m2Y = new Float64Array(outputDim);
    this.stdY = new Float64Array(outputDim);
    this.countY = 0;

    this.meanR = new Float64Array(outputDim);
    this.m2R = new Float64Array(outputDim);
    this.countR = 0;

    // Precompute positional encoding up to maxSequenceLength
    this.posEnc = new Float64Array(
      this.config.maxSequenceLength * this.config.embeddingDim,
    );
    this.precomputePosEnc();

    // Allocate parameters and buffers
    this.params = [];
    this.allocateParameters();
    this.initializeParameters();

    this.allocateBuffers();

    // last seen input buffers
    this.lastXRaw = new Float64Array(this.config.maxSequenceLength * inputDim);
    this.predictXRaw = new Float64Array(
      this.config.maxSequenceLength * inputDim,
    );
    this.lastSeqLen = 0;
  }

  private allocateParameters(): void {
    const E = this.config.embeddingDim | 0;
    const K = this.config.temporalKernelSize | 0;

    // Helper to register param
    const addParam = (
      name: string,
      size: number,
      l2: boolean,
    ): Float64Array => {
      const w = new Float64Array(size);
      const g = new Float64Array(size);
      const m = new Float64Array(size);
      const v = new Float64Array(size);
      this.params.push({ w, g, m, v, l2, name });
      return w;
    };

    // Input projection
    this.W_in = addParam("W_in", E * this.inputDim, true);
    this.b_in = addParam("b_in", E, false);

    // Temporal conv
    this.W_conv = addParam("W_conv", this.numScales * E * E * K, true);
    this.b_conv = addParam("b_conv", this.numScales * E, false);

    // Scale embeddings
    this.scaleEmb = addParam("scaleEmb", this.numScales * E, true);

    // Fusion gate
    this.W_gate = addParam("W_gate", this.numScales * E, true);
    this.b_gate = addParam("b_gate", this.numScales, false);

    // Transformer blocks
    const B = this.config.numBlocks | 0;
    const H = this.config.numHeads | 0;
    const HD = this.headDim | 0;
    const hidden = this.hiddenDim | 0;

    this.ln1Gamma = new Array(B);
    this.ln1Beta = new Array(B);
    this.ln2Gamma = new Array(B);
    this.ln2Beta = new Array(B);

    this.Wq = new Array(B);
    this.Wk = new Array(B);
    this.Wv = new Array(B);
    this.Wo = new Array(B);
    this.bo = new Array(B);

    this.W1 = new Array(B);
    this.b1 = new Array(B);
    this.W2 = new Array(B);
    this.b2 = new Array(B);

    for (let b = 0; b < B; b++) {
      this.ln1Gamma[b] = addParam(`ln1Gamma_${b}`, E, false);
      this.ln1Beta[b] = addParam(`ln1Beta_${b}`, E, false);
      this.ln2Gamma[b] = addParam(`ln2Gamma_${b}`, E, false);
      this.ln2Beta[b] = addParam(`ln2Beta_${b}`, E, false);

      this.Wq[b] = addParam(`Wq_${b}`, H * HD * E, true);
      this.Wk[b] = addParam(`Wk_${b}`, H * HD * E, true);
      this.Wv[b] = addParam(`Wv_${b}`, H * HD * E, true);
      this.Wo[b] = addParam(`Wo_${b}`, E * E, true);
      this.bo[b] = addParam(`bo_${b}`, E, false);

      this.W1[b] = addParam(`W1_${b}`, hidden * E, true);
      this.b1[b] = addParam(`b1_${b}`, hidden, false);
      this.W2[b] = addParam(`W2_${b}`, E * hidden, true);
      this.b2[b] = addParam(`b2_${b}`, E, false);
    }

    // Pooling
    this.W_pool = addParam("W_pool", E, true);
    this.b_pool = addParam("b_pool", 1, false);

    // Output head
    this.W_out = addParam("W_out", this.outputDim * E, true);
    this.b_out = addParam("b_out", this.outputDim, false);
  }

  private initializeParameters(): void {
    // Xavier/He init as described; biases 0; LayerNorm gamma=1 beta=0; scaleEmb small.
    const E = this.config.embeddingDim | 0;
    const K = this.config.temporalKernelSize | 0;
    const H = this.config.numHeads | 0;
    const HD = this.headDim | 0;
    const hidden = this.hiddenDim | 0;

    // Input projection: Xavier uniform
    this.initXavierUniform(this.W_in, this.inputDim, E);

    for (let i = 0; i < this.b_in.length; i++) this.b_in[i] = 0.0;

    // Conv: Xavier uniform with fanIn=E*K, fanOut=E
    this.initXavierUniform(this.W_conv, E * K, E);
    for (let i = 0; i < this.b_conv.length; i++) this.b_conv[i] = 0.0;

    // Scale embeddings: small uniform
    this.initUniform(this.scaleEmb, 1e-3);

    // Fusion gates: Xavier uniform (fanIn=E, fanOut=1)
    this.initXavierUniform(this.W_gate, E, 1);
    for (let i = 0; i < this.b_gate.length; i++) this.b_gate[i] = 0.0;

    // Blocks
    const B = this.config.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      // LayerNorm
      for (let d = 0; d < E; d++) {
        this.ln1Gamma[b][d] = 1.0;
        this.ln1Beta[b][d] = 0.0;
        this.ln2Gamma[b][d] = 1.0;
        this.ln2Beta[b][d] = 0.0;
      }

      // QKV: Xavier uniform (fanIn=E, fanOut=HD) per head packed
      this.initXavierUniform(this.Wq[b], E, HD);
      this.initXavierUniform(this.Wk[b], E, HD);
      this.initXavierUniform(this.Wv[b], E, HD);

      // Wo: Xavier uniform (fanIn=E, fanOut=E)
      this.initXavierUniform(this.Wo[b], E, E);
      for (let d = 0; d < E; d++) this.bo[b][d] = 0.0;

      // FFN: He uniform (fanIn=E) for W1, He uniform (fanIn=hidden) for W2
      this.initHeUniform(this.W1[b], E);
      for (let i = 0; i < this.b1[b].length; i++) this.b1[b][i] = 0.0;

      this.initHeUniform(this.W2[b], hidden);
      for (let d = 0; d < E; d++) this.b2[b][d] = 0.0;
    }

    // Pooling and output
    this.initXavierUniform(this.W_pool, E, 1);
    this.b_pool[0] = 0.0;

    this.initXavierUniform(this.W_out, E, this.outputDim);
    for (let i = 0; i < this.b_out.length; i++) this.b_out[i] = 0.0;

    // Ensure gradients/moments are zero
    for (let p = 0; p < this.params.length; p++) {
      this.params[p].g.fill(0.0);
      this.params[p].m.fill(0.0);
      this.params[p].v.fill(0.0);
    }
  }

  private allocateBuffers(): void {
    const maxL = this.config.maxSequenceLength | 0;
    const E = this.config.embeddingDim | 0;
    const hidden = this.hiddenDim | 0;

    this.xRawBuf = new Float64Array(maxL * this.inputDim);
    this.xNormBuf = new Float64Array(maxL * this.inputDim);

    this.yTargetNormBuf = new Float64Array(this.outputDim);
    this.yPredNormBuf = new Float64Array(this.outputDim);
    this.residualBuf = new Float64Array(this.outputDim);
    this.dYpredBuf = new Float64Array(this.outputDim);

    this.E0 = new Float64Array(maxL * E);
    this.dE0 = new Float64Array(maxL * E);

    // Per-scale buffers
    this.convPre = new Array(this.numScales);
    this.convAct = new Array(this.numScales);
    this.scaleE = new Array(this.numScales);
    this.dScaleE = new Array(this.numScales);
    this.dConvPre = new Array(this.numScales);

    for (let s = 0; s < this.numScales; s++) {
      const scale = this.config.temporalScales[s] | 0;
      const LsMax = ((maxL + scale - 1) / scale) | 0; // ceil
      const len = LsMax * E;
      this.convPre[s] = new Float64Array(len);
      this.convAct[s] = new Float64Array(len);
      this.scaleE[s] = new Float64Array(len);
      this.dScaleE[s] = new Float64Array(len);
      this.dConvPre[s] = new Float64Array(len);
    }

    this.gates = new Float64Array(maxL * this.numScales);

    this.fused = new Float64Array(maxL * E);
    this.dFused = new Float64Array(maxL * E);

    // Transformer caches
    const B = this.config.numBlocks | 0;
    this.blockInput = new Array(B);
    this.blockAfterAttn = new Array(B);

    this.ln1OutBuf = new Array(B);
    this.ln2OutBuf = new Array(B);
    this.ln1Mean = new Array(B);
    this.ln1InvStd = new Array(B);
    this.ln2Mean = new Array(B);
    this.ln2InvStd = new Array(B);

    this.Qbuf = new Array(B);
    this.Kbuf = new Array(B);
    this.Vbuf = new Array(B);
    this.headConcat = new Array(B);

    this.ffnPreBuf = new Array(B);
    this.ffnActBuf = new Array(B);

    for (let b = 0; b < B; b++) {
      this.blockInput[b] = new Float64Array(maxL * E);
      this.blockAfterAttn[b] = new Float64Array(maxL * E);

      this.ln1OutBuf[b] = new Float64Array(maxL * E);
      this.ln2OutBuf[b] = new Float64Array(maxL * E);

      this.ln1Mean[b] = new Float64Array(maxL);
      this.ln1InvStd[b] = new Float64Array(maxL);
      this.ln2Mean[b] = new Float64Array(maxL);
      this.ln2InvStd[b] = new Float64Array(maxL);

      this.Qbuf[b] = new Float64Array(maxL * E);
      this.Kbuf[b] = new Float64Array(maxL * E);
      this.Vbuf[b] = new Float64Array(maxL * E);
      this.headConcat[b] = new Float64Array(maxL * E);

      this.ffnPreBuf[b] = new Float64Array(maxL * hidden);
      this.ffnActBuf[b] = new Float64Array(maxL * hidden);
    }

    // Pooling caches
    this.poolLogits = new Float64Array(maxL);
    this.poolWeights = new Float64Array(maxL);
    this.dPoolTmp = new Float64Array(maxL);

    this.agg = new Float64Array(E);
    this.dAgg = new Float64Array(E);

    // temp buffers
    this.tmpEmbA = new Float64Array(maxL * E);
    this.tmpEmbB = new Float64Array(maxL * E);
    this.tmpEmbC = new Float64Array(maxL * E);
    this.tmpHidden = new Float64Array(maxL * hidden);

    // attention head temps
    this.tmpHeadQ = new Float64Array(maxL * this.headDim);
    this.tmpHeadK = new Float64Array(maxL * this.headDim);
    this.tmpHeadV = new Float64Array(maxL * this.headDim);
    this.tmpWeight = new Float64Array(maxL);
    this.tmpScore = new Float64Array(maxL);
    this.tmpDWeight = new Float64Array(maxL);
  }

  private precomputePosEnc(): void {
    // Sin/cos positional encoding:
    // PE[p,2i] = sin(p / 10000^(2i/E))
    // PE[p,2i+1] = cos(p / 10000^(2i/E))
    const maxL = this.config.maxSequenceLength | 0;
    const E = this.config.embeddingDim | 0;

    // Precompute div terms for even dimensions
    const half = (E / 2) | 0;
    // Avoid allocations: compute per i and fill per position
    for (let i = 0; i < half; i++) {
      const exponent = (2.0 * i) / E;
      const denom = Math.pow(10000.0, exponent);
      const d0 = (2 * i) | 0;
      const d1 = d0 + 1;

      for (let p = 0; p < maxL; p++) {
        const angle = p / denom;
        const off = p * E + d0;
        this.posEnc[off] = Math.sin(angle);
        if (d1 < E) this.posEnc[p * E + d1] = Math.cos(angle);
      }
    }

    // If E is odd, last dim won't be filled by above; set to 0
    if ((E & 1) === 1) {
      const dLast = E - 1;
      for (let p = 0; p < maxL; p++) {
        this.posEnc[p * E + dLast] = 0.0;
      }
    }
  }

  // ===================== normalization (Welford) =====================

  private updateInputStatsFromRaw(seqLen: number): void {
    // Welford per feature, count increments per timestep vector.
    const inputDim = this.inputDim | 0;
    for (let t = 0; t < seqLen; t++) {
      this.countX++;
      const n = this.countX;
      const base = t * inputDim;
      for (let c = 0; c < inputDim; c++) {
        const v = this.xRawBuf[base + c];
        const mean = this.meanX[c];
        const delta = v - mean;
        const meanNew = mean + delta / n;
        this.meanX[c] = meanNew;
        const delta2 = v - meanNew;
        this.m2X[c] += delta * delta2;
      }
    }

    if (this.countX > 1) {
      const denom = this.countX - 1;
      for (let c = 0; c < inputDim; c++) {
        const var_ = this.m2X[c] / denom;
        this.stdX[c] = Math.sqrt(var_ > 0 ? var_ : 0.0);
      }
    } else {
      for (let c = 0; c < inputDim; c++) this.stdX[c] = 0.0;
    }
  }

  private updateOutputStatsFromTarget(yTargetArr: number[]): void {
    this.countY++;
    const n = this.countY;

    for (let k = 0; k < this.outputDim; k++) {
      const v = yTargetArr[k];
      const mean = this.meanY[k];
      const delta = v - mean;
      const meanNew = mean + delta / n;
      this.meanY[k] = meanNew;
      const delta2 = v - meanNew;
      this.m2Y[k] += delta * delta2;
    }

    if (this.countY > 1) {
      const denom = this.countY - 1;
      for (let k = 0; k < this.outputDim; k++) {
        const var_ = this.m2Y[k] / denom;
        this.stdY[k] = Math.sqrt(var_ > 0 ? var_ : 0.0);
      }
    } else {
      for (let k = 0; k < this.outputDim; k++) this.stdY[k] = 0.0;
    }
  }

  private normalizeX(seqLen: number): void {
    const eps = this.config.epsilon;
    const inputDim = this.inputDim | 0;
    for (let t = 0; t < seqLen; t++) {
      const base = t * inputDim;
      for (let c = 0; c < inputDim; c++) {
        const denom = clampStdDenom(this.stdX[c], eps);
        this.xNormBuf[base + c] = (this.xRawBuf[base + c] - this.meanX[c]) /
          denom;
      }
    }
  }

  private normalizeYTarget(yTargetArr: number[]): void {
    const eps = this.config.epsilon;
    for (let k = 0; k < this.outputDim; k++) {
      const denom = clampStdDenom(this.stdY[k], eps);
      this.yTargetNormBuf[k] = (yTargetArr[k] - this.meanY[k]) / denom;
    }
  }

  private updateResidualStats(): void {
    // Welford on residuals in normalized space: r = yTargetNorm - yPredNorm
    this.countR++;
    const n = this.countR;

    for (let k = 0; k < this.outputDim; k++) {
      const v = this.residualBuf[k]; // already computed
      const mean = this.meanR[k];
      const delta = v - mean;
      const meanNew = mean + delta / n;
      this.meanR[k] = meanNew;
      const delta2 = v - meanNew;
      this.m2R[k] += delta * delta2;
    }
  }

  // ===================== forward/backward =====================

  private forwardTrain(
    seqLen: number,
  ): {
    mse: number;
    loss: number;
    isOutlier: boolean;
    sampleWeight: number;
    rNorm: number;
  } {
    const E = this.config.embeddingDim | 0;

    // (4) Input projection: E0[t,d] = sum_c Xnorm[t,c]*W_in[d,c] + b_in[d]
    this.forwardInputProjection(seqLen);

    // (5) Multi-scale temporal conv per scale
    this.forwardTemporalConvs(seqLen);

    // (6) Fusion to seqLen timeline (softmax gates across scales)
    this.forwardFusion(seqLen);

    // (7) Transformer blocks
    this.forwardTransformer(seqLen);

    // (8) Attention pooling
    this.forwardPooling(seqLen);

    // (9) Output head (normalized)
    for (let k = 0; k < this.outputDim; k++) {
      let sum = this.b_out[k];
      const wOff = k * E;
      for (let d = 0; d < E; d++) sum += this.agg[d] * this.W_out[wOff + d];
      this.yPredNormBuf[k] = sum;
    }

    // residuals in normalized space, mse
    let mse = 0.0;
    let r2sum = 0.0;
    for (let k = 0; k < this.outputDim; k++) {
      const r = this.yTargetNormBuf[k] - this.yPredNormBuf[k];
      this.residualBuf[k] = r;
      const r2 = r * r;
      r2sum += r2;
      mse += r2;
    }
    mse /= this.outputDim;

    const rNorm = Math.sqrt(r2sum);

    // (11) Outlier downweighting
    let isOutlier = false;
    let sampleWeight = 1.0;
    if (rNorm > this.config.outlierThreshold) {
      isOutlier = true;
      sampleWeight = 0.1;
    }

    // (10) L2 regularization (loss only, unweighted)
    const l2 = this.computeL2Loss();
    const loss = sampleWeight * mse + l2;

    return { mse, loss, isOutlier, sampleWeight, rNorm };
  }

  private forwardInferFromRaw(
    seqLen: number,
    xRaw: Float64Array,
  ): Float64Array {
    // Copy raw into xRawBuf then normalize using current stats; do not update stats.
    const n = seqLen * this.inputDim;
    this.copySlice(xRaw, this.xRawBuf, n);

    // normalize inputs into xNormBuf
    this.normalizeX(seqLen);

    // forward deterministic: inputProj -> conv -> fusion -> transformer -> pooling -> output -> denorm
    this.forwardInputProjection(seqLen);
    this.forwardTemporalConvs(seqLen);
    this.forwardFusion(seqLen);
    this.forwardTransformer(seqLen);
    this.forwardPooling(seqLen);

    const E = this.config.embeddingDim | 0;

    // output in normalized space
    for (let k = 0; k < this.outputDim; k++) {
      let sum = this.b_out[k];
      const wOff = k * E;
      for (let d = 0; d < E; d++) sum += this.agg[d] * this.W_out[wOff + d];
      this.yPredNormBuf[k] = sum;
    }

    // denormalize into residualBuf temporarily (reuse)
    const eps = this.config.epsilon;
    for (let k = 0; k < this.outputDim; k++) {
      const scale = clampStdDenom(this.stdY[k], eps);
      this.residualBuf[k] = this.yPredNormBuf[k] * scale + this.meanY[k];
    }
    // return a view-like typed array (but we must not expose internal buffers for mutation).
    // We'll return residualBuf, which is internal; caller predict() copies to JS arrays anyway.
    return this.residualBuf;
  }

  private backward(seqLen: number, sampleWeight: number): void {
    const E = this.config.embeddingDim | 0;
    const B = this.config.numBlocks | 0;

    // ===== output head backward =====
    // mse = (1/D) sum_k r^2 where r = yTarget - yPred
    // d/dyPred: (2/D) (yPred - yTarget) = (-2/D)*r
    const scale = (-2.0 / this.outputDim) * sampleWeight;

    this.dAgg.fill(0.0);

    for (let k = 0; k < this.outputDim; k++) {
      const dYpred = scale * this.residualBuf[k];
      this.dYpredBuf[k] = dYpred;

      // gradients W_out, b_out
      const wOff = k * E;
      const gW = this.getGradByWeight(this.W_out);
      const gb = this.getGradByWeight(this.b_out);

      gb[k] += dYpred;
      for (let d = 0; d < E; d++) {
        gW[wOff + d] += dYpred * this.agg[d];
        this.dAgg[d] += dYpred * this.W_out[wOff + d];
      }
    }

    // ===== pooling backward =====
    this.dFused.fill(0.0);
    this.backwardPooling(seqLen);

    // ===== transformer blocks backward (reverse) =====
    for (let b = B - 1; b >= 0; b--) {
      this.backwardBlock(seqLen, b);
    }

    // ===== fusion backward -> temporal conv -> input projection =====
    this.backwardFusion(seqLen);
    this.backwardTemporalConvs(seqLen);
    this.backwardInputProjection(seqLen);
  }

  // ---------- forward pieces ----------

  private forwardInputProjection(seqLen: number): void {
    const E = this.config.embeddingDim | 0;
    const inputDim = this.inputDim | 0;

    // E0[t,d] = b_in[d] + sum_c xNorm[t,c] * W_in[d,c]
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * inputDim;
      const eOff = t * E;
      for (let d = 0; d < E; d++) {
        let sum = this.b_in[d];
        const wOff = d * inputDim;
        for (let c = 0; c < inputDim; c++) {
          sum += this.xNormBuf[xOff + c] * this.W_in[wOff + c];
        }
        this.E0[eOff + d] = sum;
      }
    }
  }

  private forwardTemporalConvs(seqLen: number): void {
    const E = this.config.embeddingDim | 0;
    const K = this.config.temporalKernelSize | 0;
    const pad = K - 1; // causal padding (lookback)
    const maxPos = this.config.maxSequenceLength | 0;

    for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
      const scale = this.config.temporalScales[sIdx] | 0;
      const Ls = ((seqLen + scale - 1) / scale) | 0; // ceil

      const convPre = this.convPre[sIdx];
      const convAct = this.convAct[sIdx];
      const scaleE = this.scaleE[sIdx];

      // Conv_s[u,d] = b_conv + sum_{k,d2} E0[idx, d2] * W_conv[s,d,d2,k]
      for (let u = 0; u < Ls; u++) {
        let center = u * scale;
        if (center >= seqLen) center = seqLen - 1;

        const outOff = u * E;

        for (let dOut = 0; dOut < E; dOut++) {
          let sum = this.b_conv[sIdx * E + dOut];

          const baseW = (((sIdx * E + dOut) * E) * K) | 0;

          for (let k = 0; k < K; k++) {
            const idx = center - pad + k;
            if (idx < 0 || idx >= seqLen) continue;

            const inOff = idx * E;
            const wOffK = baseW + k;
            // weights are laid out ((dOut*E + dIn)*K + k)
            // so for each dIn, index = baseW + (dIn*K + k)
            for (let dIn = 0; dIn < E; dIn++) {
              const wIdx = wOffK + dIn * K;
              sum += this.E0[inOff + dIn] * this.W_conv[wIdx];
            }
          }

          convPre[outOff + dOut] = sum;
          const a = gelu(sum);
          convAct[outOff + dOut] = a;

          // E_s = F_s + PE + scaleEmb
          // PE uses u index (length Ls); for simplicity we use same PE table by u
          const peOff = (u < maxPos ? u : (maxPos - 1)) * E;
          scaleE[outOff + dOut] = a + this.posEnc[peOff + dOut] +
            this.scaleEmb[sIdx * E + dOut];
        }
      }

      // For leftover max prealloc positions (if any), do not clear; we always use Ls only.
    }
  }

  private forwardFusion(seqLen: number): void {
    const E = this.config.embeddingDim | 0;

    // gates[t,s] = softmax_s( dot(E_s_aligned[t], W_gate[s]) + b_gate[s] )
    // fused[t,d] = sum_s gates[t,s] * E_s_aligned[t,d]
    //
    // Align by nearest: u = floor(t/scale)
    for (let t = 0; t < seqLen; t++) {
      // compute logits per scale
      let maxLogit = -1e300;
      const logitOff = t * this.numScales;

      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        const scale = this.config.temporalScales[sIdx] | 0;
        const u = (t / scale) | 0;
        const eOff = u * E;

        let logit = this.b_gate[sIdx];
        const wOff = sIdx * E;
        const sE = this.scaleE[sIdx];

        for (let d = 0; d < E; d++) {
          logit += sE[eOff + d] * this.W_gate[wOff + d];
        }

        this.gates[logitOff + sIdx] = logit; // temporarily store logits
        if (logit > maxLogit) maxLogit = logit;
      }

      // softmax
      let sumExp = 0.0;
      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        const z = this.gates[logitOff + sIdx] - maxLogit;
        const e = Math.exp(z);
        this.gates[logitOff + sIdx] = e;
        sumExp += e;
      }
      if (sumExp < 1e-300) sumExp = 1e-300;
      const inv = 1.0 / sumExp;
      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        this.gates[logitOff + sIdx] *= inv; // now stores gate weights
      }

      // fused[t,d]
      const fOff = t * E;
      for (let d = 0; d < E; d++) this.fused[fOff + d] = 0.0;

      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        const g = this.gates[logitOff + sIdx];
        const scale = this.config.temporalScales[sIdx] | 0;
        const u = (t / scale) | 0;
        const eOff = u * E;
        const sE = this.scaleE[sIdx];

        for (let d = 0; d < E; d++) {
          this.fused[fOff + d] += g * sE[eOff + d];
        }
      }
    }

    // Optional fusion dropout (default 0); implemented as inverted-dropout
    const p = this.config.fusionDropout;
    if (p > 0) {
      const invKeep = 1.0 / (1.0 - p);
      const n = seqLen * E;
      for (let i = 0; i < n; i++) {
        if (this.rand() < p) this.fused[i] = 0.0;
        else this.fused[i] *= invKeep;
      }
    }
  }

  private forwardTransformer(seqLen: number): void {
    const B = this.config.numBlocks | 0;
    for (let b = 0; b < B; b++) {
      this.forwardBlock(seqLen, b);
    }
  }

  private forwardBlock(seqLen: number, b: number): void {
    const E = this.config.embeddingDim | 0;

    // cache block input
    this.copySlice(this.fused, this.blockInput[b], seqLen * E);

    // LN1
    this.layerNormForwardSeq(
      this.fused,
      this.ln1OutBuf[b],
      this.ln1Gamma[b],
      this.ln1Beta[b],
      this.ln1Mean[b],
      this.ln1InvStd[b],
      seqLen,
      E,
    );

    // Q,K,V
    this.projectQKV(seqLen, b);

    // MHA forward -> headConcat[b]
    this.attentionForward(seqLen, b);

    // Output projection + residual: fused = x + headConcat*Wo + bo
    const Wo = this.Wo[b];
    const bo = this.bo[b];
    const head = this.headConcat[b];
    const xIn = this.blockInput[b];
    for (let t = 0; t < seqLen; t++) {
      const off = t * E;
      for (let dOut = 0; dOut < E; dOut++) {
        let sum = bo[dOut];
        const wOff = dOut * E;
        for (let dIn = 0; dIn < E; dIn++) {
          sum += head[off + dIn] * Wo[wOff + dIn];
        }
        const v = xIn[off + dOut] + sum;
        this.fused[off + dOut] = v;
        this.blockAfterAttn[b][off + dOut] = v;
      }
    }

    // LN2
    this.layerNormForwardSeq(
      this.fused,
      this.ln2OutBuf[b],
      this.ln2Gamma[b],
      this.ln2Beta[b],
      this.ln2Mean[b],
      this.ln2InvStd[b],
      seqLen,
      E,
    );

    // FFN forward:
    // h = gelu(LN2 * W1^T + b1)
    // out = h * W2^T + b2
    this.ffnForward(seqLen, b);

    // residual: fused = blockAfterAttn + ffnOut (already computed in ffnForward)
  }

  private projectQKV(seqLen: number, b: number): void {
    const E = this.config.embeddingDim | 0;
    const H = this.config.numHeads | 0;
    const HD = this.headDim | 0;

    const x = this.ln1OutBuf[b];
    const Q = this.Qbuf[b];
    const K = this.Kbuf[b];
    const V = this.Vbuf[b];

    const Wq = this.Wq[b];
    const Wk = this.Wk[b];
    const Wv = this.Wv[b];

    for (let t = 0; t < seqLen; t++) {
      const xOff = t * E;
      const qOff = t * E;

      // For each head and each headDim
      for (let h = 0; h < H; h++) {
        const headBase = h * HD;
        const wHeadOff = (h * HD * E) | 0;

        for (let hd = 0; hd < HD; hd++) {
          const outIdx = qOff + headBase + hd;

          let sumQ = 0.0;
          let sumK = 0.0;
          let sumV = 0.0;

          const wRowOff = (wHeadOff + hd * E) | 0;
          for (let d = 0; d < E; d++) {
            const xv = x[xOff + d];
            sumQ += xv * Wq[wRowOff + d];
            sumK += xv * Wk[wRowOff + d];
            sumV += xv * Wv[wRowOff + d];
          }

          Q[outIdx] = sumQ;
          K[outIdx] = sumK;
          V[outIdx] = sumV;
        }
      }
    }
  }

  private attentionForward(seqLen: number, b: number): void {
    const E = this.config.embeddingDim | 0;
    const H = this.config.numHeads | 0;
    const HD = this.headDim | 0;
    const attnDrop = this.config.attentionDropout;

    const Q = this.Qbuf[b];
    const K = this.Kbuf[b];
    const V = this.Vbuf[b];
    const out = this.headConcat[b];

    const scale = 1.0 / Math.sqrt(HD);

    for (let t = 0; t < seqLen * E; t++) out[t] = 0.0;

    // Per head
    for (let h = 0; h < H; h++) {
      const headBase = h * HD;

      // For each query position i
      for (let i = 0; i < seqLen; i++) {
        const qiOff = i * E + headBase;

        // scores for j<=i
        let maxScore = -1e300;
        for (let j = 0; j <= i; j++) {
          const kjOff = j * E + headBase;
          let dot = 0.0;
          for (let hd = 0; hd < HD; hd++) dot += Q[qiOff + hd] * K[kjOff + hd];
          const s = dot * scale;
          this.tmpScore[j] = s;
          if (s > maxScore) maxScore = s;
        }

        let sumExp = 0.0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(this.tmpScore[j] - maxScore);
          this.tmpWeight[j] = e;
          sumExp += e;
        }
        if (sumExp < 1e-300) sumExp = 1e-300;
        const inv = 1.0 / sumExp;

        // attention dropout on weights (inverted)
        if (attnDrop > 0) {
          const invKeep = 1.0 / (1.0 - attnDrop);
          for (let j = 0; j <= i; j++) {
            let w = this.tmpWeight[j] * inv;
            if (this.rand() < attnDrop) w = 0.0;
            else w *= invKeep;
            this.tmpWeight[j] = w;
          }
        } else {
          for (let j = 0; j <= i; j++) this.tmpWeight[j] *= inv;
        }

        // head output
        const outOff = i * E + headBase;
        for (let hd = 0; hd < HD; hd++) {
          let sum = 0.0;
          for (let j = 0; j <= i; j++) {
            const vOff = j * E + headBase;
            sum += this.tmpWeight[j] * V[vOff + hd];
          }
          out[outOff + hd] = sum;
        }
      }
    }
  }

  private ffnForward(seqLen: number, b: number): void {
    const E = this.config.embeddingDim | 0;
    const hidden = this.hiddenDim | 0;

    const x = this.ln2OutBuf[b];
    const W1 = this.W1[b];
    const b1 = this.b1[b];
    const W2 = this.W2[b];
    const b2 = this.b2[b];

    const pre = this.ffnPreBuf[b];
    const act = this.ffnActBuf[b];

    // pre[t,h] = b1[h] + sum_d x[t,d]*W1[h,d]
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * E;
      const hOff = t * hidden;
      for (let h = 0; h < hidden; h++) {
        let sum = b1[h];
        const wOff = h * E;
        for (let d = 0; d < E; d++) sum += x[xOff + d] * W1[wOff + d];
        pre[hOff + h] = sum;
        act[hOff + h] = gelu(sum);
      }
    }

    // out[t,d] = b2[d] + sum_h act[t,h]*W2[d,h]
    // fused = blockAfterAttn + out
    const xRes = this.blockAfterAttn[b];
    for (let t = 0; t < seqLen; t++) {
      const outOff = t * E;
      const hOff = t * hidden;
      for (let d = 0; d < E; d++) {
        let sum = b2[d];
        const wOff = d * hidden;
        for (let h = 0; h < hidden; h++) sum += act[hOff + h] * W2[wOff + h];
        this.fused[outOff + d] = xRes[outOff + d] + sum;
      }
    }
  }

  private forwardPooling(seqLen: number): void {
    const E = this.config.embeddingDim | 0;

    // logits[t] = dot(fused[t], W_pool) + b_pool
    let maxLogit = -1e300;
    for (let t = 0; t < seqLen; t++) {
      const off = t * E;
      let z = this.b_pool[0];
      for (let d = 0; d < E; d++) z += this.fused[off + d] * this.W_pool[d];
      this.poolLogits[t] = z;
      if (z > maxLogit) maxLogit = z;
    }

    // softmax over t
    let sumExp = 0.0;
    for (let t = 0; t < seqLen; t++) {
      const e = Math.exp(this.poolLogits[t] - maxLogit);
      this.poolWeights[t] = e;
      sumExp += e;
    }
    if (sumExp < 1e-300) sumExp = 1e-300;
    const inv = 1.0 / sumExp;
    for (let t = 0; t < seqLen; t++) this.poolWeights[t] *= inv;

    // agg[d] = sum_t w[t] * fused[t,d]
    for (let d = 0; d < E; d++) this.agg[d] = 0.0;

    for (let t = 0; t < seqLen; t++) {
      const w = this.poolWeights[t];
      const off = t * E;
      for (let d = 0; d < E; d++) this.agg[d] += w * this.fused[off + d];
    }
  }

  // ---------- backward pieces ----------

  private backwardPooling(seqLen: number): void {
    // agg[d] = sum_t pW[t]*fused[t,d]
    // dFused += pW[t]*dAgg[d] + dLogit[t]*W_pool[d]
    // logits[t] = dot(fused[t], W_pool) + b_pool
    // pW = softmax(logits)
    const E = this.config.embeddingDim | 0;

    // dPW[t] = dot(dAgg, fused[t])
    // s = sum_t dPW[t]*pW[t]
    let s = 0.0;

    for (let t = 0; t < seqLen; t++) {
      const off = t * E;
      let dot = 0.0;
      for (let d = 0; d < E; d++) dot += this.dAgg[d] * this.fused[off + d];
      this.dPoolTmp[t] = dot;
      s += dot * this.poolWeights[t];
    }

    const gWpool = this.getGradByWeight(this.W_pool);
    const gBpool = this.getGradByWeight(this.b_pool);

    for (let t = 0; t < seqLen; t++) {
      const pw = this.poolWeights[t];
      const dLogit = pw * (this.dPoolTmp[t] - s);

      gBpool[0] += dLogit;

      const off = t * E;

      // from logits path
      for (let d = 0; d < E; d++) {
        gWpool[d] += dLogit * this.fused[off + d];
        this.dFused[off + d] += dLogit * this.W_pool[d];
      }

      // from agg direct path
      for (let d = 0; d < E; d++) {
        this.dFused[off + d] += pw * this.dAgg[d];
      }
    }
  }

  private backwardBlock(seqLen: number, b: number): void {
    const E = this.config.embeddingDim | 0;
    const hidden = this.hiddenDim | 0;

    // dFused currently holds dX2 (gradient w.r.t block output)
    // FFN residual: x2 = x1 + ffnOut
    // => dX1 starts as dX2
    // We'll compute dX1_from_LN2 and add into dFused.

    // --- FFN backward ---
    this.tmpHidden.fill(0.0);
    this.tmpEmbA.fill(0.0); // dLn2Out
    this.tmpEmbB.fill(0.0); // dX1_from_LN2

    const dX2 = this.dFused;

    const act = this.ffnActBuf[b];
    const pre = this.ffnPreBuf[b];

    const W2 = this.W2[b];
    const b2 = this.b2[b];
    const W1 = this.W1[b];
    const b1 = this.b1[b];

    const gW2 = this.getGradByWeight(W2);
    const gb2 = this.getGradByWeight(b2);
    const gW1 = this.getGradByWeight(W1);
    const gb1 = this.getGradByWeight(b1);

    // dHiddenAct = dFFNOut * W2 (backprop)
    // dFFNOut is dX2
    for (let t = 0; t < seqLen; t++) {
      const dxOff = t * E;
      const hOff = t * hidden;

      // db2 and dW2 and dHiddenAct
      for (let d = 0; d < E; d++) {
        const g = dX2[dxOff + d];
        gb2[d] += g;
        const wOff = d * hidden;

        for (let h = 0; h < hidden; h++) {
          gW2[wOff + h] += g * act[hOff + h];
          this.tmpHidden[hOff + h] += g * W2[wOff + h];
        }
      }
    }

    // GELU backward: dHiddenPre = dHiddenAct * gelu'(pre)
    for (let t = 0; t < seqLen; t++) {
      const hOff = t * hidden;
      for (let h = 0; h < hidden; h++) {
        const dh = this.tmpHidden[hOff + h];
        const z = pre[hOff + h];
        this.tmpHidden[hOff + h] = dh * geluDeriv(z);
      }
    }

    // Backprop through W1: dLn2Out = dHiddenPre * W1, and dW1/db1
    const xLn2 = this.ln2OutBuf[b];
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * E;
      const hOff = t * hidden;

      for (let h = 0; h < hidden; h++) {
        const g = this.tmpHidden[hOff + h];
        gb1[h] += g;

        const wOff = h * E;
        for (let d = 0; d < E; d++) {
          gW1[wOff + d] += g * xLn2[xOff + d];
          this.tmpEmbA[xOff + d] += g * W1[wOff + d];
        }
      }
    }

    // LN2 backward: input is x1 = blockAfterAttn[b]
    // Upstream is tmpEmbA (dLn2Out)
    this.layerNormBackwardSeq(
      this.blockAfterAttn[b],
      this.tmpEmbA,
      this.ln2Gamma[b],
      this.ln2Mean[b],
      this.ln2InvStd[b],
      this.getGradByWeight(this.ln2Gamma[b]),
      this.getGradByWeight(this.ln2Beta[b]),
      this.tmpEmbB,
      seqLen,
      E,
    );

    // Add LN2 contribution to dX1 (residual already in dFused)
    for (let i = 0; i < seqLen * E; i++) this.dFused[i] += this.tmpEmbB[i];

    // --- Attention backward ---
    // Residual: x1 = x0 + attnOut => dX0_residual = dX1 (current dFused), dAttnOut = dX1
    // We must add attention-path gradient to dX0_residual.
    //
    // attnOut = headConcat * Wo + bo
    this.tmpEmbA.fill(0.0); // dHeadConcat
    this.tmpEmbB.fill(0.0); // dLn1Out
    this.tmpEmbC.fill(0.0); // dX0_from_LN1

    const dAttnOut = this.dFused;
    const head = this.headConcat[b];
    const Wo = this.Wo[b];
    const bo = this.bo[b];

    const gWo = this.getGradByWeight(Wo);
    const gbo = this.getGradByWeight(bo);

    // dHead = dAttnOut * Wo^T
    for (let t = 0; t < seqLen; t++) {
      const off = t * E;

      for (let dOut = 0; dOut < E; dOut++) {
        const g = dAttnOut[off + dOut];
        gbo[dOut] += g;

        const wOff = dOut * E;
        for (let dIn = 0; dIn < E; dIn++) {
          gWo[wOff + dIn] += g * head[off + dIn];
          this.tmpEmbA[off + dIn] += g * Wo[wOff + dIn];
        }
      }
    }

    // Now attention backward per head: produce dLn1Out (tmpEmbB)
    this.attentionBackward(seqLen, b, this.tmpEmbA, this.tmpEmbB);

    // LN1 backward: input is blockInput[b], upstream dLn1Out=tmpEmbB
    this.layerNormBackwardSeq(
      this.blockInput[b],
      this.tmpEmbB,
      this.ln1Gamma[b],
      this.ln1Mean[b],
      this.ln1InvStd[b],
      this.getGradByWeight(this.ln1Gamma[b]),
      this.getGradByWeight(this.ln1Beta[b]),
      this.tmpEmbC,
      seqLen,
      E,
    );

    // Combine: dX0_total = dX0_residual (already in dFused) + dX0_from_LN1
    for (let i = 0; i < seqLen * E; i++) this.dFused[i] += this.tmpEmbC[i];

    // Now dFused holds gradient wrt block input (for previous block)
  }

  private attentionBackward(
    seqLen: number,
    b: number,
    dHeadConcat: Float64Array,
    dLn1Out: Float64Array,
  ): void {
    // Backprop through MHA:
    // - Inputs: Q,K,V from forward (concat heads)
    // - Upstream: dHeadConcat (seqLen*emb)
    // - Outputs:
    //   - accumulate grads for Wq,Wk,Wv
    //   - accumulate dLn1Out (seqLen*emb)
    //
    // Memory strategy:
    // - Recompute attention softmax weights on-the-fly per (head, i)
    // - Use small per-head buffers dQhead, dKhead, dVhead (seqLen*headDim)
    const E = this.config.embeddingDim | 0;
    const H = this.config.numHeads | 0;
    const HD = this.headDim | 0;

    const Q = this.Qbuf[b];
    const K = this.Kbuf[b];
    const V = this.Vbuf[b];
    const x = this.ln1OutBuf[b];

    const Wq = this.Wq[b];
    const Wk = this.Wk[b];
    const Wv = this.Wv[b];

    const gWq = this.getGradByWeight(Wq);
    const gWk = this.getGradByWeight(Wk);
    const gWv = this.getGradByWeight(Wv);

    const scale = 1.0 / Math.sqrt(HD);

    // For each head
    for (let h = 0; h < H; h++) {
      // Clear per-head grads
      const qg = this.tmpHeadQ;
      const kg = this.tmpHeadK;
      const vg = this.tmpHeadV;

      const lenHead = seqLen * HD;
      for (let i = 0; i < lenHead; i++) {
        qg[i] = 0.0;
        kg[i] = 0.0;
        vg[i] = 0.0;
      }

      const headBase = h * HD;
      const wHeadOff = (h * HD * E) | 0;

      // For each query i
      for (let i = 0; i < seqLen; i++) {
        const qiOffE = i * E + headBase;
        const qiOffH = i * HD;

        // Compute scores[j] and weights[j] for j<=i
        let maxScore = -1e300;
        for (let j = 0; j <= i; j++) {
          const kjOffE = j * E + headBase;
          let dot = 0.0;
          for (let hd = 0; hd < HD; hd++) {
            dot += Q[qiOffE + hd] * K[kjOffE + hd];
          }
          const s = dot * scale;
          this.tmpScore[j] = s;
          if (s > maxScore) maxScore = s;
        }

        let sumExp = 0.0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(this.tmpScore[j] - maxScore);
          this.tmpWeight[j] = e;
          sumExp += e;
        }
        if (sumExp < 1e-300) sumExp = 1e-300;
        const inv = 1.0 / sumExp;
        for (let j = 0; j <= i; j++) this.tmpWeight[j] *= inv;

        // dWeights[j] = dot(dHeadOut[i], V[j]) ; also accumulate dV via weights
        let sumDW = 0.0;
        for (let j = 0; j <= i; j++) {
          const vjOffE = j * E + headBase;
          let dw = 0.0;

          // dot(dHeadOut_i, V_j)
          const dHeadOffE = i * E + headBase;
          for (let hd = 0; hd < HD; hd++) {
            dw += dHeadConcat[dHeadOffE + hd] * V[vjOffE + hd];
          }
          this.tmpDWeight[j] = dw;
          sumDW += dw * this.tmpWeight[j];

          // dV[j] += weight[j] * dHeadOut[i]
          const wj = this.tmpWeight[j];
          const vjOffH = j * HD;
          for (let hd = 0; hd < HD; hd++) {
            vg[vjOffH + hd] += wj * dHeadConcat[dHeadOffE + hd];
          }
        }

        // dScore[j] = w[j]*(dW[j] - sumDW)
        // accumulate dQ[i] and dK[j]
        for (let j = 0; j <= i; j++) {
          const wj = this.tmpWeight[j];
          const dScore = wj * (this.tmpDWeight[j] - sumDW);

          const kjOffE = j * E + headBase;
          const kjOffH = j * HD;

          for (let hd = 0; hd < HD; hd++) {
            qg[qiOffH + hd] += scale * dScore * K[kjOffE + hd];
            kg[kjOffH + hd] += scale * dScore * Q[qiOffE + hd];
          }
        }
      }

      // Now backprop through linear projections:
      // Q = x * Wq^T => dWq += dQ^T * x, dX += dQ * Wq
      // similarly K,V.
      for (let t = 0; t < seqLen; t++) {
        const xOff = t * E;
        const qOffH = t * HD;
        const qOffE = t * E + headBase;

        for (let hd = 0; hd < HD; hd++) {
          const dQ = qg[qOffH + hd];
          const dK = kg[qOffH + hd];
          const dV = vg[qOffH + hd];

          const wRowOff = (wHeadOff + hd * E) | 0;

          for (let d = 0; d < E; d++) {
            const xv = x[xOff + d];

            // dW
            gWq[wRowOff + d] += dQ * xv;
            gWk[wRowOff + d] += dK * xv;
            gWv[wRowOff + d] += dV * xv;

            // dX accumulation into dLn1Out at embedding indices
            dLn1Out[xOff + d] += dQ * Wq[wRowOff + d];
            dLn1Out[xOff + d] += dK * Wk[wRowOff + d];
            dLn1Out[xOff + d] += dV * Wv[wRowOff + d];
          }

          // also keep consistent indexing usage: qOffE unused here, fine
        }
      }
    }
  }

  private backwardFusion(seqLen: number): void {
    const E = this.config.embeddingDim | 0;

    // Zero dScaleE
    for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
      this.dScaleE[sIdx].fill(0.0);
    }

    const gWgate = this.getGradByWeight(this.W_gate);
    const gbGate = this.getGradByWeight(this.b_gate);

    // dFused[t,d] -> dGate and dScaleE_aligned
    for (let t = 0; t < seqLen; t++) {
      const gOff = t * this.numScales;
      const fOff = t * E;

      // dGate[s] = dot(dFused[t], E_s_aligned[t])
      // sumDG = sum_s dGate[s] * gate[s] (softmax helper)
      let sumDG = 0.0;

      // First pass: compute dGate and sumDG; store dGate in tmpScore[s]
      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        const gate = this.gates[gOff + sIdx];
        const scale = this.config.temporalScales[sIdx] | 0;
        const u = (t / scale) | 0;
        const eOff = u * E;
        const sE = this.scaleE[sIdx];

        let dot = 0.0;
        for (let d = 0; d < E; d++) dot += this.dFused[fOff + d] * sE[eOff + d];
        this.tmpScore[sIdx] = dot;
        sumDG += dot * gate;
      }

      // Second pass: compute dLogit = gate*(dGate - sumDG); accumulate
      for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
        const gate = this.gates[gOff + sIdx];
        const dGate = this.tmpScore[sIdx];
        const dLogit = gate * (dGate - sumDG);

        const scale = this.config.temporalScales[sIdx] | 0;
        const u = (t / scale) | 0;
        const eOff = u * E;
        const sE = this.scaleE[sIdx];
        const dSE = this.dScaleE[sIdx];

        const wOff = sIdx * E;

        gbGate[sIdx] += dLogit;

        for (let d = 0; d < E; d++) {
          // dW_gate += dLogit * E_s
          gWgate[wOff + d] += dLogit * sE[eOff + d];

          // dE_s_aligned += gate*dFused + dLogit*W_gate
          dSE[eOff + d] += gate * this.dFused[fOff + d] +
            dLogit * this.W_gate[wOff + d];
        }
      }
    }

    // scale embeddings gradient: ScaleEmb added at each u
    const gScaleEmb = this.getGradByWeight(this.scaleEmb);
    for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
      const scale = this.config.temporalScales[sIdx] | 0;
      const Ls = ((seqLen + scale - 1) / scale) | 0;

      const dSE = this.dScaleE[sIdx];

      const off = sIdx * E;
      for (let d = 0; d < E; d++) {
        let sum = 0.0;
        for (let u = 0; u < Ls; u++) sum += dSE[u * E + d];
        gScaleEmb[off + d] += sum;
      }
    }
  }

  private backwardTemporalConvs(seqLen: number): void {
    // dScaleE -> dConvPre -> dE0 + grads W_conv, b_conv
    const E = this.config.embeddingDim | 0;
    const K = this.config.temporalKernelSize | 0;
    const pad = K - 1;

    // clear dE0
    this.dE0.fill(0.0);

    const gWconv = this.getGradByWeight(this.W_conv);
    const gbConv = this.getGradByWeight(this.b_conv);

    for (let sIdx = 0; sIdx < this.numScales; sIdx++) {
      const scale = this.config.temporalScales[sIdx] | 0;
      const Ls = ((seqLen + scale - 1) / scale) | 0;

      const dSE = this.dScaleE[sIdx];
      const convPre = this.convPre[sIdx];
      const dConvPre = this.dConvPre[sIdx];

      // dConvAct = dScaleE (PE constant), then gelu back to dConvPre
      const len = Ls * E;
      for (let i = 0; i < len; i++) {
        dConvPre[i] = dSE[i] * geluDeriv(convPre[i]);
      }

      // conv backward
      for (let u = 0; u < Ls; u++) {
        let center = u * scale;
        if (center >= seqLen) center = seqLen - 1;

        const outOff = u * E;

        for (let dOut = 0; dOut < E; dOut++) {
          const g = dConvPre[outOff + dOut];
          gbConv[sIdx * E + dOut] += g;

          const baseW = (((sIdx * E + dOut) * E) * K) | 0;

          for (let k = 0; k < K; k++) {
            const idx = center - pad + k;
            if (idx < 0 || idx >= seqLen) continue;

            const inOff = idx * E;
            const wOffK = baseW + k;

            for (let dIn = 0; dIn < E; dIn++) {
              const wIdx = wOffK + dIn * K;
              gWconv[wIdx] += g * this.E0[inOff + dIn];
              this.dE0[inOff + dIn] += g * this.W_conv[wIdx];
            }
          }
        }
      }
    }
  }

  private backwardInputProjection(seqLen: number): void {
    const E = this.config.embeddingDim | 0;
    const inputDim = this.inputDim | 0;

    const gWin = this.getGradByWeight(this.W_in);
    const gbin = this.getGradByWeight(this.b_in);

    // E0[t,d] = sum_c xNorm[t,c]*W_in[d,c] + b_in[d]
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * inputDim;
      const eOff = t * E;

      for (let d = 0; d < E; d++) {
        const g = this.dE0[eOff + d];
        gbin[d] += g;

        const wOff = d * inputDim;
        for (let c = 0; c < inputDim; c++) {
          gWin[wOff + c] += g * this.xNormBuf[xOff + c];
        }
      }
    }
  }

  // ===================== LayerNorm forward/backward =====================

  private layerNormForwardSeq(
    x: Float64Array,
    out: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
    seqLen: number,
    dim: number,
  ): void {
    const eps = this.config.epsilon;

    for (let t = 0; t < seqLen; t++) {
      const off = t * dim;

      let mean = 0.0;
      for (let d = 0; d < dim; d++) mean += x[off + d];
      mean /= dim;

      let var_ = 0.0;
      for (let d = 0; d < dim; d++) {
        const z = x[off + d] - mean;
        var_ += z * z;
      }
      var_ /= dim;

      const invStd = 1.0 / Math.sqrt(var_ + eps);

      meanBuf[t] = mean;
      invStdBuf[t] = invStd;

      for (let d = 0; d < dim; d++) {
        const xhat = (x[off + d] - mean) * invStd;
        out[off + d] = xhat * gamma[d] + beta[d];
      }
    }
  }

  private layerNormBackwardSeq(
    x: Float64Array,
    dOut: Float64Array,
    gamma: Float64Array,
    meanBuf: Float64Array,
    invStdBuf: Float64Array,
    dGammaAcc: Float64Array,
    dBetaAcc: Float64Array,
    dX: Float64Array,
    seqLen: number,
    dim: number,
  ): void {
    // Standard LN backward:
    // xhat = (x-mean)*invStd
    // y = gamma*xhat + beta
    // dBeta += sum(dy)
    // dGamma += sum(dy*xhat)
    // dx = (1/N)*invStd*(N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    //
    // where dxhat = dy*gamma
    for (let t = 0; t < seqLen; t++) {
      const off = t * dim;
      const mean = meanBuf[t];
      const invStd = invStdBuf[t];

      // accumulate dBeta and dGamma
      let sumDxhat = 0.0;
      let sumDxhatXhat = 0.0;

      // first pass compute xhat and dxhat sums
      for (let d = 0; d < dim; d++) {
        const xhat = (x[off + d] - mean) * invStd;
        const dy = dOut[off + d];
        dBetaAcc[d] += dy;
        dGammaAcc[d] += dy * xhat;

        const dxhat = dy * gamma[d];
        sumDxhat += dxhat;
        sumDxhatXhat += dxhat * xhat;

        // store dxhat temporarily in dX to avoid another buffer
        dX[off + d] = dxhat;
      }

      const invN = 1.0 / dim;

      // second pass: dx
      for (let d = 0; d < dim; d++) {
        const xhat = (x[off + d] - mean) * invStd;
        const dxhat = dX[off + d];
        // dx = invStd * (dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
        dX[off + d] = invStd *
          (dxhat - sumDxhat * invN - xhat * (sumDxhatXhat * invN));
      }
    }
  }

  // ===================== optimizer =====================

  private getScheduledLearningRate(stepIndex: number): number {
    const base = this.config.learningRate;
    const warm = this.config.warmupSteps | 0;
    const total = this.config.totalSteps | 0;

    if (stepIndex <= warm) {
      return base * (stepIndex / Math.max(1, warm));
    }

    const denom = Math.max(1, total - warm);
    const progress = (stepIndex - warm) / denom;
    const p = Math.max(0, Math.min(1, progress));
    // cosine decay
    return base * 0.5 * (1.0 + Math.cos(Math.PI * p));
  }

  private adamUpdate(stepIndex: number, lr: number): void {
    // Adam with bias correction:
    // m = b1*m + (1-b1)*g
    // v = b2*v + (1-b2)*g^2
    // mHat = m/(1-b1^t)
    // vHat = v/(1-b2^t)
    // w -= lr * mHat/(sqrt(vHat)+eps)
    const b1 = this.config.beta1;
    const b2 = this.config.beta2;
    const eps = this.config.epsilon;

    // Update power terms incrementally
    this.beta1Pow *= b1;
    this.beta2Pow *= b2;

    const inv1 = 1.0 / (1.0 - this.beta1Pow);
    const inv2 = 1.0 / (1.0 - this.beta2Pow);

    for (let p = 0; p < this.params.length; p++) {
      const par = this.params[p];
      const w = par.w;
      const g = par.g;
      const m = par.m;
      const v = par.v;

      const n = w.length;
      for (let i = 0; i < n; i++) {
        const gi = g[i];
        const mi = (m[i] = b1 * m[i] + (1.0 - b1) * gi);
        const vi = (v[i] = b2 * v[i] + (1.0 - b2) * (gi * gi));

        const mHat = mi * inv1;
        const vHat = vi * inv2;

        w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  private resetOptimizerMoments(): void {
    // reset m,v and bias correction powers
    for (let p = 0; p < this.params.length; p++) {
      this.params[p].m.fill(0.0);
      this.params[p].v.fill(0.0);
    }
    this.beta1Pow = 1.0;
    this.beta2Pow = 1.0;
  }

  // ===================== regularization / grad management =====================

  private computeL2Loss(): number {
    const lambda = this.config.regularizationStrength;
    if (lambda <= 0) return 0.0;

    let sumSq = 0.0;
    for (let p = 0; p < this.params.length; p++) {
      const par = this.params[p];
      if (!par.l2) continue;
      const w = par.w;
      for (let i = 0; i < w.length; i++) {
        const v = w[i];
        sumSq += v * v;
      }
    }
    return 0.5 * lambda * sumSq;
  }

  private applyL2ToGradients(): void {
    const lambda = this.config.regularizationStrength;
    if (lambda <= 0) return;

    for (let p = 0; p < this.params.length; p++) {
      const par = this.params[p];
      if (!par.l2) continue;
      const w = par.w;
      const g = par.g;
      const n = w.length;
      for (let i = 0; i < n; i++) g[i] += lambda * w[i];
    }
  }

  private zeroGradients(): void {
    for (let p = 0; p < this.params.length; p++) {
      this.params[p].g.fill(0.0);
    }
  }

  private computeGlobalGradNorm(): number {
    let sum = 0.0;
    for (let p = 0; p < this.params.length; p++) {
      const g = this.params[p].g;
      for (let i = 0; i < g.length; i++) {
        const v = g[i];
        sum += v * v;
      }
    }
    return Math.sqrt(sum);
  }

  private scaleAllGradients(scale: number): void {
    for (let p = 0; p < this.params.length; p++) {
      const g = this.params[p].g;
      for (let i = 0; i < g.length; i++) g[i] *= scale;
    }
  }

  private getGradByWeight(w: Float64Array): Float64Array {
    // Lookup by reference (O(P)), but P is small and this is not dominant vs compute.
    // Could build a Map<Float64Array, Float64Array> at init for O(1), but Map alloc/GC costs.
    for (let p = 0; p < this.params.length; p++) {
      if (this.params[p].w === w) return this.params[p].g;
    }
    throw new Error("Gradient array not found for weight tensor");
  }

  // ===================== utilities =====================

  private copyXToRawBuffer(X: number[][], start: number, seqLen: number): void {
    const inputDim = this.inputDim | 0;
    for (let t = 0; t < seqLen; t++) {
      const row = X[start + t];
      const off = t * inputDim;
      for (let c = 0; c < inputDim; c++) this.xRawBuf[off + c] = row[c];
    }
  }

  private copySlice(src: Float64Array, dst: Float64Array, len: number): void {
    // Copy first len elements; avoid subarray allocations
    for (let i = 0; i < len; i++) dst[i] = src[i];
  }

  private rand(): number {
    // LCG: fast deterministic RNG
    // state = (a*state + c) mod 2^32
    this.rngState = (Math.imul(1664525, this.rngState) + 1013904223) | 0;
    // Convert to [0,1)
    return ((this.rngState >>> 0) as number) / 4294967296.0;
  }

  private initUniform(w: Float64Array, scale: number): void {
    for (let i = 0; i < w.length; i++) {
      const r = this.rand() * 2.0 - 1.0;
      w[i] = r * scale;
    }
  }

  private initXavierUniform(
    w: Float64Array,
    fanIn: number,
    fanOut: number,
  ): void {
    const limit = Math.sqrt(6.0 / (fanIn + fanOut));
    for (let i = 0; i < w.length; i++) {
      w[i] = (this.rand() * 2.0 - 1.0) * limit;
    }
  }

  private initHeUniform(w: Float64Array, fanIn: number): void {
    const limit = Math.sqrt(6.0 / fanIn);
    for (let i = 0; i < w.length; i++) {
      w[i] = (this.rand() * 2.0 - 1.0) * limit;
    }
  }

  private getAccuracy(): number {
    return 1.0 / (1.0 + this.avgLoss);
  }

  private countParameters(): number {
    let n = 0;
    for (let p = 0; p < this.params.length; p++) n += this.params[p].w.length;
    return n;
  }

  // ===================== serialization raw helpers =====================

  private exportWeightsRaw(): any {
    // Export by parameter list order
    const out: any = { params: new Array(this.params.length) };
    for (let i = 0; i < this.params.length; i++) {
      out.params[i] = {
        name: this.params[i].name,
        w: Array.from(this.params[i].w),
        l2: this.params[i].l2,
      };
    }
    return out;
  }

  private importWeightsRaw(obj: any): void {
    const params = obj.params as Array<
      { name: string; w: number[]; l2: boolean }
    >;
    if (!params || params.length !== this.params.length) {
      throw new Error("Weight list length mismatch");
    }

    // Match by index (and optionally name)
    for (let i = 0; i < params.length; i++) {
      const src = params[i];
      const dst = this.params[i];
      if (src.name !== dst.name) {
        // allow mismatch but warn-like behavior by throwing for strict correctness
        throw new Error(
          `Weight param name mismatch at ${i}: expected ${dst.name}, got ${src.name}`,
        );
      }
      if (src.w.length !== dst.w.length) {
        throw new Error(`Weight size mismatch for ${dst.name}`);
      }
      dst.w.set(src.w);
    }
  }

  private exportMomentsRaw(): any {
    const out: any = { params: new Array(this.params.length) };
    for (let i = 0; i < this.params.length; i++) {
      out.params[i] = {
        name: this.params[i].name,
        m: Array.from(this.params[i].m),
        v: Array.from(this.params[i].v),
      };
    }
    return out;
  }

  private importMomentsRaw(obj: any): void {
    const params = obj.params as Array<
      { name: string; m: number[]; v: number[] }
    >;
    if (!params || params.length !== this.params.length) {
      throw new Error("Moment list length mismatch");
    }

    for (let i = 0; i < params.length; i++) {
      const src = params[i];
      const dst = this.params[i];
      if (src.name !== dst.name) {
        throw new Error(
          `Moment param name mismatch at ${i}: expected ${dst.name}, got ${src.name}`,
        );
      }
      if (src.m.length !== dst.m.length || src.v.length !== dst.v.length) {
        throw new Error(`Moment size mismatch for ${dst.name}`);
      }
      dst.m.set(src.m);
      dst.v.set(src.v);
    }
  }
}
