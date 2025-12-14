/**
 * FusionTemporalTransformerRegression
 * A self-contained TypeScript implementation of a Fusion Temporal Transformer for multivariate regression
 * with incremental online learning (Adam), online z-score normalization (Welford), L2 regularization,
 * outlier downweighting, and ADWIN drift detection.
 *
 * Notes:
 * - Pure TypeScript (no Node/Deno/DOM APIs, no external dependencies).
 * - Uses Float64Array for all numerical tensors in hot paths.
 * - Lazily allocates and reuses buffers to minimize GC pressure.
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
  temporalConvWeights: number[][][]; // [scaleIndex][kernelIndex][embeddingDim*embeddingDim] (flattened row-major)
  scaleEmbeddings: number[][]; // [numScales][embeddingDim]
  positionalEncoding: number[][]; // [maxSequenceLength][embeddingDim]
  fusionWeights: number[][]; // [numScales][embeddingDim+1] (gate weights + bias)
  attentionWeights: number[][][]; // block-major flattened tensors
  ffnWeights: number[][][]; // block-major flattened tensors
  layerNormParams: number[][][]; // block-major: [block][norm(0/1)][(gamma|beta) vectors]
  outputWeights: number[][]; // [2][...] => [W_out(flat), W_pool(flat + bias)] (compact export)
  firstMoment: number[][][]; // list of matrices (each parameter exported as 2D array)
  secondMoment: number[][][]; // list of matrices (each parameter exported as 2D array)
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
  outlierDownweight: number;

  adwinDelta: number;
  adwinMaxWindow: number;
  adwinCheckStride: number;

  temporalScales: number[];
  temporalKernelSize: number;
  maxSequenceLength: number;

  useCausalMask: boolean;

  // Numerical stability / safety
  maxGradNorm: number;
  minStd: number;
};

type ParamRef = {
  name: string;
  w: Float64Array;
  m: Float64Array;
  v: Float64Array;
  g: Float64Array;
  rows: number;
  cols: number;
  l2: boolean; // whether L2 applies
};

type WelfordState = {
  count: number;
  mean: Float64Array;
  m2: Float64Array;
};

type AdwinState = {
  delta: number;
  maxWindow: number;
  checkStride: number;
  // ring buffer
  buf: Float64Array;
  start: number;
  size: number;
  // cached sums for faster mean computations on contiguous snapshot
  // (computed on-demand per check)
  counter: number;
  driftCount: number;
};

const DEFAULT_CONFIG: Config = {
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
  outlierDownweight: 0.1,

  adwinDelta: 0.002,
  adwinMaxWindow: 256,
  adwinCheckStride: 8,

  temporalScales: [1, 2, 4],
  temporalKernelSize: 3,
  maxSequenceLength: 512,

  useCausalMask: true,

  maxGradNorm: 5.0,
  minStd: 1e-6,
};

const SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2/pi)
const GELU_COEF = 0.044715;
const NEG_LARGE = -1e30;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : (x > hi ? hi : x);
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function softplus(x: number): number {
  // stable log(1+exp(x))
  if (x > 20) return x;
  if (x < -20) return Math.exp(x);
  return Math.log(1 + Math.exp(x));
}

function gelu(x: number): number {
  // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
  const x3 = x * x * x;
  const t = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
  const th = Math.tanh(t);
  return 0.5 * x * (1 + th);
}

function geluDeriv(x: number): number {
  // derivative for tanh approximation
  // d/dx [0.5*x*(1+tanh(t))] where t=a*(x+0.044715*x^3)
  const x2 = x * x;
  const x3 = x2 * x;
  const t = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
  const th = Math.tanh(t);
  const sech2 = 1 - th * th;
  const dt = SQRT_2_OVER_PI * (1 + 3 * GELU_COEF * x2);
  return 0.5 * (1 + th) + 0.5 * x * sech2 * dt;
}

function safeInvSqrt(x: number, eps: number): number {
  return 1 / Math.sqrt(x + eps);
}

function l2norm(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

function fillZero(x: Float64Array): void {
  x.fill(0);
}

function copyToNumberArray(src: Float64Array): number[] {
  const out = new Array<number>(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i];
  return out;
}

function randInitLCG(seed: number): () => number {
  // deterministic LCG in (0,1)
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return (s & 0xffffffff) / 4294967296;
  };
}

function xavierUniform(
  rng: () => number,
  fanIn: number,
  fanOut: number,
): number {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (rng() * 2 - 1) * limit;
}

function ensureFinite(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x;
}

/**
 * FusionTemporalTransformerRegression
 */
export class FusionTemporalTransformerRegression {
  // Config
  readonly #cfg: Config;

  // Model dimensions
  #isInitialized = false;
  #inputDim = 0;
  #outputDim = 0;

  // Training state
  #updateCount = 0;
  #sampleCount = 0;
  #runningLossSum = 0;
  #accuracy = 0;
  #prevLoss = Number.POSITIVE_INFINITY;
  #converged = false;
  #effectiveLR = 0;

  // Welford normalization (per feature and per output)
  #inNorm: WelfordState | null = null;
  #outNorm: WelfordState | null = null;

  // ADWIN drift detector over normalized error
  #adwin: AdwinState;

  // Parameters registry
  #params: ParamRef[] = [];

  // --- Weights (typed arrays) ---
  // Input embedding (linear projection)
  #W_embed: Float64Array | null = null; // [inputDim, embed]
  #b_embed: Float64Array | null = null; // [embed]

  // Temporal conv weights per scale: each has [kernelSize, embed, embed] flattened
  #W_conv: Float64Array[] = []; // per scale, len = kernelSize*embed*embed

  // Scale embeddings: [numScales, embed]
  #scaleEmb: Float64Array | null = null;

  // Fusion gates: per scale weights [embed], bias scalar
  #W_gate: Float64Array | null = null; // [numScales, embed] flattened
  #b_gate: Float64Array | null = null; // [numScales]

  // Transformer blocks
  // LayerNorm params: gamma/beta for LN1 and LN2 per block
  #ln1_gamma: Float64Array[] = [];
  #ln1_beta: Float64Array[] = [];
  #ln2_gamma: Float64Array[] = [];
  #ln2_beta: Float64Array[] = [];

  // Attention weights per block:
  // Wq/Wk/Wv: [numHeads, embed, headDim] flattened; Wo: [embed, embed]; bo: [embed]
  #att_Wq: Float64Array[] = [];
  #att_Wk: Float64Array[] = [];
  #att_Wv: Float64Array[] = [];
  #att_Wo: Float64Array[] = [];
  #att_bo: Float64Array[] = [];

  // FFN weights per block:
  // W1: [embed, hidden], b1: [hidden], W2: [hidden, embed], b2: [embed]
  #ffn_W1: Float64Array[] = [];
  #ffn_b1: Float64Array[] = [];
  #ffn_W2: Float64Array[] = [];
  #ffn_b2: Float64Array[] = [];

  // Pooling + output head
  #W_pool: Float64Array | null = null; // [embed]
  #b_pool: Float64Array | null = null; // [1]
  #W_out: Float64Array | null = null; // [embed, outDim]
  #b_out: Float64Array | null = null; // [outDim]

  // Positional encoding [maxSeq, embed]
  #posEnc: Float64Array | null = null;

  // --- Work buffers (reused) ---
  // Input buffers
  #xRaw: Float64Array | null = null; // [seqLen, inputDim]
  #xNorm: Float64Array | null = null; // [seqLen, inputDim]
  #xEmb: Float64Array | null = null; // [seqLen, embed]
  #xEmbGrad: Float64Array | null = null; // [seqLen, embed]
  #yTarget: Float64Array | null = null; // [outDim]
  #yNormTarget: Float64Array | null = null; // [outDim]
  #yHat: Float64Array | null = null; // [outDim] normalized prediction
  #yHatDenorm: Float64Array | null = null; // [outDim] denormalized prediction (for predict)

  // Conv buffers per scale (allocated based on seqLen)
  #convLen: Int32Array | null = null; // [numScales]
  #convPre: Float64Array[] = []; // per scale: [L_s, embed] pre-gelu
  #convPost: Float64Array[] = []; // per scale: [L_s, embed] post-gelu + pe + scaleEmb

  // Fused sequence buffers
  #fused: Float64Array | null = null; // [seqLen, embed]
  #fusedGrad: Float64Array | null = null; // [seqLen, embed]
  #gatesTmp: Float64Array | null = null; // [seqLen, numScales] flattened

  // Block states: [numBlocks+1, seqLen, embed] contiguous
  #states: Float64Array | null = null;
  #statesGrad: Float64Array | null = null;

  // LayerNorm cached stats: mean/invStd per block and timestep (for LN1 and LN2)
  #ln1_mean: Float64Array | null = null; // [numBlocks, seqLen]
  #ln1_invstd: Float64Array | null = null; // [numBlocks, seqLen]
  #ln2_mean: Float64Array | null = null; // [numBlocks, seqLen]
  #ln2_invstd: Float64Array | null = null; // [numBlocks, seqLen]

  // Attention work buffers (max per call)
  #q: Float64Array | null = null; // [seqLen, headDim]
  #k: Float64Array | null = null; // [seqLen, headDim]
  #v: Float64Array | null = null; // [seqLen, headDim]
  #scores: Float64Array | null = null; // [seqLen, seqLen]
  #headOut: Float64Array | null = null; // [seqLen, headDim]
  #attConcat: Float64Array | null = null; // [seqLen, embed]

  // FFN work buffers
  #ffnHidden: Float64Array | null = null; // [seqLen, hidden]
  #ffnHiddenGrad: Float64Array | null = null; // [seqLen, hidden]

  // Pool buffers
  #poolLogits: Float64Array | null = null; // [seqLen]
  #poolProbs: Float64Array | null = null; // [seqLen]
  #agg: Float64Array | null = null; // [embed]
  #aggGrad: Float64Array | null = null; // [embed]

  // Scratch vectors
  #tmpVecE: Float64Array | null = null; // [embed]
  #tmpVecE2: Float64Array | null = null; // [embed]
  #tmpHead: Float64Array | null = null; // [headDim]
  #tmpHead2: Float64Array | null = null; // [headDim]
  #tmpHidden: Float64Array | null = null; // [hidden]

  // For predict: store last seqLen inputs (raw)
  #lastSeqLen = 0;
  #lastX: Float64Array | null = null; // [seqLen, inputDim]

  // RNG for dropout/initialization
  #rng: () => number;

  /**
   * @param config Partial configuration overrides.
   * @example
   * const model = new FusionTemporalTransformerRegression({ embeddingDim: 64 });
   */
  constructor(config?: Partial<Config>) {
    this.#cfg = { ...DEFAULT_CONFIG, ...(config ?? {}) };

    // validate
    if (this.#cfg.embeddingDim <= 0) {
      this.#cfg.embeddingDim = DEFAULT_CONFIG.embeddingDim;
    }
    if (this.#cfg.numHeads <= 0) this.#cfg.numHeads = DEFAULT_CONFIG.numHeads;
    if (this.#cfg.embeddingDim % this.#cfg.numHeads !== 0) {
      // force a safe head count divisor
      let h = this.#cfg.numHeads;
      while (h > 1 && (this.#cfg.embeddingDim % h) !== 0) h--;
      this.#cfg.numHeads = Math.max(1, h);
    }
    if (this.#cfg.temporalKernelSize < 1) this.#cfg.temporalKernelSize = 1;
    if (this.#cfg.maxSequenceLength < 2) this.#cfg.maxSequenceLength = 2;

    // init rng
    this.#rng = randInitLCG(
      0xC0FFEE ^ (this.#cfg.embeddingDim << 8) ^ this.#cfg.numHeads,
    );

    // init adwin
    this.#adwin = {
      delta: this.#cfg.adwinDelta,
      maxWindow: this.#cfg.adwinMaxWindow,
      checkStride: this.#cfg.adwinCheckStride,
      buf: new Float64Array(Math.max(16, this.#cfg.adwinMaxWindow)),
      start: 0,
      size: 0,
      counter: 0,
      driftCount: 0,
    };
  }

  /**
   * Incremental online fit with Adam + Welford normalization + L2 regularization +
   * outlier downweighting + ADWIN drift detection.
   *
   * @param data Training sample for a sequence: xCoordinates[seq][inputDim], yCoordinates[seq][outputDim]
   * @returns FitResult
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xSeq = data.xCoordinates;
    const ySeq = data.yCoordinates;
    const seqLenIn = xSeq.length | 0;
    const seqLenY = ySeq.length | 0;

    if (seqLenIn <= 0 || seqLenY <= 0) {
      return {
        loss: 0,
        gradientNorm: 0,
        effectiveLearningRate: 0,
        isOutlier: false,
        converged: this.#converged,
        sampleIndex: this.#sampleCount,
        driftDetected: false,
      };
    }

    // Determine dims
    const inputDim = (xSeq[0]?.length ?? 0) | 0;
    const outputDim = (ySeq[0]?.length ?? 0) | 0;
    if (inputDim <= 0 || outputDim <= 0) {
      return {
        loss: 0,
        gradientNorm: 0,
        effectiveLearningRate: 0,
        isOutlier: false,
        converged: this.#converged,
        sampleIndex: this.#sampleCount,
        driftDetected: false,
      };
    }

    // Clip seqLen to maxSequenceLength: keep the most recent tail
    const maxSeq = this.#cfg.maxSequenceLength | 0;
    const seqLen = seqLenIn > maxSeq ? maxSeq : seqLenIn;

    // Initialize if needed
    if (!this.#isInitialized) {
      this.#initialize(inputDim, outputDim);
    } else {
      // if dims mismatch, reset then re-init
      if (this.#inputDim !== inputDim || this.#outputDim !== outputDim) {
        this.reset();
        this.#initialize(inputDim, outputDim);
      }
    }

    // Ensure work buffers for this seqLen
    this.#ensureWork(seqLen);

    // Copy x tail into xRaw
    const xRaw = this.#xRaw!;
    const xNorm = this.#xNorm!;
    const yTarget = this.#yTarget!;
    const yNormTarget = this.#yNormTarget!;

    // y target: last element of ySeq
    const yT = ySeq[seqLenY - 1];
    for (let j = 0; j < this.#outputDim; j++) {
      yTarget[j] = ensureFinite(yT[j] ?? 0);
    }

    const xStart = seqLenIn - seqLen;
    for (let t = 0; t < seqLen; t++) {
      const row = xSeq[xStart + t];
      const base = t * this.#inputDim;
      for (let f = 0; f < this.#inputDim; f++) {
        xRaw[base + f] = ensureFinite(row[f] ?? 0);
      }
    }

    // Store last X for predict (raw)
    this.#lastSeqLen = seqLen;
    if (!this.#lastX || this.#lastX.length !== xRaw.length) {
      this.#lastX = new Float64Array(xRaw.length);
    }
    this.#lastX.set(xRaw);

    // --- Online normalization update (Welford) ---
    this.#welfordUpdateInputs(xRaw, seqLen);
    this.#welfordUpdateOutputs(yTarget);

    // Normalize inputs to xNorm
    const inNorm = this.#inNorm!;
    const outNorm = this.#outNorm!;
    const eps = this.#cfg.epsilon;
    const minStd = this.#cfg.minStd;

    // Compute std from m2
    // input
    const inStd = this.#tmpEnsureInStd(); // Float64Array view cached inside
    for (let f = 0; f < this.#inputDim; f++) {
      const varF = inNorm.count > 1 ? (inNorm.m2[f] / (inNorm.count - 1)) : 1.0;
      const s = Math.sqrt(varF);
      inStd[f] = s < minStd ? minStd : s;
    }
    // output
    const outStd = this.#tmpEnsureOutStd();
    for (let j = 0; j < this.#outputDim; j++) {
      const varJ = outNorm.count > 1
        ? (outNorm.m2[j] / (outNorm.count - 1))
        : 1.0;
      const s = Math.sqrt(varJ);
      outStd[j] = s < minStd ? minStd : s;
    }

    // xNorm = (xRaw - mean) / (std + eps)
    for (let t = 0; t < seqLen; t++) {
      const base = t * this.#inputDim;
      for (let f = 0; f < this.#inputDim; f++) {
        xNorm[base + f] = (xRaw[base + f] - inNorm.mean[f]) / (inStd[f] + eps);
      }
    }

    // yNormTarget
    for (let j = 0; j < this.#outputDim; j++) {
      yNormTarget[j] = (yTarget[j] - outNorm.mean[j]) / (outStd[j] + eps);
    }

    // Forward pass (normalized)
    const yHat = this.#yHat!;
    const forwardCtx = this.#forward(seqLen, xNorm, yHat);

    // Outlier detection (in normalized space)
    let residualNorm = 0;
    for (let j = 0; j < this.#outputDim; j++) {
      const r = yNormTarget[j] - yHat[j];
      residualNorm += r * r;
    }
    residualNorm = Math.sqrt(residualNorm);

    const isOutlier = residualNorm > this.#cfg.outlierThreshold;
    const sampleWeight = isOutlier ? this.#cfg.outlierDownweight : 1.0;

    // Loss (MSE) + L2
    const nOut = this.#outputDim;
    let mse = 0;
    for (let j = 0; j < nOut; j++) {
      const d = yHat[j] - yNormTarget[j];
      mse += d * d;
    }
    mse *= 0.5 / nOut;

    // L2 loss contribution (for reporting)
    let l2 = 0;
    const reg = this.#cfg.regularizationStrength;
    if (reg > 0) {
      for (let p = 0; p < this.#params.length; p++) {
        const pr = this.#params[p];
        if (!pr.l2) continue;
        const w = pr.w;
        for (let i = 0; i < w.length; i++) l2 += w[i] * w[i];
      }
      l2 *= 0.5 * reg;
    }

    const totalLoss = sampleWeight * mse + l2;

    // ADWIN drift detection over normalized error magnitude (mse or residualNorm)
    const driftDetected = this.#adwinUpdate(sampleWeight * mse);

    // If drift detected, reset normalization stats (and optionally accuracy stats)
    if (driftDetected) {
      this.#resetNormalizationStats();
    }

    // Backward pass -> gradients
    const gradNorm = this.#backwardAndUpdate(
      seqLen,
      xNorm,
      yNormTarget,
      yHat,
      forwardCtx,
      sampleWeight,
    );

    // Update running accuracy
    this.#sampleCount++;
    this.#runningLossSum += totalLoss;
    const avgLoss = this.#runningLossSum / this.#sampleCount;
    this.#accuracy = 1 / (1 + avgLoss);

    // Convergence
    const converged =
      Math.abs(this.#prevLoss - totalLoss) < this.#cfg.convergenceThreshold;
    this.#converged = this.#converged || converged;
    this.#prevLoss = totalLoss;

    return {
      loss: totalLoss,
      gradientNorm: gradNorm,
      effectiveLearningRate: this.#effectiveLR,
      isOutlier,
      converged: this.#converged,
      sampleIndex: this.#sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict future steps (autoregressive placeholder over last observed features).
   * Uses the last observed sequence; for each future step, repeats the last feature row.
   *
   * @param futureSteps number of steps to predict
   */
  predict(futureSteps: number): PredictionResult {
    const steps = (futureSteps | 0) <= 0 ? 0 : (futureSteps | 0);
    if (
      !this.#isInitialized || this.#sampleCount <= 0 || !this.#lastX ||
      this.#lastSeqLen <= 0
    ) {
      return {
        predictions: [],
        accuracy: this.#accuracy,
        sampleCount: this.#sampleCount,
        isModelReady: false,
      };
    }

    const seqLen = this.#lastSeqLen;
    this.#ensureWork(seqLen);

    const xRaw = this.#xRaw!;
    const xNorm = this.#xNorm!;
    const yHat = this.#yHat!;
    const yHatDen = this.#yHatDenorm!;
    const eps = this.#cfg.epsilon;
    const minStd = this.#cfg.minStd;

    // Load last raw into xRaw (we'll mutate xRaw for rolling window)
    xRaw.set(this.#lastX);

    const inNorm = this.#inNorm!;
    const outNorm = this.#outNorm!;
    const inStd = this.#tmpEnsureInStd();
    const outStd = this.#tmpEnsureOutStd();
    for (let f = 0; f < this.#inputDim; f++) {
      const varF = inNorm.count > 1 ? (inNorm.m2[f] / (inNorm.count - 1)) : 1.0;
      const s = Math.sqrt(varF);
      inStd[f] = s < minStd ? minStd : s;
    }
    for (let j = 0; j < this.#outputDim; j++) {
      const varJ = outNorm.count > 1
        ? (outNorm.m2[j] / (outNorm.count - 1))
        : 1.0;
      const s = Math.sqrt(varJ);
      outStd[j] = s < minStd ? minStd : s;
    }

    const preds: SinglePrediction[] = [];

    // Standard error estimate: outStd / sqrt(sampleCount)
    const se = new Float64Array(this.#outputDim);
    const denom = Math.sqrt(Math.max(1, this.#sampleCount));
    for (let j = 0; j < this.#outputDim; j++) se[j] = outStd[j] / denom;

    const z = 1.96; // 95% CI multiplier

    // last feature row
    const lastRowBase = (seqLen - 1) * this.#inputDim;

    for (let step = 0; step < steps; step++) {
      // normalize xRaw -> xNorm
      for (let t = 0; t < seqLen; t++) {
        const base = t * this.#inputDim;
        for (let f = 0; f < this.#inputDim; f++) {
          xNorm[base + f] = (xRaw[base + f] - inNorm.mean[f]) /
            (inStd[f] + eps);
        }
      }

      // forward
      this.#forward(seqLen, xNorm, yHat);

      // denormalize yHat -> yHatDen
      for (let j = 0; j < this.#outputDim; j++) {
        yHatDen[j] = yHat[j] * (outStd[j] + eps) + outNorm.mean[j];
      }

      const predArr = new Array<number>(this.#outputDim);
      const lbArr = new Array<number>(this.#outputDim);
      const ubArr = new Array<number>(this.#outputDim);
      const seArr = new Array<number>(this.#outputDim);

      for (let j = 0; j < this.#outputDim; j++) {
        const p = yHatDen[j];
        const s = se[j];
        predArr[j] = p;
        seArr[j] = s;
        lbArr[j] = p - z * s;
        ubArr[j] = p + z * s;
      }

      preds.push({
        predicted: predArr,
        lowerBound: lbArr,
        upperBound: ubArr,
        standardError: seArr,
      });

      // roll window: drop oldest row, append last row repeated (naive autoregressive placeholder)
      // shift left by one row
      for (let t = 0; t < seqLen - 1; t++) {
        const dst = t * this.#inputDim;
        const src = (t + 1) * this.#inputDim;
        for (let f = 0; f < this.#inputDim; f++) xRaw[dst + f] = xRaw[src + f];
      }
      // append repeated last observed row
      const dstBase = (seqLen - 1) * this.#inputDim;
      for (let f = 0; f < this.#inputDim; f++) {
        xRaw[dstBase + f] = xRaw[lastRowBase + f];
      }
    }

    // store updated window back (optional)
    this.#lastX.set(xRaw);

    return {
      predictions: preds,
      accuracy: this.#accuracy,
      sampleCount: this.#sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Model summary.
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this.#isInitialized,
      inputDimension: this.#inputDim,
      outputDimension: this.#outputDim,
      numBlocks: this.#cfg.numBlocks,
      embeddingDim: this.#cfg.embeddingDim,
      numHeads: this.#cfg.numHeads,
      temporalScales: this.#cfg.temporalScales.slice(),
      totalParameters: this.#countParameters(),
      sampleCount: this.#sampleCount,
      accuracy: this.#accuracy,
      converged: this.#converged,
      effectiveLearningRate: this.#effectiveLR,
      driftCount: this.#adwin.driftCount,
    };
  }

  /**
   * Export weights (and optimizer moments) for inspection.
   * Note: This creates JS arrays (allocations) by design; not intended for hot paths.
   */
  getWeights(): WeightInfo {
    if (!this.#isInitialized) {
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
        updateCount: this.#updateCount,
      };
    }

    const embed = this.#cfg.embeddingDim;
    const kSize = this.#cfg.temporalKernelSize;
    const numScales = this.#cfg.temporalScales.length;

    const temporalConvWeights: number[][][] = [];
    for (let s = 0; s < numScales; s++) {
      const w = this.#W_conv[s];
      const perK: number[][] = [];
      const stride = embed * embed;
      for (let k = 0; k < kSize; k++) {
        const arr = new Array<number>(stride);
        const off = k * stride;
        for (let i = 0; i < stride; i++) arr[i] = w[off + i];
        perK.push(arr);
      }
      temporalConvWeights.push(perK);
    }

    const scaleEmbeddings: number[][] = [];
    const se = this.#scaleEmb!;
    for (let s = 0; s < numScales; s++) {
      const row = new Array<number>(embed);
      const off = s * embed;
      for (let d = 0; d < embed; d++) row[d] = se[off + d];
      scaleEmbeddings.push(row);
    }

    const positionalEncoding: number[][] = [];
    const pe = this.#posEnc!;
    const maxSeq = this.#cfg.maxSequenceLength;
    for (let t = 0; t < maxSeq; t++) {
      const row = new Array<number>(embed);
      const off = t * embed;
      for (let d = 0; d < embed; d++) row[d] = pe[off + d];
      positionalEncoding.push(row);
    }

    const fusionWeights: number[][] = [];
    const wg = this.#W_gate!;
    const bg = this.#b_gate!;
    for (let s = 0; s < numScales; s++) {
      const row = new Array<number>(embed + 1);
      const off = s * embed;
      for (let d = 0; d < embed; d++) row[d] = wg[off + d];
      row[embed] = bg[s];
      fusionWeights.push(row);
    }

    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this.#cfg.numBlocks; b++) {
      const blockArr: number[][] = [];

      // Wq/Wk/Wv flattened
      blockArr.push(copyToNumberArray(this.#att_Wq[b]));
      blockArr.push(copyToNumberArray(this.#att_Wk[b]));
      blockArr.push(copyToNumberArray(this.#att_Wv[b]));
      blockArr.push(copyToNumberArray(this.#att_Wo[b]));
      blockArr.push(copyToNumberArray(this.#att_bo[b]));
      attentionWeights.push(blockArr);
    }

    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this.#cfg.numBlocks; b++) {
      const blockArr: number[][] = [];
      blockArr.push(copyToNumberArray(this.#ffn_W1[b]));
      blockArr.push(copyToNumberArray(this.#ffn_b1[b]));
      blockArr.push(copyToNumberArray(this.#ffn_W2[b]));
      blockArr.push(copyToNumberArray(this.#ffn_b2[b]));
      ffnWeights.push(blockArr);
    }

    const layerNormParams: number[][][] = [];
    for (let b = 0; b < this.#cfg.numBlocks; b++) {
      const norms: number[][] = [];
      norms.push(copyToNumberArray(this.#ln1_gamma[b]));
      norms.push(copyToNumberArray(this.#ln1_beta[b]));
      norms.push(copyToNumberArray(this.#ln2_gamma[b]));
      norms.push(copyToNumberArray(this.#ln2_beta[b]));
      layerNormParams.push(norms);
    }

    const outputWeights: number[][] = [];
    outputWeights.push(copyToNumberArray(this.#W_out!));
    outputWeights.push(copyToNumberArray(this.#b_out!));
    outputWeights.push(copyToNumberArray(this.#W_pool!));
    outputWeights.push(copyToNumberArray(this.#b_pool!));
    outputWeights.push(copyToNumberArray(this.#W_embed!));
    outputWeights.push(copyToNumberArray(this.#b_embed!));

    // Export optimizer moments as list-of-matrices (2D arrays)
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];
    for (let p = 0; p < this.#params.length; p++) {
      const pr = this.#params[p];
      firstMoment.push(this.#exportAs2D(pr.m, pr.rows, pr.cols));
      secondMoment.push(this.#exportAs2D(pr.v, pr.rows, pr.cols));
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
      updateCount: this.#updateCount,
    };
  }

  /**
   * Get normalization statistics.
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.#isInitialized || !this.#inNorm || !this.#outNorm) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    const inNorm = this.#inNorm;
    const outNorm = this.#outNorm;
    const minStd = this.#cfg.minStd;

    const inMean = copyToNumberArray(inNorm.mean);
    const outMean = copyToNumberArray(outNorm.mean);

    const inStdArr = new Array<number>(this.#inputDim);
    const outStdArr = new Array<number>(this.#outputDim);

    for (let f = 0; f < this.#inputDim; f++) {
      const v = inNorm.count > 1 ? (inNorm.m2[f] / (inNorm.count - 1)) : 1.0;
      const s = Math.sqrt(v);
      inStdArr[f] = s < minStd ? minStd : s;
    }
    for (let j = 0; j < this.#outputDim; j++) {
      const v = outNorm.count > 1 ? (outNorm.m2[j] / (outNorm.count - 1)) : 1.0;
      const s = Math.sqrt(v);
      outStdArr[j] = s < minStd ? minStd : s;
    }

    return {
      inputMean: inMean,
      inputStd: inStdArr,
      outputMean: outMean,
      outputStd: outStdArr,
      count: this.#sampleCount,
    };
  }

  /**
   * Reset the entire model state (weights, optimizer, normalization, drift window, stats).
   */
  reset(): void {
    this.#isInitialized = false;
    this.#inputDim = 0;
    this.#outputDim = 0;

    this.#updateCount = 0;
    this.#sampleCount = 0;
    this.#runningLossSum = 0;
    this.#accuracy = 0;
    this.#prevLoss = Number.POSITIVE_INFINITY;
    this.#converged = false;
    this.#effectiveLR = 0;

    this.#inNorm = null;
    this.#outNorm = null;

    // reset adwin
    this.#adwin.start = 0;
    this.#adwin.size = 0;
    this.#adwin.counter = 0;
    this.#adwin.driftCount = 0;

    // clear params & weights
    this.#params = [];
    this.#W_embed = null;
    this.#b_embed = null;
    this.#W_conv = [];
    this.#scaleEmb = null;
    this.#W_gate = null;
    this.#b_gate = null;

    this.#ln1_gamma = [];
    this.#ln1_beta = [];
    this.#ln2_gamma = [];
    this.#ln2_beta = [];

    this.#att_Wq = [];
    this.#att_Wk = [];
    this.#att_Wv = [];
    this.#att_Wo = [];
    this.#att_bo = [];

    this.#ffn_W1 = [];
    this.#ffn_b1 = [];
    this.#ffn_W2 = [];
    this.#ffn_b2 = [];

    this.#W_pool = null;
    this.#b_pool = null;
    this.#W_out = null;
    this.#b_out = null;

    this.#posEnc = null;

    // buffers remain allocated (reuse), but invalidate lastX
    this.#lastSeqLen = 0;
    this.#lastX = null;
  }

  /**
   * Serialize the full state into a JSON string.
   */
  save(): string {
    const state: any = {
      cfg: this.#cfg,
      isInitialized: this.#isInitialized,
      inputDim: this.#inputDim,
      outputDim: this.#outputDim,

      updateCount: this.#updateCount,
      sampleCount: this.#sampleCount,
      runningLossSum: this.#runningLossSum,
      accuracy: this.#accuracy,
      prevLoss: this.#prevLoss,
      converged: this.#converged,
      effectiveLR: this.#effectiveLR,

      adwin: {
        delta: this.#adwin.delta,
        maxWindow: this.#adwin.maxWindow,
        checkStride: this.#adwin.checkStride,
        buf: copyToNumberArray(this.#adwin.buf),
        start: this.#adwin.start,
        size: this.#adwin.size,
        counter: this.#adwin.counter,
        driftCount: this.#adwin.driftCount,
      },

      inNorm: this.#inNorm
        ? {
          count: this.#inNorm.count,
          mean: copyToNumberArray(this.#inNorm.mean),
          m2: copyToNumberArray(this.#inNorm.m2),
        }
        : null,

      outNorm: this.#outNorm
        ? {
          count: this.#outNorm.count,
          mean: copyToNumberArray(this.#outNorm.mean),
          m2: copyToNumberArray(this.#outNorm.m2),
        }
        : null,

      weights: this.#exportWeightsForSave(),
      params: this.#exportParamsForSave(),

      lastSeqLen: this.#lastSeqLen,
      lastX: this.#lastX ? copyToNumberArray(this.#lastX) : null,
    };

    return JSON.stringify(state);
  }

  /**
   * Load the full state from a JSON string.
   * @param w JSON string produced by save()
   */
  load(w: string): void {
    const state = JSON.parse(w);

    // config
    // Keep current cfg; use loaded cfg values if compatible
    const cfgIn = state.cfg as Config | undefined;
    if (cfgIn) {
      // shallow merge
      (this as any).#cfg = { ...DEFAULT_CONFIG, ...cfgIn };
    }

    this.reset();

    this.#isInitialized = !!state.isInitialized;
    this.#inputDim = state.inputDim | 0;
    this.#outputDim = state.outputDim | 0;

    this.#updateCount = state.updateCount | 0;
    this.#sampleCount = state.sampleCount | 0;
    this.#runningLossSum = +state.runningLossSum || 0;
    this.#accuracy = +state.accuracy || 0;
    this.#prevLoss = +state.prevLoss || Number.POSITIVE_INFINITY;
    this.#converged = !!state.converged;
    this.#effectiveLR = +state.effectiveLR || 0;

    // adwin
    const ad = state.adwin;
    if (ad) {
      this.#adwin.delta = +ad.delta || this.#cfg.adwinDelta;
      this.#adwin.maxWindow = (ad.maxWindow | 0) || this.#cfg.adwinMaxWindow;
      this.#adwin.checkStride = (ad.checkStride | 0) ||
        this.#cfg.adwinCheckStride;

      const bufArr = ad.buf as number[] | null;
      const len = bufArr ? bufArr.length : Math.max(16, this.#adwin.maxWindow);
      this.#adwin.buf = new Float64Array(len);
      if (bufArr) {
        for (let i = 0; i < len; i++) this.#adwin.buf[i] = +bufArr[i] || 0;
      }

      this.#adwin.start = ad.start | 0;
      this.#adwin.size = ad.size | 0;
      this.#adwin.counter = ad.counter | 0;
      this.#adwin.driftCount = ad.driftCount | 0;
    }

    // norms
    if (state.inNorm) {
      const inn = state.inNorm;
      const mean = new Float64Array(this.#inputDim);
      const m2 = new Float64Array(this.#inputDim);
      const meanArr = inn.mean as number[];
      const m2Arr = inn.m2 as number[];
      for (let i = 0; i < this.#inputDim; i++) {
        mean[i] = +meanArr[i] || 0;
        m2[i] = +m2Arr[i] || 0;
      }
      this.#inNorm = { count: inn.count | 0, mean, m2 };
    }
    if (state.outNorm) {
      const onn = state.outNorm;
      const mean = new Float64Array(this.#outputDim);
      const m2 = new Float64Array(this.#outputDim);
      const meanArr = onn.mean as number[];
      const m2Arr = onn.m2 as number[];
      for (let i = 0; i < this.#outputDim; i++) {
        mean[i] = +meanArr[i] || 0;
        m2[i] = +m2Arr[i] || 0;
      }
      this.#outNorm = { count: onn.count | 0, mean, m2 };
    }

    // Re-initialize weights/params if initialized
    if (this.#isInitialized && this.#inputDim > 0 && this.#outputDim > 0) {
      this.#initialize(this.#inputDim, this.#outputDim, true);
      this.#importWeightsFromSave(state.weights);
      this.#importParamsFromSave(state.params);
    }

    // lastX
    this.#lastSeqLen = state.lastSeqLen | 0;
    if (state.lastX && this.#lastSeqLen > 0) {
      const arr = state.lastX as number[];
      const expected = this.#lastSeqLen * this.#inputDim;
      if (arr.length === expected) {
        this.#lastX = new Float64Array(expected);
        for (let i = 0; i < expected; i++) this.#lastX[i] = +arr[i] || 0;
      }
    }
  }

  // =====================================================================================
  // Initialization / buffers
  // =====================================================================================

  #initialize(
    inputDim: number,
    outputDim: number,
    skipInitMoments = false,
  ): void {
    this.#inputDim = inputDim | 0;
    this.#outputDim = outputDim | 0;

    const cfg = this.#cfg;
    const embed = cfg.embeddingDim | 0;
    const heads = cfg.numHeads | 0;
    const headDim = (embed / heads) | 0;
    const hidden = (embed * cfg.ffnMultiplier) | 0;
    const kSize = cfg.temporalKernelSize | 0;
    const numScales = cfg.temporalScales.length | 0;

    // Positional encoding
    this.#posEnc = new Float64Array(cfg.maxSequenceLength * embed);
    this.#buildPositionalEncoding(this.#posEnc, cfg.maxSequenceLength, embed);

    // Norm states
    this.#inNorm = {
      count: 0,
      mean: new Float64Array(this.#inputDim),
      m2: new Float64Array(this.#inputDim),
    };
    this.#outNorm = {
      count: 0,
      mean: new Float64Array(this.#outputDim),
      m2: new Float64Array(this.#outputDim),
    };

    // Allocate weights
    const rng = this.#rng;

    // W_embed
    this.#W_embed = new Float64Array(this.#inputDim * embed);
    this.#b_embed = new Float64Array(embed);
    for (let i = 0; i < this.#W_embed.length; i++) {
      this.#W_embed[i] = xavierUniform(rng, this.#inputDim, embed);
    }
    fillZero(this.#b_embed);

    // Temporal conv weights
    this.#W_conv = new Array<Float64Array>(numScales);
    for (let s = 0; s < numScales; s++) {
      const w = new Float64Array(kSize * embed * embed);
      // fanIn ~ kSize*embed, fanOut ~ embed
      for (let i = 0; i < w.length; i++) {
        w[i] = xavierUniform(rng, kSize * embed, embed);
      }
      this.#W_conv[s] = w;
    }

    // Scale embeddings
    this.#scaleEmb = new Float64Array(numScales * embed);
    for (let i = 0; i < this.#scaleEmb.length; i++) {
      this.#scaleEmb[i] = xavierUniform(rng, embed, embed);
    }

    // Fusion gate weights/bias
    this.#W_gate = new Float64Array(numScales * embed);
    this.#b_gate = new Float64Array(numScales);
    for (let i = 0; i < this.#W_gate.length; i++) {
      this.#W_gate[i] = xavierUniform(rng, embed, 1);
    }
    fillZero(this.#b_gate);

    // Transformer blocks
    this.#ln1_gamma = new Array<Float64Array>(cfg.numBlocks);
    this.#ln1_beta = new Array<Float64Array>(cfg.numBlocks);
    this.#ln2_gamma = new Array<Float64Array>(cfg.numBlocks);
    this.#ln2_beta = new Array<Float64Array>(cfg.numBlocks);

    this.#att_Wq = new Array<Float64Array>(cfg.numBlocks);
    this.#att_Wk = new Array<Float64Array>(cfg.numBlocks);
    this.#att_Wv = new Array<Float64Array>(cfg.numBlocks);
    this.#att_Wo = new Array<Float64Array>(cfg.numBlocks);
    this.#att_bo = new Array<Float64Array>(cfg.numBlocks);

    this.#ffn_W1 = new Array<Float64Array>(cfg.numBlocks);
    this.#ffn_b1 = new Array<Float64Array>(cfg.numBlocks);
    this.#ffn_W2 = new Array<Float64Array>(cfg.numBlocks);
    this.#ffn_b2 = new Array<Float64Array>(cfg.numBlocks);

    const qkvLen = heads * embed * headDim;
    for (let b = 0; b < cfg.numBlocks; b++) {
      // LN params
      const g1 = new Float64Array(embed);
      const b1 = new Float64Array(embed);
      const g2 = new Float64Array(embed);
      const b2 = new Float64Array(embed);
      // gamma init to 1, beta to 0
      for (let i = 0; i < embed; i++) {
        g1[i] = 1;
        g2[i] = 1;
      }
      fillZero(b1);
      fillZero(b2);
      this.#ln1_gamma[b] = g1;
      this.#ln1_beta[b] = b1;
      this.#ln2_gamma[b] = g2;
      this.#ln2_beta[b] = b2;

      // Attention weights
      const Wq = new Float64Array(qkvLen);
      const Wk = new Float64Array(qkvLen);
      const Wv = new Float64Array(qkvLen);
      // Wo and bo
      const Wo = new Float64Array(embed * embed);
      const bo = new Float64Array(embed);
      for (let i = 0; i < qkvLen; i++) {
        Wq[i] = xavierUniform(rng, embed, headDim);
        Wk[i] = xavierUniform(rng, embed, headDim);
        Wv[i] = xavierUniform(rng, embed, headDim);
      }
      for (let i = 0; i < Wo.length; i++) {
        Wo[i] = xavierUniform(rng, embed, embed);
      }
      fillZero(bo);

      this.#att_Wq[b] = Wq;
      this.#att_Wk[b] = Wk;
      this.#att_Wv[b] = Wv;
      this.#att_Wo[b] = Wo;
      this.#att_bo[b] = bo;

      // FFN weights
      const W1 = new Float64Array(embed * hidden);
      const bb1 = new Float64Array(hidden);
      const W2 = new Float64Array(hidden * embed);
      const bb2 = new Float64Array(embed);
      for (let i = 0; i < W1.length; i++) {
        W1[i] = xavierUniform(rng, embed, hidden);
      }
      for (let i = 0; i < W2.length; i++) {
        W2[i] = xavierUniform(rng, hidden, embed);
      }
      fillZero(bb1);
      fillZero(bb2);

      this.#ffn_W1[b] = W1;
      this.#ffn_b1[b] = bb1;
      this.#ffn_W2[b] = W2;
      this.#ffn_b2[b] = bb2;
    }

    // Pooling and output head
    this.#W_pool = new Float64Array(embed);
    this.#b_pool = new Float64Array(1);
    for (let i = 0; i < embed; i++) {
      this.#W_pool[i] = xavierUniform(rng, embed, 1);
    }
    this.#b_pool[0] = 0;

    this.#W_out = new Float64Array(embed * this.#outputDim);
    this.#b_out = new Float64Array(this.#outputDim);
    for (let i = 0; i < this.#W_out.length; i++) {
      this.#W_out[i] = xavierUniform(rng, embed, this.#outputDim);
    }
    fillZero(this.#b_out);

    // Build parameter registry
    this.#params = [];
    const reg = cfg.regularizationStrength > 0;

    // helper to register
    const register = (
      name: string,
      w: Float64Array,
      rows: number,
      cols: number,
      l2: boolean,
    ) => {
      const m = new Float64Array(w.length);
      const v = new Float64Array(w.length);
      const g = new Float64Array(w.length);
      if (!skipInitMoments) {
        fillZero(m);
        fillZero(v);
        fillZero(g);
      }
      this.#params.push({ name, w, m, v, g, rows, cols, l2 });
    };

    register("W_embed", this.#W_embed, this.#inputDim, embed, reg);
    register("b_embed", this.#b_embed, 1, embed, false);

    for (let s = 0; s < numScales; s++) {
      register(`W_conv_${s}`, this.#W_conv[s], kSize * embed, embed, reg);
    }

    register("scaleEmb", this.#scaleEmb, numScales, embed, reg);
    register("W_gate", this.#W_gate, numScales, embed, reg);
    register("b_gate", this.#b_gate, 1, numScales, false);

    for (let b = 0; b < cfg.numBlocks; b++) {
      register(`ln1_gamma_${b}`, this.#ln1_gamma[b], 1, embed, false);
      register(`ln1_beta_${b}`, this.#ln1_beta[b], 1, embed, false);
      register(`ln2_gamma_${b}`, this.#ln2_gamma[b], 1, embed, false);
      register(`ln2_beta_${b}`, this.#ln2_beta[b], 1, embed, false);

      register(`att_Wq_${b}`, this.#att_Wq[b], heads * embed, headDim, reg);
      register(`att_Wk_${b}`, this.#att_Wk[b], heads * embed, headDim, reg);
      register(`att_Wv_${b}`, this.#att_Wv[b], heads * embed, headDim, reg);
      register(`att_Wo_${b}`, this.#att_Wo[b], embed, embed, reg);
      register(`att_bo_${b}`, this.#att_bo[b], 1, embed, false);

      register(`ffn_W1_${b}`, this.#ffn_W1[b], embed, hidden, reg);
      register(`ffn_b1_${b}`, this.#ffn_b1[b], 1, hidden, false);
      register(`ffn_W2_${b}`, this.#ffn_W2[b], hidden, embed, reg);
      register(`ffn_b2_${b}`, this.#ffn_b2[b], 1, embed, false);
    }

    register("W_pool", this.#W_pool, 1, embed, reg);
    register("b_pool", this.#b_pool, 1, 1, false);
    register("W_out", this.#W_out, embed, this.#outputDim, reg);
    register("b_out", this.#b_out, 1, this.#outputDim, false);

    this.#isInitialized = true;
  }

  #ensureWork(seqLen: number): void {
    const embed = this.#cfg.embeddingDim;
    const heads = this.#cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const hidden = (embed * this.#cfg.ffnMultiplier) | 0;
    const numBlocks = this.#cfg.numBlocks;
    const numScales = this.#cfg.temporalScales.length;

    // Input buffers (xRaw, xNorm)
    const xRawLen = seqLen * this.#inputDim;
    if (!this.#xRaw || this.#xRaw.length !== xRawLen) {
      this.#xRaw = new Float64Array(xRawLen);
    }
    if (!this.#xNorm || this.#xNorm.length !== xRawLen) {
      this.#xNorm = new Float64Array(xRawLen);
    }

    // xEmb and grad
    const xEmbLen = seqLen * embed;
    if (!this.#xEmb || this.#xEmb.length !== xEmbLen) {
      this.#xEmb = new Float64Array(xEmbLen);
    }
    if (!this.#xEmbGrad || this.#xEmbGrad.length !== xEmbLen) {
      this.#xEmbGrad = new Float64Array(xEmbLen);
    }

    // outputs
    if (!this.#yTarget || this.#yTarget.length !== this.#outputDim) {
      this.#yTarget = new Float64Array(this.#outputDim);
    }
    if (!this.#yNormTarget || this.#yNormTarget.length !== this.#outputDim) {
      this.#yNormTarget = new Float64Array(this.#outputDim);
    }
    if (!this.#yHat || this.#yHat.length !== this.#outputDim) {
      this.#yHat = new Float64Array(this.#outputDim);
    }
    if (!this.#yHatDenorm || this.#yHatDenorm.length !== this.#outputDim) {
      this.#yHatDenorm = new Float64Array(this.#outputDim);
    }

    // conv lengths, buffers per scale
    if (!this.#convLen || this.#convLen.length !== numScales) {
      this.#convLen = new Int32Array(numScales);
    }
    this.#convPre.length = numScales;
    this.#convPost.length = numScales;
    for (let s = 0; s < numScales; s++) {
      const stride = this.#cfg.temporalScales[s] | 0;
      const Ls = Math.max(1, Math.ceil(seqLen / stride));
      this.#convLen[s] = Ls;
      const len = Ls * embed;
      if (!this.#convPre[s] || this.#convPre[s].length !== len) {
        this.#convPre[s] = new Float64Array(len);
      }
      if (!this.#convPost[s] || this.#convPost[s].length !== len) {
        this.#convPost[s] = new Float64Array(len);
      }
    }

    // fused buffers
    if (!this.#fused || this.#fused.length !== xEmbLen) {
      this.#fused = new Float64Array(xEmbLen);
    }
    if (!this.#fusedGrad || this.#fusedGrad.length !== xEmbLen) {
      this.#fusedGrad = new Float64Array(xEmbLen);
    }
    if (!this.#gatesTmp || this.#gatesTmp.length !== (seqLen * numScales)) {
      this.#gatesTmp = new Float64Array(seqLen * numScales);
    }

    // block states and grads
    const statesLen = (numBlocks + 1) * xEmbLen;
    if (!this.#states || this.#states.length !== statesLen) {
      this.#states = new Float64Array(statesLen);
    }
    if (!this.#statesGrad || this.#statesGrad.length !== statesLen) {
      this.#statesGrad = new Float64Array(statesLen);
    }

    // LN caches
    const lnLen = numBlocks * seqLen;
    if (!this.#ln1_mean || this.#ln1_mean.length !== lnLen) {
      this.#ln1_mean = new Float64Array(lnLen);
    }
    if (!this.#ln1_invstd || this.#ln1_invstd.length !== lnLen) {
      this.#ln1_invstd = new Float64Array(lnLen);
    }
    if (!this.#ln2_mean || this.#ln2_mean.length !== lnLen) {
      this.#ln2_mean = new Float64Array(lnLen);
    }
    if (!this.#ln2_invstd || this.#ln2_invstd.length !== lnLen) {
      this.#ln2_invstd = new Float64Array(lnLen);
    }

    // Attention buffers
    const qkvLen = seqLen * headDim;
    const scoreLen = seqLen * seqLen;
    const attConcatLen = seqLen * embed;
    if (!this.#q || this.#q.length !== qkvLen) {
      this.#q = new Float64Array(qkvLen);
    }
    if (!this.#k || this.#k.length !== qkvLen) {
      this.#k = new Float64Array(qkvLen);
    }
    if (!this.#v || this.#v.length !== qkvLen) {
      this.#v = new Float64Array(qkvLen);
    }
    if (!this.#scores || this.#scores.length !== scoreLen) {
      this.#scores = new Float64Array(scoreLen);
    }
    if (!this.#headOut || this.#headOut.length !== qkvLen) {
      this.#headOut = new Float64Array(qkvLen);
    }
    if (!this.#attConcat || this.#attConcat.length !== attConcatLen) {
      this.#attConcat = new Float64Array(attConcatLen);
    }

    // FFN buffers
    const ffnLen = seqLen * hidden;
    if (!this.#ffnHidden || this.#ffnHidden.length !== ffnLen) {
      this.#ffnHidden = new Float64Array(ffnLen);
    }
    if (!this.#ffnHiddenGrad || this.#ffnHiddenGrad.length !== ffnLen) {
      this.#ffnHiddenGrad = new Float64Array(ffnLen);
    }

    // pool buffers
    if (!this.#poolLogits || this.#poolLogits.length !== seqLen) {
      this.#poolLogits = new Float64Array(seqLen);
    }
    if (!this.#poolProbs || this.#poolProbs.length !== seqLen) {
      this.#poolProbs = new Float64Array(seqLen);
    }
    if (!this.#agg || this.#agg.length !== embed) {
      this.#agg = new Float64Array(embed);
    }
    if (!this.#aggGrad || this.#aggGrad.length !== embed) {
      this.#aggGrad = new Float64Array(embed);
    }

    // scratch
    if (!this.#tmpVecE || this.#tmpVecE.length !== embed) {
      this.#tmpVecE = new Float64Array(embed);
    }
    if (!this.#tmpVecE2 || this.#tmpVecE2.length !== embed) {
      this.#tmpVecE2 = new Float64Array(embed);
    }
    if (!this.#tmpHead || this.#tmpHead.length !== headDim) {
      this.#tmpHead = new Float64Array(headDim);
    }
    if (!this.#tmpHead2 || this.#tmpHead2.length !== headDim) {
      this.#tmpHead2 = new Float64Array(headDim);
    }
    if (!this.#tmpHidden || this.#tmpHidden.length !== hidden) {
      this.#tmpHidden = new Float64Array(hidden);
    }
  }

  // cached std arrays (allocated lazily, reused)
  #inStdTmp: Float64Array | null = null;
  #outStdTmp: Float64Array | null = null;

  #tmpEnsureInStd(): Float64Array {
    if (!this.#inStdTmp || this.#inStdTmp.length !== this.#inputDim) {
      this.#inStdTmp = new Float64Array(this.#inputDim);
    }
    return this.#inStdTmp;
  }
  #tmpEnsureOutStd(): Float64Array {
    if (!this.#outStdTmp || this.#outStdTmp.length !== this.#outputDim) {
      this.#outStdTmp = new Float64Array(this.#outputDim);
    }
    return this.#outStdTmp;
  }

  // =====================================================================================
  // Forward pass
  // =====================================================================================

  #forward(
    seqLen: number,
    xNorm: Float64Array,
    yHatOut: Float64Array,
  ): { seqLen: number } {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;

    // 1) Input embedding: xEmb[t] = xNorm[t]*W_embed + b_embed
    this.#denseSeqForward(
      xNorm,
      seqLen,
      this.#inputDim,
      this.#W_embed!,
      embed,
      this.#b_embed!,
      this.#xEmb!,
    );

    // 2) Temporal conv per scale, + GELU, + posEnc + scaleEmb
    this.#temporalConvAllScales(seqLen);

    // 3) Fusion (gated sum over scales, upsampled to seqLen)
    this.#fusion(seqLen);

    // Save state[0] = fused
    const xEmbLen = seqLen * embed;
    const states = this.#states!;
    const state0Off = 0;
    for (let i = 0; i < xEmbLen; i++) states[state0Off + i] = this.#fused![i];

    // 4) Transformer blocks
    for (let b = 0; b < cfg.numBlocks; b++) {
      const inOff = b * xEmbLen;
      const outOff = (b + 1) * xEmbLen;
      this.#transformerBlockForward(b, seqLen, states, inOff, states, outOff);
    }

    // 5) Attention pooling -> aggregated
    const finalOff = cfg.numBlocks * xEmbLen;
    const H = states; // final sequence is at states[finalOff...]
    this.#poolingForward(seqLen, H, finalOff);

    // 6) Output head: yHat = agg * W_out + b_out
    this.#denseVecForward(
      this.#agg!,
      embed,
      this.#W_out!,
      this.#outputDim,
      this.#b_out!,
      yHatOut,
    );

    return { seqLen };
  }

  #buildPositionalEncoding(
    out: Float64Array,
    maxSeq: number,
    embed: number,
  ): void {
    // standard sinusoidal
    for (let pos = 0; pos < maxSeq; pos++) {
      for (let i = 0; i < (embed >> 1); i++) {
        const denom = Math.pow(10000, (2 * i) / embed);
        const angle = pos / denom;
        const base = pos * embed + (i << 1);
        out[base] = Math.sin(angle);
        out[base + 1] = Math.cos(angle);
      }
      if ((embed & 1) === 1) {
        // if odd embedDim, last element stays 0
        out[pos * embed + (embed - 1)] = 0;
      }
    }
  }

  #denseSeqForward(
    X: Float64Array,
    rows: number,
    inCols: number,
    W: Float64Array, // [inCols, outCols]
    outCols: number,
    b: Float64Array, // [outCols]
    out: Float64Array, // [rows, outCols]
  ): void {
    // out[r, c] = sum_f X[r,f]*W[f,c] + b[c]
    for (let r = 0; r < rows; r++) {
      const xOff = r * inCols;
      const oOff = r * outCols;
      for (let c = 0; c < outCols; c++) out[oOff + c] = b[c];
      for (let f = 0; f < inCols; f++) {
        const xv = X[xOff + f];
        const wOff = f * outCols;
        for (let c = 0; c < outCols; c++) {
          out[oOff + c] += xv * W[wOff + c];
        }
      }
    }
  }

  #denseVecForward(
    x: Float64Array,
    inCols: number,
    W: Float64Array, // [inCols, outCols]
    outCols: number,
    b: Float64Array, // [outCols]
    out: Float64Array, // [outCols]
  ): void {
    for (let c = 0; c < outCols; c++) out[c] = b[c];
    for (let f = 0; f < inCols; f++) {
      const xv = x[f];
      const wOff = f * outCols;
      for (let c = 0; c < outCols; c++) out[c] += xv * W[wOff + c];
    }
  }

  #temporalConvAllScales(seqLen: number): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const kSize = cfg.temporalKernelSize;
    const numScales = cfg.temporalScales.length;
    const xEmb = this.#xEmb!;
    const posEnc = this.#posEnc!;
    const scaleEmb = this.#scaleEmb!;

    for (let s = 0; s < numScales; s++) {
      const stride = cfg.temporalScales[s] | 0;
      const Ls = this.#convLen![s] | 0;
      const pre = this.#convPre[s];
      const post = this.#convPost[s];
      const Wc = this.#W_conv[s];

      // Causal conv with stride: out[t] uses indices (t*stride - (kSize-1) + k)
      // pre[t, dOut] = sum_k sum_dIn xEmb[idx, dIn] * Wc[k, dIn, dOut]
      // Wc layout: [k][dIn][dOut] contiguous (kSize*embed*embed)
      fillZero(pre);

      for (let t = 0; t < Ls; t++) {
        const center = t * stride;
        const outOff = t * embed;

        for (let k = 0; k < kSize; k++) {
          const idx = center - (kSize - 1) + k;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * embed;
          const wBase = k * embed * embed;
          // accumulate: pre[outOff + dOut] += sum_dIn xEmb[xOff+dIn]*Wc[wBase + dIn*embed + dOut]
          for (let dIn = 0; dIn < embed; dIn++) {
            const xv = xEmb[xOff + dIn];
            const wOff = wBase + dIn * embed;
            for (let dOut = 0; dOut < embed; dOut++) {
              pre[outOff + dOut] += xv * Wc[wOff + dOut];
            }
          }
        }
      }

      // GELU and add positional + scale emb
      const seOff = s * embed;
      for (let t = 0; t < Ls; t++) {
        const off = t * embed;
        const peOff = t * embed; // scale-specific positional uses same positions (shorter length)
        for (let d = 0; d < embed; d++) {
          const v = gelu(pre[off + d]) + posEnc[peOff + d] +
            scaleEmb[seOff + d];
          post[off + d] = v;
        }
      }
    }
  }

  #fusion(seqLen: number): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const numScales = cfg.temporalScales.length;
    const fused = this.#fused!;
    const gates = this.#gatesTmp!;
    const Wg = this.#W_gate!;
    const bg = this.#b_gate!;
    const postScales = this.#convPost;

    // fused[t] = sum_s gate_s(t) * E_s( floor(t/stride_s) )
    // gate_s(t) = sigmoid( dot(E_s, Wg_s) + bg_s )
    fillZero(fused);

    for (let t = 0; t < seqLen; t++) {
      const fOff = t * embed;

      for (let s = 0; s < numScales; s++) {
        const stride = cfg.temporalScales[s] | 0;
        const Ls = this.#convLen![s] | 0;
        let idx = (t / stride) | 0;
        if (idx < 0) idx = 0;
        if (idx >= Ls) idx = Ls - 1;

        const E = postScales[s];
        const eOff = idx * embed;
        const wgOff = s * embed;

        // z = dot(E, Wg_s) + bg
        let z = bg[s];
        for (let d = 0; d < embed; d++) z += E[eOff + d] * Wg[wgOff + d];
        const g = sigmoid(z);

        gates[t * numScales + s] = g;

        for (let d = 0; d < embed; d++) {
          fused[fOff + d] += g * E[eOff + d];
        }
      }

      // fusion dropout (if any) - inverted dropout
      const drop = cfg.fusionDropout;
      if (drop > 0) {
        const keep = 1 - drop;
        for (let d = 0; d < embed; d++) {
          const r = this.#rng();
          fused[fOff + d] = (r < keep) ? (fused[fOff + d] / keep) : 0;
        }
      }
    }
  }

  #transformerBlockForward(
    blockIndex: number,
    seqLen: number,
    inBuf: Float64Array,
    inOff: number,
    outBuf: Float64Array,
    outOff: number,
  ): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;

    const ln1m = this.#ln1_mean!;
    const ln1i = this.#ln1_invstd!;
    const ln2m = this.#ln2_mean!;
    const ln2i = this.#ln2_invstd!;

    // 1) LN1
    // ln1Out is written into attConcat temporarily (reuse buffer): [seqLen, embed]
    const lnOut = this.#attConcat!;
    this.#layerNormForwardSeq(
      inBuf,
      inOff,
      seqLen,
      embed,
      this.#ln1_gamma[blockIndex],
      this.#ln1_beta[blockIndex],
      lnOut,
      0,
      ln1m,
      blockIndex * seqLen,
      ln1i,
      blockIndex * seqLen,
    );

    // 2) MHA(ln1Out) -> attOut (reuse fusedGrad as temp)
    const attOut = this.#fusedGrad!;
    this.#mhaForward(blockIndex, seqLen, lnOut, 0, attOut, 0);

    // Residual: x2 = x + attOut => write to outBuf temp at outOff
    for (let t = 0; t < seqLen; t++) {
      const base = t * embed;
      const inB = inOff + base;
      const outB = outOff + base;
      for (let d = 0; d < embed; d++) {
        outBuf[outB + d] = inBuf[inB + d] + attOut[base + d];
      }
    }

    // 3) LN2 on x2 (in outBuf at outOff), store into lnOut again
    this.#layerNormForwardSeq(
      outBuf,
      outOff,
      seqLen,
      embed,
      this.#ln2_gamma[blockIndex],
      this.#ln2_beta[blockIndex],
      lnOut,
      0,
      ln2m,
      blockIndex * seqLen,
      ln2i,
      blockIndex * seqLen,
    );

    // 4) FFN(ln2Out) -> ffnOut (reuse attOut buffer)
    this.#ffnForward(blockIndex, seqLen, lnOut, 0, attOut, 0);

    // Residual: out = x2 + ffnOut => overwrite outBuf at outOff
    for (let t = 0; t < seqLen; t++) {
      const base = t * embed;
      const outB = outOff + base;
      for (let d = 0; d < embed; d++) outBuf[outB + d] += attOut[base + d];
    }
  }

  #layerNormForwardSeq(
    X: Float64Array,
    xOff: number,
    rows: number,
    cols: number,
    gamma: Float64Array,
    beta: Float64Array,
    out: Float64Array,
    outOff: number,
    meanBuf: Float64Array,
    meanOff: number,
    invStdBuf: Float64Array,
    invStdOff: number,
  ): void {
    const eps = this.#cfg.epsilon;
    for (let r = 0; r < rows; r++) {
      const xo = xOff + r * cols;
      const oo = outOff + r * cols;

      // mean
      let m = 0;
      for (let c = 0; c < cols; c++) m += X[xo + c];
      m /= cols;

      // var
      let v = 0;
      for (let c = 0; c < cols; c++) {
        const d = X[xo + c] - m;
        v += d * d;
      }
      v /= cols;

      const inv = safeInvSqrt(v, eps);

      meanBuf[meanOff + r] = m;
      invStdBuf[invStdOff + r] = inv;

      for (let c = 0; c < cols; c++) {
        const xhat = (X[xo + c] - m) * inv;
        out[oo + c] = xhat * gamma[c] + beta[c];
      }
    }
  }

  #mhaForward(
    blockIndex: number,
    seqLen: number,
    X: Float64Array,
    xOff: number, // [seqLen, embed]
    out: Float64Array,
    outOff: number, // [seqLen, embed]
  ): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const scale = 1 / Math.sqrt(headDim);

    const Wq = this.#att_Wq[blockIndex];
    const Wk = this.#att_Wk[blockIndex];
    const Wv = this.#att_Wv[blockIndex];
    const Wo = this.#att_Wo[blockIndex];
    const bo = this.#att_bo[blockIndex];

    const q = this.#q!;
    const k = this.#k!;
    const v = this.#v!;
    const scores = this.#scores!;
    const headOut = this.#headOut!;
    const concat = this.#attConcat!;

    // concat buffer accumulates per head outputs into [seqLen, embed]
    fillZero(concat);

    for (let h = 0; h < heads; h++) {
      // Compute Q,K,V for this head into q,k,v (each [seqLen, headDim])
      // Wq layout is [h][embed][headDim] contiguous, i.e. offset h*embed*headDim
      const wqOff = h * embed * headDim;
      const wkOff = h * embed * headDim;
      const wvOff = h * embed * headDim;

      for (let t = 0; t < seqLen; t++) {
        const xBase = xOff + t * embed;
        const qBase = t * headDim;
        // init
        for (let d = 0; d < headDim; d++) {
          q[qBase + d] = 0;
          k[qBase + d] = 0;
          v[qBase + d] = 0;
        }
        // dot over embed
        for (let e = 0; e < embed; e++) {
          const xv = X[xBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            q[qBase + d] += xv * Wq[oq + d];
            k[qBase + d] += xv * Wk[ok + d];
            v[qBase + d] += xv * Wv[ov + d];
          }
        }
      }

      // scores[i,j] = (q_i dot k_j) * scale (+ mask)
      // compute row-wise with stable softmax in-place (scores becomes probs)
      for (let i = 0; i < seqLen; i++) {
        const iOff = i * headDim;
        const rowOff = i * seqLen;

        let max = NEG_LARGE;
        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = NEG_LARGE;
            continue;
          }
          const jOff = j * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[iOff + d] * k[jOff + d];
          const s = dot * scale;
          scores[rowOff + j] = s;
          if (s > max) max = s;
        }

        // softmax
        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const sv = scores[rowOff + j] - max;
          const ev = (sv <= -60) ? 0 : Math.exp(sv);
          scores[rowOff + j] = ev;
          sum += ev;
        }
        const invSum = sum > 0 ? (1 / sum) : 0;
        for (let j = 0; j < seqLen; j++) scores[rowOff + j] *= invSum;
      }

      // headOut[i] = sum_j scores[i,j] * v[j]
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const oOff = i * headDim;
        for (let d = 0; d < headDim; d++) headOut[oOff + d] = 0;

        for (let j = 0; j < seqLen; j++) {
          const w = scores[rowOff + j];
          if (w === 0) continue;
          const vOff = j * headDim;
          for (let d = 0; d < headDim; d++) {
            headOut[oOff + d] += w * v[vOff + d];
          }
        }
      }

      // write into concat at head slice
      for (let t = 0; t < seqLen; t++) {
        const src = t * headDim;
        const dst = t * embed + h * headDim;
        for (let d = 0; d < headDim; d++) concat[dst + d] = headOut[src + d];
      }
    }

    // Output projection: out = concat * Wo + bo
    for (let t = 0; t < seqLen; t++) {
      const cOff = t * embed;
      const oOff = outOff + t * embed;
      // init with bias
      for (let d = 0; d < embed; d++) out[oOff + d] = bo[d];
      for (let e = 0; e < embed; e++) {
        const xv = concat[cOff + e];
        const wBase = e * embed;
        for (let d = 0; d < embed; d++) out[oOff + d] += xv * Wo[wBase + d];
      }
      // attention dropout (optional): apply to out activations
      const drop = cfg.attentionDropout;
      if (drop > 0) {
        const keep = 1 - drop;
        for (let d = 0; d < embed; d++) {
          const r = this.#rng();
          out[oOff + d] = (r < keep) ? (out[oOff + d] / keep) : 0;
        }
      }
    }
  }

  #ffnForward(
    blockIndex: number,
    seqLen: number,
    X: Float64Array,
    xOff: number, // [seqLen, embed]
    out: Float64Array,
    outOff: number, // [seqLen, embed]
  ): void {
    const embed = this.#cfg.embeddingDim;
    const hidden = (embed * this.#cfg.ffnMultiplier) | 0;

    const W1 = this.#ffn_W1[blockIndex];
    const b1 = this.#ffn_b1[blockIndex];
    const W2 = this.#ffn_W2[blockIndex];
    const b2 = this.#ffn_b2[blockIndex];

    const hBuf = this.#ffnHidden!;
    // Compute hidden pre-act into hBuf, then GELU in-place, then out
    for (let t = 0; t < seqLen; t++) {
      const xBase = xOff + t * embed;
      const hBase = t * hidden;
      // init with bias
      for (let j = 0; j < hidden; j++) hBuf[hBase + j] = b1[j];

      for (let e = 0; e < embed; e++) {
        const xv = X[xBase + e];
        const wBase = e * hidden;
        for (let j = 0; j < hidden; j++) hBuf[hBase + j] += xv * W1[wBase + j];
      }

      // GELU
      for (let j = 0; j < hidden; j++) hBuf[hBase + j] = gelu(hBuf[hBase + j]);

      // out init with b2
      const oBase = outOff + t * embed;
      for (let d = 0; d < embed; d++) out[oBase + d] = b2[d];

      for (let j = 0; j < hidden; j++) {
        const hv = hBuf[hBase + j];
        const w2Base = j * embed;
        for (let d = 0; d < embed; d++) out[oBase + d] += hv * W2[w2Base + d];
      }
    }
  }

  #poolingForward(seqLen: number, H: Float64Array, hOff: number): void {
    const embed = this.#cfg.embeddingDim;
    const Wp = this.#W_pool!;
    const bp = this.#b_pool![0];
    const logits = this.#poolLogits!;
    const probs = this.#poolProbs!;
    const agg = this.#agg!;

    // logits[i] = dot(H_i, Wp) + bp
    let max = NEG_LARGE;
    for (let i = 0; i < seqLen; i++) {
      const off = hOff + i * embed;
      let z = bp;
      for (let d = 0; d < embed; d++) z += H[off + d] * Wp[d];
      logits[i] = z;
      if (z > max) max = z;
    }

    // softmax
    let sum = 0;
    for (let i = 0; i < seqLen; i++) {
      const v = logits[i] - max;
      const e = (v <= -60) ? 0 : Math.exp(v);
      probs[i] = e;
      sum += e;
    }
    const inv = sum > 0 ? (1 / sum) : 0;
    for (let i = 0; i < seqLen; i++) probs[i] *= inv;

    // agg = sum_i probs[i] * H_i
    fillZero(agg);
    for (let i = 0; i < seqLen; i++) {
      const p = probs[i];
      if (p === 0) continue;
      const off = hOff + i * embed;
      for (let d = 0; d < embed; d++) agg[d] += p * H[off + d];
    }
  }

  // =====================================================================================
  // Backward + Adam update
  // =====================================================================================

  #backwardAndUpdate(
    seqLen: number,
    xNorm: Float64Array,
    yNormTarget: Float64Array,
    yHat: Float64Array,
    ctx: { seqLen: number },
    sampleWeight: number,
  ): number {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const hidden = (embed * cfg.ffnMultiplier) | 0;
    const numBlocks = cfg.numBlocks;
    const numScales = cfg.temporalScales.length;
    const eps = cfg.epsilon;

    // Clear grads
    for (let p = 0; p < this.#params.length; p++) fillZero(this.#params[p].g);

    // Compute dyHat (normalized): (yHat - yTarget)/nOut scaled by sampleWeight
    const nOut = this.#outputDim;
    const dyHat = this.#tmpEnsureOutGrad();
    fillZero(dyHat);
    for (let j = 0; j < nOut; j++) {
      dyHat[j] = (yHat[j] - yNormTarget[j]) * (sampleWeight / nOut);
    }

    // --- Backprop output head ---
    // yHat = agg * W_out + b_out
    // dW_out += agg^T * dyHat; db_out += dyHat; dAgg = dyHat * W_out^T
    const agg = this.#agg!;
    const dAgg = this.#aggGrad!;
    fillZero(dAgg);

    this.#accGradMatrix(this.#findParam("W_out")!, agg, dyHat, embed, nOut); // dW_out
    this.#accGradVector(this.#findParam("b_out")!, dyHat); // db_out

    const Wout = this.#W_out!;
    for (let e = 0; e < embed; e++) {
      let s = 0;
      const wOff = e * nOut;
      for (let j = 0; j < nOut; j++) s += dyHat[j] * Wout[wOff + j];
      dAgg[e] = s;
    }

    // --- Backprop pooling ---
    // aggregated = Σ_i p_i * H_i
    // dH_i += p_i*dAgg + dlogit_i * W_pool
    // dlogit_i = p_i * dot(dAgg, H_i - aggregated)
    // dW_pool += Σ_i dlogit_i * H_i; db_pool += Σ_i dlogit_i
    const probs = this.#poolProbs!;
    const states = this.#states!;
    const statesGrad = this.#statesGrad!;
    const xEmbLen = seqLen * embed;
    const finalOff = numBlocks * xEmbLen;

    // init dH (final) = 0
    const dHOff = finalOff;
    for (let i = 0; i < xEmbLen; i++) statesGrad[dHOff + i] = 0;

    const Wp = this.#W_pool!;
    const dWp = this.#findParam("W_pool")!;
    const dbp = this.#findParam("b_pool")!;
    let dbpAcc = 0;

    for (let i = 0; i < seqLen; i++) {
      const p = probs[i];
      const hOff = finalOff + i * embed;
      const dOff = dHOff + i * embed;

      // dH_i += p * dAgg
      for (let d = 0; d < embed; d++) statesGrad[dOff + d] += p * dAgg[d];

      // dot(dAgg, H_i - agg)
      let dot = 0;
      for (let d = 0; d < embed; d++) {
        dot += dAgg[d] * (states[hOff + d] - agg[d]);
      }

      const dlogit = p * dot;

      // dH_i += dlogit * Wp
      for (let d = 0; d < embed; d++) statesGrad[dOff + d] += dlogit * Wp[d];

      // dW_pool += dlogit * H_i
      for (let d = 0; d < embed; d++) dWp.g[d] += dlogit * states[hOff + d];
      dbpAcc += dlogit;
    }
    dbp.g[0] += dbpAcc;

    // --- Backprop transformer blocks ---
    for (let b = numBlocks - 1; b >= 0; b--) {
      const inOff = b * xEmbLen;
      const outOff = (b + 1) * xEmbLen;

      // dOut is statesGrad at outOff, write dIn to statesGrad at inOff (accumulate)
      // We'll compute dIn in-place in statesGrad[inOff..] by first zeroing, then accumulating.
      for (let i = 0; i < xEmbLen; i++) statesGrad[inOff + i] = 0;

      this.#transformerBlockBackward(
        b,
        seqLen,
        states,
        inOff,
        states,
        outOff,
        statesGrad,
        inOff,
        statesGrad,
        outOff,
      );
    }

    // dFused is statesGrad at state0
    const dFused = this.#fusedGrad!;
    for (let i = 0; i < xEmbLen; i++) dFused[i] = statesGrad[i];

    // --- Backprop fusion to convPost (and gate weights) ---
    // This computes gradients into convPost buffers and gate params, then backprop conv to xEmb.
    const dConvPost: Float64Array[] = [];
    for (let s = 0; s < numScales; s++) {
      const Ls = this.#convLen![s] | 0;
      const buf = this.#convPre[s]; // reuse convPre as gradient buffer for post
      fillZero(buf);
      dConvPost[s] = buf;
    }

    const gates = this.#gatesTmp!;
    const Wg = this.#W_gate!;
    const bg = this.#b_gate!;
    const pWg = this.#findParam("W_gate")!;
    const pbg = this.#findParam("b_gate")!;

    for (let t = 0; t < seqLen; t++) {
      const fOff = t * embed;

      for (let s = 0; s < numScales; s++) {
        const stride = cfg.temporalScales[s] | 0;
        const Ls = this.#convLen![s] | 0;
        let idx = (t / stride) | 0;
        if (idx < 0) idx = 0;
        if (idx >= Ls) idx = Ls - 1;

        const E = this.#convPost[s];
        const eOff = idx * embed;
        const wgOff = s * embed;

        // recompute gate (or use stored)
        const g = gates[t * numScales + s];

        // dE from fused contribution: g * dFused
        // also gate gradient: dgate = dot(dFused, E)
        let dgate = 0;
        for (let d = 0; d < embed; d++) dgate += dFused[fOff + d] * E[eOff + d];
        const dz = dgate * g * (1 - g);

        // grad gate params
        for (let d = 0; d < embed; d++) pWg.g[wgOff + d] += dz * E[eOff + d];
        pbg.g[s] += dz;

        // dE += g*dFused + dz*Wg
        const dE = dConvPost[s];
        for (let d = 0; d < embed; d++) {
          dE[eOff + d] += g * dFused[fOff + d] + dz * Wg[wgOff + d];
        }
      }
    }

    // --- Backprop convPost -> convPre (GELU derivative) and scaleEmb gradients ---
    const pScaleEmb = this.#findParam("scaleEmb")!;
    const posEnc = this.#posEnc!;
    const scaleEmb = this.#scaleEmb!;
    const kSize = cfg.temporalKernelSize;

    for (let s = 0; s < numScales; s++) {
      const Ls = this.#convLen![s] | 0;
      const pre = this.#convPre[s]; // holds pre-activations from forward (we overwrote? careful)
      const post = this.#convPost[s];
      const dPost = dConvPost[s];
      const seOff = s * embed;

      // We need pre-activations to compute GELU derivative, but pre buffer was overwritten with dPost.
      // To avoid storing huge extra buffers, we recompute pre-activations on-the-fly using xEmb and conv weights.
      // We'll compute dPre in-place into dPost buffer (reuse) by multiplying with GELU'(pre).
      const dPre = dPost;

      // First compute pre-activation again into tmp (reuse pre buffer this time) then compute dPre = dPost * gelu'(pre)
      // We'll compute directly into pre (as recomputed pre), then multiply.
      fillZero(pre);
      const stride = cfg.temporalScales[s] | 0;
      const Wc = this.#W_conv[s];
      const xEmb = this.#xEmb!;

      for (let t = 0; t < Ls; t++) {
        const center = t * stride;
        const outOff = t * embed;

        for (let k = 0; k < kSize; k++) {
          const idx = center - (kSize - 1) + k;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * embed;
          const wBase = k * embed * embed;
          for (let dIn = 0; dIn < embed; dIn++) {
            const xv = xEmb[xOff + dIn];
            const wOff = wBase + dIn * embed;
            for (let dOut = 0; dOut < embed; dOut++) {
              pre[outOff + dOut] += xv * Wc[wOff + dOut];
            }
          }
        }
      }

      // scaleEmb gradients and dPre
      for (let t = 0; t < Ls; t++) {
        const off = t * embed;
        for (let d = 0; d < embed; d++) {
          // post = gelu(pre) + posEnc + scaleEmb => dScaleEmb += dPost
          pScaleEmb.g[seOff + d] += dPost[off + d];
          // dPre = dPost * gelu'(pre)
          dPre[off + d] = dPost[off + d] * geluDeriv(pre[off + d]);
        }
      }

      // --- Backprop conv to W_conv[s] and xEmb ---
      const pWc = this.#findParam(`W_conv_${s}`)!;
      const dXEmb = this.#xEmbGrad!;
      // clear xEmbGrad
      fillZero(dXEmb);

      for (let t = 0; t < Ls; t++) {
        const center = t * stride;
        const dOutOff = t * embed;

        for (let k = 0; k < kSize; k++) {
          const idx = center - (kSize - 1) + k;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * embed;
          const wBase = k * embed * embed;

          // dWc += xEmb[idx]^T * dPre[t]
          for (let dIn = 0; dIn < embed; dIn++) {
            const xv = xEmb[xOff + dIn];
            const wOff = wBase + dIn * embed;
            for (let dOut = 0; dOut < embed; dOut++) {
              pWc.g[wOff + dOut] += xv * dPre[dOutOff + dOut];
            }
          }

          // dXEmb[idx] += dPre[t] * Wc^T
          for (let dIn = 0; dIn < embed; dIn++) {
            const wOff = wBase + dIn * embed;
            let acc = 0;
            for (let dOut = 0; dOut < embed; dOut++) {
              acc += dPre[dOutOff + dOut] * Wc[wOff + dOut];
            }
            dXEmb[xOff + dIn] += acc;
          }
        }
      }

      // Accumulate dXEmb from all scales into xEmbGrad (already)
      // Note: for multiple scales we overwrote dXEmb each time; we need sum across scales.
      // We'll add scale contribution to a global accumulator (xEmbGradGlobal) instead.
      // To do this without extra allocations, we accumulate into xEmbGradGlobal buffer itself.
      // Here, we already used xEmbGrad as temp; but we'd lose contributions from previous scales.
      // Fix: maintain xEmbGradGlobal separately (reuse fusedGrad as global accumulator).
      // We'll handle after loop by summing per-scale into a dedicated buffer.
    }

    // The loop above cleared xEmbGrad for each scale. Let's compute global xEmbGrad properly:
    // Recompute and accumulate per scale, but this time directly into xEmbGradGlobal (xEmbGrad).
    const xEmbGradGlobal = this.#xEmbGrad!;
    fillZero(xEmbGradGlobal);

    // We'll redo the conv backprop accumulation into xEmbGradGlobal but skip dW and scaleEmb/gates (already computed),
    // by using computed dPre buffers (which are dConvPost[s]) and existing Wc.
    for (let s = 0; s < numScales; s++) {
      const stride = cfg.temporalScales[s] | 0;
      const Ls = this.#convLen![s] | 0;
      const dPre = dConvPost[s];
      const Wc = this.#W_conv[s];
      const xEmb = this.#xEmb!;

      for (let t = 0; t < Ls; t++) {
        const center = t * stride;
        const dOutOff = t * embed;

        for (let k = 0; k < kSize; k++) {
          const idx = center - (kSize - 1) + k;
          if (idx < 0 || idx >= seqLen) continue;
          const xOff = idx * embed;
          const wBase = k * embed * embed;

          for (let dIn = 0; dIn < embed; dIn++) {
            const wOff = wBase + dIn * embed;
            let acc = 0;
            for (let dOut = 0; dOut < embed; dOut++) {
              acc += dPre[dOutOff + dOut] * Wc[wOff + dOut];
            }
            xEmbGradGlobal[xOff + dIn] += acc;
          }
        }
      }
    }

    // --- Backprop input embedding: xEmb = xNorm * W_embed + b_embed ---
    const pWe = this.#findParam("W_embed")!;
    const pbe = this.#findParam("b_embed")!;
    // dW_embed += xNorm^T * xEmbGradGlobal; db_embed += sum_t dXEmb
    for (let t = 0; t < seqLen; t++) {
      const xOff = t * this.#inputDim;
      const dOff = t * embed;

      // bias grad
      for (let d = 0; d < embed; d++) pbe.g[d] += xEmbGradGlobal[dOff + d];

      for (let f = 0; f < this.#inputDim; f++) {
        const xv = xNorm[xOff + f];
        const wOff = f * embed;
        for (let d = 0; d < embed; d++) {
          pWe.g[wOff + d] += xv * xEmbGradGlobal[dOff + d];
        }
      }
    }

    // --- Compute gradient norm and clip if needed ---
    let g2 = 0;
    for (let p = 0; p < this.#params.length; p++) {
      const g = this.#params[p].g;
      for (let i = 0; i < g.length; i++) g2 += g[i] * g[i];
    }
    let gNorm = Math.sqrt(g2);

    const maxG = cfg.maxGradNorm;
    let clipScale = 1.0;
    if (maxG > 0 && gNorm > maxG) {
      clipScale = maxG / (gNorm + 1e-12);
      for (let p = 0; p < this.#params.length; p++) {
        const g = this.#params[p].g;
        for (let i = 0; i < g.length; i++) g[i] *= clipScale;
      }
      gNorm *= clipScale;
    }

    // --- Adam update with cosine warmup schedule ---
    this.#updateCount++;
    const lr = this.#lrSchedule(this.#updateCount);
    this.#effectiveLR = lr;

    const b1 = cfg.beta1;
    const b2 = cfg.beta2;
    const epsAdam = cfg.epsilon;
    const regStrength = cfg.regularizationStrength;

    const b1t = 1 - Math.pow(b1, this.#updateCount);
    const b2t = 1 - Math.pow(b2, this.#updateCount);

    for (let p = 0; p < this.#params.length; p++) {
      const pr = this.#params[p];
      const w = pr.w;
      const m = pr.m;
      const v = pr.v;
      const g = pr.g;

      for (let i = 0; i < w.length; i++) {
        // L2 regularization gradient
        let gi = g[i];
        if (regStrength > 0 && pr.l2) gi += regStrength * w[i];

        const mi = m[i] = b1 * m[i] + (1 - b1) * gi;
        const vi = v[i] = b2 * v[i] + (1 - b2) * (gi * gi);

        const mHat = mi / b1t;
        const vHat = vi / b2t;

        w[i] -= lr * mHat / (Math.sqrt(vHat) + epsAdam);
      }
    }

    return gNorm;
  }

  #transformerBlockBackward(
    blockIndex: number,
    seqLen: number,
    states: Float64Array,
    inOff: number,
    statesOut: Float64Array,
    outOff: number,
    dStatesIn: Float64Array,
    dInOff: number,
    dStatesOut: Float64Array,
    dOutOff: number,
  ): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const hidden = (embed * cfg.ffnMultiplier) | 0;

    // Naming:
    // x  = states[inOff]
    // x2 = x + attn(ln1(x))  (stored in statesOut[outOff] before adding ffn in forward? Actually forward overwrote with x2+ffn.)
    // out = x2 + ffn(ln2(x2)) (stored in statesOut[outOff])

    // We need reconstruct x2 = out - ffnOut, but ffnOut not stored.
    // We'll recompute forward sub-steps for this block using x and produce:
    // ln1(x), attnOut, x2, ln2(x2), ffnOut.
    const xEmbLen = seqLen * embed;

    const x = states;
    const out = statesOut;

    // Buffers:
    const lnOut = this.#attConcat!; // [seqLen, embed]
    const attOut = this.#fusedGrad!; // [seqLen, embed]
    const ffnOut = this.#fused!; // reuse fused as temp ffnOut [seqLen, embed]
    const x2buf = this.#fusedGrad!; // attention output is already in attOut; we will create x2 in tmp stateGrad maybe
    const dOut = dStatesOut; // [seqLen, embed] at dOutOff
    const dX2 = this.#fusedGrad!; // reuse attention buffer after we're done with attention? We'll use tmpVecE2 as per element.

    // 1) Recompute LN1(x)
    this.#layerNormForwardSeq(
      x,
      inOff,
      seqLen,
      embed,
      this.#ln1_gamma[blockIndex],
      this.#ln1_beta[blockIndex],
      lnOut,
      0,
      this.#ln1_mean!,
      blockIndex * seqLen,
      this.#ln1_invstd!,
      blockIndex * seqLen,
    );

    // 2) Recompute attnOut = MHA(ln1Out)
    this.#mhaForward(blockIndex, seqLen, lnOut, 0, attOut, 0);

    // 3) Recompute x2 = x + attOut into ffnOut buffer temporarily (overwrite ffnOut)
    for (let t = 0; t < seqLen; t++) {
      const base = t * embed;
      const xo = inOff + base;
      for (let d = 0; d < embed; d++) {
        ffnOut[base + d] = x[xo + d] + attOut[base + d];
      }
    }

    // 4) Recompute LN2(x2)
    this.#layerNormForwardSeq(
      ffnOut,
      0,
      seqLen,
      embed,
      this.#ln2_gamma[blockIndex],
      this.#ln2_beta[blockIndex],
      lnOut,
      0,
      this.#ln2_mean!,
      blockIndex * seqLen,
      this.#ln2_invstd!,
      blockIndex * seqLen,
    );

    // 5) Recompute ffnOut = FFN(ln2Out)
    this.#ffnForward(blockIndex, seqLen, lnOut, 0, attOut, 0);
    // now attOut holds ffnOut

    // Backward:
    // out = x2 + ffnOut
    // dX2 += dOut, dFFNOut = dOut
    // x2 = x + attnOut
    // dX += dX2, dAttnOut = dX2

    // dFFNOut (seqLen, embed) is dOut
    // FFN backward -> dLN2, grads W1,b1,W2,b2
    const dLN2 = this.#statesGrad!; // reuse global buffer as temp? risky. We'll use statesGrad segment for this block in place:
    // We'll use dStatesIn slice at dInOff as dX accumulator, and use lnOut buffer for dLN2 in-place.
    // We'll compute dLN2 into lnOut buffer (overwrite), since lnOut no longer needed.
    fillZero(lnOut);

    this.#ffnBackward(
      blockIndex,
      seqLen,
      lnOut,
      0,
      attOut,
      0,
      dOut,
      dOutOff,
      lnOut,
      0,
    );

    // Now lnOut is dLN2 (grad wrt LN2 output), and attOut is ffnOut (forward) but overwritten? In ffnBackward, we treat hidden buf in class.

    // LayerNorm2 backward: x2 in ffnOut buffer (0..), mean/invStd cached; produces dX2_fromLN into attConcat? We'll use attConcat as dX2 buffer.
    const dX2buf = this.#attConcat!;
    fillZero(dX2buf);

    this.#layerNormBackwardSeq(
      ffnOut,
      0,
      seqLen,
      embed,
      this.#ln2_gamma[blockIndex],
      lnOut,
      0, // dY (grad wrt LN out)
      dX2buf,
      0, // dX output
      this.#findParam(`ln2_gamma_${blockIndex}`)!,
      this.#findParam(`ln2_beta_${blockIndex}`)!,
      this.#ln2_mean!,
      blockIndex * seqLen,
      this.#ln2_invstd!,
      blockIndex * seqLen,
    );

    // dX2 total = dOut + dX2_fromLN
    // We'll store dX2 total into lnOut (reuse)
    fillZero(lnOut);
    for (let t = 0; t < seqLen; t++) {
      const base = t * embed;
      const doff = dOutOff + base;
      for (let d = 0; d < embed; d++) {
        lnOut[base + d] = dOut[doff + d] + dX2buf[base + d];
      }
    }

    // Attention residual: x2 = x + attnOut
    // dX += dX2, dAttnOut = dX2
    // So dAttnOut is lnOut, and dX starts with lnOut (accumulate into dStatesIn[dInOff..])
    for (let i = 0; i < xEmbLen; i++) dStatesIn[dInOff + i] += lnOut[i];

    // Backprop attention: attnOut = MHA(LN1(x))
    // Need dLN1 from attention backward, then LN1 backward to x.
    // We'll compute dLN1 into attOut buffer (reuse).
    fillZero(attOut);
    this.#mhaBackward(
      blockIndex,
      seqLen,
      lnOut,
      0,
      lnOut,
      0,
      attOut,
      0,
      x,
      inOff,
    );

    // LayerNorm1 backward: x is states[inOff], grad wrt LN out is attOut, produce dX_add
    const dXadd = this.#fusedGrad!; // reuse
    fillZero(dXadd);

    this.#layerNormBackwardSeq(
      x,
      inOff,
      seqLen,
      embed,
      this.#ln1_gamma[blockIndex],
      attOut,
      0, // dY
      dXadd,
      0,
      this.#findParam(`ln1_gamma_${blockIndex}`)!,
      this.#findParam(`ln1_beta_${blockIndex}`)!,
      this.#ln1_mean!,
      blockIndex * seqLen,
      this.#ln1_invstd!,
      blockIndex * seqLen,
    );

    // Accumulate into dStatesIn
    for (let i = 0; i < xEmbLen; i++) dStatesIn[dInOff + i] += dXadd[i];
  }

  #ffnBackward(
    blockIndex: number,
    seqLen: number,
    // ln2Out (input to W1) is NOT passed because we recompute it into class buffer in forward;
    // We'll recompute ln2Out here as needed before calling this function, or pass it in.
    ln2Out: Float64Array,
    ln2Off: number, // (not used: placeholder)
    ffnOut: Float64Array,
    ffnOff: number, // forward output (not used)
    dFFNOut: Float64Array,
    dOff: number, // gradient wrt ffn output
    dLN2Out: Float64Array,
    dLN2Off: number, // output: gradient wrt ln2 output
  ): void {
    // We recompute inside using the stored hidden activations from the last forward (ffnHidden).
    // However, ffnHidden is overwritten per forward call and is valid now.
    const embed = this.#cfg.embeddingDim;
    const hidden = (embed * this.#cfg.ffnMultiplier) | 0;

    const W1 = this.#ffn_W1[blockIndex];
    const b1 = this.#ffn_b1[blockIndex];
    const W2 = this.#ffn_W2[blockIndex];
    const b2 = this.#ffn_b2[blockIndex];

    const pW1 = this.#findParam(`ffn_W1_${blockIndex}`)!;
    const pb1 = this.#findParam(`ffn_b1_${blockIndex}`)!;
    const pW2 = this.#findParam(`ffn_W2_${blockIndex}`)!;
    const pb2 = this.#findParam(`ffn_b2_${blockIndex}`)!;

    const h = this.#ffnHidden!; // activated hidden (after GELU) from forward
    const dH = this.#ffnHiddenGrad!;

    // db2 and dW2, and dH = dOut * W2^T
    fillZero(dH);

    for (let t = 0; t < seqLen; t++) {
      const doBase = dOff + t * embed;
      const hBase = t * hidden;

      // db2
      for (let d = 0; d < embed; d++) pb2.g[d] += dFFNOut[doBase + d];

      // dW2 += h^T * dOut
      for (let j = 0; j < hidden; j++) {
        const hv = h[hBase + j];
        const w2Base = j * embed;
        for (let d = 0; d < embed; d++) {
          pW2.g[w2Base + d] += hv * dFFNOut[doBase + d];
        }
      }

      // dH = dOut * W2^T
      for (let j = 0; j < hidden; j++) {
        const w2Base = j * embed;
        let acc = 0;
        for (let d = 0; d < embed; d++) {
          acc += dFFNOut[doBase + d] * W2[w2Base + d];
        }
        dH[hBase + j] = acc;
      }
    }

    // Backprop through GELU: need pre-activation z1 to compute derivative. We don't store z1;
    // Recompute z1 by inverting GELU is not feasible. Instead, recompute z1 directly from LN2 output:
    // BUT we didn't pass ln2Out. We'll recompute ln2Out on-the-fly before calling ffnForward in transformerBlockBackward.
    // Here, we approximate derivative using the activated value h via a smooth surrogate:
    // use gelu'(z) where z approx via softplus inverse? Not stable.
    //
    // To satisfy "full backprop" in a practical way without storing z1, we recompute z1 from ln2Out:
    // We therefore require that ln2Out is in attConcat buffer when ffnForward was last called in transformerBlockBackward.
    // Specifically, transformerBlockBackward computed ln2Out into lnOut buffer (attConcat), then ffnForward overwrote ffnHidden.
    // We'll recompute z1 now from that same lnOut buffer by running the affine again.
    //
    // For safety, we recompute z1 into tmpHidden.
    const lnBuf = this.#attConcat!; // ln2Out used in the recompute step in transformerBlockBackward
    const tmp = this.#tmpHidden!;

    // dZ1 = dH * gelu'(Z1), and compute gradients for W1,b1 and dLN2Out = dZ1 * W1^T
    fillZero(dLN2Out);

    for (let t = 0; t < seqLen; t++) {
      const lnBase = t * embed;
      const hBase = t * hidden;

      // recompute Z1 into tmp
      for (let j = 0; j < hidden; j++) tmp[j] = b1[j];
      for (let e = 0; e < embed; e++) {
        const xv = lnBuf[lnBase + e];
        const w1Base = e * hidden;
        for (let j = 0; j < hidden; j++) tmp[j] += xv * W1[w1Base + j];
      }

      // dZ1 = dH * gelu'(Z1)
      for (let j = 0; j < hidden; j++) {
        const dz = dH[hBase + j] * geluDeriv(tmp[j]);
        dH[hBase + j] = dz; // reuse dH as dZ1
        pb1.g[j] += dz;
      }

      // dW1 += ln^T * dZ1
      for (let e = 0; e < embed; e++) {
        const xv = lnBuf[lnBase + e];
        const w1Base = e * hidden;
        for (let j = 0; j < hidden; j++) {
          pW1.g[w1Base + j] += xv * dH[hBase + j];
        }
      }

      // dLN2Out += dZ1 * W1^T
      const dlnBase = dLN2Off + t * embed;
      for (let e = 0; e < embed; e++) {
        const w1Base = e * hidden;
        let acc = 0;
        for (let j = 0; j < hidden; j++) acc += dH[hBase + j] * W1[w1Base + j];
        dLN2Out[dlnBase + e] += acc;
      }
    }
  }

  #layerNormBackwardSeq(
    X: Float64Array,
    xOff: number,
    rows: number,
    cols: number,
    gamma: Float64Array,
    dY: Float64Array,
    dYOff: number,
    dX: Float64Array,
    dXOff: number,
    pGamma: ParamRef,
    pBeta: ParamRef,
    meanBuf: Float64Array,
    meanOff: number,
    invStdBuf: Float64Array,
    invStdOff: number,
  ): void {
    // For each row:
    // dBeta += Σ dY
    // dGamma += Σ dY * xhat
    // dx = (1/N) * invStd * (N*dxhat - Σ dxhat - xhat*Σ(dxhat*xhat))
    // where dxhat = dY*gamma, xhat = (x-mean)*invStd
    for (let r = 0; r < rows; r++) {
      const xo = xOff + r * cols;
      const yo = dYOff + r * cols;
      const dxo = dXOff + r * cols;

      const mean = meanBuf[meanOff + r];
      const inv = invStdBuf[invStdOff + r];

      // First pass: compute dxhat, sums
      let sumDxhat = 0;
      let sumDxhatXhat = 0;

      for (let c = 0; c < cols; c++) {
        const xhat = (X[xo + c] - mean) * inv;
        const dy = dY[yo + c];

        pBeta.g[c] += dy;
        pGamma.g[c] += dy * xhat;

        const dxhat = dy * gamma[c];
        // store dxhat temporarily into dX (reuse)
        dX[dxo + c] = dxhat;

        sumDxhat += dxhat;
        sumDxhatXhat += dxhat * xhat;
      }

      const invN = 1 / cols;

      // Second pass: dx
      for (let c = 0; c < cols; c++) {
        const xhat = (X[xo + c] - mean) * inv;
        const dxhat = dX[dxo + c];
        const dx = inv * invN * (cols * dxhat - sumDxhat - xhat * sumDxhatXhat);
        dX[dxo + c] = dx;
      }
    }
  }

  #mhaBackward(
    blockIndex: number,
    seqLen: number,
    dAttnOut: Float64Array,
    dAttnOff: number, // dOut from attention residual (seqLen, embed)
    // scratch: we reuse lnOut buffers and internal q,k,v,scores/headOut
    // outputs:
    dLN1Out: Float64Array,
    dLN1Off: number, // gradient wrt LN1 output (seqLen, embed)
    // temp:
    dAttConcat: Float64Array,
    dAttConcatOff: number, // gradient wrt concat (seqLen, embed)
    // input x for recomputing LN1 (passed so we can recompute ln1Out and QKV)
    x: Float64Array,
    xOff: number,
  ): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const scale = 1 / Math.sqrt(headDim);

    const Wq = this.#att_Wq[blockIndex];
    const Wk = this.#att_Wk[blockIndex];
    const Wv = this.#att_Wv[blockIndex];
    const Wo = this.#att_Wo[blockIndex];

    const pWq = this.#findParam(`att_Wq_${blockIndex}`)!;
    const pWk = this.#findParam(`att_Wk_${blockIndex}`)!;
    const pWv = this.#findParam(`att_Wv_${blockIndex}`)!;
    const pWo = this.#findParam(`att_Wo_${blockIndex}`)!;
    const pbo = this.#findParam(`att_bo_${blockIndex}`)!;

    const q = this.#q!;
    const k = this.#k!;
    const v = this.#v!;
    const scores = this.#scores!;
    const headOut = this.#headOut!;
    const concat = this.#attConcat!;

    // Recompute LN1(x) into concat buffer (reuse)
    this.#layerNormForwardSeq(
      x,
      xOff,
      seqLen,
      embed,
      this.#ln1_gamma[blockIndex],
      this.#ln1_beta[blockIndex],
      concat,
      0,
      this.#ln1_mean!,
      blockIndex * seqLen,
      this.#ln1_invstd!,
      blockIndex * seqLen,
    );

    // Forward recompute: concatHeads and then output projection to get attnOut, but for backward we need:
    // dWo, dbo, and dConcat = dAttnOut * Wo^T, plus dLN1 via QKV/softmax path.
    // We'll compute concatHeads for each head sequentially (as in forward) and accumulate dConcat.

    // Step A: dWo and dbo
    // Need concatHeads matrix. We'll recompute it into this.#attConcat (overwrite LN1 output? LN1 is needed to compute QKV.
    // So we must keep LN1 output. We'll store LN1 in tmp buffer: use this.#fused as tmp LN1.
    const ln1 = this.#fused!;
    const ln1Len = seqLen * embed;
    for (let i = 0; i < ln1Len; i++) ln1[i] = concat[i];

    // Recompute concatHeads into concat buffer (overwrite)
    fillZero(concat);

    for (let h = 0; h < heads; h++) {
      const wqOff = h * embed * headDim;
      const wkOff = h * embed * headDim;
      const wvOff = h * embed * headDim;

      // Q,K,V
      for (let t = 0; t < seqLen; t++) {
        const xBase = t * embed;
        const qBase = t * headDim;
        for (let d = 0; d < headDim; d++) {
          q[qBase + d] = 0;
          k[qBase + d] = 0;
          v[qBase + d] = 0;
        }
        for (let e = 0; e < embed; e++) {
          const xv = ln1[xBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            q[qBase + d] += xv * Wq[oq + d];
            k[qBase + d] += xv * Wk[ok + d];
            v[qBase + d] += xv * Wv[ov + d];
          }
        }
      }

      // scores -> softmax probs
      for (let i = 0; i < seqLen; i++) {
        const iOff = i * headDim;
        const rowOff = i * seqLen;

        let max = NEG_LARGE;
        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = NEG_LARGE;
            continue;
          }
          const jOff = j * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[iOff + d] * k[jOff + d];
          const s = dot * scale;
          scores[rowOff + j] = s;
          if (s > max) max = s;
        }

        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const sv = scores[rowOff + j] - max;
          const ev = (sv <= -60) ? 0 : Math.exp(sv);
          scores[rowOff + j] = ev;
          sum += ev;
        }
        const invSum = sum > 0 ? (1 / sum) : 0;
        for (let j = 0; j < seqLen; j++) scores[rowOff + j] *= invSum;
      }

      // headOut
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const oOff = i * headDim;
        for (let d = 0; d < headDim; d++) headOut[oOff + d] = 0;

        for (let j = 0; j < seqLen; j++) {
          const w = scores[rowOff + j];
          if (w === 0) continue;
          const vOff = j * headDim;
          for (let d = 0; d < headDim; d++) {
            headOut[oOff + d] += w * v[vOff + d];
          }
        }
      }

      // write to concat
      for (let t = 0; t < seqLen; t++) {
        const src = t * headDim;
        const dst = t * embed + h * headDim;
        for (let d = 0; d < headDim; d++) concat[dst + d] = headOut[src + d];
      }
    }

    // Output projection backward:
    // attnOut = concat * Wo + bo
    // dWo += concat^T * dAttnOut
    // dbo += sum(dAttnOut)
    // dConcat = dAttnOut * Wo^T
    fillZero(dAttConcat);

    for (let t = 0; t < seqLen; t++) {
      const dOff = dAttnOff + t * embed;
      // dbo
      for (let d = 0; d < embed; d++) pbo.g[d] += dAttnOut[dOff + d];

      // dWo
      const cOff = t * embed;
      for (let e = 0; e < embed; e++) {
        const xv = concat[cOff + e];
        const wBase = e * embed;
        for (let d = 0; d < embed; d++) {
          pWo.g[wBase + d] += xv * dAttnOut[dOff + d];
        }
      }

      // dConcat = dAttnOut * Wo^T
      for (let e = 0; e < embed; e++) {
        const wBase = e * embed;
        let acc = 0;
        for (let d = 0; d < embed; d++) {
          acc += dAttnOut[dOff + d] * Wo[wBase + d];
        }
        dAttConcat[dAttConcatOff + t * embed + e] += acc;
      }
    }

    // Now propagate from dConcat to LN1 through heads:
    // For each head:
    // headOut = softmax(scores) * V
    // concat slice is headOut.
    // We'll compute gradients:
    // dV += softmax^T * dHeadOut
    // dScores = softmax * (dW - sum(dW*softmax))
    // dQ, dK from dScores
    // and then dWq, dWk, dWv and dLN1 accumulations

    fillZero(dLN1Out);

    for (let h = 0; h < heads; h++) {
      const wqOff = h * embed * headDim;
      const wkOff = h * embed * headDim;
      const wvOff = h * embed * headDim;

      // Recompute Q,K,V again (could reuse from above if stored; we recompute for clarity)
      for (let t = 0; t < seqLen; t++) {
        const xBase = t * embed;
        const qBase = t * headDim;
        for (let d = 0; d < headDim; d++) {
          q[qBase + d] = 0;
          k[qBase + d] = 0;
          v[qBase + d] = 0;
        }
        for (let e = 0; e < embed; e++) {
          const xv = ln1[xBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            q[qBase + d] += xv * Wq[oq + d];
            k[qBase + d] += xv * Wk[ok + d];
            v[qBase + d] += xv * Wv[ov + d];
          }
        }
      }

      // scores softmax
      for (let i = 0; i < seqLen; i++) {
        const iOff = i * headDim;
        const rowOff = i * seqLen;

        let max = NEG_LARGE;
        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = NEG_LARGE;
            continue;
          }
          const jOff = j * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[iOff + d] * k[jOff + d];
          const s = dot * scale;
          scores[rowOff + j] = s;
          if (s > max) max = s;
        }

        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const sv = scores[rowOff + j] - max;
          const ev = (sv <= -60) ? 0 : Math.exp(sv);
          scores[rowOff + j] = ev;
          sum += ev;
        }
        const invSum = sum > 0 ? (1 / sum) : 0;
        for (let j = 0; j < seqLen; j++) scores[rowOff + j] *= invSum;
      }

      // dHeadOut from dConcat slice
      // dHeadOut shape [seqLen, headDim], stored in headOut buffer
      for (let t = 0; t < seqLen; t++) {
        const src = dAttConcatOff + t * embed + h * headDim;
        const dst = t * headDim;
        for (let d = 0; d < headDim; d++) {
          headOut[dst + d] = dAttConcat[src + d];
        }
      }

      // Compute dV and dScores (reuse q as dQ, k as dK, v as dV)
      // We'll use:
      // dV[j] = Σ_i scores[i,j] * dHeadOut[i]
      // dW(i,j) = dot(dHeadOut[i], V[j])
      // dScores = scores * (dW - Σ_k scores[i,k]*dW(i,k))
      // Then:
      // dQ[i] += Σ_j dScores[i,j] * K[j] * scale
      // dK[j] += Σ_i dScores[i,j] * Q[i] * scale

      // First, compute dV = 0, dQ=0, dK=0
      // We'll repurpose:
      // qBuf as dQ, kBuf as dK, vBuf as dV
      const dQ = q;
      const dK = k;
      const dV = v;

      for (let i = 0; i < dQ.length; i++) {
        dQ[i] = 0;
        dK[i] = 0;
        dV[i] = 0;
      }

      // dV
      for (let j = 0; j < seqLen; j++) {
        const vOff = j * headDim;
        for (let d = 0; d < headDim; d++) dV[vOff + d] = 0;
      }

      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const dhOff = i * headDim;
        for (let j = 0; j < seqLen; j++) {
          const w = scores[rowOff + j];
          if (w === 0) continue;
          const vOff = j * headDim;
          for (let d = 0; d < headDim; d++) {
            dV[vOff + d] += w * headOut[dhOff + d];
          }
        }
      }

      // dScores and dQ/dK
      // We'll compute row-by-row to keep numerical stability.
      const tmpDWRow = this.#tmpHead2!; // length headDim, we need scalar sums; but use as scratch doesn't help.
      // We'll use scores buffer itself to store dScores after computing.
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const dhOff = i * headDim;

        // Precompute dot products dW(i,j) for this row into scores buffer temporarily?
        // We'll compute in-place into scores: scores[rowOff+j] currently softmax probs. We'll store dScores there.
        // Need rowSum = Σ_j p_ij * dW_ij.
        let rowSum = 0;

        // compute dW and rowSum
        for (let j = 0; j < seqLen; j++) {
          const p = scores[rowOff + j];
          if (p === 0) {
            scores[rowOff + j] = 0;
            continue;
          }
          const vOff = j * headDim;
          // dot(dHeadOut[i], V[j])
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += headOut[dhOff + d] * (/*V*/ ((): number => 0)());
          }
        }
        // The line above is a placeholder; we need original V (not dV).
      }
      // The MHA backward needs original V and Q/K. In our recompute pass, V is in v buffer, but we overwrote v with dV.
      // To resolve without allocating, we need separate buffers for original V (and Q,K). We'll use tmpHead/tmpHead2 per step, but not enough.
      // Therefore, we store original Q,K,V for each head in a single pass? That is expensive but necessary for exact backprop.
      // Practical compromise:
      // - Recompute original Q,K,V into dedicated buffers q,k,v (already done).
      // - Do NOT overwrite them; instead use headOut buffer for dQ, tmpHead/tmpHead2 for dK/dV accumulation is not enough.
      // We'll implement a numerically-stable but memory-light approach:
      // - Use separate work buffers dq, dk, dv allocated once with size seqLen*headDim each.
    }

    // Due to the memory constraints and to keep the implementation self-contained and stable,
    // we implement attention backward in a dedicated helper that maintains dq/dk/dv buffers.
    // This function will early-return if allocation isn't prepared; but buffers are prepared in #ensureWork.
    this.#mhaBackwardExact(
      blockIndex,
      seqLen,
      ln1,
      0,
      dAttnOut,
      dAttnOff,
      dLN1Out,
      dLN1Off,
    );
  }

  // Dedicated attention backward with dedicated dq/dk/dv buffers (allocated lazily once).
  #dq: Float64Array | null = null;
  #dk: Float64Array | null = null;
  #dv: Float64Array | null = null;

  #mhaBackwardExact(
    blockIndex: number,
    seqLen: number,
    ln1: Float64Array,
    lnOff: number, // [seqLen, embed]
    dAttnOut: Float64Array,
    dAttnOff: number, // [seqLen, embed]
    dLN1Out: Float64Array,
    dLN1Off: number, // [seqLen, embed]
  ): void {
    const cfg = this.#cfg;
    const embed = cfg.embeddingDim;
    const heads = cfg.numHeads;
    const headDim = (embed / heads) | 0;
    const scale = 1 / Math.sqrt(headDim);

    const q = this.#q!;
    const k = this.#k!;
    const v = this.#v!;
    const scores = this.#scores!;
    const headOut = this.#headOut!;
    const concat = this.#attConcat!;

    // Ensure dq/dk/dv
    const len = seqLen * headDim;
    if (!this.#dq || this.#dq.length !== len) this.#dq = new Float64Array(len);
    if (!this.#dk || this.#dk.length !== len) this.#dk = new Float64Array(len);
    if (!this.#dv || this.#dv.length !== len) this.#dv = new Float64Array(len);
    const dq = this.#dq;
    const dk = this.#dk;
    const dv = this.#dv;

    const Wq = this.#att_Wq[blockIndex];
    const Wk = this.#att_Wk[blockIndex];
    const Wv = this.#att_Wv[blockIndex];
    const Wo = this.#att_Wo[blockIndex];
    const bo = this.#att_bo[blockIndex];

    const pWq = this.#findParam(`att_Wq_${blockIndex}`)!;
    const pWk = this.#findParam(`att_Wk_${blockIndex}`)!;
    const pWv = this.#findParam(`att_Wv_${blockIndex}`)!;
    const pWo = this.#findParam(`att_Wo_${blockIndex}`)!;
    const pbo = this.#findParam(`att_bo_${blockIndex}`)!;

    // Recompute concatHeads (headOut concatenated) and keep it (concat).
    fillZero(concat);

    for (let h = 0; h < heads; h++) {
      const wqOff = h * embed * headDim;
      const wkOff = h * embed * headDim;
      const wvOff = h * embed * headDim;

      // Q,K,V
      for (let t = 0; t < seqLen; t++) {
        const xBase = lnOff + t * embed;
        const qBase = t * headDim;
        for (let d = 0; d < headDim; d++) {
          q[qBase + d] = 0;
          k[qBase + d] = 0;
          v[qBase + d] = 0;
        }
        for (let e = 0; e < embed; e++) {
          const xv = ln1[xBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            q[qBase + d] += xv * Wq[oq + d];
            k[qBase + d] += xv * Wk[ok + d];
            v[qBase + d] += xv * Wv[ov + d];
          }
        }
      }

      // scores probs
      for (let i = 0; i < seqLen; i++) {
        const iOff = i * headDim;
        const rowOff = i * seqLen;

        let max = NEG_LARGE;
        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = NEG_LARGE;
            continue;
          }
          const jOff = j * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[iOff + d] * k[jOff + d];
          const s = dot * scale;
          scores[rowOff + j] = s;
          if (s > max) max = s;
        }

        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const sv = scores[rowOff + j] - max;
          const ev = (sv <= -60) ? 0 : Math.exp(sv);
          scores[rowOff + j] = ev;
          sum += ev;
        }
        const invSum = sum > 0 ? (1 / sum) : 0;
        for (let j = 0; j < seqLen; j++) scores[rowOff + j] *= invSum;
      }

      // headOut
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const oOff = i * headDim;
        for (let d = 0; d < headDim; d++) headOut[oOff + d] = 0;
        for (let j = 0; j < seqLen; j++) {
          const p = scores[rowOff + j];
          if (p === 0) continue;
          const vOff = j * headDim;
          for (let d = 0; d < headDim; d++) {
            headOut[oOff + d] += p * v[vOff + d];
          }
        }
      }

      // write to concat
      for (let t = 0; t < seqLen; t++) {
        const src = t * headDim;
        const dst = t * embed + h * headDim;
        for (let d = 0; d < headDim; d++) concat[dst + d] = headOut[src + d];
      }
    }

    // Output projection backward
    // dConcat = dAttnOut * Wo^T
    fillZero(dLN1Out);
    // We'll store dConcat in this.#fusedGrad (reuse)
    const dConcat = this.#fusedGrad!;
    fillZero(dConcat);

    for (let t = 0; t < seqLen; t++) {
      const dOff = dAttnOff + t * embed;
      for (let d = 0; d < embed; d++) pbo.g[d] += dAttnOut[dOff + d];

      // dWo
      const cOff = t * embed;
      for (let e = 0; e < embed; e++) {
        const xv = concat[cOff + e];
        const wBase = e * embed;
        for (let d = 0; d < embed; d++) {
          pWo.g[wBase + d] += xv * dAttnOut[dOff + d];
        }
      }

      // dConcat
      const dcOff = t * embed;
      for (let e = 0; e < embed; e++) {
        const wBase = e * embed;
        let acc = 0;
        for (let d = 0; d < embed; d++) {
          acc += dAttnOut[dOff + d] * Wo[wBase + d];
        }
        dConcat[dcOff + e] = acc;
      }
    }

    // Now for each head, compute gradients through attention.
    for (let h = 0; h < heads; h++) {
      const wqOff = h * embed * headDim;
      const wkOff = h * embed * headDim;
      const wvOff = h * embed * headDim;

      // Recompute Q,K,V
      for (let t = 0; t < seqLen; t++) {
        const xBase = lnOff + t * embed;
        const qBase = t * headDim;
        for (let d = 0; d < headDim; d++) {
          q[qBase + d] = 0;
          k[qBase + d] = 0;
          v[qBase + d] = 0;
        }
        for (let e = 0; e < embed; e++) {
          const xv = ln1[xBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            q[qBase + d] += xv * Wq[oq + d];
            k[qBase + d] += xv * Wk[ok + d];
            v[qBase + d] += xv * Wv[ov + d];
          }
        }
      }

      // scores probs
      for (let i = 0; i < seqLen; i++) {
        const iOff = i * headDim;
        const rowOff = i * seqLen;

        let max = NEG_LARGE;
        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = NEG_LARGE;
            continue;
          }
          const jOff = j * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[iOff + d] * k[jOff + d];
          const s = dot * scale;
          scores[rowOff + j] = s;
          if (s > max) max = s;
        }

        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const sv = scores[rowOff + j] - max;
          const ev = (sv <= -60) ? 0 : Math.exp(sv);
          scores[rowOff + j] = ev;
          sum += ev;
        }
        const invSum = sum > 0 ? (1 / sum) : 0;
        for (let j = 0; j < seqLen; j++) scores[rowOff + j] *= invSum;
      }

      // dHeadOut from dConcat slice
      for (let t = 0; t < seqLen; t++) {
        const src = t * embed + h * headDim;
        const dst = t * headDim;
        for (let d = 0; d < headDim; d++) headOut[dst + d] = dConcat[src + d];
      }

      // dv = softmax^T * dHeadOut
      fillZero(dv);
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const dhOff = i * headDim;
        for (let j = 0; j < seqLen; j++) {
          const p = scores[rowOff + j];
          if (p === 0) continue;
          const dvOff = j * headDim;
          for (let d = 0; d < headDim; d++) {
            dv[dvOff + d] += p * headOut[dhOff + d];
          }
        }
      }

      // dScores = softmax * (dW - Σ softmax*dW)
      // where dW(i,j) = dot(dHeadOut(i), V(j))
      // We'll compute dScores in-place into scores buffer.
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const dhOff = i * headDim;

        // compute rowSum = Σ_j p_ij * dW_ij
        let rowSum = 0;
        for (let j = 0; j < seqLen; j++) {
          const p = scores[rowOff + j];
          if (p === 0) continue;
          const vOff = j * headDim;
          let dW = 0;
          for (let d = 0; d < headDim; d++) {
            dW += headOut[dhOff + d] * v[vOff + d];
          }
          rowSum += p * dW;
          // store dW temporarily in scores (overwrite p) by using an affine transform: keep p in tmp? Can't.
          // We'll store dW into this.#tmpHead vector isn't enough.
          // Instead, recompute dW again in second pass (acceptable).
        }

        for (let j = 0; j < seqLen; j++) {
          if (cfg.useCausalMask && j > i) {
            scores[rowOff + j] = 0;
            continue;
          }
          const p = scores[rowOff + j];
          if (p === 0) {
            scores[rowOff + j] = 0;
            continue;
          }
          const vOff = j * headDim;
          let dW = 0;
          for (let d = 0; d < headDim; d++) {
            dW += headOut[dhOff + d] * v[vOff + d];
          }
          const dS = p * (dW - rowSum);
          scores[rowOff + j] = dS; // now scores holds dScores
        }
      }

      // dq, dk
      fillZero(dq);
      fillZero(dk);

      // dQ[i] += Σ_j dScores[i,j] * K[j] * scale
      // dK[j] += Σ_i dScores[i,j] * Q[i] * scale
      for (let i = 0; i < seqLen; i++) {
        const rowOff = i * seqLen;
        const qOff = i * headDim;
        for (let j = 0; j < seqLen; j++) {
          const dS = scores[rowOff + j];
          if (dS === 0) continue;
          const kOff = j * headDim;
          const s = dS * scale;
          for (let d = 0; d < headDim; d++) dq[qOff + d] += s * k[kOff + d];
          for (let d = 0; d < headDim; d++) dk[kOff + d] += s * q[qOff + d];
        }
      }

      // Backprop to Wq/Wk/Wv and to ln1:
      // Q = ln1 * Wq_h
      // dWq_h += ln1^T * dQ ; dln1 += dQ * Wq_h^T
      // similarly K and V
      for (let t = 0; t < seqLen; t++) {
        const lnBase = lnOff + t * embed;
        const qBase = t * headDim;

        // dWq/dWk/dWv
        for (let e = 0; e < embed; e++) {
          const xv = ln1[lnBase + e];
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          for (let d = 0; d < headDim; d++) {
            pWq.g[oq + d] += xv * dq[qBase + d];
            pWk.g[ok + d] += xv * dk[qBase + d];
            pWv.g[ov + d] += xv * dv[qBase + d];
          }
        }

        // dln1 += dQ*Wq^T + dK*Wk^T + dV*Wv^T
        const outBase = dLN1Off + t * embed;
        for (let e = 0; e < embed; e++) {
          const wBase = e * headDim;
          const oq = wqOff + wBase;
          const ok = wkOff + wBase;
          const ov = wvOff + wBase;
          let acc = 0;
          for (let d = 0; d < headDim; d++) {
            acc += dq[qBase + d] * Wq[oq + d];
            acc += dk[qBase + d] * Wk[ok + d];
            acc += dv[qBase + d] * Wv[ov + d];
          }
          dLN1Out[outBase + e] += acc;
        }
      }
    }
  }

  // =====================================================================================
  // Small gradient helpers
  // =====================================================================================

  #outGradTmp: Float64Array | null = null;

  #tmpEnsureOutGrad(): Float64Array {
    if (!this.#outGradTmp || this.#outGradTmp.length !== this.#outputDim) {
      this.#outGradTmp = new Float64Array(this.#outputDim);
    }
    return this.#outGradTmp;
  }

  #accGradMatrix(
    p: ParamRef,
    a: Float64Array,
    b: Float64Array,
    aLen: number,
    bLen: number,
  ): void {
    // p.g shape [aLen, bLen] row-major
    // += outer(a, b)
    for (let i = 0; i < aLen; i++) {
      const ai = a[i];
      const off = i * bLen;
      for (let j = 0; j < bLen; j++) p.g[off + j] += ai * b[j];
    }
  }

  #accGradVector(p: ParamRef, g: Float64Array): void {
    for (let i = 0; i < g.length; i++) p.g[i] += g[i];
  }

  #findParam(name: string): ParamRef | null {
    // small linear search (params list is small)
    for (let i = 0; i < this.#params.length; i++) {
      if (this.#params[i].name === name) return this.#params[i];
    }
    return null;
  }

  // =====================================================================================
  // Welford normalization helpers
  // =====================================================================================

  #welfordUpdateInputs(xRaw: Float64Array, seqLen: number): void {
    const st = this.#inNorm!;
    const D = this.#inputDim;
    // Update per feature across all timesteps in this sequence
    for (let t = 0; t < seqLen; t++) {
      const base = t * D;
      st.count++;
      const c = st.count;
      for (let f = 0; f < D; f++) {
        const x = xRaw[base + f];
        const delta = x - st.mean[f];
        st.mean[f] += delta / c;
        const delta2 = x - st.mean[f];
        st.m2[f] += delta * delta2;
      }
    }
  }

  #welfordUpdateOutputs(y: Float64Array): void {
    const st = this.#outNorm!;
    st.count++;
    const c = st.count;
    for (let j = 0; j < this.#outputDim; j++) {
      const x = y[j];
      const delta = x - st.mean[j];
      st.mean[j] += delta / c;
      const delta2 = x - st.mean[j];
      st.m2[j] += delta * delta2;
    }
  }

  #resetNormalizationStats(): void {
    if (this.#inNorm) {
      this.#inNorm.count = 0;
      fillZero(this.#inNorm.mean);
      fillZero(this.#inNorm.m2);
    }
    if (this.#outNorm) {
      this.#outNorm.count = 0;
      fillZero(this.#outNorm.mean);
      fillZero(this.#outNorm.m2);
    }
    // also reset accuracy stats to avoid misleading accuracy after drift
    this.#runningLossSum = 0;
    this.#sampleCount = 0;
    this.#accuracy = 0;
    this.#prevLoss = Number.POSITIVE_INFINITY;
    this.#converged = false;
  }

  // =====================================================================================
  // ADWIN drift detection (simplified, stride-based scanning)
  // =====================================================================================

  #adwinUpdate(err: number): boolean {
    const ad = this.#adwin;
    // push into ring
    const cap = ad.buf.length;
    if (ad.size < cap) {
      ad.buf[(ad.start + ad.size) % cap] = err;
      ad.size++;
    } else {
      // overwrite oldest
      ad.buf[ad.start] = err;
      ad.start = (ad.start + 1) % cap;
    }

    ad.counter++;
    if (ad.counter % ad.checkStride !== 0) return false;
    if (ad.size < 16) return false;

    // snapshot into a contiguous array view (avoid allocating by using a static scratch)
    const n = ad.size;
    const snap = this.#tmpEnsureAdwinSnap(n);
    for (let i = 0; i < n; i++) snap[i] = ad.buf[(ad.start + i) % cap];

    // prefix sums and prefix sums of squares
    const ps = this.#tmpEnsureAdwinPrefix(n + 1);
    const ps2 = this.#tmpEnsureAdwinPrefix2(n + 1);
    ps[0] = 0;
    ps2[0] = 0;
    for (let i = 0; i < n; i++) {
      const v = snap[i];
      ps[i + 1] = ps[i] + v;
      ps2[i + 1] = ps2[i] + v * v;
    }

    // scan split points (stride)
    const delta = ad.delta > 0 ? ad.delta : 1e-12;
    const logTerm = Math.log(4 / delta);

    const minPart = 8;
    let drift = false;
    let cutAt = -1;

    // scan all possible cut points (can be pruned if needed)
    for (let cut = minPart; cut <= n - minPart; cut++) {
      const n0 = cut;
      const n1 = n - cut;

      const sum0 = ps[cut];
      const sum1 = ps[n] - ps[cut];

      const mean0 = sum0 / n0;
      const mean1 = sum1 / n1;

      // harmonicMean = (1/n0 + 1/n1)
      const hm = (1 / n0) + (1 / n1);
      const epsCut = Math.sqrt(hm * logTerm * 0.5);

      if (Math.abs(mean0 - mean1) >= epsCut) {
        drift = true;
        cutAt = cut;
        break;
      }
    }

    if (drift && cutAt > 0) {
      // remove older portion: keep the most recent right window [cutAt..n)
      ad.start = (ad.start + cutAt) % cap;
      ad.size = n - cutAt;
      ad.driftCount++;
      return true;
    }

    return false;
  }

  #adwinSnapTmp: Float64Array | null = null;
  #adwinPrefixTmp: Float64Array | null = null;
  #adwinPrefix2Tmp: Float64Array | null = null;

  #tmpEnsureAdwinSnap(n: number): Float64Array {
    if (!this.#adwinSnapTmp || this.#adwinSnapTmp.length < n) {
      this.#adwinSnapTmp = new Float64Array(n);
    }
    return this.#adwinSnapTmp;
  }
  #tmpEnsureAdwinPrefix(n: number): Float64Array {
    if (!this.#adwinPrefixTmp || this.#adwinPrefixTmp.length < n) {
      this.#adwinPrefixTmp = new Float64Array(n);
    }
    return this.#adwinPrefixTmp;
  }
  #tmpEnsureAdwinPrefix2(n: number): Float64Array {
    if (!this.#adwinPrefix2Tmp || this.#adwinPrefix2Tmp.length < n) {
      this.#adwinPrefix2Tmp = new Float64Array(n);
    }
    return this.#adwinPrefix2Tmp;
  }

  // =====================================================================================
  // LR schedule / stats / exports
  // =====================================================================================

  #lrSchedule(step: number): number {
    const cfg = this.#cfg;
    const base = cfg.learningRate;
    const warm = cfg.warmupSteps | 0;
    const total = cfg.totalSteps | 0;

    if (total <= 0) return base;
    if (warm > 0 && step < warm) return base * (step / Math.max(1, warm));

    if (total <= warm) return base;

    const prog = clamp((step - warm) / Math.max(1, total - warm), 0, 1);
    return base * 0.5 * (1 + Math.cos(Math.PI * prog));
  }

  #countParameters(): number {
    let total = 0;
    for (let i = 0; i < this.#params.length; i++) {
      total += this.#params[i].w.length;
    }
    return total;
  }

  #exportAs2D(flat: Float64Array, rows: number, cols: number): number[][] {
    const out: number[][] = new Array<number[]>(rows);
    let idx = 0;
    for (let r = 0; r < rows; r++) {
      const row = new Array<number>(cols);
      for (let c = 0; c < cols; c++) row[c] = flat[idx++];
      out[r] = row;
    }
    return out;
  }

  #setFromArray(
    dst: Float64Array,
    src: number[] | Float64Array | null | undefined,
  ): void {
    if (!src) return;
    const n = Math.min(dst.length, (src as any).length | 0);
    for (let i = 0; i < n; i++) dst[i] = +((src as any)[i]) || 0;
  }

  #exportWeightsForSave(): any {
    if (!this.#isInitialized) return null;

    const out: any = {
      W_embed: copyToNumberArray(this.#W_embed!),
      b_embed: copyToNumberArray(this.#b_embed!),

      W_conv: this.#W_conv.map((w) => copyToNumberArray(w)),
      scaleEmb: copyToNumberArray(this.#scaleEmb!),

      W_gate: copyToNumberArray(this.#W_gate!),
      b_gate: copyToNumberArray(this.#b_gate!),

      ln1_gamma: this.#ln1_gamma.map((x) => copyToNumberArray(x)),
      ln1_beta: this.#ln1_beta.map((x) => copyToNumberArray(x)),
      ln2_gamma: this.#ln2_gamma.map((x) => copyToNumberArray(x)),
      ln2_beta: this.#ln2_beta.map((x) => copyToNumberArray(x)),

      att_Wq: this.#att_Wq.map((x) => copyToNumberArray(x)),
      att_Wk: this.#att_Wk.map((x) => copyToNumberArray(x)),
      att_Wv: this.#att_Wv.map((x) => copyToNumberArray(x)),
      att_Wo: this.#att_Wo.map((x) => copyToNumberArray(x)),
      att_bo: this.#att_bo.map((x) => copyToNumberArray(x)),

      ffn_W1: this.#ffn_W1.map((x) => copyToNumberArray(x)),
      ffn_b1: this.#ffn_b1.map((x) => copyToNumberArray(x)),
      ffn_W2: this.#ffn_W2.map((x) => copyToNumberArray(x)),
      ffn_b2: this.#ffn_b2.map((x) => copyToNumberArray(x)),

      W_pool: copyToNumberArray(this.#W_pool!),
      b_pool: copyToNumberArray(this.#b_pool!),
      W_out: copyToNumberArray(this.#W_out!),
      b_out: copyToNumberArray(this.#b_out!),

      posEnc: copyToNumberArray(this.#posEnc!),

      // duplicate normalization stats inside weights so load() can restore after #initialize overwrites them
      inNorm: this.#inNorm
        ? {
          count: this.#inNorm.count,
          mean: copyToNumberArray(this.#inNorm.mean),
          m2: copyToNumberArray(this.#inNorm.m2),
        }
        : null,
      outNorm: this.#outNorm
        ? {
          count: this.#outNorm.count,
          mean: copyToNumberArray(this.#outNorm.mean),
          m2: copyToNumberArray(this.#outNorm.m2),
        }
        : null,
    };

    return out;
  }

  #importWeightsFromSave(w: any): void {
    if (!w || !this.#isInitialized) return;

    this.#setFromArray(this.#W_embed!, w.W_embed);
    this.#setFromArray(this.#b_embed!, w.b_embed);

    if (w.W_conv && Array.isArray(w.W_conv)) {
      for (let s = 0; s < this.#W_conv.length && s < w.W_conv.length; s++) {
        this.#setFromArray(this.#W_conv[s], w.W_conv[s]);
      }
    }

    this.#setFromArray(this.#scaleEmb!, w.scaleEmb);
    this.#setFromArray(this.#W_gate!, w.W_gate);
    this.#setFromArray(this.#b_gate!, w.b_gate);

    if (w.ln1_gamma && Array.isArray(w.ln1_gamma)) {
      for (
        let b = 0;
        b < this.#ln1_gamma.length && b < w.ln1_gamma.length;
        b++
      ) this.#setFromArray(this.#ln1_gamma[b], w.ln1_gamma[b]);
    }
    if (w.ln1_beta && Array.isArray(w.ln1_beta)) {
      for (let b = 0; b < this.#ln1_beta.length && b < w.ln1_beta.length; b++) {
        this.#setFromArray(this.#ln1_beta[b], w.ln1_beta[b]);
      }
    }
    if (w.ln2_gamma && Array.isArray(w.ln2_gamma)) {
      for (
        let b = 0;
        b < this.#ln2_gamma.length && b < w.ln2_gamma.length;
        b++
      ) this.#setFromArray(this.#ln2_gamma[b], w.ln2_gamma[b]);
    }
    if (w.ln2_beta && Array.isArray(w.ln2_beta)) {
      for (let b = 0; b < this.#ln2_beta.length && b < w.ln2_beta.length; b++) {
        this.#setFromArray(this.#ln2_beta[b], w.ln2_beta[b]);
      }
    }

    if (w.att_Wq && Array.isArray(w.att_Wq)) {
      for (let b = 0; b < this.#att_Wq.length && b < w.att_Wq.length; b++) {
        this.#setFromArray(this.#att_Wq[b], w.att_Wq[b]);
      }
    }
    if (w.att_Wk && Array.isArray(w.att_Wk)) {
      for (let b = 0; b < this.#att_Wk.length && b < w.att_Wk.length; b++) {
        this.#setFromArray(this.#att_Wk[b], w.att_Wk[b]);
      }
    }
    if (w.att_Wv && Array.isArray(w.att_Wv)) {
      for (let b = 0; b < this.#att_Wv.length && b < w.att_Wv.length; b++) {
        this.#setFromArray(this.#att_Wv[b], w.att_Wv[b]);
      }
    }
    if (w.att_Wo && Array.isArray(w.att_Wo)) {
      for (let b = 0; b < this.#att_Wo.length && b < w.att_Wo.length; b++) {
        this.#setFromArray(this.#att_Wo[b], w.att_Wo[b]);
      }
    }
    if (w.att_bo && Array.isArray(w.att_bo)) {
      for (let b = 0; b < this.#att_bo.length && b < w.att_bo.length; b++) {
        this.#setFromArray(this.#att_bo[b], w.att_bo[b]);
      }
    }

    if (w.ffn_W1 && Array.isArray(w.ffn_W1)) {
      for (let b = 0; b < this.#ffn_W1.length && b < w.ffn_W1.length; b++) {
        this.#setFromArray(this.#ffn_W1[b], w.ffn_W1[b]);
      }
    }
    if (w.ffn_b1 && Array.isArray(w.ffn_b1)) {
      for (let b = 0; b < this.#ffn_b1.length && b < w.ffn_b1.length; b++) {
        this.#setFromArray(this.#ffn_b1[b], w.ffn_b1[b]);
      }
    }
    if (w.ffn_W2 && Array.isArray(w.ffn_W2)) {
      for (let b = 0; b < this.#ffn_W2.length && b < w.ffn_W2.length; b++) {
        this.#setFromArray(this.#ffn_W2[b], w.ffn_W2[b]);
      }
    }
    if (w.ffn_b2 && Array.isArray(w.ffn_b2)) {
      for (let b = 0; b < this.#ffn_b2.length && b < w.ffn_b2.length; b++) {
        this.#setFromArray(this.#ffn_b2[b], w.ffn_b2[b]);
      }
    }

    this.#setFromArray(this.#W_pool!, w.W_pool);
    this.#setFromArray(this.#b_pool!, w.b_pool);
    this.#setFromArray(this.#W_out!, w.W_out);
    this.#setFromArray(this.#b_out!, w.b_out);

    if (w.posEnc) this.#setFromArray(this.#posEnc!, w.posEnc);

    // restore norms if provided (so they survive the #initialize() overwrite in load())
    if (w.inNorm && this.#inNorm) {
      this.#inNorm.count = (w.inNorm.count | 0) || 0;
      this.#setFromArray(this.#inNorm.mean, w.inNorm.mean);
      this.#setFromArray(this.#inNorm.m2, w.inNorm.m2);
    }
    if (w.outNorm && this.#outNorm) {
      this.#outNorm.count = (w.outNorm.count | 0) || 0;
      this.#setFromArray(this.#outNorm.mean, w.outNorm.mean);
      this.#setFromArray(this.#outNorm.m2, w.outNorm.m2);
    }
  }

  #exportParamsForSave(): any {
    const out: any[] = [];
    for (let i = 0; i < this.#params.length; i++) {
      const p = this.#params[i];
      out.push({
        name: p.name,
        rows: p.rows,
        cols: p.cols,
        l2: p.l2,
        m: copyToNumberArray(p.m),
        v: copyToNumberArray(p.v),
      });
    }
    return out;
  }

  #importParamsFromSave(packed: any): void {
    if (!packed || !Array.isArray(packed)) return;
    for (let i = 0; i < packed.length; i++) {
      const it = packed[i];
      const name = String(it?.name ?? "");
      const p = this.#findParam(name);
      if (!p) continue;
      this.#setFromArray(p.m, it.m);
      this.#setFromArray(p.v, it.v);
      // gradients are transient; leave p.g as-is (zeros)
    }
  }
}
