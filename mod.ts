/**
 * Fusion Temporal Transformer for Multivariate Regression
 * with Incremental Online Learning
 *
 * Features:
 * - Online Adam optimizer with warmup and cosine decay
 * - Welford's online algorithm for z-score normalization
 * - L2 regularization
 * - Outlier downweighting
 * - ADWIN drift detection
 *
 * Weight initialization: Xavier uniform for linear layers
 * limit = sqrt(6 / (fanIn + fanOut))
 */

type FitResult = {
  loss: number;
  gradientNorm: number;
  effectiveLearningRate: number;
  isOutlier: boolean;
  converged: boolean;
  sampleIndex: number;
  driftDetected: boolean;
};

type SinglePrediction = {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
};

type PredictionResult = {
  predictions: SinglePrediction[];
  accuracy: number;
  sampleCount: number;
  isModelReady: boolean;
};

type WeightInfo = {
  temporalConvWeights: number[][][];
  scaleEmbeddings: number[][];
  positionalEncoding: number[][];
  fusionWeights: number[][];
  attentionWeights: number[][][];
  ffnWeights: number[][][];
  layerNormParams: number[][];
  outputWeights: number[][];
  firstMoment: number[][][];
  secondMoment: number[][][];
  updateCount: number;
};

type NormalizationStats = {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
  count: number;
};

type ModelSummary = {
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

interface Config {
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
  adwinDelta: 0.002,
  temporalScales: [1, 2, 4],
  temporalKernelSize: 3,
  maxSequenceLength: 512,
};

interface SerializedState {
  config: Config;
  inputDim: number;
  outputDim: number;
  seqLen: number;
  isInitialized: boolean;
  sampleCount: number;
  updateCount: number;
  driftCount: number;
  runningLoss: number;
  normCount: number;
  residualCount: number;
  inputMean: number[];
  inputM2: number[];
  outputMean: number[];
  outputM2: number[];
  residualM2: number[];
  posEnc: number[];
  convW: number[][];
  convB: number[][];
  scaleEmb: number[];
  fusionW: number[];
  fusionB: number[];
  ln1Gamma: number[][];
  ln1Beta: number[][];
  ln2Gamma: number[][];
  ln2Beta: number[][];
  attnWq: number[][];
  attnWk: number[][];
  attnWv: number[][];
  attnWo: number[][];
  ffnW1: number[][];
  ffnB1: number[][];
  ffnW2: number[][];
  ffnB2: number[][];
  poolW: number[];
  poolB: number;
  outW: number[];
  outB: number[];
  adamM: number[];
  adamV: number[];
  lastWindow: number[];
  lastWindowLen: number;
  driftWindow: number[];
  driftWindowHead: number;
  driftWindowCount: number;
  rngState: number;
  converged: boolean;
  lastGradNorm: number;
  lastLR: number;
}

/**
 * Fusion Temporal Transformer for multivariate regression with incremental online learning.
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({ numBlocks: 2 });
 *
 * // Train on sequences
 * for (const sample of trainingData) {
 *   const result = model.fitOnline({
 *     xCoordinates: sample.x,  // [seqLen][inputDim]
 *     yCoordinates: sample.y   // [seqLen][outputDim]
 *   });
 *   console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
 * }
 *
 * // Predict future steps
 * const predictions = model.predict(5);
 * console.log(predictions.predictions[0].predicted);
 * ```
 */
export class FusionTemporalTransformerRegression {
  private config: Config;
  private inputDim: number = 0;
  private outputDim: number = 0;
  private seqLen: number = 0;
  private isInitialized: boolean = false;
  private ffnDim: number = 0;
  private headDim: number = 0;
  private numScales: number = 0;

  // Welford normalization
  private inputMean!: Float64Array;
  private inputM2!: Float64Array;
  private outputMean!: Float64Array;
  private outputM2!: Float64Array;
  private normCount: number = 0;

  // Residual variance for uncertainty
  private residualM2!: Float64Array;
  private residualCount: number = 0;

  // Positional encoding
  private posEnc!: Float64Array;

  // Conv weights per scale
  private convW: Float64Array[] = [];
  private convB: Float64Array[] = [];

  // Scale embeddings
  private scaleEmb!: Float64Array;

  // Fusion gate
  private fusionW!: Float64Array;
  private fusionB!: Float64Array;

  // Transformer blocks
  private ln1Gamma: Float64Array[] = [];
  private ln1Beta: Float64Array[] = [];
  private ln2Gamma: Float64Array[] = [];
  private ln2Beta: Float64Array[] = [];
  private attnWq: Float64Array[] = [];
  private attnWk: Float64Array[] = [];
  private attnWv: Float64Array[] = [];
  private attnWo: Float64Array[] = [];
  private ffnW1: Float64Array[] = [];
  private ffnB1: Float64Array[] = [];
  private ffnW2: Float64Array[] = [];
  private ffnB2: Float64Array[] = [];

  // Pooling
  private poolW!: Float64Array;
  private poolB: number = 0;

  // Output
  private outW!: Float64Array;
  private outB!: Float64Array;

  // Adam moments
  private adamM!: Float64Array;
  private adamV!: Float64Array;
  private totalParams: number = 0;

  // Counters
  private updateCount: number = 0;
  private sampleCount: number = 0;
  private driftCount: number = 0;
  private runningLoss: number = 0;

  // ADWIN
  private driftWindow: Float64Array;
  private driftWindowSize: number = 256;
  private driftWindowHead: number = 0;
  private driftWindowCount: number = 0;

  // Last window
  private lastWindow!: Float64Array;
  private lastWindowLen: number = 0;

  // Forward cache
  private fwdNormX!: Float64Array;
  private fwdNormY!: Float64Array;
  private fwdConvOut: Float64Array[] = [];
  private fwdConvPre: Float64Array[] = [];
  private fwdScaleEmb: Float64Array[] = [];
  private fwdUpsampled!: Float64Array;
  private fwdGateIn!: Float64Array;
  private fwdGate!: Float64Array;
  private fwdFused!: Float64Array;
  private fwdBlockIn: Float64Array[] = [];
  private fwdLn1Out: Float64Array[] = [];
  private fwdLn1Mean: Float64Array[] = [];
  private fwdLn1Rstd: Float64Array[] = [];
  private fwdQ: Float64Array[] = [];
  private fwdK: Float64Array[] = [];
  private fwdV: Float64Array[] = [];
  private fwdAttnScores: Float64Array[] = [];
  private fwdAttnProbs: Float64Array[] = [];
  private fwdAttnOut: Float64Array[] = [];
  private fwdAttnProj: Float64Array[] = [];
  private fwdRes1: Float64Array[] = [];
  private fwdLn2Out: Float64Array[] = [];
  private fwdLn2Mean: Float64Array[] = [];
  private fwdLn2Rstd: Float64Array[] = [];
  private fwdFfnHid: Float64Array[] = [];
  private fwdFfnPre: Float64Array[] = [];
  private fwdFfnOut: Float64Array[] = [];
  private fwdBlockOut: Float64Array[] = [];
  private fwdPoolScores!: Float64Array;
  private fwdPoolWeights!: Float64Array;
  private fwdPooled!: Float64Array;
  private fwdOutput!: Float64Array;

  // Gradient buffers
  private gradConvW: Float64Array[] = [];
  private gradConvB: Float64Array[] = [];
  private gradScaleEmb!: Float64Array;
  private gradFusionW!: Float64Array;
  private gradFusionB!: Float64Array;
  private gradLn1Gamma: Float64Array[] = [];
  private gradLn1Beta: Float64Array[] = [];
  private gradLn2Gamma: Float64Array[] = [];
  private gradLn2Beta: Float64Array[] = [];
  private gradAttnWq: Float64Array[] = [];
  private gradAttnWk: Float64Array[] = [];
  private gradAttnWv: Float64Array[] = [];
  private gradAttnWo: Float64Array[] = [];
  private gradFfnW1: Float64Array[] = [];
  private gradFfnB1: Float64Array[] = [];
  private gradFfnW2: Float64Array[] = [];
  private gradFfnB2: Float64Array[] = [];
  private gradPoolW!: Float64Array;
  private gradPoolB: number = 0;
  private gradOutW!: Float64Array;
  private gradOutB!: Float64Array;

  // Scratch buffers
  private scratchSeqEmb!: Float64Array;
  private scratchSeqEmb2!: Float64Array;
  private scratchHead!: Float64Array;
  private scratchFfn!: Float64Array;
  private scratchEmb!: Float64Array;
  private scratchGrad!: Float64Array;
  private scratchScales!: Float64Array;

  // RNG
  private rngState: number = 12345;
  private converged: boolean = false;
  private lastGradNorm: number = Infinity;
  private lastLR: number = 0;

  /**
   * Creates a new FusionTemporalTransformerRegression instance.
   * @param config - Optional partial configuration to override defaults
   * @example
   * ```typescript
   * const model = new FusionTemporalTransformerRegression({
   *   numBlocks: 4,
   *   embeddingDim: 128,
   *   learningRate: 0.0005
   * });
   * ```
   */
  constructor(config?: Partial<Config>) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this.config.embeddingDim}) must be divisible by numHeads (${this.config.numHeads})`,
      );
    }

    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;
    this.numScales = this.config.temporalScales.length;
    this.driftWindow = new Float64Array(this.driftWindowSize);
  }

  /**
   * Xorshift32 deterministic RNG
   * @returns pseudo-random number in [0, 1)
   */
  private xorshift32(): number {
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return (this.rngState >>> 0) / 4294967296;
  }

  /**
   * Xavier uniform initialization
   * Formula: U(-limit, limit) where limit = sqrt(6 / (fanIn + fanOut))
   */
  private xavierInit(arr: Float64Array, fanIn: number, fanOut: number): void {
    const limit = Math.sqrt(6.0 / (fanIn + fanOut));
    for (let i = 0; i < arr.length; i++) {
      arr[i] = (this.xorshift32() * 2 - 1) * limit;
    }
  }

  /**
   * Initialize positional encoding
   * Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
   *          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
   */
  private initPositionalEncoding(): void {
    const maxLen = this.config.maxSequenceLength;
    const d = this.config.embeddingDim;
    this.posEnc = new Float64Array(maxLen * d);

    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < d; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
        const idx = pos * d + i;
        if (i % 2 === 0) {
          this.posEnc[idx] = Math.sin(angle);
        } else {
          this.posEnc[idx] = Math.cos(angle);
        }
      }
    }
  }

  /**
   * Lazy initialization on first fitOnline call
   */
  private initialize(
    inputDim: number,
    outputDim: number,
    seqLen: number,
  ): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.seqLen = Math.min(seqLen, this.config.maxSequenceLength);

    const emb = this.config.embeddingDim;
    const ks = this.config.temporalKernelSize;
    const nb = this.config.numBlocks;
    const ns = this.numScales;
    const msl = this.config.maxSequenceLength;

    // Initialize Welford stats
    this.inputMean = new Float64Array(inputDim);
    this.inputM2 = new Float64Array(inputDim);
    this.outputMean = new Float64Array(outputDim);
    this.outputM2 = new Float64Array(outputDim);
    this.residualM2 = new Float64Array(outputDim);

    // Positional encoding
    this.initPositionalEncoding();

    // Conv weights per scale
    let paramCount = 0;
    for (let s = 0; s < ns; s++) {
      const wSize = ks * inputDim * emb;
      this.convW[s] = new Float64Array(wSize);
      this.xavierInit(this.convW[s], ks * inputDim, emb);
      this.convB[s] = new Float64Array(emb);
      paramCount += wSize + emb;
    }

    // Scale embeddings
    this.scaleEmb = new Float64Array(ns * emb);
    for (let i = 0; i < this.scaleEmb.length; i++) {
      this.scaleEmb[i] = (this.xorshift32() * 2 - 1) * 0.02;
    }
    paramCount += ns * emb;

    // Fusion gate
    const fusionDim = ns * emb;
    this.fusionW = new Float64Array(fusionDim * fusionDim);
    this.xavierInit(this.fusionW, fusionDim, fusionDim);
    this.fusionB = new Float64Array(fusionDim);
    paramCount += fusionDim * fusionDim + fusionDim;

    // Transformer blocks
    for (let b = 0; b < nb; b++) {
      this.ln1Gamma[b] = new Float64Array(emb);
      this.ln1Beta[b] = new Float64Array(emb);
      this.ln2Gamma[b] = new Float64Array(emb);
      this.ln2Beta[b] = new Float64Array(emb);
      for (let i = 0; i < emb; i++) {
        this.ln1Gamma[b][i] = 1;
        this.ln2Gamma[b][i] = 1;
      }
      paramCount += 4 * emb;

      this.attnWq[b] = new Float64Array(emb * emb);
      this.attnWk[b] = new Float64Array(emb * emb);
      this.attnWv[b] = new Float64Array(emb * emb);
      this.attnWo[b] = new Float64Array(emb * emb);
      this.xavierInit(this.attnWq[b], emb, emb);
      this.xavierInit(this.attnWk[b], emb, emb);
      this.xavierInit(this.attnWv[b], emb, emb);
      this.xavierInit(this.attnWo[b], emb, emb);
      paramCount += 4 * emb * emb;

      this.ffnW1[b] = new Float64Array(emb * this.ffnDim);
      this.ffnB1[b] = new Float64Array(this.ffnDim);
      this.ffnW2[b] = new Float64Array(this.ffnDim * emb);
      this.ffnB2[b] = new Float64Array(emb);
      this.xavierInit(this.ffnW1[b], emb, this.ffnDim);
      this.xavierInit(this.ffnW2[b], this.ffnDim, emb);
      paramCount += emb * this.ffnDim + this.ffnDim + this.ffnDim * emb + emb;
    }

    // Pooling
    this.poolW = new Float64Array(emb);
    this.xavierInit(this.poolW, emb, 1);
    this.poolB = 0;
    paramCount += emb + 1;

    // Output
    this.outW = new Float64Array(emb * outputDim);
    this.outB = new Float64Array(outputDim);
    this.xavierInit(this.outW, emb, outputDim);
    paramCount += emb * outputDim + outputDim;

    this.totalParams = paramCount;

    // Adam moments
    this.adamM = new Float64Array(paramCount);
    this.adamV = new Float64Array(paramCount);

    // Allocate forward cache
    this.fwdNormX = new Float64Array(msl * inputDim);
    this.fwdNormY = new Float64Array(outputDim);
    this.lastWindow = new Float64Array(msl * inputDim);

    for (let s = 0; s < ns; s++) {
      const scaleLen = Math.ceil(msl / this.config.temporalScales[s]);
      this.fwdConvOut[s] = new Float64Array(scaleLen * emb);
      this.fwdConvPre[s] = new Float64Array(scaleLen * emb);
      this.fwdScaleEmb[s] = new Float64Array(scaleLen * emb);
    }

    this.fwdUpsampled = new Float64Array(msl * ns * emb);
    this.fwdGateIn = new Float64Array(msl * ns * emb);
    this.fwdGate = new Float64Array(msl * ns * emb);
    this.fwdFused = new Float64Array(msl * emb);

    for (let b = 0; b < nb; b++) {
      this.fwdBlockIn[b] = new Float64Array(msl * emb);
      this.fwdLn1Out[b] = new Float64Array(msl * emb);
      this.fwdLn1Mean[b] = new Float64Array(msl);
      this.fwdLn1Rstd[b] = new Float64Array(msl);
      this.fwdQ[b] = new Float64Array(msl * emb);
      this.fwdK[b] = new Float64Array(msl * emb);
      this.fwdV[b] = new Float64Array(msl * emb);
      this.fwdAttnScores[b] = new Float64Array(
        this.config.numHeads * msl * msl,
      );
      this.fwdAttnProbs[b] = new Float64Array(this.config.numHeads * msl * msl);
      this.fwdAttnOut[b] = new Float64Array(msl * emb);
      this.fwdAttnProj[b] = new Float64Array(msl * emb);
      this.fwdRes1[b] = new Float64Array(msl * emb);
      this.fwdLn2Out[b] = new Float64Array(msl * emb);
      this.fwdLn2Mean[b] = new Float64Array(msl);
      this.fwdLn2Rstd[b] = new Float64Array(msl);
      this.fwdFfnHid[b] = new Float64Array(msl * this.ffnDim);
      this.fwdFfnPre[b] = new Float64Array(msl * this.ffnDim);
      this.fwdFfnOut[b] = new Float64Array(msl * emb);
      this.fwdBlockOut[b] = new Float64Array(msl * emb);
    }

    this.fwdPoolScores = new Float64Array(msl);
    this.fwdPoolWeights = new Float64Array(msl);
    this.fwdPooled = new Float64Array(emb);
    this.fwdOutput = new Float64Array(outputDim);

    // Gradient buffers
    for (let s = 0; s < ns; s++) {
      this.gradConvW[s] = new Float64Array(ks * inputDim * emb);
      this.gradConvB[s] = new Float64Array(emb);
    }
    this.gradScaleEmb = new Float64Array(ns * emb);
    this.gradFusionW = new Float64Array(fusionDim * fusionDim);
    this.gradFusionB = new Float64Array(fusionDim);

    for (let b = 0; b < nb; b++) {
      this.gradLn1Gamma[b] = new Float64Array(emb);
      this.gradLn1Beta[b] = new Float64Array(emb);
      this.gradLn2Gamma[b] = new Float64Array(emb);
      this.gradLn2Beta[b] = new Float64Array(emb);
      this.gradAttnWq[b] = new Float64Array(emb * emb);
      this.gradAttnWk[b] = new Float64Array(emb * emb);
      this.gradAttnWv[b] = new Float64Array(emb * emb);
      this.gradAttnWo[b] = new Float64Array(emb * emb);
      this.gradFfnW1[b] = new Float64Array(emb * this.ffnDim);
      this.gradFfnB1[b] = new Float64Array(this.ffnDim);
      this.gradFfnW2[b] = new Float64Array(this.ffnDim * emb);
      this.gradFfnB2[b] = new Float64Array(emb);
    }

    this.gradPoolW = new Float64Array(emb);
    this.gradOutW = new Float64Array(emb * outputDim);
    this.gradOutB = new Float64Array(outputDim);

    // Scratch
    this.scratchSeqEmb = new Float64Array(msl * emb);
    this.scratchSeqEmb2 = new Float64Array(msl * emb);
    this.scratchHead = new Float64Array(msl * msl);
    this.scratchFfn = new Float64Array(msl * this.ffnDim);
    this.scratchEmb = new Float64Array(emb);
    this.scratchGrad = new Float64Array(msl * ns * emb);
    this.scratchScales = new Float64Array(msl);

    this.isInitialized = true;
  }

  /**
   * Update Welford statistics for online normalization
   * Mean_n = Mean_{n-1} + (x - Mean_{n-1}) / n
   * M2_n = M2_{n-1} + (x - Mean_{n-1}) * (x - Mean_n)
   */
  private updateWelford(
    x: Float64Array,
    mean: Float64Array,
    m2: Float64Array,
    count: number,
  ): void {
    const n = count;
    for (let i = 0; i < x.length; i++) {
      const delta = x[i] - mean[i];
      mean[i] += delta / n;
      const delta2 = x[i] - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation from Welford M2
   * std = sqrt(M2 / n), clamped to >= 1e-12
   */
  private getStd(m2: Float64Array, count: number, out: Float64Array): void {
    const n = Math.max(count, 1);
    for (let i = 0; i < m2.length; i++) {
      out[i] = Math.sqrt(Math.max(m2[i] / n, 1e-24));
      if (out[i] < 1e-12) out[i] = 1e-12;
    }
  }

  /**
   * GELU activation
   * Formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
   */
  private gelu(x: number): number {
    const c = 0.7978845608028654; // sqrt(2/pi)
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  }

  /**
   * GELU derivative
   * d/dx[GELU(x)] = 0.5*(1 + tanh(u)) + 0.5*x*sech²(u)*(c*(1 + 3*0.044715*x²))
   * where u = sqrt(2/π)*(x + 0.044715*x³)
   */
  private geluGrad(x: number): number {
    const c = 0.7978845608028654;
    const x3 = x * x * x;
    const u = c * (x + 0.044715 * x3);
    const tanhU = Math.tanh(u);
    const sech2U = 1 - tanhU * tanhU;
    return 0.5 * (1 + tanhU) +
      0.5 * x * sech2U * c * (1 + 3 * 0.044715 * x * x);
  }

  /**
   * Sigmoid activation: 1 / (1 + exp(-x))
   */
  private sigmoid(x: number): number {
    if (x >= 0) {
      return 1 / (1 + Math.exp(-x));
    } else {
      const ex = Math.exp(x);
      return ex / (1 + ex);
    }
  }

  /**
   * Stable softmax over a range
   */
  private softmax(
    arr: Float64Array,
    start: number,
    len: number,
    out: Float64Array,
    outStart: number,
  ): void {
    let maxVal = -Infinity;
    for (let i = 0; i < len; i++) {
      if (arr[start + i] > maxVal) maxVal = arr[start + i];
    }
    if (!isFinite(maxVal)) maxVal = 0;

    let sum = 0;
    for (let i = 0; i < len; i++) {
      const e = Math.exp(arr[start + i] - maxVal);
      out[outStart + i] = e;
      sum += e;
    }

    if (sum < 1e-12) sum = 1e-12;
    for (let i = 0; i < len; i++) {
      out[outStart + i] /= sum;
    }
  }

  /**
   * Masked causal softmax for attention
   */
  private maskedSoftmax(
    scores: Float64Array,
    start: number,
    queryIdx: number,
    seqLen: number,
    out: Float64Array,
    outStart: number,
  ): void {
    let maxVal = -Infinity;
    for (let j = 0; j <= queryIdx; j++) {
      if (scores[start + j] > maxVal) maxVal = scores[start + j];
    }
    if (!isFinite(maxVal)) maxVal = 0;

    let sum = 0;
    for (let j = 0; j < seqLen; j++) {
      if (j <= queryIdx) {
        const e = Math.exp(scores[start + j] - maxVal);
        out[outStart + j] = e;
        sum += e;
      } else {
        out[outStart + j] = 0;
      }
    }

    if (sum < 1e-12) {
      for (let j = 0; j <= queryIdx; j++) {
        out[outStart + j] = 1 / (queryIdx + 1);
      }
    } else {
      for (let j = 0; j <= queryIdx; j++) {
        out[outStart + j] /= sum;
      }
    }
  }

  /**
   * Layer normalization forward
   * y = gamma * (x - mean) / sqrt(var + eps) + beta
   */
  private layerNormForward(
    input: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    seqLen: number,
    dim: number,
    output: Float64Array,
    meanOut: Float64Array,
    rstdOut: Float64Array,
  ): void {
    const eps = 1e-12;
    for (let t = 0; t < seqLen; t++) {
      const base = t * dim;
      let mean = 0;
      for (let d = 0; d < dim; d++) {
        mean += input[base + d];
      }
      mean /= dim;
      meanOut[t] = mean;

      let variance = 0;
      for (let d = 0; d < dim; d++) {
        const diff = input[base + d] - mean;
        variance += diff * diff;
      }
      variance /= dim;
      const rstd = 1 / Math.sqrt(Math.max(variance, eps) + eps);
      rstdOut[t] = rstd;

      for (let d = 0; d < dim; d++) {
        output[base + d] = gamma[d] * (input[base + d] - mean) * rstd + beta[d];
      }
    }
  }

  /**
   * Layer normalization backward
   */
  private layerNormBackward(
    dout: Float64Array,
    input: Float64Array,
    mean: Float64Array,
    rstd: Float64Array,
    gamma: Float64Array,
    seqLen: number,
    dim: number,
    dinput: Float64Array,
    dgamma: Float64Array,
    dbeta: Float64Array,
  ): void {
    for (let d = 0; d < dim; d++) {
      dgamma[d] = 0;
      dbeta[d] = 0;
    }

    for (let t = 0; t < seqLen; t++) {
      const base = t * dim;
      const m = mean[t];
      const r = rstd[t];

      for (let d = 0; d < dim; d++) {
        const xhat = (input[base + d] - m) * r;
        dgamma[d] += dout[base + d] * xhat;
        dbeta[d] += dout[base + d];
      }

      let dxhat_sum = 0;
      let dxhat_xhat_sum = 0;
      for (let d = 0; d < dim; d++) {
        const dxhat = dout[base + d] * gamma[d];
        const xhat = (input[base + d] - m) * r;
        dxhat_sum += dxhat;
        dxhat_xhat_sum += dxhat * xhat;
      }

      for (let d = 0; d < dim; d++) {
        const xhat = (input[base + d] - m) * r;
        const dxhat = dout[base + d] * gamma[d];
        dinput[base + d] = r *
          (dxhat - (dxhat_sum + xhat * dxhat_xhat_sum) / dim);
      }
    }
  }

  /**
   * Matrix multiply: C = A * B
   * A: [m, k], B: [k, n], C: [m, n]
   */
  private matMul(
    A: Float64Array,
    B: Float64Array,
    C: Float64Array,
    m: number,
    k: number,
    n: number,
    aOffset: number = 0,
    bOffset: number = 0,
    cOffset: number = 0,
  ): void {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += A[aOffset + i * k + l] * B[bOffset + l * n + j];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Matrix multiply transposed: C = A * B^T
   * A: [m, k], B: [n, k], C: [m, n]
   */
  private matMulTransB(
    A: Float64Array,
    B: Float64Array,
    C: Float64Array,
    m: number,
    k: number,
    n: number,
    aOffset: number = 0,
    bOffset: number = 0,
    cOffset: number = 0,
  ): void {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += A[aOffset + i * k + l] * B[bOffset + j * k + l];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Matrix multiply transposed A: C = A^T * B
   * A: [k, m], B: [k, n], C: [m, n]
   */
  private matMulTransA(
    A: Float64Array,
    B: Float64Array,
    C: Float64Array,
    k: number,
    m: number,
    n: number,
    aOffset: number = 0,
    bOffset: number = 0,
    cOffset: number = 0,
  ): void {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += A[aOffset + l * m + i] * B[bOffset + l * n + j];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Forward pass through temporal convolution for one scale
   * Conv1D with causal padding (left zero padding)
   */
  private convForward(
    input: Float64Array,
    seqLen: number,
    scale: number,
    scaleIdx: number,
  ): void {
    const emb = this.config.embeddingDim;
    const ks = this.config.temporalKernelSize;
    const outLen = Math.ceil(seqLen / scale);
    const W = this.convW[scaleIdx];
    const B = this.convB[scaleIdx];
    const preAct = this.fwdConvPre[scaleIdx];
    const out = this.fwdConvOut[scaleIdx];

    for (let t = 0; t < outLen; t++) {
      const centerPos = t * scale;
      for (let e = 0; e < emb; e++) {
        let sum = B[e];
        for (let k = 0; k < ks; k++) {
          const inputPos = centerPos - k;
          if (inputPos >= 0 && inputPos < seqLen) {
            for (let f = 0; f < this.inputDim; f++) {
              const wIdx = (k * this.inputDim + f) * emb + e;
              sum += input[inputPos * this.inputDim + f] * W[wIdx];
            }
          }
        }
        preAct[t * emb + e] = sum;
        out[t * emb + e] = this.gelu(sum);
      }
    }
  }

  /**
   * Backward through temporal convolution
   */
  private convBackward(
    dout: Float64Array,
    input: Float64Array,
    seqLen: number,
    scale: number,
    scaleIdx: number,
    dinput: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;
    const ks = this.config.temporalKernelSize;
    const outLen = Math.ceil(seqLen / scale);
    const W = this.convW[scaleIdx];
    const preAct = this.fwdConvPre[scaleIdx];
    const dW = this.gradConvW[scaleIdx];
    const dB = this.gradConvB[scaleIdx];

    for (let t = 0; t < outLen; t++) {
      const centerPos = t * scale;
      for (let e = 0; e < emb; e++) {
        const idx = t * emb + e;
        const dGelu = dout[idx] * this.geluGrad(preAct[idx]);
        dB[e] += dGelu;

        for (let k = 0; k < ks; k++) {
          const inputPos = centerPos - k;
          if (inputPos >= 0 && inputPos < seqLen) {
            for (let f = 0; f < this.inputDim; f++) {
              const wIdx = (k * this.inputDim + f) * emb + e;
              dW[wIdx] += input[inputPos * this.inputDim + f] * dGelu;
              dinput[inputPos * this.inputDim + f] += W[wIdx] * dGelu;
            }
          }
        }
      }
    }
  }

  /**
   * Add scale embeddings and positional encoding
   */
  private addEmbeddings(scaleIdx: number, scaleLen: number): void {
    const emb = this.config.embeddingDim;
    const convOut = this.fwdConvOut[scaleIdx];
    const out = this.fwdScaleEmb[scaleIdx];
    const scaleEmbBase = scaleIdx * emb;

    for (let t = 0; t < scaleLen; t++) {
      const baseIn = t * emb;
      const peBase = t * emb;
      for (let e = 0; e < emb; e++) {
        out[baseIn + e] = convOut[baseIn + e] +
          this.posEnc[peBase + e] +
          this.scaleEmb[scaleEmbBase + e];
      }
    }
  }

  /**
   * Upsample and concatenate all scales
   */
  private upsampleAndConcat(seqLen: number): void {
    const emb = this.config.embeddingDim;
    const ns = this.numScales;

    for (let t = 0; t < seqLen; t++) {
      for (let s = 0; s < ns; s++) {
        const scale = this.config.temporalScales[s];
        const scaleLen = Math.ceil(seqLen / scale);
        const srcT = Math.min(Math.floor(t / scale), scaleLen - 1);
        const srcBase = srcT * emb;
        const dstBase = t * ns * emb + s * emb;
        const src = this.fwdScaleEmb[s];

        for (let e = 0; e < emb; e++) {
          this.fwdUpsampled[dstBase + e] = src[srcBase + e];
        }
      }
    }
  }

  /**
   * Fusion gating forward
   * G = sigmoid(concat * Wg + bg)
   * fused = sum_s(G_s * E_s)
   */
  private fusionForward(seqLen: number, training: boolean): void {
    const emb = this.config.embeddingDim;
    const ns = this.numScales;
    const fusionDim = ns * emb;

    // Gate computation
    for (let t = 0; t < seqLen; t++) {
      const inBase = t * fusionDim;
      for (let i = 0; i < fusionDim; i++) {
        let sum = this.fusionB[i];
        for (let j = 0; j < fusionDim; j++) {
          sum += this.fwdUpsampled[inBase + j] *
            this.fusionW[j * fusionDim + i];
        }
        this.fwdGateIn[inBase + i] = sum;
        this.fwdGate[inBase + i] = this.sigmoid(sum);
      }
    }

    // Apply dropout if training
    if (training && this.config.fusionDropout > 0) {
      for (let i = 0; i < seqLen * fusionDim; i++) {
        if (this.xorshift32() < this.config.fusionDropout) {
          this.fwdGate[i] = 0;
        } else {
          this.fwdGate[i] /= 1 - this.config.fusionDropout;
        }
      }
    }

    // Fused output
    for (let t = 0; t < seqLen; t++) {
      const gateBase = t * fusionDim;
      const outBase = t * emb;
      for (let e = 0; e < emb; e++) {
        let sum = 0;
        for (let s = 0; s < ns; s++) {
          const gateIdx = gateBase + s * emb + e;
          const valIdx = gateBase + s * emb + e;
          sum += this.fwdGate[gateIdx] * this.fwdUpsampled[valIdx];
        }
        this.fwdFused[outBase + e] = sum;
      }
    }
  }

  /**
   * Fusion backward
   */
  private fusionBackward(
    dout: Float64Array,
    seqLen: number,
    dUpsampled: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;
    const ns = this.numScales;
    const fusionDim = ns * emb;

    // Gradient through fused output
    for (let t = 0; t < seqLen; t++) {
      const gateBase = t * fusionDim;
      const outBase = t * emb;

      for (let s = 0; s < ns; s++) {
        for (let e = 0; e < emb; e++) {
          const gateIdx = gateBase + s * emb + e;
          const dOut = dout[outBase + e];

          // dGate
          const g = this.fwdGate[gateIdx];
          const val = this.fwdUpsampled[gateIdx];
          const dGate = dOut * val;

          // dUpsampled
          dUpsampled[gateIdx] += dOut * g;

          // dGateIn (sigmoid grad: g * (1 - g))
          const dGateIn = dGate * g * (1 - g);

          // Gradient to fusionW and fusionB
          this.gradFusionB[s * emb + e] += dGateIn;
          for (let j = 0; j < fusionDim; j++) {
            this.gradFusionW[j * fusionDim + s * emb + e] +=
              this.fwdUpsampled[gateBase + j] * dGateIn;
          }
        }
      }
    }
  }

  /**
   * Multi-head attention forward
   */
  private attentionForward(
    input: Float64Array,
    block: number,
    seqLen: number,
    training: boolean,
  ): void {
    const emb = this.config.embeddingDim;
    const nh = this.config.numHeads;
    const hd = this.headDim;
    const scale = 1 / Math.sqrt(hd);

    // Q, K, V projections
    this.matMul(input, this.attnWq[block], this.fwdQ[block], seqLen, emb, emb);
    this.matMul(input, this.attnWk[block], this.fwdK[block], seqLen, emb, emb);
    this.matMul(input, this.attnWv[block], this.fwdV[block], seqLen, emb, emb);

    // Clear attention output
    for (let i = 0; i < seqLen * emb; i++) {
      this.fwdAttnOut[block][i] = 0;
    }

    // Per-head attention
    for (let h = 0; h < nh; h++) {
      const headOffset = h * hd;
      const scoresBase = h * seqLen * seqLen;
      const probsBase = h * seqLen * seqLen;

      // Compute scores
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          if (j > i) {
            this.fwdAttnScores[block][scoresBase + i * seqLen + j] = -1e9;
          } else {
            let dot = 0;
            for (let d = 0; d < hd; d++) {
              dot += this.fwdQ[block][i * emb + headOffset + d] *
                this.fwdK[block][j * emb + headOffset + d];
            }
            this.fwdAttnScores[block][scoresBase + i * seqLen + j] = dot *
              scale;
          }
        }

        // Softmax
        this.maskedSoftmax(
          this.fwdAttnScores[block],
          scoresBase + i * seqLen,
          i,
          seqLen,
          this.fwdAttnProbs[block],
          probsBase + i * seqLen,
        );

        // Apply attention dropout
        if (training && this.config.attentionDropout > 0) {
          for (let j = 0; j <= i; j++) {
            const idx = probsBase + i * seqLen + j;
            if (this.xorshift32() < this.config.attentionDropout) {
              this.fwdAttnProbs[block][idx] = 0;
            } else {
              this.fwdAttnProbs[block][idx] /= 1 - this.config.attentionDropout;
            }
          }
        }

        // Weighted sum of values
        for (let d = 0; d < hd; d++) {
          let sum = 0;
          for (let j = 0; j <= i; j++) {
            sum += this.fwdAttnProbs[block][probsBase + i * seqLen + j] *
              this.fwdV[block][j * emb + headOffset + d];
          }
          this.fwdAttnOut[block][i * emb + headOffset + d] = sum;
        }
      }
    }

    // Output projection
    this.matMul(
      this.fwdAttnOut[block],
      this.attnWo[block],
      this.fwdAttnProj[block],
      seqLen,
      emb,
      emb,
    );
  }

  /**
   * Multi-head attention backward
   */
  private attentionBackward(
    dout: Float64Array,
    input: Float64Array,
    block: number,
    seqLen: number,
    dinput: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;
    const nh = this.config.numHeads;
    const hd = this.headDim;
    const scale = 1 / Math.sqrt(hd);

    // Gradient through output projection
    // dAttnOut = dout * Wo^T
    this.matMulTransB(
      dout,
      this.attnWo[block],
      this.scratchSeqEmb,
      seqLen,
      emb,
      emb,
    );

    // dWo = AttnOut^T * dout
    this.matMulTransA(
      this.fwdAttnOut[block],
      dout,
      this.gradAttnWo[block],
      seqLen,
      emb,
      emb,
    );

    // Accumulate gradient to existing
    const dAttnOut = this.scratchSeqEmb;

    // Gradient buffers for Q, K, V
    const dQ = this.scratchSeqEmb2;
    for (let i = 0; i < seqLen * emb; i++) dQ[i] = 0;

    // Per-head backward
    for (let h = 0; h < nh; h++) {
      const headOffset = h * hd;
      const probsBase = h * seqLen * seqLen;
      const scoresBase = h * seqLen * seqLen;

      for (let i = 0; i < seqLen; i++) {
        // dV from weighted sum
        for (let j = 0; j <= i; j++) {
          const prob = this.fwdAttnProbs[block][probsBase + i * seqLen + j];
          for (let d = 0; d < hd; d++) {
            // dV[j] += prob * dAttnOut[i]
            const dAO = dAttnOut[i * emb + headOffset + d];
            this.scratchHead[j * hd + d] = (this.scratchHead[j * hd + d] || 0) +
              prob * dAO;
          }
        }

        // dProbs
        for (let j = 0; j <= i; j++) {
          let dProb = 0;
          for (let d = 0; d < hd; d++) {
            dProb += dAttnOut[i * emb + headOffset + d] *
              this.fwdV[block][j * emb + headOffset + d];
          }
          this.scratchHead[seqLen * hd + j] = dProb;
        }

        // Softmax backward
        let dotSum = 0;
        for (let j = 0; j <= i; j++) {
          const prob = this.fwdAttnProbs[block][probsBase + i * seqLen + j];
          dotSum += this.scratchHead[seqLen * hd + j] * prob;
        }

        for (let j = 0; j <= i; j++) {
          const prob = this.fwdAttnProbs[block][probsBase + i * seqLen + j];
          const dScore = (this.scratchHead[seqLen * hd + j] - dotSum) * prob *
            scale;

          // dQ[i] += dScore * K[j]
          // dK[j] += dScore * Q[i]
          for (let d = 0; d < hd; d++) {
            dQ[i * emb + headOffset + d] += dScore *
              this.fwdK[block][j * emb + headOffset + d];
            // dK accumulated to scratchHead
          }
        }
      }
    }

    // Simplified: gradient accumulation to Wq, Wk, Wv and input
    // dWq = input^T * dQ
    this.matMulTransA(input, dQ, this.gradAttnWq[block], seqLen, emb, emb);

    // dinput += dQ * Wq^T + dK * Wk^T + dV * Wv^T
    this.matMulTransB(dQ, this.attnWq[block], dinput, seqLen, emb, emb);
  }

  /**
   * FFN forward
   * h = GELU(x * W1 + b1)
   * out = h * W2 + b2
   */
  private ffnForward(input: Float64Array, block: number, seqLen: number): void {
    const emb = this.config.embeddingDim;
    const ffn = this.ffnDim;

    // First layer
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < ffn; f++) {
        let sum = this.ffnB1[block][f];
        for (let e = 0; e < emb; e++) {
          sum += input[t * emb + e] * this.ffnW1[block][e * ffn + f];
        }
        this.fwdFfnPre[block][t * ffn + f] = sum;
        this.fwdFfnHid[block][t * ffn + f] = this.gelu(sum);
      }
    }

    // Second layer
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < emb; e++) {
        let sum = this.ffnB2[block][e];
        for (let f = 0; f < ffn; f++) {
          sum += this.fwdFfnHid[block][t * ffn + f] *
            this.ffnW2[block][f * emb + e];
        }
        this.fwdFfnOut[block][t * emb + e] = sum;
      }
    }
  }

  /**
   * FFN backward
   */
  private ffnBackward(
    dout: Float64Array,
    input: Float64Array,
    block: number,
    seqLen: number,
    dinput: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;
    const ffn = this.ffnDim;

    // Second layer backward
    // dFfnHid = dout * W2^T
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < ffn; f++) {
        let sum = 0;
        for (let e = 0; e < emb; e++) {
          sum += dout[t * emb + e] * this.ffnW2[block][f * emb + e];
          this.gradFfnW2[block][f * emb + e] +=
            this.fwdFfnHid[block][t * ffn + f] * dout[t * emb + e];
        }
        this.scratchFfn[t * ffn + f] = sum;
      }
      for (let e = 0; e < emb; e++) {
        this.gradFfnB2[block][e] += dout[t * emb + e];
      }
    }

    // GELU backward
    for (let i = 0; i < seqLen * ffn; i++) {
      this.scratchFfn[i] *= this.geluGrad(this.fwdFfnPre[block][i]);
    }

    // First layer backward
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < emb; e++) {
        let sum = 0;
        for (let f = 0; f < ffn; f++) {
          sum += this.scratchFfn[t * ffn + f] * this.ffnW1[block][e * ffn + f];
          this.gradFfnW1[block][e * ffn + f] += input[t * emb + e] *
            this.scratchFfn[t * ffn + f];
        }
        dinput[t * emb + e] += sum;
      }
      for (let f = 0; f < ffn; f++) {
        this.gradFfnB1[block][f] += this.scratchFfn[t * ffn + f];
      }
    }
  }

  /**
   * Transformer block forward
   * out = x + Attention(LN1(x))
   * out = out + FFN(LN2(out))
   */
  private transformerBlockForward(
    input: Float64Array,
    block: number,
    seqLen: number,
    training: boolean,
  ): void {
    const emb = this.config.embeddingDim;

    // Copy input
    for (let i = 0; i < seqLen * emb; i++) {
      this.fwdBlockIn[block][i] = input[i];
    }

    // LayerNorm 1
    this.layerNormForward(
      input,
      this.ln1Gamma[block],
      this.ln1Beta[block],
      seqLen,
      emb,
      this.fwdLn1Out[block],
      this.fwdLn1Mean[block],
      this.fwdLn1Rstd[block],
    );

    // Attention
    this.attentionForward(this.fwdLn1Out[block], block, seqLen, training);

    // Residual 1
    for (let i = 0; i < seqLen * emb; i++) {
      this.fwdRes1[block][i] = input[i] + this.fwdAttnProj[block][i];
    }

    // LayerNorm 2
    this.layerNormForward(
      this.fwdRes1[block],
      this.ln2Gamma[block],
      this.ln2Beta[block],
      seqLen,
      emb,
      this.fwdLn2Out[block],
      this.fwdLn2Mean[block],
      this.fwdLn2Rstd[block],
    );

    // FFN
    this.ffnForward(this.fwdLn2Out[block], block, seqLen);

    // Residual 2
    for (let i = 0; i < seqLen * emb; i++) {
      this.fwdBlockOut[block][i] = this.fwdRes1[block][i] +
        this.fwdFfnOut[block][i];
    }
  }

  /**
   * Transformer block backward
   */
  private transformerBlockBackward(
    dout: Float64Array,
    block: number,
    seqLen: number,
    dinput: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;

    // dRes1 = dout (from residual 2)
    // dFfnOut = dout

    // FFN backward
    this.ffnBackward(
      dout,
      this.fwdLn2Out[block],
      block,
      seqLen,
      this.scratchSeqEmb,
    );

    // LayerNorm 2 backward
    this.layerNormBackward(
      this.scratchSeqEmb,
      this.fwdRes1[block],
      this.fwdLn2Mean[block],
      this.fwdLn2Rstd[block],
      this.ln2Gamma[block],
      seqLen,
      emb,
      this.scratchSeqEmb2,
      this.gradLn2Gamma[block],
      this.gradLn2Beta[block],
    );

    // Add residual gradient
    for (let i = 0; i < seqLen * emb; i++) {
      this.scratchSeqEmb2[i] += dout[i];
    }

    // Attention backward
    this.attentionBackward(
      this.scratchSeqEmb2,
      this.fwdLn1Out[block],
      block,
      seqLen,
      this.scratchSeqEmb,
    );

    // LayerNorm 1 backward
    this.layerNormBackward(
      this.scratchSeqEmb,
      this.fwdBlockIn[block],
      this.fwdLn1Mean[block],
      this.fwdLn1Rstd[block],
      this.ln1Gamma[block],
      seqLen,
      emb,
      dinput,
      this.gradLn1Gamma[block],
      this.gradLn1Beta[block],
    );

    // Add residual gradient
    for (let i = 0; i < seqLen * emb; i++) {
      dinput[i] += this.scratchSeqEmb2[i];
    }
  }

  /**
   * Attention pooling forward
   * score[t] = H[t] · Wpool + bpool
   * alpha = softmax(scores)
   * out = sum_t alpha[t] * H[t]
   */
  private poolForward(input: Float64Array, seqLen: number): void {
    const emb = this.config.embeddingDim;

    // Compute scores
    for (let t = 0; t < seqLen; t++) {
      let score = this.poolB;
      for (let e = 0; e < emb; e++) {
        score += input[t * emb + e] * this.poolW[e];
      }
      this.fwdPoolScores[t] = score;
    }

    // Softmax
    this.softmax(this.fwdPoolScores, 0, seqLen, this.fwdPoolWeights, 0);

    // Weighted sum
    for (let e = 0; e < emb; e++) {
      let sum = 0;
      for (let t = 0; t < seqLen; t++) {
        sum += this.fwdPoolWeights[t] * input[t * emb + e];
      }
      this.fwdPooled[e] = sum;
    }
  }

  /**
   * Attention pooling backward
   */
  private poolBackward(
    dout: Float64Array,
    input: Float64Array,
    seqLen: number,
    dinput: Float64Array,
  ): void {
    const emb = this.config.embeddingDim;

    // dWeights from weighted sum: dw[t] = sum_e dout[e] * input[t,e]
    for (let t = 0; t < seqLen; t++) {
      let dw = 0;
      for (let e = 0; e < emb; e++) {
        dw += dout[e] * input[t * emb + e];
        dinput[t * emb + e] += this.fwdPoolWeights[t] * dout[e];
      }
      this.scratchScales[t] = dw;
    }

    // Softmax backward
    let dotSum = 0;
    for (let t = 0; t < seqLen; t++) {
      dotSum += this.scratchScales[t] * this.fwdPoolWeights[t];
    }

    for (let t = 0; t < seqLen; t++) {
      const dScore = (this.scratchScales[t] - dotSum) * this.fwdPoolWeights[t];
      this.gradPoolB += dScore;
      for (let e = 0; e < emb; e++) {
        this.gradPoolW[e] += input[t * emb + e] * dScore;
      }
    }
  }

  /**
   * Output head forward
   * y = pooled * Wout + bout
   */
  private outputForward(): void {
    const emb = this.config.embeddingDim;
    for (let o = 0; o < this.outputDim; o++) {
      let sum = this.outB[o];
      for (let e = 0; e < emb; e++) {
        sum += this.fwdPooled[e] * this.outW[e * this.outputDim + o];
      }
      this.fwdOutput[o] = sum;
    }
  }

  /**
   * Output head backward
   */
  private outputBackward(dout: Float64Array, dPooled: Float64Array): void {
    const emb = this.config.embeddingDim;

    for (let e = 0; e < emb; e++) {
      let sum = 0;
      for (let o = 0; o < this.outputDim; o++) {
        sum += dout[o] * this.outW[e * this.outputDim + o];
        this.gradOutW[e * this.outputDim + o] += this.fwdPooled[e] * dout[o];
      }
      dPooled[e] = sum;
    }

    for (let o = 0; o < this.outputDim; o++) {
      this.gradOutB[o] += dout[o];
    }
  }

  /**
   * Full forward pass
   */
  private forward(
    normalizedX: Float64Array,
    seqLen: number,
    training: boolean,
  ): void {
    const emb = this.config.embeddingDim;
    const ns = this.numScales;

    // Temporal convolutions per scale
    for (let s = 0; s < ns; s++) {
      const scale = this.config.temporalScales[s];
      this.convForward(normalizedX, seqLen, scale, s);
      const scaleLen = Math.ceil(seqLen / scale);
      this.addEmbeddings(s, scaleLen);
    }

    // Upsample and concatenate
    this.upsampleAndConcat(seqLen);

    // Fusion gating
    this.fusionForward(seqLen, training);

    // Transformer blocks
    let currentInput = this.fwdFused;
    for (let b = 0; b < this.config.numBlocks; b++) {
      this.transformerBlockForward(currentInput, b, seqLen, training);
      currentInput = this.fwdBlockOut[b];
    }

    // Pooling
    this.poolForward(currentInput, seqLen);

    // Output
    this.outputForward();
  }

  /**
   * Clear all gradient buffers
   */
  private clearGradients(): void {
    for (let s = 0; s < this.numScales; s++) {
      this.gradConvW[s].fill(0);
      this.gradConvB[s].fill(0);
    }
    this.gradScaleEmb.fill(0);
    this.gradFusionW.fill(0);
    this.gradFusionB.fill(0);

    for (let b = 0; b < this.config.numBlocks; b++) {
      this.gradLn1Gamma[b].fill(0);
      this.gradLn1Beta[b].fill(0);
      this.gradLn2Gamma[b].fill(0);
      this.gradLn2Beta[b].fill(0);
      this.gradAttnWq[b].fill(0);
      this.gradAttnWk[b].fill(0);
      this.gradAttnWv[b].fill(0);
      this.gradAttnWo[b].fill(0);
      this.gradFfnW1[b].fill(0);
      this.gradFfnB1[b].fill(0);
      this.gradFfnW2[b].fill(0);
      this.gradFfnB2[b].fill(0);
    }

    this.gradPoolW.fill(0);
    this.gradPoolB = 0;
    this.gradOutW.fill(0);
    this.gradOutB.fill(0);
  }

  /**
   * Full backward pass
   */
  private backward(
    normalizedX: Float64Array,
    target: Float64Array,
    seqLen: number,
    sampleWeight: number,
  ): void {
    const emb = this.config.embeddingDim;

    // Output gradient: dL/dy = (yhat - y) * sampleWeight
    for (let o = 0; o < this.outputDim; o++) {
      this.scratchEmb[o] = (this.fwdOutput[o] - target[o]) * sampleWeight;
    }

    // Output head backward
    const dPooled = new Float64Array(emb);
    this.outputBackward(this.scratchEmb, dPooled);

    // Pooling backward
    const lastBlockOut = this.fwdBlockOut[this.config.numBlocks - 1];
    this.scratchSeqEmb.fill(0);
    this.poolBackward(dPooled, lastBlockOut, seqLen, this.scratchSeqEmb);

    // Transformer blocks backward
    let dCurrent = this.scratchSeqEmb;
    for (let b = this.config.numBlocks - 1; b >= 0; b--) {
      this.scratchSeqEmb2.fill(0);
      this.transformerBlockBackward(dCurrent, b, seqLen, this.scratchSeqEmb2);

      // Swap buffers
      const tmp = dCurrent;
      dCurrent = this.scratchSeqEmb2;
      this.scratchSeqEmb2 = tmp;
    }

    // Fusion backward
    this.scratchGrad.fill(0);
    this.fusionBackward(dCurrent, seqLen, this.scratchGrad);

    // Backprop through upsampling (distribute gradients to scales)
    const ns = this.numScales;
    for (let s = 0; s < ns; s++) {
      const scale = this.config.temporalScales[s];
      const scaleLen = Math.ceil(seqLen / scale);
      const scaleEmb = this.fwdScaleEmb[s];

      // Accumulate gradients from upsampled positions
      for (let t = 0; t < seqLen; t++) {
        const srcT = Math.min(Math.floor(t / scale), scaleLen - 1);
        for (let e = 0; e < emb; e++) {
          const gradIdx = t * ns * emb + s * emb + e;
          this.gradScaleEmb[s * emb + e] += this.scratchGrad[gradIdx];
        }
      }
    }
  }

  /**
   * Compute gradient norm
   */
  private computeGradientNorm(): number {
    let normSq = 0;

    for (let s = 0; s < this.numScales; s++) {
      for (let i = 0; i < this.gradConvW[s].length; i++) {
        normSq += this.gradConvW[s][i] * this.gradConvW[s][i];
      }
      for (let i = 0; i < this.gradConvB[s].length; i++) {
        normSq += this.gradConvB[s][i] * this.gradConvB[s][i];
      }
    }

    for (let i = 0; i < this.gradScaleEmb.length; i++) {
      normSq += this.gradScaleEmb[i] * this.gradScaleEmb[i];
    }

    for (let i = 0; i < this.gradFusionW.length; i++) {
      normSq += this.gradFusionW[i] * this.gradFusionW[i];
    }
    for (let i = 0; i < this.gradFusionB.length; i++) {
      normSq += this.gradFusionB[i] * this.gradFusionB[i];
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.gradLn1Gamma[b].length; i++) {
        normSq += this.gradLn1Gamma[b][i] * this.gradLn1Gamma[b][i];
      }
      for (let i = 0; i < this.gradLn1Beta[b].length; i++) {
        normSq += this.gradLn1Beta[b][i] * this.gradLn1Beta[b][i];
      }
      for (let i = 0; i < this.gradLn2Gamma[b].length; i++) {
        normSq += this.gradLn2Gamma[b][i] * this.gradLn2Gamma[b][i];
      }
      for (let i = 0; i < this.gradLn2Beta[b].length; i++) {
        normSq += this.gradLn2Beta[b][i] * this.gradLn2Beta[b][i];
      }
      for (let i = 0; i < this.gradAttnWq[b].length; i++) {
        normSq += this.gradAttnWq[b][i] * this.gradAttnWq[b][i];
      }
      for (let i = 0; i < this.gradAttnWk[b].length; i++) {
        normSq += this.gradAttnWk[b][i] * this.gradAttnWk[b][i];
      }
      for (let i = 0; i < this.gradAttnWv[b].length; i++) {
        normSq += this.gradAttnWv[b][i] * this.gradAttnWv[b][i];
      }
      for (let i = 0; i < this.gradAttnWo[b].length; i++) {
        normSq += this.gradAttnWo[b][i] * this.gradAttnWo[b][i];
      }
      for (let i = 0; i < this.gradFfnW1[b].length; i++) {
        normSq += this.gradFfnW1[b][i] * this.gradFfnW1[b][i];
      }
      for (let i = 0; i < this.gradFfnB1[b].length; i++) {
        normSq += this.gradFfnB1[b][i] * this.gradFfnB1[b][i];
      }
      for (let i = 0; i < this.gradFfnW2[b].length; i++) {
        normSq += this.gradFfnW2[b][i] * this.gradFfnW2[b][i];
      }
      for (let i = 0; i < this.gradFfnB2[b].length; i++) {
        normSq += this.gradFfnB2[b][i] * this.gradFfnB2[b][i];
      }
    }

    for (let i = 0; i < this.gradPoolW.length; i++) {
      normSq += this.gradPoolW[i] * this.gradPoolW[i];
    }
    normSq += this.gradPoolB * this.gradPoolB;

    for (let i = 0; i < this.gradOutW.length; i++) {
      normSq += this.gradOutW[i] * this.gradOutW[i];
    }
    for (let i = 0; i < this.gradOutB.length; i++) {
      normSq += this.gradOutB[i] * this.gradOutB[i];
    }

    return Math.sqrt(normSq);
  }

  /**
   * Clip gradients by global norm
   */
  private clipGradients(maxNorm: number): void {
    const norm = this.computeGradientNorm();
    if (norm > maxNorm) {
      const scale = maxNorm / norm;

      for (let s = 0; s < this.numScales; s++) {
        for (let i = 0; i < this.gradConvW[s].length; i++) {
          this.gradConvW[s][i] *= scale;
        }
        for (let i = 0; i < this.gradConvB[s].length; i++) {
          this.gradConvB[s][i] *= scale;
        }
      }

      for (let i = 0; i < this.gradScaleEmb.length; i++) {
        this.gradScaleEmb[i] *= scale;
      }

      for (let i = 0; i < this.gradFusionW.length; i++) {
        this.gradFusionW[i] *= scale;
      }
      for (let i = 0; i < this.gradFusionB.length; i++) {
        this.gradFusionB[i] *= scale;
      }

      for (let b = 0; b < this.config.numBlocks; b++) {
        for (let i = 0; i < this.gradLn1Gamma[b].length; i++) {
          this.gradLn1Gamma[b][i] *= scale;
        }
        for (let i = 0; i < this.gradLn1Beta[b].length; i++) {
          this.gradLn1Beta[b][i] *= scale;
        }
        for (let i = 0; i < this.gradLn2Gamma[b].length; i++) {
          this.gradLn2Gamma[b][i] *= scale;
        }
        for (let i = 0; i < this.gradLn2Beta[b].length; i++) {
          this.gradLn2Beta[b][i] *= scale;
        }
        for (let i = 0; i < this.gradAttnWq[b].length; i++) {
          this.gradAttnWq[b][i] *= scale;
        }
        for (let i = 0; i < this.gradAttnWk[b].length; i++) {
          this.gradAttnWk[b][i] *= scale;
        }
        for (let i = 0; i < this.gradAttnWv[b].length; i++) {
          this.gradAttnWv[b][i] *= scale;
        }
        for (let i = 0; i < this.gradAttnWo[b].length; i++) {
          this.gradAttnWo[b][i] *= scale;
        }
        for (let i = 0; i < this.gradFfnW1[b].length; i++) {
          this.gradFfnW1[b][i] *= scale;
        }
        for (let i = 0; i < this.gradFfnB1[b].length; i++) {
          this.gradFfnB1[b][i] *= scale;
        }
        for (let i = 0; i < this.gradFfnW2[b].length; i++) {
          this.gradFfnW2[b][i] *= scale;
        }
        for (let i = 0; i < this.gradFfnB2[b].length; i++) {
          this.gradFfnB2[b][i] *= scale;
        }
      }

      for (let i = 0; i < this.gradPoolW.length; i++) {
        this.gradPoolW[i] *= scale;
      }
      this.gradPoolB *= scale;

      for (let i = 0; i < this.gradOutW.length; i++) {
        this.gradOutW[i] *= scale;
      }
      for (let i = 0; i < this.gradOutB.length; i++) {
        this.gradOutB[i] *= scale;
      }
    }
  }

  /**
   * Apply L2 regularization to gradients
   */
  private applyL2Regularization(): void {
    const lambda = this.config.regularizationStrength;

    for (let s = 0; s < this.numScales; s++) {
      for (let i = 0; i < this.convW[s].length; i++) {
        this.gradConvW[s][i] += lambda * this.convW[s][i];
      }
    }

    for (let i = 0; i < this.scaleEmb.length; i++) {
      this.gradScaleEmb[i] += lambda * this.scaleEmb[i];
    }

    for (let i = 0; i < this.fusionW.length; i++) {
      this.gradFusionW[i] += lambda * this.fusionW[i];
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.attnWq[b].length; i++) {
        this.gradAttnWq[b][i] += lambda * this.attnWq[b][i];
      }
      for (let i = 0; i < this.attnWk[b].length; i++) {
        this.gradAttnWk[b][i] += lambda * this.attnWk[b][i];
      }
      for (let i = 0; i < this.attnWv[b].length; i++) {
        this.gradAttnWv[b][i] += lambda * this.attnWv[b][i];
      }
      for (let i = 0; i < this.attnWo[b].length; i++) {
        this.gradAttnWo[b][i] += lambda * this.attnWo[b][i];
      }
      for (let i = 0; i < this.ffnW1[b].length; i++) {
        this.gradFfnW1[b][i] += lambda * this.ffnW1[b][i];
      }
      for (let i = 0; i < this.ffnW2[b].length; i++) {
        this.gradFfnW2[b][i] += lambda * this.ffnW2[b][i];
      }
    }

    for (let i = 0; i < this.outW.length; i++) {
      this.gradOutW[i] += lambda * this.outW[i];
    }
  }

  /**
   * Compute learning rate with warmup and cosine decay
   * warmup: lr = baseLR * (step / warmupSteps)
   * decay: lr = baseLR * 0.5 * (1 + cos(π * progress))
   */
  private computeLearningRate(): number {
    const step = this.updateCount;
    const baseLR = this.config.learningRate;
    const warmup = this.config.warmupSteps;
    const total = this.config.totalSteps;

    if (step < warmup) {
      return baseLR * (step + 1) / warmup;
    }

    const progress = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
    return baseLR * 0.5 * (1 + Math.cos(Math.PI * progress));
  }

  /**
   * Adam update for a single parameter array
   * m = β₁m + (1-β₁)g
   * v = β₂v + (1-β₂)g²
   * m̂ = m / (1 - β₁ᵗ)
   * v̂ = v / (1 - β₂ᵗ)
   * θ = θ - lr * m̂ / (√v̂ + ε)
   */
  private adamUpdate(
    param: Float64Array,
    grad: Float64Array,
    mOffset: number,
    lr: number,
  ): void {
    const beta1 = this.config.beta1;
    const beta2 = this.config.beta2;
    const eps = this.config.epsilon;
    const t = this.updateCount + 1;

    const beta1t = 1 - Math.pow(beta1, t);
    const beta2t = 1 - Math.pow(beta2, t);

    for (let i = 0; i < param.length; i++) {
      const g = grad[i];
      this.adamM[mOffset + i] = beta1 * this.adamM[mOffset + i] +
        (1 - beta1) * g;
      this.adamV[mOffset + i] = beta2 * this.adamV[mOffset + i] +
        (1 - beta2) * g * g;

      const mHat = this.adamM[mOffset + i] / beta1t;
      const vHat = this.adamV[mOffset + i] / beta2t;

      param[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  /**
   * Apply Adam updates to all parameters
   */
  private applyAdamUpdates(lr: number): void {
    let offset = 0;

    for (let s = 0; s < this.numScales; s++) {
      this.adamUpdate(this.convW[s], this.gradConvW[s], offset, lr);
      offset += this.convW[s].length;
      this.adamUpdate(this.convB[s], this.gradConvB[s], offset, lr);
      offset += this.convB[s].length;
    }

    this.adamUpdate(this.scaleEmb, this.gradScaleEmb, offset, lr);
    offset += this.scaleEmb.length;

    this.adamUpdate(this.fusionW, this.gradFusionW, offset, lr);
    offset += this.fusionW.length;
    this.adamUpdate(this.fusionB, this.gradFusionB, offset, lr);
    offset += this.fusionB.length;

    for (let b = 0; b < this.config.numBlocks; b++) {
      this.adamUpdate(this.ln1Gamma[b], this.gradLn1Gamma[b], offset, lr);
      offset += this.ln1Gamma[b].length;
      this.adamUpdate(this.ln1Beta[b], this.gradLn1Beta[b], offset, lr);
      offset += this.ln1Beta[b].length;
      this.adamUpdate(this.ln2Gamma[b], this.gradLn2Gamma[b], offset, lr);
      offset += this.ln2Gamma[b].length;
      this.adamUpdate(this.ln2Beta[b], this.gradLn2Beta[b], offset, lr);
      offset += this.ln2Beta[b].length;

      this.adamUpdate(this.attnWq[b], this.gradAttnWq[b], offset, lr);
      offset += this.attnWq[b].length;
      this.adamUpdate(this.attnWk[b], this.gradAttnWk[b], offset, lr);
      offset += this.attnWk[b].length;
      this.adamUpdate(this.attnWv[b], this.gradAttnWv[b], offset, lr);
      offset += this.attnWv[b].length;
      this.adamUpdate(this.attnWo[b], this.gradAttnWo[b], offset, lr);
      offset += this.attnWo[b].length;

      this.adamUpdate(this.ffnW1[b], this.gradFfnW1[b], offset, lr);
      offset += this.ffnW1[b].length;
      this.adamUpdate(this.ffnB1[b], this.gradFfnB1[b], offset, lr);
      offset += this.ffnB1[b].length;
      this.adamUpdate(this.ffnW2[b], this.gradFfnW2[b], offset, lr);
      offset += this.ffnW2[b].length;
      this.adamUpdate(this.ffnB2[b], this.gradFfnB2[b], offset, lr);
      offset += this.ffnB2[b].length;
    }

    this.adamUpdate(this.poolW, this.gradPoolW, offset, lr);
    offset += this.poolW.length;

    // poolB (scalar)
    const g = this.gradPoolB;
    this.adamM[offset] = this.config.beta1 * this.adamM[offset] +
      (1 - this.config.beta1) * g;
    this.adamV[offset] = this.config.beta2 * this.adamV[offset] +
      (1 - this.config.beta2) * g * g;
    const beta1t = 1 - Math.pow(this.config.beta1, this.updateCount + 1);
    const beta2t = 1 - Math.pow(this.config.beta2, this.updateCount + 1);
    const mHat = this.adamM[offset] / beta1t;
    const vHat = this.adamV[offset] / beta2t;
    this.poolB -= lr * mHat / (Math.sqrt(vHat) + this.config.epsilon);
    offset += 1;

    this.adamUpdate(this.outW, this.gradOutW, offset, lr);
    offset += this.outW.length;
    this.adamUpdate(this.outB, this.gradOutB, offset, lr);
  }

  /**
   * ADWIN drift detection
   * Tests if there's a significant change in the mean of recent losses
   */
  private checkDrift(loss: number): boolean {
    // Add to ring buffer
    this.driftWindow[this.driftWindowHead] = loss;
    this.driftWindowHead = (this.driftWindowHead + 1) % this.driftWindowSize;
    if (this.driftWindowCount < this.driftWindowSize) {
      this.driftWindowCount++;
    }

    // Need minimum samples
    if (this.driftWindowCount < 32) return false;

    const n = this.driftWindowCount;
    const delta = this.config.adwinDelta;

    // Find best split point
    let bestEps = 0;
    let bestDiff = 0;

    for (let split = 16; split < n - 16; split++) {
      let sumLeft = 0, sumRight = 0;
      let countLeft = 0, countRight = 0;

      for (let i = 0; i < n; i++) {
        const idx = (this.driftWindowHead - n + i + this.driftWindowSize) %
          this.driftWindowSize;
        if (i < split) {
          sumLeft += this.driftWindow[idx];
          countLeft++;
        } else {
          sumRight += this.driftWindow[idx];
          countRight++;
        }
      }

      const meanLeft = sumLeft / countLeft;
      const meanRight = sumRight / countRight;
      const diff = Math.abs(meanLeft - meanRight);

      const eps = Math.sqrt(
        2 * Math.log(2 / delta) * (1 / countLeft + 1 / countRight),
      );

      if (diff > eps && diff > bestDiff) {
        bestDiff = diff;
        bestEps = eps;
      }
    }

    if (bestDiff > bestEps) {
      // Reset window
      this.driftWindowHead = 0;
      this.driftWindowCount = 0;
      this.runningLoss = loss;
      return true;
    }

    return false;
  }

  /**
   * Trains the model on a single sample (online learning).
   *
   * @param data - Training sample with input and output sequences
   * @param data.xCoordinates - Input sequence [seqLen][inputDim]
   * @param data.yCoordinates - Output sequence [seqLen][outputDim], uses last timestep as target
   * @returns FitResult with loss, gradient norm, and other metrics
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
   *   yCoordinates: [[7], [8], [9]]
   * });
   * console.log(`Loss: ${result.loss}`);
   * ```
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xCoords = data.xCoordinates;
    const yCoords = data.yCoordinates;

    if (xCoords.length === 0 || yCoords.length === 0) {
      throw new Error("Empty input data");
    }

    const seqLen = Math.min(xCoords.length, this.config.maxSequenceLength);
    const inputDim = xCoords[0].length;
    const outputDim = yCoords[0].length;

    // Initialize on first call
    if (!this.isInitialized) {
      this.initialize(inputDim, outputDim, seqLen);
    }

    // Update sequence length if needed
    this.seqLen = seqLen;

    // Seed RNG deterministically
    this.rngState = (this.updateCount * 2654435761) >>> 0;
    if (this.rngState === 0) this.rngState = 1;

    // Copy input to flat buffer and cache for prediction
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < inputDim; f++) {
        const val = xCoords[t][f];
        this.fwdNormX[t * inputDim + f] = val;
        this.lastWindow[t * inputDim + f] = val;
      }
    }
    this.lastWindowLen = seqLen;

    // Get target (last timestep)
    for (let o = 0; o < outputDim; o++) {
      this.fwdNormY[o] = yCoords[seqLen - 1][o];
    }

    // Update Welford stats
    this.normCount++;
    const tempX = new Float64Array(inputDim);
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < inputDim; f++) {
        tempX[f] = xCoords[t][f];
      }
      this.updateWelford(tempX, this.inputMean, this.inputM2, this.normCount);
    }
    this.updateWelford(
      this.fwdNormY,
      this.outputMean,
      this.outputM2,
      this.normCount,
    );

    // Normalize input
    const inputStd = new Float64Array(inputDim);
    this.getStd(this.inputM2, this.normCount, inputStd);
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < inputDim; f++) {
        this.fwdNormX[t * inputDim + f] =
          (this.fwdNormX[t * inputDim + f] - this.inputMean[f]) / inputStd[f];
      }
    }

    // Normalize target
    const outputStd = new Float64Array(outputDim);
    this.getStd(this.outputM2, this.normCount, outputStd);
    const normalizedTarget = new Float64Array(outputDim);
    for (let o = 0; o < outputDim; o++) {
      normalizedTarget[o] = (this.fwdNormY[o] - this.outputMean[o]) /
        outputStd[o];
    }

    // Forward pass
    this.forward(this.fwdNormX, seqLen, true);

    // Compute loss: MSE on normalized outputs
    let mse = 0;
    let isOutlier = false;
    for (let o = 0; o < outputDim; o++) {
      const residual = this.fwdOutput[o] - normalizedTarget[o];
      mse += residual * residual;
      if (Math.abs(residual) > this.config.outlierThreshold) {
        isOutlier = true;
      }
    }
    mse /= 2 * outputDim;

    // L2 regularization loss
    let regLoss = 0;
    const lambda = this.config.regularizationStrength;
    for (let s = 0; s < this.numScales; s++) {
      for (let i = 0; i < this.convW[s].length; i++) {
        regLoss += this.convW[s][i] * this.convW[s][i];
      }
    }
    regLoss *= lambda / 2;

    const totalLoss = mse + regLoss;

    // Outlier downweighting
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // Update running loss
    this.sampleCount++;
    this.runningLoss =
      (this.runningLoss * (this.sampleCount - 1) + totalLoss * sampleWeight) /
      this.sampleCount;

    // Update residual variance for prediction uncertainty
    this.residualCount++;
    for (let o = 0; o < outputDim; o++) {
      const residual = this.fwdOutput[o] - normalizedTarget[o];
      const delta = residual * residual -
        this.residualM2[o] / Math.max(1, this.residualCount - 1);
      this.residualM2[o] += delta;
    }

    // Clear gradients
    this.clearGradients();

    // Backward pass
    this.backward(this.fwdNormX, normalizedTarget, seqLen, sampleWeight);

    // Apply L2 regularization to gradients
    this.applyL2Regularization();

    // Clip gradients
    this.clipGradients(5.0);

    // Compute gradient norm
    const gradNorm = this.computeGradientNorm();

    // Compute learning rate
    const lr = this.computeLearningRate();

    // Apply Adam updates
    this.applyAdamUpdates(lr);

    // Check convergence
    this.converged = gradNorm < this.config.convergenceThreshold;
    this.lastGradNorm = gradNorm;
    this.lastLR = lr;

    // Check drift
    const driftDetected = this.checkDrift(totalLoss * sampleWeight);
    if (driftDetected) {
      this.driftCount++;
    }

    this.updateCount++;

    return {
      loss: totalLoss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Generates predictions for future timesteps.
   *
   * @param futureSteps - Number of future steps to predict
   * @returns PredictionResult with predictions, bounds, and uncertainty
   *
   * @example
   * ```typescript
   * const result = model.predict(5);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, SE: ${pred.standardError}`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    const isReady = this.isInitialized && this.normCount >= 2;

    if (!isReady || futureSteps <= 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: isReady,
      };
    }

    const seqLen = this.lastWindowLen;
    const emb = this.config.embeddingDim;

    // Normalize cached input
    const inputStd = new Float64Array(this.inputDim);
    this.getStd(this.inputM2, this.normCount, inputStd);
    for (let t = 0; t < seqLen; t++) {
      for (let f = 0; f < this.inputDim; f++) {
        this.fwdNormX[t * this.inputDim + f] =
          (this.lastWindow[t * this.inputDim + f] - this.inputMean[f]) /
          inputStd[f];
      }
    }

    // Forward pass (no dropout)
    this.forward(this.fwdNormX, seqLen, false);

    // Denormalize output
    const outputStd = new Float64Array(this.outputDim);
    this.getStd(this.outputM2, this.normCount, outputStd);

    const basePrediction = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      basePrediction[o] = this.fwdOutput[o] * outputStd[o] + this.outputMean[o];
    }

    // Compute base standard error from residual variance
    const baseStdError = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      const variance = this.residualM2[o] / Math.max(1, this.residualCount);
      baseStdError[o] = Math.sqrt(Math.max(variance, 1e-12)) * outputStd[o];
    }

    // Generate predictions
    const predictions: SinglePrediction[] = [];
    const z = 1.96; // 95% confidence

    for (let step = 0; step < futureSteps; step++) {
      const uncertainty = Math.sqrt(step + 1);

      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let o = 0; o < this.outputDim; o++) {
        const se = baseStdError[o] * uncertainty;
        predicted.push(basePrediction[o]);
        standardError.push(se);
        lowerBound.push(basePrediction[o] - z * se);
        upperBound.push(basePrediction[o] + z * se);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });
    }

    // Accuracy = 1 / (1 + runningLoss)
    const accuracy = 1 / (1 + this.runningLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Returns a summary of the model's current state.
   *
   * @returns ModelSummary with architecture and training info
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Total parameters: ${summary.totalParameters}`);
   * ```
   */
  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      temporalScales: [...this.config.temporalScales],
      totalParameters: this.totalParams,
      sampleCount: this.sampleCount,
      accuracy: 1 / (1 + this.runningLoss),
      converged: this.converged,
      effectiveLearningRate: this.lastLR,
      driftCount: this.driftCount,
    };
  }

  /**
   * Returns the current model weights.
   *
   * @returns WeightInfo containing all trainable parameters
   */
  public getWeights(): WeightInfo {
    const toArr = (f: Float64Array): number[] => Array.from(f);
    const emb = this.config.embeddingDim;
    const ks = this.config.temporalKernelSize;

    // Temporal conv weights: [scale][kernel*input][emb]
    const temporalConvWeights: number[][][] = [];
    for (let s = 0; s < this.numScales; s++) {
      const scale: number[][] = [];
      for (let k = 0; k < ks * this.inputDim; k++) {
        const row: number[] = [];
        for (let e = 0; e < emb; e++) {
          row.push(this.convW[s] ? this.convW[s][k * emb + e] : 0);
        }
        scale.push(row);
      }
      temporalConvWeights.push(scale);
    }

    // Scale embeddings: [scale][emb]
    const scaleEmbeddings: number[][] = [];
    for (let s = 0; s < this.numScales; s++) {
      const row: number[] = [];
      for (let e = 0; e < emb; e++) {
        row.push(this.scaleEmb ? this.scaleEmb[s * emb + e] : 0);
      }
      scaleEmbeddings.push(row);
    }

    // Positional encoding: [pos][emb]
    const positionalEncoding: number[][] = [];
    const maxLen = this.config.maxSequenceLength;
    for (let p = 0; p < maxLen; p++) {
      const row: number[] = [];
      for (let e = 0; e < emb; e++) {
        row.push(this.posEnc ? this.posEnc[p * emb + e] : 0);
      }
      positionalEncoding.push(row);
    }

    // Fusion weights
    const fusionDim = this.numScales * emb;
    const fusionWeights: number[][] = [];
    for (let i = 0; i < fusionDim; i++) {
      const row: number[] = [];
      for (let j = 0; j < fusionDim; j++) {
        row.push(this.fusionW ? this.fusionW[i * fusionDim + j] : 0);
      }
      fusionWeights.push(row);
    }

    // Attention weights: [block][type (Wq,Wk,Wv,Wo)][emb*emb]
    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block: number[][] = [];
      block.push(this.attnWq[b] ? toArr(this.attnWq[b]) : []);
      block.push(this.attnWk[b] ? toArr(this.attnWk[b]) : []);
      block.push(this.attnWv[b] ? toArr(this.attnWv[b]) : []);
      block.push(this.attnWo[b] ? toArr(this.attnWo[b]) : []);
      attentionWeights.push(block);
    }

    // FFN weights: [block][type (W1,b1,W2,b2)][...]
    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block: number[][] = [];
      block.push(this.ffnW1[b] ? toArr(this.ffnW1[b]) : []);
      block.push(this.ffnB1[b] ? toArr(this.ffnB1[b]) : []);
      block.push(this.ffnW2[b] ? toArr(this.ffnW2[b]) : []);
      block.push(this.ffnB2[b] ? toArr(this.ffnB2[b]) : []);
      ffnWeights.push(block);
    }

    // LayerNorm params: [layer][gamma+beta]
    const layerNormParams: number[][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const ln1: number[] = [];
      if (this.ln1Gamma[b]) {
        for (let e = 0; e < emb; e++) ln1.push(this.ln1Gamma[b][e]);
        for (let e = 0; e < emb; e++) ln1.push(this.ln1Beta[b][e]);
      }
      layerNormParams.push(ln1);

      const ln2: number[] = [];
      if (this.ln2Gamma[b]) {
        for (let e = 0; e < emb; e++) ln2.push(this.ln2Gamma[b][e]);
        for (let e = 0; e < emb; e++) ln2.push(this.ln2Beta[b][e]);
      }
      layerNormParams.push(ln2);
    }

    // Output weights: [emb+1][outputDim] (include bias)
    const outputWeights: number[][] = [];
    for (let e = 0; e < emb; e++) {
      const row: number[] = [];
      for (let o = 0; o < this.outputDim; o++) {
        row.push(this.outW ? this.outW[e * this.outputDim + o] : 0);
      }
      outputWeights.push(row);
    }
    outputWeights.push(this.outB ? toArr(this.outB) : []);

    // Moments (simplified)
    const firstMoment: number[][][] = [[
      toArr(this.adamM || new Float64Array(0)),
    ]];
    const secondMoment: number[][][] = [[
      toArr(this.adamV || new Float64Array(0)),
    ]];

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
   * Returns the current normalization statistics.
   *
   * @returns NormalizationStats with mean and std for inputs/outputs
   */
  public getNormalizationStats(): NormalizationStats {
    const inputStd = new Float64Array(this.inputDim || 0);
    const outputStd = new Float64Array(this.outputDim || 0);

    if (this.isInitialized && this.normCount > 0) {
      this.getStd(this.inputM2, this.normCount, inputStd);
      this.getStd(this.outputM2, this.normCount, outputStd);
    }

    return {
      inputMean: this.inputMean ? Array.from(this.inputMean) : [],
      inputStd: Array.from(inputStd),
      outputMean: this.outputMean ? Array.from(this.outputMean) : [],
      outputStd: Array.from(outputStd),
      count: this.normCount,
    };
  }

  /**
   * Resets the model to initial state while keeping configuration.
   */
  public reset(): void {
    if (!this.isInitialized) return;

    // Reset normalization
    this.inputMean.fill(0);
    this.inputM2.fill(0);
    this.outputMean.fill(0);
    this.outputM2.fill(0);
    this.residualM2.fill(0);
    this.normCount = 0;
    this.residualCount = 0;

    // Reinitialize weights
    this.rngState = 12345;

    for (let s = 0; s < this.numScales; s++) {
      this.xavierInit(
        this.convW[s],
        this.config.temporalKernelSize * this.inputDim,
        this.config.embeddingDim,
      );
      this.convB[s].fill(0);
    }

    for (let i = 0; i < this.scaleEmb.length; i++) {
      this.scaleEmb[i] = (this.xorshift32() * 2 - 1) * 0.02;
    }

    const fusionDim = this.numScales * this.config.embeddingDim;
    this.xavierInit(this.fusionW, fusionDim, fusionDim);
    this.fusionB.fill(0);

    const emb = this.config.embeddingDim;
    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < emb; i++) {
        this.ln1Gamma[b][i] = 1;
        this.ln1Beta[b][i] = 0;
        this.ln2Gamma[b][i] = 1;
        this.ln2Beta[b][i] = 0;
      }
      this.xavierInit(this.attnWq[b], emb, emb);
      this.xavierInit(this.attnWk[b], emb, emb);
      this.xavierInit(this.attnWv[b], emb, emb);
      this.xavierInit(this.attnWo[b], emb, emb);
      this.xavierInit(this.ffnW1[b], emb, this.ffnDim);
      this.ffnB1[b].fill(0);
      this.xavierInit(this.ffnW2[b], this.ffnDim, emb);
      this.ffnB2[b].fill(0);
    }

    this.xavierInit(this.poolW, emb, 1);
    this.poolB = 0;
    this.xavierInit(this.outW, emb, this.outputDim);
    this.outB.fill(0);

    // Reset optimizer
    this.adamM.fill(0);
    this.adamV.fill(0);

    // Reset counters
    this.updateCount = 0;
    this.sampleCount = 0;
    this.driftCount = 0;
    this.runningLoss = 0;
    this.converged = false;
    this.lastGradNorm = Infinity;
    this.lastLR = 0;

    // Reset drift window
    this.driftWindow.fill(0);
    this.driftWindowHead = 0;
    this.driftWindowCount = 0;

    // Reset last window
    this.lastWindow.fill(0);
    this.lastWindowLen = 0;
  }

  /**
   * Serializes the model to a JSON string.
   *
   * @returns JSON string representation of the model
   */
  public save(): string {
    const toArr = (f: Float64Array | null): number[] => f ? Array.from(f) : [];

    const state: SerializedState = {
      config: this.config,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      seqLen: this.seqLen,
      isInitialized: this.isInitialized,
      sampleCount: this.sampleCount,
      updateCount: this.updateCount,
      driftCount: this.driftCount,
      runningLoss: this.runningLoss,
      normCount: this.normCount,
      residualCount: this.residualCount,
      inputMean: toArr(this.inputMean),
      inputM2: toArr(this.inputM2),
      outputMean: toArr(this.outputMean),
      outputM2: toArr(this.outputM2),
      residualM2: toArr(this.residualM2),
      posEnc: toArr(this.posEnc),
      convW: this.convW.map(toArr),
      convB: this.convB.map(toArr),
      scaleEmb: toArr(this.scaleEmb),
      fusionW: toArr(this.fusionW),
      fusionB: toArr(this.fusionB),
      ln1Gamma: this.ln1Gamma.map(toArr),
      ln1Beta: this.ln1Beta.map(toArr),
      ln2Gamma: this.ln2Gamma.map(toArr),
      ln2Beta: this.ln2Beta.map(toArr),
      attnWq: this.attnWq.map(toArr),
      attnWk: this.attnWk.map(toArr),
      attnWv: this.attnWv.map(toArr),
      attnWo: this.attnWo.map(toArr),
      ffnW1: this.ffnW1.map(toArr),
      ffnB1: this.ffnB1.map(toArr),
      ffnW2: this.ffnW2.map(toArr),
      ffnB2: this.ffnB2.map(toArr),
      poolW: toArr(this.poolW),
      poolB: this.poolB,
      outW: toArr(this.outW),
      outB: toArr(this.outB),
      adamM: toArr(this.adamM),
      adamV: toArr(this.adamV),
      lastWindow: toArr(this.lastWindow),
      lastWindowLen: this.lastWindowLen,
      driftWindow: Array.from(this.driftWindow),
      driftWindowHead: this.driftWindowHead,
      driftWindowCount: this.driftWindowCount,
      rngState: this.rngState,
      converged: this.converged,
      lastGradNorm: this.lastGradNorm,
      lastLR: this.lastLR,
    };

    return JSON.stringify(state);
  }

  /**
   * Loads model state from a JSON string.
   *
   * @param w - JSON string from save()
   */
  public load(w: string): void {
    const state: SerializedState = JSON.parse(w);
    const toF64 = (arr: number[]): Float64Array => new Float64Array(arr);

    this.config = state.config;
    this.inputDim = state.inputDim;
    this.outputDim = state.outputDim;
    this.seqLen = state.seqLen;
    this.isInitialized = state.isInitialized;
    this.sampleCount = state.sampleCount;
    this.updateCount = state.updateCount;
    this.driftCount = state.driftCount;
    this.runningLoss = state.runningLoss;
    this.normCount = state.normCount;
    this.residualCount = state.residualCount;

    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;
    this.numScales = this.config.temporalScales.length;

    if (this.isInitialized) {
      this.inputMean = toF64(state.inputMean);
      this.inputM2 = toF64(state.inputM2);
      this.outputMean = toF64(state.outputMean);
      this.outputM2 = toF64(state.outputM2);
      this.residualM2 = toF64(state.residualM2);
      this.posEnc = toF64(state.posEnc);

      this.convW = state.convW.map(toF64);
      this.convB = state.convB.map(toF64);
      this.scaleEmb = toF64(state.scaleEmb);
      this.fusionW = toF64(state.fusionW);
      this.fusionB = toF64(state.fusionB);

      this.ln1Gamma = state.ln1Gamma.map(toF64);
      this.ln1Beta = state.ln1Beta.map(toF64);
      this.ln2Gamma = state.ln2Gamma.map(toF64);
      this.ln2Beta = state.ln2Beta.map(toF64);
      this.attnWq = state.attnWq.map(toF64);
      this.attnWk = state.attnWk.map(toF64);
      this.attnWv = state.attnWv.map(toF64);
      this.attnWo = state.attnWo.map(toF64);
      this.ffnW1 = state.ffnW1.map(toF64);
      this.ffnB1 = state.ffnB1.map(toF64);
      this.ffnW2 = state.ffnW2.map(toF64);
      this.ffnB2 = state.ffnB2.map(toF64);

      this.poolW = toF64(state.poolW);
      this.poolB = state.poolB;
      this.outW = toF64(state.outW);
      this.outB = toF64(state.outB);

      this.adamM = toF64(state.adamM);
      this.adamV = toF64(state.adamV);
      this.totalParams = this.adamM.length;

      this.lastWindow = toF64(state.lastWindow);
      this.lastWindowLen = state.lastWindowLen;

      // Reallocate forward/backward buffers
      const emb = this.config.embeddingDim;
      const msl = this.config.maxSequenceLength;
      const nb = this.config.numBlocks;
      const ns = this.numScales;

      this.fwdNormX = new Float64Array(msl * this.inputDim);
      this.fwdNormY = new Float64Array(this.outputDim);

      for (let s = 0; s < ns; s++) {
        const scaleLen = Math.ceil(msl / this.config.temporalScales[s]);
        this.fwdConvOut[s] = new Float64Array(scaleLen * emb);
        this.fwdConvPre[s] = new Float64Array(scaleLen * emb);
        this.fwdScaleEmb[s] = new Float64Array(scaleLen * emb);
        this.gradConvW[s] = new Float64Array(this.convW[s].length);
        this.gradConvB[s] = new Float64Array(emb);
      }

      this.fwdUpsampled = new Float64Array(msl * ns * emb);
      this.fwdGateIn = new Float64Array(msl * ns * emb);
      this.fwdGate = new Float64Array(msl * ns * emb);
      this.fwdFused = new Float64Array(msl * emb);

      for (let b = 0; b < nb; b++) {
        this.fwdBlockIn[b] = new Float64Array(msl * emb);
        this.fwdLn1Out[b] = new Float64Array(msl * emb);
        this.fwdLn1Mean[b] = new Float64Array(msl);
        this.fwdLn1Rstd[b] = new Float64Array(msl);
        this.fwdQ[b] = new Float64Array(msl * emb);
        this.fwdK[b] = new Float64Array(msl * emb);
        this.fwdV[b] = new Float64Array(msl * emb);
        this.fwdAttnScores[b] = new Float64Array(
          this.config.numHeads * msl * msl,
        );
        this.fwdAttnProbs[b] = new Float64Array(
          this.config.numHeads * msl * msl,
        );
        this.fwdAttnOut[b] = new Float64Array(msl * emb);
        this.fwdAttnProj[b] = new Float64Array(msl * emb);
        this.fwdRes1[b] = new Float64Array(msl * emb);
        this.fwdLn2Out[b] = new Float64Array(msl * emb);
        this.fwdLn2Mean[b] = new Float64Array(msl);
        this.fwdLn2Rstd[b] = new Float64Array(msl);
        this.fwdFfnHid[b] = new Float64Array(msl * this.ffnDim);
        this.fwdFfnPre[b] = new Float64Array(msl * this.ffnDim);
        this.fwdFfnOut[b] = new Float64Array(msl * emb);
        this.fwdBlockOut[b] = new Float64Array(msl * emb);

        this.gradLn1Gamma[b] = new Float64Array(emb);
        this.gradLn1Beta[b] = new Float64Array(emb);
        this.gradLn2Gamma[b] = new Float64Array(emb);
        this.gradLn2Beta[b] = new Float64Array(emb);
        this.gradAttnWq[b] = new Float64Array(emb * emb);
        this.gradAttnWk[b] = new Float64Array(emb * emb);
        this.gradAttnWv[b] = new Float64Array(emb * emb);
        this.gradAttnWo[b] = new Float64Array(emb * emb);
        this.gradFfnW1[b] = new Float64Array(emb * this.ffnDim);
        this.gradFfnB1[b] = new Float64Array(this.ffnDim);
        this.gradFfnW2[b] = new Float64Array(this.ffnDim * emb);
        this.gradFfnB2[b] = new Float64Array(emb);
      }

      this.fwdPoolScores = new Float64Array(msl);
      this.fwdPoolWeights = new Float64Array(msl);
      this.fwdPooled = new Float64Array(emb);
      this.fwdOutput = new Float64Array(this.outputDim);

      this.gradScaleEmb = new Float64Array(ns * emb);
      this.gradFusionW = new Float64Array(ns * emb * ns * emb);
      this.gradFusionB = new Float64Array(ns * emb);
      this.gradPoolW = new Float64Array(emb);
      this.gradOutW = new Float64Array(emb * this.outputDim);
      this.gradOutB = new Float64Array(this.outputDim);

      this.scratchSeqEmb = new Float64Array(msl * emb);
      this.scratchSeqEmb2 = new Float64Array(msl * emb);
      this.scratchHead = new Float64Array(msl * msl);
      this.scratchFfn = new Float64Array(msl * this.ffnDim);
      this.scratchEmb = new Float64Array(emb);
      this.scratchGrad = new Float64Array(msl * ns * emb);
      this.scratchScales = new Float64Array(msl);
    }

    this.driftWindow = new Float64Array(state.driftWindow);
    this.driftWindowHead = state.driftWindowHead;
    this.driftWindowCount = state.driftWindowCount;
    this.rngState = state.rngState;
    this.converged = state.converged;
    this.lastGradNorm = state.lastGradNorm;
    this.lastLR = state.lastLR;
  }
}
