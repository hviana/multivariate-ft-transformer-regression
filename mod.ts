/**
 * FusionTemporalTransformerRegression
 *
 * A Fusion Temporal Transformer neural network for multivariate regression
 * with incremental online learning, Adam optimizer, and z-score normalization.
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({ numBlocks: 2 });
 * const result = model.fitOnline({
 *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
 *   yCoordinates: [[7, 8]]
 * });
 * const predictions = model.predict(3);
 * ```
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Result of a single online training step
 */
export interface FitResult {
  /** Current loss value (weighted MSE + L2 regularization) */
  loss: number;
  /** L2 norm of all gradients */
  gradientNorm: number;
  /** Current learning rate after warmup/decay schedule */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether training has converged based on loss change threshold */
  converged: boolean;
  /** Current sample index (total samples seen) */
  sampleIndex: number;
  /** Whether concept drift was detected by ADWIN */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds
 */
export interface SinglePrediction {
  /** Predicted output values */
  predicted: number[];
  /** Lower bound of 95% confidence interval */
  lowerBound: number[];
  /** Upper bound of 95% confidence interval */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result of prediction call
 */
export interface PredictionResult {
  /** Array of predictions for each future step */
  predictions: SinglePrediction[];
  /** Model accuracy metric: 1 / (1 + avgLoss) */
  accuracy: number;
  /** Total number of samples trained on */
  sampleCount: number;
  /** Whether model has been trained and is ready for prediction */
  isModelReady: boolean;
}

/**
 * Weight information for model introspection
 */
export interface WeightInfo {
  /** Temporal convolution weights per scale */
  temporalConvWeights: number[][][];
  /** Scale embedding vectors */
  scaleEmbeddings: number[][];
  /** Positional encoding matrices per scale */
  positionalEncoding: number[][][];
  /** Fusion gate weights per scale */
  fusionWeights: number[][];
  /** Attention weights per block */
  attentionWeights: number[][][];
  /** Feed-forward network weights per block */
  ffnWeights: number[][][];
  /** Layer normalization parameters per block */
  layerNormParams: number[][][];
  /** Output layer weights */
  outputWeights: number[][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of Adam updates performed */
  updateCount: number;
}

/**
 * Z-score normalization statistics
 */
export interface NormalizationStats {
  /** Per-feature input means */
  inputMean: number[];
  /** Per-feature input standard deviations */
  inputStd: number[];
  /** Per-feature output means */
  outputMean: number[];
  /** Per-feature output standard deviations */
  outputStd: number[];
  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Model summary information
 */
export interface ModelSummary {
  /** Whether the model has been initialized */
  isInitialized: boolean;
  /** Input feature dimension */
  inputDimension: number;
  /** Output feature dimension */
  outputDimension: number;
  /** Number of transformer blocks */
  numBlocks: number;
  /** Embedding dimension */
  embeddingDim: number;
  /** Number of attention heads */
  numHeads: number;
  /** Temporal convolution scales */
  temporalScales: number[];
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Number of samples trained on */
  sampleCount: number;
  /** Current accuracy metric */
  accuracy: number;
  /** Whether training has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/**
 * Configuration options for the model
 */
export interface FusionTemporalTransformerConfig {
  /** Number of transformer blocks (default: 3) */
  numBlocks?: number;
  /** Embedding dimension (default: 64) */
  embeddingDim?: number;
  /** Number of attention heads (default: 8) */
  numHeads?: number;
  /** FFN hidden dimension multiplier (default: 4) */
  ffnMultiplier?: number;
  /** Attention dropout rate (default: 0.0) */
  attentionDropout?: number;
  /** Base learning rate (default: 0.001) */
  learningRate?: number;
  /** Number of warmup steps (default: 100) */
  warmupSteps?: number;
  /** Total steps for cosine decay (default: 10000) */
  totalSteps?: number;
  /** Adam beta1 (default: 0.9) */
  beta1?: number;
  /** Adam beta2 (default: 0.999) */
  beta2?: number;
  /** Adam epsilon (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength (default: 1e-4) */
  regularizationStrength?: number;
  /** Convergence threshold for loss change (default: 1e-6) */
  convergenceThreshold?: number;
  /** Outlier detection threshold in std devs (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN delta parameter (default: 0.002) */
  adwinDelta?: number;
  /** Temporal convolution scales (default: [1, 2, 4]) */
  temporalScales?: number[];
  /** Temporal convolution kernel size (default: 3) */
  temporalKernelSize?: number;
  /** Maximum sequence length (default: 512) */
  maxSequenceLength?: number;
  /** Fusion layer dropout (default: 0.0) */
  fusionDropout?: number;
}

/**
 * Input data for fitOnline
 */
export interface FitOnlineInput {
  /** Input sequence: shape (seqLen, inputDim) */
  xCoordinates: number[][];
  /** Output sequence: shape (ySeqLen, outputDim) */
  yCoordinates: number[][];
}

// ============================================================================
// INTERNAL TYPES
// ============================================================================

interface InternalConfig {
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
  headDim: number;
  hiddenDim: number;
}

interface TransformerBlockWeights {
  ln1Gamma: Float64Array;
  ln1Beta: Float64Array;
  wq: Float64Array;
  wk: Float64Array;
  wv: Float64Array;
  wo: Float64Array;
  bo: Float64Array;
  relPosBias: Float64Array;
  ln2Gamma: Float64Array;
  ln2Beta: Float64Array;
  ffnW1: Float64Array;
  ffnB1: Float64Array;
  ffnW2: Float64Array;
  ffnB2: Float64Array;
}

interface ModelWeights {
  inputWeight: Float64Array;
  inputBias: Float64Array;
  convWeights: Float64Array[];
  convBiases: Float64Array[];
  scaleEmb: Float64Array[];
  gateWeights: Float64Array[];
  gateBiases: Float64Array;
  blocks: TransformerBlockWeights[];
  poolWeight: Float64Array;
  poolBias: Float64Array;
  outputWeight: Float64Array;
  outputBias: Float64Array;
}

interface ForwardCache {
  xNorm: Float64Array[];
  e0: Float64Array[];
  convOutputs: Float64Array[][];
  convActivations: Float64Array[][];
  scaleEmbeddings: Float64Array[][];
  alignedEmbeddings: Float64Array[][];
  gateLogits: Float64Array[];
  gates: Float64Array[];
  fused: Float64Array[];
  blockCaches: BlockForwardCache[];
  poolLogits: Float64Array;
  poolWeights: Float64Array;
  aggregated: Float64Array;
  yPredNorm: Float64Array;
  yTarget: Float64Array;
  yTargetNorm: Float64Array;
  seqLen: number;
}

interface BlockForwardCache {
  input: Float64Array[];
  ln1Out: Float64Array[];
  ln1Mean: Float64Array;
  ln1Var: Float64Array;
  q: Float64Array[][];
  k: Float64Array[][];
  v: Float64Array[][];
  attnScores: Float64Array[][];
  attnWeights: Float64Array[][];
  headOuts: Float64Array[][];
  attnOut: Float64Array[];
  postAttn: Float64Array[];
  ln2Out: Float64Array[];
  ln2Mean: Float64Array;
  ln2Var: Float64Array;
  ffnHidden: Float64Array[];
  ffnOut: Float64Array[];
}

interface ComputeBuffers {
  tempEmb1: Float64Array;
  tempEmb2: Float64Array;
  tempEmb3: Float64Array;
  tempHead: Float64Array;
  tempHidden: Float64Array;
  tempSeq: Float64Array[];
  tempGrad: Float64Array;
  attnBuffer: Float64Array;
}

interface AdwinBucket {
  total: number;
  variance: number;
  count: number;
}

// ============================================================================
// MATH UTILITIES
// ============================================================================

/**
 * GELU activation function
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 * @param x - Input value
 * @returns GELU activation
 */
function gelu(x: number): number {
  const c = 0.7978845608028654; // sqrt(2/π)
  const inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

/**
 * Derivative of GELU activation
 * @param x - Input value
 * @param y - GELU(x) output (for efficiency)
 * @returns Derivative
 */
function geluDerivative(x: number): number {
  const c = 0.7978845608028654;
  const x3 = x * x * x;
  const inner = c * (x + 0.044715 * x3);
  const tanhVal = Math.tanh(inner);
  const sech2 = 1 - tanhVal * tanhVal;
  const innerDeriv = c * (1 + 0.134145 * x * x);
  return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * innerDeriv;
}

/**
 * Stable softmax over array (in-place output)
 * @param input - Input array
 * @param output - Output array (can be same as input)
 * @param len - Length to process
 */
function softmax(input: Float64Array, output: Float64Array, len: number): void {
  let maxVal = -Infinity;
  for (let i = 0; i < len; i++) {
    if (input[i] > maxVal) maxVal = input[i];
  }
  let sum = 0;
  for (let i = 0; i < len; i++) {
    output[i] = Math.exp(input[i] - maxVal);
    sum += output[i];
  }
  const invSum = 1 / (sum + 1e-12);
  for (let i = 0; i < len; i++) {
    output[i] *= invSum;
  }
}

/**
 * Sigmoid function
 * @param x - Input value
 * @returns Sigmoid output
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const expX = Math.exp(x);
  return expX / (1 + expX);
}

/**
 * Xavier/Glorot initialization bound
 * @param fanIn - Number of input units
 * @param fanOut - Number of output units
 * @returns Initialization bound
 */
function xavierBound(fanIn: number, fanOut: number): number {
  return Math.sqrt(6 / (fanIn + fanOut));
}

/**
 * Simple seeded RNG for reproducibility
 */
class SeededRNG {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed;
  }

  next(): number {
    this.state = (this.state * 1103515245 + 12345) & 0x7fffffff;
    return this.state / 0x7fffffff;
  }

  uniform(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.next();
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
}

// ============================================================================
// ADWIN DRIFT DETECTOR
// ============================================================================

class AdwinDetector {
  private buckets: AdwinBucket[] = [];
  private total: number = 0;
  private count: number = 0;
  private variance: number = 0;
  private delta: number;
  private maxBuckets: number = 32;

  constructor(delta: number = 0.002) {
    this.delta = delta;
  }

  /**
   * Add a value and check for drift
   * @param value - Error value to add
   * @returns Whether drift was detected
   */
  add(value: number): boolean {
    // Update overall stats
    this.count++;
    const oldMean = this.count > 1 ? this.total / (this.count - 1) : 0;
    this.total += value;
    const newMean = this.total / this.count;
    this.variance += (value - oldMean) * (value - newMean);

    // Add to buckets
    this.buckets.push({ total: value, variance: 0, count: 1 });

    // Compress buckets if needed
    this.compress();

    // Check for drift
    return this.checkDrift();
  }

  private compress(): void {
    while (this.buckets.length > this.maxBuckets) {
      // Merge oldest two buckets
      if (this.buckets.length >= 2) {
        const b1 = this.buckets[0];
        const b2 = this.buckets[1];
        const newCount = b1.count + b2.count;
        const mean1 = b1.total / b1.count;
        const mean2 = b2.total / b2.count;
        const newTotal = b1.total + b2.total;
        const newMean = newTotal / newCount;
        const newVariance = b1.variance + b2.variance +
          b1.count * (mean1 - newMean) * (mean1 - newMean) +
          b2.count * (mean2 - newMean) * (mean2 - newMean);
        this.buckets.splice(0, 2, {
          total: newTotal,
          variance: newVariance,
          count: newCount,
        });
      }
    }
  }

  private checkDrift(): boolean {
    if (this.buckets.length < 2) return false;

    let n0 = 0, sum0 = 0;
    let n1 = this.count, sum1 = this.total;

    for (let i = 0; i < this.buckets.length - 1; i++) {
      n0 += this.buckets[i].count;
      sum0 += this.buckets[i].total;
      n1 -= this.buckets[i].count;
      sum1 -= this.buckets[i].total;

      if (n0 < 1 || n1 < 1) continue;

      const mean0 = sum0 / n0;
      const mean1 = sum1 / n1;
      const epsCut = Math.sqrt(
        (1 / n0 + 1 / n1) * Math.log(4 / this.delta) / 2,
      );

      if (Math.abs(mean0 - mean1) >= epsCut) {
        // Drift detected - shrink window
        this.shrink(i + 1);
        return true;
      }
    }
    return false;
  }

  private shrink(splitIdx: number): void {
    let removedCount = 0;
    let removedTotal = 0;
    let removedVariance = 0;

    for (let i = 0; i < splitIdx; i++) {
      removedCount += this.buckets[i].count;
      removedTotal += this.buckets[i].total;
    }

    this.buckets = this.buckets.slice(splitIdx);
    this.count -= removedCount;
    this.total -= removedTotal;

    // Recalculate variance
    if (this.count > 0) {
      const mean = this.total / this.count;
      this.variance = 0;
      for (const b of this.buckets) {
        const bMean = b.total / b.count;
        this.variance += b.variance + b.count * (bMean - mean) * (bMean - mean);
      }
    } else {
      this.variance = 0;
    }
  }

  reset(): void {
    this.buckets = [];
    this.total = 0;
    this.count = 0;
    this.variance = 0;
  }

  getState(): {
    buckets: AdwinBucket[];
    total: number;
    count: number;
    variance: number;
  } {
    return {
      buckets: this.buckets.map((b) => ({ ...b })),
      total: this.total,
      count: this.count,
      variance: this.variance,
    };
  }

  setState(
    state: {
      buckets: AdwinBucket[];
      total: number;
      count: number;
      variance: number;
    },
  ): void {
    this.buckets = state.buckets.map((b) => ({ ...b }));
    this.total = state.total;
    this.count = state.count;
    this.variance = state.variance;
  }
}

// ============================================================================
// MAIN CLASS
// ============================================================================

export class FusionTemporalTransformerRegression {
  private config: InternalConfig;
  private rng: SeededRNG;

  // Model state
  private isInitialized: boolean = false;
  private inputDim: number = 0;
  private outputDim: number = 0;

  // Weights and gradients
  private weights: ModelWeights | null = null;
  private gradients: ModelWeights | null = null;
  private firstMoment: ModelWeights | null = null;
  private secondMoment: ModelWeights | null = null;
  private step: number = 0;

  // Normalization stats (Welford's algorithm)
  private inputMean: Float64Array | null = null;
  private inputM2: Float64Array | null = null;
  private inputStd: Float64Array | null = null;
  private outputMean: Float64Array | null = null;
  private outputM2: Float64Array | null = null;
  private outputStd: Float64Array | null = null;
  private normCount: number = 0;

  // Running metrics
  private avgLoss: number = 0;
  private prevAvgLoss: number = 0;
  private sampleCount: number = 0;
  private converged: boolean = false;

  // Residual variance for uncertainty (Welford's)
  private residualMean: Float64Array | null = null;
  private residualM2: Float64Array | null = null;
  private residualCount: number = 0;

  // ADWIN drift detector
  private adwin: AdwinDetector;
  private driftCount: number = 0;

  // Preallocated buffers
  private buffers: ComputeBuffers | null = null;
  private cache: ForwardCache | null = null;

  // Last input for prediction
  private lastInput: Float64Array[] | null = null;
  private lastSeqLen: number = 0;

  /**
   * Creates a new FusionTemporalTransformerRegression model
   * @param config - Model configuration options
   * @example
   * ```typescript
   * const model = new FusionTemporalTransformerRegression({
   *   numBlocks: 3,
   *   embeddingDim: 64,
   *   learningRate: 0.001
   * });
   * ```
   */
  constructor(config: FusionTemporalTransformerConfig = {}) {
    const embeddingDim = config.embeddingDim ?? 64;
    const numHeads = config.numHeads ?? 8;
    const ffnMultiplier = config.ffnMultiplier ?? 4;

    this.config = {
      numBlocks: config.numBlocks ?? 3,
      embeddingDim,
      numHeads,
      ffnMultiplier,
      attentionDropout: config.attentionDropout ?? 0.0,
      learningRate: config.learningRate ?? 0.001,
      warmupSteps: config.warmupSteps ?? 100,
      totalSteps: config.totalSteps ?? 10000,
      beta1: config.beta1 ?? 0.9,
      beta2: config.beta2 ?? 0.999,
      epsilon: config.epsilon ?? 1e-8,
      regularizationStrength: config.regularizationStrength ?? 1e-4,
      convergenceThreshold: config.convergenceThreshold ?? 1e-6,
      outlierThreshold: config.outlierThreshold ?? 3.0,
      adwinDelta: config.adwinDelta ?? 0.002,
      temporalScales: config.temporalScales ?? [1, 2, 4],
      temporalKernelSize: config.temporalKernelSize ?? 3,
      maxSequenceLength: config.maxSequenceLength ?? 512,
      fusionDropout: config.fusionDropout ?? 0.0,
      headDim: Math.floor(embeddingDim / numHeads),
      hiddenDim: embeddingDim * ffnMultiplier,
    };

    this.rng = new SeededRNG(42);
    this.adwin = new AdwinDetector(this.config.adwinDelta);
  }

  /**
   * Initialize model weights with Xavier/Glorot initialization
   * @param inputDim - Input feature dimension
   * @param outputDim - Output feature dimension
   */
  private initializeWeights(inputDim: number, outputDim: number): void {
    const {
      embeddingDim,
      numBlocks,
      temporalScales,
      temporalKernelSize,
      hiddenDim,
      maxSequenceLength,
    } = this.config;
    const numScales = temporalScales.length;

    // Input projection: inputDim -> embeddingDim
    const inputWeight = new Float64Array(embeddingDim * inputDim);
    const inputBias = new Float64Array(embeddingDim);
    const inputBound = xavierBound(inputDim, embeddingDim);
    for (let i = 0; i < inputWeight.length; i++) {
      inputWeight[i] = this.rng.uniform(-inputBound, inputBound);
    }

    // Temporal conv weights per scale
    const convWeights: Float64Array[] = [];
    const convBiases: Float64Array[] = [];
    for (let s = 0; s < numScales; s++) {
      const w = new Float64Array(
        embeddingDim * embeddingDim * temporalKernelSize,
      );
      const b = new Float64Array(embeddingDim);
      const convBound = xavierBound(
        embeddingDim * temporalKernelSize,
        embeddingDim,
      );
      for (let i = 0; i < w.length; i++) {
        w[i] = this.rng.uniform(-convBound, convBound);
      }
      convWeights.push(w);
      convBiases.push(b);
    }

    // Scale embeddings
    const scaleEmb: Float64Array[] = [];
    for (let s = 0; s < numScales; s++) {
      const emb = new Float64Array(embeddingDim);
      for (let i = 0; i < embeddingDim; i++) {
        emb[i] = this.rng.normal(0, 0.001);
      }
      scaleEmb.push(emb);
    }

    // Fusion gate weights
    const gateWeights: Float64Array[] = [];
    const gateBiases = new Float64Array(numScales);
    for (let s = 0; s < numScales; s++) {
      const w = new Float64Array(embeddingDim);
      const gateBound = xavierBound(embeddingDim, 1);
      for (let i = 0; i < embeddingDim; i++) {
        w[i] = this.rng.uniform(-gateBound, gateBound);
      }
      gateWeights.push(w);
    }

    // Transformer blocks
    const blocks: TransformerBlockWeights[] = [];
    for (let b = 0; b < numBlocks; b++) {
      const attnBound = xavierBound(embeddingDim, embeddingDim);
      const ffnBound1 = xavierBound(embeddingDim, hiddenDim);
      const ffnBound2 = xavierBound(hiddenDim, embeddingDim);

      const wq = new Float64Array(embeddingDim * embeddingDim);
      const wk = new Float64Array(embeddingDim * embeddingDim);
      const wv = new Float64Array(embeddingDim * embeddingDim);
      const wo = new Float64Array(embeddingDim * embeddingDim);
      const bo = new Float64Array(embeddingDim);
      const relPosBias = new Float64Array(2 * maxSequenceLength - 1);

      for (let i = 0; i < wq.length; i++) {
        wq[i] = this.rng.uniform(-attnBound, attnBound);
        wk[i] = this.rng.uniform(-attnBound, attnBound);
        wv[i] = this.rng.uniform(-attnBound, attnBound);
        wo[i] = this.rng.uniform(-attnBound, attnBound);
      }

      for (let i = 0; i < relPosBias.length; i++) {
        relPosBias[i] = this.rng.normal(0, 0.02);
      }

      const ln1Gamma = new Float64Array(embeddingDim).fill(1);
      const ln1Beta = new Float64Array(embeddingDim);
      const ln2Gamma = new Float64Array(embeddingDim).fill(1);
      const ln2Beta = new Float64Array(embeddingDim);

      const ffnW1 = new Float64Array(hiddenDim * embeddingDim);
      const ffnB1 = new Float64Array(hiddenDim);
      const ffnW2 = new Float64Array(embeddingDim * hiddenDim);
      const ffnB2 = new Float64Array(embeddingDim);

      for (let i = 0; i < ffnW1.length; i++) {
        ffnW1[i] = this.rng.uniform(-ffnBound1, ffnBound1);
      }
      for (let i = 0; i < ffnW2.length; i++) {
        ffnW2[i] = this.rng.uniform(-ffnBound2, ffnBound2);
      }

      blocks.push({
        ln1Gamma,
        ln1Beta,
        wq,
        wk,
        wv,
        wo,
        bo,
        relPosBias,
        ln2Gamma,
        ln2Beta,
        ffnW1,
        ffnB1,
        ffnW2,
        ffnB2,
      });
    }

    // Pooling weights
    const poolWeight = new Float64Array(embeddingDim);
    const poolBias = new Float64Array(1);
    const poolBound = xavierBound(embeddingDim, 1);
    for (let i = 0; i < embeddingDim; i++) {
      poolWeight[i] = this.rng.uniform(-poolBound, poolBound);
    }

    // Output projection
    const outputWeight = new Float64Array(outputDim * embeddingDim);
    const outputBias = new Float64Array(outputDim);
    const outBound = xavierBound(embeddingDim, outputDim);
    for (let i = 0; i < outputWeight.length; i++) {
      outputWeight[i] = this.rng.uniform(-outBound, outBound);
    }

    this.weights = {
      inputWeight,
      inputBias,
      convWeights,
      convBiases,
      scaleEmb,
      gateWeights,
      gateBiases,
      blocks,
      poolWeight,
      poolBias,
      outputWeight,
      outputBias,
    };

    // Initialize gradients with same structure
    this.gradients = this.createZeroWeights(inputDim, outputDim);
    this.firstMoment = this.createZeroWeights(inputDim, outputDim);
    this.secondMoment = this.createZeroWeights(inputDim, outputDim);
  }

  /**
   * Create zero-initialized weight structure
   */
  private createZeroWeights(inputDim: number, outputDim: number): ModelWeights {
    const {
      embeddingDim,
      numBlocks,
      temporalScales,
      temporalKernelSize,
      hiddenDim,
      maxSequenceLength,
    } = this.config;
    const numScales = temporalScales.length;

    const blocks: TransformerBlockWeights[] = [];
    for (let b = 0; b < numBlocks; b++) {
      blocks.push({
        ln1Gamma: new Float64Array(embeddingDim),
        ln1Beta: new Float64Array(embeddingDim),
        wq: new Float64Array(embeddingDim * embeddingDim),
        wk: new Float64Array(embeddingDim * embeddingDim),
        wv: new Float64Array(embeddingDim * embeddingDim),
        wo: new Float64Array(embeddingDim * embeddingDim),
        bo: new Float64Array(embeddingDim),
        relPosBias: new Float64Array(2 * maxSequenceLength - 1),
        ln2Gamma: new Float64Array(embeddingDim),
        ln2Beta: new Float64Array(embeddingDim),
        ffnW1: new Float64Array(hiddenDim * embeddingDim),
        ffnB1: new Float64Array(hiddenDim),
        ffnW2: new Float64Array(embeddingDim * hiddenDim),
        ffnB2: new Float64Array(embeddingDim),
      });
    }

    return {
      inputWeight: new Float64Array(embeddingDim * inputDim),
      inputBias: new Float64Array(embeddingDim),
      convWeights: temporalScales.map(() =>
        new Float64Array(embeddingDim * embeddingDim * temporalKernelSize)
      ),
      convBiases: temporalScales.map(() => new Float64Array(embeddingDim)),
      scaleEmb: temporalScales.map(() => new Float64Array(embeddingDim)),
      gateWeights: temporalScales.map(() => new Float64Array(embeddingDim)),
      gateBiases: new Float64Array(numScales),
      blocks,
      poolWeight: new Float64Array(embeddingDim),
      poolBias: new Float64Array(1),
      outputWeight: new Float64Array(outputDim * embeddingDim),
      outputBias: new Float64Array(outputDim),
    };
  }

  /**
   * Initialize compute buffers for memory reuse
   */
  private initializeBuffers(seqLen: number): void {
    const { embeddingDim, hiddenDim, maxSequenceLength } = this.config;

    const tempSeq: Float64Array[] = [];
    for (let t = 0; t < maxSequenceLength; t++) {
      tempSeq.push(new Float64Array(embeddingDim));
    }

    this.buffers = {
      tempEmb1: new Float64Array(embeddingDim),
      tempEmb2: new Float64Array(embeddingDim),
      tempEmb3: new Float64Array(embeddingDim),
      tempHead: new Float64Array(this.config.headDim),
      tempHidden: new Float64Array(hiddenDim),
      tempSeq,
      tempGrad: new Float64Array(embeddingDim),
      attnBuffer: new Float64Array(maxSequenceLength),
    };
  }

  /**
   * Initialize forward cache for backpropagation
   */
  private initializeCache(seqLen: number): ForwardCache {
    const {
      embeddingDim,
      hiddenDim,
      numBlocks,
      temporalScales,
      numHeads,
      headDim,
    } = this.config;
    const numScales = temporalScales.length;

    const xNorm: Float64Array[] = [];
    const e0: Float64Array[] = [];
    const fused: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      xNorm.push(new Float64Array(this.inputDim));
      e0.push(new Float64Array(embeddingDim));
      fused.push(new Float64Array(embeddingDim));
    }

    const convOutputs: Float64Array[][] = [];
    const convActivations: Float64Array[][] = [];
    const scaleEmbeddings: Float64Array[][] = [];
    const alignedEmbeddings: Float64Array[][] = [];

    for (let s = 0; s < numScales; s++) {
      const scaleLen = Math.ceil(seqLen / temporalScales[s]);
      const scaleConvOut: Float64Array[] = [];
      const scaleConvAct: Float64Array[] = [];
      const scaleEmb: Float64Array[] = [];
      for (let t = 0; t < scaleLen; t++) {
        scaleConvOut.push(new Float64Array(embeddingDim));
        scaleConvAct.push(new Float64Array(embeddingDim));
        scaleEmb.push(new Float64Array(embeddingDim));
      }
      convOutputs.push(scaleConvOut);
      convActivations.push(scaleConvAct);
      scaleEmbeddings.push(scaleEmb);

      const aligned: Float64Array[] = [];
      for (let t = 0; t < seqLen; t++) {
        aligned.push(new Float64Array(embeddingDim));
      }
      alignedEmbeddings.push(aligned);
    }

    const gateLogits: Float64Array[] = [];
    const gates: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      gateLogits.push(new Float64Array(numScales));
      gates.push(new Float64Array(numScales));
    }

    const blockCaches: BlockForwardCache[] = [];
    for (let b = 0; b < numBlocks; b++) {
      const input: Float64Array[] = [];
      const ln1Out: Float64Array[] = [];
      const attnOut: Float64Array[] = [];
      const postAttn: Float64Array[] = [];
      const ln2Out: Float64Array[] = [];
      const ffnHidden: Float64Array[] = [];
      const ffnOut: Float64Array[] = [];

      for (let t = 0; t < seqLen; t++) {
        input.push(new Float64Array(embeddingDim));
        ln1Out.push(new Float64Array(embeddingDim));
        attnOut.push(new Float64Array(embeddingDim));
        postAttn.push(new Float64Array(embeddingDim));
        ln2Out.push(new Float64Array(embeddingDim));
        ffnHidden.push(new Float64Array(hiddenDim));
        ffnOut.push(new Float64Array(embeddingDim));
      }

      const q: Float64Array[][] = [];
      const k: Float64Array[][] = [];
      const v: Float64Array[][] = [];
      const attnScores: Float64Array[][] = [];
      const attnWeights: Float64Array[][] = [];
      const headOuts: Float64Array[][] = [];

      for (let h = 0; h < numHeads; h++) {
        const qHead: Float64Array[] = [];
        const kHead: Float64Array[] = [];
        const vHead: Float64Array[] = [];
        const scoresHead: Float64Array[] = [];
        const weightsHead: Float64Array[] = [];
        const outsHead: Float64Array[] = [];

        for (let t = 0; t < seqLen; t++) {
          qHead.push(new Float64Array(headDim));
          kHead.push(new Float64Array(headDim));
          vHead.push(new Float64Array(headDim));
          scoresHead.push(new Float64Array(seqLen));
          weightsHead.push(new Float64Array(seqLen));
          outsHead.push(new Float64Array(headDim));
        }

        q.push(qHead);
        k.push(kHead);
        v.push(vHead);
        attnScores.push(scoresHead);
        attnWeights.push(weightsHead);
        headOuts.push(outsHead);
      }

      blockCaches.push({
        input,
        ln1Out,
        ln1Mean: new Float64Array(seqLen),
        ln1Var: new Float64Array(seqLen),
        q,
        k,
        v,
        attnScores,
        attnWeights,
        headOuts,
        attnOut,
        postAttn,
        ln2Out,
        ln2Mean: new Float64Array(seqLen),
        ln2Var: new Float64Array(seqLen),
        ffnHidden,
        ffnOut,
      });
    }

    return {
      xNorm,
      e0,
      convOutputs,
      convActivations,
      scaleEmbeddings,
      alignedEmbeddings,
      gateLogits,
      gates,
      fused,
      blockCaches,
      poolLogits: new Float64Array(seqLen),
      poolWeights: new Float64Array(seqLen),
      aggregated: new Float64Array(embeddingDim),
      yPredNorm: new Float64Array(this.outputDim),
      yTarget: new Float64Array(this.outputDim),
      yTargetNorm: new Float64Array(this.outputDim),
      seqLen,
    };
  }

  /**
   * Update normalization statistics using Welford's algorithm
   * @param x - Input sequence (seqLen x inputDim)
   * @param y - Target values (outputDim)
   */
  private updateNormStats(x: number[][], y: number[]): void {
    const eps = this.config.epsilon;

    // Update input stats for each timestep
    for (let t = 0; t < x.length; t++) {
      for (let i = 0; i < this.inputDim; i++) {
        const v = x[t][i];
        this.normCount++;
        const delta = v - this.inputMean![i];
        this.inputMean![i] += delta / this.normCount;
        const delta2 = v - this.inputMean![i];
        this.inputM2![i] += delta * delta2;
      }
    }

    // Update output stats
    for (let k = 0; k < this.outputDim; k++) {
      const v = y[k];
      const count = Math.floor(this.normCount / x.length);
      const delta = v - this.outputMean![k];
      this.outputMean![k] += delta / Math.max(1, count);
      const delta2 = v - this.outputMean![k];
      this.outputM2![k] += delta * delta2;
    }

    // Compute standard deviations
    const inputCount = this.normCount;
    const outputCount = Math.max(1, Math.floor(this.normCount / x.length));

    for (let i = 0; i < this.inputDim; i++) {
      const variance = inputCount > 1 ? this.inputM2![i] / (inputCount - 1) : 0;
      this.inputStd![i] = Math.sqrt(Math.max(variance, 0)) + eps;
    }

    for (let k = 0; k < this.outputDim; k++) {
      const variance = outputCount > 1
        ? this.outputM2![k] / (outputCount - 1)
        : 0;
      this.outputStd![k] = Math.sqrt(Math.max(variance, 0)) + eps;
    }
  }

  /**
   * Normalize input sequence
   */
  private normalizeInput(x: number[][], cache: ForwardCache): void {
    const eps = this.config.epsilon;
    for (let t = 0; t < x.length; t++) {
      for (let i = 0; i < this.inputDim; i++) {
        cache.xNorm[t][i] = (x[t][i] - this.inputMean![i]) /
          (this.inputStd![i] + eps);
      }
    }
  }

  /**
   * Forward pass: input projection
   * E0[t, d] = sum_{c} Xnorm[t, c] * W_in[d, c] + b_in[d]
   */
  private forwardInputProjection(cache: ForwardCache): void {
    const { embeddingDim } = this.config;
    const w = this.weights!.inputWeight;
    const b = this.weights!.inputBias;

    for (let t = 0; t < cache.seqLen; t++) {
      for (let d = 0; d < embeddingDim; d++) {
        let sum = b[d];
        for (let c = 0; c < this.inputDim; c++) {
          sum += cache.xNorm[t][c] * w[d * this.inputDim + c];
        }
        cache.e0[t][d] = sum;
      }
    }
  }

  /**
   * Forward pass: multi-scale temporal convolution
   * For each scale s: Conv -> GELU -> Add positional encoding + scale embedding
   */
  private forwardTemporalConv(cache: ForwardCache): void {
    const {
      embeddingDim,
      temporalScales,
      temporalKernelSize,
      maxSequenceLength,
    } = this.config;
    const pad = Math.floor(temporalKernelSize / 2);

    for (let s = 0; s < temporalScales.length; s++) {
      const scale = temporalScales[s];
      const scaleLen = Math.ceil(cache.seqLen / scale);
      const w = this.weights!.convWeights[s];
      const b = this.weights!.convBiases[s];
      const scaleEmb = this.weights!.scaleEmb[s];

      // Convolution with stride
      for (let u = 0; u < scaleLen; u++) {
        const center = u * scale;

        for (let d = 0; d < embeddingDim; d++) {
          let sum = b[d];

          for (let k = 0; k < temporalKernelSize; k++) {
            const srcT = center + k - pad;
            if (srcT >= 0 && srcT < cache.seqLen) {
              for (let d2 = 0; d2 < embeddingDim; d2++) {
                sum += cache.e0[srcT][d2] *
                  w[(d * embeddingDim + d2) * temporalKernelSize + k];
              }
            }
          }

          cache.convOutputs[s][u][d] = sum;
          // GELU activation
          cache.convActivations[s][u][d] = gelu(sum);
        }

        // Add positional encoding and scale embedding
        for (let d = 0; d < embeddingDim; d++) {
          // Sinusoidal positional encoding
          // PE[u, 2i] = sin(u / 10000^(2i/embeddingDim))
          // PE[u, 2i+1] = cos(u / 10000^(2i/embeddingDim))
          const i = Math.floor(d / 2);
          const freq = u / Math.pow(10000, (2 * i) / embeddingDim);
          const pe = d % 2 === 0 ? Math.sin(freq) : Math.cos(freq);

          cache.scaleEmbeddings[s][u][d] = cache.convActivations[s][u][d] + pe +
            scaleEmb[d];
        }
      }
    }
  }

  /**
   * Forward pass: cross-scale fusion with gating
   */
  private forwardFusion(cache: ForwardCache): void {
    const { embeddingDim, temporalScales, epsilon } = this.config;
    const numScales = temporalScales.length;

    // Align all scales to finest scale (seqLen)
    for (let s = 0; s < numScales; s++) {
      const scale = temporalScales[s];
      for (let t = 0; t < cache.seqLen; t++) {
        const u = Math.min(
          Math.floor(t / scale),
          Math.ceil(cache.seqLen / scale) - 1,
        );
        for (let d = 0; d < embeddingDim; d++) {
          cache.alignedEmbeddings[s][t][d] = cache.scaleEmbeddings[s][u][d];
        }
      }
    }

    // Compute gates and fuse
    for (let t = 0; t < cache.seqLen; t++) {
      // Compute gate logits
      let maxLogit = -Infinity;
      for (let s = 0; s < numScales; s++) {
        let dot = this.weights!.gateBiases[s];
        for (let d = 0; d < embeddingDim; d++) {
          dot += cache.alignedEmbeddings[s][t][d] *
            this.weights!.gateWeights[s][d];
        }
        cache.gateLogits[t][s] = dot;
        if (dot > maxLogit) maxLogit = dot;
      }

      // Softmax over scales
      let sumExp = 0;
      for (let s = 0; s < numScales; s++) {
        cache.gates[t][s] = Math.exp(cache.gateLogits[t][s] - maxLogit);
        sumExp += cache.gates[t][s];
      }
      for (let s = 0; s < numScales; s++) {
        cache.gates[t][s] /= sumExp + epsilon;
      }

      // Weighted sum
      for (let d = 0; d < embeddingDim; d++) {
        let sum = 0;
        for (let s = 0; s < numScales; s++) {
          sum += cache.gates[t][s] * cache.alignedEmbeddings[s][t][d];
        }
        cache.fused[t][d] = sum;
      }
    }
  }

  /**
   * Layer normalization forward pass
   * LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
   */
  private layerNorm(
    input: Float64Array[],
    output: Float64Array[],
    gamma: Float64Array,
    beta: Float64Array,
    means: Float64Array,
    vars: Float64Array,
    seqLen: number,
  ): void {
    const { embeddingDim, epsilon } = this.config;

    for (let t = 0; t < seqLen; t++) {
      // Compute mean
      let mean = 0;
      for (let d = 0; d < embeddingDim; d++) {
        mean += input[t][d];
      }
      mean /= embeddingDim;
      means[t] = mean;

      // Compute variance
      let variance = 0;
      for (let d = 0; d < embeddingDim; d++) {
        const diff = input[t][d] - mean;
        variance += diff * diff;
      }
      variance /= embeddingDim;
      vars[t] = variance;

      // Normalize and scale
      const invStd = 1 / Math.sqrt(variance + epsilon);
      for (let d = 0; d < embeddingDim; d++) {
        output[t][d] = gamma[d] * (input[t][d] - mean) * invStd + beta[d];
      }
    }
  }

  /**
   * Multi-head self-attention forward pass
   */
  private forwardAttention(
    input: Float64Array[],
    blockWeights: TransformerBlockWeights,
    blockCache: BlockForwardCache,
    seqLen: number,
  ): void {
    const { embeddingDim, numHeads, headDim, maxSequenceLength } = this.config;
    const scale = 1 / Math.sqrt(headDim);

    // Compute Q, K, V for all heads
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;

      for (let t = 0; t < seqLen; t++) {
        // Q = input * Wq
        for (let hd = 0; hd < headDim; hd++) {
          let sum = 0;
          for (let d = 0; d < embeddingDim; d++) {
            sum += input[t][d] *
              blockWeights.wq[(headOffset + hd) * embeddingDim + d];
          }
          blockCache.q[h][t][hd] = sum;
        }

        // K = input * Wk
        for (let hd = 0; hd < headDim; hd++) {
          let sum = 0;
          for (let d = 0; d < embeddingDim; d++) {
            sum += input[t][d] *
              blockWeights.wk[(headOffset + hd) * embeddingDim + d];
          }
          blockCache.k[h][t][hd] = sum;
        }

        // V = input * Wv
        for (let hd = 0; hd < headDim; hd++) {
          let sum = 0;
          for (let d = 0; d < embeddingDim; d++) {
            sum += input[t][d] *
              blockWeights.wv[(headOffset + hd) * embeddingDim + d];
          }
          blockCache.v[h][t][hd] = sum;
        }
      }

      // Compute attention scores and weights
      for (let i = 0; i < seqLen; i++) {
        let maxScore = -Infinity;

        for (let j = 0; j < seqLen; j++) {
          // Dot product Q[i] · K[j]
          let dot = 0;
          for (let hd = 0; hd < headDim; hd++) {
            dot += blockCache.q[h][i][hd] * blockCache.k[h][j][hd];
          }

          // Scale and add relative position bias
          const relPos = i - j + maxSequenceLength - 1;
          dot = dot * scale + blockWeights.relPosBias[relPos];

          // Causal mask: j > i means future position, mask it out
          if (j > i) {
            dot = -1e9;
          }

          blockCache.attnScores[h][i][j] = dot;
          if (dot > maxScore) maxScore = dot;
        }

        // Softmax
        let sumExp = 0;
        for (let j = 0; j < seqLen; j++) {
          blockCache.attnWeights[h][i][j] = Math.exp(
            blockCache.attnScores[h][i][j] - maxScore,
          );
          sumExp += blockCache.attnWeights[h][i][j];
        }
        for (let j = 0; j < seqLen; j++) {
          blockCache.attnWeights[h][i][j] /= sumExp + 1e-12;
        }

        // Compute head output
        for (let hd = 0; hd < headDim; hd++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += blockCache.attnWeights[h][i][j] * blockCache.v[h][j][hd];
          }
          blockCache.headOuts[h][i][hd] = sum;
        }
      }
    }

    // Concatenate heads and project
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < embeddingDim; d++) {
        let sum = blockWeights.bo[d];
        for (let h = 0; h < numHeads; h++) {
          const headOffset = h * headDim;
          for (let hd = 0; hd < headDim; hd++) {
            sum += blockCache.headOuts[h][t][hd] *
              blockWeights.wo[d * embeddingDim + headOffset + hd];
          }
        }
        blockCache.attnOut[t][d] = sum;
      }
    }
  }

  /**
   * Feed-forward network forward pass
   * h = GELU(x * W1 + b1)
   * out = h * W2 + b2
   */
  private forwardFFN(
    input: Float64Array[],
    blockWeights: TransformerBlockWeights,
    blockCache: BlockForwardCache,
    seqLen: number,
  ): void {
    const { embeddingDim, hiddenDim } = this.config;

    for (let t = 0; t < seqLen; t++) {
      // First linear + GELU
      for (let h = 0; h < hiddenDim; h++) {
        let sum = blockWeights.ffnB1[h];
        for (let d = 0; d < embeddingDim; d++) {
          sum += input[t][d] * blockWeights.ffnW1[h * embeddingDim + d];
        }
        blockCache.ffnHidden[t][h] = gelu(sum);
      }

      // Second linear
      for (let d = 0; d < embeddingDim; d++) {
        let sum = blockWeights.ffnB2[d];
        for (let h = 0; h < hiddenDim; h++) {
          sum += blockCache.ffnHidden[t][h] *
            blockWeights.ffnW2[d * hiddenDim + h];
        }
        blockCache.ffnOut[t][d] = sum;
      }
    }
  }

  /**
   * Forward pass through all transformer blocks
   */
  private forwardTransformerBlocks(cache: ForwardCache): void {
    const { embeddingDim, numBlocks } = this.config;

    for (let b = 0; b < numBlocks; b++) {
      const blockWeights = this.weights!.blocks[b];
      const blockCache = cache.blockCaches[b];

      // Copy input
      for (let t = 0; t < cache.seqLen; t++) {
        for (let d = 0; d < embeddingDim; d++) {
          blockCache.input[t][d] = cache.fused[t][d];
        }
      }

      // Pre-attention LayerNorm
      this.layerNorm(
        blockCache.input,
        blockCache.ln1Out,
        blockWeights.ln1Gamma,
        blockWeights.ln1Beta,
        blockCache.ln1Mean,
        blockCache.ln1Var,
        cache.seqLen,
      );

      // Multi-head attention
      this.forwardAttention(
        blockCache.ln1Out,
        blockWeights,
        blockCache,
        cache.seqLen,
      );

      // Residual connection
      for (let t = 0; t < cache.seqLen; t++) {
        for (let d = 0; d < embeddingDim; d++) {
          blockCache.postAttn[t][d] = blockCache.input[t][d] +
            blockCache.attnOut[t][d];
        }
      }

      // Pre-FFN LayerNorm
      this.layerNorm(
        blockCache.postAttn,
        blockCache.ln2Out,
        blockWeights.ln2Gamma,
        blockWeights.ln2Beta,
        blockCache.ln2Mean,
        blockCache.ln2Var,
        cache.seqLen,
      );

      // FFN
      this.forwardFFN(
        blockCache.ln2Out,
        blockWeights,
        blockCache,
        cache.seqLen,
      );

      // Residual connection and update fused
      for (let t = 0; t < cache.seqLen; t++) {
        for (let d = 0; d < embeddingDim; d++) {
          cache.fused[t][d] = blockCache.postAttn[t][d] +
            blockCache.ffnOut[t][d];
        }
      }
    }
  }

  /**
   * Forward pass: attention pooling to fixed vector
   */
  private forwardPooling(cache: ForwardCache): void {
    const { embeddingDim } = this.config;
    const w = this.weights!.poolWeight;
    const b = this.weights!.poolBias[0];

    // Compute pooling logits
    let maxLogit = -Infinity;
    for (let t = 0; t < cache.seqLen; t++) {
      let dot = b;
      for (let d = 0; d < embeddingDim; d++) {
        dot += cache.fused[t][d] * w[d];
      }
      cache.poolLogits[t] = dot;
      if (dot > maxLogit) maxLogit = dot;
    }

    // Softmax
    let sumExp = 0;
    for (let t = 0; t < cache.seqLen; t++) {
      cache.poolWeights[t] = Math.exp(cache.poolLogits[t] - maxLogit);
      sumExp += cache.poolWeights[t];
    }
    for (let t = 0; t < cache.seqLen; t++) {
      cache.poolWeights[t] /= sumExp + 1e-12;
    }

    // Weighted sum
    for (let d = 0; d < embeddingDim; d++) {
      let sum = 0;
      for (let t = 0; t < cache.seqLen; t++) {
        sum += cache.poolWeights[t] * cache.fused[t][d];
      }
      cache.aggregated[d] = sum;
    }
  }

  /**
   * Forward pass: output projection
   */
  private forwardOutput(cache: ForwardCache): void {
    const { embeddingDim } = this.config;
    const w = this.weights!.outputWeight;
    const b = this.weights!.outputBias;

    for (let k = 0; k < this.outputDim; k++) {
      let sum = b[k];
      for (let d = 0; d < embeddingDim; d++) {
        sum += cache.aggregated[d] * w[k * embeddingDim + d];
      }
      cache.yPredNorm[k] = sum;
    }
  }

  /**
   * Complete forward pass
   */
  private forward(x: number[][], yTarget: number[], cache: ForwardCache): void {
    cache.seqLen = x.length;

    // Normalize inputs
    this.normalizeInput(x, cache);

    // Normalize target
    for (let k = 0; k < this.outputDim; k++) {
      cache.yTarget[k] = yTarget[k];
      cache.yTargetNorm[k] = (yTarget[k] - this.outputMean![k]) /
        (this.outputStd![k] + this.config.epsilon);
    }

    // Forward pass through layers
    this.forwardInputProjection(cache);
    this.forwardTemporalConv(cache);
    this.forwardFusion(cache);
    this.forwardTransformerBlocks(cache);
    this.forwardPooling(cache);
    this.forwardOutput(cache);
  }

  /**
   * Compute loss and check for outliers
   */
  private computeLoss(
    cache: ForwardCache,
  ): {
    mse: number;
    l2: number;
    rNorm: number;
    isOutlier: boolean;
    sampleWeight: number;
  } {
    const { regularizationStrength, outlierThreshold } = this.config;

    // MSE in normalized space
    let mse = 0;
    let rNormSq = 0;
    for (let k = 0; k < this.outputDim; k++) {
      const r = cache.yTargetNorm[k] - cache.yPredNorm[k];
      mse += r * r;
      rNormSq += r * r;
    }
    mse /= this.outputDim;
    const rNorm = Math.sqrt(rNormSq);

    // L2 regularization
    let l2 = 0;
    l2 += this.computeWeightNormSquared(this.weights!.inputWeight);
    l2 += this.computeWeightNormSquared(this.weights!.outputWeight);
    for (const cw of this.weights!.convWeights) {
      l2 += this.computeWeightNormSquared(cw);
    }
    for (const block of this.weights!.blocks) {
      l2 += this.computeWeightNormSquared(block.wq);
      l2 += this.computeWeightNormSquared(block.wk);
      l2 += this.computeWeightNormSquared(block.wv);
      l2 += this.computeWeightNormSquared(block.wo);
      l2 += this.computeWeightNormSquared(block.ffnW1);
      l2 += this.computeWeightNormSquared(block.ffnW2);
    }
    l2 *= 0.5 * regularizationStrength;

    // Outlier detection
    const isOutlier = rNorm > outlierThreshold;
    const sampleWeight = isOutlier
      ? Math.min(0.1, outlierThreshold / rNorm)
      : 1.0;

    return { mse, l2, rNorm, isOutlier, sampleWeight };
  }

  private computeWeightNormSquared(w: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < w.length; i++) {
      sum += w[i] * w[i];
    }
    return sum;
  }

  /**
   * Zero all gradients
   */
  private zeroGradients(): void {
    const g = this.gradients!;
    g.inputWeight.fill(0);
    g.inputBias.fill(0);
    g.outputWeight.fill(0);
    g.outputBias.fill(0);
    g.poolWeight.fill(0);
    g.poolBias.fill(0);
    g.gateBiases.fill(0);

    for (let s = 0; s < g.convWeights.length; s++) {
      g.convWeights[s].fill(0);
      g.convBiases[s].fill(0);
      g.scaleEmb[s].fill(0);
      g.gateWeights[s].fill(0);
    }

    for (const block of g.blocks) {
      block.ln1Gamma.fill(0);
      block.ln1Beta.fill(0);
      block.wq.fill(0);
      block.wk.fill(0);
      block.wv.fill(0);
      block.wo.fill(0);
      block.bo.fill(0);
      block.relPosBias.fill(0);
      block.ln2Gamma.fill(0);
      block.ln2Beta.fill(0);
      block.ffnW1.fill(0);
      block.ffnB1.fill(0);
      block.ffnW2.fill(0);
      block.ffnB2.fill(0);
    }
  }

  /**
   * Backward pass: compute all gradients
   */
  private backward(cache: ForwardCache, sampleWeight: number): void {
    const {
      embeddingDim,
      numBlocks,
      temporalScales,
      temporalKernelSize,
      hiddenDim,
      numHeads,
      headDim,
      maxSequenceLength,
      regularizationStrength,
    } = this.config;
    const numScales = temporalScales.length;
    const g = this.gradients!;
    const w = this.weights!;

    this.zeroGradients();

    // Gradient of loss w.r.t. yPredNorm
    // d(mse)/d(yPredNorm) = -2/outputDim * (yTargetNorm - yPredNorm) * sampleWeight
    const dYPredNorm = new Float64Array(this.outputDim);
    for (let k = 0; k < this.outputDim; k++) {
      dYPredNorm[k] = -2 * sampleWeight *
        (cache.yTargetNorm[k] - cache.yPredNorm[k]) / this.outputDim;
    }

    // Backward through output layer
    const dAgg = new Float64Array(embeddingDim);
    for (let k = 0; k < this.outputDim; k++) {
      g.outputBias[k] += dYPredNorm[k];
      for (let d = 0; d < embeddingDim; d++) {
        g.outputWeight[k * embeddingDim + d] += dYPredNorm[k] *
          cache.aggregated[d];
        dAgg[d] += dYPredNorm[k] * w.outputWeight[k * embeddingDim + d];
      }
    }

    // Backward through pooling
    const dFused: Float64Array[] = [];
    for (let t = 0; t < cache.seqLen; t++) {
      dFused.push(new Float64Array(embeddingDim));
    }

    // dPoolLogits from softmax * weighted sum
    const dPoolLogits = new Float64Array(cache.seqLen);
    for (let t = 0; t < cache.seqLen; t++) {
      let dot = 0;
      for (let d = 0; d < embeddingDim; d++) {
        dot += dAgg[d] * cache.fused[t][d];
      }
      for (let j = 0; j < cache.seqLen; j++) {
        if (t === j) {
          dPoolLogits[t] += dot * cache.poolWeights[t] *
            (1 - cache.poolWeights[t]);
        } else {
          dPoolLogits[t] -= dot * cache.poolWeights[t] * cache.poolWeights[j];
        }
      }
    }

    for (let t = 0; t < cache.seqLen; t++) {
      g.poolBias[0] += dPoolLogits[t];
      for (let d = 0; d < embeddingDim; d++) {
        g.poolWeight[d] += dPoolLogits[t] * cache.fused[t][d];
        dFused[t][d] += dAgg[d] * cache.poolWeights[t];
      }
    }

    // Backward through transformer blocks (reverse order)
    for (let b = numBlocks - 1; b >= 0; b--) {
      const blockWeights = w.blocks[b];
      const blockGrads = g.blocks[b];
      const blockCache = cache.blockCaches[b];

      // Backward through FFN residual
      const dPostAttn: Float64Array[] = [];
      const dFFNOut: Float64Array[] = [];
      for (let t = 0; t < cache.seqLen; t++) {
        dPostAttn.push(new Float64Array(embeddingDim));
        dFFNOut.push(new Float64Array(embeddingDim));
        for (let d = 0; d < embeddingDim; d++) {
          dPostAttn[t][d] = dFused[t][d];
          dFFNOut[t][d] = dFused[t][d];
        }
      }

      // Backward through FFN
      const dLn2Out: Float64Array[] = [];
      for (let t = 0; t < cache.seqLen; t++) {
        dLn2Out.push(new Float64Array(embeddingDim));
      }

      for (let t = 0; t < cache.seqLen; t++) {
        // Backward through second linear
        const dHidden = new Float64Array(hiddenDim);
        for (let d = 0; d < embeddingDim; d++) {
          blockGrads.ffnB2[d] += dFFNOut[t][d];
          for (let h = 0; h < hiddenDim; h++) {
            blockGrads.ffnW2[d * hiddenDim + h] += dFFNOut[t][d] *
              blockCache.ffnHidden[t][h];
            dHidden[h] += dFFNOut[t][d] * blockWeights.ffnW2[d * hiddenDim + h];
          }
        }

        // Backward through GELU
        for (let h = 0; h < hiddenDim; h++) {
          // Recompute pre-GELU value
          let preGelu = blockWeights.ffnB1[h];
          for (let d = 0; d < embeddingDim; d++) {
            preGelu += blockCache.ln2Out[t][d] *
              blockWeights.ffnW1[h * embeddingDim + d];
          }
          const dPreGelu = dHidden[h] * geluDerivative(preGelu);

          blockGrads.ffnB1[h] += dPreGelu;
          for (let d = 0; d < embeddingDim; d++) {
            blockGrads.ffnW1[h * embeddingDim + d] += dPreGelu *
              blockCache.ln2Out[t][d];
            dLn2Out[t][d] += dPreGelu *
              blockWeights.ffnW1[h * embeddingDim + d];
          }
        }
      }

      // Backward through LN2
      this.backwardLayerNorm(
        blockCache.postAttn,
        dLn2Out,
        dPostAttn,
        blockWeights.ln2Gamma,
        blockGrads.ln2Gamma,
        blockGrads.ln2Beta,
        blockCache.ln2Mean,
        blockCache.ln2Var,
        cache.seqLen,
      );

      // Backward through attention residual
      const dInput: Float64Array[] = [];
      const dAttnOut: Float64Array[] = [];
      for (let t = 0; t < cache.seqLen; t++) {
        dInput.push(new Float64Array(embeddingDim));
        dAttnOut.push(new Float64Array(embeddingDim));
        for (let d = 0; d < embeddingDim; d++) {
          dInput[t][d] = dPostAttn[t][d];
          dAttnOut[t][d] = dPostAttn[t][d];
        }
      }

      // Backward through attention output projection
      const dHeadOuts: Float64Array[][] = [];
      for (let h = 0; h < numHeads; h++) {
        const headOuts: Float64Array[] = [];
        for (let t = 0; t < cache.seqLen; t++) {
          headOuts.push(new Float64Array(headDim));
        }
        dHeadOuts.push(headOuts);
      }

      for (let t = 0; t < cache.seqLen; t++) {
        for (let d = 0; d < embeddingDim; d++) {
          blockGrads.bo[d] += dAttnOut[t][d];
          for (let h = 0; h < numHeads; h++) {
            const headOffset = h * headDim;
            for (let hd = 0; hd < headDim; hd++) {
              blockGrads.wo[d * embeddingDim + headOffset + hd] +=
                dAttnOut[t][d] * blockCache.headOuts[h][t][hd];
              dHeadOuts[h][t][hd] += dAttnOut[t][d] *
                blockWeights.wo[d * embeddingDim + headOffset + hd];
            }
          }
        }
      }

      // Backward through attention
      const dLn1Out: Float64Array[] = [];
      for (let t = 0; t < cache.seqLen; t++) {
        dLn1Out.push(new Float64Array(embeddingDim));
      }

      const scale = 1 / Math.sqrt(headDim);

      for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;

        // Backward through attention weighted sum and softmax
        const dV: Float64Array[] = [];
        const dAttnWeights: Float64Array[] = [];
        for (let t = 0; t < cache.seqLen; t++) {
          dV.push(new Float64Array(headDim));
          dAttnWeights.push(new Float64Array(cache.seqLen));
        }

        for (let i = 0; i < cache.seqLen; i++) {
          for (let hd = 0; hd < headDim; hd++) {
            for (let j = 0; j < cache.seqLen; j++) {
              dAttnWeights[i][j] += dHeadOuts[h][i][hd] *
                blockCache.v[h][j][hd];
              dV[j][hd] += dHeadOuts[h][i][hd] *
                blockCache.attnWeights[h][i][j];
            }
          }
        }

        // Backward through softmax
        const dScores: Float64Array[] = [];
        for (let t = 0; t < cache.seqLen; t++) {
          dScores.push(new Float64Array(cache.seqLen));
        }

        for (let i = 0; i < cache.seqLen; i++) {
          for (let j = 0; j < cache.seqLen; j++) {
            for (let k = 0; k < cache.seqLen; k++) {
              const delta = j === k ? 1 : 0;
              dScores[i][j] += dAttnWeights[i][k] *
                blockCache.attnWeights[h][i][k] *
                (delta - blockCache.attnWeights[h][i][j]);
            }
          }
        }

        // Backward through scores
        const dQ: Float64Array[] = [];
        const dK: Float64Array[] = [];
        for (let t = 0; t < cache.seqLen; t++) {
          dQ.push(new Float64Array(headDim));
          dK.push(new Float64Array(headDim));
        }

        for (let i = 0; i < cache.seqLen; i++) {
          for (let j = 0; j < cache.seqLen; j++) {
            // Skip causal masked positions
            if (j > i) continue;

            const relPos = i - j + maxSequenceLength - 1;
            blockGrads.relPosBias[relPos] += dScores[i][j];

            for (let hd = 0; hd < headDim; hd++) {
              dQ[i][hd] += dScores[i][j] * scale * blockCache.k[h][j][hd];
              dK[j][hd] += dScores[i][j] * scale * blockCache.q[h][i][hd];
            }
          }
        }

        // Backward through Q, K, V projections
        for (let t = 0; t < cache.seqLen; t++) {
          for (let hd = 0; hd < headDim; hd++) {
            for (let d = 0; d < embeddingDim; d++) {
              blockGrads.wq[(headOffset + hd) * embeddingDim + d] += dQ[t][hd] *
                blockCache.ln1Out[t][d];
              blockGrads.wk[(headOffset + hd) * embeddingDim + d] += dK[t][hd] *
                blockCache.ln1Out[t][d];
              blockGrads.wv[(headOffset + hd) * embeddingDim + d] += dV[t][hd] *
                blockCache.ln1Out[t][d];

              dLn1Out[t][d] += dQ[t][hd] *
                blockWeights.wq[(headOffset + hd) * embeddingDim + d];
              dLn1Out[t][d] += dK[t][hd] *
                blockWeights.wk[(headOffset + hd) * embeddingDim + d];
              dLn1Out[t][d] += dV[t][hd] *
                blockWeights.wv[(headOffset + hd) * embeddingDim + d];
            }
          }
        }
      }

      // Backward through LN1
      this.backwardLayerNorm(
        blockCache.input,
        dLn1Out,
        dInput,
        blockWeights.ln1Gamma,
        blockGrads.ln1Gamma,
        blockGrads.ln1Beta,
        blockCache.ln1Mean,
        blockCache.ln1Var,
        cache.seqLen,
      );

      // Pass gradient to previous block
      for (let t = 0; t < cache.seqLen; t++) {
        for (let d = 0; d < embeddingDim; d++) {
          dFused[t][d] = dInput[t][d];
        }
      }
    }

    // Backward through fusion
    const dAligned: Float64Array[][] = [];
    for (let s = 0; s < numScales; s++) {
      const aligned: Float64Array[] = [];
      for (let t = 0; t < cache.seqLen; t++) {
        aligned.push(new Float64Array(embeddingDim));
      }
      dAligned.push(aligned);
    }

    for (let t = 0; t < cache.seqLen; t++) {
      // Backward through weighted sum
      for (let s = 0; s < numScales; s++) {
        for (let d = 0; d < embeddingDim; d++) {
          dAligned[s][t][d] += dFused[t][d] * cache.gates[t][s];
        }
      }

      // Backward through softmax gates
      const dGateLogits = new Float64Array(numScales);
      for (let s = 0; s < numScales; s++) {
        let dot = 0;
        for (let d = 0; d < embeddingDim; d++) {
          dot += dFused[t][d] * cache.alignedEmbeddings[s][t][d];
        }
        for (let r = 0; r < numScales; r++) {
          const delta = s === r ? 1 : 0;
          dGateLogits[s] += dot * cache.gates[t][r] *
            (delta - cache.gates[t][s]);
        }
      }

      for (let s = 0; s < numScales; s++) {
        g.gateBiases[s] += dGateLogits[s];
        for (let d = 0; d < embeddingDim; d++) {
          g.gateWeights[s][d] += dGateLogits[s] *
            cache.alignedEmbeddings[s][t][d];
          dAligned[s][t][d] += dGateLogits[s] * w.gateWeights[s][d];
        }
      }
    }

    // Backward through scale alignment and temporal conv
    const dE0: Float64Array[] = [];
    for (let t = 0; t < cache.seqLen; t++) {
      dE0.push(new Float64Array(embeddingDim));
    }

    for (let s = 0; s < numScales; s++) {
      const scale = temporalScales[s];
      const scaleLen = Math.ceil(cache.seqLen / scale);

      // Accumulate gradients from alignment
      const dScaleEmb: Float64Array[] = [];
      for (let u = 0; u < scaleLen; u++) {
        dScaleEmb.push(new Float64Array(embeddingDim));
      }

      for (let t = 0; t < cache.seqLen; t++) {
        const u = Math.min(Math.floor(t / scale), scaleLen - 1);
        for (let d = 0; d < embeddingDim; d++) {
          dScaleEmb[u][d] += dAligned[s][t][d];
        }
      }

      // Backward through scale embedding and positional encoding
      for (let u = 0; u < scaleLen; u++) {
        for (let d = 0; d < embeddingDim; d++) {
          g.scaleEmb[s][d] += dScaleEmb[u][d];
        }
      }

      // Backward through GELU
      const dConvOut: Float64Array[] = [];
      for (let u = 0; u < scaleLen; u++) {
        dConvOut.push(new Float64Array(embeddingDim));
        for (let d = 0; d < embeddingDim; d++) {
          dConvOut[u][d] = dScaleEmb[u][d] *
            geluDerivative(cache.convOutputs[s][u][d]);
        }
      }

      // Backward through convolution
      const pad = Math.floor(temporalKernelSize / 2);
      for (let u = 0; u < scaleLen; u++) {
        const center = u * scale;

        for (let d = 0; d < embeddingDim; d++) {
          g.convBiases[s][d] += dConvOut[u][d];

          for (let k = 0; k < temporalKernelSize; k++) {
            const srcT = center + k - pad;
            if (srcT >= 0 && srcT < cache.seqLen) {
              for (let d2 = 0; d2 < embeddingDim; d2++) {
                g.convWeights[s][
                  (d * embeddingDim + d2) * temporalKernelSize + k
                ] += dConvOut[u][d] * cache.e0[srcT][d2];
                dE0[srcT][d2] += dConvOut[u][d] *
                  w.convWeights[s][
                    (d * embeddingDim + d2) * temporalKernelSize + k
                  ];
              }
            }
          }
        }
      }
    }

    // Backward through input projection
    for (let t = 0; t < cache.seqLen; t++) {
      for (let d = 0; d < embeddingDim; d++) {
        g.inputBias[d] += dE0[t][d];
        for (let c = 0; c < this.inputDim; c++) {
          g.inputWeight[d * this.inputDim + c] += dE0[t][d] * cache.xNorm[t][c];
        }
      }
    }

    // Add L2 regularization gradients
    for (let i = 0; i < w.inputWeight.length; i++) {
      g.inputWeight[i] += regularizationStrength * w.inputWeight[i];
    }
    for (let i = 0; i < w.outputWeight.length; i++) {
      g.outputWeight[i] += regularizationStrength * w.outputWeight[i];
    }
    for (let s = 0; s < numScales; s++) {
      for (let i = 0; i < w.convWeights[s].length; i++) {
        g.convWeights[s][i] += regularizationStrength * w.convWeights[s][i];
      }
    }
    for (const block of w.blocks) {
      for (let i = 0; i < block.wq.length; i++) {
        g.blocks[w.blocks.indexOf(block)].wq[i] += regularizationStrength *
          block.wq[i];
        g.blocks[w.blocks.indexOf(block)].wk[i] += regularizationStrength *
          block.wk[i];
        g.blocks[w.blocks.indexOf(block)].wv[i] += regularizationStrength *
          block.wv[i];
        g.blocks[w.blocks.indexOf(block)].wo[i] += regularizationStrength *
          block.wo[i];
      }
      for (let i = 0; i < block.ffnW1.length; i++) {
        g.blocks[w.blocks.indexOf(block)].ffnW1[i] += regularizationStrength *
          block.ffnW1[i];
      }
      for (let i = 0; i < block.ffnW2.length; i++) {
        g.blocks[w.blocks.indexOf(block)].ffnW2[i] += regularizationStrength *
          block.ffnW2[i];
      }
    }
  }

  /**
   * Backward through layer normalization
   */
  private backwardLayerNorm(
    input: Float64Array[],
    dOutput: Float64Array[],
    dInput: Float64Array[],
    gamma: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
    means: Float64Array,
    vars: Float64Array,
    seqLen: number,
  ): void {
    const { embeddingDim, epsilon } = this.config;

    for (let t = 0; t < seqLen; t++) {
      const mean = means[t];
      const variance = vars[t];
      const invStd = 1 / Math.sqrt(variance + epsilon);

      // Compute normalized values
      let dVar = 0;
      let dMean = 0;

      for (let d = 0; d < embeddingDim; d++) {
        const xHat = (input[t][d] - mean) * invStd;
        dBeta[d] += dOutput[t][d];
        dGamma[d] += dOutput[t][d] * xHat;

        const dxHat = dOutput[t][d] * gamma[d];
        dVar += dxHat * (input[t][d] - mean) * (-0.5) *
          Math.pow(variance + epsilon, -1.5);
        dMean += dxHat * (-invStd);
      }

      dMean += dVar * (-2 / embeddingDim) *
        (input[t].reduce((sum, x) => sum + x, 0) - embeddingDim * mean);

      for (let d = 0; d < embeddingDim; d++) {
        const dxHat = dOutput[t][d] * gamma[d];
        dInput[t][d] += dxHat * invStd +
          dVar * 2 * (input[t][d] - mean) / embeddingDim + dMean / embeddingDim;
      }
    }
  }

  /**
   * Compute gradient norm
   */
  private computeGradientNorm(): number {
    let norm = 0;
    const g = this.gradients!;

    norm += this.computeWeightNormSquared(g.inputWeight);
    norm += this.computeWeightNormSquared(g.inputBias);
    norm += this.computeWeightNormSquared(g.outputWeight);
    norm += this.computeWeightNormSquared(g.outputBias);
    norm += this.computeWeightNormSquared(g.poolWeight);
    norm += this.computeWeightNormSquared(g.poolBias);
    norm += this.computeWeightNormSquared(g.gateBiases);

    for (let s = 0; s < g.convWeights.length; s++) {
      norm += this.computeWeightNormSquared(g.convWeights[s]);
      norm += this.computeWeightNormSquared(g.convBiases[s]);
      norm += this.computeWeightNormSquared(g.scaleEmb[s]);
      norm += this.computeWeightNormSquared(g.gateWeights[s]);
    }

    for (const block of g.blocks) {
      norm += this.computeWeightNormSquared(block.ln1Gamma);
      norm += this.computeWeightNormSquared(block.ln1Beta);
      norm += this.computeWeightNormSquared(block.wq);
      norm += this.computeWeightNormSquared(block.wk);
      norm += this.computeWeightNormSquared(block.wv);
      norm += this.computeWeightNormSquared(block.wo);
      norm += this.computeWeightNormSquared(block.bo);
      norm += this.computeWeightNormSquared(block.relPosBias);
      norm += this.computeWeightNormSquared(block.ln2Gamma);
      norm += this.computeWeightNormSquared(block.ln2Beta);
      norm += this.computeWeightNormSquared(block.ffnW1);
      norm += this.computeWeightNormSquared(block.ffnB1);
      norm += this.computeWeightNormSquared(block.ffnW2);
      norm += this.computeWeightNormSquared(block.ffnB2);
    }

    return Math.sqrt(norm);
  }

  /**
   * Clip gradients by global norm
   */
  private clipGradients(maxNorm: number): void {
    const norm = this.computeGradientNorm();
    if (norm > maxNorm) {
      const scale = maxNorm / (norm + 1e-12);
      this.scaleGradients(scale);
    }
  }

  private scaleGradients(scale: number): void {
    const g = this.gradients!;

    for (let i = 0; i < g.inputWeight.length; i++) g.inputWeight[i] *= scale;
    for (let i = 0; i < g.inputBias.length; i++) g.inputBias[i] *= scale;
    for (let i = 0; i < g.outputWeight.length; i++) g.outputWeight[i] *= scale;
    for (let i = 0; i < g.outputBias.length; i++) g.outputBias[i] *= scale;
    for (let i = 0; i < g.poolWeight.length; i++) g.poolWeight[i] *= scale;
    for (let i = 0; i < g.poolBias.length; i++) g.poolBias[i] *= scale;
    for (let i = 0; i < g.gateBiases.length; i++) g.gateBiases[i] *= scale;

    for (let s = 0; s < g.convWeights.length; s++) {
      for (let i = 0; i < g.convWeights[s].length; i++) {
        g.convWeights[s][i] *= scale;
      }
      for (let i = 0; i < g.convBiases[s].length; i++) {
        g.convBiases[s][i] *= scale;
      }
      for (let i = 0; i < g.scaleEmb[s].length; i++) g.scaleEmb[s][i] *= scale;
      for (let i = 0; i < g.gateWeights[s].length; i++) {
        g.gateWeights[s][i] *= scale;
      }
    }

    for (const block of g.blocks) {
      for (let i = 0; i < block.ln1Gamma.length; i++) {
        block.ln1Gamma[i] *= scale;
      }
      for (let i = 0; i < block.ln1Beta.length; i++) block.ln1Beta[i] *= scale;
      for (let i = 0; i < block.wq.length; i++) block.wq[i] *= scale;
      for (let i = 0; i < block.wk.length; i++) block.wk[i] *= scale;
      for (let i = 0; i < block.wv.length; i++) block.wv[i] *= scale;
      for (let i = 0; i < block.wo.length; i++) block.wo[i] *= scale;
      for (let i = 0; i < block.bo.length; i++) block.bo[i] *= scale;
      for (let i = 0; i < block.relPosBias.length; i++) {
        block.relPosBias[i] *= scale;
      }
      for (let i = 0; i < block.ln2Gamma.length; i++) {
        block.ln2Gamma[i] *= scale;
      }
      for (let i = 0; i < block.ln2Beta.length; i++) block.ln2Beta[i] *= scale;
      for (let i = 0; i < block.ffnW1.length; i++) block.ffnW1[i] *= scale;
      for (let i = 0; i < block.ffnB1.length; i++) block.ffnB1[i] *= scale;
      for (let i = 0; i < block.ffnW2.length; i++) block.ffnW2[i] *= scale;
      for (let i = 0; i < block.ffnB2.length; i++) block.ffnB2[i] *= scale;
    }
  }

  /**
   * Get learning rate with warmup and cosine decay
   */
  private getLearningRate(): number {
    const { learningRate, warmupSteps, totalSteps } = this.config;

    if (this.step < warmupSteps) {
      return learningRate * (this.step / warmupSteps);
    }

    const progress = (this.step - warmupSteps) /
      Math.max(1, totalSteps - warmupSteps);
    return learningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
  }

  /**
   * Adam optimizer update
   */
  private adamUpdate(lr: number): void {
    const { beta1, beta2, epsilon } = this.config;
    const w = this.weights!;
    const g = this.gradients!;
    const m = this.firstMoment!;
    const v = this.secondMoment!;

    const t = this.step + 1;
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    // Update helper function
    const update = (
      weight: Float64Array,
      grad: Float64Array,
      moment1: Float64Array,
      moment2: Float64Array,
    ) => {
      for (let i = 0; i < weight.length; i++) {
        moment1[i] = beta1 * moment1[i] + (1 - beta1) * grad[i];
        moment2[i] = beta2 * moment2[i] + (1 - beta2) * grad[i] * grad[i];

        const mHat = moment1[i] / bc1;
        const vHat = moment2[i] / bc2;

        weight[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
      }
    };

    update(w.inputWeight, g.inputWeight, m.inputWeight, v.inputWeight);
    update(w.inputBias, g.inputBias, m.inputBias, v.inputBias);
    update(w.outputWeight, g.outputWeight, m.outputWeight, v.outputWeight);
    update(w.outputBias, g.outputBias, m.outputBias, v.outputBias);
    update(w.poolWeight, g.poolWeight, m.poolWeight, v.poolWeight);
    update(w.poolBias, g.poolBias, m.poolBias, v.poolBias);
    update(w.gateBiases, g.gateBiases, m.gateBiases, v.gateBiases);

    for (let s = 0; s < w.convWeights.length; s++) {
      update(
        w.convWeights[s],
        g.convWeights[s],
        m.convWeights[s],
        v.convWeights[s],
      );
      update(
        w.convBiases[s],
        g.convBiases[s],
        m.convBiases[s],
        v.convBiases[s],
      );
      update(w.scaleEmb[s], g.scaleEmb[s], m.scaleEmb[s], v.scaleEmb[s]);
      update(
        w.gateWeights[s],
        g.gateWeights[s],
        m.gateWeights[s],
        v.gateWeights[s],
      );
    }

    for (let b = 0; b < w.blocks.length; b++) {
      update(
        w.blocks[b].ln1Gamma,
        g.blocks[b].ln1Gamma,
        m.blocks[b].ln1Gamma,
        v.blocks[b].ln1Gamma,
      );
      update(
        w.blocks[b].ln1Beta,
        g.blocks[b].ln1Beta,
        m.blocks[b].ln1Beta,
        v.blocks[b].ln1Beta,
      );
      update(w.blocks[b].wq, g.blocks[b].wq, m.blocks[b].wq, v.blocks[b].wq);
      update(w.blocks[b].wk, g.blocks[b].wk, m.blocks[b].wk, v.blocks[b].wk);
      update(w.blocks[b].wv, g.blocks[b].wv, m.blocks[b].wv, v.blocks[b].wv);
      update(w.blocks[b].wo, g.blocks[b].wo, m.blocks[b].wo, v.blocks[b].wo);
      update(w.blocks[b].bo, g.blocks[b].bo, m.blocks[b].bo, v.blocks[b].bo);
      update(
        w.blocks[b].relPosBias,
        g.blocks[b].relPosBias,
        m.blocks[b].relPosBias,
        v.blocks[b].relPosBias,
      );
      update(
        w.blocks[b].ln2Gamma,
        g.blocks[b].ln2Gamma,
        m.blocks[b].ln2Gamma,
        v.blocks[b].ln2Gamma,
      );
      update(
        w.blocks[b].ln2Beta,
        g.blocks[b].ln2Beta,
        m.blocks[b].ln2Beta,
        v.blocks[b].ln2Beta,
      );
      update(
        w.blocks[b].ffnW1,
        g.blocks[b].ffnW1,
        m.blocks[b].ffnW1,
        v.blocks[b].ffnW1,
      );
      update(
        w.blocks[b].ffnB1,
        g.blocks[b].ffnB1,
        m.blocks[b].ffnB1,
        v.blocks[b].ffnB1,
      );
      update(
        w.blocks[b].ffnW2,
        g.blocks[b].ffnW2,
        m.blocks[b].ffnW2,
        v.blocks[b].ffnW2,
      );
      update(
        w.blocks[b].ffnB2,
        g.blocks[b].ffnB2,
        m.blocks[b].ffnB2,
        v.blocks[b].ffnB2,
      );
    }
  }

  /**
   * Update residual statistics for uncertainty estimation
   */
  private updateResidualStats(cache: ForwardCache): void {
    for (let k = 0; k < this.outputDim; k++) {
      const r = cache.yTargetNorm[k] - cache.yPredNorm[k];
      this.residualCount++;
      const delta = r - this.residualMean![k];
      this.residualMean![k] += delta / this.residualCount;
      const delta2 = r - this.residualMean![k];
      this.residualM2![k] += delta * delta2;
    }
  }

  /**
   * Perform one online training step
   * @param input - Training data with xCoordinates and yCoordinates
   * @returns Training result including loss, convergence status, etc.
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
   *   yCoordinates: [[10, 11]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(input: FitOnlineInput): FitResult {
    const { xCoordinates, yCoordinates } = input;

    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      throw new Error("Input sequences cannot be empty");
    }

    const seqLen = xCoordinates.length;
    const inputDim = xCoordinates[0].length;
    const yTarget = yCoordinates.length === 1
      ? yCoordinates[0]
      : yCoordinates[yCoordinates.length - 1];
    const outputDim = yTarget.length;

    // Lazy initialization
    if (!this.isInitialized) {
      this.inputDim = inputDim;
      this.outputDim = outputDim;

      this.initializeWeights(inputDim, outputDim);
      this.initializeBuffers(seqLen);

      this.inputMean = new Float64Array(inputDim);
      this.inputM2 = new Float64Array(inputDim);
      this.inputStd = new Float64Array(inputDim).fill(1);
      this.outputMean = new Float64Array(outputDim);
      this.outputM2 = new Float64Array(outputDim);
      this.outputStd = new Float64Array(outputDim).fill(1);

      this.residualMean = new Float64Array(outputDim);
      this.residualM2 = new Float64Array(outputDim);

      this.isInitialized = true;
    }

    // Reinitialize cache if sequence length changed
    if (!this.cache || this.cache.seqLen !== seqLen) {
      this.cache = this.initializeCache(seqLen);
    }

    // Update normalization statistics
    this.updateNormStats(xCoordinates, yTarget);

    // Forward pass
    this.forward(xCoordinates, yTarget, this.cache);

    // Compute loss
    const { mse, l2, rNorm, isOutlier, sampleWeight } = this.computeLoss(
      this.cache,
    );
    const loss = sampleWeight * mse + l2;

    // Backward pass
    this.backward(this.cache, sampleWeight);

    // Gradient clipping
    this.clipGradients(1.0);
    const gradientNorm = this.computeGradientNorm();

    // Get learning rate
    const lr = this.getLearningRate();

    // Adam update
    this.adamUpdate(lr);

    // Update residual statistics
    this.updateResidualStats(this.cache);

    // ADWIN drift detection
    const driftDetected = this.adwin.add(mse);
    if (driftDetected) {
      this.driftCount++;
      // Reset optimizer moments on drift
      this.firstMoment = this.createZeroWeights(this.inputDim, this.outputDim);
      this.secondMoment = this.createZeroWeights(this.inputDim, this.outputDim);
    }

    // Update running metrics
    this.sampleCount++;
    this.step++;
    this.prevAvgLoss = this.avgLoss;
    this.avgLoss += (loss - this.avgLoss) / this.sampleCount;

    // Check convergence
    this.converged = Math.abs(this.prevAvgLoss - this.avgLoss) <
      this.config.convergenceThreshold;

    // Store last input for prediction
    this.lastInput = xCoordinates.map((row) => Float64Array.from(row));
    this.lastSeqLen = seqLen;

    return {
      loss,
      gradientNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Make predictions for future steps
   * @param futureSteps - Number of future steps to predict
   * @returns Predictions with uncertainty bounds
   * @example
   * ```typescript
   * const predictions = model.predict(5);
   * predictions.predictions.forEach((p, i) => {
   *   console.log(`Step ${i}: ${p.predicted} ± ${p.standardError}`);
   * });
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.isInitialized || !this.lastInput) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const { epsilon } = this.config;

    // Use last input sequence for prediction
    let currentInput: any = this.lastInput.map((row) => Float64Array.from(row));

    for (let step = 0; step < futureSteps; step++) {
      // Create temporary cache if needed
      if (!this.cache || this.cache.seqLen !== currentInput.length) {
        this.cache = this.initializeCache(currentInput.length);
      }

      // Forward pass (with dummy target for shape)
      const dummyTarget = new Array(this.outputDim).fill(0);
      this.forward(
        currentInput.map((row: any) => Array.from(row)),
        dummyTarget,
        this.cache,
      );

      // Denormalize predictions
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let k = 0; k < this.outputDim; k++) {
        const predNorm = this.cache.yPredNorm[k];
        const pred = predNorm * (this.outputStd![k] + epsilon) +
          this.outputMean![k];
        predicted.push(pred);

        // Compute standard error
        const variance = this.residualCount > 1
          ? this.residualM2![k] / (this.residualCount - 1)
          : 0;
        const seNorm = Math.sqrt(variance) /
          Math.sqrt(Math.max(1, this.sampleCount));
        const se = seNorm * (this.outputStd![k] + epsilon);
        standardError.push(se);

        // 95% confidence interval
        lowerBound.push(pred - 1.96 * se);
        upperBound.push(pred + 1.96 * se);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });

      // Prepare next input by appending prediction (autoregressive)
      if (step < futureSteps - 1) {
        // Shift input and add prediction
        const newInput: Float64Array[] = [];
        for (let t = 1; t < currentInput.length; t++) {
          newInput.push(Float64Array.from(currentInput[t]));
        }
        // Use prediction as new input (assuming output dim <= input dim)
        const lastRow = new Float64Array(this.inputDim);
        for (let i = 0; i < Math.min(this.outputDim, this.inputDim); i++) {
          lastRow[i] = predicted[i];
        }
        newInput.push(lastRow);
        currentInput = newInput;
      }
    }

    const accuracy = 1 / (1 + this.avgLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get model summary information
   * @returns Summary including architecture, parameters, and training status
   */
  getModelSummary(): ModelSummary {
    let totalParameters = 0;

    if (this.weights) {
      totalParameters += this.weights.inputWeight.length +
        this.weights.inputBias.length;
      totalParameters += this.weights.outputWeight.length +
        this.weights.outputBias.length;
      totalParameters += this.weights.poolWeight.length +
        this.weights.poolBias.length;
      totalParameters += this.weights.gateBiases.length;

      for (let s = 0; s < this.weights.convWeights.length; s++) {
        totalParameters += this.weights.convWeights[s].length;
        totalParameters += this.weights.convBiases[s].length;
        totalParameters += this.weights.scaleEmb[s].length;
        totalParameters += this.weights.gateWeights[s].length;
      }

      for (const block of this.weights.blocks) {
        totalParameters += block.ln1Gamma.length + block.ln1Beta.length;
        totalParameters += block.wq.length + block.wk.length + block.wv.length;
        totalParameters += block.wo.length + block.bo.length;
        totalParameters += block.relPosBias.length;
        totalParameters += block.ln2Gamma.length + block.ln2Beta.length;
        totalParameters += block.ffnW1.length + block.ffnB1.length;
        totalParameters += block.ffnW2.length + block.ffnB2.length;
      }
    }

    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      temporalScales: [...this.config.temporalScales],
      totalParameters,
      sampleCount: this.sampleCount,
      accuracy: 1 / (1 + this.avgLoss),
      converged: this.converged,
      effectiveLearningRate: this.getLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Get all model weights
   * @returns Weight information including optimizer state
   */
  getWeights(): WeightInfo {
    if (!this.weights) {
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
        updateCount: 0,
      };
    }

    const toNestedArray = (arr: Float64Array, dims: number[]): number[][] => {
      const result: number[][] = [];
      let offset = 0;
      const rowSize = dims.length > 1 ? dims[1] : arr.length;
      const numRows = dims[0];
      for (let i = 0; i < numRows; i++) {
        result.push(Array.from(arr.slice(offset, offset + rowSize)));
        offset += rowSize;
      }
      return result;
    };

    const temporalConvWeights = this.weights.convWeights.map((w) =>
      toNestedArray(w, [
        this.config.embeddingDim,
        w.length / this.config.embeddingDim,
      ])
    );

    const scaleEmbeddings = this.weights.scaleEmb.map((e) => Array.from(e));

    const positionalEncoding: number[][][] = [];
    // Positional encoding is computed dynamically, return placeholder

    const fusionWeights = this.weights.gateWeights.map((w) => Array.from(w));

    const attentionWeights = this.weights.blocks.map((b) => [
      Array.from(b.wq),
      Array.from(b.wk),
      Array.from(b.wv),
      Array.from(b.wo),
    ]);

    const ffnWeights = this.weights.blocks.map((b) => [
      Array.from(b.ffnW1),
      Array.from(b.ffnW2),
    ]);

    const layerNormParams = this.weights.blocks.map((b) => [
      Array.from(b.ln1Gamma),
      Array.from(b.ln1Beta),
      Array.from(b.ln2Gamma),
      Array.from(b.ln2Beta),
    ]);

    const outputWeights = [
      Array.from(this.weights.outputWeight),
      Array.from(this.weights.outputBias),
    ];

    const firstMoment = this.firstMoment
      ? [
        [Array.from(this.firstMoment.inputWeight)],
        [Array.from(this.firstMoment.outputWeight)],
      ]
      : [];

    const secondMoment = this.secondMoment
      ? [
        [Array.from(this.secondMoment.inputWeight)],
        [Array.from(this.secondMoment.outputWeight)],
      ]
      : [];

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
      updateCount: this.step,
    };
  }

  /**
   * Get normalization statistics
   * @returns Current normalization statistics for inputs and outputs
   */
  getNormalizationStats(): NormalizationStats {
    return {
      inputMean: this.inputMean ? Array.from(this.inputMean) : [],
      inputStd: this.inputStd ? Array.from(this.inputStd) : [],
      outputMean: this.outputMean ? Array.from(this.outputMean) : [],
      outputStd: this.outputStd ? Array.from(this.outputStd) : [],
      count: this.normCount,
    };
  }

  /**
   * Reset model to initial state
   */
  reset(): void {
    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.weights = null;
    this.gradients = null;
    this.firstMoment = null;
    this.secondMoment = null;
    this.step = 0;
    this.inputMean = null;
    this.inputM2 = null;
    this.inputStd = null;
    this.outputMean = null;
    this.outputM2 = null;
    this.outputStd = null;
    this.normCount = 0;
    this.avgLoss = 0;
    this.prevAvgLoss = 0;
    this.sampleCount = 0;
    this.converged = false;
    this.residualMean = null;
    this.residualM2 = null;
    this.residualCount = 0;
    this.adwin.reset();
    this.driftCount = 0;
    this.buffers = null;
    this.cache = null;
    this.lastInput = null;
    this.lastSeqLen = 0;
    this.rng = new SeededRNG(42);
  }

  /**
   * Serialize model to JSON string
   * @returns JSON string containing all model state
   * @example
   * ```typescript
   * const savedModel = model.save();
   * localStorage.setItem('model', savedModel);
   * ```
   */
  save(): string {
    const serializeWeights = (w: ModelWeights | null) => {
      if (!w) return null;
      return {
        inputWeight: Array.from(w.inputWeight),
        inputBias: Array.from(w.inputBias),
        convWeights: w.convWeights.map((cw) => Array.from(cw)),
        convBiases: w.convBiases.map((cb) => Array.from(cb)),
        scaleEmb: w.scaleEmb.map((se) => Array.from(se)),
        gateWeights: w.gateWeights.map((gw) => Array.from(gw)),
        gateBiases: Array.from(w.gateBiases),
        blocks: w.blocks.map((b) => ({
          ln1Gamma: Array.from(b.ln1Gamma),
          ln1Beta: Array.from(b.ln1Beta),
          wq: Array.from(b.wq),
          wk: Array.from(b.wk),
          wv: Array.from(b.wv),
          wo: Array.from(b.wo),
          bo: Array.from(b.bo),
          relPosBias: Array.from(b.relPosBias),
          ln2Gamma: Array.from(b.ln2Gamma),
          ln2Beta: Array.from(b.ln2Beta),
          ffnW1: Array.from(b.ffnW1),
          ffnB1: Array.from(b.ffnB1),
          ffnW2: Array.from(b.ffnW2),
          ffnB2: Array.from(b.ffnB2),
        })),
        poolWeight: Array.from(w.poolWeight),
        poolBias: Array.from(w.poolBias),
        outputWeight: Array.from(w.outputWeight),
        outputBias: Array.from(w.outputBias),
      };
    };

    const state = {
      config: this.config,
      isInitialized: this.isInitialized,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      weights: serializeWeights(this.weights),
      firstMoment: serializeWeights(this.firstMoment),
      secondMoment: serializeWeights(this.secondMoment),
      step: this.step,
      inputMean: this.inputMean ? Array.from(this.inputMean) : null,
      inputM2: this.inputM2 ? Array.from(this.inputM2) : null,
      inputStd: this.inputStd ? Array.from(this.inputStd) : null,
      outputMean: this.outputMean ? Array.from(this.outputMean) : null,
      outputM2: this.outputM2 ? Array.from(this.outputM2) : null,
      outputStd: this.outputStd ? Array.from(this.outputStd) : null,
      normCount: this.normCount,
      avgLoss: this.avgLoss,
      prevAvgLoss: this.prevAvgLoss,
      sampleCount: this.sampleCount,
      converged: this.converged,
      residualMean: this.residualMean ? Array.from(this.residualMean) : null,
      residualM2: this.residualM2 ? Array.from(this.residualM2) : null,
      residualCount: this.residualCount,
      adwinState: this.adwin.getState(),
      driftCount: this.driftCount,
      lastInput: this.lastInput
        ? this.lastInput.map((row) => Array.from(row))
        : null,
      lastSeqLen: this.lastSeqLen,
    };

    return JSON.stringify(state);
  }

  /**
   * Load model from JSON string
   * @param json - JSON string from save()
   * @example
   * ```typescript
   * const savedModel = localStorage.getItem('model');
   * if (savedModel) {
   *   model.load(savedModel);
   * }
   * ```
   */
  load(json: string): void {
    const state = JSON.parse(json);

    const deserializeWeights = (w: any): ModelWeights | null => {
      if (!w) return null;
      return {
        inputWeight: Float64Array.from(w.inputWeight),
        inputBias: Float64Array.from(w.inputBias),
        convWeights: w.convWeights.map((cw: number[]) => Float64Array.from(cw)),
        convBiases: w.convBiases.map((cb: number[]) => Float64Array.from(cb)),
        scaleEmb: w.scaleEmb.map((se: number[]) => Float64Array.from(se)),
        gateWeights: w.gateWeights.map((gw: number[]) => Float64Array.from(gw)),
        gateBiases: Float64Array.from(w.gateBiases),
        blocks: w.blocks.map((b: any) => ({
          ln1Gamma: Float64Array.from(b.ln1Gamma),
          ln1Beta: Float64Array.from(b.ln1Beta),
          wq: Float64Array.from(b.wq),
          wk: Float64Array.from(b.wk),
          wv: Float64Array.from(b.wv),
          wo: Float64Array.from(b.wo),
          bo: Float64Array.from(b.bo),
          relPosBias: Float64Array.from(b.relPosBias),
          ln2Gamma: Float64Array.from(b.ln2Gamma),
          ln2Beta: Float64Array.from(b.ln2Beta),
          ffnW1: Float64Array.from(b.ffnW1),
          ffnB1: Float64Array.from(b.ffnB1),
          ffnW2: Float64Array.from(b.ffnW2),
          ffnB2: Float64Array.from(b.ffnB2),
        })),
        poolWeight: Float64Array.from(w.poolWeight),
        poolBias: Float64Array.from(w.poolBias),
        outputWeight: Float64Array.from(w.outputWeight),
        outputBias: Float64Array.from(w.outputBias),
      };
    };

    this.config = state.config;
    this.isInitialized = state.isInitialized;
    this.inputDim = state.inputDim;
    this.outputDim = state.outputDim;
    this.weights = deserializeWeights(state.weights);
    this.gradients = this.weights
      ? this.createZeroWeights(this.inputDim, this.outputDim)
      : null;
    this.firstMoment = deserializeWeights(state.firstMoment);
    this.secondMoment = deserializeWeights(state.secondMoment);
    this.step = state.step;
    this.inputMean = state.inputMean
      ? Float64Array.from(state.inputMean)
      : null;
    this.inputM2 = state.inputM2 ? Float64Array.from(state.inputM2) : null;
    this.inputStd = state.inputStd ? Float64Array.from(state.inputStd) : null;
    this.outputMean = state.outputMean
      ? Float64Array.from(state.outputMean)
      : null;
    this.outputM2 = state.outputM2 ? Float64Array.from(state.outputM2) : null;
    this.outputStd = state.outputStd
      ? Float64Array.from(state.outputStd)
      : null;
    this.normCount = state.normCount;
    this.avgLoss = state.avgLoss;
    this.prevAvgLoss = state.prevAvgLoss;
    this.sampleCount = state.sampleCount;
    this.converged = state.converged;
    this.residualMean = state.residualMean
      ? Float64Array.from(state.residualMean)
      : null;
    this.residualM2 = state.residualM2
      ? Float64Array.from(state.residualM2)
      : null;
    this.residualCount = state.residualCount;
    this.adwin.setState(state.adwinState);
    this.driftCount = state.driftCount;
    this.lastInput = state.lastInput
      ? state.lastInput.map((row: number[]) => Float64Array.from(row))
      : null;
    this.lastSeqLen = state.lastSeqLen;

    // Reinitialize buffers and cache
    if (this.isInitialized) {
      this.initializeBuffers(this.lastSeqLen || this.config.maxSequenceLength);
      this.cache = null; // Will be reinitialized on next forward pass
    }
  }
}

export default FusionTemporalTransformerRegression;
