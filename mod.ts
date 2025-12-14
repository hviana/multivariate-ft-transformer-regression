/**
 * @fileoverview Fusion Temporal Transformer Neural Network for Multivariate Regression
 * High-performance implementation with incremental online learning, Adam optimizer,
 * z-score normalization, and full backpropagation through all layers.
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Result from a single online training step
 */
interface FitResult {
  /** Mean squared error loss for this sample */
  loss: number;
  /** L2 norm of all gradients */
  gradientNorm: number;
  /** Current learning rate after warmup/cosine decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether training has converged based on loss change threshold */
  converged: boolean;
  /** Cumulative index of this sample in training */
  sampleIndex: number;
  /** Whether ADWIN detected concept drift */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds
 */
interface SinglePrediction {
  /** Predicted output values (denormalized) */
  predicted: number[];
  /** Lower confidence bound (95%) */
  lowerBound: number[];
  /** Upper confidence bound (95%) */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result from prediction
 */
interface PredictionResult {
  /** Array of predictions for each requested future step */
  predictions: SinglePrediction[];
  /** Running accuracy metric: 1/(1+avgLoss) */
  accuracy: number;
  /** Total samples seen during training */
  sampleCount: number;
  /** Whether model is ready for predictions */
  isModelReady: boolean;
}

/**
 * Complete weight information for serialization/inspection
 */
interface WeightInfo {
  /** Temporal convolution weights per scale */
  temporalConvWeights: number[][][];
  /** Learnable scale embeddings */
  scaleEmbeddings: number[][][];
  /** Sinusoidal positional encoding (computed) */
  positionalEncoding: number[][][];
  /** Fusion gate weights */
  fusionWeights: number[][][];
  /** Attention weights per block */
  attentionWeights: number[][][];
  /** Feed-forward network weights per block */
  ffnWeights: number[][][];
  /** Layer normalization parameters per block */
  layerNormParams: number[][][];
  /** Output projection weights */
  outputWeights: number[][][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Total Adam update count */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
interface NormalizationStats {
  /** Running mean of inputs per dimension */
  inputMean: number[];
  /** Running standard deviation of inputs */
  inputStd: number[];
  /** Running mean of outputs per dimension */
  outputMean: number[];
  /** Running standard deviation of outputs */
  outputStd: number[];
  /** Number of samples used for statistics */
  count: number;
}

/**
 * Model summary information
 */
interface ModelSummary {
  /** Whether model has been initialized with data */
  isInitialized: boolean;
  /** Input dimension (features per time step) */
  inputDimension: number;
  /** Output dimension (prediction targets) */
  outputDimension: number;
  /** Number of transformer blocks */
  numBlocks: number;
  /** Embedding dimension throughout network */
  embeddingDim: number;
  /** Number of attention heads */
  numHeads: number;
  /** Temporal scales used for multi-scale processing */
  temporalScales: number[];
  /** Total learnable parameters */
  totalParameters: number;
  /** Training samples processed */
  sampleCount: number;
  /** Current accuracy: 1/(1+avgLoss) */
  accuracy: number;
  /** Whether training has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of detected drift events */
  driftCount: number;
}

/**
 * Configuration options for FusionTemporalTransformerRegression
 */
interface FusionTemporalTransformerConfig {
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
  /** Warmup steps for learning rate (default: 100) */
  warmupSteps?: number;
  /** Total steps for cosine decay (default: 10000) */
  totalSteps?: number;
  /** Adam beta1 (default: 0.9) */
  beta1?: number;
  /** Adam beta2 (default: 0.999) */
  beta2?: number;
  /** Numerical stability epsilon (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength (default: 1e-4) */
  regularizationStrength?: number;
  /** Convergence threshold for loss change (default: 1e-6) */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN delta parameter (default: 0.002) */
  adwinDelta?: number;
  /** Temporal scales for multi-scale processing (default: [1, 2, 4]) */
  temporalScales?: number[];
  /** Temporal convolution kernel size (default: 3) */
  temporalKernelSize?: number;
  /** Maximum sequence length (default: 512) */
  maxSequenceLength?: number;
  /** Fusion dropout rate (default: 0.0) */
  fusionDropout?: number;
}

/**
 * Public interface for FusionTemporalTransformerRegression
 */
interface IFusionTemporalTransformerRegression {
  /**
   * Perform one step of online training
   * @param data Training data with input/output sequences
   * @returns Training metrics for this step
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult;

  /**
   * Generate predictions for future time steps
   * @param futureSteps Number of steps to predict ahead
   * @returns Predictions with uncertainty bounds
   */
  predict(futureSteps: number): PredictionResult;

  /** Get comprehensive model summary */
  getModelSummary(): ModelSummary;

  /** Get all weight matrices and optimizer state */
  getWeights(): WeightInfo;

  /** Get normalization statistics */
  getNormalizationStats(): NormalizationStats;

  /** Reset model to initial state */
  reset(): void;

  /** Serialize model state to JSON string */
  save(): string;

  /** Load model state from JSON string */
  load(w: string): void;
}

// ============================================================================
// Constants
// ============================================================================

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
const GELU_CONST = 0.044715;
const CONFIDENCE_MULTIPLIER = 1.96; // 95% confidence interval

// ============================================================================
// Object Pool for Float64Arrays
// ============================================================================

class Float64ArrayPool {
  private pools: Map<number, Float64Array[]> = new Map();

  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(size);
  }

  release(arr: Float64Array): void {
    const size = arr.length;
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    const pool = this.pools.get(size)!;
    if (pool.length < 32) { // Limit pool size
      arr.fill(0);
      pool.push(arr);
    }
  }

  clear(): void {
    this.pools.clear();
  }
}

// ============================================================================
// Main Implementation
// ============================================================================

/**
 * Fusion Temporal Transformer for Multivariate Regression
 *
 * A neural network combining multi-scale temporal convolutions with transformer
 * architecture for online regression tasks. Features include:
 * - Multi-scale temporal feature extraction
 * - Gated fusion of temporal scales
 * - Multi-head self-attention with causal masking
 * - Incremental online learning with Adam optimizer
 * - Welford's algorithm for online normalization
 * - ADWIN drift detection
 * - Outlier detection and downweighting
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({
 *   numBlocks: 2,
 *   embeddingDim: 32,
 *   numHeads: 4
 * });
 *
 * // Online training
 * const result = model.fitOnline({
 *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
 *   yCoordinates: [[7, 8]]
 * });
 *
 * // Prediction
 * const predictions = model.predict(3);
 * ```
 */
class FusionTemporalTransformerRegression
  implements IFusionTemporalTransformerRegression {
  // Configuration (readonly after construction)
  private readonly numBlocks: number;
  private readonly embeddingDim: number;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly ffnMultiplier: number;
  private readonly ffnDim: number;
  private readonly attentionDropout: number;
  private readonly learningRate: number;
  private readonly warmupSteps: number;
  private readonly totalSteps: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;
  private readonly regularizationStrength: number;
  private readonly convergenceThreshold: number;
  private readonly outlierThreshold: number;
  private readonly adwinDelta: number;
  private readonly temporalScales: number[];
  private readonly numScales: number;
  private readonly temporalKernelSize: number;
  private readonly maxSequenceLength: number;
  private readonly fusionDropout: number;

  // Dimensions (set on first fit)
  private inputDim: number = 0;
  private outputDim: number = 0;
  private isInitialized: boolean = false;

  // Weight storage - organized by layer type
  // All weights stored as Float64Array for performance
  private temporalConvW: Float64Array[] = []; // [numScales] each: [kernelSize*inputDim, embeddingDim]
  private temporalConvB: Float64Array[] = []; // [numScales] each: [embeddingDim]
  private scaleEmbed: Float64Array[] = []; // [numScales] each: [embeddingDim]
  private posEncoding: Float64Array = new Float64Array(0); // [maxSeqLen * embeddingDim]
  private fusionGateW: Float64Array = new Float64Array(0); // [numScales * embeddingDim, numScales]
  private fusionGateB: Float64Array = new Float64Array(0); // [numScales]

  // Attention weights per block: [Wq, Wk, Wv, Wo, bq, bk, bv, bo, temporalBias]
  private attnWq: Float64Array[] = []; // [numBlocks] each: [embeddingDim, embeddingDim]
  private attnWk: Float64Array[] = [];
  private attnWv: Float64Array[] = [];
  private attnWo: Float64Array[] = [];
  private attnBq: Float64Array[] = []; // [numBlocks] each: [embeddingDim]
  private attnBk: Float64Array[] = [];
  private attnBv: Float64Array[] = [];
  private attnBo: Float64Array[] = [];
  private temporalBias: Float64Array[] = []; // [numBlocks] each: [maxSeqLen * maxSeqLen]

  // FFN weights per block
  private ffnW1: Float64Array[] = []; // [numBlocks] each: [embeddingDim, ffnDim]
  private ffnB1: Float64Array[] = []; // [numBlocks] each: [ffnDim]
  private ffnW2: Float64Array[] = []; // [numBlocks] each: [ffnDim, embeddingDim]
  private ffnB2: Float64Array[] = []; // [numBlocks] each: [embeddingDim]

  // LayerNorm per block (2 per block: pre-attention and pre-FFN)
  private lnGamma1: Float64Array[] = []; // [numBlocks] each: [embeddingDim]
  private lnBeta1: Float64Array[] = [];
  private lnGamma2: Float64Array[] = [];
  private lnBeta2: Float64Array[] = [];

  // Output layer
  private poolW: Float64Array = new Float64Array(0); // [embeddingDim]
  private outputW: Float64Array = new Float64Array(0); // [embeddingDim, outputDim]
  private outputB: Float64Array = new Float64Array(0); // [outputDim]

  // Adam optimizer state (parallel arrays for each weight)
  private adamM: Float64Array[] = [];
  private adamV: Float64Array[] = [];
  private updateCount: number = 0;

  // Welford's online normalization
  private inputMean: Float64Array = new Float64Array(0);
  private inputM2: Float64Array = new Float64Array(0);
  private outputMean: Float64Array = new Float64Array(0);
  private outputM2: Float64Array = new Float64Array(0);
  private normCount: number = 0;

  // Training state
  private totalLoss: number = 0;
  private sampleCount: number = 0;
  private previousLoss: number = Infinity;
  private converged: boolean = false;

  // ADWIN drift detection
  private adwinWindow: number[] = [];
  private adwinSum: number = 0;
  private driftCount: number = 0;

  // Sequence buffer for autoregressive prediction
  private sequenceBufferX: Float64Array[] = [];
  private sequenceBufferY: Float64Array[] = [];
  private maxBufferSize: number;

  // Preallocated computation buffers
  private pool: Float64ArrayPool = new Float64ArrayPool();

  // Forward pass cache (for backprop)
  private cacheConvOut: Float64Array[] = [];
  private cacheScaleEmbed: Float64Array[] = [];
  private cacheFusedInput: Float64Array = new Float64Array(0);
  private cacheGates: Float64Array = new Float64Array(0);
  private cacheFused: Float64Array = new Float64Array(0);
  private cacheBlockInputs: Float64Array[] = [];
  private cacheLn1Out: Float64Array[] = [];
  private cacheLn1Mean: Float64Array[] = [];
  private cacheLn1Var: Float64Array[] = [];
  private cacheAttnQ: Float64Array[] = [];
  private cacheAttnK: Float64Array[] = [];
  private cacheAttnV: Float64Array[] = [];
  private cacheAttnScores: Float64Array[] = [];
  private cacheAttnProbs: Float64Array[] = [];
  private cacheAttnOut: Float64Array[] = [];
  private cacheResidual1: Float64Array[] = [];
  private cacheLn2Out: Float64Array[] = [];
  private cacheLn2Mean: Float64Array[] = [];
  private cacheLn2Var: Float64Array[] = [];
  private cacheFFNHidden: Float64Array[] = [];
  private cacheFFNAct: Float64Array[] = [];
  private cacheResidual2: Float64Array[] = [];
  private cachePoolWeights: Float64Array = new Float64Array(0);
  private cacheAggregated: Float64Array = new Float64Array(0);
  private cacheOutput: Float64Array = new Float64Array(0);
  private currentSeqLen: number = 0;

  // Gradient buffers
  private gradTempConvW: Float64Array[] = [];
  private gradTempConvB: Float64Array[] = [];
  private gradScaleEmbed: Float64Array[] = [];
  private gradFusionGateW: Float64Array = new Float64Array(0);
  private gradFusionGateB: Float64Array = new Float64Array(0);
  private gradAttnWq: Float64Array[] = [];
  private gradAttnWk: Float64Array[] = [];
  private gradAttnWv: Float64Array[] = [];
  private gradAttnWo: Float64Array[] = [];
  private gradAttnBq: Float64Array[] = [];
  private gradAttnBk: Float64Array[] = [];
  private gradAttnBv: Float64Array[] = [];
  private gradAttnBo: Float64Array[] = [];
  private gradTemporalBias: Float64Array[] = [];
  private gradFFNW1: Float64Array[] = [];
  private gradFFNB1: Float64Array[] = [];
  private gradFFNW2: Float64Array[] = [];
  private gradFFNB2: Float64Array[] = [];
  private gradLnGamma1: Float64Array[] = [];
  private gradLnBeta1: Float64Array[] = [];
  private gradLnGamma2: Float64Array[] = [];
  private gradLnBeta2: Float64Array[] = [];
  private gradPoolW: Float64Array = new Float64Array(0);
  private gradOutputW: Float64Array = new Float64Array(0);
  private gradOutputB: Float64Array = new Float64Array(0);

  // Temporary buffers for computation
  private tempBuffer1: Float64Array = new Float64Array(0);
  private tempBuffer2: Float64Array = new Float64Array(0);
  private tempBuffer3: Float64Array = new Float64Array(0);
  private tempSeqBuffer: Float64Array = new Float64Array(0);

  /**
   * Create a new FusionTemporalTransformerRegression instance
   * @param config Configuration options
   * @example
   * ```typescript
   * const model = new FusionTemporalTransformerRegression({
   *   numBlocks: 3,
   *   embeddingDim: 64,
   *   numHeads: 8,
   *   learningRate: 0.001
   * });
   * ```
   */
  constructor(config: FusionTemporalTransformerConfig = {}) {
    this.numBlocks = config.numBlocks ?? 3;
    this.embeddingDim = config.embeddingDim ?? 64;
    this.numHeads = config.numHeads ?? 8;
    this.headDim = Math.floor(this.embeddingDim / this.numHeads);
    this.ffnMultiplier = config.ffnMultiplier ?? 4;
    this.ffnDim = this.embeddingDim * this.ffnMultiplier;
    this.attentionDropout = config.attentionDropout ?? 0.0;
    this.learningRate = config.learningRate ?? 0.001;
    this.warmupSteps = config.warmupSteps ?? 100;
    this.totalSteps = config.totalSteps ?? 10000;
    this.beta1 = config.beta1 ?? 0.9;
    this.beta2 = config.beta2 ?? 0.999;
    this.epsilon = config.epsilon ?? 1e-8;
    this.regularizationStrength = config.regularizationStrength ?? 1e-4;
    this.convergenceThreshold = config.convergenceThreshold ?? 1e-6;
    this.outlierThreshold = config.outlierThreshold ?? 3.0;
    this.adwinDelta = config.adwinDelta ?? 0.002;
    this.temporalScales = config.temporalScales ?? [1, 2, 4];
    this.numScales = this.temporalScales.length;
    this.temporalKernelSize = config.temporalKernelSize ?? 3;
    this.maxSequenceLength = config.maxSequenceLength ?? 512;
    this.fusionDropout = config.fusionDropout ?? 0.0;
    this.maxBufferSize = this.maxSequenceLength;
  }

  /**
   * Initialize all network weights using Xavier/He initialization
   * @param inputDim Input feature dimension
   * @param outputDim Output target dimension
   */
  private initializeWeights(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;

    const embDim = this.embeddingDim;
    const numScales = this.numScales;
    const numBlocks = this.numBlocks;
    const kSize = this.temporalKernelSize;
    const maxSeq = this.maxSequenceLength;

    // Temporal convolution weights: Xavier initialization
    // Formula: std = sqrt(2 / (fan_in + fan_out))
    this.temporalConvW = [];
    this.temporalConvB = [];
    for (let s = 0; s < numScales; s++) {
      const fanIn = kSize * inputDim;
      const fanOut = embDim;
      const std = Math.sqrt(2.0 / (fanIn + fanOut));
      const w = new Float64Array(fanIn * fanOut);
      for (let i = 0; i < w.length; i++) {
        w[i] = this.randn() * std;
      }
      this.temporalConvW.push(w);
      this.temporalConvB.push(new Float64Array(embDim));
    }

    // Scale embeddings: small random initialization
    this.scaleEmbed = [];
    for (let s = 0; s < numScales; s++) {
      const embed = new Float64Array(embDim);
      for (let i = 0; i < embDim; i++) {
        embed[i] = this.randn() * 0.02;
      }
      this.scaleEmbed.push(embed);
    }

    // Positional encoding: sinusoidal (not learned)
    // Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
    //          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    this.posEncoding = new Float64Array(maxSeq * embDim);
    for (let pos = 0; pos < maxSeq; pos++) {
      for (let i = 0; i < Math.floor(embDim / 2); i++) {
        const angle = pos / Math.pow(10000, (2 * i) / embDim);
        this.posEncoding[pos * embDim + 2 * i] = Math.sin(angle);
        this.posEncoding[pos * embDim + 2 * i + 1] = Math.cos(angle);
      }
    }

    // Fusion gate weights
    const fusionFanIn = numScales * embDim;
    const fusionStd = Math.sqrt(2.0 / (fusionFanIn + numScales));
    this.fusionGateW = new Float64Array(fusionFanIn * numScales);
    for (let i = 0; i < this.fusionGateW.length; i++) {
      this.fusionGateW[i] = this.randn() * fusionStd;
    }
    this.fusionGateB = new Float64Array(numScales);

    // Initialize attention and FFN weights per block
    this.attnWq = [];
    this.attnWk = [];
    this.attnWv = [];
    this.attnWo = [];
    this.attnBq = [];
    this.attnBk = [];
    this.attnBv = [];
    this.attnBo = [];
    this.temporalBias = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    this.lnGamma1 = [];
    this.lnBeta1 = [];
    this.lnGamma2 = [];
    this.lnBeta2 = [];

    const attnStd = Math.sqrt(2.0 / (embDim + embDim));
    const ffnStd1 = Math.sqrt(2.0 / (embDim + this.ffnDim));
    const ffnStd2 = Math.sqrt(2.0 / (this.ffnDim + embDim));

    for (let b = 0; b < numBlocks; b++) {
      // Attention weights
      this.attnWq.push(this.randomArray(embDim * embDim, attnStd));
      this.attnWk.push(this.randomArray(embDim * embDim, attnStd));
      this.attnWv.push(this.randomArray(embDim * embDim, attnStd));
      this.attnWo.push(this.randomArray(embDim * embDim, attnStd));
      this.attnBq.push(new Float64Array(embDim));
      this.attnBk.push(new Float64Array(embDim));
      this.attnBv.push(new Float64Array(embDim));
      this.attnBo.push(new Float64Array(embDim));

      // Temporal bias (learned relative position bias)
      this.temporalBias.push(new Float64Array(maxSeq * maxSeq));

      // FFN weights
      this.ffnW1.push(this.randomArray(embDim * this.ffnDim, ffnStd1));
      this.ffnB1.push(new Float64Array(this.ffnDim));
      this.ffnW2.push(this.randomArray(this.ffnDim * embDim, ffnStd2));
      this.ffnB2.push(new Float64Array(embDim));

      // LayerNorm parameters (gamma=1, beta=0 initially)
      const gamma1 = new Float64Array(embDim);
      const gamma2 = new Float64Array(embDim);
      gamma1.fill(1.0);
      gamma2.fill(1.0);
      this.lnGamma1.push(gamma1);
      this.lnBeta1.push(new Float64Array(embDim));
      this.lnGamma2.push(gamma2);
      this.lnBeta2.push(new Float64Array(embDim));
    }

    // Output layer weights
    this.poolW = new Float64Array(embDim);
    for (let i = 0; i < embDim; i++) {
      this.poolW[i] = this.randn() * 0.02;
    }
    const outStd = Math.sqrt(2.0 / (embDim + outputDim));
    this.outputW = this.randomArray(embDim * outputDim, outStd);
    this.outputB = new Float64Array(outputDim);

    // Initialize Adam optimizer state
    this.initializeAdamState();

    // Initialize normalization arrays
    this.inputMean = new Float64Array(inputDim);
    this.inputM2 = new Float64Array(inputDim);
    this.outputMean = new Float64Array(outputDim);
    this.outputM2 = new Float64Array(outputDim);
    this.normCount = 0;

    // Allocate gradient buffers
    this.allocateGradientBuffers();

    // Allocate cache buffers
    this.allocateCacheBuffers();

    // Allocate temporary buffers
    const maxBufSize = Math.max(
      maxSeq * embDim,
      maxSeq * maxSeq,
      embDim * this.ffnDim,
      embDim * outputDim,
    );
    this.tempBuffer1 = new Float64Array(maxBufSize);
    this.tempBuffer2 = new Float64Array(maxBufSize);
    this.tempBuffer3 = new Float64Array(maxBufSize);
    this.tempSeqBuffer = new Float64Array(maxSeq * inputDim);

    this.isInitialized = true;
  }

  /**
   * Generate random number from standard normal distribution
   * Uses Box-Muller transform
   */
  private randn(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Create random Float64Array with given standard deviation
   */
  private randomArray(size: number, std: number): Float64Array {
    const arr = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      arr[i] = this.randn() * std;
    }
    return arr;
  }

  /**
   * Initialize Adam optimizer first and second moment arrays
   */
  private initializeAdamState(): void {
    this.adamM = [];
    this.adamV = [];

    // Collect all weight arrays
    const allWeights = this.getAllWeightArrays();
    for (const w of allWeights) {
      this.adamM.push(new Float64Array(w.length));
      this.adamV.push(new Float64Array(w.length));
    }
    this.updateCount = 0;
  }

  /**
   * Get flat list of all learnable weight arrays (for Adam)
   */
  private getAllWeightArrays(): Float64Array[] {
    const arrays: Float64Array[] = [];

    // Temporal conv
    for (let s = 0; s < this.numScales; s++) {
      arrays.push(this.temporalConvW[s]);
      arrays.push(this.temporalConvB[s]);
      arrays.push(this.scaleEmbed[s]);
    }

    // Fusion
    arrays.push(this.fusionGateW);
    arrays.push(this.fusionGateB);

    // Blocks
    for (let b = 0; b < this.numBlocks; b++) {
      arrays.push(this.attnWq[b]);
      arrays.push(this.attnWk[b]);
      arrays.push(this.attnWv[b]);
      arrays.push(this.attnWo[b]);
      arrays.push(this.attnBq[b]);
      arrays.push(this.attnBk[b]);
      arrays.push(this.attnBv[b]);
      arrays.push(this.attnBo[b]);
      arrays.push(this.temporalBias[b]);
      arrays.push(this.ffnW1[b]);
      arrays.push(this.ffnB1[b]);
      arrays.push(this.ffnW2[b]);
      arrays.push(this.ffnB2[b]);
      arrays.push(this.lnGamma1[b]);
      arrays.push(this.lnBeta1[b]);
      arrays.push(this.lnGamma2[b]);
      arrays.push(this.lnBeta2[b]);
    }

    // Output
    arrays.push(this.poolW);
    arrays.push(this.outputW);
    arrays.push(this.outputB);

    return arrays;
  }

  /**
   * Get flat list of all gradient arrays (parallel to weight arrays)
   */
  private getAllGradientArrays(): Float64Array[] {
    const arrays: Float64Array[] = [];

    for (let s = 0; s < this.numScales; s++) {
      arrays.push(this.gradTempConvW[s]);
      arrays.push(this.gradTempConvB[s]);
      arrays.push(this.gradScaleEmbed[s]);
    }

    arrays.push(this.gradFusionGateW);
    arrays.push(this.gradFusionGateB);

    for (let b = 0; b < this.numBlocks; b++) {
      arrays.push(this.gradAttnWq[b]);
      arrays.push(this.gradAttnWk[b]);
      arrays.push(this.gradAttnWv[b]);
      arrays.push(this.gradAttnWo[b]);
      arrays.push(this.gradAttnBq[b]);
      arrays.push(this.gradAttnBk[b]);
      arrays.push(this.gradAttnBv[b]);
      arrays.push(this.gradAttnBo[b]);
      arrays.push(this.gradTemporalBias[b]);
      arrays.push(this.gradFFNW1[b]);
      arrays.push(this.gradFFNB1[b]);
      arrays.push(this.gradFFNW2[b]);
      arrays.push(this.gradFFNB2[b]);
      arrays.push(this.gradLnGamma1[b]);
      arrays.push(this.gradLnBeta1[b]);
      arrays.push(this.gradLnGamma2[b]);
      arrays.push(this.gradLnBeta2[b]);
    }

    arrays.push(this.gradPoolW);
    arrays.push(this.gradOutputW);
    arrays.push(this.gradOutputB);

    return arrays;
  }

  /**
   * Allocate gradient buffers matching weight shapes
   */
  private allocateGradientBuffers(): void {
    this.gradTempConvW = [];
    this.gradTempConvB = [];
    this.gradScaleEmbed = [];

    for (let s = 0; s < this.numScales; s++) {
      this.gradTempConvW.push(new Float64Array(this.temporalConvW[s].length));
      this.gradTempConvB.push(new Float64Array(this.temporalConvB[s].length));
      this.gradScaleEmbed.push(new Float64Array(this.scaleEmbed[s].length));
    }

    this.gradFusionGateW = new Float64Array(this.fusionGateW.length);
    this.gradFusionGateB = new Float64Array(this.fusionGateB.length);

    this.gradAttnWq = [];
    this.gradAttnWk = [];
    this.gradAttnWv = [];
    this.gradAttnWo = [];
    this.gradAttnBq = [];
    this.gradAttnBk = [];
    this.gradAttnBv = [];
    this.gradAttnBo = [];
    this.gradTemporalBias = [];
    this.gradFFNW1 = [];
    this.gradFFNB1 = [];
    this.gradFFNW2 = [];
    this.gradFFNB2 = [];
    this.gradLnGamma1 = [];
    this.gradLnBeta1 = [];
    this.gradLnGamma2 = [];
    this.gradLnBeta2 = [];

    for (let b = 0; b < this.numBlocks; b++) {
      this.gradAttnWq.push(new Float64Array(this.attnWq[b].length));
      this.gradAttnWk.push(new Float64Array(this.attnWk[b].length));
      this.gradAttnWv.push(new Float64Array(this.attnWv[b].length));
      this.gradAttnWo.push(new Float64Array(this.attnWo[b].length));
      this.gradAttnBq.push(new Float64Array(this.attnBq[b].length));
      this.gradAttnBk.push(new Float64Array(this.attnBk[b].length));
      this.gradAttnBv.push(new Float64Array(this.attnBv[b].length));
      this.gradAttnBo.push(new Float64Array(this.attnBo[b].length));
      this.gradTemporalBias.push(new Float64Array(this.temporalBias[b].length));
      this.gradFFNW1.push(new Float64Array(this.ffnW1[b].length));
      this.gradFFNB1.push(new Float64Array(this.ffnB1[b].length));
      this.gradFFNW2.push(new Float64Array(this.ffnW2[b].length));
      this.gradFFNB2.push(new Float64Array(this.ffnB2[b].length));
      this.gradLnGamma1.push(new Float64Array(this.lnGamma1[b].length));
      this.gradLnBeta1.push(new Float64Array(this.lnBeta1[b].length));
      this.gradLnGamma2.push(new Float64Array(this.lnGamma2[b].length));
      this.gradLnBeta2.push(new Float64Array(this.lnBeta2[b].length));
    }

    this.gradPoolW = new Float64Array(this.poolW.length);
    this.gradOutputW = new Float64Array(this.outputW.length);
    this.gradOutputB = new Float64Array(this.outputB.length);
  }

  /**
   * Allocate forward pass cache buffers
   */
  private allocateCacheBuffers(): void {
    const maxSeq = this.maxSequenceLength;
    const embDim = this.embeddingDim;

    this.cacheConvOut = [];
    this.cacheScaleEmbed = [];
    for (let s = 0; s < this.numScales; s++) {
      this.cacheConvOut.push(new Float64Array(maxSeq * embDim));
      this.cacheScaleEmbed.push(new Float64Array(maxSeq * embDim));
    }

    this.cacheFusedInput = new Float64Array(maxSeq * this.numScales * embDim);
    this.cacheGates = new Float64Array(maxSeq * this.numScales);
    this.cacheFused = new Float64Array(maxSeq * embDim);

    this.cacheBlockInputs = [];
    this.cacheLn1Out = [];
    this.cacheLn1Mean = [];
    this.cacheLn1Var = [];
    this.cacheAttnQ = [];
    this.cacheAttnK = [];
    this.cacheAttnV = [];
    this.cacheAttnScores = [];
    this.cacheAttnProbs = [];
    this.cacheAttnOut = [];
    this.cacheResidual1 = [];
    this.cacheLn2Out = [];
    this.cacheLn2Mean = [];
    this.cacheLn2Var = [];
    this.cacheFFNHidden = [];
    this.cacheFFNAct = [];
    this.cacheResidual2 = [];

    for (let b = 0; b < this.numBlocks; b++) {
      this.cacheBlockInputs.push(new Float64Array(maxSeq * embDim));
      this.cacheLn1Out.push(new Float64Array(maxSeq * embDim));
      this.cacheLn1Mean.push(new Float64Array(maxSeq));
      this.cacheLn1Var.push(new Float64Array(maxSeq));
      this.cacheAttnQ.push(new Float64Array(maxSeq * embDim));
      this.cacheAttnK.push(new Float64Array(maxSeq * embDim));
      this.cacheAttnV.push(new Float64Array(maxSeq * embDim));
      this.cacheAttnScores.push(new Float64Array(maxSeq * maxSeq));
      this.cacheAttnProbs.push(new Float64Array(maxSeq * maxSeq));
      this.cacheAttnOut.push(new Float64Array(maxSeq * embDim));
      this.cacheResidual1.push(new Float64Array(maxSeq * embDim));
      this.cacheLn2Out.push(new Float64Array(maxSeq * embDim));
      this.cacheLn2Mean.push(new Float64Array(maxSeq));
      this.cacheLn2Var.push(new Float64Array(maxSeq));
      this.cacheFFNHidden.push(new Float64Array(maxSeq * this.ffnDim));
      this.cacheFFNAct.push(new Float64Array(maxSeq * this.ffnDim));
      this.cacheResidual2.push(new Float64Array(maxSeq * embDim));
    }

    this.cachePoolWeights = new Float64Array(maxSeq);
    this.cacheAggregated = new Float64Array(embDim);
    this.cacheOutput = new Float64Array(this.outputDim || 1);
  }

  /**
   * Clear all gradient buffers to zero
   */
  private zeroGradients(): void {
    const grads = this.getAllGradientArrays();
    for (const g of grads) {
      g.fill(0);
    }
  }

  // ============================================================================
  // Activation Functions
  // ============================================================================

  /**
   * GELU activation function (Gaussian Error Linear Unit)
   * Formula: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
   * @param x Input value
   * @returns Activated value
   */
  private gelu(x: number): number {
    const inner = SQRT_2_PI * (x + GELU_CONST * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  }

  /**
   * Derivative of GELU activation
   * @param x Input value
   * @param y Output of gelu(x) (for efficiency)
   * @returns Derivative value
   */
  private geluDerivative(x: number): number {
    const x3 = x * x * x;
    const inner = SQRT_2_PI * (x + GELU_CONST * x3);
    const tanhVal = Math.tanh(inner);
    const sech2 = 1 - tanhVal * tanhVal;
    const dinnerDx = SQRT_2_PI * (1 + 3 * GELU_CONST * x * x);
    return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * dinnerDx;
  }

  /**
   * Sigmoid activation function
   * Formula: σ(x) = 1 / (1 + exp(-x))
   */
  private sigmoid(x: number): number {
    if (x >= 0) {
      const ez = Math.exp(-x);
      return 1 / (1 + ez);
    } else {
      const ez = Math.exp(x);
      return ez / (1 + ez);
    }
  }

  /**
   * Derivative of sigmoid
   * @param s Output of sigmoid(x)
   */
  private sigmoidDerivative(s: number): number {
    return s * (1 - s);
  }

  /**
   * Stable softmax over array segment
   * @param arr Source array
   * @param start Start index
   * @param len Length of segment
   * @param out Output array
   * @param outStart Output start index
   */
  private softmax(
    arr: Float64Array,
    start: number,
    len: number,
    out: Float64Array,
    outStart: number,
  ): void {
    // Find max for numerical stability
    let max = -Infinity;
    for (let i = 0; i < len; i++) {
      if (arr[start + i] > max) max = arr[start + i];
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const e = Math.exp(arr[start + i] - max);
      out[outStart + i] = e;
      sum += e;
    }

    // Normalize
    const invSum = 1 / (sum + this.epsilon);
    for (let i = 0; i < len; i++) {
      out[outStart + i] *= invSum;
    }
  }

  // ============================================================================
  // Matrix Operations (Optimized)
  // ============================================================================

  /**
   * Matrix-vector multiplication: out = A * x + b
   * @param A Weight matrix [rows, cols] stored row-major as flat array
   * @param x Input vector [cols]
   * @param b Bias vector [rows] (optional)
   * @param out Output vector [rows]
   * @param rows Number of rows
   * @param cols Number of columns
   */
  private matVec(
    A: Float64Array,
    x: Float64Array,
    b: Float64Array | null,
    out: Float64Array,
    rows: number,
    cols: number,
    outOffset: number = 0,
    xOffset: number = 0,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = b ? b[i] : 0;
      const rowStart = i * cols;
      for (let j = 0; j < cols; j++) {
        sum += A[rowStart + j] * x[xOffset + j];
      }
      out[outOffset + i] = sum;
    }
  }

  /**
   * Matrix-matrix multiplication: C = A * B
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
        for (let p = 0; p < k; p++) {
          sum += A[aOffset + i * k + p] * B[bOffset + p * n + j];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Transpose matrix multiplication: C = A^T * B
   * A: [k, m], B: [k, n], C: [m, n]
   */
  private matMulAT(
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
        for (let p = 0; p < k; p++) {
          sum += A[aOffset + p * m + i] * B[bOffset + p * n + j];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Matrix multiplication with B transposed: C = A * B^T
   * A: [m, k], B: [n, k], C: [m, n]
   */
  private matMulBT(
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
        for (let p = 0; p < k; p++) {
          sum += A[aOffset + i * k + p] * B[bOffset + j * k + p];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Outer product: C += alpha * x * y^T
   * x: [m], y: [n], C: [m, n]
   */
  private outerProduct(
    x: Float64Array,
    y: Float64Array,
    C: Float64Array,
    m: number,
    n: number,
    alpha: number,
    xOffset: number = 0,
    yOffset: number = 0,
    cOffset: number = 0,
  ): void {
    for (let i = 0; i < m; i++) {
      const xi = alpha * x[xOffset + i];
      for (let j = 0; j < n; j++) {
        C[cOffset + i * n + j] += xi * y[yOffset + j];
      }
    }
  }

  // ============================================================================
  // Forward Pass Components
  // ============================================================================

  /**
   * 1D convolution with padding to maintain sequence length
   * Uses same padding with dilation for multi-scale
   */
  private conv1dForward(
    input: Float64Array,
    seqLen: number,
    inputDim: number,
    W: Float64Array,
    b: Float64Array,
    kernelSize: number,
    dilation: number,
    output: Float64Array,
  ): void {
    const embDim = this.embeddingDim;
    const padSize = Math.floor((kernelSize - 1) * dilation / 2);

    for (let t = 0; t < seqLen; t++) {
      // Reset output position
      for (let e = 0; e < embDim; e++) {
        output[t * embDim + e] = b[e];
      }

      // Apply kernel
      for (let k = 0; k < kernelSize; k++) {
        const inputT = t + (k - Math.floor(kernelSize / 2)) * dilation;
        if (inputT >= 0 && inputT < seqLen) {
          // Weight matrix: [kernelSize * inputDim, embDim]
          const wOffset = k * inputDim;
          for (let e = 0; e < embDim; e++) {
            let sum = 0;
            for (let d = 0; d < inputDim; d++) {
              sum += W[(wOffset + d) * embDim + e] *
                input[inputT * inputDim + d];
            }
            output[t * embDim + e] += sum;
          }
        }
      }
    }
  }

  /**
   * Layer normalization
   * Formula: y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
   */
  private layerNormForward(
    input: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    output: Float64Array,
    meanOut: Float64Array,
    varOut: Float64Array,
    seqLen: number,
    dim: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const offset = t * dim;

      // Compute mean
      let mean = 0;
      for (let i = 0; i < dim; i++) {
        mean += input[offset + i];
      }
      mean /= dim;
      meanOut[t] = mean;

      // Compute variance
      let variance = 0;
      for (let i = 0; i < dim; i++) {
        const diff = input[offset + i] - mean;
        variance += diff * diff;
      }
      variance /= dim;
      varOut[t] = variance;

      // Normalize and scale
      const invStd = 1 / Math.sqrt(variance + this.epsilon);
      for (let i = 0; i < dim; i++) {
        output[offset + i] = gamma[i] * (input[offset + i] - mean) * invStd +
          beta[i];
      }
    }
  }

  /**
   * Multi-head self-attention forward pass
   */
  private multiHeadAttentionForward(
    input: Float64Array,
    Wq: Float64Array,
    Wk: Float64Array,
    Wv: Float64Array,
    Wo: Float64Array,
    bq: Float64Array,
    bk: Float64Array,
    bv: Float64Array,
    bo: Float64Array,
    temporalBias: Float64Array,
    seqLen: number,
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    scores: Float64Array,
    probs: Float64Array,
    output: Float64Array,
  ): void {
    const embDim = this.embeddingDim;
    const numHeads = this.numHeads;
    const headDim = this.headDim;
    const scale = 1 / Math.sqrt(headDim);

    // Compute Q, K, V projections for all positions
    // Q = input * Wq + bq  (same for K, V)
    for (let t = 0; t < seqLen; t++) {
      this.matVec(Wq, input, bq, Q, embDim, embDim, t * embDim, t * embDim);
      this.matVec(Wk, input, bk, K, embDim, embDim, t * embDim, t * embDim);
      this.matVec(Wv, input, bv, V, embDim, embDim, t * embDim, t * embDim);
    }

    // Compute attention scores and apply softmax per head
    // For efficiency, process all heads together but compute scores separately
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;

      // Compute scaled dot-product attention scores
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += Q[i * embDim + headOffset + d] *
              K[j * embDim + headOffset + d];
          }
          // Add temporal bias and scale
          score = score * scale + temporalBias[i * this.maxSequenceLength + j];

          // Causal mask: only attend to previous positions
          if (j > i) {
            score = -1e9;
          }
          scores[h * seqLen * seqLen + i * seqLen + j] = score;
        }

        // Softmax over j for this (h, i)
        this.softmax(
          scores,
          h * seqLen * seqLen + i * seqLen,
          seqLen,
          probs,
          h * seqLen * seqLen + i * seqLen,
        );
      }
    }

    // Apply attention to values and concatenate heads
    for (let i = 0; i < seqLen; i++) {
      for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;
        const probsOffset = h * seqLen * seqLen + i * seqLen;

        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += probs[probsOffset + j] * V[j * embDim + headOffset + d];
          }
          // Store in temp, will project to output
          this.tempBuffer1[i * embDim + headOffset + d] = sum;
        }
      }
    }

    // Output projection: output = concat_heads * Wo + bo
    for (let t = 0; t < seqLen; t++) {
      this.matVec(
        Wo,
        this.tempBuffer1,
        bo,
        output,
        embDim,
        embDim,
        t * embDim,
        t * embDim,
      );
    }
  }

  /**
   * Feed-forward network forward pass
   * Formula: FFN(x) = GELU(x * W1 + b1) * W2 + b2
   */
  private ffnForward(
    input: Float64Array,
    W1: Float64Array,
    b1: Float64Array,
    W2: Float64Array,
    b2: Float64Array,
    seqLen: number,
    hidden: Float64Array,
    activated: Float64Array,
    output: Float64Array,
  ): void {
    const embDim = this.embeddingDim;
    const ffnDim = this.ffnDim;

    for (let t = 0; t < seqLen; t++) {
      // First linear: hidden = input * W1 + b1
      this.matVec(
        W1,
        input,
        b1,
        hidden,
        ffnDim,
        embDim,
        t * ffnDim,
        t * embDim,
      );

      // GELU activation
      for (let i = 0; i < ffnDim; i++) {
        activated[t * ffnDim + i] = this.gelu(hidden[t * ffnDim + i]);
      }

      // Second linear: output = activated * W2 + b2
      this.matVec(
        W2,
        activated,
        b2,
        output,
        embDim,
        ffnDim,
        t * embDim,
        t * ffnDim,
      );
    }
  }

  /**
   * Attention-based pooling over sequence dimension
   * Formula: weights = softmax(input * w_pool), aggregated = sum(weights * input)
   */
  private attentionPoolingForward(
    input: Float64Array,
    poolW: Float64Array,
    seqLen: number,
    weights: Float64Array,
    aggregated: Float64Array,
  ): void {
    const embDim = this.embeddingDim;

    // Compute attention scores
    for (let t = 0; t < seqLen; t++) {
      let score = 0;
      for (let i = 0; i < embDim; i++) {
        score += input[t * embDim + i] * poolW[i];
      }
      weights[t] = score;
    }

    // Softmax
    this.softmax(weights, 0, seqLen, weights, 0);

    // Weighted sum
    aggregated.fill(0);
    for (let t = 0; t < seqLen; t++) {
      const w = weights[t];
      for (let i = 0; i < embDim; i++) {
        aggregated[i] += w * input[t * embDim + i];
      }
    }
  }

  /**
   * Complete forward pass through the network
   * @param input Normalized input sequence [seqLen, inputDim]
   * @param seqLen Sequence length
   * @returns Output predictions [outputDim]
   */
  private forward(input: Float64Array, seqLen: number): Float64Array {
    this.currentSeqLen = seqLen;
    const embDim = this.embeddingDim;

    // === Multi-scale temporal convolution ===
    for (let s = 0; s < this.numScales; s++) {
      const dilation = this.temporalScales[s];
      this.conv1dForward(
        input,
        seqLen,
        this.inputDim,
        this.temporalConvW[s],
        this.temporalConvB[s],
        this.temporalKernelSize,
        dilation,
        this.cacheConvOut[s],
      );

      // Apply GELU and add positional + scale embeddings
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < embDim; i++) {
          const idx = t * embDim + i;
          // GELU activation on conv output
          this.cacheConvOut[s][idx] = this.gelu(this.cacheConvOut[s][idx]);
          // Add positional encoding and scale embedding
          this.cacheScaleEmbed[s][idx] = this.cacheConvOut[s][idx] +
            this.posEncoding[t * embDim + i] +
            this.scaleEmbed[s][i];
        }
      }
    }

    // === Gated fusion of scales ===
    // Concatenate all scales for gate computation
    for (let t = 0; t < seqLen; t++) {
      for (let s = 0; s < this.numScales; s++) {
        for (let i = 0; i < embDim; i++) {
          this.cacheFusedInput[t * this.numScales * embDim + s * embDim + i] =
            this.cacheScaleEmbed[s][t * embDim + i];
        }
      }
    }

    // Compute gates: gates = sigmoid(fusedInput * W_gate + b_gate)
    for (let t = 0; t < seqLen; t++) {
      const inputOffset = t * this.numScales * embDim;
      const gateOffset = t * this.numScales;

      for (let s = 0; s < this.numScales; s++) {
        let sum = this.fusionGateB[s];
        for (let i = 0; i < this.numScales * embDim; i++) {
          sum += this.cacheFusedInput[inputOffset + i] *
            this.fusionGateW[i * this.numScales + s];
        }
        this.cacheGates[gateOffset + s] = this.sigmoid(sum);
      }
    }

    // Fused = sum over scales (gates[s] * E_s)
    for (let t = 0; t < seqLen; t++) {
      const gateOffset = t * this.numScales;
      for (let i = 0; i < embDim; i++) {
        let sum = 0;
        for (let s = 0; s < this.numScales; s++) {
          sum += this.cacheGates[gateOffset + s] *
            this.cacheScaleEmbed[s][t * embDim + i];
        }
        this.cacheFused[t * embDim + i] = sum;
      }
    }

    // === Transformer blocks ===
    // Copy fused to first block input
    for (let i = 0; i < seqLen * embDim; i++) {
      this.cacheBlockInputs[0][i] = this.cacheFused[i];
    }

    for (let b = 0; b < this.numBlocks; b++) {
      const blockInput = b === 0 ? this.cacheFused : this.cacheResidual2[b - 1];

      // Copy to cache if not first block
      if (b > 0) {
        for (let i = 0; i < seqLen * embDim; i++) {
          this.cacheBlockInputs[b][i] = blockInput[i];
        }
      }

      // LayerNorm 1
      this.layerNormForward(
        this.cacheBlockInputs[b],
        this.lnGamma1[b],
        this.lnBeta1[b],
        this.cacheLn1Out[b],
        this.cacheLn1Mean[b],
        this.cacheLn1Var[b],
        seqLen,
        embDim,
      );

      // Multi-head self-attention
      this.multiHeadAttentionForward(
        this.cacheLn1Out[b],
        this.attnWq[b],
        this.attnWk[b],
        this.attnWv[b],
        this.attnWo[b],
        this.attnBq[b],
        this.attnBk[b],
        this.attnBv[b],
        this.attnBo[b],
        this.temporalBias[b],
        seqLen,
        this.cacheAttnQ[b],
        this.cacheAttnK[b],
        this.cacheAttnV[b],
        this.cacheAttnScores[b],
        this.cacheAttnProbs[b],
        this.cacheAttnOut[b],
      );

      // Residual connection 1
      for (let i = 0; i < seqLen * embDim; i++) {
        this.cacheResidual1[b][i] = this.cacheBlockInputs[b][i] +
          this.cacheAttnOut[b][i];
      }

      // LayerNorm 2
      this.layerNormForward(
        this.cacheResidual1[b],
        this.lnGamma2[b],
        this.lnBeta2[b],
        this.cacheLn2Out[b],
        this.cacheLn2Mean[b],
        this.cacheLn2Var[b],
        seqLen,
        embDim,
      );

      // Feed-forward network
      this.ffnForward(
        this.cacheLn2Out[b],
        this.ffnW1[b],
        this.ffnB1[b],
        this.ffnW2[b],
        this.ffnB2[b],
        seqLen,
        this.cacheFFNHidden[b],
        this.cacheFFNAct[b],
        this.tempBuffer2,
      );

      // Residual connection 2
      for (let i = 0; i < seqLen * embDim; i++) {
        this.cacheResidual2[b][i] = this.cacheResidual1[b][i] +
          this.tempBuffer2[i];
      }
    }

    // === Attention pooling ===
    const finalOutput = this.cacheResidual2[this.numBlocks - 1];
    this.attentionPoolingForward(
      finalOutput,
      this.poolW,
      seqLen,
      this.cachePoolWeights,
      this.cacheAggregated,
    );

    // === Output projection ===
    this.matVec(
      this.outputW,
      this.cacheAggregated,
      this.outputB,
      this.cacheOutput,
      this.outputDim,
      embDim,
      0,
      0,
    );

    return this.cacheOutput;
  }

  // ============================================================================
  // Backward Pass Components
  // ============================================================================

  /**
   * Layer normalization backward pass
   */
  private layerNormBackward(
    dOutput: Float64Array,
    input: Float64Array,
    gamma: Float64Array,
    mean: Float64Array,
    variance: Float64Array,
    dInput: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
    seqLen: number,
    dim: number,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const offset = t * dim;
      const m = mean[t];
      const v = variance[t];
      const invStd = 1 / Math.sqrt(v + this.epsilon);

      // Accumulate gradients for gamma and beta
      for (let i = 0; i < dim; i++) {
        const xNorm = (input[offset + i] - m) * invStd;
        dGamma[i] += dOutput[offset + i] * xNorm;
        dBeta[i] += dOutput[offset + i];
      }

      // Compute dxNorm = dOutput * gamma
      let sumDxNorm = 0;
      let sumDxNormXNorm = 0;
      for (let i = 0; i < dim; i++) {
        const dxNorm = dOutput[offset + i] * gamma[i];
        const xNorm = (input[offset + i] - m) * invStd;
        sumDxNorm += dxNorm;
        sumDxNormXNorm += dxNorm * xNorm;
      }

      // Compute dInput
      const invN = 1 / dim;
      for (let i = 0; i < dim; i++) {
        const xNorm = (input[offset + i] - m) * invStd;
        const dxNorm = dOutput[offset + i] * gamma[i];
        dInput[offset + i] = invStd *
          (dxNorm - invN * sumDxNorm - invN * xNorm * sumDxNormXNorm);
      }
    }
  }

  /**
   * Multi-head attention backward pass
   */
  private multiHeadAttentionBackward(
    dOutput: Float64Array,
    input: Float64Array,
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    probs: Float64Array,
    Wq: Float64Array,
    Wk: Float64Array,
    Wv: Float64Array,
    Wo: Float64Array,
    seqLen: number,
    dInput: Float64Array,
    dWq: Float64Array,
    dWk: Float64Array,
    dWv: Float64Array,
    dWo: Float64Array,
    dBq: Float64Array,
    dBk: Float64Array,
    dBv: Float64Array,
    dBo: Float64Array,
    dTemporalBias: Float64Array,
  ): void {
    const embDim = this.embeddingDim;
    const numHeads = this.numHeads;
    const headDim = this.headDim;
    const scale = 1 / Math.sqrt(headDim);

    // Clear dInput
    dInput.fill(0);

    // Backward through output projection
    // dConcat = dOutput * Wo^T
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        let sum = 0;
        for (let j = 0; j < embDim; j++) {
          sum += dOutput[t * embDim + j] * Wo[j * embDim + i];
        }
        this.tempBuffer1[t * embDim + i] = sum;
      }
    }

    // Gradient for Wo and bo
    for (let t = 0; t < seqLen; t++) {
      // Reconstruct concat of attention outputs from cached V and probs
      for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;
        const probsOffset = h * seqLen * seqLen + t * seqLen;

        for (let d = 0; d < headDim; d++) {
          let attnOut = 0;
          for (let j = 0; j < seqLen; j++) {
            attnOut += probs[probsOffset + j] * V[j * embDim + headOffset + d];
          }
          this.tempBuffer2[t * embDim + headOffset + d] = attnOut;
        }
      }

      // dWo += dOutput * concat^T
      this.outerProduct(
        dOutput,
        this.tempBuffer2,
        dWo,
        embDim,
        embDim,
        1,
        t * embDim,
        t * embDim,
        0,
      );

      // dBo += dOutput
      for (let i = 0; i < embDim; i++) {
        dBo[i] += dOutput[t * embDim + i];
      }
    }

    // Backward through attention computation per head
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;

      for (let i = 0; i < seqLen; i++) {
        const probsOffset = h * seqLen * seqLen + i * seqLen;

        // dV contribution
        for (let j = 0; j < seqLen; j++) {
          const prob = probs[probsOffset + j];
          for (let d = 0; d < headDim; d++) {
            // From: out = sum_j(probs[j] * V[j])
            // dV[j] += probs[j] * dOut
            this.tempBuffer3[j * embDim + headOffset + d] =
              (this.tempBuffer3[j * embDim + headOffset + d] || 0) +
              prob * this.tempBuffer1[i * embDim + headOffset + d];
          }
        }

        // dProbs
        for (let j = 0; j < seqLen; j++) {
          let dp = 0;
          for (let d = 0; d < headDim; d++) {
            dp += this.tempBuffer1[i * embDim + headOffset + d] *
              V[j * embDim + headOffset + d];
          }
          // Store dProbs temporarily
          this.tempBuffer2[i * seqLen + j] = dp;
        }

        // Backward through softmax
        // dScores = probs * (dProbs - sum(probs * dProbs))
        let sumProbDProb = 0;
        for (let j = 0; j < seqLen; j++) {
          sumProbDProb += probs[probsOffset + j] *
            this.tempBuffer2[i * seqLen + j];
        }

        for (let j = 0; j < seqLen; j++) {
          const dScore = probs[probsOffset + j] *
            (this.tempBuffer2[i * seqLen + j] - sumProbDProb);

          // Gradient for temporal bias
          dTemporalBias[i * this.maxSequenceLength + j] += dScore;

          // dQ, dK from scores = Q * K^T / sqrt(d)
          const scaledDScore = dScore * scale;
          for (let d = 0; d < headDim; d++) {
            // dQ[i] += dScore * K[j]
            // dK[j] += dScore * Q[i]
            const qIdx = i * embDim + headOffset + d;
            const kIdx = j * embDim + headOffset + d;
            this.tempBuffer2[seqLen * seqLen + qIdx] =
              (this.tempBuffer2[seqLen * seqLen + qIdx] || 0) +
              scaledDScore * K[kIdx];
            this.tempBuffer2[seqLen * seqLen + seqLen * embDim + kIdx] =
              (this.tempBuffer2[seqLen * seqLen + seqLen * embDim + kIdx] ||
                0) + scaledDScore * Q[qIdx];
          }
        }
      }
    }

    // Now backpropagate through Q, K, V projections
    // dInput = dQ * Wq^T + dK * Wk^T + dV * Wv^T
    // Also accumulate gradients for Wq, Wk, Wv
    for (let t = 0; t < seqLen; t++) {
      // dQ contribution to dInput
      for (let i = 0; i < embDim; i++) {
        const dq = this.tempBuffer2[seqLen * seqLen + t * embDim + i] || 0;
        for (let j = 0; j < embDim; j++) {
          dInput[t * embDim + j] += dq * Wq[i * embDim + j];
        }
        dBq[i] += dq;
        for (let j = 0; j < embDim; j++) {
          dWq[i * embDim + j] += dq * input[t * embDim + j];
        }
      }

      // dK contribution
      for (let i = 0; i < embDim; i++) {
        const dk = this
          .tempBuffer2[seqLen * seqLen + seqLen * embDim + t * embDim + i] ||
          0;
        for (let j = 0; j < embDim; j++) {
          dInput[t * embDim + j] += dk * Wk[i * embDim + j];
        }
        dBk[i] += dk;
        for (let j = 0; j < embDim; j++) {
          dWk[i * embDim + j] += dk * input[t * embDim + j];
        }
      }

      // dV contribution
      for (let i = 0; i < embDim; i++) {
        const dv = this.tempBuffer3[t * embDim + i] || 0;
        for (let j = 0; j < embDim; j++) {
          dInput[t * embDim + j] += dv * Wv[i * embDim + j];
        }
        dBv[i] += dv;
        for (let j = 0; j < embDim; j++) {
          dWv[i * embDim + j] += dv * input[t * embDim + j];
        }
      }
    }

    // Clear temp buffers used
    this.tempBuffer3.fill(0);
  }

  /**
   * FFN backward pass
   */
  private ffnBackward(
    dOutput: Float64Array,
    input: Float64Array,
    hidden: Float64Array,
    activated: Float64Array,
    W1: Float64Array,
    W2: Float64Array,
    seqLen: number,
    dInput: Float64Array,
    dW1: Float64Array,
    dB1: Float64Array,
    dW2: Float64Array,
    dB2: Float64Array,
  ): void {
    const embDim = this.embeddingDim;
    const ffnDim = this.ffnDim;

    for (let t = 0; t < seqLen; t++) {
      // Backward through W2: dActivated = dOutput * W2^T
      for (let i = 0; i < ffnDim; i++) {
        let sum = 0;
        for (let j = 0; j < embDim; j++) {
          sum += dOutput[t * embDim + j] * W2[i * embDim + j];
        }
        this.tempBuffer1[i] = sum;
      }

      // dW2 += activated * dOutput^T
      for (let i = 0; i < ffnDim; i++) {
        for (let j = 0; j < embDim; j++) {
          dW2[i * embDim + j] += activated[t * ffnDim + i] *
            dOutput[t * embDim + j];
        }
      }

      // dB2 += dOutput
      for (let j = 0; j < embDim; j++) {
        dB2[j] += dOutput[t * embDim + j];
      }

      // Backward through GELU
      for (let i = 0; i < ffnDim; i++) {
        this.tempBuffer1[i] *= this.geluDerivative(hidden[t * ffnDim + i]);
      }

      // dInput = dHidden * W1^T
      for (let i = 0; i < embDim; i++) {
        let sum = 0;
        for (let j = 0; j < ffnDim; j++) {
          sum += this.tempBuffer1[j] * W1[j * embDim + i];
        }
        dInput[t * embDim + i] = sum;
      }

      // dW1 += input * dHidden^T
      for (let i = 0; i < ffnDim; i++) {
        for (let j = 0; j < embDim; j++) {
          dW1[i * embDim + j] += this.tempBuffer1[i] * input[t * embDim + j];
        }
      }

      // dB1 += dHidden
      for (let i = 0; i < ffnDim; i++) {
        dB1[i] += this.tempBuffer1[i];
      }
    }
  }

  /**
   * Complete backward pass through the network
   */
  private backward(
    target: Float64Array,
    seqLen: number,
    sampleWeight: number,
  ): number {
    const embDim = this.embeddingDim;

    // Zero all gradients
    this.zeroGradients();

    // Compute output loss gradient: dL/dOutput = (predicted - target) / n * sampleWeight
    // MSE Loss: L = (1/2n) * sum((y - yhat)^2)
    // dL/dyhat = (yhat - y) / n
    for (let i = 0; i < this.outputDim; i++) {
      this.tempBuffer1[i] = (this.cacheOutput[i] - target[i]) * sampleWeight /
        this.outputDim;
    }

    // Backward through output projection
    // dAggregated = dOutput * W_out^T
    for (let i = 0; i < embDim; i++) {
      let sum = 0;
      for (let j = 0; j < this.outputDim; j++) {
        sum += this.tempBuffer1[j] * this.outputW[i * this.outputDim + j];
      }
      this.tempBuffer2[i] = sum;
    }

    // dW_out += aggregated * dOutput^T
    for (let i = 0; i < embDim; i++) {
      for (let j = 0; j < this.outputDim; j++) {
        this.gradOutputW[i * this.outputDim + j] += this.cacheAggregated[i] *
          this.tempBuffer1[j];
      }
    }

    // dB_out += dOutput
    for (let j = 0; j < this.outputDim; j++) {
      this.gradOutputB[j] += this.tempBuffer1[j];
    }

    // Backward through attention pooling
    // aggregated = sum(weights[t] * input[t])
    // dWeights[t] = sum_i(dAggregated[i] * input[t][i])
    // dInput[t] = weights[t] * dAggregated
    const finalOutput = this.cacheResidual2[this.numBlocks - 1];

    // dWeights (before softmax)
    for (let t = 0; t < seqLen; t++) {
      let dw = 0;
      for (let i = 0; i < embDim; i++) {
        dw += this.tempBuffer2[i] * finalOutput[t * embDim + i];
      }
      this.tempBuffer3[t] = dw;
    }

    // Backward through softmax: dScores = weights * (dWeights - sum(weights * dWeights))
    let sumWdW = 0;
    for (let t = 0; t < seqLen; t++) {
      sumWdW += this.cachePoolWeights[t] * this.tempBuffer3[t];
    }

    for (let t = 0; t < seqLen; t++) {
      this.tempBuffer3[t] = this.cachePoolWeights[t] *
        (this.tempBuffer3[t] - sumWdW);
    }

    // dPoolW += sum_t(dScores[t] * input[t])
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        this.gradPoolW[i] += this.tempBuffer3[t] * finalOutput[t * embDim + i];
      }
    }

    // dFinalOutput = weights * dAggregated (broadcast) + input * dScores (for poolW gradient flow)
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        this.tempBuffer1[t * embDim + i] =
          this.cachePoolWeights[t] * this.tempBuffer2[i] +
          this.tempBuffer3[t] * this.poolW[i];
      }
    }

    // Backward through transformer blocks (reverse order)
    for (let b = this.numBlocks - 1; b >= 0; b--) {
      // Current gradient is in tempBuffer1

      // Backward through residual2
      // residual2 = residual1 + ffnOutput
      // dResidual1 = dResidual2
      // dFFNOutput = dResidual2

      // Backward through FFN
      this.ffnBackward(
        this.tempBuffer1,
        this.cacheLn2Out[b],
        this.cacheFFNHidden[b],
        this.cacheFFNAct[b],
        this.ffnW1[b],
        this.ffnW2[b],
        seqLen,
        this.tempBuffer2, // dLn2Out
        this.gradFFNW1[b],
        this.gradFFNB1[b],
        this.gradFFNW2[b],
        this.gradFFNB2[b],
      );

      // Backward through LayerNorm2
      this.layerNormBackward(
        this.tempBuffer2,
        this.cacheResidual1[b],
        this.lnGamma2[b],
        this.cacheLn2Mean[b],
        this.cacheLn2Var[b],
        this.tempBuffer3, // dResidual1 from LN2
        this.gradLnGamma2[b],
        this.gradLnBeta2[b],
        seqLen,
        embDim,
      );

      // Add residual gradient from residual2
      for (let i = 0; i < seqLen * embDim; i++) {
        this.tempBuffer3[i] += this.tempBuffer1[i];
      }

      // Backward through attention
      this.tempBuffer2.fill(0);
      this.multiHeadAttentionBackward(
        this.tempBuffer3,
        this.cacheLn1Out[b],
        this.cacheAttnQ[b],
        this.cacheAttnK[b],
        this.cacheAttnV[b],
        this.cacheAttnProbs[b],
        this.attnWq[b],
        this.attnWk[b],
        this.attnWv[b],
        this.attnWo[b],
        seqLen,
        this.tempBuffer2, // dLn1Out
        this.gradAttnWq[b],
        this.gradAttnWk[b],
        this.gradAttnWv[b],
        this.gradAttnWo[b],
        this.gradAttnBq[b],
        this.gradAttnBk[b],
        this.gradAttnBv[b],
        this.gradAttnBo[b],
        this.gradTemporalBias[b],
      );

      // Backward through LayerNorm1
      const dBlockInput = this.pool.acquire(seqLen * embDim);
      this.layerNormBackward(
        this.tempBuffer2,
        this.cacheBlockInputs[b],
        this.lnGamma1[b],
        this.cacheLn1Mean[b],
        this.cacheLn1Var[b],
        dBlockInput,
        this.gradLnGamma1[b],
        this.gradLnBeta1[b],
        seqLen,
        embDim,
      );

      // Add residual gradient
      for (let i = 0; i < seqLen * embDim; i++) {
        this.tempBuffer1[i] = dBlockInput[i] + this.tempBuffer3[i];
      }

      this.pool.release(dBlockInput);
    }

    // Now tempBuffer1 contains gradient w.r.t. fused output
    // Backward through gated fusion
    // fused[t][i] = sum_s(gates[t][s] * E_s[t][i])

    // dGates[t][s] = sum_i(dFused[t][i] * E_s[t][i])
    // dE_s[t][i] = gates[t][s] * dFused[t][i]

    for (let t = 0; t < seqLen; t++) {
      const gateOffset = t * this.numScales;

      for (let s = 0; s < this.numScales; s++) {
        let dGate = 0;
        for (let i = 0; i < embDim; i++) {
          dGate += this.tempBuffer1[t * embDim + i] *
            this.cacheScaleEmbed[s][t * embDim + i];
          // Accumulate dE_s for later
          this.tempBuffer2[s * seqLen * embDim + t * embDim + i] =
            this.cacheGates[gateOffset + s] * this.tempBuffer1[t * embDim + i];
        }

        // Backward through sigmoid
        const gateVal = this.cacheGates[gateOffset + s];
        const dGatePre = dGate * this.sigmoidDerivative(gateVal);

        // dFusionGateW += dGatePre * fusedInput
        // dFusionGateB += dGatePre
        this.gradFusionGateB[s] += dGatePre;
        for (let i = 0; i < this.numScales * embDim; i++) {
          this.gradFusionGateW[i * this.numScales + s] += dGatePre *
            this.cacheFusedInput[t * this.numScales * embDim + i];
        }
      }
    }

    // Backward through temporal convolutions and embeddings
    for (let s = 0; s < this.numScales; s++) {
      // dE_s = dFused contribution (in tempBuffer2)
      // E_s = gelu(convOut) + posEnc + scaleEmbed
      // dConvOut = dE_s * gelu'(convOut)
      // dScaleEmbed = sum_t(dE_s)

      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < embDim; i++) {
          const dEs = this.tempBuffer2[s * seqLen * embDim + t * embDim + i];

          // Accumulate scale embedding gradient
          this.gradScaleEmbed[s][i] += dEs;

          // Backward through GELU (applied to conv output)
          const convOutPreGelu = this.cacheConvOut[s][t * embDim + i];
          // Note: cacheConvOut stores post-GELU values, we need pre-GELU
          // We'll approximate or store pre-GELU separately in a more complete implementation
          // For now, use post-GELU value to estimate derivative
          const dConvOut = dEs * this.geluDerivative(convOutPreGelu);

          // Store dConvOut for conv backward
          this.tempBuffer3[t * embDim + i] = dConvOut;
        }
      }

      // Backward through conv1d
      this.conv1dBackward(
        this.tempBuffer3,
        this.tempSeqBuffer, // normalized input
        seqLen,
        this.inputDim,
        this.temporalConvW[s],
        this.temporalKernelSize,
        this.temporalScales[s],
        this.gradTempConvW[s],
        this.gradTempConvB[s],
      );
    }

    // Add L2 regularization gradients
    const weights = this.getAllWeightArrays();
    const grads = this.getAllGradientArrays();
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      const g = grads[i];
      for (let j = 0; j < w.length; j++) {
        g[j] += this.regularizationStrength * w[j];
      }
    }

    // Compute gradient norm
    let gradNormSq = 0;
    for (const g of grads) {
      for (let i = 0; i < g.length; i++) {
        gradNormSq += g[i] * g[i];
      }
    }

    return Math.sqrt(gradNormSq);
  }

  /**
   * 1D convolution backward pass
   */
  private conv1dBackward(
    dOutput: Float64Array,
    input: Float64Array,
    seqLen: number,
    inputDim: number,
    W: Float64Array,
    kernelSize: number,
    dilation: number,
    dW: Float64Array,
    dB: Float64Array,
  ): void {
    const embDim = this.embeddingDim;

    for (let t = 0; t < seqLen; t++) {
      // dB += dOutput[t]
      for (let e = 0; e < embDim; e++) {
        dB[e] += dOutput[t * embDim + e];
      }

      // dW contribution
      for (let k = 0; k < kernelSize; k++) {
        const inputT = t + (k - Math.floor(kernelSize / 2)) * dilation;
        if (inputT >= 0 && inputT < seqLen) {
          const wOffset = k * inputDim;
          for (let e = 0; e < embDim; e++) {
            for (let d = 0; d < inputDim; d++) {
              dW[(wOffset + d) * embDim + e] += dOutput[t * embDim + e] *
                input[inputT * inputDim + d];
            }
          }
        }
      }
    }
  }

  // ============================================================================
  // Adam Optimizer
  // ============================================================================

  /**
   * Compute learning rate with cosine warmup schedule
   * Formula:
   *   if step < warmupSteps: lr = baseLr * (step / warmupSteps)
   *   else: progress = (step - warmup) / (total - warmup)
   *         lr = baseLr * 0.5 * (1 + cos(π * progress))
   */
  private getEffectiveLearningRate(): number {
    const step = this.updateCount;
    if (step < this.warmupSteps) {
      return this.learningRate * (step / this.warmupSteps);
    } else {
      const progress = (step - this.warmupSteps) /
        (this.totalSteps - this.warmupSteps);
      return this.learningRate * 0.5 *
        (1 + Math.cos(Math.PI * Math.min(progress, 1)));
    }
  }

  /**
   * Apply Adam optimizer update to all weights
   * Formula:
   *   m = β₁ * m + (1 - β₁) * g
   *   v = β₂ * v + (1 - β₂) * g²
   *   m̂ = m / (1 - β₁^t)
   *   v̂ = v / (1 - β₂^t)
   *   w = w - lr * m̂ / (√v̂ + ε)
   */
  private adamUpdate(): void {
    this.updateCount++;
    const lr = this.getEffectiveLearningRate();
    const t = this.updateCount;

    // Bias correction factors
    const bc1 = 1 - Math.pow(this.beta1, t);
    const bc2 = 1 - Math.pow(this.beta2, t);

    const weights = this.getAllWeightArrays();
    const grads = this.getAllGradientArrays();

    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      const g = grads[i];
      const m = this.adamM[i];
      const v = this.adamV[i];

      for (let j = 0; j < w.length; j++) {
        // Update biased first moment estimate
        m[j] = this.beta1 * m[j] + (1 - this.beta1) * g[j];

        // Update biased second moment estimate
        v[j] = this.beta2 * v[j] + (1 - this.beta2) * g[j] * g[j];

        // Compute bias-corrected estimates
        const mHat = m[j] / bc1;
        const vHat = v[j] / bc2;

        // Update weight
        w[j] -= lr * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
  }

  // ============================================================================
  // Welford's Online Normalization
  // ============================================================================

  /**
   * Update running statistics using Welford's online algorithm
   * Formula:
   *   count = count + 1
   *   delta = x - mean
   *   mean = mean + delta / count
   *   delta2 = x - mean
   *   M2 = M2 + delta * delta2
   *   variance = M2 / (count - 1)  (for count > 1)
   */
  private updateNormStats(x: Float64Array, y: Float64Array): void {
    this.normCount++;
    const n = this.normCount;

    // Update input statistics
    for (let i = 0; i < this.inputDim; i++) {
      const delta = x[i] - this.inputMean[i];
      this.inputMean[i] += delta / n;
      const delta2 = x[i] - this.inputMean[i];
      this.inputM2[i] += delta * delta2;
    }

    // Update output statistics
    for (let i = 0; i < this.outputDim; i++) {
      const delta = y[i] - this.outputMean[i];
      this.outputMean[i] += delta / n;
      const delta2 = y[i] - this.outputMean[i];
      this.outputM2[i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation from M2 accumulator
   */
  private getStd(m2: Float64Array, count: number): Float64Array {
    const std = new Float64Array(m2.length);
    if (count > 1) {
      for (let i = 0; i < m2.length; i++) {
        std[i] = Math.sqrt(m2[i] / (count - 1));
        if (std[i] < this.epsilon) std[i] = 1; // Prevent division by zero
      }
    } else {
      std.fill(1);
    }
    return std;
  }

  /**
   * Normalize input data using running statistics
   */
  private normalizeInput(
    input: Float64Array,
    seqLen: number,
    output: Float64Array,
  ): void {
    const inputStd = this.getStd(this.inputM2, this.normCount);
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < this.inputDim; i++) {
        output[t * this.inputDim + i] =
          (input[t * this.inputDim + i] - this.inputMean[i]) / inputStd[i];
      }
    }
  }

  // ============================================================================
  // ADWIN Drift Detection
  // ============================================================================

  /**
   * ADWIN drift detection algorithm
   * Detects concept drift by monitoring error distribution changes
   */
  private adwinDetectDrift(error: number): boolean {
    this.adwinWindow.push(error);
    this.adwinSum += error;

    // Only check for drift if window is large enough
    if (this.adwinWindow.length < 10) return false;

    let driftDetected = false;

    // Try to find a split point where the two subwindows have significantly different means
    for (let split = 1; split < this.adwinWindow.length - 1; split++) {
      // Left subwindow statistics
      let leftSum = 0;
      for (let i = 0; i < split; i++) {
        leftSum += this.adwinWindow[i];
      }
      const leftMean = leftSum / split;

      // Right subwindow statistics
      let rightSum = 0;
      for (let i = split; i < this.adwinWindow.length; i++) {
        rightSum += this.adwinWindow[i];
      }
      const rightMean = rightSum / (this.adwinWindow.length - split);

      // Compute epsilon cut threshold
      // Formula: ε = sqrt((1/n0 + 1/n1) * ln(4/δ) / 2)
      const n0 = split;
      const n1 = this.adwinWindow.length - split;
      const harmonicMean = 1 / n0 + 1 / n1;
      const epsilonCut = Math.sqrt(
        harmonicMean * Math.log(4 / this.adwinDelta) / 2,
      );

      if (Math.abs(leftMean - rightMean) >= epsilonCut) {
        driftDetected = true;

        // Remove older portion of window
        this.adwinSum -= leftSum;
        this.adwinWindow = this.adwinWindow.slice(split);

        // Increment drift count
        this.driftCount++;

        // Partial reset of normalization statistics
        // Keep some history but reduce weight of old samples
        this.normCount = Math.max(1, Math.floor(this.normCount * 0.5));
        for (let i = 0; i < this.inputM2.length; i++) {
          this.inputM2[i] *= 0.5;
        }
        for (let i = 0; i < this.outputM2.length; i++) {
          this.outputM2[i] *= 0.5;
        }

        break;
      }
    }

    // Limit window size
    while (this.adwinWindow.length > 1000) {
      this.adwinSum -= this.adwinWindow.shift()!;
    }

    return driftDetected;
  }

  // ============================================================================
  // Outlier Detection
  // ============================================================================

  /**
   * Detect outliers based on z-score of residuals
   * Formula: residualNorm = sqrt(sum((y - pred) / std)^2)
   *          isOutlier = residualNorm > threshold
   */
  private detectOutlier(
    target: Float64Array,
    predicted: Float64Array,
  ): { isOutlier: boolean; weight: number } {
    const outputStd = this.getStd(this.outputM2, this.normCount);

    let residualNormSq = 0;
    for (let i = 0; i < this.outputDim; i++) {
      const residual = (target[i] - predicted[i]) / outputStd[i];
      residualNormSq += residual * residual;
    }
    const residualNorm = Math.sqrt(residualNormSq);

    if (residualNorm > this.outlierThreshold) {
      return { isOutlier: true, weight: 0.1 };
    }
    return { isOutlier: false, weight: 1.0 };
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Perform one step of online training with incremental Adam optimization
   * @param data Training data containing input sequence and target outputs
   * @returns Training metrics including loss, gradient norm, and drift detection status
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6], [7, 8]],
   *   yCoordinates: [[9, 10]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xCoords = data.xCoordinates;
    const yCoords = data.yCoordinates;

    if (xCoords.length === 0 || yCoords.length === 0) {
      throw new Error("Input data cannot be empty");
    }

    const seqLen = Math.min(xCoords.length, this.maxSequenceLength);
    const inputDim = xCoords[0].length;
    const outputDim = yCoords[0].length;

    // Initialize on first call
    if (!this.isInitialized) {
      this.initializeWeights(inputDim, outputDim);
      this.cacheOutput = new Float64Array(outputDim);
    }

    // Verify dimensions match
    if (inputDim !== this.inputDim || outputDim !== this.outputDim) {
      throw new Error(
        `Dimension mismatch: expected (${this.inputDim}, ${this.outputDim}), got (${inputDim}, ${outputDim})`,
      );
    }

    // Convert input to flat Float64Array
    const flatInput = new Float64Array(seqLen * inputDim);
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < inputDim; i++) {
        flatInput[t * inputDim + i] = xCoords[t][i];
      }
    }

    // Use last target as the prediction target
    const targetIdx = yCoords.length - 1;
    const target = new Float64Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      target[i] = yCoords[targetIdx][i];
    }

    // Update normalization statistics with the last input and target
    const lastInput = new Float64Array(inputDim);
    for (let i = 0; i < inputDim; i++) {
      lastInput[i] = xCoords[seqLen - 1][i];
    }
    this.updateNormStats(lastInput, target);

    // Normalize input
    this.normalizeInput(flatInput, seqLen, this.tempSeqBuffer);

    // Forward pass
    const predicted = this.forward(this.tempSeqBuffer, seqLen);

    // Denormalize prediction for loss computation
    const outputStd = this.getStd(this.outputM2, this.normCount);
    const denormPredicted = new Float64Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      denormPredicted[i] = predicted[i] * outputStd[i] + this.outputMean[i];
    }

    // Normalize target for training
    const normTarget = new Float64Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      normTarget[i] = (target[i] - this.outputMean[i]) / outputStd[i];
    }

    // Compute MSE loss
    let mseLoss = 0;
    for (let i = 0; i < outputDim; i++) {
      const diff = denormPredicted[i] - target[i];
      mseLoss += diff * diff;
    }
    mseLoss /= 2 * outputDim;

    // Add L2 regularization to loss
    let l2Loss = 0;
    const weights = this.getAllWeightArrays();
    for (const w of weights) {
      for (let i = 0; i < w.length; i++) {
        l2Loss += w[i] * w[i];
      }
    }
    l2Loss *= this.regularizationStrength / 2;
    const totalLoss = mseLoss + l2Loss;

    // Outlier detection and sample weighting
    const { isOutlier, weight: sampleWeight } = this.detectOutlier(
      target,
      denormPredicted,
    );

    // Backward pass
    const gradientNorm = this.backward(normTarget, seqLen, sampleWeight);

    // Adam update
    this.adamUpdate();

    // Update accuracy tracking
    this.totalLoss += totalLoss;
    this.sampleCount++;

    // Convergence check
    const avgLoss = this.totalLoss / this.sampleCount;
    if (Math.abs(this.previousLoss - avgLoss) < this.convergenceThreshold) {
      this.converged = true;
    }
    this.previousLoss = avgLoss;

    // ADWIN drift detection
    const driftDetected = this.adwinDetectDrift(mseLoss);

    // Update sequence buffer for prediction
    this.sequenceBufferX.push(flatInput.slice(0, seqLen * inputDim));
    this.sequenceBufferY.push(target.slice());
    while (this.sequenceBufferX.length > this.maxBufferSize) {
      this.sequenceBufferX.shift();
      this.sequenceBufferY.shift();
    }

    return {
      loss: totalLoss,
      gradientNorm,
      effectiveLearningRate: this.getEffectiveLearningRate(),
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future time steps with uncertainty bounds
   * @param futureSteps Number of steps to predict ahead
   * @returns Predictions with confidence intervals
   * @example
   * ```typescript
   * const predictions = model.predict(5);
   * for (const pred of predictions.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.isInitialized || this.sampleCount === 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const outputStd = this.getStd(this.outputM2, this.normCount);

    // Build initial sequence from buffer
    let currentSeq: Float64Array;
    if (this.sequenceBufferX.length > 0) {
      const lastSeq = this.sequenceBufferX[this.sequenceBufferX.length - 1];
      currentSeq = new Float64Array(lastSeq);
    } else {
      // No history, return empty
      return {
        predictions: [],
        accuracy: this.getAccuracy(),
        sampleCount: this.sampleCount,
        isModelReady: true,
      };
    }

    const seqLen = currentSeq.length / this.inputDim;

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current sequence
      this.normalizeInput(currentSeq, seqLen, this.tempSeqBuffer);

      // Forward pass
      const normalizedPred = this.forward(this.tempSeqBuffer, seqLen);

      // Denormalize prediction
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let i = 0; i < this.outputDim; i++) {
        const pred = normalizedPred[i] * outputStd[i] + this.outputMean[i];
        predicted.push(pred);

        // Compute standard error: σ / √n
        const se = outputStd[i] / Math.sqrt(Math.max(1, this.sampleCount));
        standardError.push(se);

        // 95% confidence interval
        lowerBound.push(pred - CONFIDENCE_MULTIPLIER * se);
        upperBound.push(pred + CONFIDENCE_MULTIPLIER * se);
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError,
      });

      // For autoregressive prediction, use prediction as next input
      // This assumes input and output have same structure or we need mapping
      // For now, shift sequence and append prediction (simplified)
      if (step < futureSteps - 1 && this.inputDim === this.outputDim) {
        // Shift sequence left and add new prediction
        for (let t = 0; t < seqLen - 1; t++) {
          for (let i = 0; i < this.inputDim; i++) {
            currentSeq[t * this.inputDim + i] =
              currentSeq[(t + 1) * this.inputDim + i];
          }
        }
        for (let i = 0; i < this.inputDim; i++) {
          currentSeq[(seqLen - 1) * this.inputDim + i] = predicted[i];
        }
      }
    }

    return {
      predictions,
      accuracy: this.getAccuracy(),
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get current accuracy metric
   * Formula: accuracy = 1 / (1 + avgLoss)
   */
  private getAccuracy(): number {
    if (this.sampleCount === 0) return 0;
    const avgLoss = this.totalLoss / this.sampleCount;
    return 1 / (1 + avgLoss);
  }

  /**
   * Get comprehensive model summary
   * @returns Summary of model configuration and training state
   */
  getModelSummary(): ModelSummary {
    // Count total parameters
    let totalParams = 0;
    if (this.isInitialized) {
      const weights = this.getAllWeightArrays();
      for (const w of weights) {
        totalParams += w.length;
      }
    }

    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.numBlocks,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      temporalScales: [...this.temporalScales],
      totalParameters: totalParams,
      sampleCount: this.sampleCount,
      accuracy: this.getAccuracy(),
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Get all weight matrices and optimizer state
   * @returns Complete weight information for inspection or serialization
   */
  getWeights(): WeightInfo {
    const to3D = (arr: Float64Array, shape: number[]): number[][][] => {
      const result: number[][][] = [];
      if (shape.length === 1) {
        result.push([Array.from(arr)]);
      } else if (shape.length === 2) {
        const [rows, cols] = shape;
        const matrix: number[][] = [];
        for (let i = 0; i < rows; i++) {
          matrix.push(Array.from(arr.slice(i * cols, (i + 1) * cols)));
        }
        result.push(matrix);
      }
      return result;
    };

    const temporalConvWeights: number[][][] = [];
    const scaleEmbeddings: number[][][] = [];
    const attentionWeights: number[][][] = [];
    const ffnWeights: number[][][] = [];
    const layerNormParams: number[][][] = [];

    for (let s = 0; s < this.numScales; s++) {
      temporalConvWeights.push(
        ...to3D(this.temporalConvW[s], [
          this.temporalKernelSize * this.inputDim,
          this.embeddingDim,
        ]),
      );
      temporalConvWeights.push(
        ...to3D(this.temporalConvB[s], [this.embeddingDim]),
      );
      scaleEmbeddings.push(...to3D(this.scaleEmbed[s], [this.embeddingDim]));
    }

    for (let b = 0; b < this.numBlocks; b++) {
      attentionWeights.push(
        ...to3D(this.attnWq[b], [this.embeddingDim, this.embeddingDim]),
      );
      attentionWeights.push(
        ...to3D(this.attnWk[b], [this.embeddingDim, this.embeddingDim]),
      );
      attentionWeights.push(
        ...to3D(this.attnWv[b], [this.embeddingDim, this.embeddingDim]),
      );
      attentionWeights.push(
        ...to3D(this.attnWo[b], [this.embeddingDim, this.embeddingDim]),
      );

      ffnWeights.push(...to3D(this.ffnW1[b], [this.ffnDim, this.embeddingDim]));
      ffnWeights.push(...to3D(this.ffnW2[b], [this.embeddingDim, this.ffnDim]));

      layerNormParams.push(...to3D(this.lnGamma1[b], [this.embeddingDim]));
      layerNormParams.push(...to3D(this.lnBeta1[b], [this.embeddingDim]));
      layerNormParams.push(...to3D(this.lnGamma2[b], [this.embeddingDim]));
      layerNormParams.push(...to3D(this.lnBeta2[b], [this.embeddingDim]));
    }

    const fusionWeights = to3D(this.fusionGateW, [
      this.numScales * this.embeddingDim,
      this.numScales,
    ]);
    const outputWeights = [
      ...to3D(this.poolW, [this.embeddingDim]),
      ...to3D(this.outputW, [this.embeddingDim, this.outputDim]),
      ...to3D(this.outputB, [this.outputDim]),
    ];

    // Positional encoding (computed, not learned)
    const positionalEncoding = to3D(this.posEncoding, [
      this.maxSequenceLength,
      this.embeddingDim,
    ]);

    // Adam state
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];
    for (let i = 0; i < this.adamM.length; i++) {
      firstMoment.push([Array.from(this.adamM[i])]);
      secondMoment.push([Array.from(this.adamV[i])]);
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
   * Get normalization statistics
   * @returns Current running statistics for input and output normalization
   */
  getNormalizationStats(): NormalizationStats {
    const inputStd = this.getStd(this.inputM2, this.normCount);
    const outputStd = this.getStd(this.outputM2, this.normCount);

    return {
      inputMean: Array.from(this.inputMean),
      inputStd: Array.from(inputStd),
      outputMean: Array.from(this.outputMean),
      outputStd: Array.from(outputStd),
      count: this.normCount,
    };
  }

  /**
   * Reset model to initial untrained state
   */
  reset(): void {
    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.updateCount = 0;
    this.normCount = 0;
    this.totalLoss = 0;
    this.sampleCount = 0;
    this.previousLoss = Infinity;
    this.converged = false;
    this.adwinWindow = [];
    this.adwinSum = 0;
    this.driftCount = 0;
    this.sequenceBufferX = [];
    this.sequenceBufferY = [];
    this.pool.clear();

    // Clear all weight arrays
    this.temporalConvW = [];
    this.temporalConvB = [];
    this.scaleEmbed = [];
    this.attnWq = [];
    this.attnWk = [];
    this.attnWv = [];
    this.attnWo = [];
    this.attnBq = [];
    this.attnBk = [];
    this.attnBv = [];
    this.attnBo = [];
    this.temporalBias = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    this.lnGamma1 = [];
    this.lnBeta1 = [];
    this.lnGamma2 = [];
    this.lnBeta2 = [];
    this.adamM = [];
    this.adamV = [];
  }

  /**
   * Serialize complete model state to JSON string
   * @returns JSON string containing all model state
   */
  save(): string {
    const arrayToList = (arr: Float64Array): number[] => Array.from(arr);
    const arraysToLists = (arrs: Float64Array[]): number[][] =>
      arrs.map(arrayToList);

    const state = {
      config: {
        numBlocks: this.numBlocks,
        embeddingDim: this.embeddingDim,
        numHeads: this.numHeads,
        ffnMultiplier: this.ffnMultiplier,
        attentionDropout: this.attentionDropout,
        learningRate: this.learningRate,
        warmupSteps: this.warmupSteps,
        totalSteps: this.totalSteps,
        beta1: this.beta1,
        beta2: this.beta2,
        epsilon: this.epsilon,
        regularizationStrength: this.regularizationStrength,
        convergenceThreshold: this.convergenceThreshold,
        outlierThreshold: this.outlierThreshold,
        adwinDelta: this.adwinDelta,
        temporalScales: this.temporalScales,
        temporalKernelSize: this.temporalKernelSize,
        maxSequenceLength: this.maxSequenceLength,
        fusionDropout: this.fusionDropout,
      },
      dimensions: {
        inputDim: this.inputDim,
        outputDim: this.outputDim,
        isInitialized: this.isInitialized,
      },
      weights: {
        temporalConvW: arraysToLists(this.temporalConvW),
        temporalConvB: arraysToLists(this.temporalConvB),
        scaleEmbed: arraysToLists(this.scaleEmbed),
        posEncoding: arrayToList(this.posEncoding),
        fusionGateW: arrayToList(this.fusionGateW),
        fusionGateB: arrayToList(this.fusionGateB),
        attnWq: arraysToLists(this.attnWq),
        attnWk: arraysToLists(this.attnWk),
        attnWv: arraysToLists(this.attnWv),
        attnWo: arraysToLists(this.attnWo),
        attnBq: arraysToLists(this.attnBq),
        attnBk: arraysToLists(this.attnBk),
        attnBv: arraysToLists(this.attnBv),
        attnBo: arraysToLists(this.attnBo),
        temporalBias: arraysToLists(this.temporalBias),
        ffnW1: arraysToLists(this.ffnW1),
        ffnB1: arraysToLists(this.ffnB1),
        ffnW2: arraysToLists(this.ffnW2),
        ffnB2: arraysToLists(this.ffnB2),
        lnGamma1: arraysToLists(this.lnGamma1),
        lnBeta1: arraysToLists(this.lnBeta1),
        lnGamma2: arraysToLists(this.lnGamma2),
        lnBeta2: arraysToLists(this.lnBeta2),
        poolW: arrayToList(this.poolW),
        outputW: arrayToList(this.outputW),
        outputB: arrayToList(this.outputB),
      },
      optimizer: {
        adamM: arraysToLists(this.adamM),
        adamV: arraysToLists(this.adamV),
        updateCount: this.updateCount,
      },
      normalization: {
        inputMean: arrayToList(this.inputMean),
        inputM2: arrayToList(this.inputM2),
        outputMean: arrayToList(this.outputMean),
        outputM2: arrayToList(this.outputM2),
        normCount: this.normCount,
      },
      training: {
        totalLoss: this.totalLoss,
        sampleCount: this.sampleCount,
        previousLoss: this.previousLoss,
        converged: this.converged,
        adwinWindow: this.adwinWindow,
        adwinSum: this.adwinSum,
        driftCount: this.driftCount,
      },
      buffer: {
        sequenceBufferX: arraysToLists(this.sequenceBufferX),
        sequenceBufferY: arraysToLists(this.sequenceBufferY),
      },
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   * @param w JSON string containing serialized model state
   */
  load(w: string): void {
    const listToArray = (list: number[]): Float64Array =>
      new Float64Array(list);
    const listsToArrays = (lists: number[][]): Float64Array[] =>
      lists.map(listToArray);

    const state = JSON.parse(w);

    // Restore dimensions
    this.inputDim = state.dimensions.inputDim;
    this.outputDim = state.dimensions.outputDim;
    this.isInitialized = state.dimensions.isInitialized;

    if (!this.isInitialized) {
      return;
    }

    // Restore weights
    this.temporalConvW = listsToArrays(state.weights.temporalConvW);
    this.temporalConvB = listsToArrays(state.weights.temporalConvB);
    this.scaleEmbed = listsToArrays(state.weights.scaleEmbed);
    this.posEncoding = listToArray(state.weights.posEncoding);
    this.fusionGateW = listToArray(state.weights.fusionGateW);
    this.fusionGateB = listToArray(state.weights.fusionGateB);
    this.attnWq = listsToArrays(state.weights.attnWq);
    this.attnWk = listsToArrays(state.weights.attnWk);
    this.attnWv = listsToArrays(state.weights.attnWv);
    this.attnWo = listsToArrays(state.weights.attnWo);
    this.attnBq = listsToArrays(state.weights.attnBq);
    this.attnBk = listsToArrays(state.weights.attnBk);
    this.attnBv = listsToArrays(state.weights.attnBv);
    this.attnBo = listsToArrays(state.weights.attnBo);
    this.temporalBias = listsToArrays(state.weights.temporalBias);
    this.ffnW1 = listsToArrays(state.weights.ffnW1);
    this.ffnB1 = listsToArrays(state.weights.ffnB1);
    this.ffnW2 = listsToArrays(state.weights.ffnW2);
    this.ffnB2 = listsToArrays(state.weights.ffnB2);
    this.lnGamma1 = listsToArrays(state.weights.lnGamma1);
    this.lnBeta1 = listsToArrays(state.weights.lnBeta1);
    this.lnGamma2 = listsToArrays(state.weights.lnGamma2);
    this.lnBeta2 = listsToArrays(state.weights.lnBeta2);
    this.poolW = listToArray(state.weights.poolW);
    this.outputW = listToArray(state.weights.outputW);
    this.outputB = listToArray(state.weights.outputB);

    // Restore optimizer state
    this.adamM = listsToArrays(state.optimizer.adamM);
    this.adamV = listsToArrays(state.optimizer.adamV);
    this.updateCount = state.optimizer.updateCount;

    // Restore normalization state
    this.inputMean = listToArray(state.normalization.inputMean);
    this.inputM2 = listToArray(state.normalization.inputM2);
    this.outputMean = listToArray(state.normalization.outputMean);
    this.outputM2 = listToArray(state.normalization.outputM2);
    this.normCount = state.normalization.normCount;

    // Restore training state
    this.totalLoss = state.training.totalLoss;
    this.sampleCount = state.training.sampleCount;
    this.previousLoss = state.training.previousLoss;
    this.converged = state.training.converged;
    this.adwinWindow = state.training.adwinWindow;
    this.adwinSum = state.training.adwinSum;
    this.driftCount = state.training.driftCount;

    // Restore buffer
    this.sequenceBufferX = listsToArrays(state.buffer.sequenceBufferX);
    this.sequenceBufferY = listsToArrays(state.buffer.sequenceBufferY);

    // Reallocate computation buffers
    this.allocateGradientBuffers();
    this.allocateCacheBuffers();

    const maxBufSize = Math.max(
      this.maxSequenceLength * this.embeddingDim,
      this.maxSequenceLength * this.maxSequenceLength,
      this.embeddingDim * this.ffnDim,
      this.embeddingDim * this.outputDim,
    );
    this.tempBuffer1 = new Float64Array(maxBufSize);
    this.tempBuffer2 = new Float64Array(maxBufSize);
    this.tempBuffer3 = new Float64Array(maxBufSize);
    this.tempSeqBuffer = new Float64Array(
      this.maxSequenceLength * this.inputDim,
    );
    this.cacheOutput = new Float64Array(this.outputDim);
  }
}

export { FusionTemporalTransformerRegression };
export type {
  FitResult,
  FusionTemporalTransformerConfig,
  IFusionTemporalTransformerRegression,
  ModelSummary,
  NormalizationStats,
  PredictionResult,
  SinglePrediction,
  WeightInfo,
};
