/**
 * @fileoverview Fusion Temporal Transformer for Multivariate Regression
 * with Incremental Online Learning, Adam Optimizer, and Z-Score Normalization.
 *
 * Mathematical Foundations:
 * - Temporal Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
 * - Multi-Head Self-Attention: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
 * - GELU Activation: x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
 * - Layer Normalization: (x - μ) / (σ + ε) · γ + β
 * - Adam Update: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², θ -= η·m̂/(√v̂ + ε)
 * - Welford's Online Variance: δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Result returned from a single online training step
 */
export interface FitResult {
  /** Mean squared error loss for this sample */
  loss: number;
  /** L2 norm of the gradient */
  gradientNorm: number;
  /** Current learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether the model has converged */
  converged: boolean;
  /** Index of this sample in the training sequence */
  sampleIndex: number;
  /** Whether concept drift was detected */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds
 */
export interface SinglePrediction {
  /** Predicted values for each output dimension */
  predicted: number[];
  /** Lower confidence bound (predicted - 1.96 * standardError) */
  lowerBound: number[];
  /** Upper confidence bound (predicted + 1.96 * standardError) */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result from prediction method
 */
export interface PredictionResult {
  /** Array of predictions for each future step */
  predictions: SinglePrediction[];
  /** Model accuracy: 1 / (1 + averageLoss) */
  accuracy: number;
  /** Total number of training samples seen */
  sampleCount: number;
  /** Whether the model is ready for prediction */
  isModelReady: boolean;
}

/**
 * Model weights and optimizer state
 */
export interface WeightInfo {
  /** Temporal convolution weights [scale][kernel][inputDim × embeddingDim] */
  temporalConvWeights: number[][][];
  /** Scale-specific embeddings [scale][embeddingDim] */
  scaleEmbeddings: number[][][];
  /** Positional encoding [maxSeqLen][embeddingDim] */
  positionalEncoding: number[][][];
  /** Fusion gate weights */
  fusionWeights: number[][][];
  /** Attention weights per block [block][Q/K/V/O][weights] */
  attentionWeights: number[][][];
  /** Feed-forward network weights per block */
  ffnWeights: number[][][];
  /** Layer normalization parameters */
  layerNormParams: number[][][];
  /** Output layer weights */
  outputWeights: number[][][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of parameter updates performed */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean for each input dimension */
  inputMean: number[];
  /** Running standard deviation for each input dimension */
  inputStd: number[];
  /** Running mean for each output dimension */
  outputMean: number[];
  /** Running standard deviation for each output dimension */
  outputStd: number[];
  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Summary of model configuration and state
 */
export interface ModelSummary {
  /** Whether the model has been initialized with data */
  isInitialized: boolean;
  /** Number of input features */
  inputDimension: number;
  /** Number of output features */
  outputDimension: number;
  /** Number of transformer blocks */
  numBlocks: number;
  /** Embedding dimension */
  embeddingDim: number;
  /** Number of attention heads */
  numHeads: number;
  /** Temporal scales used */
  temporalScales: number[];
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Number of training samples seen */
  sampleCount: number;
  /** Current model accuracy */
  accuracy: number;
  /** Whether training has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/**
 * Internal configuration interface
 */
interface FusionTemporalConfig {
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

/**
 * Serializable state for save/load
 */
interface SerializableState {
  config: FusionTemporalConfig;
  inputDim: number;
  outputDim: number;
  seqLen: number;
  isInitialized: boolean;
  sampleCount: number;
  updateCount: number;
  converged: boolean;
  driftCount: number;
  runningLossSum: number;
  runningLossCount: number;
  inputMean: number[] | null;
  inputM2: number[] | null;
  outputMean: number[] | null;
  outputM2: number[] | null;
  adwinWindow: number[];
  weights: {
    temporalConvW: number[][][];
    temporalConvB: number[][];
    scaleEmb: number[][];
    posEnc: number[];
    fusionGateW: number[];
    fusionGateB: number[];
    fusionQueryW: number[];
    fusionKeyW: number[];
    fusionValueW: number[];
    attQW: number[][];
    attKW: number[][];
    attVW: number[][];
    attOW: number[][];
    attOB: number[][];
    ffnW1: number[][];
    ffnB1: number[][];
    ffnW2: number[][];
    ffnB2: number[][];
    lnGamma1: number[][];
    lnBeta1: number[][];
    lnGamma2: number[][];
    lnBeta2: number[][];
    outputW: number[];
    outputB: number[];
    poolW: number[];
  };
  adamState: {
    m: { [key: string]: number[] };
    v: { [key: string]: number[] };
  };
  predictionVariance: number[] | null;
}

// ============================================================================
// Buffer Pool for Memory Efficiency
// ============================================================================

/**
 * Object pool for Float64Array to minimize allocations
 */
class BufferPool {
  private readonly pools: Map<number, Float64Array[]> = new Map();
  private readonly maxPoolSize: number = 32;

  /**
   * Acquire a buffer of specified size
   * @param size - Required buffer size
   * @returns Float64Array of requested size
   */
  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(size);
  }

  /**
   * Release a buffer back to the pool
   * @param buffer - Buffer to release
   */
  release(buffer: Float64Array): void {
    const size = buffer.length;
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      // Zero out for security and consistency
      for (let i = 0; i < size; i++) {
        buffer[i] = 0;
      }
      pool.push(buffer);
    }
  }

  /**
   * Clear all pooled buffers
   */
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
 * A neural network architecture combining multi-scale temporal convolutions,
 * cross-scale attention fusion, and transformer blocks for time series regression
 * with incremental online learning capabilities.
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({
 *   numBlocks: 3,
 *   embeddingDim: 64,
 *   numHeads: 8
 * });
 *
 * // Train incrementally
 * const result = model.fitOnline({
 *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
 *   yCoordinates: [[7], [8], [9]]
 * });
 *
 * // Predict future steps
 * const predictions = model.predict(5);
 *
 * // Save and restore
 * const saved = model.save();
 * model.load(saved);
 * ```
 */
export class FusionTemporalTransformerRegression {
  // ============================================================================
  // Private Fields
  // ============================================================================

  /** Model configuration */
  private readonly config: FusionTemporalConfig;

  /** Input feature dimension */
  private inputDim: number = 0;

  /** Output feature dimension */
  private outputDim: number = 0;

  /** Sequence length */
  private seqLen: number = 0;

  /** Dimension per attention head: embeddingDim / numHeads */
  private headDim: number = 0;

  /** FFN hidden dimension: embeddingDim × ffnMultiplier */
  private ffnDim: number = 0;

  /** Whether model is initialized */
  private isInitialized: boolean = false;

  /** Total samples processed */
  private sampleCount: number = 0;

  /** Adam update count */
  private updateCount: number = 0;

  /** Convergence flag */
  private converged: boolean = false;

  /** Drift event count */
  private driftCount: number = 0;

  /** Running loss sum for accuracy */
  private runningLossSum: number = 0;

  /** Running loss count */
  private runningLossCount: number = 0;

  // Welford's algorithm state for normalization
  private inputMean: Float64Array | null = null;
  private inputM2: Float64Array | null = null;
  private outputMean: Float64Array | null = null;
  private outputM2: Float64Array | null = null;

  // ADWIN state
  private adwinWindow: number[] = [];

  // Buffer pool
  private readonly bufferPool: BufferPool = new BufferPool();

  // ============================================================================
  // Model Weights (Float64Array for performance)
  // ============================================================================

  // Temporal convolution: [scale][kernel_pos * inputDim + embDim]
  private temporalConvW: Float64Array[] | null = null;
  private temporalConvB: Float64Array[] | null = null;

  // Scale embeddings: [scale][embeddingDim]
  private scaleEmb: Float64Array[] | null = null;

  // Positional encoding: [maxSeqLen * embeddingDim]
  private posEnc: Float64Array | null = null;

  // Cross-scale fusion
  private fusionGateW: Float64Array | null = null;
  private fusionGateB: Float64Array | null = null;
  private fusionQueryW: Float64Array | null = null;
  private fusionKeyW: Float64Array | null = null;
  private fusionValueW: Float64Array | null = null;

  // Attention weights per block
  private attQW: Float64Array[] | null = null;
  private attKW: Float64Array[] | null = null;
  private attVW: Float64Array[] | null = null;
  private attOW: Float64Array[] | null = null;
  private attOB: Float64Array[] | null = null;

  // FFN weights per block
  private ffnW1: Float64Array[] | null = null;
  private ffnB1: Float64Array[] | null = null;
  private ffnW2: Float64Array[] | null = null;
  private ffnB2: Float64Array[] | null = null;

  // Layer norm per block
  private lnGamma1: Float64Array[] | null = null;
  private lnBeta1: Float64Array[] | null = null;
  private lnGamma2: Float64Array[] | null = null;
  private lnBeta2: Float64Array[] | null = null;

  // Output layer
  private outputW: Float64Array | null = null;
  private outputB: Float64Array | null = null;
  private poolW: Float64Array | null = null;

  // ============================================================================
  // Adam Optimizer State
  // ============================================================================

  private adamM: Map<string, Float64Array> = new Map();
  private adamV: Map<string, Float64Array> = new Map();

  // ============================================================================
  // Preallocated Buffers for Forward/Backward Pass
  // ============================================================================

  // Forward pass caches
  private cacheNormX: Float64Array | null = null;
  private cacheConvOutputs: Float64Array[] | null = null;
  private cacheFusedOutput: Float64Array | null = null;
  private cacheBlockInputs: Float64Array[] | null = null;
  private cacheBlockLN1: Float64Array[] | null = null;
  private cacheAttentionOut: Float64Array[] | null = null;
  private cacheBlockLN2: Float64Array[] | null = null;
  private cacheFFNHidden: Float64Array[] | null = null;
  private cacheBlockOutputs: Float64Array[] | null = null;
  private cachePoolingWeights: Float64Array | null = null;
  private cachePooledOutput: Float64Array | null = null;
  private cachePrediction: Float64Array | null = null;

  // Attention score caches
  private cacheAttScores: Float64Array[] | null = null;
  private cacheQ: Float64Array[] | null = null;
  private cacheK: Float64Array[] | null = null;
  private cacheV: Float64Array[] | null = null;

  // Layer norm caches
  private cacheLN1Mean: Float64Array[] | null = null;
  private cacheLN1Var: Float64Array[] | null = null;
  private cacheLN1Norm: Float64Array[] | null = null;
  private cacheLN2Mean: Float64Array[] | null = null;
  private cacheLN2Var: Float64Array[] | null = null;
  private cacheLN2Norm: Float64Array[] | null = null;

  // Gradient buffers
  private gradOutputW: Float64Array | null = null;
  private gradOutputB: Float64Array | null = null;
  private gradPoolW: Float64Array | null = null;

  // Prediction variance tracking
  private predictionVariance: Float64Array | null = null;

  // Last input sequence for prediction
  private lastInputSequence: Float64Array | null = null;

  // ============================================================================
  // Constructor
  // ============================================================================

  /**
   * Create a new Fusion Temporal Transformer Regression model
   *
   * @param config - Partial configuration to override defaults
   *
   * @example
   * ```typescript
   * const model = new FusionTemporalTransformerRegression({
   *   numBlocks: 4,
   *   embeddingDim: 128,
   *   learningRate: 0.0005
   * });
   * ```
   */
  constructor(config: Partial<FusionTemporalConfig> = {}) {
    this.config = {
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
      ...config,
    };

    // Validate config
    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this.config.embeddingDim}) must be divisible by numHeads (${this.config.numHeads})`,
      );
    }

    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;
  }

  // ============================================================================
  // Private Initialization Methods
  // ============================================================================

  /**
   * Initialize model with detected dimensions
   * @param inputDim - Number of input features
   * @param outputDim - Number of output features
   * @param seqLen - Sequence length
   */
  private initialize(
    inputDim: number,
    outputDim: number,
    seqLen: number,
  ): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.seqLen = Math.min(seqLen, this.config.maxSequenceLength);

    this.initializeNormalization();
    this.initializeWeights();
    this.initializeAdamState();
    this.allocateBuffers();

    this.isInitialized = true;
  }

  /**
   * Initialize Welford's algorithm state for z-score normalization
   */
  private initializeNormalization(): void {
    this.inputMean = new Float64Array(this.inputDim);
    this.inputM2 = new Float64Array(this.inputDim);
    this.outputMean = new Float64Array(this.outputDim);
    this.outputM2 = new Float64Array(this.outputDim);
    this.predictionVariance = new Float64Array(this.outputDim);

    // Initialize variance estimates
    for (let i = 0; i < this.outputDim; i++) {
      this.predictionVariance[i] = 1.0;
    }
  }

  /**
   * Xavier/Glorot initialization for a weight matrix
   * Formula: W ~ U(-√(6/(fanIn+fanOut)), √(6/(fanIn+fanOut)))
   *
   * @param size - Total number of parameters
   * @param fanIn - Input dimension
   * @param fanOut - Output dimension
   * @returns Initialized Float64Array
   */
  private xavierInit(
    size: number,
    fanIn: number,
    fanOut: number,
  ): Float64Array {
    const arr = new Float64Array(size);
    const limit = Math.sqrt(6.0 / (fanIn + fanOut));
    for (let i = 0; i < size; i++) {
      arr[i] = (Math.random() * 2 - 1) * limit;
    }
    return arr;
  }

  /**
   * Initialize all model weights
   */
  private initializeWeights(): void {
    const { embeddingDim, numBlocks, temporalScales, temporalKernelSize } =
      this.config;

    // Temporal convolution weights: [scale][kernelSize * inputDim * embeddingDim]
    this.temporalConvW = [];
    this.temporalConvB = [];
    for (let s = 0; s < temporalScales.length; s++) {
      const wSize = temporalKernelSize * this.inputDim * embeddingDim;
      this.temporalConvW.push(
        this.xavierInit(
          wSize,
          temporalKernelSize * this.inputDim,
          embeddingDim,
        ),
      );
      this.temporalConvB.push(new Float64Array(embeddingDim));
    }

    // Scale embeddings: [scale][embeddingDim]
    this.scaleEmb = [];
    for (let s = 0; s < temporalScales.length; s++) {
      this.scaleEmb.push(this.xavierInit(embeddingDim, 1, embeddingDim));
    }

    // Positional encoding (precomputed, not learned)
    // PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
    this.posEnc = new Float64Array(
      this.config.maxSequenceLength * embeddingDim,
    );
    for (let pos = 0; pos < this.config.maxSequenceLength; pos++) {
      for (let i = 0; i < embeddingDim; i++) {
        const angle = pos /
          Math.pow(10000, (2 * Math.floor(i / 2)) / embeddingDim);
        const offset = pos * embeddingDim + i;
        this.posEnc[offset] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }

    // Cross-scale fusion weights
    const totalScaleLen = temporalScales.reduce(
      (sum, s) => sum + Math.ceil(this.seqLen / s),
      0,
    );
    this.fusionGateW = this.xavierInit(
      totalScaleLen * embeddingDim * temporalScales.length,
      totalScaleLen * embeddingDim,
      temporalScales.length,
    );
    this.fusionGateB = new Float64Array(temporalScales.length);
    this.fusionQueryW = this.xavierInit(
      embeddingDim * embeddingDim,
      embeddingDim,
      embeddingDim,
    );
    this.fusionKeyW = this.xavierInit(
      embeddingDim * embeddingDim,
      embeddingDim,
      embeddingDim,
    );
    this.fusionValueW = this.xavierInit(
      embeddingDim * embeddingDim,
      embeddingDim,
      embeddingDim,
    );

    // Attention weights per block: Q, K, V, O projections
    this.attQW = [];
    this.attKW = [];
    this.attVW = [];
    this.attOW = [];
    this.attOB = [];
    for (let b = 0; b < numBlocks; b++) {
      this.attQW.push(
        this.xavierInit(
          embeddingDim * embeddingDim,
          embeddingDim,
          embeddingDim,
        ),
      );
      this.attKW.push(
        this.xavierInit(
          embeddingDim * embeddingDim,
          embeddingDim,
          embeddingDim,
        ),
      );
      this.attVW.push(
        this.xavierInit(
          embeddingDim * embeddingDim,
          embeddingDim,
          embeddingDim,
        ),
      );
      this.attOW.push(
        this.xavierInit(
          embeddingDim * embeddingDim,
          embeddingDim,
          embeddingDim,
        ),
      );
      this.attOB.push(new Float64Array(embeddingDim));
    }

    // FFN weights per block
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    for (let b = 0; b < numBlocks; b++) {
      this.ffnW1.push(
        this.xavierInit(embeddingDim * this.ffnDim, embeddingDim, this.ffnDim),
      );
      this.ffnB1.push(new Float64Array(this.ffnDim));
      this.ffnW2.push(
        this.xavierInit(this.ffnDim * embeddingDim, this.ffnDim, embeddingDim),
      );
      this.ffnB2.push(new Float64Array(embeddingDim));
    }

    // Layer norm parameters per block
    this.lnGamma1 = [];
    this.lnBeta1 = [];
    this.lnGamma2 = [];
    this.lnBeta2 = [];
    for (let b = 0; b < numBlocks; b++) {
      const g1 = new Float64Array(embeddingDim);
      const g2 = new Float64Array(embeddingDim);
      g1.fill(1.0);
      g2.fill(1.0);
      this.lnGamma1.push(g1);
      this.lnBeta1.push(new Float64Array(embeddingDim));
      this.lnGamma2.push(g2);
      this.lnBeta2.push(new Float64Array(embeddingDim));
    }

    // Output layer
    this.outputW = this.xavierInit(
      embeddingDim * this.outputDim,
      embeddingDim,
      this.outputDim,
    );
    this.outputB = new Float64Array(this.outputDim);
    this.poolW = this.xavierInit(embeddingDim, embeddingDim, 1);
  }

  /**
   * Initialize Adam optimizer state (first and second moment estimates)
   */
  private initializeAdamState(): void {
    this.adamM.clear();
    this.adamV.clear();

    const initMV = (name: string, arr: Float64Array | null): void => {
      if (arr) {
        this.adamM.set(name, new Float64Array(arr.length));
        this.adamV.set(name, new Float64Array(arr.length));
      }
    };

    // Temporal conv
    for (let s = 0; s < this.config.temporalScales.length; s++) {
      initMV(`temporalConvW_${s}`, this.temporalConvW![s]);
      initMV(`temporalConvB_${s}`, this.temporalConvB![s]);
      initMV(`scaleEmb_${s}`, this.scaleEmb![s]);
    }

    // Fusion
    initMV("fusionGateW", this.fusionGateW);
    initMV("fusionGateB", this.fusionGateB);
    initMV("fusionQueryW", this.fusionQueryW);
    initMV("fusionKeyW", this.fusionKeyW);
    initMV("fusionValueW", this.fusionValueW);

    // Per-block weights
    for (let b = 0; b < this.config.numBlocks; b++) {
      initMV(`attQW_${b}`, this.attQW![b]);
      initMV(`attKW_${b}`, this.attKW![b]);
      initMV(`attVW_${b}`, this.attVW![b]);
      initMV(`attOW_${b}`, this.attOW![b]);
      initMV(`attOB_${b}`, this.attOB![b]);
      initMV(`ffnW1_${b}`, this.ffnW1![b]);
      initMV(`ffnB1_${b}`, this.ffnB1![b]);
      initMV(`ffnW2_${b}`, this.ffnW2![b]);
      initMV(`ffnB2_${b}`, this.ffnB2![b]);
      initMV(`lnGamma1_${b}`, this.lnGamma1![b]);
      initMV(`lnBeta1_${b}`, this.lnBeta1![b]);
      initMV(`lnGamma2_${b}`, this.lnGamma2![b]);
      initMV(`lnBeta2_${b}`, this.lnBeta2![b]);
    }

    // Output
    initMV("outputW", this.outputW);
    initMV("outputB", this.outputB);
    initMV("poolW", this.poolW);
  }

  /**
   * Preallocate buffers for forward/backward passes
   */
  private allocateBuffers(): void {
    const { embeddingDim, numBlocks, temporalScales } = this.config;
    const seqLen = this.seqLen;

    // Normalized input cache
    this.cacheNormX = new Float64Array(seqLen * this.inputDim);

    // Temporal conv outputs per scale
    this.cacheConvOutputs = [];
    for (let s = 0; s < temporalScales.length; s++) {
      const outLen = Math.ceil(seqLen / temporalScales[s]);
      this.cacheConvOutputs.push(new Float64Array(outLen * embeddingDim));
    }

    // Fused output
    this.cacheFusedOutput = new Float64Array(seqLen * embeddingDim);

    // Block-level caches
    this.cacheBlockInputs = [];
    this.cacheBlockLN1 = [];
    this.cacheAttentionOut = [];
    this.cacheBlockLN2 = [];
    this.cacheFFNHidden = [];
    this.cacheBlockOutputs = [];
    this.cacheLN1Mean = [];
    this.cacheLN1Var = [];
    this.cacheLN1Norm = [];
    this.cacheLN2Mean = [];
    this.cacheLN2Var = [];
    this.cacheLN2Norm = [];
    this.cacheAttScores = [];
    this.cacheQ = [];
    this.cacheK = [];
    this.cacheV = [];

    for (let b = 0; b < numBlocks; b++) {
      this.cacheBlockInputs.push(new Float64Array(seqLen * embeddingDim));
      this.cacheBlockLN1.push(new Float64Array(seqLen * embeddingDim));
      this.cacheAttentionOut.push(new Float64Array(seqLen * embeddingDim));
      this.cacheBlockLN2.push(new Float64Array(seqLen * embeddingDim));
      this.cacheFFNHidden.push(new Float64Array(seqLen * this.ffnDim));
      this.cacheBlockOutputs.push(new Float64Array(seqLen * embeddingDim));
      this.cacheLN1Mean.push(new Float64Array(seqLen));
      this.cacheLN1Var.push(new Float64Array(seqLen));
      this.cacheLN1Norm.push(new Float64Array(seqLen * embeddingDim));
      this.cacheLN2Mean.push(new Float64Array(seqLen));
      this.cacheLN2Var.push(new Float64Array(seqLen));
      this.cacheLN2Norm.push(new Float64Array(seqLen * embeddingDim));
      this.cacheAttScores.push(
        new Float64Array(seqLen * seqLen * this.config.numHeads),
      );
      this.cacheQ.push(new Float64Array(seqLen * embeddingDim));
      this.cacheK.push(new Float64Array(seqLen * embeddingDim));
      this.cacheV.push(new Float64Array(seqLen * embeddingDim));
    }

    // Pooling and output caches
    this.cachePoolingWeights = new Float64Array(seqLen);
    this.cachePooledOutput = new Float64Array(embeddingDim);
    this.cachePrediction = new Float64Array(this.outputDim);

    // Gradient buffers
    this.gradOutputW = new Float64Array(embeddingDim * this.outputDim);
    this.gradOutputB = new Float64Array(this.outputDim);
    this.gradPoolW = new Float64Array(embeddingDim);

    // Last input sequence
    this.lastInputSequence = new Float64Array(seqLen * this.inputDim);
  }

  // ============================================================================
  // Mathematical Operations (Optimized, In-Place)
  // ============================================================================

  /**
   * GELU activation function
   * Formula: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
   *
   * @param x - Input value
   * @returns GELU(x)
   */
  private gelu(x: number): number {
    const c = 0.7978845608028654; // √(2/π)
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  }

  /**
   * GELU derivative
   * @param x - Input value
   * @param geluX - Precomputed GELU(x) if available
   * @returns d(GELU)/dx
   */
  private geluDerivative(x: number): number {
    const c = 0.7978845608028654;
    const inner = c * (x + 0.044715 * x * x * x);
    const tanhInner = Math.tanh(inner);
    const sech2 = 1 - tanhInner * tanhInner;
    const innerDeriv = c * (1 + 3 * 0.044715 * x * x);
    return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerDeriv;
  }

  /**
   * Softmax with numerical stability (in-place)
   * Formula: softmax(x)_i = exp(x_i - max(x)) / Σexp(x_j - max(x))
   *
   * @param arr - Input array
   * @param start - Start index
   * @param length - Length of softmax region
   */
  private softmaxInPlace(
    arr: Float64Array,
    start: number,
    length: number,
  ): void {
    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < length; i++) {
      if (arr[start + i] > maxVal) {
        maxVal = arr[start + i];
      }
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < length; i++) {
      arr[start + i] = Math.exp(arr[start + i] - maxVal);
      sum += arr[start + i];
    }

    // Normalize
    const invSum = 1.0 / (sum + this.config.epsilon);
    for (let i = 0; i < length; i++) {
      arr[start + i] *= invSum;
    }
  }

  /**
   * Layer normalization forward pass (in-place output)
   * Formula: y = γ * (x - μ) / √(σ² + ε) + β
   *
   * @param input - Input array [seqLen * dim]
   * @param output - Output array [seqLen * dim]
   * @param gamma - Scale parameters [dim]
   * @param beta - Shift parameters [dim]
   * @param seqLen - Sequence length
   * @param dim - Feature dimension
   * @param meanCache - Cache for mean values [seqLen]
   * @param varCache - Cache for variance values [seqLen]
   * @param normCache - Cache for normalized values [seqLen * dim]
   */
  private layerNormForward(
    input: Float64Array,
    output: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    seqLen: number,
    dim: number,
    meanCache: Float64Array,
    varCache: Float64Array,
    normCache: Float64Array,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const offset = t * dim;

      // Compute mean
      let mean = 0;
      for (let i = 0; i < dim; i++) {
        mean += input[offset + i];
      }
      mean /= dim;
      meanCache[t] = mean;

      // Compute variance
      let variance = 0;
      for (let i = 0; i < dim; i++) {
        const diff = input[offset + i] - mean;
        variance += diff * diff;
      }
      variance /= dim;
      varCache[t] = variance;

      // Normalize and scale
      const invStd = 1.0 / Math.sqrt(variance + this.config.epsilon);
      for (let i = 0; i < dim; i++) {
        const normalized = (input[offset + i] - mean) * invStd;
        normCache[offset + i] = normalized;
        output[offset + i] = gamma[i] * normalized + beta[i];
      }
    }
  }

  /**
   * Layer normalization backward pass
   *
   * @param dOutput - Gradient w.r.t. output
   * @param normCache - Cached normalized values
   * @param varCache - Cached variance values
   * @param gamma - Scale parameters
   * @param seqLen - Sequence length
   * @param dim - Feature dimension
   * @param dInput - Gradient w.r.t. input (output)
   * @param dGamma - Gradient w.r.t. gamma (accumulated)
   * @param dBeta - Gradient w.r.t. beta (accumulated)
   */
  private layerNormBackward(
    dOutput: Float64Array,
    normCache: Float64Array,
    varCache: Float64Array,
    gamma: Float64Array,
    seqLen: number,
    dim: number,
    dInput: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      const offset = t * dim;
      const invStd = 1.0 / Math.sqrt(varCache[t] + this.config.epsilon);

      // Accumulate dGamma and dBeta
      for (let i = 0; i < dim; i++) {
        dGamma[i] += dOutput[offset + i] * normCache[offset + i];
        dBeta[i] += dOutput[offset + i];
      }

      // Compute dNorm
      let sumDNorm = 0;
      let sumDNormNorm = 0;
      for (let i = 0; i < dim; i++) {
        const dNorm = dOutput[offset + i] * gamma[i];
        sumDNorm += dNorm;
        sumDNormNorm += dNorm * normCache[offset + i];
      }

      // Compute dInput
      const invDim = 1.0 / dim;
      for (let i = 0; i < dim; i++) {
        const dNorm = dOutput[offset + i] * gamma[i];
        dInput[offset + i] = invStd *
          (dNorm - invDim * sumDNorm -
            invDim * normCache[offset + i] * sumDNormNorm);
      }
    }
  }

  /**
   * Matrix multiply: C = A × B (A: [m×k], B: [k×n], C: [m×n])
   * Optimized loop order for cache efficiency
   */
  private matmul(
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
    // Clear output
    for (let i = 0; i < m * n; i++) {
      C[cOffset + i] = 0;
    }

    // Cache-friendly multiply
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const aVal = A[aOffset + i * k + p];
        const bRow = bOffset + p * n;
        const cRow = cOffset + i * n;
        for (let j = 0; j < n; j++) {
          C[cRow + j] += aVal * B[bRow + j];
        }
      }
    }
  }

  /**
   * Matrix multiply with transpose of B: C = A × Bᵀ
   */
  private matmulBT(
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
        const aRow = aOffset + i * k;
        const bRow = bOffset + j * k;
        for (let p = 0; p < k; p++) {
          sum += A[aRow + p] * B[bRow + p];
        }
        C[cOffset + i * n + j] = sum;
      }
    }
  }

  /**
   * Matrix multiply with transpose of A: C = Aᵀ × B
   */
  private matmulAT(
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
    // Clear output
    for (let i = 0; i < m * n; i++) {
      C[cOffset + i] = 0;
    }

    for (let p = 0; p < k; p++) {
      for (let i = 0; i < m; i++) {
        const aVal = A[aOffset + p * m + i];
        const cRow = cOffset + i * n;
        const bRow = bOffset + p * n;
        for (let j = 0; j < n; j++) {
          C[cRow + j] += aVal * B[bRow + j];
        }
      }
    }
  }

  // ============================================================================
  // Forward Pass Components
  // ============================================================================

  /**
   * Temporal 1D convolution forward pass
   * Formula: F_s = GELU(Conv1D(X, kernel=k, stride=s))
   *
   * @param input - Normalized input [seqLen × inputDim]
   * @param scaleIdx - Index of temporal scale
   * @param output - Output buffer
   */
  private temporalConvForward(
    input: Float64Array,
    scaleIdx: number,
    output: Float64Array,
  ): void {
    const { embeddingDim, temporalKernelSize, temporalScales } = this.config;
    const stride = temporalScales[scaleIdx];
    const weights = this.temporalConvW![scaleIdx];
    const bias = this.temporalConvB![scaleIdx];
    const outLen = Math.ceil(this.seqLen / stride);
    const padSize = Math.floor(temporalKernelSize / 2);

    for (let t = 0; t < outLen; t++) {
      const centerPos = t * stride;

      for (let d = 0; d < embeddingDim; d++) {
        let sum = bias[d];

        for (let k = 0; k < temporalKernelSize; k++) {
          const inputPos = centerPos + k - padSize;

          if (inputPos >= 0 && inputPos < this.seqLen) {
            const inputOffset = inputPos * this.inputDim;
            const wOffset = (k * this.inputDim + 0) * embeddingDim + d;

            for (let i = 0; i < this.inputDim; i++) {
              sum += input[inputOffset + i] *
                weights[
                  i * embeddingDim +
                  k * this.inputDim * embeddingDim / this.inputDim + d
                ];
            }
          }
        }

        // Apply GELU
        output[t * embeddingDim + d] = this.gelu(sum);
      }
    }
  }

  /**
   * Simplified temporal convolution forward (linear projection + GELU)
   */
  private temporalConvForwardSimple(
    input: Float64Array,
    scaleIdx: number,
    output: Float64Array,
  ): void {
    const { embeddingDim, temporalScales } = this.config;
    const stride = temporalScales[scaleIdx];
    const weights = this.temporalConvW![scaleIdx];
    const bias = this.temporalConvB![scaleIdx];
    const outLen = Math.ceil(this.seqLen / stride);

    // Simple strided projection with aggregation
    for (let t = 0; t < outLen; t++) {
      const startPos = t * stride;
      const endPos = Math.min(startPos + stride, this.seqLen);

      for (let d = 0; d < embeddingDim; d++) {
        let sum = bias[d];

        // Aggregate over stride window
        for (let p = startPos; p < endPos; p++) {
          for (let i = 0; i < this.inputDim; i++) {
            sum += input[p * this.inputDim + i] * weights[i * embeddingDim + d];
          }
        }

        sum /= endPos - startPos;
        output[t * embeddingDim + d] = this.gelu(sum);
      }
    }
  }

  /**
   * Add positional encoding and scale embedding
   *
   * @param buffer - Feature buffer to modify in-place
   * @param scaleIdx - Scale index for scale embedding
   * @param len - Sequence length at this scale
   */
  private addPositionalAndScaleEmb(
    buffer: Float64Array,
    scaleIdx: number,
    len: number,
  ): void {
    const { embeddingDim } = this.config;
    const scaleEmb = this.scaleEmb![scaleIdx];

    for (let t = 0; t < len; t++) {
      const offset = t * embeddingDim;
      const posOffset = t * embeddingDim;

      for (let d = 0; d < embeddingDim; d++) {
        buffer[offset + d] += this.posEnc![posOffset + d] + scaleEmb[d];
      }
    }
  }

  /**
   * Cross-scale attention fusion
   * Query from finest scale, keys/values from all scales
   *
   * @param scaleOutputs - Outputs from each temporal scale
   * @param output - Fused output buffer
   */
  private crossScaleFusion(
    scaleOutputs: Float64Array[],
    output: Float64Array,
  ): void {
    const { embeddingDim, temporalScales } = this.config;
    const finestLen = Math.ceil(this.seqLen / temporalScales[0]);

    // Use finest scale as query
    const Q = this.bufferPool.acquire(finestLen * embeddingDim);
    this.matmul(
      scaleOutputs[0],
      this.fusionQueryW!,
      Q,
      finestLen,
      embeddingDim,
      embeddingDim,
    );

    // Concatenate all scales for keys and values
    let totalLen = 0;
    for (let s = 0; s < temporalScales.length; s++) {
      totalLen += Math.ceil(this.seqLen / temporalScales[s]);
    }

    const concat = this.bufferPool.acquire(totalLen * embeddingDim);
    let offset = 0;
    for (let s = 0; s < temporalScales.length; s++) {
      const scaleLen = Math.ceil(this.seqLen / temporalScales[s]);
      for (let i = 0; i < scaleLen * embeddingDim; i++) {
        concat[offset + i] = scaleOutputs[s][i];
      }
      offset += scaleLen * embeddingDim;
    }

    // Compute K and V
    const K = this.bufferPool.acquire(totalLen * embeddingDim);
    const V = this.bufferPool.acquire(totalLen * embeddingDim);
    this.matmul(
      concat,
      this.fusionKeyW!,
      K,
      totalLen,
      embeddingDim,
      embeddingDim,
    );
    this.matmul(
      concat,
      this.fusionValueW!,
      V,
      totalLen,
      embeddingDim,
      embeddingDim,
    );

    // Attention: softmax(QK^T / √d) V
    const scores = this.bufferPool.acquire(finestLen * totalLen);
    const scale = 1.0 / Math.sqrt(embeddingDim);

    this.matmulBT(Q, K, scores, finestLen, embeddingDim, totalLen);

    // Scale and softmax
    for (let i = 0; i < finestLen; i++) {
      for (let j = 0; j < totalLen; j++) {
        scores[i * totalLen + j] *= scale;
      }
      this.softmaxInPlace(scores, i * totalLen, totalLen);
    }

    // Output
    this.matmul(scores, V, output, finestLen, totalLen, embeddingDim);

    // Upsample to original sequence length if needed
    if (finestLen < this.seqLen) {
      const tempOutput = this.bufferPool.acquire(this.seqLen * embeddingDim);
      for (let t = 0; t < this.seqLen; t++) {
        const srcIdx = Math.min(
          Math.floor(t / temporalScales[0]),
          finestLen - 1,
        );
        for (let d = 0; d < embeddingDim; d++) {
          tempOutput[t * embeddingDim + d] = output[srcIdx * embeddingDim + d];
        }
      }
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        output[i] = tempOutput[i];
      }
      this.bufferPool.release(tempOutput);
    }

    this.bufferPool.release(Q);
    this.bufferPool.release(K);
    this.bufferPool.release(V);
    this.bufferPool.release(concat);
    this.bufferPool.release(scores);
  }

  /**
   * Multi-head self-attention forward pass
   *
   * @param input - Input [seqLen × embeddingDim]
   * @param blockIdx - Transformer block index
   * @param output - Output buffer
   */
  private multiHeadAttentionForward(
    input: Float64Array,
    blockIdx: number,
    output: Float64Array,
  ): void {
    const { embeddingDim, numHeads } = this.config;
    const headDim = this.headDim;
    const seqLen = this.seqLen;

    const Q = this.cacheQ![blockIdx];
    const K = this.cacheK![blockIdx];
    const V = this.cacheV![blockIdx];
    const scores = this.cacheAttScores![blockIdx];

    // Project to Q, K, V
    this.matmul(
      input,
      this.attQW![blockIdx],
      Q,
      seqLen,
      embeddingDim,
      embeddingDim,
    );
    this.matmul(
      input,
      this.attKW![blockIdx],
      K,
      seqLen,
      embeddingDim,
      embeddingDim,
    );
    this.matmul(
      input,
      this.attVW![blockIdx],
      V,
      seqLen,
      embeddingDim,
      embeddingDim,
    );

    // Multi-head attention
    const scale = 1.0 / Math.sqrt(headDim);
    const attOut = this.bufferPool.acquire(seqLen * embeddingDim);

    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scoreOffset = h * seqLen * seqLen;

      // Compute attention scores for this head
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += Q[i * embeddingDim + headOffset + d] *
              K[j * embeddingDim + headOffset + d];
          }
          scores[scoreOffset + i * seqLen + j] = score * scale;

          // Causal mask (optional - for autoregressive)
          if (j > i) {
            scores[scoreOffset + i * seqLen + j] = -1e9;
          }
        }

        // Softmax over keys
        this.softmaxInPlace(scores, scoreOffset + i * seqLen, seqLen);
      }

      // Attention-weighted values
      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += scores[scoreOffset + i * seqLen + j] *
              V[j * embeddingDim + headOffset + d];
          }
          attOut[i * embeddingDim + headOffset + d] = sum;
        }
      }
    }

    // Output projection
    this.matmul(
      attOut,
      this.attOW![blockIdx],
      output,
      seqLen,
      embeddingDim,
      embeddingDim,
    );

    // Add bias
    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < embeddingDim; d++) {
        output[i * embeddingDim + d] += this.attOB![blockIdx][d];
      }
    }

    this.bufferPool.release(attOut);
  }

  /**
   * Feed-forward network forward pass
   * Formula: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
   *
   * @param input - Input [seqLen × embeddingDim]
   * @param blockIdx - Block index
   * @param hiddenCache - Cache for hidden activations
   * @param output - Output buffer
   */
  private ffnForward(
    input: Float64Array,
    blockIdx: number,
    hiddenCache: Float64Array,
    output: Float64Array,
  ): void {
    const { embeddingDim } = this.config;
    const seqLen = this.seqLen;

    // First linear + GELU
    this.matmul(
      input,
      this.ffnW1![blockIdx],
      hiddenCache,
      seqLen,
      embeddingDim,
      this.ffnDim,
    );

    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < this.ffnDim; d++) {
        const idx = i * this.ffnDim + d;
        hiddenCache[idx] = this.gelu(
          hiddenCache[idx] + this.ffnB1![blockIdx][d],
        );
      }
    }

    // Second linear
    this.matmul(
      hiddenCache,
      this.ffnW2![blockIdx],
      output,
      seqLen,
      this.ffnDim,
      embeddingDim,
    );

    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < embeddingDim; d++) {
        output[i * embeddingDim + d] += this.ffnB2![blockIdx][d];
      }
    }
  }

  /**
   * Attention-weighted pooling
   * Formula: α = softmax(HW_pool), out = Σα_i h_i
   *
   * @param input - Transformer output [seqLen × embeddingDim]
   * @param poolWeights - Attention weights buffer
   * @param output - Pooled output [embeddingDim]
   */
  private attentionPooling(
    input: Float64Array,
    poolWeights: Float64Array,
    output: Float64Array,
  ): void {
    const { embeddingDim } = this.config;
    const seqLen = this.seqLen;

    // Compute attention scores
    for (let t = 0; t < seqLen; t++) {
      let score = 0;
      for (let d = 0; d < embeddingDim; d++) {
        score += input[t * embeddingDim + d] * this.poolW![d];
      }
      poolWeights[t] = score;
    }

    // Softmax
    this.softmaxInPlace(poolWeights, 0, seqLen);

    // Weighted sum
    output.fill(0);
    for (let t = 0; t < seqLen; t++) {
      const w = poolWeights[t];
      for (let d = 0; d < embeddingDim; d++) {
        output[d] += w * input[t * embeddingDim + d];
      }
    }
  }

  /**
   * Full forward pass
   *
   * @param x - Input sequence [seqLen × inputDim]
   * @param y - Optional target for loss computation
   * @returns Prediction and loss
   */
  private forward(
    x: Float64Array,
    y?: Float64Array,
  ): { prediction: Float64Array; loss: number } {
    const { embeddingDim, numBlocks, temporalScales } = this.config;

    // 1. Normalize input
    this.normalizeInput(x, this.cacheNormX!);

    // 2. Multi-scale temporal convolution
    for (let s = 0; s < temporalScales.length; s++) {
      this.temporalConvForwardSimple(
        this.cacheNormX!,
        s,
        this.cacheConvOutputs![s],
      );
      const scaleLen = Math.ceil(this.seqLen / temporalScales[s]);
      this.addPositionalAndScaleEmb(this.cacheConvOutputs![s], s, scaleLen);
    }

    // 3. Cross-scale fusion
    this.crossScaleFusion(this.cacheConvOutputs!, this.cacheFusedOutput!);

    // 4. Transformer blocks
    let currentInput = this.cacheFusedOutput!;

    for (let b = 0; b < numBlocks; b++) {
      // Cache block input
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        this.cacheBlockInputs![b][i] = currentInput[i];
      }

      // LayerNorm 1
      this.layerNormForward(
        currentInput,
        this.cacheBlockLN1![b],
        this.lnGamma1![b],
        this.lnBeta1![b],
        this.seqLen,
        embeddingDim,
        this.cacheLN1Mean![b],
        this.cacheLN1Var![b],
        this.cacheLN1Norm![b],
      );

      // Multi-head attention
      this.multiHeadAttentionForward(
        this.cacheBlockLN1![b],
        b,
        this.cacheAttentionOut![b],
      );

      // Residual connection
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        this.cacheAttentionOut![b][i] += this.cacheBlockInputs![b][i];
      }

      // LayerNorm 2
      this.layerNormForward(
        this.cacheAttentionOut![b],
        this.cacheBlockLN2![b],
        this.lnGamma2![b],
        this.lnBeta2![b],
        this.seqLen,
        embeddingDim,
        this.cacheLN2Mean![b],
        this.cacheLN2Var![b],
        this.cacheLN2Norm![b],
      );

      // FFN
      this.ffnForward(
        this.cacheBlockLN2![b],
        b,
        this.cacheFFNHidden![b],
        this.cacheBlockOutputs![b],
      );

      // Residual connection
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        this.cacheBlockOutputs![b][i] += this.cacheAttentionOut![b][i];
      }

      currentInput = this.cacheBlockOutputs![b];
    }

    // 5. Attention pooling
    this.attentionPooling(
      currentInput,
      this.cachePoolingWeights!,
      this.cachePooledOutput!,
    );

    // 6. Output projection
    for (let d = 0; d < this.outputDim; d++) {
      let sum = this.outputB![d];
      for (let i = 0; i < embeddingDim; i++) {
        sum += this.cachePooledOutput![i] *
          this.outputW![i * this.outputDim + d];
      }
      this.cachePrediction![d] = sum;
    }

    // Denormalize prediction
    const prediction = new Float64Array(this.outputDim);
    this.denormalizePrediction(this.cachePrediction!, prediction);

    // Compute loss if target provided
    let loss = 0;
    if (y) {
      for (let d = 0; d < this.outputDim; d++) {
        const diff = prediction[d] - y[d];
        loss += diff * diff;
      }
      loss = loss / (2 * this.outputDim);

      // Add L2 regularization
      loss += this.computeL2Regularization();
    }

    return { prediction, loss };
  }

  // ============================================================================
  // Backward Pass
  // ============================================================================

  /**
   * Full backward pass - computes gradients for all parameters
   *
   * @param y - Target values
   * @param prediction - Model prediction (denormalized)
   * @param sampleWeight - Weight for this sample (for outlier downweighting)
   * @returns Gradient norm
   */
  private backward(
    y: Float64Array,
    prediction: Float64Array,
    sampleWeight: number,
  ): number {
    const { embeddingDim, numBlocks } = this.config;
    let gradientNormSq = 0;

    // Normalize target for gradient computation
    const normalizedY = this.bufferPool.acquire(this.outputDim);
    const normalizedPred = this.bufferPool.acquire(this.outputDim);
    this.normalizeOutput(y, normalizedY);

    for (let d = 0; d < this.outputDim; d++) {
      normalizedPred[d] = this.cachePrediction![d];
    }

    // Output layer gradient: dL/dPred = (pred - y) / outputDim
    const dPred = this.bufferPool.acquire(this.outputDim);
    for (let d = 0; d < this.outputDim; d++) {
      dPred[d] = sampleWeight * (normalizedPred[d] - normalizedY[d]) /
        this.outputDim;
    }

    // dL/dOutputW, dL/dOutputB
    this.gradOutputW!.fill(0);
    this.gradOutputB!.fill(0);

    for (let i = 0; i < embeddingDim; i++) {
      for (let d = 0; d < this.outputDim; d++) {
        this.gradOutputW![i * this.outputDim + d] = this.cachePooledOutput![i] *
          dPred[d];
        gradientNormSq += this.gradOutputW![i * this.outputDim + d] ** 2;
      }
    }

    for (let d = 0; d < this.outputDim; d++) {
      this.gradOutputB![d] = dPred[d];
      gradientNormSq += this.gradOutputB![d] ** 2;
    }

    // dL/dPooledOutput
    const dPooledOutput = this.bufferPool.acquire(embeddingDim);
    for (let i = 0; i < embeddingDim; i++) {
      let grad = 0;
      for (let d = 0; d < this.outputDim; d++) {
        grad += this.outputW![i * this.outputDim + d] * dPred[d];
      }
      dPooledOutput[i] = grad;
    }

    // dL/dPoolW and backprop through attention pooling
    this.gradPoolW!.fill(0);
    const lastBlockOutput = this.cacheBlockOutputs![numBlocks - 1];
    const dLastBlockOutput = this.bufferPool.acquire(
      this.seqLen * embeddingDim,
    );
    dLastBlockOutput.fill(0);

    // Gradient through attention pooling
    for (let t = 0; t < this.seqLen; t++) {
      const w = this.cachePoolingWeights![t];

      // Gradient w.r.t. hidden states
      for (let d = 0; d < embeddingDim; d++) {
        dLastBlockOutput[t * embeddingDim + d] += w * dPooledOutput[d];
      }

      // Gradient w.r.t. attention weights (softmax backward)
      let dScore = 0;
      for (let d = 0; d < embeddingDim; d++) {
        dScore += lastBlockOutput[t * embeddingDim + d] * dPooledOutput[d];
      }

      // Softmax jacobian contribution
      for (let t2 = 0; t2 < this.seqLen; t2++) {
        const w2 = this.cachePoolingWeights![t2];
        const kronecker = t === t2 ? 1 : 0;
        const softmaxGrad = w * (kronecker - w2);

        for (let d = 0; d < embeddingDim; d++) {
          this.gradPoolW![d] += softmaxGrad *
            lastBlockOutput[t * embeddingDim + d] *
            (lastBlockOutput[t2 * embeddingDim + d] -
              this.cachePooledOutput![d]);
        }
      }
    }

    // Backprop through transformer blocks (reverse order)
    let dBlockOutput = dLastBlockOutput;

    for (let b = numBlocks - 1; b >= 0; b--) {
      // Through FFN residual
      const dFFNOutput = this.bufferPool.acquire(this.seqLen * embeddingDim);
      const dAttentionResidual = this.bufferPool.acquire(
        this.seqLen * embeddingDim,
      );

      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        dFFNOutput[i] = dBlockOutput[i];
        dAttentionResidual[i] = dBlockOutput[i];
      }

      // Through FFN
      const dLN2Output = this.bufferPool.acquire(this.seqLen * embeddingDim);
      this.ffnBackward(dFFNOutput, b, dLN2Output);

      // Through LayerNorm 2
      const dAttentionOut = this.bufferPool.acquire(this.seqLen * embeddingDim);
      const dLNGamma2 = this.bufferPool.acquire(embeddingDim);
      const dLNBeta2 = this.bufferPool.acquire(embeddingDim);
      dLNGamma2.fill(0);
      dLNBeta2.fill(0);

      this.layerNormBackward(
        dLN2Output,
        this.cacheLN2Norm![b],
        this.cacheLN2Var![b],
        this.lnGamma2![b],
        this.seqLen,
        embeddingDim,
        dAttentionOut,
        dLNGamma2,
        dLNBeta2,
      );

      // Update layer norm gradients
      this.updateAdamParam(`lnGamma2_${b}`, this.lnGamma2![b], dLNGamma2);
      this.updateAdamParam(`lnBeta2_${b}`, this.lnBeta2![b], dLNBeta2);

      // Add residual gradient
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        dAttentionOut[i] += dAttentionResidual[i];
      }

      // Through attention residual
      const dAttention = this.bufferPool.acquire(this.seqLen * embeddingDim);
      const dBlockInput = this.bufferPool.acquire(this.seqLen * embeddingDim);

      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        dAttention[i] = dAttentionOut[i];
        dBlockInput[i] = dAttentionOut[i];
      }

      // Through attention
      const dLN1Output = this.bufferPool.acquire(this.seqLen * embeddingDim);
      this.attentionBackward(dAttention, b, dLN1Output);

      // Through LayerNorm 1
      const dLN1Input = this.bufferPool.acquire(this.seqLen * embeddingDim);
      const dLNGamma1 = this.bufferPool.acquire(embeddingDim);
      const dLNBeta1 = this.bufferPool.acquire(embeddingDim);
      dLNGamma1.fill(0);
      dLNBeta1.fill(0);

      this.layerNormBackward(
        dLN1Output,
        this.cacheLN1Norm![b],
        this.cacheLN1Var![b],
        this.lnGamma1![b],
        this.seqLen,
        embeddingDim,
        dLN1Input,
        dLNGamma1,
        dLNBeta1,
      );

      // Update layer norm gradients
      this.updateAdamParam(`lnGamma1_${b}`, this.lnGamma1![b], dLNGamma1);
      this.updateAdamParam(`lnBeta1_${b}`, this.lnBeta1![b], dLNBeta1);

      // Add residual gradient
      for (let i = 0; i < this.seqLen * embeddingDim; i++) {
        dBlockInput[i] += dLN1Input[i];
      }

      // Prepare for next block
      if (b > 0) {
        this.bufferPool.release(dBlockOutput);
        dBlockOutput = dBlockInput;
      }

      // Release buffers
      this.bufferPool.release(dFFNOutput);
      this.bufferPool.release(dAttentionResidual);
      this.bufferPool.release(dLN2Output);
      this.bufferPool.release(dAttentionOut);
      this.bufferPool.release(dAttention);
      this.bufferPool.release(dLN1Output);
      this.bufferPool.release(dLNGamma2);
      this.bufferPool.release(dLNBeta2);
      this.bufferPool.release(dLNGamma1);
      this.bufferPool.release(dLNBeta1);

      if (b > 0) {
        this.bufferPool.release(dLN1Input);
      }
    }

    // Update output layer
    this.updateAdamParam("outputW", this.outputW!, this.gradOutputW!);
    this.updateAdamParam("outputB", this.outputB!, this.gradOutputB!);
    this.updateAdamParam("poolW", this.poolW!, this.gradPoolW!);

    // Release buffers
    this.bufferPool.release(normalizedY);
    this.bufferPool.release(normalizedPred);
    this.bufferPool.release(dPred);
    this.bufferPool.release(dPooledOutput);

    return Math.sqrt(gradientNormSq);
  }

  /**
   * FFN backward pass
   */
  private ffnBackward(
    dOutput: Float64Array,
    blockIdx: number,
    dInput: Float64Array,
  ): void {
    const { embeddingDim } = this.config;
    const seqLen = this.seqLen;

    // dL/dW2, dL/dB2
    const dW2 = this.bufferPool.acquire(this.ffnDim * embeddingDim);
    const dB2 = this.bufferPool.acquire(embeddingDim);
    dW2.fill(0);
    dB2.fill(0);

    this.matmulAT(
      this.cacheFFNHidden![blockIdx],
      dOutput,
      dW2,
      this.ffnDim,
      seqLen,
      embeddingDim,
    );

    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < embeddingDim; d++) {
        dB2[d] += dOutput[i * embeddingDim + d];
      }
    }

    // dL/dHidden
    const dHidden = this.bufferPool.acquire(seqLen * this.ffnDim);
    this.matmul(
      dOutput,
      this.ffnW2![blockIdx],
      dHidden,
      seqLen,
      embeddingDim,
      this.ffnDim,
    );

    // Through GELU
    const dPreGelu = this.bufferPool.acquire(seqLen * this.ffnDim);
    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < this.ffnDim; d++) {
        const idx = i * this.ffnDim + d;
        // Need pre-activation value for GELU derivative
        // Approximate from cached hidden (post-GELU)
        const postGelu = this.cacheFFNHidden![blockIdx][idx];
        // Inverse GELU approximation (not exact, but works for training)
        const preGelu = postGelu; // Simplified - use stored pre-activation if needed
        dPreGelu[idx] = dHidden[idx] * this.geluDerivative(preGelu);
      }
    }

    // dL/dW1, dL/dB1
    const dW1 = this.bufferPool.acquire(embeddingDim * this.ffnDim);
    const dB1 = this.bufferPool.acquire(this.ffnDim);
    dW1.fill(0);
    dB1.fill(0);

    this.matmulAT(
      this.cacheBlockLN2![blockIdx],
      dPreGelu,
      dW1,
      embeddingDim,
      seqLen,
      this.ffnDim,
    );

    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < this.ffnDim; d++) {
        dB1[d] += dPreGelu[i * this.ffnDim + d];
      }
    }

    // dL/dInput
    this.matmul(
      dPreGelu,
      this.ffnW1![blockIdx],
      dInput,
      seqLen,
      this.ffnDim,
      embeddingDim,
    );

    // Update weights
    this.updateAdamParam(`ffnW1_${blockIdx}`, this.ffnW1![blockIdx], dW1);
    this.updateAdamParam(`ffnB1_${blockIdx}`, this.ffnB1![blockIdx], dB1);
    this.updateAdamParam(`ffnW2_${blockIdx}`, this.ffnW2![blockIdx], dW2);
    this.updateAdamParam(`ffnB2_${blockIdx}`, this.ffnB2![blockIdx], dB2);

    this.bufferPool.release(dW2);
    this.bufferPool.release(dB2);
    this.bufferPool.release(dHidden);
    this.bufferPool.release(dPreGelu);
    this.bufferPool.release(dW1);
    this.bufferPool.release(dB1);
  }

  /**
   * Attention backward pass (simplified)
   */
  private attentionBackward(
    dOutput: Float64Array,
    blockIdx: number,
    dInput: Float64Array,
  ): void {
    const { embeddingDim, numHeads } = this.config;
    const headDim = this.headDim;
    const seqLen = this.seqLen;

    // dL/dOW, dL/dOB
    const dOW = this.bufferPool.acquire(embeddingDim * embeddingDim);
    const dOB = this.bufferPool.acquire(embeddingDim);
    dOW.fill(0);
    dOB.fill(0);

    // Reconstruct attention output before output projection
    const attOut = this.bufferPool.acquire(seqLen * embeddingDim);
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scoreOffset = h * seqLen * seqLen;

      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum +=
              this.cacheAttScores![blockIdx][scoreOffset + i * seqLen + j] *
              this.cacheV![blockIdx][j * embeddingDim + headOffset + d];
          }
          attOut[i * embeddingDim + headOffset + d] = sum;
        }
      }
    }

    this.matmulAT(attOut, dOutput, dOW, embeddingDim, seqLen, embeddingDim);

    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < embeddingDim; d++) {
        dOB[d] += dOutput[i * embeddingDim + d];
      }
    }

    // dL/dAttOut
    const dAttOut = this.bufferPool.acquire(seqLen * embeddingDim);
    this.matmul(
      dOutput,
      this.attOW![blockIdx],
      dAttOut,
      seqLen,
      embeddingDim,
      embeddingDim,
    );

    // Gradient w.r.t Q, K, V (simplified)
    const dQ = this.bufferPool.acquire(seqLen * embeddingDim);
    const dK = this.bufferPool.acquire(seqLen * embeddingDim);
    const dV = this.bufferPool.acquire(seqLen * embeddingDim);
    dQ.fill(0);
    dK.fill(0);
    dV.fill(0);

    const scale = 1.0 / Math.sqrt(headDim);

    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scoreOffset = h * seqLen * seqLen;

      // dV = A^T @ dOut_head
      for (let j = 0; j < seqLen; j++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let i = 0; i < seqLen; i++) {
            sum +=
              this.cacheAttScores![blockIdx][scoreOffset + i * seqLen + j] *
              dAttOut[i * embeddingDim + headOffset + d];
          }
          dV[j * embeddingDim + headOffset + d] += sum;
        }
      }

      // dA = dOut_head @ V^T
      const dA = this.bufferPool.acquire(seqLen * seqLen);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let sum = 0;
          for (let d = 0; d < headDim; d++) {
            sum += dAttOut[i * embeddingDim + headOffset + d] *
              this.cacheV![blockIdx][j * embeddingDim + headOffset + d];
          }
          dA[i * seqLen + j] = sum;
        }
      }

      // Softmax backward
      const dScores = this.bufferPool.acquire(seqLen * seqLen);
      for (let i = 0; i < seqLen; i++) {
        let dotProduct = 0;
        for (let j = 0; j < seqLen; j++) {
          dotProduct += dA[i * seqLen + j] *
            this.cacheAttScores![blockIdx][scoreOffset + i * seqLen + j];
        }
        for (let j = 0; j < seqLen; j++) {
          const a =
            this.cacheAttScores![blockIdx][scoreOffset + i * seqLen + j];
          dScores[i * seqLen + j] = a * (dA[i * seqLen + j] - dotProduct) *
            scale;
        }
      }

      // dQ = dScores @ K_head
      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += dScores[i * seqLen + j] *
              this.cacheK![blockIdx][j * embeddingDim + headOffset + d];
          }
          dQ[i * embeddingDim + headOffset + d] += sum;
        }
      }

      // dK = dScores^T @ Q_head
      for (let j = 0; j < seqLen; j++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let i = 0; i < seqLen; i++) {
            sum += dScores[i * seqLen + j] *
              this.cacheQ![blockIdx][i * embeddingDim + headOffset + d];
          }
          dK[j * embeddingDim + headOffset + d] += sum;
        }
      }

      this.bufferPool.release(dA);
      this.bufferPool.release(dScores);
    }

    // Gradient w.r.t projection weights
    const dQW = this.bufferPool.acquire(embeddingDim * embeddingDim);
    const dKW = this.bufferPool.acquire(embeddingDim * embeddingDim);
    const dVW = this.bufferPool.acquire(embeddingDim * embeddingDim);

    this.matmulAT(
      this.cacheBlockLN1![blockIdx],
      dQ,
      dQW,
      embeddingDim,
      seqLen,
      embeddingDim,
    );
    this.matmulAT(
      this.cacheBlockLN1![blockIdx],
      dK,
      dKW,
      embeddingDim,
      seqLen,
      embeddingDim,
    );
    this.matmulAT(
      this.cacheBlockLN1![blockIdx],
      dV,
      dVW,
      embeddingDim,
      seqLen,
      embeddingDim,
    );

    // dL/dInput (through Q, K, V projections)
    dInput.fill(0);
    const dInputQ = this.bufferPool.acquire(seqLen * embeddingDim);
    const dInputK = this.bufferPool.acquire(seqLen * embeddingDim);
    const dInputV = this.bufferPool.acquire(seqLen * embeddingDim);

    this.matmul(
      dQ,
      this.attQW![blockIdx],
      dInputQ,
      seqLen,
      embeddingDim,
      embeddingDim,
    );
    this.matmul(
      dK,
      this.attKW![blockIdx],
      dInputK,
      seqLen,
      embeddingDim,
      embeddingDim,
    );
    this.matmul(
      dV,
      this.attVW![blockIdx],
      dInputV,
      seqLen,
      embeddingDim,
      embeddingDim,
    );

    for (let i = 0; i < seqLen * embeddingDim; i++) {
      dInput[i] = dInputQ[i] + dInputK[i] + dInputV[i];
    }

    // Update weights
    this.updateAdamParam(`attQW_${blockIdx}`, this.attQW![blockIdx], dQW);
    this.updateAdamParam(`attKW_${blockIdx}`, this.attKW![blockIdx], dKW);
    this.updateAdamParam(`attVW_${blockIdx}`, this.attVW![blockIdx], dVW);
    this.updateAdamParam(`attOW_${blockIdx}`, this.attOW![blockIdx], dOW);
    this.updateAdamParam(`attOB_${blockIdx}`, this.attOB![blockIdx], dOB);

    // Release buffers
    this.bufferPool.release(dOW);
    this.bufferPool.release(dOB);
    this.bufferPool.release(attOut);
    this.bufferPool.release(dAttOut);
    this.bufferPool.release(dQ);
    this.bufferPool.release(dK);
    this.bufferPool.release(dV);
    this.bufferPool.release(dQW);
    this.bufferPool.release(dKW);
    this.bufferPool.release(dVW);
    this.bufferPool.release(dInputQ);
    this.bufferPool.release(dInputK);
    this.bufferPool.release(dInputV);
  }

  // ============================================================================
  // Adam Optimizer
  // ============================================================================

  /**
   * Update a parameter using Adam optimizer
   * Formulas:
   *   m = β₁m + (1-β₁)g
   *   v = β₂v + (1-β₂)g²
   *   m̂ = m / (1 - β₁^t)
   *   v̂ = v / (1 - β₂^t)
   *   θ = θ - η · m̂ / (√v̂ + ε)
   *
   * @param name - Parameter name
   * @param param - Parameter array
   * @param grad - Gradient array
   */
  private updateAdamParam(
    name: string,
    param: Float64Array,
    grad: Float64Array,
  ): void {
    const m = this.adamM.get(name);
    const v = this.adamV.get(name);

    if (!m || !v) return;

    const { beta1, beta2, epsilon, regularizationStrength } = this.config;
    const lr = this.getEffectiveLearningRate();
    const t = this.updateCount + 1;

    // Bias correction terms
    const biasCorrection1 = 1 - Math.pow(beta1, t);
    const biasCorrection2 = 1 - Math.pow(beta2, t);

    for (let i = 0; i < param.length; i++) {
      // Add L2 regularization to gradient
      const g = grad[i] + regularizationStrength * param[i];

      // Update biased first moment
      m[i] = beta1 * m[i] + (1 - beta1) * g;

      // Update biased second moment
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;

      // Compute bias-corrected estimates
      const mHat = m[i] / biasCorrection1;
      const vHat = v[i] / biasCorrection2;

      // Update parameter
      param[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  /**
   * Get effective learning rate with warmup and cosine decay
   * Formula:
   *   warmup: lr * step / warmupSteps
   *   decay: lr * 0.5 * (1 + cos(π * (step - warmup) / (total - warmup)))
   *
   * @returns Effective learning rate
   */
  private getEffectiveLearningRate(): number {
    const { learningRate, warmupSteps, totalSteps } = this.config;
    const step = this.updateCount;

    if (step < warmupSteps) {
      // Linear warmup
      return learningRate * (step + 1) / warmupSteps;
    } else {
      // Cosine decay
      const progress = (step - warmupSteps) /
        Math.max(1, totalSteps - warmupSteps);
      return learningRate * 0.5 *
        (1 + Math.cos(Math.PI * Math.min(progress, 1)));
    }
  }

  // ============================================================================
  // Normalization (Welford's Algorithm)
  // ============================================================================

  /**
   * Update running statistics using Welford's online algorithm
   * Formulas:
   *   δ = x - μ
   *   μ_new = μ + δ/n
   *   M₂_new = M₂ + δ(x - μ_new)
   *   σ² = M₂ / (n - 1)
   *
   * @param x - New input sample [inputDim]
   * @param y - New output sample [outputDim]
   */
  private updateNormalizationStats(x: Float64Array, y: Float64Array): void {
    const n = this.sampleCount + 1;

    // Update input stats
    for (let i = 0; i < this.inputDim; i++) {
      const delta = x[i] - this.inputMean![i];
      this.inputMean![i] += delta / n;
      const delta2 = x[i] - this.inputMean![i];
      this.inputM2![i] += delta * delta2;
    }

    // Update output stats
    for (let i = 0; i < this.outputDim; i++) {
      const delta = y[i] - this.outputMean![i];
      this.outputMean![i] += delta / n;
      const delta2 = y[i] - this.outputMean![i];
      this.outputM2![i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation from Welford's M2 statistic
   */
  private getStd(m2: Float64Array, count: number): Float64Array {
    const std = new Float64Array(m2.length);
    const variance = count > 1 ? count - 1 : 1;
    for (let i = 0; i < m2.length; i++) {
      std[i] = Math.sqrt(m2[i] / variance + this.config.epsilon);
    }
    return std;
  }

  /**
   * Normalize input sequence
   * Formula: x̃ = (x - μ) / (σ + ε)
   */
  private normalizeInput(input: Float64Array, output: Float64Array): void {
    const std = this.getStd(this.inputM2!, Math.max(1, this.sampleCount));

    for (let t = 0; t < this.seqLen; t++) {
      for (let i = 0; i < this.inputDim; i++) {
        const idx = t * this.inputDim + i;
        output[idx] = (input[idx] - this.inputMean![i]) / std[i];
      }
    }
  }

  /**
   * Normalize output (for training)
   */
  private normalizeOutput(
    output: Float64Array,
    normalized: Float64Array,
  ): void {
    const std = this.getStd(this.outputM2!, Math.max(1, this.sampleCount));

    for (let i = 0; i < this.outputDim; i++) {
      normalized[i] = (output[i] - this.outputMean![i]) / std[i];
    }
  }

  /**
   * Denormalize prediction
   * Formula: y = ŷ · σ + μ
   */
  private denormalizePrediction(
    normalized: Float64Array,
    output: Float64Array,
  ): void {
    const std = this.getStd(this.outputM2!, Math.max(1, this.sampleCount));

    for (let i = 0; i < this.outputDim; i++) {
      output[i] = normalized[i] * std[i] + this.outputMean![i];
    }
  }

  // ============================================================================
  // ADWIN Drift Detection
  // ============================================================================

  /**
   * ADWIN drift detection algorithm
   * Detects concept drift by monitoring error distribution changes
   *
   * @param error - Current prediction error
   * @returns Whether drift was detected
   */
  private detectDrift(error: number): boolean {
    this.adwinWindow.push(error);

    // Minimum window size
    if (this.adwinWindow.length < 10) {
      return false;
    }

    // Limit window size
    const maxWindowSize = 1000;
    if (this.adwinWindow.length > maxWindowSize) {
      this.adwinWindow.shift();
    }

    const n = this.adwinWindow.length;
    const delta = this.config.adwinDelta;

    // Try different split points
    for (
      let splitPoint = Math.floor(n * 0.25);
      splitPoint < Math.floor(n * 0.75);
      splitPoint++
    ) {
      const n0 = splitPoint;
      const n1 = n - splitPoint;

      if (n0 < 5 || n1 < 5) continue;

      // Compute means of both windows
      let sum0 = 0, sum1 = 0;
      for (let i = 0; i < splitPoint; i++) {
        sum0 += this.adwinWindow[i];
      }
      for (let i = splitPoint; i < n; i++) {
        sum1 += this.adwinWindow[i];
      }

      const mean0 = sum0 / n0;
      const mean1 = sum1 / n1;

      // Hoeffding bound
      const m = 1 / (1 / n0 + 1 / n1);
      const epsilonCut = Math.sqrt((1 / (2 * m)) * Math.log(4 / delta));

      if (Math.abs(mean0 - mean1) > epsilonCut) {
        // Drift detected - shrink window
        this.adwinWindow = this.adwinWindow.slice(splitPoint);
        this.driftCount++;
        return true;
      }
    }

    return false;
  }

  // ============================================================================
  // Outlier Detection
  // ============================================================================

  /**
   * Detect outliers using residual z-score
   * Formula: r = (y - ŷ) / σ; outlier if |r| > threshold
   *
   * @param y - Target values
   * @param prediction - Model predictions
   * @returns Whether sample is outlier and weight to apply
   */
  private detectOutlier(
    y: Float64Array,
    prediction: Float64Array,
  ): { isOutlier: boolean; weight: number } {
    const std = this.getStd(this.outputM2!, Math.max(1, this.sampleCount));
    let maxResidual = 0;

    for (let i = 0; i < this.outputDim; i++) {
      const residual = Math.abs((y[i] - prediction[i]) / std[i]);
      if (residual > maxResidual) {
        maxResidual = residual;
      }
    }

    const isOutlier = maxResidual > this.config.outlierThreshold;
    const weight = isOutlier ? 0.1 : 1.0;

    return { isOutlier, weight };
  }

  // ============================================================================
  // L2 Regularization
  // ============================================================================

  /**
   * Compute L2 regularization loss
   * Formula: L_reg = (λ/2) Σ ||W||²
   */
  private computeL2Regularization(): number {
    let l2 = 0;
    const lambda = this.config.regularizationStrength;

    // Sum squared weights from all layers
    const addL2 = (arr: Float64Array | null): void => {
      if (!arr) return;
      for (let i = 0; i < arr.length; i++) {
        l2 += arr[i] * arr[i];
      }
    };

    // Temporal conv
    for (let s = 0; s < this.config.temporalScales.length; s++) {
      addL2(this.temporalConvW![s]);
    }

    // Attention and FFN per block
    for (let b = 0; b < this.config.numBlocks; b++) {
      addL2(this.attQW![b]);
      addL2(this.attKW![b]);
      addL2(this.attVW![b]);
      addL2(this.attOW![b]);
      addL2(this.ffnW1![b]);
      addL2(this.ffnW2![b]);
    }

    // Output
    addL2(this.outputW);
    addL2(this.poolW);

    return 0.5 * lambda * l2;
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Perform incremental online learning step
   *
   * @param data - Training data with input and output coordinates
   * @returns Fit result with loss and convergence info
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
   *   yCoordinates: [[10], [11], [12]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Validate input
    if (!xCoordinates || !yCoordinates || xCoordinates.length === 0) {
      throw new Error("Input data cannot be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error("xCoordinates and yCoordinates must have same length");
    }

    const seqLen = xCoordinates.length;
    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;

    // Initialize on first call
    if (!this.isInitialized) {
      this.initialize(inputDim, outputDim, seqLen);
    }

    // Convert to Float64Array
    const x = new Float64Array(seqLen * inputDim);
    const y = new Float64Array(outputDim);

    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < inputDim; i++) {
        x[t * inputDim + i] = xCoordinates[t][i];
      }
    }

    // Use last timestep target for regression
    for (let i = 0; i < outputDim; i++) {
      y[i] = yCoordinates[seqLen - 1][i];
    }

    // Update normalization stats (use mean of sequence for input stats)
    const meanX = new Float64Array(inputDim);
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < inputDim; i++) {
        meanX[i] += x[t * inputDim + i] / seqLen;
      }
    }
    this.updateNormalizationStats(meanX, y);

    // Store input sequence for prediction
    for (
      let i = 0;
      i < Math.min(x.length, this.lastInputSequence!.length);
      i++
    ) {
      this.lastInputSequence![i] = x[i];
    }

    // Forward pass
    const { prediction, loss } = this.forward(x, y);

    // Outlier detection
    const { isOutlier, weight } = this.detectOutlier(y, prediction);

    // Backward pass
    const gradientNorm = this.backward(y, prediction, weight);

    // Update count
    this.updateCount++;
    this.sampleCount++;

    // Update running loss
    this.runningLossSum += loss;
    this.runningLossCount++;

    // Update prediction variance
    for (let i = 0; i < this.outputDim; i++) {
      const diff = prediction[i] - y[i];
      const oldVar = this.predictionVariance![i];
      this.predictionVariance![i] = 0.99 * oldVar + 0.01 * diff * diff;
    }

    // Drift detection
    const error = loss;
    const driftDetected = this.detectDrift(error);

    // Reset stats on drift
    if (driftDetected) {
      this.runningLossSum = loss;
      this.runningLossCount = 1;
    }

    // Check convergence
    const avgLoss = this.runningLossSum / this.runningLossCount;
    this.converged = avgLoss < this.config.convergenceThreshold;

    return {
      loss,
      gradientNorm,
      effectiveLearningRate: this.getEffectiveLearningRate(),
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future timesteps
   *
   * @param futureSteps - Number of future steps to predict
   * @returns Prediction result with uncertainty bounds
   *
   * @example
   * ```typescript
   * const predictions = model.predict(10);
   * for (const pred of predictions.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, SE: ${pred.standardError}`);
   * }
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.isInitialized || this.sampleCount < 1) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const outputStd = this.getStd(
      this.outputM2!,
      Math.max(1, this.sampleCount),
    );

    // Generate predictions autoregressively
    const currentInput = new Float64Array(this.lastInputSequence!);

    for (let step = 0; step < futureSteps; step++) {
      // Forward pass
      const { prediction } = this.forward(currentInput);

      // Compute standard error with uncertainty growth
      const standardError: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const uncertaintyFactor = 1 + 0.1 * step; // Uncertainty grows with horizon

      for (let i = 0; i < this.outputDim; i++) {
        const se = Math.sqrt(this.predictionVariance![i]) * uncertaintyFactor;
        standardError.push(se);
        lowerBound.push(prediction[i] - 1.96 * se);
        upperBound.push(prediction[i] + 1.96 * se);
      }

      predictions.push({
        predicted: Array.from(prediction),
        lowerBound,
        upperBound,
        standardError,
      });

      // Shift input and add prediction for autoregressive step
      if (step < futureSteps - 1) {
        // Shift sequence
        for (let t = 0; t < this.seqLen - 1; t++) {
          for (let i = 0; i < this.inputDim; i++) {
            currentInput[t * this.inputDim + i] =
              currentInput[(t + 1) * this.inputDim + i];
          }
        }

        // Add prediction as new input (use first outputDim features if dimensions match)
        const lastT = this.seqLen - 1;
        for (let i = 0; i < Math.min(this.inputDim, this.outputDim); i++) {
          currentInput[lastT * this.inputDim + i] = prediction[i];
        }
      }
    }

    // Compute accuracy: 1 / (1 + avgLoss)
    const avgLoss = this.runningLossCount > 0
      ? this.runningLossSum / this.runningLossCount
      : 1;
    const accuracy = 1 / (1 + avgLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get summary of model configuration and state
   *
   * @returns Model summary object
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}`);
   * console.log(`Accuracy: ${summary.accuracy}`);
   * ```
   */
  getModelSummary(): ModelSummary {
    let totalParameters = 0;

    if (this.isInitialized) {
      // Count parameters
      const countParams = (arr: Float64Array | null): number =>
        arr ? arr.length : 0;

      // Temporal conv
      for (let s = 0; s < this.config.temporalScales.length; s++) {
        totalParameters += countParams(this.temporalConvW![s]);
        totalParameters += countParams(this.temporalConvB![s]);
        totalParameters += countParams(this.scaleEmb![s]);
      }

      // Fusion
      totalParameters += countParams(this.fusionGateW);
      totalParameters += countParams(this.fusionGateB);
      totalParameters += countParams(this.fusionQueryW);
      totalParameters += countParams(this.fusionKeyW);
      totalParameters += countParams(this.fusionValueW);

      // Per block
      for (let b = 0; b < this.config.numBlocks; b++) {
        totalParameters += countParams(this.attQW![b]);
        totalParameters += countParams(this.attKW![b]);
        totalParameters += countParams(this.attVW![b]);
        totalParameters += countParams(this.attOW![b]);
        totalParameters += countParams(this.attOB![b]);
        totalParameters += countParams(this.ffnW1![b]);
        totalParameters += countParams(this.ffnB1![b]);
        totalParameters += countParams(this.ffnW2![b]);
        totalParameters += countParams(this.ffnB2![b]);
        totalParameters += countParams(this.lnGamma1![b]);
        totalParameters += countParams(this.lnBeta1![b]);
        totalParameters += countParams(this.lnGamma2![b]);
        totalParameters += countParams(this.lnBeta2![b]);
      }

      // Output
      totalParameters += countParams(this.outputW);
      totalParameters += countParams(this.outputB);
      totalParameters += countParams(this.poolW);
    }

    const avgLoss = this.runningLossCount > 0
      ? this.runningLossSum / this.runningLossCount
      : 1;

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
      accuracy: 1 / (1 + avgLoss),
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Get all model weights and optimizer state
   *
   * @returns Weight information object
   */
  getWeights(): WeightInfo {
    const toArray = (arr: Float64Array | null): number[] =>
      arr ? Array.from(arr) : [];

    const toArray2D = (arrs: Float64Array[] | null): number[][] =>
      arrs ? arrs.map((a) => Array.from(a)) : [];

    return {
      temporalConvWeights: this.temporalConvW
        ? this.temporalConvW.map((w) => [Array.from(w)])
        : [],
      scaleEmbeddings: this.scaleEmb
        ? this.scaleEmb.map((e) => [Array.from(e)])
        : [],
      positionalEncoding: this.posEnc ? [[toArray(this.posEnc)]] : [],
      fusionWeights: [
        [toArray(this.fusionGateW)],
        [toArray(this.fusionQueryW)],
        [toArray(this.fusionKeyW)],
        [toArray(this.fusionValueW)],
      ],
      attentionWeights: this.attQW
        ? [
          toArray2D(this.attQW),
          toArray2D(this.attKW),
          toArray2D(this.attVW),
          toArray2D(this.attOW),
        ]
        : [],
      ffnWeights: this.ffnW1
        ? [
          toArray2D(this.ffnW1),
          toArray2D(this.ffnW2),
        ]
        : [],
      layerNormParams: this.lnGamma1
        ? [
          toArray2D(this.lnGamma1),
          toArray2D(this.lnBeta1),
          toArray2D(this.lnGamma2),
          toArray2D(this.lnBeta2),
        ]
        : [],
      outputWeights: [
        [toArray(this.outputW)],
        [toArray(this.outputB)],
        [toArray(this.poolW)],
      ],
      firstMoment: Array.from(this.adamM.entries()).map((
        [_, v],
      ) => [Array.from(v)]),
      secondMoment: Array.from(this.adamV.entries()).map((
        [_, v],
      ) => [Array.from(v)]),
      updateCount: this.updateCount,
    };
  }

  /**
   * Get normalization statistics
   *
   * @returns Normalization stats from Welford's algorithm
   */
  getNormalizationStats(): NormalizationStats {
    return {
      inputMean: this.inputMean ? Array.from(this.inputMean) : [],
      inputStd: this.inputM2
        ? Array.from(this.getStd(this.inputM2, Math.max(1, this.sampleCount)))
        : [],
      outputMean: this.outputMean ? Array.from(this.outputMean) : [],
      outputStd: this.outputM2
        ? Array.from(this.getStd(this.outputM2, Math.max(1, this.sampleCount)))
        : [],
      count: this.sampleCount,
    };
  }

  /**
   * Reset model to initial state
   */
  reset(): void {
    this.isInitialized = false;
    this.sampleCount = 0;
    this.updateCount = 0;
    this.converged = false;
    this.driftCount = 0;
    this.runningLossSum = 0;
    this.runningLossCount = 0;
    this.inputMean = null;
    this.inputM2 = null;
    this.outputMean = null;
    this.outputM2 = null;
    this.adwinWindow = [];
    this.bufferPool.clear();
    this.adamM.clear();
    this.adamV.clear();

    // Null out all weights
    this.temporalConvW = null;
    this.temporalConvB = null;
    this.scaleEmb = null;
    this.posEnc = null;
    this.fusionGateW = null;
    this.fusionGateB = null;
    this.fusionQueryW = null;
    this.fusionKeyW = null;
    this.fusionValueW = null;
    this.attQW = null;
    this.attKW = null;
    this.attVW = null;
    this.attOW = null;
    this.attOB = null;
    this.ffnW1 = null;
    this.ffnB1 = null;
    this.ffnW2 = null;
    this.ffnB2 = null;
    this.lnGamma1 = null;
    this.lnBeta1 = null;
    this.lnGamma2 = null;
    this.lnBeta2 = null;
    this.outputW = null;
    this.outputB = null;
    this.poolW = null;

    // Null out caches
    this.cacheNormX = null;
    this.cacheConvOutputs = null;
    this.cacheFusedOutput = null;
    this.cacheBlockInputs = null;
    this.cacheBlockLN1 = null;
    this.cacheAttentionOut = null;
    this.cacheBlockLN2 = null;
    this.cacheFFNHidden = null;
    this.cacheBlockOutputs = null;
    this.cachePoolingWeights = null;
    this.cachePooledOutput = null;
    this.cachePrediction = null;
    this.cacheAttScores = null;
    this.cacheQ = null;
    this.cacheK = null;
    this.cacheV = null;
    this.cacheLN1Mean = null;
    this.cacheLN1Var = null;
    this.cacheLN1Norm = null;
    this.cacheLN2Mean = null;
    this.cacheLN2Var = null;
    this.cacheLN2Norm = null;
    this.gradOutputW = null;
    this.gradOutputB = null;
    this.gradPoolW = null;
    this.predictionVariance = null;
    this.lastInputSequence = null;
  }

  /**
   * Serialize model state to JSON string
   *
   * @returns JSON string containing all model state
   *
   * @example
   * ```typescript
   * const savedState = model.save();
   * localStorage.setItem('model', savedState);
   * ```
   */
  save(): string {
    const toArray = (arr: Float64Array | null): number[] | null =>
      arr ? Array.from(arr) : null;

    const toArray2D = (arrs: Float64Array[] | null): number[][] | null =>
      arrs ? arrs.map((a) => Array.from(a)) : null;

    const adamMObj: { [key: string]: number[] } = {};
    const adamVObj: { [key: string]: number[] } = {};

    this.adamM.forEach((v, k) => {
      adamMObj[k] = Array.from(v);
    });
    this.adamV.forEach((v, k) => {
      adamVObj[k] = Array.from(v);
    });

    const state: SerializableState = {
      config: { ...this.config },
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      seqLen: this.seqLen,
      isInitialized: this.isInitialized,
      sampleCount: this.sampleCount,
      updateCount: this.updateCount,
      converged: this.converged,
      driftCount: this.driftCount,
      runningLossSum: this.runningLossSum,
      runningLossCount: this.runningLossCount,
      inputMean: toArray(this.inputMean),
      inputM2: toArray(this.inputM2),
      outputMean: toArray(this.outputMean),
      outputM2: toArray(this.outputM2),
      adwinWindow: [...this.adwinWindow],
      weights: {
        temporalConvW: toArray2D(this.temporalConvW) || [],
        temporalConvB: toArray2D(this.temporalConvB) || [],
        scaleEmb: toArray2D(this.scaleEmb) || [],
        posEnc: toArray(this.posEnc) || [],
        fusionGateW: toArray(this.fusionGateW) || [],
        fusionGateB: toArray(this.fusionGateB) || [],
        fusionQueryW: toArray(this.fusionQueryW) || [],
        fusionKeyW: toArray(this.fusionKeyW) || [],
        fusionValueW: toArray(this.fusionValueW) || [],
        attQW: toArray2D(this.attQW) || [],
        attKW: toArray2D(this.attKW) || [],
        attVW: toArray2D(this.attVW) || [],
        attOW: toArray2D(this.attOW) || [],
        attOB: toArray2D(this.attOB) || [],
        ffnW1: toArray2D(this.ffnW1) || [],
        ffnB1: toArray2D(this.ffnB1) || [],
        ffnW2: toArray2D(this.ffnW2) || [],
        ffnB2: toArray2D(this.ffnB2) || [],
        lnGamma1: toArray2D(this.lnGamma1) || [],
        lnBeta1: toArray2D(this.lnBeta1) || [],
        lnGamma2: toArray2D(this.lnGamma2) || [],
        lnBeta2: toArray2D(this.lnBeta2) || [],
        outputW: toArray(this.outputW) || [],
        outputB: toArray(this.outputB) || [],
        poolW: toArray(this.poolW) || [],
      },
      adamState: {
        m: adamMObj,
        v: adamVObj,
      },
      predictionVariance: toArray(this.predictionVariance),
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   *
   * @param w - JSON string containing model state
   *
   * @example
   * ```typescript
   * const savedState = localStorage.getItem('model');
   * if (savedState) {
   *   model.load(savedState);
   * }
   * ```
   */
  load(w: string): void {
    const state: SerializableState = JSON.parse(w);

    const fromArray = (arr: number[] | null): Float64Array | null =>
      arr ? new Float64Array(arr) : null;

    const fromArray2D = (arrs: number[][] | null): Float64Array[] | null =>
      arrs && arrs.length > 0 ? arrs.map((a) => new Float64Array(a)) : null;

    // Restore config (merge with defaults)
    Object.assign(this.config, state.config);
    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;

    // Restore dimensions and state
    this.inputDim = state.inputDim;
    this.outputDim = state.outputDim;
    this.seqLen = state.seqLen;
    this.isInitialized = state.isInitialized;
    this.sampleCount = state.sampleCount;
    this.updateCount = state.updateCount;
    this.converged = state.converged;
    this.driftCount = state.driftCount;
    this.runningLossSum = state.runningLossSum;
    this.runningLossCount = state.runningLossCount;

    // Restore normalization stats
    this.inputMean = fromArray(state.inputMean);
    this.inputM2 = fromArray(state.inputM2);
    this.outputMean = fromArray(state.outputMean);
    this.outputM2 = fromArray(state.outputM2);

    // Restore ADWIN
    this.adwinWindow = [...state.adwinWindow];

    // Restore weights
    const wts = state.weights;
    this.temporalConvW = fromArray2D(wts.temporalConvW);
    this.temporalConvB = fromArray2D(wts.temporalConvB);
    this.scaleEmb = fromArray2D(wts.scaleEmb);
    this.posEnc = fromArray(wts.posEnc);
    this.fusionGateW = fromArray(wts.fusionGateW);
    this.fusionGateB = fromArray(wts.fusionGateB);
    this.fusionQueryW = fromArray(wts.fusionQueryW);
    this.fusionKeyW = fromArray(wts.fusionKeyW);
    this.fusionValueW = fromArray(wts.fusionValueW);
    this.attQW = fromArray2D(wts.attQW);
    this.attKW = fromArray2D(wts.attKW);
    this.attVW = fromArray2D(wts.attVW);
    this.attOW = fromArray2D(wts.attOW);
    this.attOB = fromArray2D(wts.attOB);
    this.ffnW1 = fromArray2D(wts.ffnW1);
    this.ffnB1 = fromArray2D(wts.ffnB1);
    this.ffnW2 = fromArray2D(wts.ffnW2);
    this.ffnB2 = fromArray2D(wts.ffnB2);
    this.lnGamma1 = fromArray2D(wts.lnGamma1);
    this.lnBeta1 = fromArray2D(wts.lnBeta1);
    this.lnGamma2 = fromArray2D(wts.lnGamma2);
    this.lnBeta2 = fromArray2D(wts.lnBeta2);
    this.outputW = fromArray(wts.outputW);
    this.outputB = fromArray(wts.outputB);
    this.poolW = fromArray(wts.poolW);

    // Restore Adam state
    this.adamM.clear();
    this.adamV.clear();
    for (const [key, val] of Object.entries(state.adamState.m)) {
      this.adamM.set(key, new Float64Array(val));
    }
    for (const [key, val] of Object.entries(state.adamState.v)) {
      this.adamV.set(key, new Float64Array(val));
    }

    // Restore prediction variance
    this.predictionVariance = fromArray(state.predictionVariance);

    // Reallocate buffers if initialized
    if (this.isInitialized) {
      this.allocateBuffers();
    }
  }
}
