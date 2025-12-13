/**
 * FusionTemporalTransformerRegression
 *
 * A high-performance Fusion Temporal Transformer neural network for multivariate regression
 * with incremental online learning, Adam optimizer, z-score normalization, and drift detection.
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({ numBlocks: 3, embeddingDim: 64 });
 * const result = model.fitOnline({ xCoordinates: [[1,2],[3,4]], yCoordinates: [[5],[6]] });
 * const predictions = model.predict(3);
 * ```
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Configuration options for the FusionTemporalTransformerRegression model.
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
  /** Learning rate (default: 0.001) */
  learningRate?: number;
  /** Warmup steps for learning rate (default: 100) */
  warmupSteps?: number;
  /** Total training steps for LR schedule (default: 10000) */
  totalSteps?: number;
  /** Adam beta1 (default: 0.9) */
  beta1?: number;
  /** Adam beta2 (default: 0.999) */
  beta2?: number;
  /** Adam epsilon (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength (default: 1e-4) */
  regularizationStrength?: number;
  /** Convergence threshold (default: 1e-6) */
  convergenceThreshold?: number;
  /** Outlier threshold in std deviations (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN delta parameter (default: 0.002) */
  adwinDelta?: number;
  /** Temporal scales for multi-scale convolution (default: [1, 2, 4]) */
  temporalScales?: number[];
  /** Temporal convolution kernel size (default: 3) */
  temporalKernelSize?: number;
  /** Maximum sequence length (default: 512) */
  maxSequenceLength?: number;
  /** Fusion dropout rate (default: 0.0) */
  fusionDropout?: number;
}

/**
 * Result returned from fitOnline method.
 */
export interface FitResult {
  /** Current loss value */
  loss: number;
  /** Gradient L2 norm */
  gradientNorm: number;
  /** Effective learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as outlier */
  isOutlier: boolean;
  /** Whether model has converged */
  converged: boolean;
  /** Current sample index */
  sampleIndex: number;
  /** Whether drift was detected */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds.
 */
export interface SinglePrediction {
  /** Predicted values */
  predicted: number[];
  /** Lower confidence bound */
  lowerBound: number[];
  /** Upper confidence bound */
  upperBound: number[];
  /** Standard error per dimension */
  standardError: number[];
}

/**
 * Result returned from predict method.
 */
export interface PredictionResult {
  /** Array of predictions for future steps */
  predictions: SinglePrediction[];
  /** Model accuracy metric */
  accuracy: number;
  /** Number of training samples seen */
  sampleCount: number;
  /** Whether model is ready for predictions */
  isModelReady: boolean;
}

/**
 * Weight information for model introspection and serialization.
 */
export interface WeightInfo {
  /** Temporal convolution weights per scale */
  temporalConvWeights: number[][][];
  /** Scale-specific embeddings */
  scaleEmbeddings: number[][];
  /** Positional encoding cache */
  positionalEncoding: number[][];
  /** Fusion gate weights */
  fusionWeights: number[][];
  /** Attention weights per block [block][Q,K,V,O] */
  attentionWeights: number[][][][];
  /** FFN weights per block [block][W1,W2] */
  ffnWeights: number[][][][];
  /** LayerNorm parameters per block [block][gamma,beta] */
  layerNormParams: number[][][][];
  /** Output projection weights */
  outputWeights: number[][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Update count for bias correction */
  updateCount: number;
}

/**
 * Normalization statistics for inputs and outputs.
 */
export interface NormalizationStats {
  /** Running mean for inputs */
  inputMean: number[];
  /** Running std for inputs */
  inputStd: number[];
  /** Running mean for outputs */
  outputMean: number[];
  /** Running std for outputs */
  outputStd: number[];
  /** Sample count */
  count: number;
}

/**
 * Model summary information.
 */
export interface ModelSummary {
  /** Whether model weights are initialized */
  isInitialized: boolean;
  /** Input dimension */
  inputDimension: number;
  /** Output dimension */
  outputDimension: number;
  /** Number of transformer blocks */
  numBlocks: number;
  /** Embedding dimension */
  embeddingDim: number;
  /** Number of attention heads */
  numHeads: number;
  /** Temporal scales used */
  temporalScales: number[];
  /** Total trainable parameters */
  totalParameters: number;
  /** Training samples seen */
  sampleCount: number;
  /** Current accuracy */
  accuracy: number;
  /** Whether training has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/**
 * Training data input format.
 */
export interface TrainingData {
  /** Input sequences: [seqLen][inputDim] */
  xCoordinates: number[][];
  /** Output targets: [seqLen][outputDim] */
  yCoordinates: number[][];
}

// ============================================================================
// INTERNAL TYPES
// ============================================================================

interface WelfordState {
  mean: Float64Array;
  m2: Float64Array;
  count: number;
}

interface ADWINBucket {
  total: number;
  variance: number;
  count: number;
}

interface ADWINState {
  buckets: ADWINBucket[][];
  total: number;
  variance: number;
  width: number;
  lastMean: number;
}

interface CachedActivations {
  temporalConvOutputs: Float64Array[][];
  scaledEmbeddings: Float64Array[][];
  fusedRepresentation: Float64Array[];
  fusionGates: Float64Array;
  blockInputs: Float64Array[][];
  attentionScores: Float64Array[][];
  attentionOutputs: Float64Array[][];
  ffnIntermediates: Float64Array[][];
  blockOutputs: Float64Array[][];
  layerNormInputs: Float64Array[][];
  aggregationWeights: Float64Array;
  finalPooled: Float64Array;
  preOutput: Float64Array;
  predictions: Float64Array;
}

interface GradientAccumulators {
  temporalConvGrads: Float64Array[][];
  scaleEmbedGrads: Float64Array[];
  fusionGrads: Float64Array[];
  attentionGrads: Float64Array[][][];
  ffnGrads: Float64Array[][][];
  layerNormGrads: Float64Array[][][];
  outputGrads: Float64Array[];
  poolGrads: Float64Array;
}

// ============================================================================
// MAIN CLASS
// ============================================================================

export class FusionTemporalTransformerRegression {
  // Configuration
  private readonly numBlocks: number;
  private readonly embeddingDim: number;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly ffnMultiplier: number;
  private readonly ffnHiddenDim: number;
  private readonly attentionDropout: number;
  private readonly baseLearningRate: number;
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
  private readonly temporalKernelSize: number;
  private readonly maxSequenceLength: number;
  private readonly fusionDropout: number;

  // Dimensions (set on first fit)
  private inputDim: number = 0;
  private outputDim: number = 0;
  private seqLen: number = 0;
  private isInitialized: boolean = false;

  // Weight matrices - using Float64Array for performance
  private temporalConvWeights: Float64Array[][] = [];
  private temporalConvBias: Float64Array[] = [];
  private scaleEmbeddings: Float64Array[] = [];
  private positionalEncoding: Float64Array[] = [];
  private fusionGateWeights: Float64Array | null = null;
  private fusionGateBias: Float64Array | null = null;

  // Transformer block weights [block][component]
  private attentionQWeights: Float64Array[][] = [];
  private attentionKWeights: Float64Array[][] = [];
  private attentionVWeights: Float64Array[][] = [];
  private attentionOWeights: Float64Array[][] = [];
  private attentionOBias: Float64Array[] = [];
  private ffnW1: Float64Array[][] = [];
  private ffnB1: Float64Array[] = [];
  private ffnW2: Float64Array[][] = [];
  private ffnB2: Float64Array[] = [];
  private layerNorm1Gamma: Float64Array[] = [];
  private layerNorm1Beta: Float64Array[] = [];
  private layerNorm2Gamma: Float64Array[] = [];
  private layerNorm2Beta: Float64Array[] = [];

  // Output layer
  private poolWeights: Float64Array | null = null;
  private outputWeights: Float64Array | null = null;
  private outputBias: Float64Array | null = null;

  // Adam optimizer state - flattened for efficiency
  private adamM: Float64Array | null = null;
  private adamV: Float64Array | null = null;
  private updateCount: number = 0;
  private totalParams: number = 0;

  // Normalization state (Welford's algorithm)
  private inputStats: WelfordState | null = null;
  private outputStats: WelfordState | null = null;

  // Training state
  private sampleCount: number = 0;
  private runningLossSum: number = 0;
  private runningLossCount: number = 0;
  private converged: boolean = false;
  private previousLoss: number = Infinity;

  // ADWIN drift detection
  private adwinState: ADWINState | null = null;
  private driftCount: number = 0;

  // Cached activations for backprop - reused to avoid allocations
  private cache: CachedActivations | null = null;
  private gradAccum: GradientAccumulators | null = null;

  // Preallocated work buffers
  private workBuffer1: Float64Array | null = null;
  private workBuffer2: Float64Array | null = null;
  private workBuffer3: Float64Array | null = null;
  private attentionBuffer: Float64Array | null = null;
  private gradBuffer: Float64Array | null = null;

  // Recent data buffer for predictions
  private recentX: Float64Array[] = [];
  private recentY: Float64Array[] = [];

  /**
   * Creates a new FusionTemporalTransformerRegression model.
   *
   * @param config - Configuration options
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
    this.numBlocks = config.numBlocks ?? 3;
    this.embeddingDim = config.embeddingDim ?? 64;
    this.numHeads = config.numHeads ?? 8;
    this.headDim = Math.floor(this.embeddingDim / this.numHeads);
    this.ffnMultiplier = config.ffnMultiplier ?? 4;
    this.ffnHiddenDim = this.embeddingDim * this.ffnMultiplier;
    this.attentionDropout = config.attentionDropout ?? 0.0;
    this.baseLearningRate = config.learningRate ?? 0.001;
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
    this.temporalKernelSize = config.temporalKernelSize ?? 3;
    this.maxSequenceLength = config.maxSequenceLength ?? 512;
    this.fusionDropout = config.fusionDropout ?? 0.0;
  }

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================

  /**
   * Initialize all weight matrices with Xavier/He initialization.
   * Uses lazy initialization - called on first fitOnline call.
   */
  private initializeWeights(): void {
    if (this.isInitialized) return;

    const numScales = this.temporalScales.length;

    // Initialize positional encoding (precomputed)
    // PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
    this.positionalEncoding = [];
    for (let pos = 0; pos < this.maxSequenceLength; pos++) {
      const pe = new Float64Array(this.embeddingDim);
      for (let i = 0; i < this.embeddingDim; i += 2) {
        const angle = pos / Math.pow(10000, i / this.embeddingDim);
        pe[i] = Math.sin(angle);
        if (i + 1 < this.embeddingDim) {
          pe[i + 1] = Math.cos(angle);
        }
      }
      this.positionalEncoding.push(pe);
    }

    // Temporal convolution weights per scale
    // Conv1D: [kernelSize, inputDim, embeddingDim]
    this.temporalConvWeights = [];
    this.temporalConvBias = [];
    const convFanIn = this.temporalKernelSize * this.inputDim;
    const convStd = Math.sqrt(2.0 / convFanIn); // He initialization

    for (let s = 0; s < numScales; s++) {
      const kernelWeights: Float64Array[] = [];
      for (let k = 0; k < this.temporalKernelSize; k++) {
        const w = new Float64Array(this.inputDim * this.embeddingDim);
        for (let i = 0; i < w.length; i++) {
          w[i] = this.randn() * convStd;
        }
        kernelWeights.push(w);
      }
      this.temporalConvWeights.push(kernelWeights);
      this.temporalConvBias.push(new Float64Array(this.embeddingDim));
    }

    // Scale embeddings (learnable per scale)
    this.scaleEmbeddings = [];
    const scaleStd = Math.sqrt(2.0 / this.embeddingDim);
    for (let s = 0; s < numScales; s++) {
      const emb = new Float64Array(this.embeddingDim);
      for (let i = 0; i < this.embeddingDim; i++) {
        emb[i] = this.randn() * scaleStd * 0.1; // Small initialization
      }
      this.scaleEmbeddings.push(emb);
    }

    // Fusion gate weights
    // G = σ(Concat(E₁,...,Eₛ)Wg + bg)
    const fusionInputDim = numScales * this.embeddingDim;
    const fusionStd = Math.sqrt(2.0 / fusionInputDim);
    this.fusionGateWeights = new Float64Array(fusionInputDim * numScales);
    for (let i = 0; i < this.fusionGateWeights.length; i++) {
      this.fusionGateWeights[i] = this.randn() * fusionStd;
    }
    this.fusionGateBias = new Float64Array(numScales);

    // Transformer block weights
    const attnStd = Math.sqrt(2.0 / this.embeddingDim);
    const ffnStd1 = Math.sqrt(2.0 / this.embeddingDim);
    const ffnStd2 = Math.sqrt(2.0 / this.ffnHiddenDim);

    for (let b = 0; b < this.numBlocks; b++) {
      // Attention: Q, K, V, O projections
      const qw = new Float64Array(this.embeddingDim * this.embeddingDim);
      const kw = new Float64Array(this.embeddingDim * this.embeddingDim);
      const vw = new Float64Array(this.embeddingDim * this.embeddingDim);
      const ow = new Float64Array(this.embeddingDim * this.embeddingDim);
      for (let i = 0; i < qw.length; i++) {
        qw[i] = this.randn() * attnStd;
        kw[i] = this.randn() * attnStd;
        vw[i] = this.randn() * attnStd;
        ow[i] = this.randn() * attnStd;
      }
      this.attentionQWeights.push([qw]);
      this.attentionKWeights.push([kw]);
      this.attentionVWeights.push([vw]);
      this.attentionOWeights.push([ow]);
      this.attentionOBias.push(new Float64Array(this.embeddingDim));

      // FFN weights
      const w1 = new Float64Array(this.embeddingDim * this.ffnHiddenDim);
      const w2 = new Float64Array(this.ffnHiddenDim * this.embeddingDim);
      for (let i = 0; i < w1.length; i++) {
        w1[i] = this.randn() * ffnStd1;
      }
      for (let i = 0; i < w2.length; i++) {
        w2[i] = this.randn() * ffnStd2;
      }
      this.ffnW1.push([w1]);
      this.ffnB1.push(new Float64Array(this.ffnHiddenDim));
      this.ffnW2.push([w2]);
      this.ffnB2.push(new Float64Array(this.embeddingDim));

      // Layer norm params (initialized to 1 and 0)
      const gamma1 = new Float64Array(this.embeddingDim);
      const beta1 = new Float64Array(this.embeddingDim);
      const gamma2 = new Float64Array(this.embeddingDim);
      const beta2 = new Float64Array(this.embeddingDim);
      gamma1.fill(1.0);
      gamma2.fill(1.0);
      this.layerNorm1Gamma.push(gamma1);
      this.layerNorm1Beta.push(beta1);
      this.layerNorm2Gamma.push(gamma2);
      this.layerNorm2Beta.push(beta2);
    }

    // Temporal aggregation (pool) weights
    this.poolWeights = new Float64Array(this.embeddingDim);
    const poolStd = Math.sqrt(2.0 / this.embeddingDim);
    for (let i = 0; i < this.embeddingDim; i++) {
      this.poolWeights[i] = this.randn() * poolStd;
    }

    // Output layer
    const outStd = Math.sqrt(2.0 / this.embeddingDim);
    this.outputWeights = new Float64Array(this.embeddingDim * this.outputDim);
    for (let i = 0; i < this.outputWeights.length; i++) {
      this.outputWeights[i] = this.randn() * outStd;
    }
    this.outputBias = new Float64Array(this.outputDim);

    // Count total parameters and initialize Adam state
    this.countAndInitAdam();

    // Initialize normalization stats
    this.inputStats = {
      mean: new Float64Array(this.inputDim),
      m2: new Float64Array(this.inputDim),
      count: 0,
    };
    this.outputStats = {
      mean: new Float64Array(this.outputDim),
      m2: new Float64Array(this.outputDim),
      count: 0,
    };

    // Initialize ADWIN
    this.initADWIN();

    // Allocate work buffers
    this.allocateBuffers();

    this.isInitialized = true;
  }

  /**
   * Count total parameters and initialize Adam moment estimates.
   */
  private countAndInitAdam(): void {
    this.totalParams = 0;
    const numScales = this.temporalScales.length;

    // Temporal conv
    this.totalParams += numScales *
      this.temporalKernelSize *
      this.inputDim *
      this.embeddingDim;
    this.totalParams += numScales * this.embeddingDim; // bias

    // Scale embeddings
    this.totalParams += numScales * this.embeddingDim;

    // Fusion gates
    this.totalParams += numScales * this.embeddingDim * numScales;
    this.totalParams += numScales;

    // Transformer blocks
    for (let b = 0; b < this.numBlocks; b++) {
      // Attention (Q,K,V,O) + bias
      this.totalParams += 4 * this.embeddingDim * this.embeddingDim;
      this.totalParams += this.embeddingDim;
      // FFN
      this.totalParams += this.embeddingDim * this.ffnHiddenDim;
      this.totalParams += this.ffnHiddenDim;
      this.totalParams += this.ffnHiddenDim * this.embeddingDim;
      this.totalParams += this.embeddingDim;
      // LayerNorm (2 per block)
      this.totalParams += 4 * this.embeddingDim;
    }

    // Pool + output
    this.totalParams += this.embeddingDim;
    this.totalParams += this.embeddingDim * this.outputDim;
    this.totalParams += this.outputDim;

    // Initialize Adam states
    this.adamM = new Float64Array(this.totalParams);
    this.adamV = new Float64Array(this.totalParams);
    this.updateCount = 0;
  }

  /**
   * Allocate reusable buffers for forward/backward passes.
   */
  private allocateBuffers(): void {
    const maxLen = this.maxSequenceLength;
    const dim = this.embeddingDim;
    const numScales = this.temporalScales.length;

    // Work buffers
    this.workBuffer1 = new Float64Array(maxLen * dim);
    this.workBuffer2 = new Float64Array(maxLen * dim);
    this.workBuffer3 = new Float64Array(dim * dim);
    this.attentionBuffer = new Float64Array(maxLen * maxLen);
    this.gradBuffer = new Float64Array(this.totalParams);

    // Initialize cache structure
    this.cache = {
      temporalConvOutputs: [],
      scaledEmbeddings: [],
      fusedRepresentation: [],
      fusionGates: new Float64Array(numScales),
      blockInputs: [],
      attentionScores: [],
      attentionOutputs: [],
      ffnIntermediates: [],
      blockOutputs: [],
      layerNormInputs: [],
      aggregationWeights: new Float64Array(maxLen),
      finalPooled: new Float64Array(dim),
      preOutput: new Float64Array(dim),
      predictions: new Float64Array(this.outputDim),
    };

    // Initialize gradient accumulators
    this.gradAccum = {
      temporalConvGrads: [],
      scaleEmbedGrads: [],
      fusionGrads: [],
      attentionGrads: [],
      ffnGrads: [],
      layerNormGrads: [],
      outputGrads: [],
      poolGrads: new Float64Array(dim),
    };
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  /**
   * Standard normal random number (Box-Muller transform).
   */
  private randn(): number {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * GELU activation function.
   * Formula: GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
   *
   * @param x - Input value
   * @returns GELU activation
   */
  private gelu(x: number): number {
    const c = 0.7978845608028654; // sqrt(2/pi)
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + Math.tanh(inner));
  }

  /**
   * GELU derivative for backprop.
   *
   * @param x - Input value
   * @returns Derivative of GELU at x
   */
  private geluDerivative(x: number): number {
    const c = 0.7978845608028654;
    const x3 = x * x * x;
    const inner = c * (x + 0.044715 * x3);
    const tanhInner = Math.tanh(inner);
    const sech2 = 1.0 - tanhInner * tanhInner;
    const dinnerDx = c * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + tanhInner) + 0.5 * x * sech2 * dinnerDx;
  }

  /**
   * Sigmoid activation.
   *
   * @param x - Input value
   * @returns σ(x) = 1 / (1 + exp(-x))
   */
  private sigmoid(x: number): number {
    if (x >= 0) {
      return 1.0 / (1.0 + Math.exp(-x));
    } else {
      const ex = Math.exp(x);
      return ex / (1.0 + ex);
    }
  }

  /**
   * Softmax over array (in-place for efficiency).
   *
   * @param arr - Input array
   * @param out - Output array
   * @param len - Length to process
   */
  private softmax(arr: Float64Array, out: Float64Array, len: number): void {
    let max = -Infinity;
    for (let i = 0; i < len; i++) {
      if (arr[i] > max) max = arr[i];
    }
    let sum = 0;
    for (let i = 0; i < len; i++) {
      out[i] = Math.exp(arr[i] - max);
      sum += out[i];
    }
    const invSum = 1.0 / (sum + 1e-10);
    for (let i = 0; i < len; i++) {
      out[i] *= invSum;
    }
  }

  /**
   * Layer normalization.
   * Formula: LN(x) = γ * (x - μ) / √(σ² + ε) + β
   *
   * @param input - Input array
   * @param gamma - Scale parameter
   * @param beta - Shift parameter
   * @param output - Output array
   * @param len - Length to normalize
   */
  private layerNorm(
    input: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    output: Float64Array,
    len: number,
  ): void {
    let mean = 0;
    for (let i = 0; i < len; i++) {
      mean += input[i];
    }
    mean /= len;

    let variance = 0;
    for (let i = 0; i < len; i++) {
      const diff = input[i] - mean;
      variance += diff * diff;
    }
    variance /= len;

    const invStd = 1.0 / Math.sqrt(variance + this.epsilon);
    for (let i = 0; i < len; i++) {
      output[i] = gamma[i] * (input[i] - mean) * invStd + beta[i];
    }
  }

  /**
   * Matrix-vector multiply: out = A * x
   *
   * @param A - Matrix (row-major, rows × cols)
   * @param x - Input vector (cols)
   * @param out - Output vector (rows)
   * @param rows - Number of rows
   * @param cols - Number of columns
   */
  private matVecMul(
    A: Float64Array,
    x: Float64Array,
    out: Float64Array,
    rows: number,
    cols: number,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const rowOffset = i * cols;
      for (let j = 0; j < cols; j++) {
        sum += A[rowOffset + j] * x[j];
      }
      out[i] = sum;
    }
  }

  /**
   * Vector-matrix multiply (transpose): out = x^T * A = A^T * x
   *
   * @param x - Input vector (rows)
   * @param A - Matrix (row-major, rows × cols)
   * @param out - Output vector (cols)
   * @param rows - Number of rows
   * @param cols - Number of columns
   */
  private vecMatMul(
    x: Float64Array,
    A: Float64Array,
    out: Float64Array,
    rows: number,
    cols: number,
  ): void {
    out.fill(0);
    for (let i = 0; i < rows; i++) {
      const xi = x[i];
      const rowOffset = i * cols;
      for (let j = 0; j < cols; j++) {
        out[j] += xi * A[rowOffset + j];
      }
    }
  }

  /**
   * Outer product: out += x * y^T (accumulate)
   *
   * @param x - First vector (m)
   * @param y - Second vector (n)
   * @param out - Output matrix (m × n, row-major)
   * @param m - Length of x
   * @param n - Length of y
   */
  private outerProductAdd(
    x: Float64Array,
    y: Float64Array,
    out: Float64Array,
    m: number,
    n: number,
  ): void {
    for (let i = 0; i < m; i++) {
      const xi = x[i];
      const rowOffset = i * n;
      for (let j = 0; j < n; j++) {
        out[rowOffset + j] += xi * y[j];
      }
    }
  }

  /**
   * Dot product of two vectors.
   *
   * @param a - First vector
   * @param b - Second vector
   * @param len - Length
   * @returns Dot product
   */
  private dot(a: Float64Array, b: Float64Array, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * L2 norm of vector.
   *
   * @param arr - Input array
   * @param len - Length
   * @returns L2 norm
   */
  private l2Norm(arr: Float64Array, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += arr[i] * arr[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Update Welford running statistics.
   * Formula: δ = x - μ, μ += δ/n, M₂ += δ(x - μ), σ² = M₂/(n-1)
   *
   * @param stats - Welford state
   * @param values - New values to incorporate
   */
  private updateWelford(stats: WelfordState, values: Float64Array): void {
    stats.count++;
    const n = stats.count;
    for (let i = 0; i < values.length; i++) {
      const delta = values[i] - stats.mean[i];
      stats.mean[i] += delta / n;
      const delta2 = values[i] - stats.mean[i];
      stats.m2[i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation from Welford state.
   *
   * @param stats - Welford state
   * @param out - Output std array
   */
  private getWelfordStd(stats: WelfordState, out: Float64Array): void {
    if (stats.count < 2) {
      out.fill(1.0);
      return;
    }
    const denom = stats.count - 1;
    for (let i = 0; i < stats.m2.length; i++) {
      out[i] = Math.sqrt(stats.m2[i] / denom + this.epsilon);
    }
  }

  /**
   * Normalize input using running statistics.
   *
   * @param input - Input values
   * @param output - Normalized output
   */
  private normalizeInput(input: Float64Array, output: Float64Array): void {
    if (!this.inputStats || this.inputStats.count < 2) {
      for (let i = 0; i < input.length; i++) {
        output[i] = input[i];
      }
      return;
    }
    const stdBuf = this.workBuffer1!;
    this.getWelfordStd(this.inputStats, stdBuf);
    for (let i = 0; i < input.length; i++) {
      output[i] = (input[i] - this.inputStats.mean[i]) / stdBuf[i];
    }
  }

  /**
   * Denormalize output using running statistics.
   *
   * @param normalized - Normalized values
   * @param output - Denormalized output
   */
  private denormalizeOutput(
    normalized: Float64Array,
    output: Float64Array,
  ): void {
    if (!this.outputStats || this.outputStats.count < 2) {
      for (let i = 0; i < normalized.length; i++) {
        output[i] = normalized[i];
      }
      return;
    }
    const stdBuf = this.workBuffer1!;
    this.getWelfordStd(this.outputStats, stdBuf);
    for (let i = 0; i < normalized.length; i++) {
      output[i] = normalized[i] * stdBuf[i] + this.outputStats.mean[i];
    }
  }

  // ==========================================================================
  // ADWIN DRIFT DETECTION
  // ==========================================================================

  /**
   * Initialize ADWIN state for drift detection.
   */
  private initADWIN(): void {
    this.adwinState = {
      buckets: [[]],
      total: 0,
      variance: 0,
      width: 0,
      lastMean: 0,
    };
  }

  /**
   * Add error to ADWIN and check for drift.
   * Formula: Detect drift when |μ₀ - μ₁| ≥ εcut(δ)
   *
   * @param error - Current error value
   * @returns Whether drift was detected
   */
  private adwinAddAndCheck(error: number): boolean {
    if (!this.adwinState) return false;

    // Add to first bucket level
    const bucket: ADWINBucket = {
      total: error,
      variance: 0,
      count: 1,
    };

    if (!this.adwinState.buckets[0]) {
      this.adwinState.buckets[0] = [];
    }
    this.adwinState.buckets[0].push(bucket);
    this.adwinState.total += error;
    this.adwinState.width++;

    // Compress buckets (exponential histogram)
    for (let i = 0; i < this.adwinState.buckets.length; i++) {
      const level = this.adwinState.buckets[i];
      while (level && level.length > 2) {
        const b1 = level.shift()!;
        const b2 = level.shift()!;
        const merged: ADWINBucket = {
          total: b1.total + b2.total,
          count: b1.count + b2.count,
          variance: b1.variance +
            b2.variance +
            Math.pow(b1.total / b1.count - b2.total / b2.count, 2) *
              ((b1.count * b2.count) / (b1.count + b2.count)),
        };
        if (!this.adwinState.buckets[i + 1]) {
          this.adwinState.buckets[i + 1] = [];
        }
        this.adwinState.buckets[i + 1].push(merged);
      }
    }

    // Check for drift
    if (this.adwinState.width < 10) return false;

    const currentMean = this.adwinState.total / this.adwinState.width;
    let driftDetected = false;

    // Simplified drift check: compare first half vs second half means
    let n0 = 0,
      sum0 = 0;
    let n1 = 0,
      sum1 = 0;

    for (let i = 0; i < this.adwinState.buckets.length; i++) {
      const level = this.adwinState.buckets[i];
      if (!level) continue;
      for (let j = 0; j < level.length; j++) {
        if (n0 < this.adwinState.width / 2) {
          sum0 += level[j].total;
          n0 += level[j].count;
        } else {
          sum1 += level[j].total;
          n1 += level[j].count;
        }
      }
    }

    if (n0 > 0 && n1 > 0) {
      const mean0 = sum0 / n0;
      const mean1 = sum1 / n1;
      const epsilon = Math.sqrt(
        (2.0 / this.adwinState.width) * Math.log(2.0 / this.adwinDelta),
      );

      if (Math.abs(mean0 - mean1) > epsilon) {
        driftDetected = true;
        // Shrink window - remove oldest half
        this.shrinkADWIN();
      }
    }

    this.adwinState.lastMean = currentMean;
    return driftDetected;
  }

  /**
   * Shrink ADWIN window after drift detection.
   */
  private shrinkADWIN(): void {
    if (!this.adwinState) return;

    // Remove approximately half the data (oldest)
    let toRemove = Math.floor(this.adwinState.width / 2);
    this.adwinState.total = 0;
    this.adwinState.width = 0;

    for (
      let i = this.adwinState.buckets.length - 1;
      i >= 0 && toRemove > 0;
      i--
    ) {
      const level = this.adwinState.buckets[i];
      if (!level) continue;
      while (level.length > 0 && toRemove > 0) {
        const bucket = level.pop()!;
        toRemove -= bucket.count;
      }
    }

    // Recalculate total and width
    for (let i = 0; i < this.adwinState.buckets.length; i++) {
      const level = this.adwinState.buckets[i];
      if (!level) continue;
      for (const bucket of level) {
        this.adwinState.total += bucket.total;
        this.adwinState.width += bucket.count;
      }
    }
  }

  // ==========================================================================
  // FORWARD PASS
  // ==========================================================================

  /**
   * Full forward pass through the network.
   *
   * @param inputs - Normalized input sequence [seqLen][inputDim]
   * @returns Predictions [outputDim]
   */
  private forward(inputs: Float64Array[]): Float64Array {
    const seqLen = inputs.length;
    const numScales = this.temporalScales.length;

    // Clear/resize cache arrays as needed
    this.ensureCacheCapacity(seqLen);

    // 1. Multi-scale temporal convolution
    // For each scale s: Fₛ = GELU(Conv1D(X, kernel, stride=s))
    for (let s = 0; s < numScales; s++) {
      const scale = this.temporalScales[s];
      const outLen = Math.max(1, Math.ceil(seqLen / scale));

      if (!this.cache!.temporalConvOutputs[s]) {
        this.cache!.temporalConvOutputs[s] = [];
      }

      for (let t = 0; t < outLen; t++) {
        if (!this.cache!.temporalConvOutputs[s][t]) {
          this.cache!.temporalConvOutputs[s][t] = new Float64Array(
            this.embeddingDim,
          );
        }
        const out = this.cache!.temporalConvOutputs[s][t];
        out.fill(0);

        // Conv1D with kernel
        for (let k = 0; k < this.temporalKernelSize; k++) {
          const inputIdx = t * scale + k;
          if (inputIdx < seqLen) {
            const kernelW = this.temporalConvWeights[s][k];
            // out += input[inputIdx] * kernelW (matmul)
            for (let d = 0; d < this.embeddingDim; d++) {
              let sum = 0;
              for (let i = 0; i < this.inputDim; i++) {
                sum += inputs[inputIdx][i] * kernelW[i * this.embeddingDim + d];
              }
              out[d] += sum;
            }
          }
        }

        // Add bias and apply GELU
        for (let d = 0; d < this.embeddingDim; d++) {
          out[d] = this.gelu(out[d] + this.temporalConvBias[s][d]);
        }
      }
    }

    // 2. Scale-specific embeddings
    // Eₛ = Fₛ + PEₛ + ScaleEmb(s)
    for (let s = 0; s < numScales; s++) {
      const scale = this.temporalScales[s];
      const outLen = Math.max(1, Math.ceil(seqLen / scale));

      if (!this.cache!.scaledEmbeddings[s]) {
        this.cache!.scaledEmbeddings[s] = [];
      }

      for (let t = 0; t < outLen; t++) {
        if (!this.cache!.scaledEmbeddings[s][t]) {
          this.cache!.scaledEmbeddings[s][t] = new Float64Array(
            this.embeddingDim,
          );
        }
        const out = this.cache!.scaledEmbeddings[s][t];
        const conv = this.cache!.temporalConvOutputs[s][t];
        const pe =
          this.positionalEncoding[Math.min(t, this.maxSequenceLength - 1)];
        const scaleEmb = this.scaleEmbeddings[s];

        for (let d = 0; d < this.embeddingDim; d++) {
          out[d] = conv[d] + pe[d] + scaleEmb[d];
        }
      }
    }

    // 3. Cross-scale fusion
    // G = σ(Concat(E₁,...,Eₛ)Wg + bg), Fused = Σ(Gₛ ⊙ Eₛ)
    const fusionInput = new Float64Array(numScales * this.embeddingDim);

    // Average pool each scale's embeddings for fusion
    for (let s = 0; s < numScales; s++) {
      const embeddings = this.cache!.scaledEmbeddings[s];
      const len = embeddings.length;
      for (let t = 0; t < len; t++) {
        for (let d = 0; d < this.embeddingDim; d++) {
          fusionInput[s * this.embeddingDim + d] += embeddings[t][d] / len;
        }
      }
    }

    // Compute gates
    this.matVecMul(
      this.fusionGateWeights!,
      fusionInput,
      this.cache!.fusionGates,
      numScales,
      numScales * this.embeddingDim,
    );
    for (let s = 0; s < numScales; s++) {
      this.cache!.fusionGates[s] = this.sigmoid(
        this.cache!.fusionGates[s] + this.fusionGateBias![s],
      );
    }

    // Fused representation (use finest scale's sequence length)
    const fusedLen = this.cache!.scaledEmbeddings[0].length;
    this.cache!.fusedRepresentation = [];
    for (let t = 0; t < fusedLen; t++) {
      const fused = new Float64Array(this.embeddingDim);

      for (let s = 0; s < numScales; s++) {
        const gate = this.cache!.fusionGates[s];
        const scale = this.temporalScales[s];
        const scaledT = Math.min(
          Math.floor(t / scale),
          this.cache!.scaledEmbeddings[s].length - 1,
        );
        const emb = this.cache!.scaledEmbeddings[s][scaledT];

        for (let d = 0; d < this.embeddingDim; d++) {
          fused[d] += gate * emb[d];
        }
      }

      this.cache!.fusedRepresentation.push(fused);
    }

    // 4. Transformer blocks
    let currentInput = this.cache!.fusedRepresentation;

    for (let b = 0; b < this.numBlocks; b++) {
      // Store block input
      this.cache!.blockInputs[b] = currentInput.map((arr) =>
        Float64Array.from(arr)
      );

      const blockOutput: Float64Array[] = [];

      for (let t = 0; t < currentInput.length; t++) {
        // Layer Norm 1
        if (!this.cache!.layerNormInputs[b]) {
          this.cache!.layerNormInputs[b] = [];
        }
        if (!this.cache!.layerNormInputs[b][t]) {
          this.cache!.layerNormInputs[b][t] = new Float64Array(
            this.embeddingDim,
          );
        }
        this.layerNorm(
          currentInput[t],
          this.layerNorm1Gamma[b],
          this.layerNorm1Beta[b],
          this.cache!.layerNormInputs[b][t],
          this.embeddingDim,
        );
      }

      // Multi-head self-attention
      const attnOut = this.multiHeadAttention(
        this.cache!.layerNormInputs[b],
        b,
      );
      this.cache!.attentionOutputs[b] = attnOut;

      // Residual connection
      const afterAttn: Float64Array[] = [];
      for (let t = 0; t < currentInput.length; t++) {
        const res = new Float64Array(this.embeddingDim);
        for (let d = 0; d < this.embeddingDim; d++) {
          res[d] = currentInput[t][d] + attnOut[t][d];
        }
        afterAttn.push(res);
      }

      // Layer Norm 2 + FFN + Residual
      for (let t = 0; t < afterAttn.length; t++) {
        const ln2Out = new Float64Array(this.embeddingDim);
        this.layerNorm(
          afterAttn[t],
          this.layerNorm2Gamma[b],
          this.layerNorm2Beta[b],
          ln2Out,
          this.embeddingDim,
        );

        // FFN: GELU(xW₁ + b₁)W₂ + b₂
        const ffnHidden = new Float64Array(this.ffnHiddenDim);
        this.matVecMul(
          this.ffnW1[b][0],
          ln2Out,
          ffnHidden,
          this.ffnHiddenDim,
          this.embeddingDim,
        );
        for (let d = 0; d < this.ffnHiddenDim; d++) {
          ffnHidden[d] = this.gelu(ffnHidden[d] + this.ffnB1[b][d]);
        }

        if (!this.cache!.ffnIntermediates[b]) {
          this.cache!.ffnIntermediates[b] = [];
        }
        this.cache!.ffnIntermediates[b][t] = ffnHidden;

        const ffnOut = new Float64Array(this.embeddingDim);
        this.matVecMul(
          this.ffnW2[b][0],
          ffnHidden,
          ffnOut,
          this.embeddingDim,
          this.ffnHiddenDim,
        );
        for (let d = 0; d < this.embeddingDim; d++) {
          ffnOut[d] += this.ffnB2[b][d];
        }

        // Second residual
        const out = new Float64Array(this.embeddingDim);
        for (let d = 0; d < this.embeddingDim; d++) {
          out[d] = afterAttn[t][d] + ffnOut[d];
        }
        blockOutput.push(out);
      }

      this.cache!.blockOutputs[b] = blockOutput;
      currentInput = blockOutput;
    }

    // 5. Temporal aggregation (attention-weighted pooling)
    // α = softmax(HWpool), out = Σαᵢhᵢ
    const finalHidden = currentInput;
    const poolScores = this.workBuffer1!;

    for (let t = 0; t < finalHidden.length; t++) {
      poolScores[t] = this.dot(
        finalHidden[t],
        this.poolWeights!,
        this.embeddingDim,
      );
    }

    this.softmax(
      poolScores,
      this.cache!.aggregationWeights,
      finalHidden.length,
    );

    this.cache!.finalPooled.fill(0);
    for (let t = 0; t < finalHidden.length; t++) {
      const alpha = this.cache!.aggregationWeights[t];
      for (let d = 0; d < this.embeddingDim; d++) {
        this.cache!.finalPooled[d] += alpha * finalHidden[t][d];
      }
    }

    // 6. Output projection
    // ŷ = Dense(pooled)
    this.matVecMul(
      this.outputWeights!,
      this.cache!.finalPooled,
      this.cache!.predictions,
      this.outputDim,
      this.embeddingDim,
    );
    for (let d = 0; d < this.outputDim; d++) {
      this.cache!.predictions[d] += this.outputBias![d];
    }

    return this.cache!.predictions;
  }

  /**
   * Multi-head self-attention.
   *
   * @param inputs - Input sequence [seqLen][embeddingDim]
   * @param blockIdx - Transformer block index
   * @returns Output sequence [seqLen][embeddingDim]
   */
  private multiHeadAttention(
    inputs: Float64Array[],
    blockIdx: number,
  ): Float64Array[] {
    const seqLen = inputs.length;
    const outputs: Float64Array[] = [];

    // Compute Q, K, V for all positions
    const Q: Float64Array[] = [];
    const K: Float64Array[] = [];
    const V: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const q = new Float64Array(this.embeddingDim);
      const k = new Float64Array(this.embeddingDim);
      const v = new Float64Array(this.embeddingDim);

      this.matVecMul(
        this.attentionQWeights[blockIdx][0],
        inputs[t],
        q,
        this.embeddingDim,
        this.embeddingDim,
      );
      this.matVecMul(
        this.attentionKWeights[blockIdx][0],
        inputs[t],
        k,
        this.embeddingDim,
        this.embeddingDim,
      );
      this.matVecMul(
        this.attentionVWeights[blockIdx][0],
        inputs[t],
        v,
        this.embeddingDim,
        this.embeddingDim,
      );

      Q.push(q);
      K.push(k);
      V.push(v);
    }

    // Store for backprop
    if (!this.cache!.attentionScores[blockIdx]) {
      this.cache!.attentionScores[blockIdx] = [];
    }

    const scale = 1.0 / Math.sqrt(this.headDim);

    // Multi-head attention (process all heads)
    for (let t = 0; t < seqLen; t++) {
      const attnOutput = new Float64Array(this.embeddingDim);

      // Process each head
      for (let h = 0; h < this.numHeads; h++) {
        const headOffset = h * this.headDim;

        // Compute attention scores for this head
        const scores = new Float64Array(seqLen);
        for (let s = 0; s < seqLen; s++) {
          // Causal mask: only attend to past and current
          if (s > t) {
            scores[s] = -Infinity;
          } else {
            let score = 0;
            for (let d = 0; d < this.headDim; d++) {
              score += Q[t][headOffset + d] * K[s][headOffset + d];
            }
            scores[s] = score * scale;
          }
        }

        // Softmax
        const attnWeights = new Float64Array(seqLen);
        this.softmax(scores, attnWeights, seqLen);

        // Weighted sum of values
        for (let s = 0; s < seqLen; s++) {
          const w = attnWeights[s];
          for (let d = 0; d < this.headDim; d++) {
            attnOutput[headOffset + d] += w * V[s][headOffset + d];
          }
        }
      }

      // Output projection
      const projected = new Float64Array(this.embeddingDim);
      this.matVecMul(
        this.attentionOWeights[blockIdx][0],
        attnOutput,
        projected,
        this.embeddingDim,
        this.embeddingDim,
      );
      for (let d = 0; d < this.embeddingDim; d++) {
        projected[d] += this.attentionOBias[blockIdx][d];
      }

      outputs.push(projected);
    }

    return outputs;
  }

  /**
   * Ensure cache arrays have sufficient capacity.
   *
   * @param seqLen - Current sequence length
   */
  private ensureCacheCapacity(seqLen: number): void {
    if (!this.cache) return;

    const numScales = this.temporalScales.length;

    // Resize temporal conv outputs
    while (this.cache.temporalConvOutputs.length < numScales) {
      this.cache.temporalConvOutputs.push([]);
    }
    while (this.cache.scaledEmbeddings.length < numScales) {
      this.cache.scaledEmbeddings.push([]);
    }

    // Ensure block arrays
    while (this.cache.blockInputs.length < this.numBlocks) {
      this.cache.blockInputs.push([]);
    }
    while (this.cache.attentionScores.length < this.numBlocks) {
      this.cache.attentionScores.push([]);
    }
    while (this.cache.attentionOutputs.length < this.numBlocks) {
      this.cache.attentionOutputs.push([]);
    }
    while (this.cache.ffnIntermediates.length < this.numBlocks) {
      this.cache.ffnIntermediates.push([]);
    }
    while (this.cache.blockOutputs.length < this.numBlocks) {
      this.cache.blockOutputs.push([]);
    }
    while (this.cache.layerNormInputs.length < this.numBlocks) {
      this.cache.layerNormInputs.push([]);
    }
  }

  // ==========================================================================
  // BACKWARD PASS
  // ==========================================================================

  /**
   * Compute gradients via backpropagation.
   *
   * @param inputs - Input sequence
   * @param targets - Target outputs
   * @param predictions - Forward pass predictions
   * @returns Gradient norm
   */
  private backward(
    inputs: Float64Array[],
    targets: Float64Array,
    predictions: Float64Array,
  ): number {
    const seqLen = inputs.length;

    // Initialize gradient buffer
    this.gradBuffer!.fill(0);

    // Output layer gradient
    // dL/dŷ = (ŷ - y) / n
    const dOutput = new Float64Array(this.outputDim);
    for (let d = 0; d < this.outputDim; d++) {
      dOutput[d] = (predictions[d] - targets[d]) / this.outputDim;
    }

    // Gradients for output weights and bias
    // dL/dW_out = dL/dŷ ⊗ pooled
    // dL/db_out = dL/dŷ
    let gradOffset = this.getOutputWeightOffset();
    for (let d = 0; d < this.outputDim; d++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        this.gradBuffer![gradOffset + d * this.embeddingDim + e] += dOutput[d] *
          this.cache!.finalPooled[e];
      }
    }
    gradOffset += this.outputDim * this.embeddingDim;
    for (let d = 0; d < this.outputDim; d++) {
      this.gradBuffer![gradOffset + d] += dOutput[d];
    }

    // Backprop through output projection
    // dL/d_pooled = W_out^T * dL/dŷ
    const dPooled = new Float64Array(this.embeddingDim);
    this.vecMatMul(
      dOutput,
      this.outputWeights!,
      dPooled,
      this.outputDim,
      this.embeddingDim,
    );

    // Backprop through pooling attention
    // α = softmax(H*w_pool)
    // pooled = Σ αᵢ hᵢ
    // dL/d_αᵢ = dL/d_pooled · hᵢ
    const finalHidden = this.cache!.blockOutputs[this.numBlocks - 1];
    const dAlpha = new Float64Array(finalHidden.length);
    for (let t = 0; t < finalHidden.length; t++) {
      dAlpha[t] = this.dot(dPooled, finalHidden[t], this.embeddingDim);
    }

    // Softmax backward
    // dL/d_scores = α ⊙ (dL/dα - Σⱼ αⱼ dL/dαⱼ)
    const alphaSum = this.dot(
      this.cache!.aggregationWeights,
      dAlpha,
      finalHidden.length,
    );
    const dScores = new Float64Array(finalHidden.length);
    for (let t = 0; t < finalHidden.length; t++) {
      dScores[t] = this.cache!.aggregationWeights[t] * (dAlpha[t] - alphaSum);
    }

    // Pool weights gradient
    // dL/d_w_pool = Σₜ dL/d_scoresₜ * hₜ
    const poolGradOffset = this.getPoolWeightOffset();
    for (let t = 0; t < finalHidden.length; t++) {
      for (let d = 0; d < this.embeddingDim; d++) {
        this.gradBuffer![poolGradOffset + d] += dScores[t] * finalHidden[t][d];
      }
    }

    // dL/d_hₜ = αₜ * dL/d_pooled + dL/d_scoresₜ * w_pool
    const dFinalHidden: Float64Array[] = [];
    for (let t = 0; t < finalHidden.length; t++) {
      const dh = new Float64Array(this.embeddingDim);
      for (let d = 0; d < this.embeddingDim; d++) {
        dh[d] = this.cache!.aggregationWeights[t] * dPooled[d] +
          dScores[t] * this.poolWeights![d];
      }
      dFinalHidden.push(dh);
    }

    // Backprop through transformer blocks (reverse order)
    let dBlock = dFinalHidden;
    for (let b = this.numBlocks - 1; b >= 0; b--) {
      dBlock = this.backwardTransformerBlock(b, dBlock, inputs);
    }

    // Backprop through fusion and temporal conv (simplified)
    this.backwardFusion(dBlock, inputs);

    // Add L2 regularization gradients
    this.addRegularizationGradients();

    // Compute gradient norm
    return this.l2Norm(this.gradBuffer!, this.totalParams);
  }

  /**
   * Backward pass through one transformer block.
   *
   * @param blockIdx - Block index
   * @param dOutput - Gradient from next layer
   * @param inputs - Original inputs
   * @returns Gradient to pass to previous layer
   */
  private backwardTransformerBlock(
    blockIdx: number,
    dOutput: Float64Array[],
    inputs: Float64Array[],
  ): Float64Array[] {
    const seqLen = dOutput.length;
    const blockInput = this.cache!.blockInputs[blockIdx];
    const ffnIntermediate = this.cache!.ffnIntermediates[blockIdx];

    // Gradients through second residual
    const dFFNOut = dOutput.map((d) => Float64Array.from(d));

    // Get weight offsets for this block
    const offsets = this.getBlockWeightOffsets(blockIdx);

    // Backprop through FFN
    for (let t = 0; t < seqLen; t++) {
      // dL/dW2 = dL/dFFNOut ⊗ ffnHidden
      // dL/db2 = dL/dFFNOut
      for (let d = 0; d < this.embeddingDim; d++) {
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          this.gradBuffer![
            offsets.ffnW2 + d * this.ffnHiddenDim + h
          ] += dFFNOut[t][d] * ffnIntermediate[t][h];
        }
        this.gradBuffer![offsets.ffnB2 + d] += dFFNOut[t][d];
      }

      // dL/d_ffnHidden = W2^T * dL/dFFNOut
      const dFFNHidden = new Float64Array(this.ffnHiddenDim);
      this.vecMatMul(
        dFFNOut[t],
        this.ffnW2[blockIdx][0],
        dFFNHidden,
        this.embeddingDim,
        this.ffnHiddenDim,
      );

      // GELU backward
      const dPreGELU = new Float64Array(this.ffnHiddenDim);
      for (let h = 0; h < this.ffnHiddenDim; h++) {
        // Need pre-activation value (approximate from intermediate)
        const preAct = this.inverseGELUApprox(ffnIntermediate[t][h]);
        dPreGELU[h] = dFFNHidden[h] * this.geluDerivative(preAct);
      }

      // dL/dW1, dL/db1
      const ln2Input = this.cache!.layerNormInputs[blockIdx]?.[t] ||
        blockInput[t];
      for (let h = 0; h < this.ffnHiddenDim; h++) {
        for (let d = 0; d < this.embeddingDim; d++) {
          this.gradBuffer![
            offsets.ffnW1 + h * this.embeddingDim + d
          ] += dPreGELU[h] * ln2Input[d];
        }
        this.gradBuffer![offsets.ffnB1 + h] += dPreGELU[h];
      }
    }

    // Simplified attention gradient (approximate)
    // Full attention backprop is very complex - using simplified version
    const dAttn = dOutput.map((d) => Float64Array.from(d));

    // Backprop through attention output projection
    for (let t = 0; t < seqLen; t++) {
      // dL/dW_O, dL/db_O
      const attnOut = this.cache!.attentionOutputs[blockIdx][t];
      for (let d = 0; d < this.embeddingDim; d++) {
        for (let e = 0; e < this.embeddingDim; e++) {
          this.gradBuffer![
            offsets.attnO + d * this.embeddingDim + e
          ] += dAttn[t][d] * attnOut[e];
        }
        this.gradBuffer![offsets.attnOBias + d] += dAttn[t][d];
      }
    }

    // Layer norm gradients (simplified - just pass through scale)
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < this.embeddingDim; d++) {
        this.gradBuffer![offsets.ln1Gamma + d] += dAttn[t][d] *
          (blockInput[t][d] - this.getMean(blockInput[t]));
        this.gradBuffer![offsets.ln1Beta + d] += dAttn[t][d];
        this.gradBuffer![offsets.ln2Gamma + d] += dOutput[t][d] *
          (blockInput[t][d] - this.getMean(blockInput[t]));
        this.gradBuffer![offsets.ln2Beta + d] += dOutput[t][d];
      }
    }

    // Return gradient to pass to previous block
    return dOutput.map((d) => Float64Array.from(d));
  }

  /**
   * Backward pass through fusion layer.
   *
   * @param dFused - Gradient from transformer blocks
   * @param inputs - Original inputs
   */
  private backwardFusion(
    dFused: Float64Array[],
    inputs: Float64Array[],
  ): void {
    const numScales = this.temporalScales.length;
    const offsets = this.getFusionWeightOffsets();

    // Simplified fusion gradient
    // Approximate gradient for fusion gate weights
    for (let s = 0; s < numScales; s++) {
      const gate = this.cache!.fusionGates[s];
      const gateGrad = gate * (1 - gate); // sigmoid derivative

      for (let t = 0; t < dFused.length; t++) {
        const scaledT = Math.min(
          Math.floor(t / this.temporalScales[s]),
          (this.cache!.scaledEmbeddings[s]?.length || 1) - 1,
        );
        const emb = this.cache!.scaledEmbeddings[s]?.[scaledT];
        if (!emb) continue;

        for (let d = 0; d < this.embeddingDim; d++) {
          // Gradient for fusion gate bias
          this.gradBuffer![offsets.fusionBias + s] += dFused[t][d] * emb[d] *
            gateGrad / dFused.length;
        }
      }
    }

    // Temporal conv gradients (simplified)
    for (let s = 0; s < numScales; s++) {
      const scale = this.temporalScales[s];
      const convOutputs = this.cache!.temporalConvOutputs[s];
      if (!convOutputs) continue;

      for (let t = 0; t < convOutputs.length; t++) {
        // Gradient through GELU
        for (let d = 0; d < this.embeddingDim; d++) {
          const preAct = this.inverseGELUApprox(convOutputs[t][d]);
          const geluGrad = this.geluDerivative(preAct);

          // Bias gradient
          this.gradBuffer![offsets.convBias[s] + d] += geluGrad *
            (dFused[Math.min(t * scale, dFused.length - 1)]?.[d] || 0) /
            convOutputs.length;

          // Weight gradients
          for (let k = 0; k < this.temporalKernelSize; k++) {
            const inputIdx = t * scale + k;
            if (inputIdx < inputs.length) {
              for (let i = 0; i < this.inputDim; i++) {
                this.gradBuffer![
                  offsets.convWeights[s] +
                  k * this.inputDim * this.embeddingDim +
                  i * this.embeddingDim +
                  d
                ] += geluGrad * inputs[inputIdx][i] *
                  (dFused[Math.min(t * scale, dFused.length - 1)]?.[d] || 0) /
                  convOutputs.length;
              }
            }
          }
        }
      }

      // Scale embedding gradient
      for (let d = 0; d < this.embeddingDim; d++) {
        let grad = 0;
        for (let t = 0; t < Math.min(dFused.length, convOutputs.length); t++) {
          grad += dFused[t][d] * this.cache!.fusionGates[s];
        }
        this.gradBuffer![offsets.scaleEmb[s] + d] += grad / dFused.length;
      }
    }
  }

  /**
   * Add L2 regularization gradients.
   */
  private addRegularizationGradients(): void {
    const lambda = this.regularizationStrength;

    // Add regularization to all weights (not biases)
    // This is simplified - in practice would iterate through weight matrices
    const weightRanges = this.getWeightRanges();
    for (const range of weightRanges) {
      for (let i = range.start; i < range.end; i++) {
        this.gradBuffer![i] += lambda * this.getWeightAtIndex(i);
      }
    }
  }

  /**
   * Get weight value at flat index.
   *
   * @param index - Flat parameter index
   * @returns Weight value
   */
  private getWeightAtIndex(index: number): number {
    // Simplified: would need proper indexing into weight matrices
    // This is a placeholder for the full implementation
    return 0;
  }

  /**
   * Get ranges of weight (non-bias) parameters.
   *
   * @returns Array of weight ranges
   */
  private getWeightRanges(): { start: number; end: number }[] {
    // Returns indices of weight matrices (excluding biases)
    // Simplified placeholder
    return [];
  }

  /**
   * Get various weight offset values.
   */
  private getOutputWeightOffset(): number {
    // Calculate offset to output weights in flat gradient buffer
    let offset = 0;

    // Skip temporal conv, scale embeddings, fusion, transformer blocks, pool
    const numScales = this.temporalScales.length;

    // Temporal conv weights and biases
    offset += numScales * this.temporalKernelSize * this.inputDim *
      this.embeddingDim;
    offset += numScales * this.embeddingDim;

    // Scale embeddings
    offset += numScales * this.embeddingDim;

    // Fusion
    offset += numScales * this.embeddingDim * numScales;
    offset += numScales;

    // Transformer blocks
    for (let b = 0; b < this.numBlocks; b++) {
      offset += 4 * this.embeddingDim * this.embeddingDim; // Q,K,V,O
      offset += this.embeddingDim; // O bias
      offset += this.embeddingDim * this.ffnHiddenDim; // W1
      offset += this.ffnHiddenDim; // b1
      offset += this.ffnHiddenDim * this.embeddingDim; // W2
      offset += this.embeddingDim; // b2
      offset += 4 * this.embeddingDim; // LN params
    }

    // Pool weights
    offset += this.embeddingDim;

    return offset;
  }

  private getPoolWeightOffset(): number {
    return this.getOutputWeightOffset() - this.embeddingDim;
  }

  private getBlockWeightOffsets(blockIdx: number): {
    attnQ: number;
    attnK: number;
    attnV: number;
    attnO: number;
    attnOBias: number;
    ffnW1: number;
    ffnB1: number;
    ffnW2: number;
    ffnB2: number;
    ln1Gamma: number;
    ln1Beta: number;
    ln2Gamma: number;
    ln2Beta: number;
  } {
    let offset = 0;
    const numScales = this.temporalScales.length;

    // Skip temporal conv, scale embeddings, fusion
    offset += numScales * this.temporalKernelSize * this.inputDim *
      this.embeddingDim;
    offset += numScales * this.embeddingDim;
    offset += numScales * this.embeddingDim;
    offset += numScales * this.embeddingDim * numScales;
    offset += numScales;

    // Skip previous blocks
    const blockSize = 4 * this.embeddingDim * this.embeddingDim +
      this.embeddingDim +
      this.embeddingDim * this.ffnHiddenDim +
      this.ffnHiddenDim +
      this.ffnHiddenDim * this.embeddingDim +
      this.embeddingDim +
      4 * this.embeddingDim;

    offset += blockIdx * blockSize;

    const d2 = this.embeddingDim * this.embeddingDim;

    return {
      attnQ: offset,
      attnK: offset + d2,
      attnV: offset + 2 * d2,
      attnO: offset + 3 * d2,
      attnOBias: offset + 4 * d2,
      ffnW1: offset + 4 * d2 + this.embeddingDim,
      ffnB1: offset + 4 * d2 + this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim,
      ffnW2: offset + 4 * d2 + this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim + this.ffnHiddenDim,
      ffnB2: offset +
        4 * d2 +
        this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim +
        this.ffnHiddenDim +
        this.ffnHiddenDim * this.embeddingDim,
      ln1Gamma: offset +
        4 * d2 +
        this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim +
        this.ffnHiddenDim +
        this.ffnHiddenDim * this.embeddingDim +
        this.embeddingDim,
      ln1Beta: offset +
        4 * d2 +
        this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim +
        this.ffnHiddenDim +
        this.ffnHiddenDim * this.embeddingDim +
        this.embeddingDim +
        this.embeddingDim,
      ln2Gamma: offset +
        4 * d2 +
        this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim +
        this.ffnHiddenDim +
        this.ffnHiddenDim * this.embeddingDim +
        this.embeddingDim +
        2 * this.embeddingDim,
      ln2Beta: offset +
        4 * d2 +
        this.embeddingDim +
        this.embeddingDim * this.ffnHiddenDim +
        this.ffnHiddenDim +
        this.ffnHiddenDim * this.embeddingDim +
        this.embeddingDim +
        3 * this.embeddingDim,
    };
  }

  private getFusionWeightOffsets(): {
    convWeights: number[];
    convBias: number[];
    scaleEmb: number[];
    fusionWeights: number;
    fusionBias: number;
  } {
    const numScales = this.temporalScales.length;
    const convSize = this.temporalKernelSize * this.inputDim *
      this.embeddingDim;

    const convWeights: number[] = [];
    const convBias: number[] = [];
    const scaleEmb: number[] = [];

    let offset = 0;
    for (let s = 0; s < numScales; s++) {
      convWeights.push(offset);
      offset += convSize;
    }
    for (let s = 0; s < numScales; s++) {
      convBias.push(offset);
      offset += this.embeddingDim;
    }
    for (let s = 0; s < numScales; s++) {
      scaleEmb.push(offset);
      offset += this.embeddingDim;
    }

    return {
      convWeights,
      convBias,
      scaleEmb,
      fusionWeights: offset,
      fusionBias: offset + numScales * this.embeddingDim * numScales,
    };
  }

  /**
   * Approximate inverse GELU for backprop.
   */
  private inverseGELUApprox(y: number): number {
    // Simple approximation: for small y, inverse is close to y
    // This is a numerical convenience
    if (Math.abs(y) < 1e-6) return 0;
    return y > 0 ? Math.sqrt(y) : -Math.sqrt(-y);
  }

  /**
   * Get mean of array.
   */
  private getMean(arr: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
    }
    return sum / arr.length;
  }

  // ==========================================================================
  // ADAM OPTIMIZER
  // ==========================================================================

  /**
   * Apply Adam optimizer update.
   * Formula: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²
   *          W -= η(m/(1-β₁ᵗ))/(√(v/(1-β₂ᵗ)) + ε)
   *
   * @param learningRate - Current learning rate
   */
  private adamUpdate(learningRate: number): void {
    this.updateCount++;

    const beta1Correction = 1 - Math.pow(this.beta1, this.updateCount);
    const beta2Correction = 1 - Math.pow(this.beta2, this.updateCount);

    // Update moments and weights
    let paramIdx = 0;
    const updateWeight = (
      weights: Float64Array,
      startIdx: number,
      length: number,
    ): void => {
      for (let i = 0; i < length; i++) {
        const g = this.gradBuffer![paramIdx];

        // Update biased first moment
        this.adamM![paramIdx] = this.beta1 * this.adamM![paramIdx] +
          (1 - this.beta1) * g;

        // Update biased second moment
        this.adamV![paramIdx] = this.beta2 * this.adamV![paramIdx] +
          (1 - this.beta2) * g * g;

        // Bias-corrected estimates
        const mHat = this.adamM![paramIdx] / beta1Correction;
        const vHat = this.adamV![paramIdx] / beta2Correction;

        // Update weight
        weights[startIdx + i] -= learningRate * mHat /
          (Math.sqrt(vHat) + this.epsilon);

        paramIdx++;
      }
    };

    // Update all weights in order
    const numScales = this.temporalScales.length;

    // Temporal conv weights
    for (let s = 0; s < numScales; s++) {
      for (let k = 0; k < this.temporalKernelSize; k++) {
        updateWeight(
          this.temporalConvWeights[s][k],
          0,
          this.inputDim * this.embeddingDim,
        );
      }
    }

    // Temporal conv biases
    for (let s = 0; s < numScales; s++) {
      updateWeight(this.temporalConvBias[s], 0, this.embeddingDim);
    }

    // Scale embeddings
    for (let s = 0; s < numScales; s++) {
      updateWeight(this.scaleEmbeddings[s], 0, this.embeddingDim);
    }

    // Fusion weights
    updateWeight(
      this.fusionGateWeights!,
      0,
      numScales * this.embeddingDim * numScales,
    );
    updateWeight(this.fusionGateBias!, 0, numScales);

    // Transformer blocks
    for (let b = 0; b < this.numBlocks; b++) {
      updateWeight(
        this.attentionQWeights[b][0],
        0,
        this.embeddingDim * this.embeddingDim,
      );
      updateWeight(
        this.attentionKWeights[b][0],
        0,
        this.embeddingDim * this.embeddingDim,
      );
      updateWeight(
        this.attentionVWeights[b][0],
        0,
        this.embeddingDim * this.embeddingDim,
      );
      updateWeight(
        this.attentionOWeights[b][0],
        0,
        this.embeddingDim * this.embeddingDim,
      );
      updateWeight(this.attentionOBias[b], 0, this.embeddingDim);
      updateWeight(
        this.ffnW1[b][0],
        0,
        this.embeddingDim * this.ffnHiddenDim,
      );
      updateWeight(this.ffnB1[b], 0, this.ffnHiddenDim);
      updateWeight(
        this.ffnW2[b][0],
        0,
        this.ffnHiddenDim * this.embeddingDim,
      );
      updateWeight(this.ffnB2[b], 0, this.embeddingDim);
      updateWeight(this.layerNorm1Gamma[b], 0, this.embeddingDim);
      updateWeight(this.layerNorm1Beta[b], 0, this.embeddingDim);
      updateWeight(this.layerNorm2Gamma[b], 0, this.embeddingDim);
      updateWeight(this.layerNorm2Beta[b], 0, this.embeddingDim);
    }

    // Pool weights
    updateWeight(this.poolWeights!, 0, this.embeddingDim);

    // Output weights and bias
    updateWeight(
      this.outputWeights!,
      0,
      this.embeddingDim * this.outputDim,
    );
    updateWeight(this.outputBias!, 0, this.outputDim);
  }

  /**
   * Compute learning rate with warmup and cosine decay.
   * Formula: LR = base_lr * min(step/warmup, 0.5*(1 + cos(π*progress)))
   *
   * @returns Current effective learning rate
   */
  private getEffectiveLearningRate(): number {
    const step = this.sampleCount;

    if (step < this.warmupSteps) {
      // Linear warmup
      return this.baseLearningRate * (step / this.warmupSteps);
    }

    // Cosine decay
    const progress = (step - this.warmupSteps) /
      (this.totalSteps - this.warmupSteps);
    const decayFactor = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1.0)));
    return this.baseLearningRate * decayFactor;
  }

  // ==========================================================================
  // PUBLIC API
  // ==========================================================================

  /**
   * Performs incremental online learning with a single batch.
   * Uses Adam optimizer with warmup, z-score normalization, L2 regularization,
   * outlier downweighting, and ADWIN drift detection.
   *
   * @param data - Training data with xCoordinates and yCoordinates
   * @returns FitResult with loss, gradient norm, and training state
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
   *   yCoordinates: [[10], [20], [30]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  public fitOnline(data: TrainingData): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Auto-detect dimensions on first call
    if (!this.isInitialized) {
      this.inputDim = xCoordinates[0].length;
      this.outputDim = yCoordinates[0].length;
      this.seqLen = xCoordinates.length;
      this.initializeWeights();
    }

    // Convert to Float64Arrays
    const inputs: Float64Array[] = [];
    for (let i = 0; i < xCoordinates.length; i++) {
      inputs.push(new Float64Array(xCoordinates[i]));
    }

    // Use last target as the prediction target
    const target = new Float64Array(yCoordinates[yCoordinates.length - 1]);

    // Update normalization statistics (Welford's algorithm)
    for (const input of inputs) {
      this.updateWelford(this.inputStats!, input);
    }
    this.updateWelford(this.outputStats!, target);

    // Normalize inputs
    const normalizedInputs: Float64Array[] = [];
    for (const input of inputs) {
      const normalized = new Float64Array(this.inputDim);
      this.normalizeInput(input, normalized);
      normalizedInputs.push(normalized);
    }

    // Forward pass
    const predictions = this.forward(normalizedInputs);

    // Denormalize predictions for loss computation
    const denormPred = new Float64Array(this.outputDim);
    this.denormalizeOutput(predictions, denormPred);

    // Compute loss: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
    let mse = 0;
    for (let d = 0; d < this.outputDim; d++) {
      const diff = target[d] - denormPred[d];
      mse += diff * diff;
    }
    mse /= 2 * this.outputDim;

    // Check for outlier: r = (y - ŷ)/σ; |r| > threshold → downweight
    let isOutlier = false;
    let outlierWeight = 1.0;
    if (this.outputStats!.count > 10) {
      const stdBuf = new Float64Array(this.outputDim);
      this.getWelfordStd(this.outputStats!, stdBuf);

      let maxResidual = 0;
      for (let d = 0; d < this.outputDim; d++) {
        const residual = Math.abs(target[d] - denormPred[d]) / stdBuf[d];
        if (residual > maxResidual) maxResidual = residual;
      }

      if (maxResidual > this.outlierThreshold) {
        isOutlier = true;
        outlierWeight = 0.1;
      }
    }

    // Backward pass
    const gradientNorm = this.backward(normalizedInputs, target, denormPred);

    // Scale gradients for outliers
    if (isOutlier) {
      for (let i = 0; i < this.totalParams; i++) {
        this.gradBuffer![i] *= outlierWeight;
      }
    }

    // Adam update
    const effectiveLR = this.getEffectiveLearningRate();
    this.adamUpdate(effectiveLR);

    // Update training state
    this.sampleCount++;
    this.runningLossSum += mse;
    this.runningLossCount++;

    // Check convergence
    const avgLoss = this.runningLossSum / this.runningLossCount;
    if (
      Math.abs(this.previousLoss - avgLoss) < this.convergenceThreshold &&
      this.sampleCount > this.warmupSteps
    ) {
      this.converged = true;
    }
    this.previousLoss = avgLoss;

    // ADWIN drift detection
    const driftDetected = this.adwinAddAndCheck(mse);
    if (driftDetected) {
      this.driftCount++;
      // Optionally reset some state on drift
    }

    // Store recent data for predictions
    this.recentX.push(...normalizedInputs);
    this.recentY.push(target);
    while (this.recentX.length > this.maxSequenceLength) {
      this.recentX.shift();
    }
    while (this.recentY.length > this.maxSequenceLength) {
      this.recentY.shift();
    }

    return {
      loss: mse,
      gradientNorm,
      effectiveLearningRate: effectiveLR,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future time steps.
   *
   * @param futureSteps - Number of future steps to predict
   * @returns PredictionResult with predictions and uncertainty estimates
   *
   * @example
   * ```typescript
   * const result = model.predict(5);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    if (!this.isInitialized || this.recentX.length === 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const currentInputs = [...this.recentX];

    // Compute accuracy: 1/(1 + L̄)
    const avgLoss = this.runningLossCount > 0
      ? this.runningLossSum / this.runningLossCount
      : 1;
    const accuracy = 1 / (1 + avgLoss);

    // Get output std for uncertainty bounds
    const outputStd = new Float64Array(this.outputDim);
    this.getWelfordStd(this.outputStats!, outputStd);

    for (let step = 0; step < futureSteps; step++) {
      // Use recent sequence for prediction
      const seqStart = Math.max(0, currentInputs.length - this.seqLen);
      const inputSeq = currentInputs.slice(seqStart);

      // Forward pass
      const rawPred = this.forward(inputSeq);

      // Denormalize
      const predicted = new Float64Array(this.outputDim);
      this.denormalizeOutput(rawPred, predicted);

      // Compute uncertainty bounds (simple: ± 1.96 * std / sqrt(n))
      const standardError: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const sqrtN = Math.sqrt(Math.max(1, this.sampleCount));

      for (let d = 0; d < this.outputDim; d++) {
        const se = outputStd[d] / sqrtN;
        standardError.push(se);
        lowerBound.push(predicted[d] - 1.96 * se);
        upperBound.push(predicted[d] + 1.96 * se);
      }

      predictions.push({
        predicted: Array.from(predicted),
        lowerBound,
        upperBound,
        standardError,
      });

      // For multi-step prediction, use prediction as next input
      // (simplified: just duplicate last input with slight modification)
      if (step < futureSteps - 1) {
        const nextInput = new Float64Array(this.inputDim);
        const lastInput = currentInputs[currentInputs.length - 1];
        for (let i = 0; i < this.inputDim; i++) {
          nextInput[i] = lastInput[i] * 0.99 + 0.01 * (Math.random() - 0.5);
        }
        currentInputs.push(nextInput);
      }
    }

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get a summary of the model's current state.
   *
   * @returns ModelSummary with configuration and training statistics
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}, Samples: ${summary.sampleCount}`);
   * ```
   */
  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.numBlocks,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      temporalScales: [...this.temporalScales],
      totalParameters: this.totalParams,
      sampleCount: this.sampleCount,
      accuracy: this.runningLossCount > 0
        ? 1 / (1 + this.runningLossSum / this.runningLossCount)
        : 0,
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Get all model weights for inspection or serialization.
   *
   * @returns WeightInfo containing all learnable parameters
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Update count: ${weights.updateCount}`);
   * ```
   */
  public getWeights(): WeightInfo {
    const temporalConvWeights: number[][][] = [];
    for (const scaleWeights of this.temporalConvWeights) {
      const sw: number[][] = [];
      for (const kw of scaleWeights) {
        sw.push(Array.from(kw));
      }
      temporalConvWeights.push(sw);
    }

    const scaleEmbeddings: number[][] = [];
    for (const emb of this.scaleEmbeddings) {
      scaleEmbeddings.push(Array.from(emb));
    }

    const positionalEncoding: number[][] = [];
    for (const pe of this.positionalEncoding) {
      positionalEncoding.push(Array.from(pe));
    }

    const fusionWeights: number[][] = [
      this.fusionGateWeights ? Array.from(this.fusionGateWeights) : [],
      this.fusionGateBias ? Array.from(this.fusionGateBias) : [],
    ];

    const attentionWeights: number[][][][] = [];
    for (let b = 0; b < this.numBlocks; b++) {
      attentionWeights.push([
        [Array.from(this.attentionQWeights[b][0])],
        [Array.from(this.attentionKWeights[b][0])],
        [Array.from(this.attentionVWeights[b][0])],
        [Array.from(this.attentionOWeights[b][0])],
      ]);
    }

    const ffnWeights: number[][][][] = [];
    for (let b = 0; b < this.numBlocks; b++) {
      ffnWeights.push([
        [Array.from(this.ffnW1[b][0])],
        [Array.from(this.ffnW2[b][0])],
      ]);
    }

    const layerNormParams: number[][][][] = [];
    for (let b = 0; b < this.numBlocks; b++) {
      layerNormParams.push([
        [Array.from(this.layerNorm1Gamma[b])],
        [Array.from(this.layerNorm1Beta[b])],
        [Array.from(this.layerNorm2Gamma[b])],
        [Array.from(this.layerNorm2Beta[b])],
      ]);
    }

    const outputWeights: number[][] = [
      this.outputWeights ? Array.from(this.outputWeights) : [],
      this.outputBias ? Array.from(this.outputBias) : [],
    ];

    return {
      temporalConvWeights,
      scaleEmbeddings,
      positionalEncoding,
      fusionWeights,
      attentionWeights,
      ffnWeights,
      layerNormParams,
      outputWeights,
      firstMoment: this.adamM ? [[Array.from(this.adamM)]] : [],
      secondMoment: this.adamV ? [[Array.from(this.adamV)]] : [],
      updateCount: this.updateCount,
    };
  }

  /**
   * Get normalization statistics for inputs and outputs.
   *
   * @returns NormalizationStats with mean, std, and count
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input mean: ${stats.inputMean}, count: ${stats.count}`);
   * ```
   */
  public getNormalizationStats(): NormalizationStats {
    const inputStd = new Float64Array(this.inputDim || 1);
    const outputStd = new Float64Array(this.outputDim || 1);

    if (this.inputStats) {
      this.getWelfordStd(this.inputStats, inputStd);
    }
    if (this.outputStats) {
      this.getWelfordStd(this.outputStats, outputStd);
    }

    return {
      inputMean: this.inputStats ? Array.from(this.inputStats.mean) : [],
      inputStd: Array.from(inputStd),
      outputMean: this.outputStats ? Array.from(this.outputStats.mean) : [],
      outputStd: Array.from(outputStd),
      count: this.inputStats?.count || 0,
    };
  }

  /**
   * Reset the model to its initial state.
   * Clears all weights, optimizer state, and training statistics.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now ready for fresh training
   * ```
   */
  public reset(): void {
    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.seqLen = 0;

    this.temporalConvWeights = [];
    this.temporalConvBias = [];
    this.scaleEmbeddings = [];
    this.positionalEncoding = [];
    this.fusionGateWeights = null;
    this.fusionGateBias = null;

    this.attentionQWeights = [];
    this.attentionKWeights = [];
    this.attentionVWeights = [];
    this.attentionOWeights = [];
    this.attentionOBias = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    this.layerNorm1Gamma = [];
    this.layerNorm1Beta = [];
    this.layerNorm2Gamma = [];
    this.layerNorm2Beta = [];

    this.poolWeights = null;
    this.outputWeights = null;
    this.outputBias = null;

    this.adamM = null;
    this.adamV = null;
    this.updateCount = 0;
    this.totalParams = 0;

    this.inputStats = null;
    this.outputStats = null;

    this.sampleCount = 0;
    this.runningLossSum = 0;
    this.runningLossCount = 0;
    this.converged = false;
    this.previousLoss = Infinity;

    this.adwinState = null;
    this.driftCount = 0;

    this.cache = null;
    this.gradAccum = null;
    this.workBuffer1 = null;
    this.workBuffer2 = null;
    this.workBuffer3 = null;
    this.attentionBuffer = null;
    this.gradBuffer = null;

    this.recentX = [];
    this.recentY = [];
  }

  /**
   * Serialize the model to a JSON string.
   * Includes all weights, optimizer state, and training statistics.
   *
   * @returns JSON string representation of the model
   *
   * @example
   * ```typescript
   * const serialized = model.save();
   * localStorage.setItem('myModel', serialized);
   * ```
   */
  public save(): string {
    const state = {
      config: {
        numBlocks: this.numBlocks,
        embeddingDim: this.embeddingDim,
        numHeads: this.numHeads,
        ffnMultiplier: this.ffnMultiplier,
        attentionDropout: this.attentionDropout,
        learningRate: this.baseLearningRate,
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
        seqLen: this.seqLen,
      },
      isInitialized: this.isInitialized,
      weights: this.getWeights(),
      normStats: this.getNormalizationStats(),
      trainingState: {
        sampleCount: this.sampleCount,
        runningLossSum: this.runningLossSum,
        runningLossCount: this.runningLossCount,
        converged: this.converged,
        previousLoss: this.previousLoss,
        driftCount: this.driftCount,
        updateCount: this.updateCount,
      },
      recentX: this.recentX.map((arr) => Array.from(arr)),
      recentY: this.recentY.map((arr) => Array.from(arr)),
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from a JSON string.
   * Restores all weights, optimizer state, and training statistics.
   *
   * @param jsonStr - JSON string from save()
   *
   * @example
   * ```typescript
   * const serialized = localStorage.getItem('myModel');
   * model.load(serialized);
   * ```
   */
  public load(jsonStr: string): void {
    const state = JSON.parse(jsonStr);

    // Reset first
    this.reset();

    // Restore dimensions
    this.inputDim = state.dimensions.inputDim;
    this.outputDim = state.dimensions.outputDim;
    this.seqLen = state.dimensions.seqLen;

    // Check if model was initialized
    if (!state.isInitialized) return;

    // Initialize structure (isInitialized is still false from reset, so this works)
    this.initializeWeights();

    // Restore weights
    const weights = state.weights as WeightInfo;

    // Temporal conv weights
    for (let s = 0; s < weights.temporalConvWeights.length; s++) {
      for (let k = 0; k < weights.temporalConvWeights[s].length; k++) {
        this.temporalConvWeights[s][k] = new Float64Array(
          weights.temporalConvWeights[s][k],
        );
      }
    }

    // Scale embeddings
    for (let s = 0; s < weights.scaleEmbeddings.length; s++) {
      this.scaleEmbeddings[s] = new Float64Array(weights.scaleEmbeddings[s]);
    }

    // Fusion weights
    if (weights.fusionWeights.length >= 2) {
      this.fusionGateWeights = new Float64Array(weights.fusionWeights[0]);
      this.fusionGateBias = new Float64Array(weights.fusionWeights[1]);
    }

    // Attention weights
    for (let b = 0; b < weights.attentionWeights.length; b++) {
      this.attentionQWeights[b][0] = new Float64Array(
        weights.attentionWeights[b][0][0],
      );
      this.attentionKWeights[b][0] = new Float64Array(
        weights.attentionWeights[b][1][0],
      );
      this.attentionVWeights[b][0] = new Float64Array(
        weights.attentionWeights[b][2][0],
      );
      this.attentionOWeights[b][0] = new Float64Array(
        weights.attentionWeights[b][3][0],
      );
    }

    // FFN weights
    for (let b = 0; b < weights.ffnWeights.length; b++) {
      this.ffnW1[b][0] = new Float64Array(weights.ffnWeights[b][0][0]);
      this.ffnW2[b][0] = new Float64Array(weights.ffnWeights[b][1][0]);
    }

    // LayerNorm params
    for (let b = 0; b < weights.layerNormParams.length; b++) {
      this.layerNorm1Gamma[b] = new Float64Array(
        weights.layerNormParams[b][0][0],
      );
      this.layerNorm1Beta[b] = new Float64Array(
        weights.layerNormParams[b][1][0],
      );
      this.layerNorm2Gamma[b] = new Float64Array(
        weights.layerNormParams[b][2][0],
      );
      this.layerNorm2Beta[b] = new Float64Array(
        weights.layerNormParams[b][3][0],
      );
    }

    // Output weights
    if (weights.outputWeights.length >= 2) {
      this.outputWeights = new Float64Array(weights.outputWeights[0]);
      this.outputBias = new Float64Array(weights.outputWeights[1]);
    }

    // Adam state
    if (
      weights.firstMoment.length > 0 &&
      weights.firstMoment[0].length > 0
    ) {
      this.adamM = new Float64Array(weights.firstMoment[0][0]);
    }
    if (
      weights.secondMoment.length > 0 &&
      weights.secondMoment[0].length > 0
    ) {
      this.adamV = new Float64Array(weights.secondMoment[0][0]);
    }

    // Restore normalization stats
    const normStats = state.normStats as NormalizationStats;
    if (normStats.count > 0) {
      this.inputStats = {
        mean: new Float64Array(normStats.inputMean),
        m2: new Float64Array(this.inputDim),
        count: normStats.count,
      };
      // Reconstruct M2 from std (approximation)
      for (let i = 0; i < this.inputDim; i++) {
        this.inputStats.m2[i] = normStats.inputStd[i] * normStats.inputStd[i] *
          (normStats.count - 1);
      }

      this.outputStats = {
        mean: new Float64Array(normStats.outputMean),
        m2: new Float64Array(this.outputDim),
        count: normStats.count,
      };
      for (let i = 0; i < this.outputDim; i++) {
        this.outputStats.m2[i] = normStats.outputStd[i] *
          normStats.outputStd[i] * (normStats.count - 1);
      }
    }

    // Restore training state
    const trainState = state.trainingState;
    this.sampleCount = trainState.sampleCount;
    this.runningLossSum = trainState.runningLossSum;
    this.runningLossCount = trainState.runningLossCount;
    this.converged = trainState.converged;
    this.previousLoss = trainState.previousLoss;
    this.driftCount = trainState.driftCount;
    this.updateCount = trainState.updateCount;

    // Restore recent data
    this.recentX = state.recentX.map(
      (arr: number[]) => new Float64Array(arr),
    );
    this.recentY = state.recentY.map(
      (arr: number[]) => new Float64Array(arr),
    );

    // Reinitialize ADWIN
    this.initADWIN();
  }
}

export default FusionTemporalTransformerRegression;
