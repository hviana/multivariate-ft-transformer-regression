/**
 * @fileoverview Fusion Temporal Transformer for Multivariate Regression
 * High-performance implementation with online learning, Adam optimizer,
 * and z-score normalization.
 */

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Result from a single online training step
 */
interface FitResult {
  loss: number;
  gradientNorm: number;
  effectiveLearningRate: number;
  isOutlier: boolean;
  converged: boolean;
  sampleIndex: number;
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty estimates
 */
interface SinglePrediction {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
}

/**
 * Result from prediction
 */
interface PredictionResult {
  predictions: SinglePrediction[];
  accuracy: number;
  sampleCount: number;
  isModelReady: boolean;
}

/**
 * Complete model weight information
 */
interface WeightInfo {
  temporalConvWeights: number[][][];
  scaleEmbeddings: number[][][];
  positionalEncoding: number[][];
  fusionWeights: number[][][];
  attentionWeights: number[][][];
  ffnWeights: number[][][];
  layerNormParams: number[][][];
  outputWeights: number[][][];
  firstMoment: number[][][];
  secondMoment: number[][][];
  updateCount: number;
}

/**
 * Normalization statistics for z-score transform
 */
interface NormalizationStats {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
  count: number;
}

/**
 * Summary of model configuration and state
 */
interface ModelSummary {
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
 * Configuration options for the model
 */
interface FusionTemporalConfig {
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

/**
 * Internal serialization state
 */
interface SerializedState {
  config: Required<FusionTemporalConfig>;
  isInitialized: boolean;
  inputDim: number;
  outputDim: number;
  seqLen: number;
  weights: number[];
  weightOffsets: Record<string, number>;
  inputMean: number[];
  inputM2: number[];
  outputMean: number[];
  outputM2: number[];
  normCount: number;
  sampleCount: number;
  totalLoss: number;
  updateCount: number;
  converged: boolean;
  driftCount: number;
  adwinWindow: number[];
  firstMoment: number[];
  secondMoment: number[];
  inputHistory: number[][];
}

// ============================================================================
// Buffer Pool for Memory Efficiency
// ============================================================================

/**
 * Object pool for Float64Array reuse to minimize GC pressure
 */
class BufferPool {
  private readonly _pools: Map<number, Float64Array[]> = new Map();
  private readonly _inUse: Set<Float64Array> = new Set();

  /**
   * Acquire a buffer of specified size
   * @param size - Required buffer size
   * @returns Reused or new Float64Array
   */
  acquire(size: number): Float64Array {
    const pool = this._pools.get(size);
    if (pool && pool.length > 0) {
      const buf = pool.pop()!;
      this._inUse.add(buf);
      // Zero out for clean slate
      buf.fill(0);
      return buf;
    }
    const newBuf = new Float64Array(size);
    this._inUse.add(newBuf);
    return newBuf;
  }

  /**
   * Release a buffer back to the pool
   * @param buf - Buffer to release
   */
  release(buf: Float64Array): void {
    if (!this._inUse.has(buf)) return;
    this._inUse.delete(buf);
    const size = buf.length;
    let pool = this._pools.get(size);
    if (!pool) {
      pool = [];
      this._pools.set(size, pool);
    }
    if (pool.length < 32) {
      pool.push(buf);
    }
  }

  /**
   * Clear all pools
   */
  clear(): void {
    this._pools.clear();
    this._inUse.clear();
  }
}

// ============================================================================
// Main Class Implementation
// ============================================================================

/**
 * Fusion Temporal Transformer for Multivariate Regression
 *
 * A high-performance transformer architecture with multi-scale temporal
 * processing, designed for online learning on time series data.
 *
 * Architecture:
 * 1. Multi-scale temporal convolution extracts features at different resolutions
 * 2. Scale-specific embeddings with learnable scale tokens
 * 3. Cross-scale gated attention fusion
 * 4. Transformer blocks with temporal self-attention
 * 5. Attention-weighted temporal aggregation
 * 6. Dense output projection
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({
 *   numBlocks: 3,
 *   embeddingDim: 64,
 *   numHeads: 8,
 *   learningRate: 0.001
 * });
 *
 * // Incremental online training
 * for (const batch of dataStream) {
 *   const result = model.fitOnline({
 *     xCoordinates: batch.x,
 *     yCoordinates: batch.y
 *   });
 *   console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
 * }
 *
 * // Make predictions
 * const predictions = model.predict(5);
 * ```
 */
export class FusionTemporalTransformerRegression {
  // -------------------------------------------------------------------------
  // Configuration (immutable after construction)
  // -------------------------------------------------------------------------
  private readonly _numBlocks: number;
  private readonly _embeddingDim: number;
  private readonly _numHeads: number;
  private readonly _ffnMultiplier: number;
  private readonly _attentionDropout: number;
  private readonly _learningRate: number;
  private readonly _warmupSteps: number;
  private readonly _totalSteps: number;
  private readonly _beta1: number;
  private readonly _beta2: number;
  private readonly _epsilon: number;
  private readonly _regularizationStrength: number;
  private readonly _convergenceThreshold: number;
  private readonly _outlierThreshold: number;
  private readonly _adwinDelta: number;
  private readonly _temporalScales: readonly number[];
  private readonly _temporalKernelSize: number;
  private readonly _maxSequenceLength: number;
  private readonly _fusionDropout: number;

  // Derived configuration
  private readonly _headDim: number;
  private readonly _ffnDim: number;
  private readonly _numScales: number;

  // -------------------------------------------------------------------------
  // Model State
  // -------------------------------------------------------------------------
  private _isInitialized: boolean = false;
  private _inputDim: number = 0;
  private _outputDim: number = 0;
  private _seqLen: number = 0;

  // -------------------------------------------------------------------------
  // Weight Storage (flat Float64Array for cache efficiency)
  // -------------------------------------------------------------------------
  private _weights!: Float64Array;
  private _gradients!: Float64Array;
  private _firstMoment!: Float64Array;
  private _secondMoment!: Float64Array;

  // Weight offsets for indexing into flat arrays
  private _weightOffsets: Record<string, number> = {};
  private _totalParams: number = 0;

  // -------------------------------------------------------------------------
  // Normalization Statistics (Welford's algorithm)
  // -------------------------------------------------------------------------
  private _inputMean!: Float64Array;
  private _inputM2!: Float64Array;
  private _outputMean!: Float64Array;
  private _outputM2!: Float64Array;
  private _normCount: number = 0;

  // -------------------------------------------------------------------------
  // Training State
  // -------------------------------------------------------------------------
  private _sampleCount: number = 0;
  private _totalLoss: number = 0;
  private _updateCount: number = 0;
  private _converged: boolean = false;
  private _driftCount: number = 0;
  private _prevLoss: number = Infinity;

  // -------------------------------------------------------------------------
  // ADWIN Drift Detection
  // -------------------------------------------------------------------------
  private _adwinWindow: number[] = [];

  // -------------------------------------------------------------------------
  // Input History for Prediction
  // -------------------------------------------------------------------------
  private _inputHistory: Float64Array[] = [];

  // -------------------------------------------------------------------------
  // Preallocated Buffers for Forward/Backward Pass
  // -------------------------------------------------------------------------
  private readonly _bufferPool: BufferPool = new BufferPool();
  private _activationCache: Map<string, Float64Array> = new Map();

  // Reusable computation buffers
  private _tempBuffer1!: Float64Array;
  private _tempBuffer2!: Float64Array;
  private _tempBuffer3!: Float64Array;
  private _attentionScores!: Float64Array;
  private _softmaxBuffer!: Float64Array;

  /**
   * Creates a new Fusion Temporal Transformer model
   *
   * @param config - Model configuration options
   *
   * @example
   * ```typescript
   * const model = new FusionTemporalTransformerRegression({
   *   numBlocks: 4,
   *   embeddingDim: 128,
   *   numHeads: 8,
   *   temporalScales: [1, 2, 4, 8]
   * });
   * ```
   */
  constructor(config: FusionTemporalConfig = {}) {
    // Apply defaults
    this._numBlocks = config.numBlocks ?? 3;
    this._embeddingDim = config.embeddingDim ?? 64;
    this._numHeads = config.numHeads ?? 8;
    this._ffnMultiplier = config.ffnMultiplier ?? 4;
    this._attentionDropout = config.attentionDropout ?? 0.0;
    this._learningRate = config.learningRate ?? 0.001;
    this._warmupSteps = config.warmupSteps ?? 100;
    this._totalSteps = config.totalSteps ?? 10000;
    this._beta1 = config.beta1 ?? 0.9;
    this._beta2 = config.beta2 ?? 0.999;
    this._epsilon = config.epsilon ?? 1e-8;
    this._regularizationStrength = config.regularizationStrength ?? 1e-4;
    this._convergenceThreshold = config.convergenceThreshold ?? 1e-6;
    this._outlierThreshold = config.outlierThreshold ?? 3.0;
    this._adwinDelta = config.adwinDelta ?? 0.002;
    this._temporalScales = Object.freeze([
      ...(config.temporalScales ?? [1, 2, 4]),
    ]);
    this._temporalKernelSize = config.temporalKernelSize ?? 3;
    this._maxSequenceLength = config.maxSequenceLength ?? 512;
    this._fusionDropout = config.fusionDropout ?? 0.0;

    // Derived values
    this._headDim = Math.floor(this._embeddingDim / this._numHeads);
    this._ffnDim = this._embeddingDim * this._ffnMultiplier;
    this._numScales = this._temporalScales.length;

    // Validate configuration
    if (this._embeddingDim % this._numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this._embeddingDim}) must be divisible by numHeads (${this._numHeads})`,
      );
    }
  }

  // =========================================================================
  // Private Initialization Methods
  // =========================================================================

  /**
   * Initialize model weights and buffers based on input/output dimensions
   * @param inputDim - Input feature dimension
   * @param outputDim - Output dimension
   * @param seqLen - Sequence length
   */
  private _initialize(
    inputDim: number,
    outputDim: number,
    seqLen: number,
  ): void {
    this._inputDim = inputDim;
    this._outputDim = outputDim;
    this._seqLen = Math.min(seqLen, this._maxSequenceLength);

    // Calculate weight sizes and offsets
    this._calculateWeightLayout();

    // Allocate weight arrays
    this._weights = new Float64Array(this._totalParams);
    this._gradients = new Float64Array(this._totalParams);
    this._firstMoment = new Float64Array(this._totalParams);
    this._secondMoment = new Float64Array(this._totalParams);

    // Initialize weights with Xavier/He initialization
    this._initializeWeights();

    // Allocate normalization statistics
    this._inputMean = new Float64Array(inputDim);
    this._inputM2 = new Float64Array(inputDim);
    this._outputMean = new Float64Array(outputDim);
    this._outputM2 = new Float64Array(outputDim);

    // Allocate reusable buffers
    const maxBufSize = Math.max(
      this._seqLen * this._embeddingDim,
      this._seqLen * this._seqLen * this._numHeads,
      this._ffnDim * this._seqLen,
    );
    this._tempBuffer1 = new Float64Array(maxBufSize);
    this._tempBuffer2 = new Float64Array(maxBufSize);
    this._tempBuffer3 = new Float64Array(maxBufSize);
    this._attentionScores = new Float64Array(
      this._seqLen * this._seqLen * this._numHeads,
    );
    this._softmaxBuffer = new Float64Array(this._seqLen);

    this._isInitialized = true;
  }

  /**
   * Calculate memory layout for all weights
   */
  private _calculateWeightLayout(): void {
    let offset = 0;
    const D = this._embeddingDim;
    const H = this._numHeads;
    const Dh = this._headDim;
    const Df = this._ffnDim;
    const K = this._temporalKernelSize;

    // Temporal convolution weights for each scale: [K × inputDim × D]
    for (let s = 0; s < this._numScales; s++) {
      this._weightOffsets[`conv_w_${s}`] = offset;
      offset += K * this._inputDim * D;
      this._weightOffsets[`conv_b_${s}`] = offset;
      offset += D;
    }

    // Scale embeddings: [numScales × D]
    this._weightOffsets["scale_emb"] = offset;
    offset += this._numScales * D;

    // Fusion gate weights: [numScales × D × numScales]
    this._weightOffsets["fusion_w"] = offset;
    offset += this._numScales * D * this._numScales;
    this._weightOffsets["fusion_b"] = offset;
    offset += this._numScales;

    // Cross-scale attention weights
    this._weightOffsets["cross_q"] = offset;
    offset += D * D;
    this._weightOffsets["cross_k"] = offset;
    offset += D * D;
    this._weightOffsets["cross_v"] = offset;
    offset += D * D;
    this._weightOffsets["cross_o"] = offset;
    offset += D * D;

    // Transformer blocks
    for (let b = 0; b < this._numBlocks; b++) {
      // LayerNorm 1
      this._weightOffsets[`ln1_g_${b}`] = offset;
      offset += D;
      this._weightOffsets[`ln1_b_${b}`] = offset;
      offset += D;

      // Multi-head attention: Q, K, V, O
      this._weightOffsets[`attn_q_${b}`] = offset;
      offset += D * D;
      this._weightOffsets[`attn_k_${b}`] = offset;
      offset += D * D;
      this._weightOffsets[`attn_v_${b}`] = offset;
      offset += D * D;
      this._weightOffsets[`attn_o_${b}`] = offset;
      offset += D * D;
      this._weightOffsets[`attn_ob_${b}`] = offset;
      offset += D;

      // Temporal bias for attention
      this._weightOffsets[`temp_bias_${b}`] = offset;
      offset += this._seqLen * this._seqLen;

      // LayerNorm 2
      this._weightOffsets[`ln2_g_${b}`] = offset;
      offset += D;
      this._weightOffsets[`ln2_b_${b}`] = offset;
      offset += D;

      // FFN: W1, b1, W2, b2
      this._weightOffsets[`ffn_w1_${b}`] = offset;
      offset += D * Df;
      this._weightOffsets[`ffn_b1_${b}`] = offset;
      offset += Df;
      this._weightOffsets[`ffn_w2_${b}`] = offset;
      offset += Df * D;
      this._weightOffsets[`ffn_b2_${b}`] = offset;
      offset += D;
    }

    // Pooling attention weights
    this._weightOffsets["pool_w"] = offset;
    offset += D;

    // Output projection
    this._weightOffsets["out_w"] = offset;
    offset += D * this._outputDim;
    this._weightOffsets["out_b"] = offset;
    offset += this._outputDim;

    this._totalParams = offset;
  }

  /**
   * Initialize weights using Xavier/He initialization
   * Formula: W ~ N(0, sqrt(2/fan_in)) for ReLU-like activations
   */
  private _initializeWeights(): void {
    const D = this._embeddingDim;
    const K = this._temporalKernelSize;

    // Helper for Xavier initialization
    const xavier = (fanIn: number, fanOut: number): number => {
      const std = Math.sqrt(2.0 / (fanIn + fanOut));
      return this._gaussianRandom() * std;
    };

    // Helper for He initialization (for GELU)
    const he = (fanIn: number): number => {
      const std = Math.sqrt(2.0 / fanIn);
      return this._gaussianRandom() * std;
    };

    // Initialize temporal convolution weights
    for (let s = 0; s < this._numScales; s++) {
      const wOffset = this._weightOffsets[`conv_w_${s}`];
      const bOffset = this._weightOffsets[`conv_b_${s}`];
      const fanIn = K * this._inputDim;
      for (let i = 0; i < K * this._inputDim * D; i++) {
        this._weights[wOffset + i] = he(fanIn);
      }
      // Bias initialized to zero
      for (let i = 0; i < D; i++) {
        this._weights[bOffset + i] = 0;
      }
    }

    // Scale embeddings: small random init
    const seOffset = this._weightOffsets["scale_emb"];
    for (let i = 0; i < this._numScales * D; i++) {
      this._weights[seOffset + i] = this._gaussianRandom() * 0.02;
    }

    // Fusion weights
    const fwOffset = this._weightOffsets["fusion_w"];
    const fbOffset = this._weightOffsets["fusion_b"];
    for (let i = 0; i < this._numScales * D * this._numScales; i++) {
      this._weights[fwOffset + i] = xavier(
        D * this._numScales,
        this._numScales,
      );
    }
    for (let i = 0; i < this._numScales; i++) {
      this._weights[fbOffset + i] = 0;
    }

    // Cross-scale attention
    const crossKeys = ["cross_q", "cross_k", "cross_v", "cross_o"];
    for (const key of crossKeys) {
      const off = this._weightOffsets[key];
      for (let i = 0; i < D * D; i++) {
        this._weights[off + i] = xavier(D, D);
      }
    }

    // Transformer blocks
    for (let b = 0; b < this._numBlocks; b++) {
      // LayerNorm: gamma=1, beta=0
      const ln1gOff = this._weightOffsets[`ln1_g_${b}`];
      const ln1bOff = this._weightOffsets[`ln1_b_${b}`];
      const ln2gOff = this._weightOffsets[`ln2_g_${b}`];
      const ln2bOff = this._weightOffsets[`ln2_b_${b}`];
      for (let i = 0; i < D; i++) {
        this._weights[ln1gOff + i] = 1.0;
        this._weights[ln1bOff + i] = 0.0;
        this._weights[ln2gOff + i] = 1.0;
        this._weights[ln2bOff + i] = 0.0;
      }

      // Attention weights
      const attnKeys = [
        `attn_q_${b}`,
        `attn_k_${b}`,
        `attn_v_${b}`,
        `attn_o_${b}`,
      ];
      for (const key of attnKeys) {
        const off = this._weightOffsets[key];
        for (let i = 0; i < D * D; i++) {
          this._weights[off + i] = xavier(D, D);
        }
      }
      const obOff = this._weightOffsets[`attn_ob_${b}`];
      for (let i = 0; i < D; i++) {
        this._weights[obOff + i] = 0;
      }

      // Temporal bias: initialize to 0
      const tbOff = this._weightOffsets[`temp_bias_${b}`];
      for (let i = 0; i < this._seqLen * this._seqLen; i++) {
        this._weights[tbOff + i] = 0;
      }

      // FFN weights
      const ffnW1Off = this._weightOffsets[`ffn_w1_${b}`];
      const ffnB1Off = this._weightOffsets[`ffn_b1_${b}`];
      const ffnW2Off = this._weightOffsets[`ffn_w2_${b}`];
      const ffnB2Off = this._weightOffsets[`ffn_b2_${b}`];

      for (let i = 0; i < D * this._ffnDim; i++) {
        this._weights[ffnW1Off + i] = he(D);
      }
      for (let i = 0; i < this._ffnDim; i++) {
        this._weights[ffnB1Off + i] = 0;
      }
      for (let i = 0; i < this._ffnDim * D; i++) {
        this._weights[ffnW2Off + i] = xavier(this._ffnDim, D);
      }
      for (let i = 0; i < D; i++) {
        this._weights[ffnB2Off + i] = 0;
      }
    }

    // Pooling weights
    const poolOff = this._weightOffsets["pool_w"];
    for (let i = 0; i < D; i++) {
      this._weights[poolOff + i] = xavier(D, 1);
    }

    // Output weights
    const outWOff = this._weightOffsets["out_w"];
    const outBOff = this._weightOffsets["out_b"];
    for (let i = 0; i < D * this._outputDim; i++) {
      this._weights[outWOff + i] = xavier(D, this._outputDim);
    }
    for (let i = 0; i < this._outputDim; i++) {
      this._weights[outBOff + i] = 0;
    }
  }

  /**
   * Generate Gaussian random number using Box-Muller transform
   * @returns Random value from N(0, 1)
   */
  private _gaussianRandom(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // =========================================================================
  // Activation Functions
  // =========================================================================

  /**
   * GELU activation function (Gaussian Error Linear Unit)
   * Formula: GELU(x) = x * Φ(x) ≈ x * σ(1.702x)
   * @param x - Input value
   * @returns Activated value
   */
  private _gelu(x: number): number {
    // Fast approximation: x * sigmoid(1.702 * x)
    return x * (1.0 / (1.0 + Math.exp(-1.702 * x)));
  }

  /**
   * GELU derivative for backpropagation
   * @param x - Input value
   * @returns Derivative of GELU at x
   */
  private _geluDerivative(x: number): number {
    const sig = 1.0 / (1.0 + Math.exp(-1.702 * x));
    return sig + x * 1.702 * sig * (1.0 - sig);
  }

  /**
   * Sigmoid activation
   * Formula: σ(x) = 1 / (1 + e^(-x))
   * @param x - Input value
   * @returns Sigmoid output
   */
  private _sigmoid(x: number): number {
    if (x >= 0) {
      return 1.0 / (1.0 + Math.exp(-x));
    }
    const expX = Math.exp(x);
    return expX / (1.0 + expX);
  }

  // =========================================================================
  // Matrix Operations (In-Place, Cache Efficient)
  // =========================================================================

  /**
   * Matrix-vector multiplication: y = Wx + b (in-place)
   * @param W - Weight matrix [outDim × inDim] stored row-major
   * @param wOffset - Offset into weight array
   * @param x - Input vector [inDim]
   * @param xOffset - Offset into input
   * @param b - Bias vector [outDim] (optional)
   * @param bOffset - Offset into bias
   * @param y - Output vector [outDim]
   * @param yOffset - Offset into output
   * @param inDim - Input dimension
   * @param outDim - Output dimension
   */
  private _matVecMul(
    W: Float64Array,
    wOffset: number,
    x: Float64Array,
    xOffset: number,
    b: Float64Array | null,
    bOffset: number,
    y: Float64Array,
    yOffset: number,
    inDim: number,
    outDim: number,
  ): void {
    for (let i = 0; i < outDim; i++) {
      let sum = b !== null ? b[bOffset + i] : 0;
      const rowOff = wOffset + i * inDim;
      for (let j = 0; j < inDim; j++) {
        sum += W[rowOff + j] * x[xOffset + j];
      }
      y[yOffset + i] = sum;
    }
  }

  /**
   * Matrix-matrix multiplication: C = AB (in-place)
   * A: [M × K], B: [K × N], C: [M × N]
   */
  private _matMul(
    A: Float64Array,
    aOffset: number,
    B: Float64Array,
    bOffset: number,
    C: Float64Array,
    cOffset: number,
    M: number,
    K: number,
    N: number,
  ): void {
    // Clear output
    for (let i = 0; i < M * N; i++) {
      C[cOffset + i] = 0;
    }
    // Cache-friendly loop order
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aVal = A[aOffset + i * K + k];
        const bRowOff = bOffset + k * N;
        const cRowOff = cOffset + i * N;
        for (let j = 0; j < N; j++) {
          C[cRowOff + j] += aVal * B[bRowOff + j];
        }
      }
    }
  }

  /**
   * Softmax over array segment (in-place, numerically stable)
   * Formula: softmax(x)_i = exp(x_i - max(x)) / Σexp(x_j - max(x))
   * @param arr - Array to apply softmax to
   * @param offset - Start offset
   * @param len - Length of segment
   */
  private _softmax(arr: Float64Array, offset: number, len: number): void {
    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < len; i++) {
      if (arr[offset + i] > maxVal) maxVal = arr[offset + i];
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const expVal = Math.exp(arr[offset + i] - maxVal);
      arr[offset + i] = expVal;
      sum += expVal;
    }

    // Normalize
    const invSum = 1.0 / (sum + this._epsilon);
    for (let i = 0; i < len; i++) {
      arr[offset + i] *= invSum;
    }
  }

  /**
   * Layer normalization (in-place)
   * Formula: y = γ * (x - μ) / sqrt(σ² + ε) + β
   * @param x - Input array
   * @param xOffset - Input offset
   * @param gamma - Scale parameters
   * @param gOffset - Gamma offset
   * @param beta - Shift parameters
   * @param bOffset - Beta offset
   * @param y - Output array
   * @param yOffset - Output offset
   * @param dim - Dimension to normalize over
   * @param meanOut - Output mean (for backprop cache)
   * @param varOut - Output variance (for backprop cache)
   */
  private _layerNorm(
    x: Float64Array,
    xOffset: number,
    gamma: Float64Array,
    gOffset: number,
    beta: Float64Array,
    bOffset: number,
    y: Float64Array,
    yOffset: number,
    dim: number,
    meanOut: Float64Array,
    varOut: Float64Array,
    cacheIdx: number,
  ): void {
    // Compute mean
    let mean = 0;
    for (let i = 0; i < dim; i++) {
      mean += x[xOffset + i];
    }
    mean /= dim;
    meanOut[cacheIdx] = mean;

    // Compute variance
    let variance = 0;
    for (let i = 0; i < dim; i++) {
      const diff = x[xOffset + i] - mean;
      variance += diff * diff;
    }
    variance /= dim;
    varOut[cacheIdx] = variance;

    // Normalize
    const invStd = 1.0 / Math.sqrt(variance + this._epsilon);
    for (let i = 0; i < dim; i++) {
      y[yOffset + i] = gamma[gOffset + i] * (x[xOffset + i] - mean) * invStd +
        beta[bOffset + i];
    }
  }

  // =========================================================================
  // Positional Encoding
  // =========================================================================

  /**
   * Compute sinusoidal positional encoding
   * Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
   *          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
   * @param output - Output buffer
   * @param seqLen - Sequence length
   * @param dim - Embedding dimension
   */
  private _computePositionalEncoding(
    output: Float64Array,
    seqLen: number,
    dim: number,
  ): void {
    for (let pos = 0; pos < seqLen; pos++) {
      for (let i = 0; i < dim; i += 2) {
        const divTerm = Math.exp(-(i / dim) * Math.log(10000.0));
        const angle = pos * divTerm;
        output[pos * dim + i] = Math.sin(angle);
        if (i + 1 < dim) {
          output[pos * dim + i + 1] = Math.cos(angle);
        }
      }
    }
  }

  // =========================================================================
  // Welford's Online Statistics
  // =========================================================================

  /**
   * Update running statistics using Welford's algorithm
   * Formula: δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
   * @param x - New data point
   * @param mean - Running mean array
   * @param m2 - Running M2 array
   * @param dim - Dimension of data
   */
  private _welfordUpdate(
    x: Float64Array,
    xOffset: number,
    mean: Float64Array,
    m2: Float64Array,
    dim: number,
  ): void {
    this._normCount++;
    const n = this._normCount;
    for (let i = 0; i < dim; i++) {
      const delta = x[xOffset + i] - mean[i];
      mean[i] += delta / n;
      const delta2 = x[xOffset + i] - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation from Welford statistics
   * Formula: σ = sqrt(M₂ / (n-1))
   * @param m2 - M2 statistics
   * @param output - Output array for std
   * @param dim - Dimension
   */
  private _getStd(m2: Float64Array, output: Float64Array, dim: number): void {
    const n = Math.max(2, this._normCount);
    for (let i = 0; i < dim; i++) {
      output[i] = Math.sqrt(m2[i] / (n - 1) + this._epsilon);
    }
  }

  // =========================================================================
  // Forward Pass
  // =========================================================================

  /**
   * Execute full forward pass through the network
   * @param input - Normalized input [seqLen × inputDim]
   * @param seqLen - Actual sequence length
   * @param cacheActivations - Whether to cache for backprop
   * @returns Output predictions [outputDim]
   */
  private _forward(
    input: Float64Array,
    seqLen: number,
    cacheActivations: boolean,
  ): Float64Array {
    const D = this._embeddingDim;

    // Allocate output buffer
    const output = this._bufferPool.acquire(this._outputDim);

    // Stage 1: Multi-scale temporal convolution
    const scaleOutputs: Float64Array[] = [];
    const scaleLengths: number[] = [];

    for (let s = 0; s < this._numScales; s++) {
      const scale = this._temporalScales[s];
      const outLen = Math.max(1, Math.floor(seqLen / scale));
      scaleLengths.push(outLen);
      const convOut = this._bufferPool.acquire(outLen * D);

      this._temporalConv(input, seqLen, s, scale, convOut, outLen);

      if (cacheActivations) {
        this._activationCache.set(`conv_out_${s}`, convOut);
        this._activationCache.set(`conv_in_${s}`, input);
      }
      scaleOutputs.push(convOut);
    }

    // Stage 2: Add positional encoding and scale embeddings
    const peBuffer = this._bufferPool.acquire(this._maxSequenceLength * D);
    this._computePositionalEncoding(peBuffer, this._maxSequenceLength, D);

    for (let s = 0; s < this._numScales; s++) {
      const outLen = scaleLengths[s];
      const seOff = this._weightOffsets["scale_emb"] + s * D;
      const scale = this._temporalScales[s];

      for (let t = 0; t < outLen; t++) {
        // Adjust positional index for the scale
        const peIdx = Math.min(t * scale, this._maxSequenceLength - 1);
        for (let d = 0; d < D; d++) {
          scaleOutputs[s][t * D + d] += peBuffer[peIdx * D + d] +
            this._weights[seOff + d];
        }
      }
    }
    this._bufferPool.release(peBuffer);

    // Stage 3: Cross-scale fusion
    // Upsample all scales to finest resolution and fuse
    const finestLen = scaleLengths[0];
    const fused = this._bufferPool.acquire(finestLen * D);
    this._crossScaleFusion(
      scaleOutputs,
      scaleLengths,
      fused,
      finestLen,
      cacheActivations,
    );

    if (cacheActivations) {
      this._activationCache.set("fused", fused);
    }

    // Release scale outputs (except fused which we keep)
    for (let s = 0; s < this._numScales; s++) {
      if (!cacheActivations) {
        this._bufferPool.release(scaleOutputs[s]);
      }
    }

    // Stage 4: Transformer blocks
    let current = fused;
    const currentLen = finestLen;

    for (let b = 0; b < this._numBlocks; b++) {
      const blockOut = this._bufferPool.acquire(currentLen * D);
      this._transformerBlock(
        current,
        currentLen,
        b,
        blockOut,
        cacheActivations,
      );

      if (!cacheActivations && current !== fused) {
        this._bufferPool.release(current);
      }
      if (cacheActivations) {
        this._activationCache.set(`block_out_${b}`, blockOut);
      }
      current = blockOut;
    }

    // Stage 5: Temporal aggregation via attention pooling
    const pooled = this._bufferPool.acquire(D);
    this._attentionPool(current, currentLen, pooled, cacheActivations);

    if (cacheActivations) {
      this._activationCache.set("pooled", pooled);
      this._activationCache.set("final_hidden", current);
    } else {
      this._bufferPool.release(current);
    }

    // Stage 6: Output projection
    const outWOff = this._weightOffsets["out_w"];
    const outBOff = this._weightOffsets["out_b"];
    this._matVecMul(
      this._weights,
      outWOff,
      pooled,
      0,
      this._weights,
      outBOff,
      output,
      0,
      D,
      this._outputDim,
    );

    if (!cacheActivations) {
      this._bufferPool.release(pooled);
    }

    return output;
  }

  /**
   * Temporal 1D convolution for a specific scale
   * Formula: F_s = GELU(Conv1D(X, kernel=K, stride=s))
   */
  private _temporalConv(
    input: Float64Array,
    seqLen: number,
    scaleIdx: number,
    stride: number,
    output: Float64Array,
    outLen: number,
  ): void {
    const D = this._embeddingDim;
    const K = this._temporalKernelSize;
    const wOff = this._weightOffsets[`conv_w_${scaleIdx}`];
    const bOff = this._weightOffsets[`conv_b_${scaleIdx}`];

    // Cache pre-activation for backprop
    const preAct = this._bufferPool.acquire(outLen * D);

    for (let t = 0; t < outLen; t++) {
      const startPos = t * stride;
      for (let d = 0; d < D; d++) {
        let sum = this._weights[bOff + d];
        for (let k = 0; k < K; k++) {
          const pos = Math.min(startPos + k, seqLen - 1);
          for (let i = 0; i < this._inputDim; i++) {
            const wIdx = wOff + (k * this._inputDim + i) * D + d;
            sum += this._weights[wIdx] * input[pos * this._inputDim + i];
          }
        }
        preAct[t * D + d] = sum;
        output[t * D + d] = this._gelu(sum);
      }
    }

    this._activationCache.set(`conv_preact_${scaleIdx}`, preAct);
  }

  /**
   * Cross-scale attention fusion
   * Formula: G = σ(Concat(E₁,...,Eₛ)Wg), Fused = Σ(Gₛ ⊙ Eₛ)
   */
  private _crossScaleFusion(
    scaleOutputs: Float64Array[],
    scaleLengths: number[],
    output: Float64Array,
    finestLen: number,
    cacheActivations: boolean,
  ): void {
    const D = this._embeddingDim;
    const S = this._numScales;
    const fwOff = this._weightOffsets["fusion_w"];
    const fbOff = this._weightOffsets["fusion_b"];

    // Upsample all scales and compute gating
    const upsampled: Float64Array[] = [];
    for (let s = 0; s < S; s++) {
      const up = this._bufferPool.acquire(finestLen * D);
      const srcLen = scaleLengths[s];
      const scale = this._temporalScales[s];

      for (let t = 0; t < finestLen; t++) {
        const srcT = Math.min(Math.floor(t / scale), srcLen - 1);
        for (let d = 0; d < D; d++) {
          up[t * D + d] = scaleOutputs[s][srcT * D + d];
        }
      }
      upsampled.push(up);
    }

    // Compute gating weights for each position
    const gates = this._bufferPool.acquire(finestLen * S);

    for (let t = 0; t < finestLen; t++) {
      // Concatenate features from all scales
      for (let s = 0; s < S; s++) {
        let gate = this._weights[fbOff + s];
        for (let s2 = 0; s2 < S; s2++) {
          for (let d = 0; d < D; d++) {
            const wIdx = fwOff + (s2 * D + d) * S + s;
            gate += this._weights[wIdx] * upsampled[s2][t * D + d];
          }
        }
        gates[t * S + s] = this._sigmoid(gate);
      }

      // Apply gated fusion
      for (let d = 0; d < D; d++) {
        let sum = 0;
        for (let s = 0; s < S; s++) {
          sum += gates[t * S + s] * upsampled[s][t * D + d];
        }
        output[t * D + d] = sum;
      }
    }

    if (cacheActivations) {
      this._activationCache.set("fusion_gates", gates);
      for (let s = 0; s < S; s++) {
        this._activationCache.set(`upsampled_${s}`, upsampled[s]);
      }
    } else {
      this._bufferPool.release(gates);
      for (let s = 0; s < S; s++) {
        this._bufferPool.release(upsampled[s]);
      }
    }
  }

  /**
   * Single transformer block
   * Structure: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual
   */
  private _transformerBlock(
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    output: Float64Array,
    cacheActivations: boolean,
  ): void {
    const D = this._embeddingDim;
    const b = blockIdx;

    // Allocate temporaries
    const normed1 = this._bufferPool.acquire(seqLen * D);
    const attnOut = this._bufferPool.acquire(seqLen * D);
    const residual1 = this._bufferPool.acquire(seqLen * D);
    const normed2 = this._bufferPool.acquire(seqLen * D);
    const ffnOut = this._bufferPool.acquire(seqLen * D);

    // Cache for layer norm backprop
    const lnMean = this._bufferPool.acquire(seqLen * 2);
    const lnVar = this._bufferPool.acquire(seqLen * 2);

    // LayerNorm 1
    const ln1gOff = this._weightOffsets[`ln1_g_${b}`];
    const ln1bOff = this._weightOffsets[`ln1_b_${b}`];
    for (let t = 0; t < seqLen; t++) {
      this._layerNorm(
        input,
        t * D,
        this._weights,
        ln1gOff,
        this._weights,
        ln1bOff,
        normed1,
        t * D,
        D,
        lnMean,
        lnVar,
        t,
      );
    }

    // Multi-head self-attention
    this._multiHeadAttention(
      normed1,
      seqLen,
      blockIdx,
      attnOut,
      cacheActivations,
    );

    // Residual connection 1
    for (let i = 0; i < seqLen * D; i++) {
      residual1[i] = input[i] + attnOut[i];
    }

    // LayerNorm 2
    const ln2gOff = this._weightOffsets[`ln2_g_${b}`];
    const ln2bOff = this._weightOffsets[`ln2_b_${b}`];
    for (let t = 0; t < seqLen; t++) {
      this._layerNorm(
        residual1,
        t * D,
        this._weights,
        ln2gOff,
        this._weights,
        ln2bOff,
        normed2,
        t * D,
        D,
        lnMean,
        lnVar,
        seqLen + t,
      );
    }

    // Feed-forward network
    this._feedForward(normed2, seqLen, blockIdx, ffnOut, cacheActivations);

    // Residual connection 2
    for (let i = 0; i < seqLen * D; i++) {
      output[i] = residual1[i] + ffnOut[i];
    }

    // Cache if needed
    if (cacheActivations) {
      this._activationCache.set(`block_${b}_input`, input);
      this._activationCache.set(`block_${b}_normed1`, normed1);
      this._activationCache.set(`block_${b}_attn_out`, attnOut);
      this._activationCache.set(`block_${b}_residual1`, residual1);
      this._activationCache.set(`block_${b}_normed2`, normed2);
      this._activationCache.set(`block_${b}_ffn_out`, ffnOut);
      this._activationCache.set(`block_${b}_ln_mean`, lnMean);
      this._activationCache.set(`block_${b}_ln_var`, lnVar);
    } else {
      this._bufferPool.release(normed1);
      this._bufferPool.release(attnOut);
      this._bufferPool.release(residual1);
      this._bufferPool.release(normed2);
      this._bufferPool.release(ffnOut);
      this._bufferPool.release(lnMean);
      this._bufferPool.release(lnVar);
    }
  }

  /**
   * Multi-head self-attention with temporal bias
   * Formula: Attention(Q,K,V) = softmax(QK^T/√d_k + TemporalBias)V
   */
  private _multiHeadAttention(
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    output: Float64Array,
    cacheActivations: boolean,
  ): void {
    const D = this._embeddingDim;
    const H = this._numHeads;
    const Dh = this._headDim;
    const b = blockIdx;

    const qOff = this._weightOffsets[`attn_q_${b}`];
    const kOff = this._weightOffsets[`attn_k_${b}`];
    const vOff = this._weightOffsets[`attn_v_${b}`];
    const oOff = this._weightOffsets[`attn_o_${b}`];
    const obOff = this._weightOffsets[`attn_ob_${b}`];
    const tbOff = this._weightOffsets[`temp_bias_${b}`];

    // Compute Q, K, V
    const Q = this._bufferPool.acquire(seqLen * D);
    const K = this._bufferPool.acquire(seqLen * D);
    const V = this._bufferPool.acquire(seqLen * D);

    // Linear projections
    for (let t = 0; t < seqLen; t++) {
      this._matVecMul(
        this._weights,
        qOff,
        input,
        t * D,
        null,
        0,
        Q,
        t * D,
        D,
        D,
      );
      this._matVecMul(
        this._weights,
        kOff,
        input,
        t * D,
        null,
        0,
        K,
        t * D,
        D,
        D,
      );
      this._matVecMul(
        this._weights,
        vOff,
        input,
        t * D,
        null,
        0,
        V,
        t * D,
        D,
        D,
      );
    }

    // Compute attention scores for each head
    const scores = this._bufferPool.acquire(H * seqLen * seqLen);
    const attnWeights = this._bufferPool.acquire(H * seqLen * seqLen);
    const headOut = this._bufferPool.acquire(seqLen * D);

    const scale = 1.0 / Math.sqrt(Dh);

    for (let h = 0; h < H; h++) {
      const headOff = h * Dh;

      // Compute QK^T for this head
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let score = 0;
          for (let d = 0; d < Dh; d++) {
            score += Q[i * D + headOff + d] * K[j * D + headOff + d];
          }
          // Add temporal bias and scale
          score = score * scale + this._weights[tbOff + i * seqLen + j];
          scores[h * seqLen * seqLen + i * seqLen + j] = score;
        }
      }

      // Apply softmax per query
      for (let i = 0; i < seqLen; i++) {
        const scoreOff = h * seqLen * seqLen + i * seqLen;
        // Copy to attnWeights before softmax
        for (let j = 0; j < seqLen; j++) {
          attnWeights[scoreOff + j] = scores[scoreOff + j];
        }
        this._softmax(attnWeights, scoreOff, seqLen);
      }

      // Compute attention output for this head
      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < Dh; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += attnWeights[h * seqLen * seqLen + i * seqLen + j] *
              V[j * D + headOff + d];
          }
          headOut[i * D + headOff + d] = sum;
        }
      }
    }

    // Output projection
    for (let t = 0; t < seqLen; t++) {
      this._matVecMul(
        this._weights,
        oOff,
        headOut,
        t * D,
        this._weights,
        obOff,
        output,
        t * D,
        D,
        D,
      );
    }

    if (cacheActivations) {
      this._activationCache.set(`attn_${b}_Q`, Q);
      this._activationCache.set(`attn_${b}_K`, K);
      this._activationCache.set(`attn_${b}_V`, V);
      this._activationCache.set(`attn_${b}_scores`, scores);
      this._activationCache.set(`attn_${b}_weights`, attnWeights);
      this._activationCache.set(`attn_${b}_head_out`, headOut);
    } else {
      this._bufferPool.release(Q);
      this._bufferPool.release(K);
      this._bufferPool.release(V);
      this._bufferPool.release(scores);
      this._bufferPool.release(attnWeights);
      this._bufferPool.release(headOut);
    }
  }

  /**
   * Feed-forward network
   * Formula: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
   */
  private _feedForward(
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    output: Float64Array,
    cacheActivations: boolean,
  ): void {
    const D = this._embeddingDim;
    const Df = this._ffnDim;
    const b = blockIdx;

    const w1Off = this._weightOffsets[`ffn_w1_${b}`];
    const b1Off = this._weightOffsets[`ffn_b1_${b}`];
    const w2Off = this._weightOffsets[`ffn_w2_${b}`];
    const b2Off = this._weightOffsets[`ffn_b2_${b}`];

    const hidden = this._bufferPool.acquire(seqLen * Df);
    const preAct = this._bufferPool.acquire(seqLen * Df);

    for (let t = 0; t < seqLen; t++) {
      // First linear layer
      this._matVecMul(
        this._weights,
        w1Off,
        input,
        t * D,
        this._weights,
        b1Off,
        preAct,
        t * Df,
        D,
        Df,
      );

      // GELU activation
      for (let d = 0; d < Df; d++) {
        hidden[t * Df + d] = this._gelu(preAct[t * Df + d]);
      }

      // Second linear layer
      this._matVecMul(
        this._weights,
        w2Off,
        hidden,
        t * Df,
        this._weights,
        b2Off,
        output,
        t * D,
        Df,
        D,
      );
    }

    if (cacheActivations) {
      this._activationCache.set(`ffn_${b}_preact`, preAct);
      this._activationCache.set(`ffn_${b}_hidden`, hidden);
    } else {
      this._bufferPool.release(hidden);
      this._bufferPool.release(preAct);
    }
  }

  /**
   * Attention-weighted temporal pooling
   * Formula: α = softmax(HW_pool), out = Σαᵢhᵢ
   */
  private _attentionPool(
    input: Float64Array,
    seqLen: number,
    output: Float64Array,
    cacheActivations: boolean,
  ): void {
    const D = this._embeddingDim;
    const poolOff = this._weightOffsets["pool_w"];

    const scores = this._bufferPool.acquire(seqLen);

    // Compute attention scores
    for (let t = 0; t < seqLen; t++) {
      let score = 0;
      for (let d = 0; d < D; d++) {
        score += input[t * D + d] * this._weights[poolOff + d];
      }
      scores[t] = score;
    }

    // Softmax
    this._softmax(scores, 0, seqLen);

    // Weighted sum
    output.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < D; d++) {
        output[d] += scores[t] * input[t * D + d];
      }
    }

    if (cacheActivations) {
      this._activationCache.set("pool_scores", scores);
    } else {
      this._bufferPool.release(scores);
    }
  }

  // =========================================================================
  // Backward Pass
  // =========================================================================

  /**
   * Execute full backward pass
   * @param target - Target output [outputDim]
   * @param predicted - Predicted output [outputDim]
   * @param seqLen - Sequence length
   * @returns Loss value
   */
  private _backward(
    target: Float64Array,
    predicted: Float64Array,
    seqLen: number,
  ): number {
    const D = this._embeddingDim;

    // Clear gradients
    this._gradients.fill(0);

    // Compute MSE loss and initial gradient
    // Formula: L = (1/2n)Σ‖y - ŷ‖²
    let loss = 0;
    const dOutput = this._bufferPool.acquire(this._outputDim);
    for (let i = 0; i < this._outputDim; i++) {
      const diff = predicted[i] - target[i];
      loss += diff * diff;
      dOutput[i] = diff / this._outputDim; // Gradient of MSE
    }
    loss = loss / (2 * this._outputDim);

    // Add L2 regularization loss
    // Formula: L_reg = (λ/2)Σ‖W‖²
    let regLoss = 0;
    for (let i = 0; i < this._totalParams; i++) {
      regLoss += this._weights[i] * this._weights[i];
    }
    regLoss *= this._regularizationStrength / 2;
    loss += regLoss;

    // Backprop through output layer
    const pooled = this._activationCache.get("pooled")!;
    const outWOff = this._weightOffsets["out_w"];
    const outBOff = this._weightOffsets["out_b"];

    const dPooled = this._bufferPool.acquire(D);

    // Gradient w.r.t. output weights: dW = dL/dŷ * pooled^T
    for (let i = 0; i < this._outputDim; i++) {
      for (let j = 0; j < D; j++) {
        this._gradients[outWOff + i * D + j] += dOutput[i] * pooled[j];
      }
      this._gradients[outBOff + i] += dOutput[i];
    }

    // Gradient w.r.t. pooled: dPooled = W^T * dOutput
    dPooled.fill(0);
    for (let j = 0; j < D; j++) {
      for (let i = 0; i < this._outputDim; i++) {
        dPooled[j] += this._weights[outWOff + i * D + j] * dOutput[i];
      }
    }

    // Backprop through attention pooling
    const finalHidden = this._activationCache.get("final_hidden")!;
    const poolScores = this._activationCache.get("pool_scores")!;
    const finestLen = Math.max(1, Math.floor(seqLen / this._temporalScales[0]));
    const dHidden = this._bufferPool.acquire(finestLen * D);

    this._backwardAttentionPool(
      dPooled,
      finalHidden,
      poolScores,
      finestLen,
      dHidden,
    );

    // Backprop through transformer blocks (reverse order)
    let dCurrent = dHidden;
    for (let b = this._numBlocks - 1; b >= 0; b--) {
      const dBlockIn = this._bufferPool.acquire(finestLen * D);
      this._backwardTransformerBlock(dCurrent, finestLen, b, dBlockIn);
      if (b < this._numBlocks - 1) {
        this._bufferPool.release(dCurrent);
      }
      dCurrent = dBlockIn;
    }

    // Backprop through cross-scale fusion
    const dScales: Float64Array[] = [];
    for (let s = 0; s < this._numScales; s++) {
      const scaleLen = Math.max(
        1,
        Math.floor(seqLen / this._temporalScales[s]),
      );
      dScales.push(this._bufferPool.acquire(scaleLen * D));
    }
    this._backwardCrossScaleFusion(dCurrent, finestLen, dScales);

    // Backprop through temporal convolutions
    for (let s = 0; s < this._numScales; s++) {
      const scaleLen = Math.max(
        1,
        Math.floor(seqLen / this._temporalScales[s]),
      );
      this._backwardTemporalConv(dScales[s], scaleLen, s, seqLen);
      this._bufferPool.release(dScales[s]);
    }

    // Add L2 regularization gradient
    for (let i = 0; i < this._totalParams; i++) {
      this._gradients[i] += this._regularizationStrength * this._weights[i];
    }

    // Cleanup
    this._bufferPool.release(dOutput);
    this._bufferPool.release(dPooled);
    this._bufferPool.release(dHidden);
    this._bufferPool.release(dCurrent);

    return loss;
  }

  /**
   * Backprop through attention pooling
   */
  private _backwardAttentionPool(
    dOutput: Float64Array,
    input: Float64Array,
    scores: Float64Array,
    seqLen: number,
    dInput: Float64Array,
  ): void {
    const D = this._embeddingDim;
    const poolOff = this._weightOffsets["pool_w"];

    // dInput[t] = scores[t] * dOutput
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < D; d++) {
        dInput[t * D + d] = scores[t] * dOutput[d];
      }
    }

    // Gradient through softmax and pooling weights
    // dScore_t = Σ_d (input[t,d] * dOutput[d])
    const dScores = this._bufferPool.acquire(seqLen);
    for (let t = 0; t < seqLen; t++) {
      let ds = 0;
      for (let d = 0; d < D; d++) {
        ds += input[t * D + d] * dOutput[d];
      }
      dScores[t] = ds;
    }

    // Softmax backward: dPre = (dScore - dot(dScore, score)) * score
    let dotProd = 0;
    for (let t = 0; t < seqLen; t++) {
      dotProd += dScores[t] * scores[t];
    }
    for (let t = 0; t < seqLen; t++) {
      const dPre = (dScores[t] - dotProd) * scores[t];
      for (let d = 0; d < D; d++) {
        this._gradients[poolOff + d] += dPre * input[t * D + d];
      }
    }

    this._bufferPool.release(dScores);
  }

  /**
   * Backprop through transformer block
   */
  private _backwardTransformerBlock(
    dOutput: Float64Array,
    seqLen: number,
    blockIdx: number,
    dInput: Float64Array,
  ): void {
    const D = this._embeddingDim;
    const b = blockIdx;

    // Retrieve cached activations
    const residual1 = this._activationCache.get(`block_${b}_residual1`)!;
    const normed2 = this._activationCache.get(`block_${b}_normed2`)!;
    const normed1 = this._activationCache.get(`block_${b}_normed1`)!;
    const blockInput = this._activationCache.get(`block_${b}_input`)!;

    // Backprop through residual 2: dResidual1 = dOutput, dFFN = dOutput
    const dFFN = this._bufferPool.acquire(seqLen * D);
    for (let i = 0; i < seqLen * D; i++) {
      dFFN[i] = dOutput[i];
    }

    // Backprop through FFN
    const dNormed2 = this._bufferPool.acquire(seqLen * D);
    this._backwardFFN(dFFN, normed2, seqLen, blockIdx, dNormed2);
    this._bufferPool.release(dFFN);

    // Backprop through LayerNorm 2
    const dResidual1 = this._bufferPool.acquire(seqLen * D);
    const lnMean = this._activationCache.get(`block_${b}_ln_mean`)!;
    const lnVar = this._activationCache.get(`block_${b}_ln_var`)!;
    this._backwardLayerNorm(
      dNormed2,
      residual1,
      seqLen,
      blockIdx,
      2,
      dResidual1,
      lnMean,
      lnVar,
      seqLen,
    );
    this._bufferPool.release(dNormed2);

    // Add residual gradient: dResidual1 += dOutput
    for (let i = 0; i < seqLen * D; i++) {
      dResidual1[i] += dOutput[i];
    }

    // Backprop through attention
    const dAttn = this._bufferPool.acquire(seqLen * D);
    for (let i = 0; i < seqLen * D; i++) {
      dAttn[i] = dResidual1[i];
    }

    const dNormed1 = this._bufferPool.acquire(seqLen * D);
    this._backwardMHA(dAttn, normed1, seqLen, blockIdx, dNormed1);
    this._bufferPool.release(dAttn);

    // Backprop through LayerNorm 1
    this._backwardLayerNorm(
      dNormed1,
      blockInput,
      seqLen,
      blockIdx,
      1,
      dInput,
      lnMean,
      lnVar,
      0,
    );
    this._bufferPool.release(dNormed1);

    // Add residual gradient: dInput += dResidual1
    for (let i = 0; i < seqLen * D; i++) {
      dInput[i] += dResidual1[i];
    }

    this._bufferPool.release(dResidual1);
  }

  /**
   * Backprop through FFN
   */
  private _backwardFFN(
    dOutput: Float64Array,
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    dInput: Float64Array,
  ): void {
    const D = this._embeddingDim;
    const Df = this._ffnDim;
    const b = blockIdx;

    const w1Off = this._weightOffsets[`ffn_w1_${b}`];
    const b1Off = this._weightOffsets[`ffn_b1_${b}`];
    const w2Off = this._weightOffsets[`ffn_w2_${b}`];
    const b2Off = this._weightOffsets[`ffn_b2_${b}`];

    const preAct = this._activationCache.get(`ffn_${b}_preact`)!;
    const hidden = this._activationCache.get(`ffn_${b}_hidden`)!;

    const dHidden = this._bufferPool.acquire(seqLen * Df);
    const dPreAct = this._bufferPool.acquire(seqLen * Df);

    for (let t = 0; t < seqLen; t++) {
      // Backprop through W2
      // dHidden = W2^T * dOutput
      for (let d = 0; d < Df; d++) {
        let sum = 0;
        for (let i = 0; i < D; i++) {
          sum += this._weights[w2Off + d * D + i] * dOutput[t * D + i];
        }
        dHidden[t * Df + d] = sum;
      }

      // Gradient w.r.t W2: dW2 += dOutput * hidden^T
      for (let d = 0; d < Df; d++) {
        for (let i = 0; i < D; i++) {
          this._gradients[w2Off + d * D + i] += hidden[t * Df + d] *
            dOutput[t * D + i];
        }
      }
      for (let i = 0; i < D; i++) {
        this._gradients[b2Off + i] += dOutput[t * D + i];
      }

      // Backprop through GELU
      for (let d = 0; d < Df; d++) {
        dPreAct[t * Df + d] = dHidden[t * Df + d] *
          this._geluDerivative(preAct[t * Df + d]);
      }

      // Backprop through W1
      // dInput = W1^T * dPreAct
      for (let i = 0; i < D; i++) {
        let sum = 0;
        for (let d = 0; d < Df; d++) {
          sum += this._weights[w1Off + i * Df + d] * dPreAct[t * Df + d];
        }
        dInput[t * D + i] = sum;
      }

      // Gradient w.r.t W1
      for (let i = 0; i < D; i++) {
        for (let d = 0; d < Df; d++) {
          this._gradients[w1Off + i * Df + d] += input[t * D + i] *
            dPreAct[t * Df + d];
        }
      }
      for (let d = 0; d < Df; d++) {
        this._gradients[b1Off + d] += dPreAct[t * Df + d];
      }
    }

    this._bufferPool.release(dHidden);
    this._bufferPool.release(dPreAct);
  }

  /**
   * Backprop through multi-head attention
   */
  private _backwardMHA(
    dOutput: Float64Array,
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    dInput: Float64Array,
  ): void {
    const D = this._embeddingDim;
    const H = this._numHeads;
    const Dh = this._headDim;
    const b = blockIdx;

    const qOff = this._weightOffsets[`attn_q_${b}`];
    const kOff = this._weightOffsets[`attn_k_${b}`];
    const vOff = this._weightOffsets[`attn_v_${b}`];
    const oOff = this._weightOffsets[`attn_o_${b}`];
    const obOff = this._weightOffsets[`attn_ob_${b}`];
    const tbOff = this._weightOffsets[`temp_bias_${b}`];

    const Q = this._activationCache.get(`attn_${b}_Q`)!;
    const K = this._activationCache.get(`attn_${b}_K`)!;
    const V = this._activationCache.get(`attn_${b}_V`)!;
    const attnWeights = this._activationCache.get(`attn_${b}_weights`)!;
    const headOut = this._activationCache.get(`attn_${b}_head_out`)!;

    const dHeadOut = this._bufferPool.acquire(seqLen * D);
    const dQ = this._bufferPool.acquire(seqLen * D);
    const dK = this._bufferPool.acquire(seqLen * D);
    const dV = this._bufferPool.acquire(seqLen * D);

    dQ.fill(0);
    dK.fill(0);
    dV.fill(0);

    // Backprop through output projection
    for (let t = 0; t < seqLen; t++) {
      // dHeadOut = Wo^T * dOutput
      for (let d = 0; d < D; d++) {
        let sum = 0;
        for (let i = 0; i < D; i++) {
          sum += this._weights[oOff + d * D + i] * dOutput[t * D + i];
        }
        dHeadOut[t * D + d] = sum;
      }

      // Gradient w.r.t Wo
      for (let d = 0; d < D; d++) {
        for (let i = 0; i < D; i++) {
          this._gradients[oOff + d * D + i] += headOut[t * D + d] *
            dOutput[t * D + i];
        }
      }
      for (let i = 0; i < D; i++) {
        this._gradients[obOff + i] += dOutput[t * D + i];
      }
    }

    const scale = 1.0 / Math.sqrt(Dh);

    // Backprop through attention for each head
    for (let h = 0; h < H; h++) {
      const headOff = h * Dh;

      // For each query position
      for (let i = 0; i < seqLen; i++) {
        // dAttnWeights = dHeadOut * V^T
        const dAttnW = this._bufferPool.acquire(seqLen);
        for (let j = 0; j < seqLen; j++) {
          let sum = 0;
          for (let d = 0; d < Dh; d++) {
            sum += dHeadOut[i * D + headOff + d] * V[j * D + headOff + d];
          }
          dAttnW[j] = sum;
        }

        // dV += attnWeights * dHeadOut
        for (let j = 0; j < seqLen; j++) {
          for (let d = 0; d < Dh; d++) {
            dV[j * D + headOff + d] +=
              attnWeights[h * seqLen * seqLen + i * seqLen + j] *
              dHeadOut[i * D + headOff + d];
          }
        }

        // Backprop through softmax
        const awOff = h * seqLen * seqLen + i * seqLen;
        let dotProd = 0;
        for (let j = 0; j < seqLen; j++) {
          dotProd += dAttnW[j] * attnWeights[awOff + j];
        }

        for (let j = 0; j < seqLen; j++) {
          const dScore = (dAttnW[j] - dotProd) * attnWeights[awOff + j] * scale;

          // Gradient for temporal bias
          this._gradients[tbOff + i * seqLen + j] += dScore / scale;

          // dQ += dScore * K
          for (let d = 0; d < Dh; d++) {
            dQ[i * D + headOff + d] += dScore * K[j * D + headOff + d];
            dK[j * D + headOff + d] += dScore * Q[i * D + headOff + d];
          }
        }

        this._bufferPool.release(dAttnW);
      }
    }

    // Backprop through Q, K, V projections
    dInput.fill(0);
    for (let t = 0; t < seqLen; t++) {
      // dInput += Wq^T * dQ + Wk^T * dK + Wv^T * dV
      for (let i = 0; i < D; i++) {
        for (let d = 0; d < D; d++) {
          dInput[t * D + i] += this._weights[qOff + i * D + d] * dQ[t * D + d];
          dInput[t * D + i] += this._weights[kOff + i * D + d] * dK[t * D + d];
          dInput[t * D + i] += this._weights[vOff + i * D + d] * dV[t * D + d];
        }
      }

      // Gradients for projection weights
      for (let i = 0; i < D; i++) {
        for (let d = 0; d < D; d++) {
          this._gradients[qOff + i * D + d] += input[t * D + i] * dQ[t * D + d];
          this._gradients[kOff + i * D + d] += input[t * D + i] * dK[t * D + d];
          this._gradients[vOff + i * D + d] += input[t * D + i] * dV[t * D + d];
        }
      }
    }

    this._bufferPool.release(dHeadOut);
    this._bufferPool.release(dQ);
    this._bufferPool.release(dK);
    this._bufferPool.release(dV);
  }

  /**
   * Backprop through layer normalization
   */
  private _backwardLayerNorm(
    dOutput: Float64Array,
    input: Float64Array,
    seqLen: number,
    blockIdx: number,
    lnIdx: number,
    dInput: Float64Array,
    meanCache: Float64Array,
    varCache: Float64Array,
    cacheOffset: number,
  ): void {
    const D = this._embeddingDim;
    const gOff = this._weightOffsets[`ln${lnIdx}_g_${blockIdx}`];
    const bOff = this._weightOffsets[`ln${lnIdx}_b_${blockIdx}`];

    for (let t = 0; t < seqLen; t++) {
      const mean = meanCache[cacheOffset + t];
      const variance = varCache[cacheOffset + t];
      const invStd = 1.0 / Math.sqrt(variance + this._epsilon);

      // Compute xhat = (x - mean) * invStd
      let dVar = 0;
      let dMean = 0;

      for (let d = 0; d < D; d++) {
        const xhat = (input[t * D + d] - mean) * invStd;

        // Gradient w.r.t gamma and beta
        this._gradients[gOff + d] += dOutput[t * D + d] * xhat;
        this._gradients[bOff + d] += dOutput[t * D + d];

        // Gradient w.r.t xhat
        const dxhat = dOutput[t * D + d] * this._weights[gOff + d];

        // Accumulate for dVar and dMean
        dVar += dxhat * (input[t * D + d] - mean) * (-0.5) *
          Math.pow(variance + this._epsilon, -1.5);
        dMean += dxhat * (-invStd);
      }

      dMean += dVar * (-2.0 / D) * (input[t * D] - mean); // Simplified

      // Final gradient w.r.t input
      for (let d = 0; d < D; d++) {
        const xhat = (input[t * D + d] - mean) * invStd;
        const dxhat = dOutput[t * D + d] * this._weights[gOff + d];
        dInput[t * D + d] = dxhat * invStd +
          dVar * 2.0 * (input[t * D + d] - mean) / D + dMean / D;
      }
    }
  }

  /**
   * Backprop through cross-scale fusion
   */
  private _backwardCrossScaleFusion(
    dOutput: Float64Array,
    finestLen: number,
    dScales: Float64Array[],
  ): void {
    const D = this._embeddingDim;
    const S = this._numScales;
    const fwOff = this._weightOffsets["fusion_w"];
    const fbOff = this._weightOffsets["fusion_b"];

    const gates = this._activationCache.get("fusion_gates")!;
    const upsampled: Float64Array[] = [];
    for (let s = 0; s < S; s++) {
      upsampled.push(this._activationCache.get(`upsampled_${s}`)!);
    }

    // Initialize dScales to zero
    for (let s = 0; s < S; s++) {
      dScales[s].fill(0);
    }

    // For each position
    for (let t = 0; t < finestLen; t++) {
      // Gradient w.r.t gates: dGate = dOutput ⊙ upsampled
      const dGates = this._bufferPool.acquire(S);
      for (let s = 0; s < S; s++) {
        let dg = 0;
        for (let d = 0; d < D; d++) {
          dg += dOutput[t * D + d] * upsampled[s][t * D + d];
        }
        dGates[s] = dg;
      }

      // Gradient w.r.t upsampled: dUpsampled = dOutput ⊙ gates
      for (let s = 0; s < S; s++) {
        const scale = this._temporalScales[s];
        const srcT = Math.min(
          Math.floor(t / scale),
          Math.floor(finestLen / scale) - 1,
        );
        const srcT_clamped = Math.max(0, srcT);
        for (let d = 0; d < D; d++) {
          dScales[s][srcT_clamped * D + d] += dOutput[t * D + d] *
            gates[t * S + s];
        }
      }

      // Backprop through sigmoid: dPre = dGate * gate * (1 - gate)
      for (let s = 0; s < S; s++) {
        const gate = gates[t * S + s];
        const dPre = dGates[s] * gate * (1 - gate);

        // Gradient w.r.t fusion weights and bias
        this._gradients[fbOff + s] += dPre;
        for (let s2 = 0; s2 < S; s2++) {
          for (let d = 0; d < D; d++) {
            const wIdx = fwOff + (s2 * D + d) * S + s;
            this._gradients[wIdx] += dPre * upsampled[s2][t * D + d];
          }
        }
      }

      this._bufferPool.release(dGates);
    }
  }

  /**
   * Backprop through temporal convolution
   */
  private _backwardTemporalConv(
    dOutput: Float64Array,
    outLen: number,
    scaleIdx: number,
    seqLen: number,
  ): void {
    const D = this._embeddingDim;
    const K = this._temporalKernelSize;
    const stride = this._temporalScales[scaleIdx];
    const wOff = this._weightOffsets[`conv_w_${scaleIdx}`];
    const bOff = this._weightOffsets[`conv_b_${scaleIdx}`];

    const preAct = this._activationCache.get(`conv_preact_${scaleIdx}`)!;
    const input = this._activationCache.get(`conv_in_${scaleIdx}`)!;

    for (let t = 0; t < outLen; t++) {
      const startPos = t * stride;

      for (let d = 0; d < D; d++) {
        // Backprop through GELU
        const dPreAct = dOutput[t * D + d] *
          this._geluDerivative(preAct[t * D + d]);

        // Gradient w.r.t bias
        this._gradients[bOff + d] += dPreAct;

        // Gradient w.r.t weights
        for (let k = 0; k < K; k++) {
          const pos = Math.min(startPos + k, seqLen - 1);
          for (let i = 0; i < this._inputDim; i++) {
            const wIdx = wOff + (k * this._inputDim + i) * D + d;
            this._gradients[wIdx] += dPreAct * input[pos * this._inputDim + i];
          }
        }
      }
    }

    // Add gradients for scale embeddings
    const seOff = this._weightOffsets["scale_emb"] + scaleIdx * D;
    for (let t = 0; t < outLen; t++) {
      for (let d = 0; d < D; d++) {
        this._gradients[seOff + d] += dOutput[t * D + d];
      }
    }
  }

  // =========================================================================
  // Optimizer
  // =========================================================================

  /**
   * Compute effective learning rate with warmup and cosine decay
   * Formula: warmup → cosine decay
   * @returns Current effective learning rate
   */
  private _getEffectiveLearningRate(): number {
    const step = this._updateCount;

    if (step < this._warmupSteps) {
      // Linear warmup
      return this._learningRate * (step + 1) / this._warmupSteps;
    }

    // Cosine decay
    const progress = (step - this._warmupSteps) /
      (this._totalSteps - this._warmupSteps);
    const decay = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
    return this._learningRate * decay;
  }

  /**
   * Adam optimizer update step
   * Formula: m = β₁m + (1-β₁)g
   *          v = β₂v + (1-β₂)g²
   *          W -= η(m/(1-β₁ᵗ))/(√(v/(1-β₂ᵗ)) + ε)
   */
  private _adamUpdate(): void {
    this._updateCount++;
    const lr = this._getEffectiveLearningRate();
    const t = this._updateCount;

    // Bias correction factors
    const bc1 = 1 - Math.pow(this._beta1, t);
    const bc2 = 1 - Math.pow(this._beta2, t);

    for (let i = 0; i < this._totalParams; i++) {
      const g = this._gradients[i];

      // Update first moment (momentum)
      this._firstMoment[i] = this._beta1 * this._firstMoment[i] +
        (1 - this._beta1) * g;

      // Update second moment (RMSprop)
      this._secondMoment[i] = this._beta2 * this._secondMoment[i] +
        (1 - this._beta2) * g * g;

      // Bias-corrected estimates
      const mHat = this._firstMoment[i] / bc1;
      const vHat = this._secondMoment[i] / bc2;

      // Update weights
      this._weights[i] -= lr * mHat / (Math.sqrt(vHat) + this._epsilon);
    }
  }

  /**
   * Compute gradient norm for monitoring
   * @returns L2 norm of gradient
   */
  private _computeGradientNorm(): number {
    let norm = 0;
    for (let i = 0; i < this._totalParams; i++) {
      norm += this._gradients[i] * this._gradients[i];
    }
    return Math.sqrt(norm);
  }

  // =========================================================================
  // ADWIN Drift Detection
  // =========================================================================

  /**
   * ADWIN drift detection algorithm
   * Detects concept drift by monitoring error distribution changes
   * Formula: |μ₀ - μ₁| ≥ εcut(δ)
   * @param error - Current error value
   * @returns Whether drift was detected
   */
  private _adwinDetectDrift(error: number): boolean {
    this._adwinWindow.push(error);

    if (this._adwinWindow.length < 10) {
      return false;
    }

    // Try to find a split point where distributions differ significantly
    const n = this._adwinWindow.length;

    for (let i = Math.floor(n * 0.3); i < Math.floor(n * 0.7); i++) {
      // Compute mean and variance for both windows
      let sum0 = 0, sum1 = 0;
      for (let j = 0; j < i; j++) sum0 += this._adwinWindow[j];
      for (let j = i; j < n; j++) sum1 += this._adwinWindow[j];

      const mean0 = sum0 / i;
      const mean1 = sum1 / (n - i);
      const diff = Math.abs(mean0 - mean1);

      // Compute epsilon cutoff
      const m = 1.0 / (1.0 / i + 1.0 / (n - i));
      const deltaPrime = this._adwinDelta / Math.log(n);
      const epsilon = Math.sqrt((1.0 / (2.0 * m)) * Math.log(4.0 / deltaPrime));

      if (diff >= epsilon) {
        // Drift detected - shrink window
        this._adwinWindow = this._adwinWindow.slice(i);
        return true;
      }
    }

    // Limit window size
    if (this._adwinWindow.length > 1000) {
      this._adwinWindow = this._adwinWindow.slice(-500);
    }

    return false;
  }

  // =========================================================================
  // Outlier Detection
  // =========================================================================

  /**
   * Detect outliers using residual-based approach
   * Formula: r = (y - ŷ)/σ; |r| > threshold → outlier
   * @param target - Target values
   * @param predicted - Predicted values
   * @returns Whether sample is an outlier and weight factor
   */
  private _detectOutlier(
    target: Float64Array,
    predicted: Float64Array,
  ): { isOutlier: boolean; weight: number } {
    const std = this._bufferPool.acquire(this._outputDim);
    this._getStd(this._outputM2, std, this._outputDim);

    let maxResidual = 0;
    for (let i = 0; i < this._outputDim; i++) {
      const residual = Math.abs(target[i] - predicted[i]) /
        (std[i] + this._epsilon);
      if (residual > maxResidual) maxResidual = residual;
    }

    this._bufferPool.release(std);

    const isOutlier = maxResidual > this._outlierThreshold;
    const weight = isOutlier ? 0.1 : 1.0;

    return { isOutlier, weight };
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Perform incremental online learning step
   *
   * Updates model weights using Adam optimizer with:
   * - Welford's running statistics for z-score normalization
   * - L2 regularization
   * - Outlier downweighting
   * - ADWIN drift detection
   *
   * @param data - Training data with input and output coordinates
   * @param data.xCoordinates - Input sequence [seqLen × inputDim]
   * @param data.yCoordinates - Output targets [seqLen × outputDim]
   * @returns Training step result
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
   *   yCoordinates: [[7], [8], [9]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Validate input
    if (
      !xCoordinates || !yCoordinates || xCoordinates.length === 0 ||
      yCoordinates.length === 0
    ) {
      throw new Error(
        "Invalid input: xCoordinates and yCoordinates must be non-empty arrays",
      );
    }

    const seqLen = xCoordinates.length;
    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;

    // Initialize model if needed
    if (!this._isInitialized) {
      this._initialize(inputDim, outputDim, seqLen);
    }

    // Validate dimensions
    if (inputDim !== this._inputDim || outputDim !== this._outputDim) {
      throw new Error(
        `Dimension mismatch: expected (${this._inputDim}, ${this._outputDim}), got (${inputDim}, ${outputDim})`,
      );
    }

    // Convert to Float64Array and flatten input
    const inputFlat = this._bufferPool.acquire(seqLen * inputDim);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < inputDim; d++) {
        inputFlat[t * inputDim + d] = xCoordinates[t][d];
      }
    }

    // Get last target for training (use last timestep)
    const target = this._bufferPool.acquire(outputDim);
    for (let d = 0; d < outputDim; d++) {
      target[d] = yCoordinates[yCoordinates.length - 1][d];
    }

    // Update normalization statistics
    for (let t = 0; t < seqLen; t++) {
      this._welfordUpdate(
        inputFlat,
        t * inputDim,
        this._inputMean,
        this._inputM2,
        inputDim,
      );
    }
    this._welfordUpdate(target, 0, this._outputMean, this._outputM2, outputDim);

    // Normalize input using z-score
    const inputStd = this._bufferPool.acquire(inputDim);
    this._getStd(this._inputM2, inputStd, inputDim);
    const normalizedInput = this._bufferPool.acquire(seqLen * inputDim);

    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < inputDim; d++) {
        normalizedInput[t * inputDim + d] =
          (inputFlat[t * inputDim + d] - this._inputMean[d]) /
          (inputStd[d] + this._epsilon);
      }
    }

    // Store in history for prediction
    if (this._inputHistory.length >= this._maxSequenceLength) {
      this._inputHistory.shift();
    }
    const historyEntry = new Float64Array(seqLen * inputDim);
    historyEntry.set(normalizedInput);
    this._inputHistory.push(historyEntry);

    // Forward pass with activation caching
    const effectiveSeqLen = Math.min(seqLen, this._seqLen);
    const predicted = this._forward(normalizedInput, effectiveSeqLen, true);

    // Denormalize prediction for outlier detection
    const outputStd = this._bufferPool.acquire(outputDim);
    this._getStd(this._outputM2, outputStd, outputDim);
    const denormPredicted = this._bufferPool.acquire(outputDim);
    for (let d = 0; d < outputDim; d++) {
      denormPredicted[d] = predicted[d] * outputStd[d] + this._outputMean[d];
    }

    // Detect outlier
    const { isOutlier, weight } = this._detectOutlier(target, denormPredicted);

    // Normalize target for loss computation
    const normalizedTarget = this._bufferPool.acquire(outputDim);
    for (let d = 0; d < outputDim; d++) {
      normalizedTarget[d] = (target[d] - this._outputMean[d]) /
        (outputStd[d] + this._epsilon);
    }

    // Backward pass
    let loss = this._backward(normalizedTarget, predicted, effectiveSeqLen);

    // Apply outlier weight to gradients
    if (isOutlier) {
      for (let i = 0; i < this._totalParams; i++) {
        this._gradients[i] *= weight;
      }
      loss *= weight;
    }

    // Compute gradient norm before update
    const gradientNorm = this._computeGradientNorm();

    // Adam update
    this._adamUpdate();

    // Update training statistics
    this._sampleCount++;
    this._totalLoss += loss;

    // Check convergence
    const lossDiff = Math.abs(loss - this._prevLoss);
    this._converged = lossDiff < this._convergenceThreshold;
    this._prevLoss = loss;

    // ADWIN drift detection
    const driftDetected = this._adwinDetectDrift(loss);
    if (driftDetected) {
      this._driftCount++;
      // Optionally reset some statistics on drift
      this._totalLoss = loss;
      this._sampleCount = 1;
    }

    // Clear activation cache
    this._activationCache.forEach((buf) => this._bufferPool.release(buf));
    this._activationCache.clear();

    // Cleanup buffers
    this._bufferPool.release(inputFlat);
    this._bufferPool.release(target);
    this._bufferPool.release(inputStd);
    this._bufferPool.release(normalizedInput);
    this._bufferPool.release(predicted);
    this._bufferPool.release(outputStd);
    this._bufferPool.release(denormPredicted);
    this._bufferPool.release(normalizedTarget);

    return {
      loss,
      gradientNorm,
      effectiveLearningRate: this._getEffectiveLearningRate(),
      isOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future timesteps
   *
   * Uses the learned model to predict future values with uncertainty estimates.
   *
   * @param futureSteps - Number of future steps to predict
   * @returns Prediction results with uncertainty bounds
   *
   * @example
   * ```typescript
   * const predictions = model.predict(5);
   * for (const pred of predictions.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this._isInitialized || this._inputHistory.length === 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this._sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const outputStd = this._bufferPool.acquire(this._outputDim);
    this._getStd(this._outputM2, outputStd, this._outputDim);

    // Use most recent input history
    const lastInput = this._inputHistory[this._inputHistory.length - 1];
    const seqLen = lastInput.length / this._inputDim;
    const effectiveSeqLen = Math.min(seqLen, this._seqLen);

    // Forward pass without caching
    const predicted = this._forward(lastInput, effectiveSeqLen, false);

    // Compute accuracy from running loss
    const avgLoss = this._sampleCount > 0
      ? this._totalLoss / this._sampleCount
      : Infinity;
    const accuracy = 1 / (1 + avgLoss);

    // Generate predictions for each future step
    for (let step = 0; step < futureSteps; step++) {
      const pred: number[] = [];
      const lower: number[] = [];
      const upper: number[] = [];
      const stdErr: number[] = [];

      for (let d = 0; d < this._outputDim; d++) {
        // Denormalize prediction
        const denorm = predicted[d] * outputStd[d] + this._outputMean[d];
        pred.push(denorm);

        // Estimate standard error based on running statistics
        // Uncertainty grows with prediction horizon
        const se = outputStd[d] * Math.sqrt(1 + step * 0.1);
        stdErr.push(se);

        // 95% confidence interval
        const z = 1.96;
        lower.push(denorm - z * se);
        upper.push(denorm + z * se);
      }

      predictions.push({
        predicted: pred,
        lowerBound: lower,
        upperBound: upper,
        standardError: stdErr,
      });
    }

    this._bufferPool.release(outputStd);
    this._bufferPool.release(predicted);

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get comprehensive model summary
   *
   * @returns Summary of model configuration and current state
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Parameters: ${summary.totalParameters}`);
   * console.log(`Accuracy: ${summary.accuracy}`);
   * ```
   */
  getModelSummary(): ModelSummary {
    const avgLoss = this._sampleCount > 0
      ? this._totalLoss / this._sampleCount
      : 0;

    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDim,
      outputDimension: this._outputDim,
      numBlocks: this._numBlocks,
      embeddingDim: this._embeddingDim,
      numHeads: this._numHeads,
      temporalScales: [...this._temporalScales],
      totalParameters: this._totalParams,
      sampleCount: this._sampleCount,
      accuracy: 1 / (1 + avgLoss),
      converged: this._converged,
      effectiveLearningRate: this._getEffectiveLearningRate(),
      driftCount: this._driftCount,
    };
  }

  /**
   * Get all model weights and optimizer state
   *
   * @returns Complete weight information including Adam moments
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Update count: ${weights.updateCount}`);
   * ```
   */
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

    const D = this._embeddingDim;
    const K = this._temporalKernelSize;

    // Extract temporal conv weights
    const temporalConvWeights: number[][][] = [];
    for (let s = 0; s < this._numScales; s++) {
      const wOff = this._weightOffsets[`conv_w_${s}`];
      const bOff = this._weightOffsets[`conv_b_${s}`];
      const layer: number[][] = [];

      // Weights
      const weights: number[] = [];
      for (let i = 0; i < K * this._inputDim * D; i++) {
        weights.push(this._weights[wOff + i]);
      }
      layer.push(weights);

      // Bias
      const bias: number[] = [];
      for (let i = 0; i < D; i++) {
        bias.push(this._weights[bOff + i]);
      }
      layer.push(bias);

      temporalConvWeights.push(layer);
    }

    // Extract scale embeddings
    const scaleEmbeddings: number[][][] = [];
    const seOff = this._weightOffsets["scale_emb"];
    for (let s = 0; s < this._numScales; s++) {
      const emb: number[] = [];
      for (let d = 0; d < D; d++) {
        emb.push(this._weights[seOff + s * D + d]);
      }
      scaleEmbeddings.push([emb]);
    }

    // Positional encoding (computed, not learned)
    const pe = new Float64Array(this._maxSequenceLength * D);
    this._computePositionalEncoding(pe, this._maxSequenceLength, D);
    const positionalEncoding: number[][] = [];
    for (let t = 0; t < this._maxSequenceLength; t++) {
      const row: number[] = [];
      for (let d = 0; d < D; d++) {
        row.push(pe[t * D + d]);
      }
      positionalEncoding.push(row);
    }

    // Fusion weights
    const fusionWeights: number[][][] = [];
    const fwOff = this._weightOffsets["fusion_w"];
    const fbOff = this._weightOffsets["fusion_b"];
    const fw: number[] = [];
    for (let i = 0; i < this._numScales * D * this._numScales; i++) {
      fw.push(this._weights[fwOff + i]);
    }
    const fb: number[] = [];
    for (let i = 0; i < this._numScales; i++) {
      fb.push(this._weights[fbOff + i]);
    }
    fusionWeights.push([fw, fb]);

    // Attention weights per block
    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this._numBlocks; b++) {
      const blockWeights: number[][] = [];
      for (const key of ["q", "k", "v", "o"]) {
        const off = this._weightOffsets[`attn_${key}_${b}`];
        const w: number[] = [];
        for (let i = 0; i < D * D; i++) {
          w.push(this._weights[off + i]);
        }
        blockWeights.push(w);
      }
      attentionWeights.push(blockWeights);
    }

    // FFN weights per block
    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this._numBlocks; b++) {
      const blockWeights: number[][] = [];
      for (const key of ["w1", "b1", "w2", "b2"]) {
        const off = this._weightOffsets[`ffn_${key}_${b}`];
        const size = key.includes("1")
          ? (key === "w1" ? D * this._ffnDim : this._ffnDim)
          : (key === "w2" ? this._ffnDim * D : D);
        const w: number[] = [];
        for (let i = 0; i < size; i++) {
          w.push(this._weights[off + i]);
        }
        blockWeights.push(w);
      }
      ffnWeights.push(blockWeights);
    }

    // Layer norm params
    const layerNormParams: number[][][] = [];
    for (let b = 0; b < this._numBlocks; b++) {
      const blockParams: number[][] = [];
      for (let ln = 1; ln <= 2; ln++) {
        const gOff = this._weightOffsets[`ln${ln}_g_${b}`];
        const bOff = this._weightOffsets[`ln${ln}_b_${b}`];
        const g: number[] = [];
        const beta: number[] = [];
        for (let d = 0; d < D; d++) {
          g.push(this._weights[gOff + d]);
          beta.push(this._weights[bOff + d]);
        }
        blockParams.push(g, beta);
      }
      layerNormParams.push(blockParams);
    }

    // Output weights
    const outputWeights: number[][][] = [];
    const outWOff = this._weightOffsets["out_w"];
    const outBOff = this._weightOffsets["out_b"];
    const outW: number[] = [];
    for (let i = 0; i < D * this._outputDim; i++) {
      outW.push(this._weights[outWOff + i]);
    }
    const outB: number[] = [];
    for (let i = 0; i < this._outputDim; i++) {
      outB.push(this._weights[outBOff + i]);
    }
    outputWeights.push([outW, outB]);

    // First and second moments
    const firstMoment: number[][][] = [[Array.from(this._firstMoment)]];
    const secondMoment: number[][][] = [[Array.from(this._secondMoment)]];

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

  /**
   * Get current normalization statistics
   *
   * @returns Running mean and standard deviation for inputs and outputs
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input mean: ${stats.inputMean}`);
   * console.log(`Sample count: ${stats.count}`);
   * ```
   */
  getNormalizationStats(): NormalizationStats {
    if (!this._isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    const inputStd = new Float64Array(this._inputDim);
    const outputStd = new Float64Array(this._outputDim);
    this._getStd(this._inputM2, inputStd, this._inputDim);
    this._getStd(this._outputM2, outputStd, this._outputDim);

    return {
      inputMean: Array.from(this._inputMean),
      inputStd: Array.from(inputStd),
      outputMean: Array.from(this._outputMean),
      outputStd: Array.from(outputStd),
      count: this._normCount,
    };
  }

  /**
   * Reset model to initial state
   *
   * Clears all learned weights, statistics, and history while preserving configuration.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now in its initial state
   * ```
   */
  reset(): void {
    this._isInitialized = false;
    this._inputDim = 0;
    this._outputDim = 0;
    this._seqLen = 0;
    this._normCount = 0;
    this._sampleCount = 0;
    this._totalLoss = 0;
    this._updateCount = 0;
    this._converged = false;
    this._driftCount = 0;
    this._prevLoss = Infinity;
    this._adwinWindow = [];
    this._inputHistory = [];
    this._activationCache.clear();
    this._bufferPool.clear();
    this._weightOffsets = {};
    this._totalParams = 0;
  }

  /**
   * Serialize model state to JSON string
   *
   * Saves all weights, optimizer state, and statistics for later restoration.
   *
   * @returns JSON string containing complete model state
   *
   * @example
   * ```typescript
   * const json = model.save();
   * localStorage.setItem('model', json);
   * ```
   */
  save(): string {
    const state: SerializedState = {
      config: {
        numBlocks: this._numBlocks,
        embeddingDim: this._embeddingDim,
        numHeads: this._numHeads,
        ffnMultiplier: this._ffnMultiplier,
        attentionDropout: this._attentionDropout,
        learningRate: this._learningRate,
        warmupSteps: this._warmupSteps,
        totalSteps: this._totalSteps,
        beta1: this._beta1,
        beta2: this._beta2,
        epsilon: this._epsilon,
        regularizationStrength: this._regularizationStrength,
        convergenceThreshold: this._convergenceThreshold,
        outlierThreshold: this._outlierThreshold,
        adwinDelta: this._adwinDelta,
        temporalScales: [...this._temporalScales],
        temporalKernelSize: this._temporalKernelSize,
        maxSequenceLength: this._maxSequenceLength,
        fusionDropout: this._fusionDropout,
      },
      isInitialized: this._isInitialized,
      inputDim: this._inputDim,
      outputDim: this._outputDim,
      seqLen: this._seqLen,
      weights: this._isInitialized ? Array.from(this._weights) : [],
      weightOffsets: { ...this._weightOffsets },
      inputMean: this._isInitialized ? Array.from(this._inputMean) : [],
      inputM2: this._isInitialized ? Array.from(this._inputM2) : [],
      outputMean: this._isInitialized ? Array.from(this._outputMean) : [],
      outputM2: this._isInitialized ? Array.from(this._outputM2) : [],
      normCount: this._normCount,
      sampleCount: this._sampleCount,
      totalLoss: this._totalLoss,
      updateCount: this._updateCount,
      converged: this._converged,
      driftCount: this._driftCount,
      adwinWindow: [...this._adwinWindow],
      firstMoment: this._isInitialized ? Array.from(this._firstMoment) : [],
      secondMoment: this._isInitialized ? Array.from(this._secondMoment) : [],
      inputHistory: this._inputHistory.map((arr) => Array.from(arr)),
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   *
   * Restores all weights, optimizer state, and statistics from a saved state.
   *
   * @param json - JSON string from save()
   *
   * @example
   * ```typescript
   * const json = localStorage.getItem('model');
   * if (json) {
   *   model.load(json);
   * }
   * ```
   */
  load(json: string): void {
    const state: SerializedState = JSON.parse(json);

    // Validate configuration compatibility
    if (
      state.config.numBlocks !== this._numBlocks ||
      state.config.embeddingDim !== this._embeddingDim ||
      state.config.numHeads !== this._numHeads
    ) {
      throw new Error(
        "Configuration mismatch: saved model has different architecture",
      );
    }

    this._isInitialized = state.isInitialized;
    this._inputDim = state.inputDim;
    this._outputDim = state.outputDim;
    this._seqLen = state.seqLen;

    if (state.isInitialized) {
      // Restore weight layout
      this._weightOffsets = { ...state.weightOffsets };
      this._totalParams = state.weights.length;

      // Allocate and restore arrays
      this._weights = new Float64Array(state.weights);
      this._gradients = new Float64Array(this._totalParams);
      this._firstMoment = new Float64Array(state.firstMoment);
      this._secondMoment = new Float64Array(state.secondMoment);

      this._inputMean = new Float64Array(state.inputMean);
      this._inputM2 = new Float64Array(state.inputM2);
      this._outputMean = new Float64Array(state.outputMean);
      this._outputM2 = new Float64Array(state.outputM2);

      // Restore buffers
      const maxBufSize = Math.max(
        this._seqLen * this._embeddingDim,
        this._seqLen * this._seqLen * this._numHeads,
        this._ffnDim * this._seqLen,
      );
      this._tempBuffer1 = new Float64Array(maxBufSize);
      this._tempBuffer2 = new Float64Array(maxBufSize);
      this._tempBuffer3 = new Float64Array(maxBufSize);
      this._attentionScores = new Float64Array(
        this._seqLen * this._seqLen * this._numHeads,
      );
      this._softmaxBuffer = new Float64Array(this._seqLen);
    }

    // Restore statistics
    this._normCount = state.normCount;
    this._sampleCount = state.sampleCount;
    this._totalLoss = state.totalLoss;
    this._updateCount = state.updateCount;
    this._converged = state.converged;
    this._driftCount = state.driftCount;
    this._adwinWindow = [...state.adwinWindow];

    // Restore input history
    this._inputHistory = state.inputHistory.map((arr) => new Float64Array(arr));
  }
}
