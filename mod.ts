/**
 * Fusion Temporal Transformer for Multivariate Regression
 *
 * Features:
 * - Incremental online learning with Adam optimizer
 * - Online z-score normalization using Welford's algorithm
 * - L2 regularization
 * - Outlier downweighting
 * - ADWIN drift detection
 * - Multi-scale temporal convolution
 * - Cross-scale fusion with gating
 * - Transformer blocks with multi-head self-attention
 * - Attention-weighted temporal pooling
 *
 * All computations use Float64Array for numerical precision.
 * Buffers are preallocated and reused to avoid hot-path allocations.
 *
 * Weight initialization: Xavier uniform for linear layers
 * limit = sqrt(6 / (fanIn + fanOut))
 */

// ============================================================================
// Type Definitions
// ============================================================================

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
  fusionWeights: number[][];
  attentionWeights: number[][][];
  ffnWeights: number[][][];
  layerNormParams: number[][];
  outputWeights: number[][];
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

// ============================================================================
// Internal Interfaces
// ============================================================================

interface WelfordStats {
  mean: Float64Array;
  m2: Float64Array;
  count: number;
}

interface AdamMoments {
  m: Float64Array;
  v: Float64Array;
}

// ============================================================================
// Helper Functions (Module-level for reuse)
// ============================================================================

/**
 * Deterministic xorshift32 PRNG
 * @param state - Current state (will be mutated)
 * @returns Random number in [0, 1)
 */
function xorshift32(state: { seed: number }): number {
  let x = state.seed;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  state.seed = x >>> 0;
  return (x >>> 0) / 4294967296;
}

/**
 * GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * @param x - Input value
 * @returns GELU(x)
 */
function gelu(x: number): number {
  const c = 0.7978845608028654; // sqrt(2/pi)
  const inner = c * (x + 0.044715 * x * x * x);
  // Clamp inner to prevent tanh overflow
  const clampedInner = Math.max(-20, Math.min(20, inner));
  return 0.5 * x * (1 + Math.tanh(clampedInner));
}

/**
 * GELU derivative: d/dx GELU(x)
 * @param x - Input value
 * @returns d(GELU)/dx
 */
function geluDerivative(x: number): number {
  const c = 0.7978845608028654;
  const k = 0.044715;
  const inner = c * (x + k * x * x * x);
  const clampedInner = Math.max(-20, Math.min(20, inner));
  const tanhVal = Math.tanh(clampedInner);
  const sech2 = 1 - tanhVal * tanhVal;
  const dinnerDx = c * (1 + 3 * k * x * x);
  return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * dinnerDx;
}

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 * @param x - Input value
 * @returns sigmoid(x)
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    const ez = Math.exp(-x);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(x);
    return ez / (1 + ez);
  }
}

// ============================================================================
// Main Class
// ============================================================================

export class FusionTemporalTransformerRegression {
  // Configuration
  private readonly config: Config;

  // Dimensions (set on first fitOnline)
  private inputDim: number = 0;
  private outputDim: number = 0;
  private seqLen: number = 0;
  private isInitialized: boolean = false;

  // Derived dimensions
  private headDim: number = 0;
  private ffnHiddenDim: number = 0;
  private nScales: number = 0;
  private fusionInputDim: number = 0;

  // Training state
  private sampleCount: number = 0;
  private updateCount: number = 0;
  private driftCount: number = 0;
  private runningLoss: number = 0;
  private converged: boolean = false;

  // Welford normalization stats
  private inputStats!: WelfordStats;
  private outputStats!: WelfordStats;

  // Residual variance for prediction uncertainty
  private residualM2!: Float64Array;
  private residualCount: number = 0;

  // ADWIN drift detection
  private readonly adwinWindowCap: number = 256;
  private readonly adwinMinSize: number = 32;
  private adwinBuffer!: Float64Array;
  private adwinHead: number = 0;
  private adwinSize: number = 0;

  // Cached input window for predict()
  private cachedWindow!: Float64Array; // seqLen x inputDim
  private cachedWindowLen: number = 0;

  // Positional encoding cache (maxSequenceLength x embeddingDim)
  private positionalEncodingCache!: Float64Array;

  // ========== WEIGHTS ==========
  // Temporal conv weights per scale: [kernelSize, inputDim, embeddingDim]
  private convWeights!: Float64Array[];
  private convBiases!: Float64Array[];

  // Scale embeddings: [nScales, embeddingDim]
  private scaleEmbeddings!: Float64Array;

  // Fusion gate weights: [fusionInputDim, fusionInputDim]
  private fusionWg!: Float64Array;
  private fusionBg!: Float64Array;

  // Transformer block weights (per block)
  // Each block: LN1 gamma/beta, Wq/Wk/Wv/Wo, LN2 gamma/beta, FFN W1/b1/W2/b2
  private ln1Gamma!: Float64Array[]; // [numBlocks][embeddingDim]
  private ln1Beta!: Float64Array[];
  private Wq!: Float64Array[]; // [numBlocks][embeddingDim x embeddingDim]
  private Wk!: Float64Array[];
  private Wv!: Float64Array[];
  private Wo!: Float64Array[];
  private ln2Gamma!: Float64Array[];
  private ln2Beta!: Float64Array[];
  private ffnW1!: Float64Array[]; // [numBlocks][embeddingDim x ffnHiddenDim]
  private ffnB1!: Float64Array[];
  private ffnW2!: Float64Array[]; // [numBlocks][ffnHiddenDim x embeddingDim]
  private ffnB2!: Float64Array[];

  // Temporal aggregation pooling
  private Wpool!: Float64Array; // embeddingDim
  private bpool: number = 0;

  // Output head
  private Wout!: Float64Array; // embeddingDim x outputDim
  private bout!: Float64Array; // outputDim

  // ========== ADAM MOMENTS ==========
  private convWeightsMoments!: AdamMoments[];
  private convBiasesMoments!: AdamMoments[];
  private scaleEmbeddingsMoments!: AdamMoments;
  private fusionWgMoments!: AdamMoments;
  private fusionBgMoments!: AdamMoments;
  private ln1GammaMoments!: AdamMoments[];
  private ln1BetaMoments!: AdamMoments[];
  private WqMoments!: AdamMoments[];
  private WkMoments!: AdamMoments[];
  private WvMoments!: AdamMoments[];
  private WoMoments!: AdamMoments[];
  private ln2GammaMoments!: AdamMoments[];
  private ln2BetaMoments!: AdamMoments[];
  private ffnW1Moments!: AdamMoments[];
  private ffnB1Moments!: AdamMoments[];
  private ffnW2Moments!: AdamMoments[];
  private ffnB2Moments!: AdamMoments[];
  private WpoolMoments!: AdamMoments;
  private bpoolMoments!: AdamMoments;
  private WoutMoments!: AdamMoments;
  private boutMoments!: AdamMoments;

  // ========== FORWARD CACHE ==========
  private normalizedInput!: Float64Array; // seqLen x inputDim
  private convOutputs!: Float64Array[]; // per scale: Ls x embeddingDim
  private scaleOutputLengths!: number[];
  private scaleEmbeddingsAdded!: Float64Array[]; // per scale: Ls x embeddingDim
  private upsampledScales!: Float64Array[]; // per scale: seqLen x embeddingDim
  private fusionConcat!: Float64Array; // seqLen x fusionInputDim
  private fusionGates!: Float64Array; // seqLen x fusionInputDim
  private fusedOutput!: Float64Array; // seqLen x embeddingDim
  private fusionDropoutMask!: Float64Array;

  // Transformer cache per block
  private blockInputs!: Float64Array[]; // [numBlocks+1][seqLen x embeddingDim]
  private ln1Outputs!: Float64Array[];
  private ln1Means!: Float64Array[]; // [numBlocks][seqLen]
  private ln1Vars!: Float64Array[];
  private QKV!: Float64Array[][]; // [numBlocks][3][seqLen x embeddingDim]
  private attnScores!: Float64Array[][]; // [numBlocks][numHeads][seqLen x seqLen]
  private attnProbs!: Float64Array[][]; // same shape
  private attnDropoutMasks!: Float64Array[][];
  private attnOutputs!: Float64Array[]; // [numBlocks][seqLen x embeddingDim]
  private attnProjected!: Float64Array[]; // after Wo
  private residual1!: Float64Array[];
  private ln2Outputs!: Float64Array[];
  private ln2Means!: Float64Array[];
  private ln2Vars!: Float64Array[];
  private ffnHidden!: Float64Array[];
  private ffnHiddenPreAct!: Float64Array[];
  private ffnOutputs!: Float64Array[];

  // Pooling cache
  private poolingScores!: Float64Array; // seqLen
  private poolingAlpha!: Float64Array; // seqLen
  private pooledOutput!: Float64Array; // embeddingDim
  private outputPred!: Float64Array; // outputDim (before denorm)

  // ========== GRADIENT BUFFERS ==========
  private gradConvWeights!: Float64Array[];
  private gradConvBiases!: Float64Array[];
  private gradScaleEmbeddings!: Float64Array;
  private gradFusionWg!: Float64Array;
  private gradFusionBg!: Float64Array;
  private gradLn1Gamma!: Float64Array[];
  private gradLn1Beta!: Float64Array[];
  private gradWq!: Float64Array[];
  private gradWk!: Float64Array[];
  private gradWv!: Float64Array[];
  private gradWo!: Float64Array[];
  private gradLn2Gamma!: Float64Array[];
  private gradLn2Beta!: Float64Array[];
  private gradFfnW1!: Float64Array[];
  private gradFfnB1!: Float64Array[];
  private gradFfnW2!: Float64Array[];
  private gradFfnB2!: Float64Array[];
  private gradWpool!: Float64Array;
  private gradBpool: number = 0;
  private gradWout!: Float64Array;
  private gradBout!: Float64Array;

  // Backward scratch buffers
  private dPooledOutput!: Float64Array;
  private dPoolingAlpha!: Float64Array;
  private dBlockOutput!: Float64Array;
  private dLn2Output!: Float64Array;
  private dFfnHidden!: Float64Array;
  private dFfnOutput!: Float64Array;
  private dResidual1!: Float64Array;
  private dAttnProjected!: Float64Array;
  private dAttnOutput!: Float64Array;
  private dAttnProbs!: Float64Array[];
  private dV!: Float64Array;
  private dQ!: Float64Array;
  private dK!: Float64Array;
  private dLn1Output!: Float64Array;
  private dFusedOutput!: Float64Array;
  private dFusionGates!: Float64Array;
  private dFusionConcat!: Float64Array;
  private dUpsampledScales!: Float64Array[];
  private dScaleEmbeddingsAdded!: Float64Array[];
  private dConvOutputs!: Float64Array[];

  // RNG state
  private rngState: { seed: number } = { seed: 12345 };

  /**
   * Creates a new FusionTemporalTransformerRegression instance
   * @param config - Optional partial configuration
   * @example
   * const model = new FusionTemporalTransformerRegression({ numBlocks: 4 });
   */
  constructor(config?: Partial<Config>) {
    this.config = {
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
      ...config,
    };

    // Validate numHeads divides embeddingDim
    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this.config.embeddingDim}) must be divisible by numHeads (${this.config.numHeads})`,
      );
    }

    this.nScales = this.config.temporalScales.length;
    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnHiddenDim = this.config.embeddingDim * this.config.ffnMultiplier;
    this.fusionInputDim = this.nScales * this.config.embeddingDim;

    // Initialize ADWIN buffer
    this.adwinBuffer = new Float64Array(this.adwinWindowCap);
  }

  /**
   * Performs one online training step
   * @param data - Training sample with xCoordinates (seqLen x inputDim) and yCoordinates (seqLen x outputDim)
   * @returns FitResult with loss, gradient norm, and training metrics
   * @example
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
   *   yCoordinates: [[0.1], [0.2], [0.3]]
   * });
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Initialize on first call
    if (!this.isInitialized) {
      this.initializeModel(xCoordinates, yCoordinates);
    }

    // Get actual sequence length for this sample
    const actualSeqLen = Math.min(
      xCoordinates.length,
      this.config.maxSequenceLength,
    );

    // Update Welford stats for normalization
    this.updateWelfordStats(xCoordinates, yCoordinates, actualSeqLen);

    // Cache the window for predict()
    this.cacheWindow(xCoordinates, actualSeqLen);

    // Get target (last timestep of yCoordinates)
    const targetY = yCoordinates[yCoordinates.length - 1];

    // Normalize input
    this.normalizeInput(xCoordinates, actualSeqLen);

    // Normalize target
    const normalizedTarget = new Float64Array(this.outputDim);
    for (let d = 0; d < this.outputDim; d++) {
      const std = Math.max(
        1e-12,
        Math.sqrt(this.outputStats.m2[d] / Math.max(1, this.outputStats.count)),
      );
      normalizedTarget[d] = (targetY[d] - this.outputStats.mean[d]) / std;
    }

    // Forward pass
    this.forward(actualSeqLen, true);

    // Compute loss and check outlier
    let loss = 0;
    let isOutlier = false;
    for (let d = 0; d < this.outputDim; d++) {
      const residual = normalizedTarget[d] - this.outputPred[d];
      loss += 0.5 * residual * residual;
      if (Math.abs(residual) > this.config.outlierThreshold) {
        isOutlier = true;
      }
    }
    loss /= this.outputDim;

    // Add L2 regularization to loss
    const l2Loss = this.computeL2Loss();
    loss += 0.5 * this.config.regularizationStrength * l2Loss;

    // Sample weight for outlier handling
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // Update running loss
    this.sampleCount++;
    this.runningLoss =
      (this.runningLoss * (this.sampleCount - 1) + loss * sampleWeight) /
      this.sampleCount;

    // Update residual variance for prediction uncertainty
    this.updateResidualVariance(normalizedTarget);

    // Backward pass
    const gradNorm = this.backward(
      normalizedTarget,
      actualSeqLen,
      sampleWeight,
    );

    // Compute learning rate
    const lr = this.computeLearningRate();

    // Adam update
    this.updateCount++;
    this.adamUpdate(lr);

    // Check convergence
    this.converged = gradNorm < this.config.convergenceThreshold;

    // ADWIN drift detection
    const driftDetected = this.adwinDetect(loss * sampleWeight);

    return {
      loss: loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier: isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected: driftDetected,
    };
  }

  /**
   * Generates predictions for future timesteps
   * @param futureSteps - Number of steps to predict
   * @returns PredictionResult with predictions and confidence bounds
   * @example
   * const result = model.predict(5);
   * console.log(result.predictions[0].predicted);
   */
  predict(futureSteps: number): PredictionResult {
    const isModelReady = this.isInitialized && this.sampleCount >= 2;

    if (!isModelReady || futureSteps <= 0) {
      return {
        predictions: [],
        accuracy: this.getAccuracy(),
        sampleCount: this.sampleCount,
        isModelReady: isModelReady,
      };
    }

    // Load cached window into normalizedInput
    this.normalizeFromCache();

    // Forward pass (no dropout)
    this.forward(this.cachedWindowLen, false);

    // Denormalize prediction
    const basePrediction = new Float64Array(this.outputDim);
    const baseStdError = new Float64Array(this.outputDim);

    for (let d = 0; d < this.outputDim; d++) {
      const std = Math.max(
        1e-12,
        Math.sqrt(this.outputStats.m2[d] / Math.max(1, this.outputStats.count)),
      );
      basePrediction[d] = this.outputPred[d] * std + this.outputStats.mean[d];

      // Standard error from residual variance
      const residualVar = this.residualCount > 1
        ? this.residualM2[d] / (this.residualCount - 1)
        : 1.0;
      baseStdError[d] = Math.sqrt(Math.max(0, residualVar)) * std;
    }

    // Generate predictions with increasing uncertainty
    const predictions: SinglePrediction[] = [];
    for (let step = 0; step < futureSteps; step++) {
      const multiplier = Math.sqrt(step + 1);
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let d = 0; d < this.outputDim; d++) {
        const se = baseStdError[d] * multiplier;
        predicted.push(basePrediction[d]);
        standardError.push(se);
        lowerBound.push(basePrediction[d] - 1.96 * se);
        upperBound.push(basePrediction[d] + 1.96 * se);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });
    }

    return {
      predictions: predictions,
      accuracy: this.getAccuracy(),
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Returns a summary of the model's current state
   * @returns ModelSummary with architecture and training info
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      temporalScales: [...this.config.temporalScales],
      totalParameters: this.isInitialized ? this.countParameters() : 0,
      sampleCount: this.sampleCount,
      accuracy: this.getAccuracy(),
      converged: this.converged,
      effectiveLearningRate: this.computeLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Returns all model weights and optimizer state
   * @returns WeightInfo with all weight matrices as number arrays
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

    // Pack temporal conv weights: [nScales][kernelSize * inputDim][embeddingDim]
    const temporalConvWeights: number[][][] = [];
    for (let s = 0; s < this.nScales; s++) {
      const w = this.unpack2D(
        this.convWeights[s],
        this.config.temporalKernelSize * this.inputDim,
        this.config.embeddingDim,
      );
      temporalConvWeights.push(w);
    }

    // Scale embeddings: [nScales][embeddingDim]
    const scaleEmbeddings = this.unpack2D(
      this.scaleEmbeddings,
      this.nScales,
      this.config.embeddingDim,
    );

    // Positional encoding: [maxSequenceLength][embeddingDim]
    const positionalEncoding = this.unpack2D(
      this.positionalEncodingCache,
      this.config.maxSequenceLength,
      this.config.embeddingDim,
    );

    // Fusion weights: [fusionInputDim][fusionInputDim] and bias
    const fusionWeights: number[][] = this.unpack2D(
      this.fusionWg,
      this.fusionInputDim,
      this.fusionInputDim,
    );
    fusionWeights.push(Array.from(this.fusionBg));

    // Attention weights per block: Wq, Wk, Wv, Wo flattened
    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const blockWeights: number[][] = [];
      blockWeights.push(
        ...this.unpack2D(
          this.Wq[b],
          this.config.embeddingDim,
          this.config.embeddingDim,
        ),
      );
      blockWeights.push(
        ...this.unpack2D(
          this.Wk[b],
          this.config.embeddingDim,
          this.config.embeddingDim,
        ),
      );
      blockWeights.push(
        ...this.unpack2D(
          this.Wv[b],
          this.config.embeddingDim,
          this.config.embeddingDim,
        ),
      );
      blockWeights.push(
        ...this.unpack2D(
          this.Wo[b],
          this.config.embeddingDim,
          this.config.embeddingDim,
        ),
      );
      attentionWeights.push(blockWeights);
    }

    // FFN weights per block: W1, b1, W2, b2
    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const blockWeights: number[][] = [];
      blockWeights.push(
        ...this.unpack2D(
          this.ffnW1[b],
          this.config.embeddingDim,
          this.ffnHiddenDim,
        ),
      );
      blockWeights.push(Array.from(this.ffnB1[b]));
      blockWeights.push(
        ...this.unpack2D(
          this.ffnW2[b],
          this.ffnHiddenDim,
          this.config.embeddingDim,
        ),
      );
      blockWeights.push(Array.from(this.ffnB2[b]));
      ffnWeights.push(blockWeights);
    }

    // Layer norm params: gamma and beta for LN1 and LN2 per block
    const layerNormParams: number[][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      layerNormParams.push(Array.from(this.ln1Gamma[b]));
      layerNormParams.push(Array.from(this.ln1Beta[b]));
      layerNormParams.push(Array.from(this.ln2Gamma[b]));
      layerNormParams.push(Array.from(this.ln2Beta[b]));
    }

    // Output weights: Wout and bout
    const outputWeights = this.unpack2D(
      this.Wout,
      this.config.embeddingDim,
      this.outputDim,
    );
    outputWeights.push(Array.from(this.bout));
    outputWeights.push(Array.from(this.Wpool));
    outputWeights.push([this.bpool]);

    // First and second moments (simplified - just conv and output for brevity)
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];
    for (let s = 0; s < this.nScales; s++) {
      firstMoment.push(this.unpack2D(
        this.convWeightsMoments[s].m,
        this.config.temporalKernelSize * this.inputDim,
        this.config.embeddingDim,
      ));
      secondMoment.push(this.unpack2D(
        this.convWeightsMoments[s].v,
        this.config.temporalKernelSize * this.inputDim,
        this.config.embeddingDim,
      ));
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
   * Returns current normalization statistics
   * @returns NormalizationStats with mean/std for inputs and outputs
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

    const inputStd: number[] = [];
    const outputStd: number[] = [];

    for (let i = 0; i < this.inputDim; i++) {
      const variance = this.inputStats.count > 0
        ? this.inputStats.m2[i] / this.inputStats.count
        : 0;
      inputStd.push(Math.sqrt(Math.max(1e-12, variance)));
    }

    for (let d = 0; d < this.outputDim; d++) {
      const variance = this.outputStats.count > 0
        ? this.outputStats.m2[d] / this.outputStats.count
        : 0;
      outputStd.push(Math.sqrt(Math.max(1e-12, variance)));
    }

    return {
      inputMean: Array.from(this.inputStats.mean),
      inputStd: inputStd,
      outputMean: Array.from(this.outputStats.mean),
      outputStd: outputStd,
      count: this.inputStats.count,
    };
  }

  /**
   * Resets model to initial state while preserving configuration
   */
  reset(): void {
    if (!this.isInitialized) return;

    this.sampleCount = 0;
    this.updateCount = 0;
    this.driftCount = 0;
    this.runningLoss = 0;
    this.converged = false;
    this.residualCount = 0;
    this.adwinHead = 0;
    this.adwinSize = 0;
    this.cachedWindowLen = 0;

    // Reset Welford stats
    this.inputStats.mean.fill(0);
    this.inputStats.m2.fill(0);
    this.inputStats.count = 0;
    this.outputStats.mean.fill(0);
    this.outputStats.m2.fill(0);
    this.outputStats.count = 0;
    this.residualM2.fill(0);

    // Reinitialize weights
    this.initializeWeights();

    // Reset all moments
    this.resetMoments();
  }

  /**
   * Serializes model state to JSON string
   * @returns JSON string containing full model state
   */
  save(): string {
    const state: any = {
      config: this.config,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      seqLen: this.seqLen,
      isInitialized: this.isInitialized,
      sampleCount: this.sampleCount,
      updateCount: this.updateCount,
      driftCount: this.driftCount,
      runningLoss: this.runningLoss,
      converged: this.converged,
      residualCount: this.residualCount,
      cachedWindowLen: this.cachedWindowLen,
    };

    if (this.isInitialized) {
      state.inputStats = {
        mean: Array.from(this.inputStats.mean),
        m2: Array.from(this.inputStats.m2),
        count: this.inputStats.count,
      };
      state.outputStats = {
        mean: Array.from(this.outputStats.mean),
        m2: Array.from(this.outputStats.m2),
        count: this.outputStats.count,
      };
      state.residualM2 = Array.from(this.residualM2);
      state.cachedWindow = Array.from(this.cachedWindow);

      // Save weights
      state.convWeights = this.convWeights.map((w) => Array.from(w));
      state.convBiases = this.convBiases.map((b) => Array.from(b));
      state.scaleEmbeddings = Array.from(this.scaleEmbeddings);
      state.fusionWg = Array.from(this.fusionWg);
      state.fusionBg = Array.from(this.fusionBg);
      state.ln1Gamma = this.ln1Gamma.map((g) => Array.from(g));
      state.ln1Beta = this.ln1Beta.map((b) => Array.from(b));
      state.Wq = this.Wq.map((w) => Array.from(w));
      state.Wk = this.Wk.map((w) => Array.from(w));
      state.Wv = this.Wv.map((w) => Array.from(w));
      state.Wo = this.Wo.map((w) => Array.from(w));
      state.ln2Gamma = this.ln2Gamma.map((g) => Array.from(g));
      state.ln2Beta = this.ln2Beta.map((b) => Array.from(b));
      state.ffnW1 = this.ffnW1.map((w) => Array.from(w));
      state.ffnB1 = this.ffnB1.map((b) => Array.from(b));
      state.ffnW2 = this.ffnW2.map((w) => Array.from(w));
      state.ffnB2 = this.ffnB2.map((b) => Array.from(b));
      state.Wpool = Array.from(this.Wpool);
      state.bpool = this.bpool;
      state.Wout = Array.from(this.Wout);
      state.bout = Array.from(this.bout);

      // Save moments
      state.convWeightsMoments = this.convWeightsMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.convBiasesMoments = this.convBiasesMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.scaleEmbeddingsMoments = {
        m: Array.from(this.scaleEmbeddingsMoments.m),
        v: Array.from(this.scaleEmbeddingsMoments.v),
      };
      state.fusionWgMoments = {
        m: Array.from(this.fusionWgMoments.m),
        v: Array.from(this.fusionWgMoments.v),
      };
      state.fusionBgMoments = {
        m: Array.from(this.fusionBgMoments.m),
        v: Array.from(this.fusionBgMoments.v),
      };
      state.ln1GammaMoments = this.ln1GammaMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ln1BetaMoments = this.ln1BetaMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.WqMoments = this.WqMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.WkMoments = this.WkMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.WvMoments = this.WvMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.WoMoments = this.WoMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ln2GammaMoments = this.ln2GammaMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ln2BetaMoments = this.ln2BetaMoments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ffnW1Moments = this.ffnW1Moments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ffnB1Moments = this.ffnB1Moments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ffnW2Moments = this.ffnW2Moments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.ffnB2Moments = this.ffnB2Moments.map((m) => ({
        m: Array.from(m.m),
        v: Array.from(m.v),
      }));
      state.WpoolMoments = {
        m: Array.from(this.WpoolMoments.m),
        v: Array.from(this.WpoolMoments.v),
      };
      state.bpoolMoments = {
        m: Array.from(this.bpoolMoments.m),
        v: Array.from(this.bpoolMoments.v),
      };
      state.WoutMoments = {
        m: Array.from(this.WoutMoments.m),
        v: Array.from(this.WoutMoments.v),
      };
      state.boutMoments = {
        m: Array.from(this.boutMoments.m),
        v: Array.from(this.boutMoments.v),
      };
    }

    return JSON.stringify(state);
  }

  /**
   * Restores model state from JSON string
   * @param w - JSON string from save()
   */
  load(w: string): void {
    const state = JSON.parse(w);

    // Restore config (immutable after construction, but we validate)
    Object.assign(this.config, state.config);

    this.inputDim = state.inputDim;
    this.outputDim = state.outputDim;
    this.seqLen = state.seqLen;
    this.isInitialized = state.isInitialized;
    this.sampleCount = state.sampleCount;
    this.updateCount = state.updateCount;
    this.driftCount = state.driftCount;
    this.runningLoss = state.runningLoss;
    this.converged = state.converged;
    this.residualCount = state.residualCount;
    this.cachedWindowLen = state.cachedWindowLen;

    if (state.isInitialized) {
      // Allocate buffers
      this.allocateBuffers();

      // Restore stats
      this.inputStats = {
        mean: new Float64Array(state.inputStats.mean),
        m2: new Float64Array(state.inputStats.m2),
        count: state.inputStats.count,
      };
      this.outputStats = {
        mean: new Float64Array(state.outputStats.mean),
        m2: new Float64Array(state.outputStats.m2),
        count: state.outputStats.count,
      };
      this.residualM2 = new Float64Array(state.residualM2);
      this.cachedWindow = new Float64Array(state.cachedWindow);

      // Restore weights
      for (let s = 0; s < this.nScales; s++) {
        this.convWeights[s] = new Float64Array(state.convWeights[s]);
        this.convBiases[s] = new Float64Array(state.convBiases[s]);
      }
      this.scaleEmbeddings = new Float64Array(state.scaleEmbeddings);
      this.fusionWg = new Float64Array(state.fusionWg);
      this.fusionBg = new Float64Array(state.fusionBg);

      for (let b = 0; b < this.config.numBlocks; b++) {
        this.ln1Gamma[b] = new Float64Array(state.ln1Gamma[b]);
        this.ln1Beta[b] = new Float64Array(state.ln1Beta[b]);
        this.Wq[b] = new Float64Array(state.Wq[b]);
        this.Wk[b] = new Float64Array(state.Wk[b]);
        this.Wv[b] = new Float64Array(state.Wv[b]);
        this.Wo[b] = new Float64Array(state.Wo[b]);
        this.ln2Gamma[b] = new Float64Array(state.ln2Gamma[b]);
        this.ln2Beta[b] = new Float64Array(state.ln2Beta[b]);
        this.ffnW1[b] = new Float64Array(state.ffnW1[b]);
        this.ffnB1[b] = new Float64Array(state.ffnB1[b]);
        this.ffnW2[b] = new Float64Array(state.ffnW2[b]);
        this.ffnB2[b] = new Float64Array(state.ffnB2[b]);
      }

      this.Wpool = new Float64Array(state.Wpool);
      this.bpool = state.bpool;
      this.Wout = new Float64Array(state.Wout);
      this.bout = new Float64Array(state.bout);

      // Restore moments
      for (let s = 0; s < this.nScales; s++) {
        this.convWeightsMoments[s] = {
          m: new Float64Array(state.convWeightsMoments[s].m),
          v: new Float64Array(state.convWeightsMoments[s].v),
        };
        this.convBiasesMoments[s] = {
          m: new Float64Array(state.convBiasesMoments[s].m),
          v: new Float64Array(state.convBiasesMoments[s].v),
        };
      }
      this.scaleEmbeddingsMoments = {
        m: new Float64Array(state.scaleEmbeddingsMoments.m),
        v: new Float64Array(state.scaleEmbeddingsMoments.v),
      };
      this.fusionWgMoments = {
        m: new Float64Array(state.fusionWgMoments.m),
        v: new Float64Array(state.fusionWgMoments.v),
      };
      this.fusionBgMoments = {
        m: new Float64Array(state.fusionBgMoments.m),
        v: new Float64Array(state.fusionBgMoments.v),
      };

      for (let b = 0; b < this.config.numBlocks; b++) {
        this.ln1GammaMoments[b] = {
          m: new Float64Array(state.ln1GammaMoments[b].m),
          v: new Float64Array(state.ln1GammaMoments[b].v),
        };
        this.ln1BetaMoments[b] = {
          m: new Float64Array(state.ln1BetaMoments[b].m),
          v: new Float64Array(state.ln1BetaMoments[b].v),
        };
        this.WqMoments[b] = {
          m: new Float64Array(state.WqMoments[b].m),
          v: new Float64Array(state.WqMoments[b].v),
        };
        this.WkMoments[b] = {
          m: new Float64Array(state.WkMoments[b].m),
          v: new Float64Array(state.WkMoments[b].v),
        };
        this.WvMoments[b] = {
          m: new Float64Array(state.WvMoments[b].m),
          v: new Float64Array(state.WvMoments[b].v),
        };
        this.WoMoments[b] = {
          m: new Float64Array(state.WoMoments[b].m),
          v: new Float64Array(state.WoMoments[b].v),
        };
        this.ln2GammaMoments[b] = {
          m: new Float64Array(state.ln2GammaMoments[b].m),
          v: new Float64Array(state.ln2GammaMoments[b].v),
        };
        this.ln2BetaMoments[b] = {
          m: new Float64Array(state.ln2BetaMoments[b].m),
          v: new Float64Array(state.ln2BetaMoments[b].v),
        };
        this.ffnW1Moments[b] = {
          m: new Float64Array(state.ffnW1Moments[b].m),
          v: new Float64Array(state.ffnW1Moments[b].v),
        };
        this.ffnB1Moments[b] = {
          m: new Float64Array(state.ffnB1Moments[b].m),
          v: new Float64Array(state.ffnB1Moments[b].v),
        };
        this.ffnW2Moments[b] = {
          m: new Float64Array(state.ffnW2Moments[b].m),
          v: new Float64Array(state.ffnW2Moments[b].v),
        };
        this.ffnB2Moments[b] = {
          m: new Float64Array(state.ffnB2Moments[b].m),
          v: new Float64Array(state.ffnB2Moments[b].v),
        };
      }

      this.WpoolMoments = {
        m: new Float64Array(state.WpoolMoments.m),
        v: new Float64Array(state.WpoolMoments.v),
      };
      this.bpoolMoments = {
        m: new Float64Array(state.bpoolMoments.m),
        v: new Float64Array(state.bpoolMoments.v),
      };
      this.WoutMoments = {
        m: new Float64Array(state.WoutMoments.m),
        v: new Float64Array(state.WoutMoments.v),
      };
      this.boutMoments = {
        m: new Float64Array(state.boutMoments.m),
        v: new Float64Array(state.boutMoments.v),
      };

      // Rebuild positional encoding cache
      this.buildPositionalEncoding();
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  /**
   * Initialize model on first fitOnline call
   */
  private initializeModel(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    this.inputDim = xCoordinates[0].length;
    this.outputDim = yCoordinates[0].length;
    this.seqLen = Math.min(xCoordinates.length, this.config.maxSequenceLength);

    this.allocateBuffers();
    this.initializeWeights();
    this.buildPositionalEncoding();

    this.isInitialized = true;
  }

  /**
   * Allocate all buffers based on dimensions
   */
  private allocateBuffers(): void {
    const maxSeq = this.config.maxSequenceLength;
    const embDim = this.config.embeddingDim;
    const numBlocks = this.config.numBlocks;
    const numHeads = this.config.numHeads;

    // Welford stats
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
    this.residualM2 = new Float64Array(this.outputDim);

    // Cached window
    this.cachedWindow = new Float64Array(maxSeq * this.inputDim);

    // Positional encoding
    this.positionalEncodingCache = new Float64Array(maxSeq * embDim);

    // Conv weights and biases
    this.convWeights = [];
    this.convBiases = [];
    this.convWeightsMoments = [];
    this.convBiasesMoments = [];
    this.convOutputs = [];
    this.scaleOutputLengths = [];
    this.scaleEmbeddingsAdded = [];
    this.upsampledScales = [];
    this.gradConvWeights = [];
    this.gradConvBiases = [];
    this.dConvOutputs = [];
    this.dUpsampledScales = [];
    this.dScaleEmbeddingsAdded = [];

    for (let s = 0; s < this.nScales; s++) {
      const convSize = this.config.temporalKernelSize * this.inputDim * embDim;
      this.convWeights.push(new Float64Array(convSize));
      this.convBiases.push(new Float64Array(embDim));
      this.convWeightsMoments.push({
        m: new Float64Array(convSize),
        v: new Float64Array(convSize),
      });
      this.convBiasesMoments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });
      this.gradConvWeights.push(new Float64Array(convSize));
      this.gradConvBiases.push(new Float64Array(embDim));

      const scaleLen = Math.ceil(maxSeq / this.config.temporalScales[s]);
      this.convOutputs.push(new Float64Array(scaleLen * embDim));
      this.scaleOutputLengths.push(scaleLen);
      this.scaleEmbeddingsAdded.push(new Float64Array(scaleLen * embDim));
      this.upsampledScales.push(new Float64Array(maxSeq * embDim));
      this.dConvOutputs.push(new Float64Array(scaleLen * embDim));
      this.dUpsampledScales.push(new Float64Array(maxSeq * embDim));
      this.dScaleEmbeddingsAdded.push(new Float64Array(scaleLen * embDim));
    }

    // Scale embeddings
    this.scaleEmbeddings = new Float64Array(this.nScales * embDim);
    this.scaleEmbeddingsMoments = {
      m: new Float64Array(this.nScales * embDim),
      v: new Float64Array(this.nScales * embDim),
    };
    this.gradScaleEmbeddings = new Float64Array(this.nScales * embDim);

    // Fusion
    this.fusionConcat = new Float64Array(maxSeq * this.fusionInputDim);
    this.fusionGates = new Float64Array(maxSeq * this.fusionInputDim);
    this.fusedOutput = new Float64Array(maxSeq * embDim);
    this.fusionDropoutMask = new Float64Array(maxSeq * embDim);
    this.fusionWg = new Float64Array(this.fusionInputDim * this.fusionInputDim);
    this.fusionBg = new Float64Array(this.fusionInputDim);
    this.fusionWgMoments = {
      m: new Float64Array(this.fusionInputDim * this.fusionInputDim),
      v: new Float64Array(this.fusionInputDim * this.fusionInputDim),
    };
    this.fusionBgMoments = {
      m: new Float64Array(this.fusionInputDim),
      v: new Float64Array(this.fusionInputDim),
    };
    this.gradFusionWg = new Float64Array(
      this.fusionInputDim * this.fusionInputDim,
    );
    this.gradFusionBg = new Float64Array(this.fusionInputDim);
    this.dFusedOutput = new Float64Array(maxSeq * embDim);
    this.dFusionGates = new Float64Array(maxSeq * this.fusionInputDim);
    this.dFusionConcat = new Float64Array(maxSeq * this.fusionInputDim);

    // Normalized input
    this.normalizedInput = new Float64Array(maxSeq * this.inputDim);

    // Transformer blocks
    this.blockInputs = [];
    this.ln1Outputs = [];
    this.ln1Means = [];
    this.ln1Vars = [];
    this.QKV = [];
    this.attnScores = [];
    this.attnProbs = [];
    this.attnDropoutMasks = [];
    this.attnOutputs = [];
    this.attnProjected = [];
    this.residual1 = [];
    this.ln2Outputs = [];
    this.ln2Means = [];
    this.ln2Vars = [];
    this.ffnHidden = [];
    this.ffnHiddenPreAct = [];
    this.ffnOutputs = [];

    this.ln1Gamma = [];
    this.ln1Beta = [];
    this.Wq = [];
    this.Wk = [];
    this.Wv = [];
    this.Wo = [];
    this.ln2Gamma = [];
    this.ln2Beta = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];

    this.ln1GammaMoments = [];
    this.ln1BetaMoments = [];
    this.WqMoments = [];
    this.WkMoments = [];
    this.WvMoments = [];
    this.WoMoments = [];
    this.ln2GammaMoments = [];
    this.ln2BetaMoments = [];
    this.ffnW1Moments = [];
    this.ffnB1Moments = [];
    this.ffnW2Moments = [];
    this.ffnB2Moments = [];

    this.gradLn1Gamma = [];
    this.gradLn1Beta = [];
    this.gradWq = [];
    this.gradWk = [];
    this.gradWv = [];
    this.gradWo = [];
    this.gradLn2Gamma = [];
    this.gradLn2Beta = [];
    this.gradFfnW1 = [];
    this.gradFfnB1 = [];
    this.gradFfnW2 = [];
    this.gradFfnB2 = [];

    this.dAttnProbs = [];

    for (let b = 0; b <= numBlocks; b++) {
      this.blockInputs.push(new Float64Array(maxSeq * embDim));
    }

    for (let b = 0; b < numBlocks; b++) {
      this.ln1Outputs.push(new Float64Array(maxSeq * embDim));
      this.ln1Means.push(new Float64Array(maxSeq));
      this.ln1Vars.push(new Float64Array(maxSeq));
      this.QKV.push([
        new Float64Array(maxSeq * embDim),
        new Float64Array(maxSeq * embDim),
        new Float64Array(maxSeq * embDim),
      ]);

      const headScores: Float64Array[] = [];
      const headProbs: Float64Array[] = [];
      const headMasks: Float64Array[] = [];
      for (let h = 0; h < numHeads; h++) {
        headScores.push(new Float64Array(maxSeq * maxSeq));
        headProbs.push(new Float64Array(maxSeq * maxSeq));
        headMasks.push(new Float64Array(maxSeq * maxSeq));
      }
      this.attnScores.push(headScores);
      this.attnProbs.push(headProbs);
      this.attnDropoutMasks.push(headMasks);
      this.dAttnProbs.push(new Float64Array(maxSeq * maxSeq));

      this.attnOutputs.push(new Float64Array(maxSeq * embDim));
      this.attnProjected.push(new Float64Array(maxSeq * embDim));
      this.residual1.push(new Float64Array(maxSeq * embDim));
      this.ln2Outputs.push(new Float64Array(maxSeq * embDim));
      this.ln2Means.push(new Float64Array(maxSeq));
      this.ln2Vars.push(new Float64Array(maxSeq));
      this.ffnHidden.push(new Float64Array(maxSeq * this.ffnHiddenDim));
      this.ffnHiddenPreAct.push(new Float64Array(maxSeq * this.ffnHiddenDim));
      this.ffnOutputs.push(new Float64Array(maxSeq * embDim));

      // Weights
      this.ln1Gamma.push(new Float64Array(embDim));
      this.ln1Beta.push(new Float64Array(embDim));
      this.Wq.push(new Float64Array(embDim * embDim));
      this.Wk.push(new Float64Array(embDim * embDim));
      this.Wv.push(new Float64Array(embDim * embDim));
      this.Wo.push(new Float64Array(embDim * embDim));
      this.ln2Gamma.push(new Float64Array(embDim));
      this.ln2Beta.push(new Float64Array(embDim));
      this.ffnW1.push(new Float64Array(embDim * this.ffnHiddenDim));
      this.ffnB1.push(new Float64Array(this.ffnHiddenDim));
      this.ffnW2.push(new Float64Array(this.ffnHiddenDim * embDim));
      this.ffnB2.push(new Float64Array(embDim));

      // Moments
      this.ln1GammaMoments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });
      this.ln1BetaMoments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });
      this.WqMoments.push({
        m: new Float64Array(embDim * embDim),
        v: new Float64Array(embDim * embDim),
      });
      this.WkMoments.push({
        m: new Float64Array(embDim * embDim),
        v: new Float64Array(embDim * embDim),
      });
      this.WvMoments.push({
        m: new Float64Array(embDim * embDim),
        v: new Float64Array(embDim * embDim),
      });
      this.WoMoments.push({
        m: new Float64Array(embDim * embDim),
        v: new Float64Array(embDim * embDim),
      });
      this.ln2GammaMoments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });
      this.ln2BetaMoments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });
      this.ffnW1Moments.push({
        m: new Float64Array(embDim * this.ffnHiddenDim),
        v: new Float64Array(embDim * this.ffnHiddenDim),
      });
      this.ffnB1Moments.push({
        m: new Float64Array(this.ffnHiddenDim),
        v: new Float64Array(this.ffnHiddenDim),
      });
      this.ffnW2Moments.push({
        m: new Float64Array(this.ffnHiddenDim * embDim),
        v: new Float64Array(this.ffnHiddenDim * embDim),
      });
      this.ffnB2Moments.push({
        m: new Float64Array(embDim),
        v: new Float64Array(embDim),
      });

      // Gradients
      this.gradLn1Gamma.push(new Float64Array(embDim));
      this.gradLn1Beta.push(new Float64Array(embDim));
      this.gradWq.push(new Float64Array(embDim * embDim));
      this.gradWk.push(new Float64Array(embDim * embDim));
      this.gradWv.push(new Float64Array(embDim * embDim));
      this.gradWo.push(new Float64Array(embDim * embDim));
      this.gradLn2Gamma.push(new Float64Array(embDim));
      this.gradLn2Beta.push(new Float64Array(embDim));
      this.gradFfnW1.push(new Float64Array(embDim * this.ffnHiddenDim));
      this.gradFfnB1.push(new Float64Array(this.ffnHiddenDim));
      this.gradFfnW2.push(new Float64Array(this.ffnHiddenDim * embDim));
      this.gradFfnB2.push(new Float64Array(embDim));
    }

    // Pooling
    this.poolingScores = new Float64Array(maxSeq);
    this.poolingAlpha = new Float64Array(maxSeq);
    this.pooledOutput = new Float64Array(embDim);
    this.Wpool = new Float64Array(embDim);
    this.WpoolMoments = {
      m: new Float64Array(embDim),
      v: new Float64Array(embDim),
    };
    this.gradWpool = new Float64Array(embDim);
    this.bpoolMoments = { m: new Float64Array(1), v: new Float64Array(1) };

    // Output head
    this.outputPred = new Float64Array(this.outputDim);
    this.Wout = new Float64Array(embDim * this.outputDim);
    this.bout = new Float64Array(this.outputDim);
    this.WoutMoments = {
      m: new Float64Array(embDim * this.outputDim),
      v: new Float64Array(embDim * this.outputDim),
    };
    this.boutMoments = {
      m: new Float64Array(this.outputDim),
      v: new Float64Array(this.outputDim),
    };
    this.gradWout = new Float64Array(embDim * this.outputDim);
    this.gradBout = new Float64Array(this.outputDim);

    // Backward scratch
    this.dPooledOutput = new Float64Array(embDim);
    this.dPoolingAlpha = new Float64Array(maxSeq);
    this.dBlockOutput = new Float64Array(maxSeq * embDim);
    this.dLn2Output = new Float64Array(maxSeq * embDim);
    this.dFfnHidden = new Float64Array(maxSeq * this.ffnHiddenDim);
    this.dFfnOutput = new Float64Array(maxSeq * embDim);
    this.dResidual1 = new Float64Array(maxSeq * embDim);
    this.dAttnProjected = new Float64Array(maxSeq * embDim);
    this.dAttnOutput = new Float64Array(maxSeq * embDim);
    this.dV = new Float64Array(maxSeq * embDim);
    this.dQ = new Float64Array(maxSeq * embDim);
    this.dK = new Float64Array(maxSeq * embDim);
    this.dLn1Output = new Float64Array(maxSeq * embDim);
  }

  /**
   * Initialize weights using Xavier uniform initialization
   * limit = sqrt(6 / (fanIn + fanOut))
   */
  private initializeWeights(): void {
    this.rngState.seed = 42;

    const embDim = this.config.embeddingDim;

    // Conv weights: Xavier
    for (let s = 0; s < this.nScales; s++) {
      const fanIn = this.config.temporalKernelSize * this.inputDim;
      const fanOut = embDim;
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      for (let i = 0; i < this.convWeights[s].length; i++) {
        this.convWeights[s][i] = (xorshift32(this.rngState) * 2 - 1) * limit;
      }
      this.convBiases[s].fill(0);
    }

    // Scale embeddings: small uniform ±0.02
    for (let i = 0; i < this.scaleEmbeddings.length; i++) {
      this.scaleEmbeddings[i] = (xorshift32(this.rngState) * 2 - 1) * 0.02;
    }

    // Fusion weights: Xavier
    {
      const fanIn = this.fusionInputDim;
      const fanOut = this.fusionInputDim;
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      for (let i = 0; i < this.fusionWg.length; i++) {
        this.fusionWg[i] = (xorshift32(this.rngState) * 2 - 1) * limit;
      }
      this.fusionBg.fill(0);
    }

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      // LayerNorm: gamma=1, beta=0
      this.ln1Gamma[b].fill(1);
      this.ln1Beta[b].fill(0);
      this.ln2Gamma[b].fill(1);
      this.ln2Beta[b].fill(0);

      // Attention weights: Xavier
      const attnLimit = Math.sqrt(6 / (embDim + embDim));
      for (let i = 0; i < embDim * embDim; i++) {
        this.Wq[b][i] = (xorshift32(this.rngState) * 2 - 1) * attnLimit;
        this.Wk[b][i] = (xorshift32(this.rngState) * 2 - 1) * attnLimit;
        this.Wv[b][i] = (xorshift32(this.rngState) * 2 - 1) * attnLimit;
        this.Wo[b][i] = (xorshift32(this.rngState) * 2 - 1) * attnLimit;
      }

      // FFN weights: Xavier
      const ffn1Limit = Math.sqrt(6 / (embDim + this.ffnHiddenDim));
      for (let i = 0; i < embDim * this.ffnHiddenDim; i++) {
        this.ffnW1[b][i] = (xorshift32(this.rngState) * 2 - 1) * ffn1Limit;
      }
      this.ffnB1[b].fill(0);

      const ffn2Limit = Math.sqrt(6 / (this.ffnHiddenDim + embDim));
      for (let i = 0; i < this.ffnHiddenDim * embDim; i++) {
        this.ffnW2[b][i] = (xorshift32(this.rngState) * 2 - 1) * ffn2Limit;
      }
      this.ffnB2[b].fill(0);
    }

    // Pooling weights
    const poolLimit = Math.sqrt(6 / (embDim + 1));
    for (let i = 0; i < embDim; i++) {
      this.Wpool[i] = (xorshift32(this.rngState) * 2 - 1) * poolLimit;
    }
    this.bpool = 0;

    // Output weights
    const outLimit = Math.sqrt(6 / (embDim + this.outputDim));
    for (let i = 0; i < embDim * this.outputDim; i++) {
      this.Wout[i] = (xorshift32(this.rngState) * 2 - 1) * outLimit;
    }
    this.bout.fill(0);
  }

  /**
   * Reset all Adam moments to zero
   */
  private resetMoments(): void {
    for (let s = 0; s < this.nScales; s++) {
      this.convWeightsMoments[s].m.fill(0);
      this.convWeightsMoments[s].v.fill(0);
      this.convBiasesMoments[s].m.fill(0);
      this.convBiasesMoments[s].v.fill(0);
    }
    this.scaleEmbeddingsMoments.m.fill(0);
    this.scaleEmbeddingsMoments.v.fill(0);
    this.fusionWgMoments.m.fill(0);
    this.fusionWgMoments.v.fill(0);
    this.fusionBgMoments.m.fill(0);
    this.fusionBgMoments.v.fill(0);

    for (let b = 0; b < this.config.numBlocks; b++) {
      this.ln1GammaMoments[b].m.fill(0);
      this.ln1GammaMoments[b].v.fill(0);
      this.ln1BetaMoments[b].m.fill(0);
      this.ln1BetaMoments[b].v.fill(0);
      this.WqMoments[b].m.fill(0);
      this.WqMoments[b].v.fill(0);
      this.WkMoments[b].m.fill(0);
      this.WkMoments[b].v.fill(0);
      this.WvMoments[b].m.fill(0);
      this.WvMoments[b].v.fill(0);
      this.WoMoments[b].m.fill(0);
      this.WoMoments[b].v.fill(0);
      this.ln2GammaMoments[b].m.fill(0);
      this.ln2GammaMoments[b].v.fill(0);
      this.ln2BetaMoments[b].m.fill(0);
      this.ln2BetaMoments[b].v.fill(0);
      this.ffnW1Moments[b].m.fill(0);
      this.ffnW1Moments[b].v.fill(0);
      this.ffnB1Moments[b].m.fill(0);
      this.ffnB1Moments[b].v.fill(0);
      this.ffnW2Moments[b].m.fill(0);
      this.ffnW2Moments[b].v.fill(0);
      this.ffnB2Moments[b].m.fill(0);
      this.ffnB2Moments[b].v.fill(0);
    }

    this.WpoolMoments.m.fill(0);
    this.WpoolMoments.v.fill(0);
    this.bpoolMoments.m.fill(0);
    this.bpoolMoments.v.fill(0);
    this.WoutMoments.m.fill(0);
    this.WoutMoments.v.fill(0);
    this.boutMoments.m.fill(0);
    this.boutMoments.v.fill(0);
  }

  /**
   * Build positional encoding cache
   * PE(pos, 2i) = sin(pos / 10000^(2i/d))
   * PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
   */
  private buildPositionalEncoding(): void {
    const maxSeq = this.config.maxSequenceLength;
    const embDim = this.config.embeddingDim;

    for (let pos = 0; pos < maxSeq; pos++) {
      for (let i = 0; i < embDim; i++) {
        const idx = pos * embDim + i;
        const dimIdx = Math.floor(i / 2);
        const angle = pos / Math.pow(10000, (2 * dimIdx) / embDim);
        if (i % 2 === 0) {
          this.positionalEncodingCache[idx] = Math.sin(angle);
        } else {
          this.positionalEncodingCache[idx] = Math.cos(angle);
        }
      }
    }
  }

  /**
   * Update Welford statistics for online normalization
   */
  private updateWelfordStats(
    xCoordinates: number[][],
    yCoordinates: number[][],
    actualSeqLen: number,
  ): void {
    // Update input stats for all timesteps
    for (let t = 0; t < actualSeqLen; t++) {
      this.inputStats.count++;
      const n = this.inputStats.count;
      for (let f = 0; f < this.inputDim; f++) {
        const x = xCoordinates[t][f];
        const delta = x - this.inputStats.mean[f];
        this.inputStats.mean[f] += delta / n;
        const delta2 = x - this.inputStats.mean[f];
        this.inputStats.m2[f] += delta * delta2;
      }
    }

    // Update output stats for all timesteps
    for (let t = 0; t < Math.min(yCoordinates.length, actualSeqLen); t++) {
      this.outputStats.count++;
      const n = this.outputStats.count;
      for (let d = 0; d < this.outputDim; d++) {
        const y = yCoordinates[t][d];
        const delta = y - this.outputStats.mean[d];
        this.outputStats.mean[d] += delta / n;
        const delta2 = y - this.outputStats.mean[d];
        this.outputStats.m2[d] += delta * delta2;
      }
    }
  }

  /**
   * Cache the input window for predict()
   */
  private cacheWindow(xCoordinates: number[][], actualSeqLen: number): void {
    this.cachedWindowLen = actualSeqLen;
    for (let t = 0; t < actualSeqLen; t++) {
      for (let f = 0; f < this.inputDim; f++) {
        this.cachedWindow[t * this.inputDim + f] = xCoordinates[t][f];
      }
    }
  }

  /**
   * Normalize input data using Welford stats
   */
  private normalizeInput(xCoordinates: number[][], actualSeqLen: number): void {
    for (let t = 0; t < actualSeqLen; t++) {
      for (let f = 0; f < this.inputDim; f++) {
        const variance = this.inputStats.count > 0
          ? this.inputStats.m2[f] / this.inputStats.count
          : 1;
        const std = Math.max(1e-12, Math.sqrt(variance));
        this.normalizedInput[t * this.inputDim + f] =
          (xCoordinates[t][f] - this.inputStats.mean[f]) / std;
      }
    }
  }

  /**
   * Normalize from cached window for prediction
   */
  private normalizeFromCache(): void {
    for (let t = 0; t < this.cachedWindowLen; t++) {
      for (let f = 0; f < this.inputDim; f++) {
        const variance = this.inputStats.count > 0
          ? this.inputStats.m2[f] / this.inputStats.count
          : 1;
        const std = Math.max(1e-12, Math.sqrt(variance));
        this.normalizedInput[t * this.inputDim + f] =
          (this.cachedWindow[t * this.inputDim + f] - this.inputStats.mean[f]) /
          std;
      }
    }
  }

  /**
   * Forward pass through the network
   */
  private forward(seqLen: number, training: boolean): void {
    const embDim = this.config.embeddingDim;
    const kernelSize = this.config.temporalKernelSize;

    // Set RNG seed for deterministic dropout
    this.rngState.seed = (this.updateCount * 12345 + 67890) >>> 0;

    // 1. Multi-scale temporal convolution
    for (let s = 0; s < this.nScales; s++) {
      const scale = this.config.temporalScales[s];
      const outLen = Math.ceil(seqLen / scale);
      this.scaleOutputLengths[s] = outLen;

      // Conv1D with causal padding
      for (let t = 0; t < outLen; t++) {
        const centerT = t * scale;
        for (let e = 0; e < embDim; e++) {
          let sum = this.convBiases[s][e];
          for (let k = 0; k < kernelSize; k++) {
            const inputT = centerT - k;
            if (inputT >= 0 && inputT < seqLen) {
              for (let f = 0; f < this.inputDim; f++) {
                const wIdx = (k * this.inputDim + f) * embDim + e;
                sum += this.normalizedInput[inputT * this.inputDim + f] *
                  this.convWeights[s][wIdx];
              }
            }
          }
          // GELU activation
          this.convOutputs[s][t * embDim + e] = gelu(sum);
        }
      }
    }

    // 2. Add scale embeddings and positional encoding
    for (let s = 0; s < this.nScales; s++) {
      const outLen = this.scaleOutputLengths[s];
      for (let t = 0; t < outLen; t++) {
        for (let e = 0; e < embDim; e++) {
          const idx = t * embDim + e;
          this.scaleEmbeddingsAdded[s][idx] = this.convOutputs[s][idx] +
            this.positionalEncodingCache[t * embDim + e] +
            this.scaleEmbeddings[s * embDim + e];
        }
      }
    }

    // 3. Upsample to fine scale length (seqLen)
    for (let s = 0; s < this.nScales; s++) {
      const scale = this.config.temporalScales[s];
      const outLen = this.scaleOutputLengths[s];
      for (let t = 0; t < seqLen; t++) {
        const srcT = Math.min(Math.floor(t / scale), outLen - 1);
        for (let e = 0; e < embDim; e++) {
          this.upsampledScales[s][t * embDim + e] =
            this.scaleEmbeddingsAdded[s][srcT * embDim + e];
        }
      }
    }

    // 4. Cross-scale fusion with gating
    // Concatenate all scales
    for (let t = 0; t < seqLen; t++) {
      for (let s = 0; s < this.nScales; s++) {
        for (let e = 0; e < embDim; e++) {
          this.fusionConcat[t * this.fusionInputDim + s * embDim + e] =
            this.upsampledScales[s][t * embDim + e];
        }
      }
    }

    // Compute gates: G = sigmoid(Concat * Wg + bg)
    for (let t = 0; t < seqLen; t++) {
      for (let o = 0; o < this.fusionInputDim; o++) {
        let sum = this.fusionBg[o];
        for (let i = 0; i < this.fusionInputDim; i++) {
          sum += this.fusionConcat[t * this.fusionInputDim + i] *
            this.fusionWg[i * this.fusionInputDim + o];
        }
        this.fusionGates[t * this.fusionInputDim + o] = sigmoid(sum);
      }
    }

    // Fused output: sum over scales of gated embeddings
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        let sum = 0;
        for (let s = 0; s < this.nScales; s++) {
          const gate =
            this.fusionGates[t * this.fusionInputDim + s * embDim + e];
          const value = this.upsampledScales[s][t * embDim + e];
          sum += gate * value;
        }
        this.fusedOutput[t * embDim + e] = sum;
      }
    }

    // Apply fusion dropout
    if (training && this.config.fusionDropout > 0) {
      for (let i = 0; i < seqLen * embDim; i++) {
        const keep = xorshift32(this.rngState) >= this.config.fusionDropout;
        this.fusionDropoutMask[i] = keep
          ? 1 / (1 - this.config.fusionDropout)
          : 0;
        this.fusedOutput[i] *= this.fusionDropoutMask[i];
      }
    } else {
      for (let i = 0; i < seqLen * embDim; i++) {
        this.fusionDropoutMask[i] = 1;
      }
    }

    // Copy to block input
    for (let i = 0; i < seqLen * embDim; i++) {
      this.blockInputs[0][i] = this.fusedOutput[i];
    }

    // 5. Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      this.transformerBlockForward(b, seqLen, training);
    }

    // 6. Temporal aggregation (attention-weighted pooling)
    const finalOut = this.blockInputs[this.config.numBlocks];

    // Compute scores
    let maxScore = -Infinity;
    for (let t = 0; t < seqLen; t++) {
      let score = this.bpool;
      for (let e = 0; e < embDim; e++) {
        score += finalOut[t * embDim + e] * this.Wpool[e];
      }
      this.poolingScores[t] = score;
      if (score > maxScore) maxScore = score;
    }

    // Softmax
    let sumExp = 0;
    for (let t = 0; t < seqLen; t++) {
      const exp = Math.exp(this.poolingScores[t] - maxScore);
      this.poolingAlpha[t] = exp;
      sumExp += exp;
    }
    if (sumExp < 1e-12) sumExp = 1e-12;
    for (let t = 0; t < seqLen; t++) {
      this.poolingAlpha[t] /= sumExp;
    }

    // Weighted sum
    this.pooledOutput.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        this.pooledOutput[e] += this.poolingAlpha[t] * finalOut[t * embDim + e];
      }
    }

    // 7. Output head
    for (let d = 0; d < this.outputDim; d++) {
      let sum = this.bout[d];
      for (let e = 0; e < embDim; e++) {
        sum += this.pooledOutput[e] * this.Wout[e * this.outputDim + d];
      }
      this.outputPred[d] = sum;
    }
  }

  /**
   * Forward pass through a single transformer block
   */
  private transformerBlockForward(
    blockIdx: number,
    seqLen: number,
    training: boolean,
  ): void {
    const embDim = this.config.embeddingDim;
    const numHeads = this.config.numHeads;
    const headDim = this.headDim;
    const input = this.blockInputs[blockIdx];

    // LayerNorm 1
    this.layerNormForward(
      input,
      seqLen,
      embDim,
      this.ln1Gamma[blockIdx],
      this.ln1Beta[blockIdx],
      this.ln1Outputs[blockIdx],
      this.ln1Means[blockIdx],
      this.ln1Vars[blockIdx],
    );

    const ln1Out = this.ln1Outputs[blockIdx];

    // Compute Q, K, V
    this.linearForward(
      ln1Out,
      seqLen,
      embDim,
      embDim,
      this.Wq[blockIdx],
      this.QKV[blockIdx][0],
    );
    this.linearForward(
      ln1Out,
      seqLen,
      embDim,
      embDim,
      this.Wk[blockIdx],
      this.QKV[blockIdx][1],
    );
    this.linearForward(
      ln1Out,
      seqLen,
      embDim,
      embDim,
      this.Wv[blockIdx],
      this.QKV[blockIdx][2],
    );

    const Q = this.QKV[blockIdx][0];
    const K = this.QKV[blockIdx][1];
    const V = this.QKV[blockIdx][2];

    // Multi-head attention
    const attnOut = this.attnOutputs[blockIdx];
    attnOut.fill(0);

    const scale = 1 / Math.sqrt(headDim);

    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scores = this.attnScores[blockIdx][h];
      const probs = this.attnProbs[blockIdx][h];
      const mask = this.attnDropoutMasks[blockIdx][h];

      // Compute attention scores with causal mask
      for (let i = 0; i < seqLen; i++) {
        let maxScore = -Infinity;
        for (let j = 0; j < seqLen; j++) {
          if (j > i) {
            scores[i * seqLen + j] = -1e9;
          } else {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += Q[i * embDim + headOffset + d] *
                K[j * embDim + headOffset + d];
            }
            scores[i * seqLen + j] = dot * scale;
          }
          if (scores[i * seqLen + j] > maxScore) {
            maxScore = scores[i * seqLen + j];
          }
        }

        // Softmax
        let sumExp = 0;
        for (let j = 0; j < seqLen; j++) {
          const exp = Math.exp(scores[i * seqLen + j] - maxScore);
          probs[i * seqLen + j] = exp;
          sumExp += exp;
        }
        if (sumExp < 1e-12) sumExp = 1e-12;
        for (let j = 0; j < seqLen; j++) {
          probs[i * seqLen + j] /= sumExp;
        }

        // Attention dropout
        if (training && this.config.attentionDropout > 0) {
          for (let j = 0; j < seqLen; j++) {
            const keep =
              xorshift32(this.rngState) >= this.config.attentionDropout;
            mask[i * seqLen + j] = keep
              ? 1 / (1 - this.config.attentionDropout)
              : 0;
            probs[i * seqLen + j] *= mask[i * seqLen + j];
          }
        } else {
          for (let j = 0; j < seqLen; j++) {
            mask[i * seqLen + j] = 1;
          }
        }

        // Compute weighted sum of values
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += probs[i * seqLen + j] * V[j * embDim + headOffset + d];
          }
          attnOut[i * embDim + headOffset + d] = sum;
        }
      }
    }

    // Project with Wo
    this.linearForward(
      attnOut,
      seqLen,
      embDim,
      embDim,
      this.Wo[blockIdx],
      this.attnProjected[blockIdx],
    );

    // Residual connection
    for (let i = 0; i < seqLen * embDim; i++) {
      this.residual1[blockIdx][i] = input[i] + this.attnProjected[blockIdx][i];
    }

    // LayerNorm 2
    this.layerNormForward(
      this.residual1[blockIdx],
      seqLen,
      embDim,
      this.ln2Gamma[blockIdx],
      this.ln2Beta[blockIdx],
      this.ln2Outputs[blockIdx],
      this.ln2Means[blockIdx],
      this.ln2Vars[blockIdx],
    );

    // FFN
    const ln2Out = this.ln2Outputs[blockIdx];

    // First linear + GELU
    for (let t = 0; t < seqLen; t++) {
      for (let h = 0; h < this.ffnHiddenDim; h++) {
        let sum = this.ffnB1[blockIdx][h];
        for (let e = 0; e < embDim; e++) {
          sum += ln2Out[t * embDim + e] *
            this.ffnW1[blockIdx][e * this.ffnHiddenDim + h];
        }
        this.ffnHiddenPreAct[blockIdx][t * this.ffnHiddenDim + h] = sum;
        this.ffnHidden[blockIdx][t * this.ffnHiddenDim + h] = gelu(sum);
      }
    }

    // Second linear
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        let sum = this.ffnB2[blockIdx][e];
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          sum += this.ffnHidden[blockIdx][t * this.ffnHiddenDim + h] *
            this.ffnW2[blockIdx][h * embDim + e];
        }
        this.ffnOutputs[blockIdx][t * embDim + e] = sum;
      }
    }

    // Residual connection
    const output = this.blockInputs[blockIdx + 1];
    for (let i = 0; i < seqLen * embDim; i++) {
      output[i] = this.residual1[blockIdx][i] + this.ffnOutputs[blockIdx][i];
    }
  }

  /**
   * LayerNorm forward pass
   * output = gamma * (x - mean) / sqrt(var + eps) + beta
   */
  private layerNormForward(
    input: Float64Array,
    seqLen: number,
    dim: number,
    gamma: Float64Array,
    beta: Float64Array,
    output: Float64Array,
    means: Float64Array,
    vars: Float64Array,
  ): void {
    const eps = 1e-12;

    for (let t = 0; t < seqLen; t++) {
      // Compute mean
      let mean = 0;
      for (let d = 0; d < dim; d++) {
        mean += input[t * dim + d];
      }
      mean /= dim;
      means[t] = mean;

      // Compute variance
      let variance = 0;
      for (let d = 0; d < dim; d++) {
        const diff = input[t * dim + d] - mean;
        variance += diff * diff;
      }
      variance /= dim;
      vars[t] = variance;

      // Normalize
      const invStd = 1 / Math.sqrt(Math.max(eps, variance));
      for (let d = 0; d < dim; d++) {
        output[t * dim + d] = gamma[d] * (input[t * dim + d] - mean) * invStd +
          beta[d];
      }
    }
  }

  /**
   * Linear layer forward (no bias)
   * output[t, o] = sum_i input[t, i] * W[i, o]
   */
  private linearForward(
    input: Float64Array,
    seqLen: number,
    inDim: number,
    outDim: number,
    W: Float64Array,
    output: Float64Array,
  ): void {
    for (let t = 0; t < seqLen; t++) {
      for (let o = 0; o < outDim; o++) {
        let sum = 0;
        for (let i = 0; i < inDim; i++) {
          sum += input[t * inDim + i] * W[i * outDim + o];
        }
        output[t * outDim + o] = sum;
      }
    }
  }

  /**
   * Compute L2 regularization loss
   */
  private computeL2Loss(): number {
    let sum = 0;

    // Conv weights
    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.convWeights[s].length; i++) {
        sum += this.convWeights[s][i] * this.convWeights[s][i];
      }
    }

    // Scale embeddings
    for (let i = 0; i < this.scaleEmbeddings.length; i++) {
      sum += this.scaleEmbeddings[i] * this.scaleEmbeddings[i];
    }

    // Fusion weights
    for (let i = 0; i < this.fusionWg.length; i++) {
      sum += this.fusionWg[i] * this.fusionWg[i];
    }

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.Wq[b].length; i++) {
        sum += this.Wq[b][i] * this.Wq[b][i];
        sum += this.Wk[b][i] * this.Wk[b][i];
        sum += this.Wv[b][i] * this.Wv[b][i];
        sum += this.Wo[b][i] * this.Wo[b][i];
      }
      for (let i = 0; i < this.ffnW1[b].length; i++) {
        sum += this.ffnW1[b][i] * this.ffnW1[b][i];
      }
      for (let i = 0; i < this.ffnW2[b].length; i++) {
        sum += this.ffnW2[b][i] * this.ffnW2[b][i];
      }
    }

    // Output weights
    for (let i = 0; i < this.Wout.length; i++) {
      sum += this.Wout[i] * this.Wout[i];
    }
    for (let i = 0; i < this.Wpool.length; i++) {
      sum += this.Wpool[i] * this.Wpool[i];
    }

    return sum;
  }

  /**
   * Update residual variance for prediction uncertainty
   */
  private updateResidualVariance(normalizedTarget: Float64Array): void {
    this.residualCount++;
    for (let d = 0; d < this.outputDim; d++) {
      const residual = normalizedTarget[d] - this.outputPred[d];
      // Welford update for residual variance
      const delta = residual * residual -
        (this.residualCount > 1
          ? this.residualM2[d] / (this.residualCount - 1)
          : 0);
      this.residualM2[d] += delta;
    }
  }

  /**
   * Backward pass through the network
   * @returns gradient norm
   */
  private backward(
    normalizedTarget: Float64Array,
    seqLen: number,
    sampleWeight: number,
  ): number {
    const embDim = this.config.embeddingDim;
    const lambda = this.config.regularizationStrength;

    // Zero all gradients
    this.zeroGradients();

    // d(loss)/d(outputPred) = (outputPred - target) / outputDim * sampleWeight
    for (let d = 0; d < this.outputDim; d++) {
      this.gradBout[d] = (this.outputPred[d] - normalizedTarget[d]) /
        this.outputDim * sampleWeight;
    }

    // Backward through output head
    // output = pooled * Wout + bout
    this.dPooledOutput.fill(0);
    for (let e = 0; e < embDim; e++) {
      for (let d = 0; d < this.outputDim; d++) {
        this.gradWout[e * this.outputDim + d] += this.pooledOutput[e] *
          this.gradBout[d];
        this.dPooledOutput[e] += this.Wout[e * this.outputDim + d] *
          this.gradBout[d];
      }
    }

    // Backward through pooling softmax
    // pooled[e] = sum_t alpha[t] * H[t, e]
    // d(pooled)/d(H[t,e]) = alpha[t]
    // d(pooled)/d(alpha[t]) = H[t, e]
    const finalOut = this.blockInputs[this.config.numBlocks];

    // d(loss)/d(alpha) before softmax
    this.dPoolingAlpha.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        this.dPoolingAlpha[t] += this.dPooledOutput[e] *
          finalOut[t * embDim + e];
      }
    }

    // Softmax backward
    // d(loss)/d(scores[i]) = sum_j alpha[j] * (d(loss)/d(alpha[j])) * (delta[i,j] - alpha[i])
    let dotProd = 0;
    for (let t = 0; t < seqLen; t++) {
      dotProd += this.poolingAlpha[t] * this.dPoolingAlpha[t];
    }

    // d(loss)/d(H) through alpha
    this.dBlockOutput.fill(0);
    for (let t = 0; t < seqLen; t++) {
      const dScore = this.poolingAlpha[t] * (this.dPoolingAlpha[t] - dotProd);

      // score[t] = dot(H[t], Wpool) + bpool
      this.gradBpool += dScore;
      for (let e = 0; e < embDim; e++) {
        this.gradWpool[e] += finalOut[t * embDim + e] * dScore;
        this.dBlockOutput[t * embDim + e] += this.Wpool[e] * dScore;
      }

      // Direct contribution: pooled[e] = sum_t alpha[t] * H[t, e]
      for (let e = 0; e < embDim; e++) {
        this.dBlockOutput[t * embDim + e] += this.poolingAlpha[t] *
          this.dPooledOutput[e];
      }
    }

    // Backward through transformer blocks
    for (let b = this.config.numBlocks - 1; b >= 0; b--) {
      this.transformerBlockBackward(b, seqLen);
    }

    // Backward through fusion
    this.fusionBackward(seqLen);

    // Backward through conv
    this.convBackward(seqLen);

    // Add L2 regularization gradients
    this.addL2Gradients(lambda);

    // Compute gradient norm and clip if needed
    let gradNorm = this.computeGradientNorm();

    // Clip gradients if norm > 5.0
    if (gradNorm > 5.0) {
      const scale = 5.0 / gradNorm;
      this.scaleGradients(scale);
      gradNorm = 5.0;
    }

    return gradNorm;
  }

  /**
   * Zero all gradient buffers
   */
  private zeroGradients(): void {
    for (let s = 0; s < this.nScales; s++) {
      this.gradConvWeights[s].fill(0);
      this.gradConvBiases[s].fill(0);
    }
    this.gradScaleEmbeddings.fill(0);
    this.gradFusionWg.fill(0);
    this.gradFusionBg.fill(0);

    for (let b = 0; b < this.config.numBlocks; b++) {
      this.gradLn1Gamma[b].fill(0);
      this.gradLn1Beta[b].fill(0);
      this.gradWq[b].fill(0);
      this.gradWk[b].fill(0);
      this.gradWv[b].fill(0);
      this.gradWo[b].fill(0);
      this.gradLn2Gamma[b].fill(0);
      this.gradLn2Beta[b].fill(0);
      this.gradFfnW1[b].fill(0);
      this.gradFfnB1[b].fill(0);
      this.gradFfnW2[b].fill(0);
      this.gradFfnB2[b].fill(0);
    }

    this.gradWpool.fill(0);
    this.gradBpool = 0;
    this.gradWout.fill(0);
    this.gradBout.fill(0);
  }

  /**
   * Backward through transformer block
   */
  private transformerBlockBackward(blockIdx: number, seqLen: number): void {
    const embDim = this.config.embeddingDim;
    const numHeads = this.config.numHeads;
    const headDim = this.headDim;

    // dBlockOutput is gradient w.r.t. blockInputs[blockIdx + 1]
    // residual2: output = residual1 + ffnOutput

    // Gradient flows to both residual1 and ffnOutput
    // dFFNOutput = dBlockOutput
    // dResidual1 += dBlockOutput

    // Backward through FFN
    // ffnOutput[t, e] = sum_h ffnHidden[t, h] * ffnW2[h, e] + ffnB2[e]
    this.dFfnOutput.fill(0);
    for (let i = 0; i < seqLen * embDim; i++) {
      this.dFfnOutput[i] = this.dBlockOutput[i];
      this.dResidual1[i] = this.dBlockOutput[i];
    }

    // Backward through second linear
    this.dFfnHidden.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        const dOut = this.dFfnOutput[t * embDim + e];
        this.gradFfnB2[blockIdx][e] += dOut;
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          this.gradFfnW2[blockIdx][h * embDim + e] +=
            this.ffnHidden[blockIdx][t * this.ffnHiddenDim + h] * dOut;
          this.dFfnHidden[t * this.ffnHiddenDim + h] +=
            this.ffnW2[blockIdx][h * embDim + e] * dOut;
        }
      }
    }

    // Backward through GELU
    for (let i = 0; i < seqLen * this.ffnHiddenDim; i++) {
      this.dFfnHidden[i] *= geluDerivative(this.ffnHiddenPreAct[blockIdx][i]);
    }

    // Backward through first linear
    this.dLn2Output.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let h = 0; h < this.ffnHiddenDim; h++) {
        const dHidden = this.dFfnHidden[t * this.ffnHiddenDim + h];
        this.gradFfnB1[blockIdx][h] += dHidden;
        for (let e = 0; e < embDim; e++) {
          this.gradFfnW1[blockIdx][e * this.ffnHiddenDim + h] +=
            this.ln2Outputs[blockIdx][t * embDim + e] * dHidden;
          this.dLn2Output[t * embDim + e] +=
            this.ffnW1[blockIdx][e * this.ffnHiddenDim + h] * dHidden;
        }
      }
    }

    // Backward through LayerNorm 2
    this.layerNormBackward(
      this.dLn2Output,
      this.residual1[blockIdx],
      this.ln2Means[blockIdx],
      this.ln2Vars[blockIdx],
      this.ln2Gamma[blockIdx],
      seqLen,
      embDim,
      this.gradLn2Gamma[blockIdx],
      this.gradLn2Beta[blockIdx],
      this.dResidual1, // accumulate
    );

    // Now dResidual1 has gradient w.r.t. residual1
    // residual1 = input + attnProjected

    // Backward through attention projection
    this.dAttnProjected.fill(0);
    for (let i = 0; i < seqLen * embDim; i++) {
      this.dAttnProjected[i] = this.dResidual1[i];
    }

    // Backward through Wo
    this.dAttnOutput.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let o = 0; o < embDim; o++) {
        const dProj = this.dAttnProjected[t * embDim + o];
        for (let i = 0; i < embDim; i++) {
          this.gradWo[blockIdx][i * embDim + o] +=
            this.attnOutputs[blockIdx][t * embDim + i] * dProj;
          this.dAttnOutput[t * embDim + i] +=
            this.Wo[blockIdx][i * embDim + o] * dProj;
        }
      }
    }

    // Backward through multi-head attention
    this.dQ.fill(0);
    this.dK.fill(0);
    this.dV.fill(0);

    const scale = 1 / Math.sqrt(headDim);

    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const probs = this.attnProbs[blockIdx][h];
      const scores = this.attnScores[blockIdx][h];
      const mask = this.attnDropoutMasks[blockIdx][h];
      const Q = this.QKV[blockIdx][0];
      const K = this.QKV[blockIdx][1];
      const V = this.QKV[blockIdx][2];

      // Backward from attnOut to probs and V
      // attnOut[i, d] = sum_j probs[i, j] * V[j, d]
      const dProbs = this.dAttnProbs[blockIdx];
      dProbs.fill(0);

      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          const dOut = this.dAttnOutput[i * embDim + headOffset + d];
          for (let j = 0; j < seqLen; j++) {
            dProbs[i * seqLen + j] += dOut * V[j * embDim + headOffset + d];
            this.dV[j * embDim + headOffset + d] += probs[i * seqLen + j] *
              dOut;
          }
        }
      }

      // Backward through dropout
      for (let i = 0; i < seqLen * seqLen; i++) {
        dProbs[i] *= mask[i];
      }

      // Backward through softmax
      for (let i = 0; i < seqLen; i++) {
        let dot = 0;
        for (let j = 0; j < seqLen; j++) {
          // Need pre-dropout probs for softmax backward
          const pNoDropout = probs[i * seqLen + j] /
            (mask[i * seqLen + j] || 1);
          dot += pNoDropout * dProbs[i * seqLen + j];
        }

        for (let j = 0; j < seqLen; j++) {
          if (scores[i * seqLen + j] > -1e8) { // not masked
            const pNoDropout = probs[i * seqLen + j] /
              (mask[i * seqLen + j] || 1);
            const dScore = pNoDropout * (dProbs[i * seqLen + j] - dot) * scale;

            // score = Q[i] · K[j] / sqrt(d)
            // dQ[i] += dScore * K[j]
            // dK[j] += dScore * Q[i]
            for (let d = 0; d < headDim; d++) {
              this.dQ[i * embDim + headOffset + d] += dScore *
                K[j * embDim + headOffset + d];
              this.dK[j * embDim + headOffset + d] += dScore *
                Q[i * embDim + headOffset + d];
            }
          }
        }
      }
    }

    // Backward through Q, K, V projections
    this.dLn1Output.fill(0);

    // dQ = dLn1 * Wq^T
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        for (let o = 0; o < embDim; o++) {
          this.gradWq[blockIdx][i * embDim + o] +=
            this.ln1Outputs[blockIdx][t * embDim + i] * this.dQ[t * embDim + o];
          this.dLn1Output[t * embDim + i] += this.Wq[blockIdx][i * embDim + o] *
            this.dQ[t * embDim + o];
        }
      }
    }

    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        for (let o = 0; o < embDim; o++) {
          this.gradWk[blockIdx][i * embDim + o] +=
            this.ln1Outputs[blockIdx][t * embDim + i] * this.dK[t * embDim + o];
          this.dLn1Output[t * embDim + i] += this.Wk[blockIdx][i * embDim + o] *
            this.dK[t * embDim + o];
        }
      }
    }

    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < embDim; i++) {
        for (let o = 0; o < embDim; o++) {
          this.gradWv[blockIdx][i * embDim + o] +=
            this.ln1Outputs[blockIdx][t * embDim + i] * this.dV[t * embDim + o];
          this.dLn1Output[t * embDim + i] += this.Wv[blockIdx][i * embDim + o] *
            this.dV[t * embDim + o];
        }
      }
    }

    // Backward through LayerNorm 1
    // Clear dBlockOutput for next iteration to receive gradients
    const dInput = this.dBlockOutput;
    dInput.fill(0);

    // Add residual gradient
    for (let i = 0; i < seqLen * embDim; i++) {
      dInput[i] = this.dResidual1[i];
    }

    this.layerNormBackward(
      this.dLn1Output,
      this.blockInputs[blockIdx],
      this.ln1Means[blockIdx],
      this.ln1Vars[blockIdx],
      this.ln1Gamma[blockIdx],
      seqLen,
      embDim,
      this.gradLn1Gamma[blockIdx],
      this.gradLn1Beta[blockIdx],
      dInput, // accumulate
    );
  }

  /**
   * LayerNorm backward
   */
  private layerNormBackward(
    dOut: Float64Array,
    input: Float64Array,
    means: Float64Array,
    vars: Float64Array,
    gamma: Float64Array,
    seqLen: number,
    dim: number,
    dGamma: Float64Array,
    dBeta: Float64Array,
    dInput: Float64Array, // will accumulate
  ): void {
    const eps = 1e-12;

    for (let t = 0; t < seqLen; t++) {
      const mean = means[t];
      const variance = vars[t];
      const invStd = 1 / Math.sqrt(Math.max(eps, variance));

      // Compute intermediate values
      let dVar = 0;
      let dMean = 0;

      for (let d = 0; d < dim; d++) {
        const xHat = (input[t * dim + d] - mean) * invStd;
        dGamma[d] += dOut[t * dim + d] * xHat;
        dBeta[d] += dOut[t * dim + d];

        const dxHat = dOut[t * dim + d] * gamma[d];
        dVar += dxHat * (input[t * dim + d] - mean) * (-0.5) *
          Math.pow(variance + eps, -1.5);
        dMean += dxHat * (-invStd);
      }

      dMean += dVar * (-2 / dim) * 0; // This term cancels out for mean-centered

      for (let d = 0; d < dim; d++) {
        const dxHat = dOut[t * dim + d] * gamma[d];
        dInput[t * dim + d] += dxHat * invStd +
          dVar * 2 * (input[t * dim + d] - mean) / dim +
          dMean / dim;
      }
    }
  }

  /**
   * Backward through fusion layer
   */
  private fusionBackward(seqLen: number): void {
    const embDim = this.config.embeddingDim;

    // dBlockOutput[0] contains gradient w.r.t. fusedOutput (after dropout)
    this.dFusedOutput.fill(0);
    for (let i = 0; i < seqLen * embDim; i++) {
      this.dFusedOutput[i] = this.dBlockOutput[i] * this.fusionDropoutMask[i];
    }

    // fusedOutput[t, e] = sum_s gate[t, s*embDim + e] * upsampled[s][t, e]
    this.dFusionGates.fill(0);
    for (let s = 0; s < this.nScales; s++) {
      this.dUpsampledScales[s].fill(0);
    }

    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < embDim; e++) {
        const dFused = this.dFusedOutput[t * embDim + e];
        for (let s = 0; s < this.nScales; s++) {
          const gateIdx = t * this.fusionInputDim + s * embDim + e;
          const gate = this.fusionGates[gateIdx];
          const value = this.upsampledScales[s][t * embDim + e];

          this.dFusionGates[gateIdx] += dFused * value;
          this.dUpsampledScales[s][t * embDim + e] += dFused * gate;
        }
      }
    }

    // Backward through sigmoid for gates
    // G = sigmoid(Concat * Wg + bg)
    // dG/d(pre-sigmoid) = G * (1 - G)
    this.dFusionConcat.fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let o = 0; o < this.fusionInputDim; o++) {
        const gate = this.fusionGates[t * this.fusionInputDim + o];
        const dGate = this.dFusionGates[t * this.fusionInputDim + o];
        const dPreSigmoid = dGate * gate * (1 - gate);

        this.gradFusionBg[o] += dPreSigmoid;
        for (let i = 0; i < this.fusionInputDim; i++) {
          this.gradFusionWg[i * this.fusionInputDim + o] +=
            this.fusionConcat[t * this.fusionInputDim + i] * dPreSigmoid;
          this.dFusionConcat[t * this.fusionInputDim + i] +=
            this.fusionWg[i * this.fusionInputDim + o] * dPreSigmoid;
        }
      }
    }

    // Backward through concatenation
    for (let t = 0; t < seqLen; t++) {
      for (let s = 0; s < this.nScales; s++) {
        for (let e = 0; e < embDim; e++) {
          this.dUpsampledScales[s][t * embDim + e] +=
            this.dFusionConcat[t * this.fusionInputDim + s * embDim + e];
        }
      }
    }

    // Backward through upsampling
    for (let s = 0; s < this.nScales; s++) {
      const scale = this.config.temporalScales[s];
      const outLen = this.scaleOutputLengths[s];
      this.dScaleEmbeddingsAdded[s].fill(0);

      for (let t = 0; t < seqLen; t++) {
        const srcT = Math.min(Math.floor(t / scale), outLen - 1);
        for (let e = 0; e < embDim; e++) {
          this.dScaleEmbeddingsAdded[s][srcT * embDim + e] +=
            this.dUpsampledScales[s][t * embDim + e];
        }
      }
    }

    // Backward through scale embeddings and positional encoding addition
    for (let s = 0; s < this.nScales; s++) {
      const outLen = this.scaleOutputLengths[s];
      this.dConvOutputs[s].fill(0);

      for (let t = 0; t < outLen; t++) {
        for (let e = 0; e < embDim; e++) {
          const dOut = this.dScaleEmbeddingsAdded[s][t * embDim + e];
          this.dConvOutputs[s][t * embDim + e] = dOut;
          this.gradScaleEmbeddings[s * embDim + e] += dOut;
          // Positional encoding is fixed, no gradient
        }
      }
    }
  }

  /**
   * Backward through convolution layers
   */
  private convBackward(seqLen: number): void {
    const embDim = this.config.embeddingDim;
    const kernelSize = this.config.temporalKernelSize;

    for (let s = 0; s < this.nScales; s++) {
      const scale = this.config.temporalScales[s];
      const outLen = this.scaleOutputLengths[s];

      for (let t = 0; t < outLen; t++) {
        const centerT = t * scale;
        for (let e = 0; e < embDim; e++) {
          // Backward through GELU
          // Need to reconstruct pre-activation
          let preAct = this.convBiases[s][e];
          for (let k = 0; k < kernelSize; k++) {
            const inputT = centerT - k;
            if (inputT >= 0 && inputT < seqLen) {
              for (let f = 0; f < this.inputDim; f++) {
                const wIdx = (k * this.inputDim + f) * embDim + e;
                preAct += this.normalizedInput[inputT * this.inputDim + f] *
                  this.convWeights[s][wIdx];
              }
            }
          }

          const dOut = this.dConvOutputs[s][t * embDim + e];
          const dPreAct = dOut * geluDerivative(preAct);

          this.gradConvBiases[s][e] += dPreAct;
          for (let k = 0; k < kernelSize; k++) {
            const inputT = centerT - k;
            if (inputT >= 0 && inputT < seqLen) {
              for (let f = 0; f < this.inputDim; f++) {
                const wIdx = (k * this.inputDim + f) * embDim + e;
                this.gradConvWeights[s][wIdx] +=
                  this.normalizedInput[inputT * this.inputDim + f] * dPreAct;
              }
            }
          }
        }
      }
    }
  }

  /**
   * Add L2 regularization gradients
   */
  private addL2Gradients(lambda: number): void {
    // Conv weights
    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.convWeights[s].length; i++) {
        this.gradConvWeights[s][i] += lambda * this.convWeights[s][i];
      }
    }

    // Scale embeddings
    for (let i = 0; i < this.scaleEmbeddings.length; i++) {
      this.gradScaleEmbeddings[i] += lambda * this.scaleEmbeddings[i];
    }

    // Fusion weights
    for (let i = 0; i < this.fusionWg.length; i++) {
      this.gradFusionWg[i] += lambda * this.fusionWg[i];
    }

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.Wq[b].length; i++) {
        this.gradWq[b][i] += lambda * this.Wq[b][i];
        this.gradWk[b][i] += lambda * this.Wk[b][i];
        this.gradWv[b][i] += lambda * this.Wv[b][i];
        this.gradWo[b][i] += lambda * this.Wo[b][i];
      }
      for (let i = 0; i < this.ffnW1[b].length; i++) {
        this.gradFfnW1[b][i] += lambda * this.ffnW1[b][i];
      }
      for (let i = 0; i < this.ffnW2[b].length; i++) {
        this.gradFfnW2[b][i] += lambda * this.ffnW2[b][i];
      }
    }

    // Output weights
    for (let i = 0; i < this.Wout.length; i++) {
      this.gradWout[i] += lambda * this.Wout[i];
    }
    for (let i = 0; i < this.Wpool.length; i++) {
      this.gradWpool[i] += lambda * this.Wpool[i];
    }
  }

  /**
   * Compute gradient norm
   */
  private computeGradientNorm(): number {
    let sum = 0;

    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.gradConvWeights[s].length; i++) {
        sum += this.gradConvWeights[s][i] * this.gradConvWeights[s][i];
      }
      for (let i = 0; i < this.gradConvBiases[s].length; i++) {
        sum += this.gradConvBiases[s][i] * this.gradConvBiases[s][i];
      }
    }

    for (let i = 0; i < this.gradScaleEmbeddings.length; i++) {
      sum += this.gradScaleEmbeddings[i] * this.gradScaleEmbeddings[i];
    }

    for (let i = 0; i < this.gradFusionWg.length; i++) {
      sum += this.gradFusionWg[i] * this.gradFusionWg[i];
    }
    for (let i = 0; i < this.gradFusionBg.length; i++) {
      sum += this.gradFusionBg[i] * this.gradFusionBg[i];
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.gradLn1Gamma[b].length; i++) {
        sum += this.gradLn1Gamma[b][i] * this.gradLn1Gamma[b][i];
        sum += this.gradLn1Beta[b][i] * this.gradLn1Beta[b][i];
        sum += this.gradLn2Gamma[b][i] * this.gradLn2Gamma[b][i];
        sum += this.gradLn2Beta[b][i] * this.gradLn2Beta[b][i];
      }
      for (let i = 0; i < this.gradWq[b].length; i++) {
        sum += this.gradWq[b][i] * this.gradWq[b][i];
        sum += this.gradWk[b][i] * this.gradWk[b][i];
        sum += this.gradWv[b][i] * this.gradWv[b][i];
        sum += this.gradWo[b][i] * this.gradWo[b][i];
      }
      for (let i = 0; i < this.gradFfnW1[b].length; i++) {
        sum += this.gradFfnW1[b][i] * this.gradFfnW1[b][i];
      }
      for (let i = 0; i < this.gradFfnB1[b].length; i++) {
        sum += this.gradFfnB1[b][i] * this.gradFfnB1[b][i];
      }
      for (let i = 0; i < this.gradFfnW2[b].length; i++) {
        sum += this.gradFfnW2[b][i] * this.gradFfnW2[b][i];
      }
      for (let i = 0; i < this.gradFfnB2[b].length; i++) {
        sum += this.gradFfnB2[b][i] * this.gradFfnB2[b][i];
      }
    }

    for (let i = 0; i < this.gradWpool.length; i++) {
      sum += this.gradWpool[i] * this.gradWpool[i];
    }
    sum += this.gradBpool * this.gradBpool;

    for (let i = 0; i < this.gradWout.length; i++) {
      sum += this.gradWout[i] * this.gradWout[i];
    }
    for (let i = 0; i < this.gradBout.length; i++) {
      sum += this.gradBout[i] * this.gradBout[i];
    }

    return Math.sqrt(sum);
  }

  /**
   * Scale all gradients
   */
  private scaleGradients(scale: number): void {
    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.gradConvWeights[s].length; i++) {
        this.gradConvWeights[s][i] *= scale;
      }
      for (let i = 0; i < this.gradConvBiases[s].length; i++) {
        this.gradConvBiases[s][i] *= scale;
      }
    }

    for (let i = 0; i < this.gradScaleEmbeddings.length; i++) {
      this.gradScaleEmbeddings[i] *= scale;
    }

    for (let i = 0; i < this.gradFusionWg.length; i++) {
      this.gradFusionWg[i] *= scale;
    }
    for (let i = 0; i < this.gradFusionBg.length; i++) {
      this.gradFusionBg[i] *= scale;
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.gradLn1Gamma[b].length; i++) {
        this.gradLn1Gamma[b][i] *= scale;
        this.gradLn1Beta[b][i] *= scale;
        this.gradLn2Gamma[b][i] *= scale;
        this.gradLn2Beta[b][i] *= scale;
      }
      for (let i = 0; i < this.gradWq[b].length; i++) {
        this.gradWq[b][i] *= scale;
        this.gradWk[b][i] *= scale;
        this.gradWv[b][i] *= scale;
        this.gradWo[b][i] *= scale;
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

    for (let i = 0; i < this.gradWpool.length; i++) {
      this.gradWpool[i] *= scale;
    }
    this.gradBpool *= scale;

    for (let i = 0; i < this.gradWout.length; i++) {
      this.gradWout[i] *= scale;
    }
    for (let i = 0; i < this.gradBout.length; i++) {
      this.gradBout[i] *= scale;
    }
  }

  /**
   * Compute learning rate with warmup and cosine decay
   */
  private computeLearningRate(): number {
    const step = this.updateCount;
    const { learningRate, warmupSteps, totalSteps } = this.config;

    if (step < warmupSteps) {
      return learningRate * (step / warmupSteps);
    }

    const progress = (step - warmupSteps) /
      Math.max(1, totalSteps - warmupSteps);
    return learningRate * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)));
  }

  /**
   * Adam optimizer update
   */
  private adamUpdate(lr: number): void {
    const { beta1, beta2, epsilon } = this.config;
    const t = this.updateCount;

    // Bias correction
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    // Helper function
    const update = (
      weights: Float64Array,
      grads: Float64Array,
      moments: AdamMoments,
    ) => {
      for (let i = 0; i < weights.length; i++) {
        moments.m[i] = beta1 * moments.m[i] + (1 - beta1) * grads[i];
        moments.v[i] = beta2 * moments.v[i] + (1 - beta2) * grads[i] * grads[i];
        const mHat = moments.m[i] / bc1;
        const vHat = moments.v[i] / bc2;
        weights[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
      }
    };

    // Conv weights
    for (let s = 0; s < this.nScales; s++) {
      update(
        this.convWeights[s],
        this.gradConvWeights[s],
        this.convWeightsMoments[s],
      );
      update(
        this.convBiases[s],
        this.gradConvBiases[s],
        this.convBiasesMoments[s],
      );
    }

    // Scale embeddings
    update(
      this.scaleEmbeddings,
      this.gradScaleEmbeddings,
      this.scaleEmbeddingsMoments,
    );

    // Fusion
    update(this.fusionWg, this.gradFusionWg, this.fusionWgMoments);
    update(this.fusionBg, this.gradFusionBg, this.fusionBgMoments);

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      update(this.ln1Gamma[b], this.gradLn1Gamma[b], this.ln1GammaMoments[b]);
      update(this.ln1Beta[b], this.gradLn1Beta[b], this.ln1BetaMoments[b]);
      update(this.Wq[b], this.gradWq[b], this.WqMoments[b]);
      update(this.Wk[b], this.gradWk[b], this.WkMoments[b]);
      update(this.Wv[b], this.gradWv[b], this.WvMoments[b]);
      update(this.Wo[b], this.gradWo[b], this.WoMoments[b]);
      update(this.ln2Gamma[b], this.gradLn2Gamma[b], this.ln2GammaMoments[b]);
      update(this.ln2Beta[b], this.gradLn2Beta[b], this.ln2BetaMoments[b]);
      update(this.ffnW1[b], this.gradFfnW1[b], this.ffnW1Moments[b]);
      update(this.ffnB1[b], this.gradFfnB1[b], this.ffnB1Moments[b]);
      update(this.ffnW2[b], this.gradFfnW2[b], this.ffnW2Moments[b]);
      update(this.ffnB2[b], this.gradFfnB2[b], this.ffnB2Moments[b]);
    }

    // Pooling
    update(this.Wpool, this.gradWpool, this.WpoolMoments);

    // bpool (scalar)
    this.bpoolMoments.m[0] = beta1 * this.bpoolMoments.m[0] +
      (1 - beta1) * this.gradBpool;
    this.bpoolMoments.v[0] = beta2 * this.bpoolMoments.v[0] +
      (1 - beta2) * this.gradBpool * this.gradBpool;
    const mHat = this.bpoolMoments.m[0] / bc1;
    const vHat = this.bpoolMoments.v[0] / bc2;
    this.bpool -= lr * mHat / (Math.sqrt(vHat) + epsilon);

    // Output
    update(this.Wout, this.gradWout, this.WoutMoments);
    update(this.bout, this.gradBout, this.boutMoments);
  }

  /**
   * ADWIN drift detection
   * @returns true if drift detected
   */
  private adwinDetect(loss: number): boolean {
    // Add to ring buffer
    this.adwinBuffer[this.adwinHead] = loss;
    this.adwinHead = (this.adwinHead + 1) % this.adwinWindowCap;
    if (this.adwinSize < this.adwinWindowCap) {
      this.adwinSize++;
    }

    // Need minimum size
    if (this.adwinSize < this.adwinMinSize) {
      return false;
    }

    const delta = this.config.adwinDelta;
    let driftDetected = false;

    // Try different split points
    for (
      let split = this.adwinMinSize / 2;
      split < this.adwinSize - this.adwinMinSize / 2;
      split++
    ) {
      // Compute means of left and right windows
      let sumLeft = 0;
      let sumRight = 0;
      const nLeft = split;
      const nRight = this.adwinSize - split;

      // Calculate indices in ring buffer
      const startIdx = (this.adwinHead - this.adwinSize + this.adwinWindowCap) %
        this.adwinWindowCap;

      for (let i = 0; i < nLeft; i++) {
        const idx = (startIdx + i) % this.adwinWindowCap;
        sumLeft += this.adwinBuffer[idx];
      }
      for (let i = nLeft; i < this.adwinSize; i++) {
        const idx = (startIdx + i) % this.adwinWindowCap;
        sumRight += this.adwinBuffer[idx];
      }

      const meanLeft = sumLeft / nLeft;
      const meanRight = sumRight / nRight;

      // Simplified epsilon cut
      const eps = Math.sqrt(
        (2 * Math.log(2 / delta)) * (1 / nLeft + 1 / nRight),
      );

      if (Math.abs(meanLeft - meanRight) > eps) {
        driftDetected = true;
        break;
      }
    }

    if (driftDetected) {
      this.driftCount++;
      // Reset window
      this.adwinSize = 1;
      this.adwinHead = 1;
      this.adwinBuffer[0] = loss;
      // Reset running loss to current
      this.runningLoss = loss;
    }

    return driftDetected;
  }

  /**
   * Get current accuracy metric
   */
  private getAccuracy(): number {
    if (this.sampleCount === 0) return 0;
    return 1 / (1 + this.runningLoss);
  }

  /**
   * Count total parameters
   */
  private countParameters(): number {
    let count = 0;

    for (let s = 0; s < this.nScales; s++) {
      count += this.convWeights[s].length + this.convBiases[s].length;
    }

    count += this.scaleEmbeddings.length;
    count += this.fusionWg.length + this.fusionBg.length;

    for (let b = 0; b < this.config.numBlocks; b++) {
      count += this.ln1Gamma[b].length + this.ln1Beta[b].length;
      count += this.Wq[b].length + this.Wk[b].length + this.Wv[b].length +
        this.Wo[b].length;
      count += this.ln2Gamma[b].length + this.ln2Beta[b].length;
      count += this.ffnW1[b].length + this.ffnB1[b].length;
      count += this.ffnW2[b].length + this.ffnB2[b].length;
    }

    count += this.Wpool.length + 1; // +1 for bpool
    count += this.Wout.length + this.bout.length;

    return count;
  }

  /**
   * Unpack Float64Array to 2D number array
   */
  private unpack2D(arr: Float64Array, rows: number, cols: number): number[][] {
    const result: number[][] = [];
    for (let r = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c = 0; c < cols; c++) {
        row.push(arr[r * cols + c]);
      }
      result.push(row);
    }
    return result;
  }
}
