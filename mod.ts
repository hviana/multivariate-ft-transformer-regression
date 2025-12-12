/**
 * FT-Transformer for Multivariate Regression with Incremental Online Learning
 *
 * Implements the Feature Tokenizer Transformer architecture with:
 * - Feature-wise linear embeddings: Each input feature → embedding vector
 * - Learnable [CLS] token for aggregating sequence information
 * - Multi-head self-attention with scaled dot-product attention
 * - Feed-forward networks with GELU activation
 * - Layer normalization with learnable parameters
 * - Adam optimizer with warmup and cosine decay learning rate schedule
 * - Welford's algorithm for online z-score normalization
 * - ADWIN algorithm for concept drift detection
 * - Outlier downweighting using standardized residuals
 * - L2 regularization for weight decay
 *
 * @module FTTransformerRegression
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Result returned from a single online fitting step
 */
export interface FitResult {
  /** Mean squared error loss for this sample */
  loss: number;
  /** L2 norm of the gradient vector */
  gradientNorm: number;
  /** Current learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether model has converged based on loss threshold */
  converged: boolean;
  /** Current sample index (total samples seen) */
  sampleIndex: number;
  /** Whether concept drift was detected via ADWIN */
  driftDetected: boolean;
}

/**
 * Single prediction with uncertainty bounds
 */
export interface SinglePrediction {
  /** Predicted values for each output dimension */
  predicted: number[];
  /** Lower confidence bound (predicted - 2*standardError) */
  lowerBound: number[];
  /** Upper confidence bound (predicted + 2*standardError) */
  upperBound: number[];
  /** Standard error estimate for each output */
  standardError: number[];
}

/**
 * Result from prediction operation
 */
export interface PredictionResult {
  /** Array of predictions for requested future steps */
  predictions: SinglePrediction[];
  /** Model accuracy: 1/(1 + averageLoss) */
  accuracy: number;
  /** Total number of training samples seen */
  sampleCount: number;
  /** Whether model is ready for predictions (has been trained) */
  isModelReady: boolean;
}

/**
 * Complete weight information for model inspection
 */
export interface WeightInfo {
  /** Feature embedding weights [inputDim][2][embeddingDim] (weights, biases) */
  featureEmbeddings: number[][][];
  /** Learnable CLS token [embeddingDim] */
  clsToken: number[];
  /** Attention weights per block [numBlocks][4][...] (Wq, Wk, Wv, Wo) */
  attentionWeights: number[][][];
  /** FFN weights per block [numBlocks][4][...] (W1, b1, W2, b2) */
  ffnWeights: number[][][];
  /** Layer norm params per block [numBlocks][4][...] (gamma1, beta1, gamma2, beta2) */
  layerNormParams: number[][][];
  /** Output layer weights [2][...] (weights, biases) */
  outputWeights: number[][];
  /** Adam first moment estimates */
  firstMoment: number[][][];
  /** Adam second moment estimates */
  secondMoment: number[][][];
  /** Number of Adam updates performed */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean of input features */
  inputMean: number[];
  /** Running standard deviation of input features */
  inputStd: number[];
  /** Running mean of output targets */
  outputMean: number[];
  /** Running standard deviation of output targets */
  outputStd: number[];
  /** Number of samples used for statistics */
  count: number;
}

/**
 * Summary of model configuration and state
 */
export interface ModelSummary {
  /** Whether model has been initialized with data */
  isInitialized: boolean;
  /** Auto-detected input dimension */
  inputDimension: number;
  /** Auto-detected output dimension */
  outputDimension: number;
  /** Number of transformer blocks */
  numBlocks: number;
  /** Embedding dimension */
  embeddingDim: number;
  /** Number of attention heads */
  numHeads: number;
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Total training samples seen */
  sampleCount: number;
  /** Current model accuracy */
  accuracy: number;
  /** Whether model has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

/**
 * Input data format for online fitting
 */
interface FitInput {
  /** Input feature vectors [numSamples][inputDim] */
  xCoordinates: number[][];
  /** Target output vectors [numSamples][outputDim] */
  yCoordinates: number[][];
}

/**
 * Configuration options with defaults
 */
interface Config {
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
}

/**
 * Serializable state for save/load
 */
interface SerializedState {
  config: Config;
  inputDim: number;
  outputDim: number;
  sampleCount: number;
  updateCount: number;
  runningLossSum: number;
  runningLossCount: number;
  converged: boolean;
  driftCount: number;
  lastInput: number[] | null;
  weights: {
    featureWeights: number[][];
    featureBiases: number[][];
    clsToken: number[];
    blocks: Array<{
      ln1Gamma: number[];
      ln1Beta: number[];
      wq: number[];
      wk: number[];
      wv: number[];
      wo: number[];
      bo: number[];
      ln2Gamma: number[];
      ln2Beta: number[];
      w1: number[];
      b1: number[];
      w2: number[];
      b2: number[];
    }>;
    wout: number[];
    bout: number[];
  };
  adam: {
    featureWeightsM: number[][];
    featureWeightsV: number[][];
    featureBiasesM: number[][];
    featureBiasesV: number[][];
    clsTokenM: number[];
    clsTokenV: number[];
    blocks: Array<{
      ln1GammaM: number[];
      ln1GammaV: number[];
      ln1BetaM: number[];
      ln1BetaV: number[];
      wqM: number[];
      wqV: number[];
      wkM: number[];
      wkV: number[];
      wvM: number[];
      wvV: number[];
      woM: number[];
      woV: number[];
      boM: number[];
      boV: number[];
      ln2GammaM: number[];
      ln2GammaV: number[];
      ln2BetaM: number[];
      ln2BetaV: number[];
      w1M: number[];
      w1V: number[];
      b1M: number[];
      b1V: number[];
      w2M: number[];
      w2V: number[];
      b2M: number[];
      b2V: number[];
    }>;
    woutM: number[];
    woutV: number[];
    boutM: number[];
    boutV: number[];
  };
  normalization: {
    inputMean: number[];
    inputM2: number[];
    outputMean: number[];
    outputM2: number[];
    count: number;
  };
  adwin: {
    window: number[];
    windowSum: number;
    windowSize: number;
  };
  predictionVariance: number[];
}

// ============================================================================
// OBJECT POOL FOR MEMORY REUSE
// ============================================================================

/**
 * Pool for reusing Float64Array instances to minimize GC pressure
 * @private
 */
class ArrayPool {
  private readonly pools: Map<number, Float64Array[]> = new Map();
  private readonly maxPoolSize: number = 50;

  /**
   * Acquire an array of specified size from pool or create new
   * @param size - Required array length
   * @returns Float64Array of requested size (zeroed)
   */
  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      const arr = pool.pop()!;
      arr.fill(0);
      return arr;
    }
    return new Float64Array(size);
  }

  /**
   * Release array back to pool for reuse
   * @param arr - Array to release
   */
  release(arr: Float64Array): void {
    if (!arr || arr.length === 0) return;
    const size = arr.length;
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      pool.push(arr);
    }
  }

  /**
   * Clear all pooled arrays
   */
  clear(): void {
    this.pools.clear();
  }
}

// ============================================================================
// WELFORD'S ONLINE STATISTICS
// ============================================================================

/**
 * Implements Welford's online algorithm for computing running mean and variance
 *
 * Update formulas:
 * - δ = x - μ
 * - μ += δ/n
 * - M₂ += δ(x - μ)
 * - σ² = M₂/(n-1)
 *
 * @private
 */
class WelfordStats {
  private mean: Float64Array;
  private m2: Float64Array;
  private count: number = 0;
  private readonly dim: number;

  constructor(dimension: number) {
    this.dim = dimension;
    this.mean = new Float64Array(dimension);
    this.m2 = new Float64Array(dimension);
  }

  /**
   * Update statistics with new observation
   * @param values - New observation vector
   */
  update(values: Float64Array | number[]): void {
    this.count++;
    const n = this.count;
    for (let i = 0; i < this.dim; i++) {
      const x = values[i];
      const delta = x - this.mean[i];
      this.mean[i] += delta / n;
      const delta2 = x - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  /**
   * Get current running mean
   */
  getMean(): Float64Array {
    return this.mean;
  }

  /**
   * Get current standard deviation with numerical stability
   * @param epsilon - Small value to prevent division by zero
   */
  getStd(epsilon: number = 1e-8): Float64Array {
    const std = new Float64Array(this.dim);
    if (this.count > 1) {
      for (let i = 0; i < this.dim; i++) {
        std[i] = Math.sqrt(this.m2[i] / (this.count - 1) + epsilon);
      }
    } else {
      std.fill(1);
    }
    return std;
  }

  /**
   * Get M2 values for serialization
   */
  getM2(): Float64Array {
    return this.m2;
  }

  /**
   * Get sample count
   */
  getCount(): number {
    return this.count;
  }

  /**
   * Reset all statistics
   */
  reset(): void {
    this.mean.fill(0);
    this.m2.fill(0);
    this.count = 0;
  }

  /**
   * Load state from serialized data
   */
  loadState(mean: number[], m2: number[], count: number): void {
    for (let i = 0; i < this.dim; i++) {
      this.mean[i] = mean[i] || 0;
      this.m2[i] = m2[i] || 0;
    }
    this.count = count;
  }
}

// ============================================================================
// ADWIN DRIFT DETECTION
// ============================================================================

/**
 * ADWIN (ADaptive WINdowing) algorithm for concept drift detection
 *
 * Maintains a variable-length window of recent observations and detects
 * drift when the difference between two sub-windows exceeds a threshold:
 * |μ₀ - μ₁| ≥ εcut where εcut = sqrt((1/2m)ln(4/δ))
 *
 * @private
 */
class ADWIN {
  private window: number[] = [];
  private windowSum: number = 0;
  private readonly delta: number;
  private readonly maxWindowSize: number = 1000;

  /**
   * @param delta - Confidence parameter (default 0.002)
   */
  constructor(delta: number = 0.002) {
    this.delta = delta;
  }

  /**
   * Add new value and check for drift
   * @param value - New observation (typically error/loss value)
   * @returns true if drift detected
   */
  update(value: number): boolean {
    this.window.push(value);
    this.windowSum += value;

    // Limit window size for memory efficiency
    while (this.window.length > this.maxWindowSize) {
      this.windowSum -= this.window.shift()!;
    }

    if (this.window.length < 10) return false;

    // Check for drift by comparing sub-windows
    return this.detectDrift();
  }

  /**
   * Detect drift by finding optimal split point
   * @private
   */
  private detectDrift(): boolean {
    const n = this.window.length;
    let sum0 = 0;

    for (let i = 1; i < n; i++) {
      sum0 += this.window[i - 1];
      const n0 = i;
      const n1 = n - i;

      if (n0 < 5 || n1 < 5) continue;

      const mean0 = sum0 / n0;
      const mean1 = (this.windowSum - sum0) / n1;

      // Hoeffding bound: εcut = sqrt((1/2m)ln(4/δ))
      const m = 1 / (1 / n0 + 1 / n1);
      const epsilon = Math.sqrt(Math.log(4 / this.delta) / (2 * m));

      if (Math.abs(mean0 - mean1) >= epsilon) {
        // Drift detected - shrink window
        this.shrinkWindow(i);
        return true;
      }
    }
    return false;
  }

  /**
   * Shrink window after drift detection
   * @private
   */
  private shrinkWindow(splitPoint: number): void {
    // Keep only the newer portion
    const newWindow = this.window.slice(splitPoint);
    this.window = newWindow;
    this.windowSum = 0;
    for (let i = 0; i < this.window.length; i++) {
      this.windowSum += this.window[i];
    }
  }

  /**
   * Get current window state for serialization
   */
  getState(): { window: number[]; windowSum: number; windowSize: number } {
    return {
      window: this.window.slice(),
      windowSum: this.windowSum,
      windowSize: this.window.length,
    };
  }

  /**
   * Load state from serialized data
   */
  loadState(window: number[], windowSum: number): void {
    this.window = window.slice();
    this.windowSum = windowSum;
  }

  /**
   * Reset drift detector
   */
  reset(): void {
    this.window = [];
    this.windowSum = 0;
  }
}

// ============================================================================
// TRANSFORMER BLOCK DATA STRUCTURES
// ============================================================================

/**
 * Stores weights and Adam state for a single transformer block
 * @private
 */
class TransformerBlockWeights {
  // Layer norm 1 (pre-attention)
  ln1Gamma: Float64Array;
  ln1Beta: Float64Array;

  // Multi-head attention weights
  // Shape: [embeddingDim, embeddingDim] stored as flat array
  wq: Float64Array;
  wk: Float64Array;
  wv: Float64Array;
  wo: Float64Array;
  bo: Float64Array;

  // Layer norm 2 (pre-FFN)
  ln2Gamma: Float64Array;
  ln2Beta: Float64Array;

  // FFN weights
  // W1: [embeddingDim, ffnDim], W2: [ffnDim, embeddingDim]
  w1: Float64Array;
  b1: Float64Array;
  w2: Float64Array;
  b2: Float64Array;

  // Adam first moments
  ln1GammaM: Float64Array;
  ln1BetaM: Float64Array;
  wqM: Float64Array;
  wkM: Float64Array;
  wvM: Float64Array;
  woM: Float64Array;
  boM: Float64Array;
  ln2GammaM: Float64Array;
  ln2BetaM: Float64Array;
  w1M: Float64Array;
  b1M: Float64Array;
  w2M: Float64Array;
  b2M: Float64Array;

  // Adam second moments
  ln1GammaV: Float64Array;
  ln1BetaV: Float64Array;
  wqV: Float64Array;
  wkV: Float64Array;
  wvV: Float64Array;
  woV: Float64Array;
  boV: Float64Array;
  ln2GammaV: Float64Array;
  ln2BetaV: Float64Array;
  w1V: Float64Array;
  b1V: Float64Array;
  w2V: Float64Array;
  b2V: Float64Array;

  constructor(embeddingDim: number, ffnDim: number) {
    const ed = embeddingDim;
    const fd = ffnDim;
    const attnSize = ed * ed;
    const ffn1Size = ed * fd;
    const ffn2Size = fd * ed;

    // Initialize weights
    this.ln1Gamma = new Float64Array(ed);
    this.ln1Beta = new Float64Array(ed);
    this.wq = new Float64Array(attnSize);
    this.wk = new Float64Array(attnSize);
    this.wv = new Float64Array(attnSize);
    this.wo = new Float64Array(attnSize);
    this.bo = new Float64Array(ed);
    this.ln2Gamma = new Float64Array(ed);
    this.ln2Beta = new Float64Array(ed);
    this.w1 = new Float64Array(ffn1Size);
    this.b1 = new Float64Array(fd);
    this.w2 = new Float64Array(ffn2Size);
    this.b2 = new Float64Array(ed);

    // Initialize Adam moments (all zeros)
    this.ln1GammaM = new Float64Array(ed);
    this.ln1BetaM = new Float64Array(ed);
    this.wqM = new Float64Array(attnSize);
    this.wkM = new Float64Array(attnSize);
    this.wvM = new Float64Array(attnSize);
    this.woM = new Float64Array(attnSize);
    this.boM = new Float64Array(ed);
    this.ln2GammaM = new Float64Array(ed);
    this.ln2BetaM = new Float64Array(ed);
    this.w1M = new Float64Array(ffn1Size);
    this.b1M = new Float64Array(fd);
    this.w2M = new Float64Array(ffn2Size);
    this.b2M = new Float64Array(ed);

    this.ln1GammaV = new Float64Array(ed);
    this.ln1BetaV = new Float64Array(ed);
    this.wqV = new Float64Array(attnSize);
    this.wkV = new Float64Array(attnSize);
    this.wvV = new Float64Array(attnSize);
    this.woV = new Float64Array(attnSize);
    this.boV = new Float64Array(ed);
    this.ln2GammaV = new Float64Array(ed);
    this.ln2BetaV = new Float64Array(ed);
    this.w1V = new Float64Array(ffn1Size);
    this.b1V = new Float64Array(fd);
    this.w2V = new Float64Array(ffn2Size);
    this.b2V = new Float64Array(ed);

    // Initialize layer norm gamma to 1
    this.ln1Gamma.fill(1);
    this.ln2Gamma.fill(1);
  }

  /**
   * Initialize weights with Xavier/Glorot initialization
   * @param scale - Scale factor for initialization
   */
  initializeWeights(scale: number): void {
    this.initArray(this.wq, scale);
    this.initArray(this.wk, scale);
    this.initArray(this.wv, scale);
    this.initArray(this.wo, scale);
    this.initArray(this.w1, scale);
    this.initArray(this.w2, scale);
  }

  /**
   * @private
   */
  private initArray(arr: Float64Array, scale: number): void {
    for (let i = 0; i < arr.length; i++) {
      // Xavier initialization with normal distribution approximation
      arr[i] = (Math.random() - 0.5) * 2 * scale;
    }
  }

  /**
   * Count total parameters in this block
   */
  countParameters(): number {
    return (
      this.ln1Gamma.length + this.ln1Beta.length +
      this.wq.length + this.wk.length + this.wv.length +
      this.wo.length + this.bo.length +
      this.ln2Gamma.length + this.ln2Beta.length +
      this.w1.length + this.b1.length +
      this.w2.length + this.b2.length
    );
  }
}

// ============================================================================
// FORWARD PASS CACHE
// ============================================================================

/**
 * Cache for storing intermediate values during forward pass for backpropagation
 * @private
 */
class ForwardCache {
  // Input embeddings [seqLen, embeddingDim]
  embeddings: Float64Array;

  // Per-block caches
  blockCaches: BlockCache[];

  // Final CLS representation
  clsFinal: Float64Array;

  // Output
  output: Float64Array;

  constructor(
    seqLen: number,
    embeddingDim: number,
    ffnDim: number,
    numBlocks: number,
    numHeads: number,
    outputDim: number,
  ) {
    const seqEmbed = seqLen * embeddingDim;
    this.embeddings = new Float64Array(seqEmbed);
    this.blockCaches = [];
    for (let i = 0; i < numBlocks; i++) {
      this.blockCaches.push(
        new BlockCache(seqLen, embeddingDim, ffnDim, numHeads),
      );
    }
    this.clsFinal = new Float64Array(embeddingDim);
    this.output = new Float64Array(outputDim);
  }

  reset(): void {
    this.embeddings.fill(0);
    for (let i = 0; i < this.blockCaches.length; i++) {
      this.blockCaches[i].reset();
    }
    this.clsFinal.fill(0);
    this.output.fill(0);
  }
}

/**
 * Cache for a single transformer block
 * @private
 */
class BlockCache {
  // Input to block
  input: Float64Array;

  // Layer norm 1 outputs
  ln1Out: Float64Array;
  ln1Mean: Float64Array;
  ln1Var: Float64Array;

  // Attention intermediates
  q: Float64Array;
  k: Float64Array;
  v: Float64Array;
  attnScores: Float64Array; // [numHeads, seqLen, seqLen]
  attnProbs: Float64Array; // After softmax
  attnOut: Float64Array;
  multiHeadOut: Float64Array;

  // Residual after attention
  residual1: Float64Array;

  // Layer norm 2 outputs
  ln2Out: Float64Array;
  ln2Mean: Float64Array;
  ln2Var: Float64Array;

  // FFN intermediates
  ffnHidden: Float64Array;
  ffnHiddenPreAct: Float64Array; // Before GELU
  ffnOut: Float64Array;

  // Final output (residual2)
  output: Float64Array;

  constructor(
    seqLen: number,
    embeddingDim: number,
    ffnDim: number,
    numHeads: number,
  ) {
    const seqEmbed = seqLen * embeddingDim;
    const attnScoreSize = numHeads * seqLen * seqLen;
    const seqFfn = seqLen * ffnDim;

    this.input = new Float64Array(seqEmbed);
    this.ln1Out = new Float64Array(seqEmbed);
    this.ln1Mean = new Float64Array(seqLen);
    this.ln1Var = new Float64Array(seqLen);
    this.q = new Float64Array(seqEmbed);
    this.k = new Float64Array(seqEmbed);
    this.v = new Float64Array(seqEmbed);
    this.attnScores = new Float64Array(attnScoreSize);
    this.attnProbs = new Float64Array(attnScoreSize);
    this.attnOut = new Float64Array(seqEmbed);
    this.multiHeadOut = new Float64Array(seqEmbed);
    this.residual1 = new Float64Array(seqEmbed);
    this.ln2Out = new Float64Array(seqEmbed);
    this.ln2Mean = new Float64Array(seqLen);
    this.ln2Var = new Float64Array(seqLen);
    this.ffnHidden = new Float64Array(seqFfn);
    this.ffnHiddenPreAct = new Float64Array(seqFfn);
    this.ffnOut = new Float64Array(seqEmbed);
    this.output = new Float64Array(seqEmbed);
  }

  reset(): void {
    this.input.fill(0);
    this.ln1Out.fill(0);
    this.ln1Mean.fill(0);
    this.ln1Var.fill(0);
    this.q.fill(0);
    this.k.fill(0);
    this.v.fill(0);
    this.attnScores.fill(0);
    this.attnProbs.fill(0);
    this.attnOut.fill(0);
    this.multiHeadOut.fill(0);
    this.residual1.fill(0);
    this.ln2Out.fill(0);
    this.ln2Mean.fill(0);
    this.ln2Var.fill(0);
    this.ffnHidden.fill(0);
    this.ffnHiddenPreAct.fill(0);
    this.ffnOut.fill(0);
    this.output.fill(0);
  }
}

// ============================================================================
// GRADIENT STORAGE
// ============================================================================

/**
 * Storage for gradients during backpropagation
 * @private
 */
class GradientStorage {
  // Feature embeddings gradients
  featureWeightsGrad: Float64Array[];
  featureBiasesGrad: Float64Array[];
  clsTokenGrad: Float64Array;

  // Block gradients
  blockGrads: BlockGradients[];

  // Output layer gradients
  woutGrad: Float64Array;
  boutGrad: Float64Array;

  constructor(
    inputDim: number,
    embeddingDim: number,
    ffnDim: number,
    outputDim: number,
    numBlocks: number,
  ) {
    this.featureWeightsGrad = [];
    this.featureBiasesGrad = [];
    for (let i = 0; i < inputDim; i++) {
      this.featureWeightsGrad.push(new Float64Array(embeddingDim));
      this.featureBiasesGrad.push(new Float64Array(embeddingDim));
    }
    this.clsTokenGrad = new Float64Array(embeddingDim);

    this.blockGrads = [];
    for (let i = 0; i < numBlocks; i++) {
      this.blockGrads.push(new BlockGradients(embeddingDim, ffnDim));
    }

    this.woutGrad = new Float64Array(embeddingDim * outputDim);
    this.boutGrad = new Float64Array(outputDim);
  }

  reset(): void {
    for (let i = 0; i < this.featureWeightsGrad.length; i++) {
      this.featureWeightsGrad[i].fill(0);
      this.featureBiasesGrad[i].fill(0);
    }
    this.clsTokenGrad.fill(0);
    for (let i = 0; i < this.blockGrads.length; i++) {
      this.blockGrads[i].reset();
    }
    this.woutGrad.fill(0);
    this.boutGrad.fill(0);
  }

  /**
   * Compute total L2 norm of all gradients
   */
  computeNorm(): number {
    let sum = 0;

    for (let i = 0; i < this.featureWeightsGrad.length; i++) {
      sum += this.sumSquared(this.featureWeightsGrad[i]);
      sum += this.sumSquared(this.featureBiasesGrad[i]);
    }
    sum += this.sumSquared(this.clsTokenGrad);

    for (let i = 0; i < this.blockGrads.length; i++) {
      sum += this.blockGrads[i].computeNorm();
    }

    sum += this.sumSquared(this.woutGrad);
    sum += this.sumSquared(this.boutGrad);

    return Math.sqrt(sum);
  }

  private sumSquared(arr: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i] * arr[i];
    }
    return sum;
  }
}

/**
 * Gradients for a single transformer block
 * @private
 */
class BlockGradients {
  ln1GammaGrad: Float64Array;
  ln1BetaGrad: Float64Array;
  wqGrad: Float64Array;
  wkGrad: Float64Array;
  wvGrad: Float64Array;
  woGrad: Float64Array;
  boGrad: Float64Array;
  ln2GammaGrad: Float64Array;
  ln2BetaGrad: Float64Array;
  w1Grad: Float64Array;
  b1Grad: Float64Array;
  w2Grad: Float64Array;
  b2Grad: Float64Array;

  constructor(embeddingDim: number, ffnDim: number) {
    const ed = embeddingDim;
    const fd = ffnDim;
    const attnSize = ed * ed;
    const ffn1Size = ed * fd;
    const ffn2Size = fd * ed;

    this.ln1GammaGrad = new Float64Array(ed);
    this.ln1BetaGrad = new Float64Array(ed);
    this.wqGrad = new Float64Array(attnSize);
    this.wkGrad = new Float64Array(attnSize);
    this.wvGrad = new Float64Array(attnSize);
    this.woGrad = new Float64Array(attnSize);
    this.boGrad = new Float64Array(ed);
    this.ln2GammaGrad = new Float64Array(ed);
    this.ln2BetaGrad = new Float64Array(ed);
    this.w1Grad = new Float64Array(ffn1Size);
    this.b1Grad = new Float64Array(fd);
    this.w2Grad = new Float64Array(ffn2Size);
    this.b2Grad = new Float64Array(ed);
  }

  reset(): void {
    this.ln1GammaGrad.fill(0);
    this.ln1BetaGrad.fill(0);
    this.wqGrad.fill(0);
    this.wkGrad.fill(0);
    this.wvGrad.fill(0);
    this.woGrad.fill(0);
    this.boGrad.fill(0);
    this.ln2GammaGrad.fill(0);
    this.ln2BetaGrad.fill(0);
    this.w1Grad.fill(0);
    this.b1Grad.fill(0);
    this.w2Grad.fill(0);
    this.b2Grad.fill(0);
  }

  computeNorm(): number {
    let sum = 0;
    const arrays = [
      this.ln1GammaGrad,
      this.ln1BetaGrad,
      this.wqGrad,
      this.wkGrad,
      this.wvGrad,
      this.woGrad,
      this.boGrad,
      this.ln2GammaGrad,
      this.ln2BetaGrad,
      this.w1Grad,
      this.b1Grad,
      this.w2Grad,
      this.b2Grad,
    ];
    for (let a = 0; a < arrays.length; a++) {
      const arr = arrays[a];
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i] * arr[i];
      }
    }
    return sum;
  }
}

// ============================================================================
// MAIN CLASS
// ============================================================================

/**
 * FT-Transformer for Multivariate Regression with Incremental Online Learning
 *
 * Architecture:
 * 1. Feature Tokenizer: Each input feature xᵢ → eᵢ = xᵢ · Wᵢ + bᵢ ∈ ℝ^embeddingDim
 * 2. CLS Token: Prepend learnable [CLS] ∈ ℝ^embeddingDim → sequence [CLS, e₁, ..., eₙ]
 * 3. Transformer Blocks × numBlocks: LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN → Residual
 * 4. Output: Dense([CLS]_final) → ŷ ∈ ℝ^outputDim
 *
 * @example
 * ```typescript
 * const model = new FTTransformerRegression();
 *
 * // Online training
 * const result = model.fitOnline({
 *   xCoordinates: [[1.0, 2.0, 3.0]],
 *   yCoordinates: [[4.0, 5.0]]
 * });
 *
 * // Make predictions
 * const predictions = model.predict(1);
 *
 * // Save and load
 * const state = model.save();
 * model.load(state);
 * ```
 */
export class FTTransformerRegression {
  // Configuration
  private readonly config: Config;

  // Dimensions (auto-detected from first data)
  private inputDim: number = 0;
  private outputDim: number = 0;
  private seqLen: number = 0; // inputDim + 1 (for CLS token)
  private headDim: number = 0;
  private ffnDim: number = 0;

  // Model state
  private isInitialized: boolean = false;
  private sampleCount: number = 0;
  private updateCount: number = 0;
  private runningLossSum: number = 0;
  private runningLossCount: number = 0;
  private converged: boolean = false;
  private driftCount: number = 0;

  // Weights
  private featureWeights: Float64Array[] = []; // [inputDim][embeddingDim]
  private featureBiases: Float64Array[] = []; // [inputDim][embeddingDim]
  private clsToken: Float64Array | null = null; // [embeddingDim]
  private blocks: TransformerBlockWeights[] = [];
  private wout: Float64Array | null = null; // [embeddingDim * outputDim]
  private bout: Float64Array | null = null; // [outputDim]

  // Adam state for non-block weights
  private featureWeightsM: Float64Array[] = [];
  private featureWeightsV: Float64Array[] = [];
  private featureBiasesM: Float64Array[] = [];
  private featureBiasesV: Float64Array[] = [];
  private clsTokenM: Float64Array | null = null;
  private clsTokenV: Float64Array | null = null;
  private woutM: Float64Array | null = null;
  private woutV: Float64Array | null = null;
  private boutM: Float64Array | null = null;
  private boutV: Float64Array | null = null;

  // Statistics
  private inputStats: WelfordStats | null = null;
  private outputStats: WelfordStats | null = null;

  // Drift detection
  private adwin: ADWIN;

  // Forward/backward caches (preallocated)
  private forwardCache: ForwardCache | null = null;
  private gradients: GradientStorage | null = null;

  // Object pool for temporary arrays
  private readonly pool: ArrayPool;

  // Prediction variance tracking
  private predictionVariance: Float64Array | null = null;

  // Last input for auto-regressive prediction
  private lastInput: Float64Array | null = null;

  // Temporary buffers (preallocated)
  private tempInput: Float64Array | null = null;
  private tempOutput: Float64Array | null = null;
  private tempSeqGrad: Float64Array | null = null;

  /**
   * Create a new FT-Transformer regression model
   *
   * @param options - Configuration options (all optional with defaults)
   * @param options.numBlocks - Number of transformer blocks (default: 3)
   * @param options.embeddingDim - Embedding dimension (default: 64)
   * @param options.numHeads - Number of attention heads (default: 8)
   * @param options.ffnMultiplier - FFN hidden dimension multiplier (default: 4)
   * @param options.attentionDropout - Attention dropout rate (default: 0.0)
   * @param options.learningRate - Base learning rate for Adam (default: 0.001)
   * @param options.warmupSteps - Number of warmup steps (default: 100)
   * @param options.totalSteps - Total steps for learning rate schedule (default: 10000)
   * @param options.beta1 - Adam beta1 (default: 0.9)
   * @param options.beta2 - Adam beta2 (default: 0.999)
   * @param options.epsilon - Numerical stability epsilon (default: 1e-8)
   * @param options.regularizationStrength - L2 regularization strength (default: 1e-4)
   * @param options.convergenceThreshold - Loss threshold for convergence (default: 1e-6)
   * @param options.outlierThreshold - Z-score threshold for outlier detection (default: 3.0)
   * @param options.adwinDelta - ADWIN confidence parameter (default: 0.002)
   */
  constructor(options: Partial<Config> = {}) {
    this.config = {
      numBlocks: options.numBlocks ?? 3,
      embeddingDim: options.embeddingDim ?? 64,
      numHeads: options.numHeads ?? 8,
      ffnMultiplier: options.ffnMultiplier ?? 4,
      attentionDropout: options.attentionDropout ?? 0.0,
      learningRate: options.learningRate ?? 0.001,
      warmupSteps: options.warmupSteps ?? 100,
      totalSteps: options.totalSteps ?? 10000,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      epsilon: options.epsilon ?? 1e-8,
      regularizationStrength: options.regularizationStrength ?? 1e-4,
      convergenceThreshold: options.convergenceThreshold ?? 1e-6,
      outlierThreshold: options.outlierThreshold ?? 3.0,
      adwinDelta: options.adwinDelta ?? 0.002,
    };

    // Validate configuration
    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${this.config.embeddingDim}) must be divisible by numHeads (${this.config.numHeads})`,
      );
    }

    this.pool = new ArrayPool();
    this.adwin = new ADWIN(this.config.adwinDelta);
  }

  /**
   * Initialize model architecture based on detected dimensions
   * @private
   */
  private initialize(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.seqLen = inputDim + 1; // +1 for CLS token
    this.headDim = this.config.embeddingDim / this.config.numHeads;
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;

    const ed = this.config.embeddingDim;

    // Xavier initialization scale
    const embScale = Math.sqrt(2 / (1 + ed));
    const attnScale = Math.sqrt(2 / (ed + ed));
    const ffnScale = Math.sqrt(2 / (ed + this.ffnDim));
    const outScale = Math.sqrt(2 / (ed + outputDim));

    // Feature embeddings: one weight vector per input feature
    // Each feature xᵢ → eᵢ = xᵢ · Wᵢ + bᵢ
    this.featureWeights = [];
    this.featureBiases = [];
    this.featureWeightsM = [];
    this.featureWeightsV = [];
    this.featureBiasesM = [];
    this.featureBiasesV = [];

    for (let i = 0; i < inputDim; i++) {
      const w = new Float64Array(ed);
      const b = new Float64Array(ed);
      for (let j = 0; j < ed; j++) {
        w[j] = (Math.random() - 0.5) * 2 * embScale;
      }
      this.featureWeights.push(w);
      this.featureBiases.push(b);
      this.featureWeightsM.push(new Float64Array(ed));
      this.featureWeightsV.push(new Float64Array(ed));
      this.featureBiasesM.push(new Float64Array(ed));
      this.featureBiasesV.push(new Float64Array(ed));
    }

    // CLS token
    this.clsToken = new Float64Array(ed);
    for (let i = 0; i < ed; i++) {
      this.clsToken[i] = (Math.random() - 0.5) * 2 * embScale;
    }
    this.clsTokenM = new Float64Array(ed);
    this.clsTokenV = new Float64Array(ed);

    // Transformer blocks
    this.blocks = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = new TransformerBlockWeights(ed, this.ffnDim);
      // Initialize attention weights
      for (let i = 0; i < block.wq.length; i++) {
        block.wq[i] = (Math.random() - 0.5) * 2 * attnScale;
        block.wk[i] = (Math.random() - 0.5) * 2 * attnScale;
        block.wv[i] = (Math.random() - 0.5) * 2 * attnScale;
        block.wo[i] = (Math.random() - 0.5) * 2 * attnScale;
      }
      // Initialize FFN weights
      for (let i = 0; i < block.w1.length; i++) {
        block.w1[i] = (Math.random() - 0.5) * 2 * ffnScale;
      }
      for (let i = 0; i < block.w2.length; i++) {
        block.w2[i] = (Math.random() - 0.5) * 2 * ffnScale;
      }
      this.blocks.push(block);
    }

    // Output layer
    this.wout = new Float64Array(ed * outputDim);
    this.bout = new Float64Array(outputDim);
    for (let i = 0; i < this.wout.length; i++) {
      this.wout[i] = (Math.random() - 0.5) * 2 * outScale;
    }
    this.woutM = new Float64Array(ed * outputDim);
    this.woutV = new Float64Array(ed * outputDim);
    this.boutM = new Float64Array(outputDim);
    this.boutV = new Float64Array(outputDim);

    // Statistics
    this.inputStats = new WelfordStats(inputDim);
    this.outputStats = new WelfordStats(outputDim);
    this.predictionVariance = new Float64Array(outputDim);
    this.predictionVariance.fill(1);

    // Caches
    this.forwardCache = new ForwardCache(
      this.seqLen,
      ed,
      this.ffnDim,
      this.config.numBlocks,
      this.config.numHeads,
      outputDim,
    );
    this.gradients = new GradientStorage(
      inputDim,
      ed,
      this.ffnDim,
      outputDim,
      this.config.numBlocks,
    );

    // Temp buffers
    this.tempInput = new Float64Array(inputDim);
    this.tempOutput = new Float64Array(outputDim);
    this.tempSeqGrad = new Float64Array(this.seqLen * ed);
    this.lastInput = new Float64Array(inputDim);

    this.isInitialized = true;
  }

  /**
   * Perform online learning with a batch of samples
   *
   * Processes samples incrementally, updating model weights after each sample.
   *
   * Algorithm:
   * 1. Normalize input using Welford's running statistics: x̃ = (x - μ)/(σ + ε)
   * 2. Forward pass through transformer blocks
   * 3. Compute MSE loss: L = (1/2n)Σ‖y - ŷ‖² + (λ/2)Σ‖W‖²
   * 4. Backward pass to compute gradients
   * 5. Update weights with Adam optimizer
   * 6. Detect outliers and drift
   *
   * @param data - Training data with xCoordinates and yCoordinates
   * @returns FitResult for the last processed sample
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2, 3], [4, 5, 6]],
   *   yCoordinates: [[7, 8], [9, 10]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(data: FitInput): FitResult {
    const { xCoordinates, yCoordinates } = data;

    if (!xCoordinates || !yCoordinates || xCoordinates.length === 0) {
      throw new Error("Input data cannot be empty");
    }

    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error("xCoordinates and yCoordinates must have same length");
    }

    // Auto-detect dimensions from first sample if not initialized
    if (!this.isInitialized) {
      this.initialize(xCoordinates[0].length, yCoordinates[0].length);
    }

    let lastResult: FitResult = {
      loss: 0,
      gradientNorm: 0,
      effectiveLearningRate: 0,
      isOutlier: false,
      converged: false,
      sampleIndex: this.sampleCount,
      driftDetected: false,
    };

    // Process each sample
    for (let i = 0; i < xCoordinates.length; i++) {
      lastResult = this.fitSingleSample(xCoordinates[i], yCoordinates[i]);
    }

    return lastResult;
  }

  /**
   * Process a single training sample
   * @private
   */
  private fitSingleSample(x: number[], y: number[]): FitResult {
    this.sampleCount++;
    this.updateCount++;

    // Copy input to typed array and update statistics
    for (let i = 0; i < this.inputDim; i++) {
      this.tempInput![i] = x[i];
    }
    for (let i = 0; i < this.outputDim; i++) {
      this.tempOutput![i] = y[i];
    }

    // Update running statistics with Welford's algorithm
    this.inputStats!.update(this.tempInput!);
    this.outputStats!.update(this.tempOutput!);

    // Normalize input: x̃ = (x - μ)/(σ + ε)
    const inputMean = this.inputStats!.getMean();
    const inputStd = this.inputStats!.getStd(this.config.epsilon);
    const normalizedInput = this.pool.acquire(this.inputDim);
    for (let i = 0; i < this.inputDim; i++) {
      normalizedInput[i] = (this.tempInput![i] - inputMean[i]) / inputStd[i];
    }

    // Normalize target
    const outputMean = this.outputStats!.getMean();
    const outputStd = this.outputStats!.getStd(this.config.epsilon);
    const normalizedTarget = this.pool.acquire(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) {
      normalizedTarget[i] = (this.tempOutput![i] - outputMean[i]) /
        outputStd[i];
    }

    // Forward pass
    this.forward(normalizedInput);

    // Compute loss: L = (1/2)‖y - ŷ‖² + (λ/2)‖W‖²
    let loss = 0;
    for (let i = 0; i < this.outputDim; i++) {
      const diff = normalizedTarget[i] - this.forwardCache!.output[i];
      loss += diff * diff;
    }
    loss = loss / 2;

    // Add L2 regularization
    const regLoss = this.computeRegularizationLoss();
    const totalLoss = loss + regLoss;

    // Check for outlier using standardized residuals
    // r = (y - ŷ)/σ; outlier if |r| > threshold
    let isOutlier = false;
    let maxResidual = 0;
    for (let i = 0; i < this.outputDim; i++) {
      const residual = Math.abs(
        normalizedTarget[i] - this.forwardCache!.output[i],
      );
      if (residual > maxResidual) maxResidual = residual;
    }
    if (maxResidual > this.config.outlierThreshold) {
      isOutlier = true;
    }

    // Compute weight for outlier downweighting
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // Backward pass
    this.gradients!.reset();
    this.backward(normalizedInput, normalizedTarget, sampleWeight);

    // Compute gradient norm
    const gradientNorm = this.gradients!.computeNorm();

    // Compute learning rate with warmup and cosine decay
    // warmup: lr * step / warmupSteps
    // decay: lr * 0.5 * (1 + cos(π * (step - warmupSteps) / (totalSteps - warmupSteps)))
    let effectiveLR = this.config.learningRate;
    if (this.updateCount < this.config.warmupSteps) {
      effectiveLR = this.config.learningRate * this.updateCount /
        this.config.warmupSteps;
    } else {
      const decayProgress = (this.updateCount - this.config.warmupSteps) /
        Math.max(1, this.config.totalSteps - this.config.warmupSteps);
      effectiveLR = this.config.learningRate * 0.5 *
        (1 + Math.cos(Math.PI * Math.min(1, decayProgress)));
    }

    // Adam update
    this.adamUpdate(effectiveLR);

    // Update running loss for accuracy tracking
    this.runningLossSum += loss;
    this.runningLossCount++;

    // Update prediction variance
    for (let i = 0; i < this.outputDim; i++) {
      const diff = normalizedTarget[i] - this.forwardCache!.output[i];
      const oldVar = this.predictionVariance![i];
      // Exponential moving average of variance
      this.predictionVariance![i] = 0.99 * oldVar + 0.01 * diff * diff;
    }

    // Check for drift using ADWIN
    const driftDetected = this.adwin.update(loss);
    if (driftDetected) {
      this.driftCount++;
      // Optionally reset some statistics on drift
      // For now, we just track it
    }

    // Check convergence
    const avgLoss = this.runningLossSum / this.runningLossCount;
    this.converged = avgLoss < this.config.convergenceThreshold;

    // Store last input for prediction
    for (let i = 0; i < this.inputDim; i++) {
      this.lastInput![i] = this.tempInput![i];
    }

    // Release temporary arrays
    this.pool.release(normalizedInput);
    this.pool.release(normalizedTarget);

    return {
      loss: totalLoss,
      gradientNorm,
      effectiveLearningRate: effectiveLR,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Forward pass through the transformer
   *
   * Architecture:
   * 1. Feature embedding: eᵢ = xᵢ · Wᵢ + bᵢ
   * 2. Prepend CLS: sequence = [CLS, e₁, ..., eₙ]
   * 3. For each block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
   * 4. Output: Dense(CLS_final) → ŷ
   *
   * @private
   */
  private forward(normalizedInput: Float64Array): void {
    const ed = this.config.embeddingDim;
    const cache = this.forwardCache!;
    cache.reset();

    // Build embeddings sequence: [CLS, e₁, ..., eₙ]
    // CLS token at position 0
    for (let j = 0; j < ed; j++) {
      cache.embeddings[j] = this.clsToken![j];
    }

    // Feature embeddings at positions 1 to inputDim
    for (let i = 0; i < this.inputDim; i++) {
      const offset = (i + 1) * ed;
      const xi = normalizedInput[i];
      const w = this.featureWeights[i];
      const b = this.featureBiases[i];
      for (let j = 0; j < ed; j++) {
        cache.embeddings[offset + j] = xi * w[j] + b[j];
      }
    }

    // Process through transformer blocks
    let currentInput = cache.embeddings;
    for (let blockIdx = 0; blockIdx < this.config.numBlocks; blockIdx++) {
      const block = this.blocks[blockIdx];
      const blockCache = cache.blockCaches[blockIdx];

      // Copy input to block cache
      for (let i = 0; i < currentInput.length; i++) {
        blockCache.input[i] = currentInput[i];
      }

      // Layer Norm 1
      this.layerNorm(
        currentInput,
        blockCache.ln1Out,
        block.ln1Gamma,
        block.ln1Beta,
        blockCache.ln1Mean,
        blockCache.ln1Var,
      );

      // Multi-Head Self-Attention
      this.multiHeadAttention(
        blockCache.ln1Out,
        block,
        blockCache.q,
        blockCache.k,
        blockCache.v,
        blockCache.attnScores,
        blockCache.attnProbs,
        blockCache.attnOut,
        blockCache.multiHeadOut,
      );

      // Residual connection 1
      for (let i = 0; i < blockCache.residual1.length; i++) {
        blockCache.residual1[i] = currentInput[i] + blockCache.multiHeadOut[i];
      }

      // Layer Norm 2
      this.layerNorm(
        blockCache.residual1,
        blockCache.ln2Out,
        block.ln2Gamma,
        block.ln2Beta,
        blockCache.ln2Mean,
        blockCache.ln2Var,
      );

      // Feed-Forward Network
      this.feedForward(
        blockCache.ln2Out,
        block,
        blockCache.ffnHiddenPreAct,
        blockCache.ffnHidden,
        blockCache.ffnOut,
      );

      // Residual connection 2
      for (let i = 0; i < blockCache.output.length; i++) {
        blockCache.output[i] = blockCache.residual1[i] + blockCache.ffnOut[i];
      }

      currentInput = blockCache.output;
    }

    // Extract CLS token (first position)
    for (let j = 0; j < ed; j++) {
      cache.clsFinal[j] = currentInput[j];
    }

    // Output projection: ŷ = CLS_final · Wout + bout
    for (let i = 0; i < this.outputDim; i++) {
      let sum = this.bout![i];
      for (let j = 0; j < ed; j++) {
        sum += cache.clsFinal[j] * this.wout![i * ed + j];
      }
      cache.output[i] = sum;
    }
  }

  /**
   * Layer Normalization: LN(x) = γ · (x - μ) / √(σ² + ε) + β
   * @private
   */
  private layerNorm(
    input: Float64Array,
    output: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    mean: Float64Array,
    variance: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const eps = this.config.epsilon;

    // For each position in sequence
    for (let pos = 0; pos < this.seqLen; pos++) {
      const offset = pos * ed;

      // Compute mean
      let sum = 0;
      for (let j = 0; j < ed; j++) {
        sum += input[offset + j];
      }
      const m = sum / ed;
      mean[pos] = m;

      // Compute variance
      let varSum = 0;
      for (let j = 0; j < ed; j++) {
        const diff = input[offset + j] - m;
        varSum += diff * diff;
      }
      const v = varSum / ed;
      variance[pos] = v;

      // Normalize
      const invStd = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < ed; j++) {
        output[offset + j] = gamma[j] * (input[offset + j] - m) * invStd +
          beta[j];
      }
    }
  }

  /**
   * Multi-Head Self-Attention
   *
   * Q = X·Wq, K = X·Wk, V = X·Wv
   * Attention(Q,K,V) = softmax(QK^T / √d_k)V
   * MultiHead = Concat(head_1, ..., head_h)·Wo + bo
   *
   * @private
   */
  private multiHeadAttention(
    input: Float64Array,
    block: TransformerBlockWeights,
    q: Float64Array,
    k: Float64Array,
    v: Float64Array,
    attnScores: Float64Array,
    attnProbs: Float64Array,
    attnOut: Float64Array,
    output: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const numHeads = this.config.numHeads;
    const headDim = this.headDim;
    const scale = 1 / Math.sqrt(headDim);

    // Compute Q, K, V projections
    // Q = input · Wq, etc. where Wq is [ed, ed]
    this.matmul(input, this.seqLen, ed, block.wq, ed, ed, q);
    this.matmul(input, this.seqLen, ed, block.wk, ed, ed, k);
    this.matmul(input, this.seqLen, ed, block.wv, ed, ed, v);

    // For each head
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scoreOffset = h * this.seqLen * this.seqLen;

      // Compute attention scores: Q_h · K_h^T / √d_k
      for (let i = 0; i < this.seqLen; i++) {
        for (let j = 0; j < this.seqLen; j++) {
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += q[i * ed + headOffset + d] * k[j * ed + headOffset + d];
          }
          attnScores[scoreOffset + i * this.seqLen + j] = score * scale;
        }
      }

      // Softmax over keys for each query
      for (let i = 0; i < this.seqLen; i++) {
        const rowOffset = scoreOffset + i * this.seqLen;

        // Find max for numerical stability
        let max = -Infinity;
        for (let j = 0; j < this.seqLen; j++) {
          if (attnScores[rowOffset + j] > max) {
            max = attnScores[rowOffset + j];
          }
        }

        // Compute exp and sum
        let sum = 0;
        for (let j = 0; j < this.seqLen; j++) {
          attnProbs[rowOffset + j] = Math.exp(attnScores[rowOffset + j] - max);
          sum += attnProbs[rowOffset + j];
        }

        // Normalize
        const invSum = 1 / (sum + this.config.epsilon);
        for (let j = 0; j < this.seqLen; j++) {
          attnProbs[rowOffset + j] *= invSum;
        }
      }

      // Compute attention output: attnProbs · V_h
      for (let i = 0; i < this.seqLen; i++) {
        const rowOffset = scoreOffset + i * this.seqLen;
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < this.seqLen; j++) {
            sum += attnProbs[rowOffset + j] * v[j * ed + headOffset + d];
          }
          attnOut[i * ed + headOffset + d] = sum;
        }
      }
    }

    // Output projection: attnOut · Wo + bo
    this.matmul(attnOut, this.seqLen, ed, block.wo, ed, ed, output);
    for (let i = 0; i < this.seqLen; i++) {
      const offset = i * ed;
      for (let j = 0; j < ed; j++) {
        output[offset + j] += block.bo[j];
      }
    }
  }

  /**
   * Feed-Forward Network with GELU activation
   * FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
   *
   * GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
   *
   * @private
   */
  private feedForward(
    input: Float64Array,
    block: TransformerBlockWeights,
    hiddenPreAct: Float64Array,
    hidden: Float64Array,
    output: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const fd = this.ffnDim;

    // First layer: hidden = input · W1 + b1
    this.matmul(input, this.seqLen, ed, block.w1, ed, fd, hiddenPreAct);
    for (let i = 0; i < this.seqLen; i++) {
      const offset = i * fd;
      for (let j = 0; j < fd; j++) {
        hiddenPreAct[offset + j] += block.b1[j];
      }
    }

    // GELU activation
    const SQRT_2_PI = 0.7978845608028654;
    for (let i = 0; i < hiddenPreAct.length; i++) {
      const x = hiddenPreAct[i];
      const t = SQRT_2_PI * (x + 0.044715 * x * x * x);
      hidden[i] = 0.5 * x * (1 + Math.tanh(t));
    }

    // Second layer: output = hidden · W2 + b2
    this.matmul(hidden, this.seqLen, fd, block.w2, fd, ed, output);
    for (let i = 0; i < this.seqLen; i++) {
      const offset = i * ed;
      for (let j = 0; j < ed; j++) {
        output[offset + j] += block.b2[j];
      }
    }
  }

  /**
   * Matrix multiplication: C = A × B
   * A: [m, k], B: [k, n], C: [m, n]
   * @private
   */
  private matmul(
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    _bRows: number,
    n: number,
    C: Float64Array,
  ): void {
    // Initialize C to zeros
    C.fill(0);

    // Standard matrix multiplication with loop reordering for cache efficiency
    for (let i = 0; i < m; i++) {
      const rowOffsetA = i * k;
      const rowOffsetC = i * n;
      for (let p = 0; p < k; p++) {
        const aVal = A[rowOffsetA + p];
        const rowOffsetB = p * n;
        for (let j = 0; j < n; j++) {
          C[rowOffsetC + j] += aVal * B[rowOffsetB + j];
        }
      }
    }
  }

  /**
   * Backward pass to compute gradients
   * @private
   */
  private backward(
    normalizedInput: Float64Array,
    normalizedTarget: Float64Array,
    sampleWeight: number,
  ): void {
    const ed = this.config.embeddingDim;
    const cache = this.forwardCache!;
    const grads = this.gradients!;

    // Gradient of output loss: dL/dŷ = (ŷ - y) * weight
    const dOutput = this.pool.acquire(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) {
      dOutput[i] = (cache.output[i] - normalizedTarget[i]) * sampleWeight;
    }

    // Gradient of output layer
    // ŷ = CLS · Wout + bout
    // dL/dWout = CLS^T · dOutput
    // dL/dbout = dOutput
    // dL/dCLS = dOutput · Wout^T
    for (let i = 0; i < this.outputDim; i++) {
      grads.boutGrad[i] = dOutput[i];
      for (let j = 0; j < ed; j++) {
        grads.woutGrad[i * ed + j] = dOutput[i] * cache.clsFinal[j];
      }
    }

    // dCLS = dOutput · Wout^T
    const dCLS = this.pool.acquire(ed);
    for (let j = 0; j < ed; j++) {
      let sum = 0;
      for (let i = 0; i < this.outputDim; i++) {
        sum += dOutput[i] * this.wout![i * ed + j];
      }
      dCLS[j] = sum;
    }

    // Initialize sequence gradient (only CLS position has gradient initially)
    const dSeq = this.tempSeqGrad!;
    dSeq.fill(0);
    for (let j = 0; j < ed; j++) {
      dSeq[j] = dCLS[j];
    }

    // Backward through transformer blocks (reverse order)
    for (let blockIdx = this.config.numBlocks - 1; blockIdx >= 0; blockIdx--) {
      this.backwardBlock(blockIdx, dSeq, normalizedInput);
    }

    // Gradient of embeddings
    // CLS token gradient
    for (let j = 0; j < ed; j++) {
      grads.clsTokenGrad[j] = dSeq[j];
    }

    // Feature embedding gradients
    for (let i = 0; i < this.inputDim; i++) {
      const offset = (i + 1) * ed;
      const xi = normalizedInput[i];
      for (let j = 0; j < ed; j++) {
        // eᵢ = xᵢ · Wᵢ + bᵢ
        // dL/dWᵢ = dL/deᵢ · xᵢ
        // dL/dbᵢ = dL/deᵢ
        grads.featureWeightsGrad[i][j] = dSeq[offset + j] * xi;
        grads.featureBiasesGrad[i][j] = dSeq[offset + j];
      }
    }

    // Add L2 regularization gradients
    this.addRegularizationGradients();

    this.pool.release(dOutput);
    this.pool.release(dCLS);
  }

  /**
   * Backward pass through a single transformer block
   * @private
   */
  private backwardBlock(
    blockIdx: number,
    dSeq: Float64Array,
    normalizedInput: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const fd = this.ffnDim;
    const block = this.blocks[blockIdx];
    const blockCache = this.forwardCache!.blockCaches[blockIdx];
    const blockGrads = this.gradients!.blockGrads[blockIdx];

    // dSeq is gradient w.r.t. block output
    // output = residual1 + ffnOut
    // dResidual1 = dSeq
    // dFfnOut = dSeq

    // Backward through FFN
    const dFfnOut = this.pool.acquire(this.seqLen * ed);
    for (let i = 0; i < dSeq.length; i++) {
      dFfnOut[i] = dSeq[i];
    }

    // FFN backward: output = hidden · W2 + b2
    const dHidden = this.pool.acquire(this.seqLen * fd);
    const dLn2Out = this.pool.acquire(this.seqLen * ed);

    // dHidden = dFfnOut · W2^T
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < fd; j++) {
        let sum = 0;
        for (let k = 0; k < ed; k++) {
          sum += dFfnOut[i * ed + k] * block.w2[j * ed + k];
        }
        dHidden[i * fd + j] = sum;
      }
    }

    // dW2 = hidden^T · dFfnOut
    for (let j = 0; j < fd; j++) {
      for (let k = 0; k < ed; k++) {
        let sum = 0;
        for (let i = 0; i < this.seqLen; i++) {
          sum += blockCache.ffnHidden[i * fd + j] * dFfnOut[i * ed + k];
        }
        blockGrads.w2Grad[j * ed + k] = sum;
      }
    }

    // db2 = sum over sequence of dFfnOut
    for (let k = 0; k < ed; k++) {
      let sum = 0;
      for (let i = 0; i < this.seqLen; i++) {
        sum += dFfnOut[i * ed + k];
      }
      blockGrads.b2Grad[k] = sum;
    }

    // Backward through GELU
    // GELU'(x) ≈ 0.5(1 + tanh(t)) + 0.5x · sech²(t) · (√(2/π) + 3·0.044715·√(2/π)·x²)
    const dHiddenPreAct = this.pool.acquire(this.seqLen * fd);
    const SQRT_2_PI = 0.7978845608028654;
    for (let i = 0; i < this.seqLen * fd; i++) {
      const x = blockCache.ffnHiddenPreAct[i];
      const t = SQRT_2_PI * (x + 0.044715 * x * x * x);
      const tanhT = Math.tanh(t);
      const sech2T = 1 - tanhT * tanhT;
      const dtdx = SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
      const geluGrad = 0.5 * (1 + tanhT) + 0.5 * x * sech2T * dtdx;
      dHiddenPreAct[i] = dHidden[i] * geluGrad;
    }

    // dLn2Out = dHiddenPreAct · W1^T
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < ed; j++) {
        let sum = 0;
        for (let k = 0; k < fd; k++) {
          sum += dHiddenPreAct[i * fd + k] * block.w1[j * fd + k];
        }
        dLn2Out[i * ed + j] = sum;
      }
    }

    // dW1 = ln2Out^T · dHiddenPreAct
    for (let j = 0; j < ed; j++) {
      for (let k = 0; k < fd; k++) {
        let sum = 0;
        for (let i = 0; i < this.seqLen; i++) {
          sum += blockCache.ln2Out[i * ed + j] * dHiddenPreAct[i * fd + k];
        }
        blockGrads.w1Grad[j * fd + k] = sum;
      }
    }

    // db1 = sum over sequence of dHiddenPreAct
    for (let k = 0; k < fd; k++) {
      let sum = 0;
      for (let i = 0; i < this.seqLen; i++) {
        sum += dHiddenPreAct[i * fd + k];
      }
      blockGrads.b1Grad[k] = sum;
    }

    // Backward through LayerNorm 2
    const dResidual1 = this.pool.acquire(this.seqLen * ed);
    this.backwardLayerNorm(
      dLn2Out,
      blockCache.residual1,
      blockCache.ln2Mean,
      blockCache.ln2Var,
      block.ln2Gamma,
      dResidual1,
      blockGrads.ln2GammaGrad,
      blockGrads.ln2BetaGrad,
    );

    // Add residual gradient: dResidual1 += dSeq (from residual connection)
    for (let i = 0; i < dSeq.length; i++) {
      dResidual1[i] += dSeq[i];
    }

    // Backward through attention
    const dMultiHeadOut = this.pool.acquire(this.seqLen * ed);
    for (let i = 0; i < dResidual1.length; i++) {
      dMultiHeadOut[i] = dResidual1[i];
    }

    const dLn1Out = this.pool.acquire(this.seqLen * ed);
    this.backwardAttention(blockIdx, dMultiHeadOut, dLn1Out);

    // Backward through LayerNorm 1
    const dInput = this.pool.acquire(this.seqLen * ed);
    this.backwardLayerNorm(
      dLn1Out,
      blockCache.input,
      blockCache.ln1Mean,
      blockCache.ln1Var,
      block.ln1Gamma,
      dInput,
      blockGrads.ln1GammaGrad,
      blockGrads.ln1BetaGrad,
    );

    // Add residual gradient: dInput += dResidual1 (from residual connection)
    for (let i = 0; i < dSeq.length; i++) {
      dSeq[i] = dInput[i] + dResidual1[i];
    }

    // Release temporary arrays
    this.pool.release(dFfnOut);
    this.pool.release(dHidden);
    this.pool.release(dHiddenPreAct);
    this.pool.release(dLn2Out);
    this.pool.release(dResidual1);
    this.pool.release(dMultiHeadOut);
    this.pool.release(dLn1Out);
    this.pool.release(dInput);
  }

  /**
   * Backward pass through layer normalization
   * @private
   */
  private backwardLayerNorm(
    dOut: Float64Array,
    input: Float64Array,
    mean: Float64Array,
    variance: Float64Array,
    gamma: Float64Array,
    dInput: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const eps = this.config.epsilon;

    // Initialize gradients
    dGamma.fill(0);
    dBeta.fill(0);

    for (let pos = 0; pos < this.seqLen; pos++) {
      const offset = pos * ed;
      const m = mean[pos];
      const v = variance[pos];
      const invStd = 1 / Math.sqrt(v + eps);

      // Accumulate gamma and beta gradients
      for (let j = 0; j < ed; j++) {
        const normalized = (input[offset + j] - m) * invStd;
        dGamma[j] += dOut[offset + j] * normalized;
        dBeta[j] += dOut[offset + j];
      }

      // Compute dInput
      // Let x̂ = (x - μ) / σ, y = γx̂ + β
      // dy/dx = γ/σ - γ/(nσ) - γx̂²/(nσ) + terms...
      // This is the full layer norm gradient
      let dxhatSum = 0;
      let dxhatXhatSum = 0;
      for (let j = 0; j < ed; j++) {
        const xhat = (input[offset + j] - m) * invStd;
        const dxhat = dOut[offset + j] * gamma[j];
        dxhatSum += dxhat;
        dxhatXhatSum += dxhat * xhat;
      }

      for (let j = 0; j < ed; j++) {
        const xhat = (input[offset + j] - m) * invStd;
        const dxhat = dOut[offset + j] * gamma[j];
        dInput[offset + j] = invStd *
          (dxhat - dxhatSum / ed - xhat * dxhatXhatSum / ed);
      }
    }
  }

  /**
   * Backward pass through multi-head attention
   * @private
   */
  private backwardAttention(
    blockIdx: number,
    dOutput: Float64Array,
    dInput: Float64Array,
  ): void {
    const ed = this.config.embeddingDim;
    const numHeads = this.config.numHeads;
    const headDim = this.headDim;
    const scale = 1 / Math.sqrt(headDim);
    const block = this.blocks[blockIdx];
    const blockCache = this.forwardCache!.blockCaches[blockIdx];
    const blockGrads = this.gradients!.blockGrads[blockIdx];

    // dOutput is [seqLen, ed]
    // output = attnOut · Wo + bo

    // dAttnOut = dOutput · Wo^T
    const dAttnOut = this.pool.acquire(this.seqLen * ed);
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < ed; j++) {
        let sum = 0;
        for (let k = 0; k < ed; k++) {
          sum += dOutput[i * ed + k] * block.wo[j * ed + k];
        }
        dAttnOut[i * ed + j] = sum;
      }
    }

    // dWo = attnOut^T · dOutput
    for (let j = 0; j < ed; j++) {
      for (let k = 0; k < ed; k++) {
        let sum = 0;
        for (let i = 0; i < this.seqLen; i++) {
          sum += blockCache.attnOut[i * ed + j] * dOutput[i * ed + k];
        }
        blockGrads.woGrad[j * ed + k] = sum;
      }
    }

    // dbo = sum of dOutput over sequence
    for (let k = 0; k < ed; k++) {
      let sum = 0;
      for (let i = 0; i < this.seqLen; i++) {
        sum += dOutput[i * ed + k];
      }
      blockGrads.boGrad[k] = sum;
    }

    // Initialize Q, K, V gradients
    const dQ = this.pool.acquire(this.seqLen * ed);
    const dK = this.pool.acquire(this.seqLen * ed);
    const dV = this.pool.acquire(this.seqLen * ed);
    dQ.fill(0);
    dK.fill(0);
    dV.fill(0);

    // Backward through each attention head
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      const scoreOffset = h * this.seqLen * this.seqLen;

      // attnOut[i,h,:] = sum_j attnProbs[i,j] * V[j,h,:]
      // dAttnProbs[i,j] = sum_d dAttnOut[i,h,d] * V[j,h,d]
      // dV[j,h,d] += sum_i attnProbs[i,j] * dAttnOut[i,h,d]

      const dAttnProbs = this.pool.acquire(this.seqLen * this.seqLen);

      for (let i = 0; i < this.seqLen; i++) {
        for (let j = 0; j < this.seqLen; j++) {
          let sum = 0;
          for (let d = 0; d < headDim; d++) {
            sum += dAttnOut[i * ed + headOffset + d] *
              blockCache.v[j * ed + headOffset + d];
          }
          dAttnProbs[i * this.seqLen + j] = sum;
        }
      }

      for (let j = 0; j < this.seqLen; j++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let i = 0; i < this.seqLen; i++) {
            sum += blockCache.attnProbs[scoreOffset + i * this.seqLen + j] *
              dAttnOut[i * ed + headOffset + d];
          }
          dV[j * ed + headOffset + d] += sum;
        }
      }

      // Backward through softmax
      // If S = softmax(Z), dZ = S ⊙ (dS - sum(dS ⊙ S))
      const dScores = this.pool.acquire(this.seqLen * this.seqLen);
      for (let i = 0; i < this.seqLen; i++) {
        const rowOffset = i * this.seqLen;
        let dotProduct = 0;
        for (let j = 0; j < this.seqLen; j++) {
          dotProduct += dAttnProbs[rowOffset + j] *
            blockCache.attnProbs[scoreOffset + rowOffset + j];
        }
        for (let j = 0; j < this.seqLen; j++) {
          const prob = blockCache.attnProbs[scoreOffset + rowOffset + j];
          dScores[rowOffset + j] = prob *
            (dAttnProbs[rowOffset + j] - dotProduct);
        }
      }

      // scores = Q · K^T / sqrt(d_k)
      // dQ += dScores · K / sqrt(d_k)
      // dK += dScores^T · Q / sqrt(d_k)
      for (let i = 0; i < this.seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          let sumQ = 0;
          let sumK = 0;
          for (let j = 0; j < this.seqLen; j++) {
            sumQ += dScores[i * this.seqLen + j] *
              blockCache.k[j * ed + headOffset + d];
            sumK += dScores[j * this.seqLen + i] *
              blockCache.q[j * ed + headOffset + d];
          }
          dQ[i * ed + headOffset + d] += sumQ * scale;
          dK[i * ed + headOffset + d] += sumK * scale;
        }
      }

      this.pool.release(dAttnProbs);
      this.pool.release(dScores);
    }

    // Backward through Q, K, V projections
    // Q = ln1Out · Wq, so dLn1Out += dQ · Wq^T, dWq = ln1Out^T · dQ
    dInput.fill(0);

    // dWq, dWk, dWv
    for (let j = 0; j < ed; j++) {
      for (let k = 0; k < ed; k++) {
        let sumQ = 0, sumK = 0, sumV = 0;
        for (let i = 0; i < this.seqLen; i++) {
          sumQ += blockCache.ln1Out[i * ed + j] * dQ[i * ed + k];
          sumK += blockCache.ln1Out[i * ed + j] * dK[i * ed + k];
          sumV += blockCache.ln1Out[i * ed + j] * dV[i * ed + k];
        }
        blockGrads.wqGrad[j * ed + k] = sumQ;
        blockGrads.wkGrad[j * ed + k] = sumK;
        blockGrads.wvGrad[j * ed + k] = sumV;
      }
    }

    // dInput = dQ · Wq^T + dK · Wk^T + dV · Wv^T
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < ed; j++) {
        let sum = 0;
        for (let k = 0; k < ed; k++) {
          sum += dQ[i * ed + k] * block.wq[j * ed + k];
          sum += dK[i * ed + k] * block.wk[j * ed + k];
          sum += dV[i * ed + k] * block.wv[j * ed + k];
        }
        dInput[i * ed + j] = sum;
      }
    }

    this.pool.release(dAttnOut);
    this.pool.release(dQ);
    this.pool.release(dK);
    this.pool.release(dV);
  }

  /**
   * Compute L2 regularization loss: (λ/2)Σ‖W‖²
   * @private
   */
  private computeRegularizationLoss(): number {
    const lambda = this.config.regularizationStrength;
    let sum = 0;

    // Feature embeddings
    for (let i = 0; i < this.inputDim; i++) {
      sum += this.sumSquared(this.featureWeights[i]);
    }

    // CLS token
    sum += this.sumSquared(this.clsToken!);

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      sum += this.sumSquared(block.wq);
      sum += this.sumSquared(block.wk);
      sum += this.sumSquared(block.wv);
      sum += this.sumSquared(block.wo);
      sum += this.sumSquared(block.w1);
      sum += this.sumSquared(block.w2);
    }

    // Output layer
    sum += this.sumSquared(this.wout!);

    return 0.5 * lambda * sum;
  }

  /**
   * Add L2 regularization gradients to computed gradients
   * @private
   */
  private addRegularizationGradients(): void {
    const lambda = this.config.regularizationStrength;
    const grads = this.gradients!;

    // Feature embeddings
    for (let i = 0; i < this.inputDim; i++) {
      for (let j = 0; j < this.config.embeddingDim; j++) {
        grads.featureWeightsGrad[i][j] += lambda * this.featureWeights[i][j];
      }
    }

    // CLS token
    for (let j = 0; j < this.config.embeddingDim; j++) {
      grads.clsTokenGrad[j] += lambda * this.clsToken![j];
    }

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      const blockGrads = grads.blockGrads[b];

      this.addRegArray(blockGrads.wqGrad, block.wq, lambda);
      this.addRegArray(blockGrads.wkGrad, block.wk, lambda);
      this.addRegArray(blockGrads.wvGrad, block.wv, lambda);
      this.addRegArray(blockGrads.woGrad, block.wo, lambda);
      this.addRegArray(blockGrads.w1Grad, block.w1, lambda);
      this.addRegArray(blockGrads.w2Grad, block.w2, lambda);
    }

    // Output layer
    this.addRegArray(grads.woutGrad, this.wout!, lambda);
  }

  /**
   * Helper to add regularization gradient
   * @private
   */
  private addRegArray(
    grad: Float64Array,
    weight: Float64Array,
    lambda: number,
  ): void {
    for (let i = 0; i < grad.length; i++) {
      grad[i] += lambda * weight[i];
    }
  }

  /**
   * Sum of squared elements
   * @private
   */
  private sumSquared(arr: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i] * arr[i];
    }
    return sum;
  }

  /**
   * Adam optimizer update
   *
   * m = β₁m + (1-β₁)g
   * v = β₂v + (1-β₂)g²
   * m̂ = m/(1-β₁ᵗ)
   * v̂ = v/(1-β₂ᵗ)
   * W -= η · m̂/(√v̂ + ε)
   *
   * @private
   */
  private adamUpdate(lr: number): void {
    const beta1 = this.config.beta1;
    const beta2 = this.config.beta2;
    const eps = this.config.epsilon;
    const t = this.updateCount;

    // Bias correction
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    const grads = this.gradients!;

    // Update feature embeddings
    for (let i = 0; i < this.inputDim; i++) {
      this.adamUpdateArray(
        this.featureWeights[i],
        grads.featureWeightsGrad[i],
        this.featureWeightsM[i],
        this.featureWeightsV[i],
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        this.featureBiases[i],
        grads.featureBiasesGrad[i],
        this.featureBiasesM[i],
        this.featureBiasesV[i],
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
    }

    // Update CLS token
    this.adamUpdateArray(
      this.clsToken!,
      grads.clsTokenGrad,
      this.clsTokenM!,
      this.clsTokenV!,
      lr,
      beta1,
      beta2,
      bc1,
      bc2,
      eps,
    );

    // Update transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      const blockGrads = grads.blockGrads[b];

      this.adamUpdateArray(
        block.ln1Gamma,
        blockGrads.ln1GammaGrad,
        block.ln1GammaM,
        block.ln1GammaV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.ln1Beta,
        blockGrads.ln1BetaGrad,
        block.ln1BetaM,
        block.ln1BetaV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.wq,
        blockGrads.wqGrad,
        block.wqM,
        block.wqV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.wk,
        blockGrads.wkGrad,
        block.wkM,
        block.wkV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.wv,
        blockGrads.wvGrad,
        block.wvM,
        block.wvV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.wo,
        blockGrads.woGrad,
        block.woM,
        block.woV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.bo,
        blockGrads.boGrad,
        block.boM,
        block.boV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.ln2Gamma,
        blockGrads.ln2GammaGrad,
        block.ln2GammaM,
        block.ln2GammaV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.ln2Beta,
        blockGrads.ln2BetaGrad,
        block.ln2BetaM,
        block.ln2BetaV,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.w1,
        blockGrads.w1Grad,
        block.w1M,
        block.w1V,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.b1,
        blockGrads.b1Grad,
        block.b1M,
        block.b1V,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.w2,
        blockGrads.w2Grad,
        block.w2M,
        block.w2V,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
      this.adamUpdateArray(
        block.b2,
        blockGrads.b2Grad,
        block.b2M,
        block.b2V,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
      );
    }

    // Update output layer
    this.adamUpdateArray(
      this.wout!,
      grads.woutGrad,
      this.woutM!,
      this.woutV!,
      lr,
      beta1,
      beta2,
      bc1,
      bc2,
      eps,
    );
    this.adamUpdateArray(
      this.bout!,
      grads.boutGrad,
      this.boutM!,
      this.boutV!,
      lr,
      beta1,
      beta2,
      bc1,
      bc2,
      eps,
    );
  }

  /**
   * Adam update for a single array
   * @private
   */
  private adamUpdateArray(
    weights: Float64Array,
    grads: Float64Array,
    m: Float64Array,
    v: Float64Array,
    lr: number,
    beta1: number,
    beta2: number,
    bc1: number,
    bc2: number,
    eps: number,
  ): void {
    for (let i = 0; i < weights.length; i++) {
      const g = grads[i];
      // Update biased first moment estimate
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      // Update biased second moment estimate
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      // Compute bias-corrected estimates
      const mHat = m[i] / bc1;
      const vHat = v[i] / bc2;
      // Update weights
      weights[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  /**
   * Generate predictions for future steps
   *
   * Uses the last seen input to make predictions. For multi-step prediction,
   * predictions are made autoregressively (each prediction becomes input for next).
   *
   * @param futureSteps - Number of prediction steps to generate
   * @returns PredictionResult containing predictions with uncertainty bounds
   *
   * @example
   * ```typescript
   * const result = model.predict(3);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}`);
   *   console.log(`95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.isInitialized) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const accuracy = this.getAccuracy();

    // Use last input as starting point
    const currentInput = this.pool.acquire(this.inputDim);
    for (let i = 0; i < this.inputDim; i++) {
      currentInput[i] = this.lastInput![i];
    }

    // Get normalization stats
    const inputMean = this.inputStats!.getMean();
    const inputStd = this.inputStats!.getStd(this.config.epsilon);
    const outputMean = this.outputStats!.getMean();
    const outputStd = this.outputStats!.getStd(this.config.epsilon);

    for (let step = 0; step < futureSteps; step++) {
      // Normalize input
      const normalizedInput = this.pool.acquire(this.inputDim);
      for (let i = 0; i < this.inputDim; i++) {
        normalizedInput[i] = (currentInput[i] - inputMean[i]) / inputStd[i];
      }

      // Forward pass
      this.forward(normalizedInput);

      // Denormalize output
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let i = 0; i < this.outputDim; i++) {
        const normalizedPred = this.forwardCache!.output[i];
        const pred = normalizedPred * outputStd[i] + outputMean[i];
        const se = Math.sqrt(this.predictionVariance![i]) * outputStd[i];

        predicted.push(pred);
        standardError.push(se);
        lowerBound.push(pred - 2 * se); // ~95% confidence interval
        upperBound.push(pred + 2 * se);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });

      // For autoregressive prediction, use output as next input if dimensions match
      if (this.outputDim <= this.inputDim) {
        // Shift input left and append prediction
        for (let i = 0; i < this.inputDim - this.outputDim; i++) {
          currentInput[i] = currentInput[i + this.outputDim];
        }
        for (let i = 0; i < this.outputDim; i++) {
          currentInput[this.inputDim - this.outputDim + i] = predicted[i];
        }
      }

      this.pool.release(normalizedInput);
    }

    this.pool.release(currentInput);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get current model accuracy: 1/(1 + averageLoss)
   * @private
   */
  private getAccuracy(): number {
    if (this.runningLossCount === 0) return 0;
    const avgLoss = this.runningLossSum / this.runningLossCount;
    return 1 / (1 + avgLoss);
  }

  /**
   * Get summary of model configuration and current state
   *
   * @returns ModelSummary with configuration and statistics
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
      // Feature embeddings
      totalParameters += this.inputDim * this.config.embeddingDim * 2; // weights + biases
      // CLS token
      totalParameters += this.config.embeddingDim;
      // Transformer blocks
      for (let b = 0; b < this.config.numBlocks; b++) {
        totalParameters += this.blocks[b].countParameters();
      }
      // Output layer
      totalParameters += this.config.embeddingDim * this.outputDim +
        this.outputDim;
    }

    // Compute effective learning rate
    let effectiveLR = this.config.learningRate;
    if (this.updateCount > 0) {
      if (this.updateCount < this.config.warmupSteps) {
        effectiveLR = this.config.learningRate * this.updateCount /
          this.config.warmupSteps;
      } else {
        const decayProgress = (this.updateCount - this.config.warmupSteps) /
          Math.max(1, this.config.totalSteps - this.config.warmupSteps);
        effectiveLR = this.config.learningRate * 0.5 *
          (1 + Math.cos(Math.PI * Math.min(1, decayProgress)));
      }
    }

    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      totalParameters,
      sampleCount: this.sampleCount,
      accuracy: this.getAccuracy(),
      converged: this.converged,
      effectiveLearningRate: effectiveLR,
      driftCount: this.driftCount,
    };
  }

  /**
   * Get detailed weight information for inspection or external use
   *
   * @returns WeightInfo containing all model weights and optimizer state
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`CLS token: ${weights.clsToken}`);
   * ```
   */
  getWeights(): WeightInfo {
    if (!this.isInitialized) {
      return {
        featureEmbeddings: [],
        clsToken: [],
        attentionWeights: [],
        ffnWeights: [],
        layerNormParams: [],
        outputWeights: [],
        firstMoment: [],
        secondMoment: [],
        updateCount: 0,
      };
    }

    // Feature embeddings [inputDim][2][embeddingDim]
    const featureEmbeddings: number[][][] = [];
    for (let i = 0; i < this.inputDim; i++) {
      featureEmbeddings.push([
        Array.from(this.featureWeights[i]),
        Array.from(this.featureBiases[i]),
      ]);
    }

    // Attention weights [numBlocks][4][...]
    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      attentionWeights.push([
        Array.from(block.wq),
        Array.from(block.wk),
        Array.from(block.wv),
        Array.from(block.wo),
      ]);
    }

    // FFN weights [numBlocks][4][...]
    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      ffnWeights.push([
        Array.from(block.w1),
        Array.from(block.b1),
        Array.from(block.w2),
        Array.from(block.b2),
      ]);
    }

    // Layer norm params [numBlocks][4][...]
    const layerNormParams: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      layerNormParams.push([
        Array.from(block.ln1Gamma),
        Array.from(block.ln1Beta),
        Array.from(block.ln2Gamma),
        Array.from(block.ln2Beta),
      ]);
    }

    // First moments for Adam
    const firstMoment: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      firstMoment.push([
        Array.from(block.wqM),
        Array.from(block.wkM),
        Array.from(block.wvM),
        Array.from(block.woM),
      ]);
    }

    // Second moments for Adam
    const secondMoment: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      secondMoment.push([
        Array.from(block.wqV),
        Array.from(block.wkV),
        Array.from(block.wvV),
        Array.from(block.woV),
      ]);
    }

    return {
      featureEmbeddings,
      clsToken: Array.from(this.clsToken!),
      attentionWeights,
      ffnWeights,
      layerNormParams,
      outputWeights: [
        Array.from(this.wout!),
        Array.from(this.bout!),
      ],
      firstMoment,
      secondMoment,
      updateCount: this.updateCount,
    };
  }

  /**
   * Get normalization statistics for inputs and outputs
   *
   * @returns NormalizationStats with mean, std, and count
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input mean: ${stats.inputMean}`);
   * console.log(`Samples: ${stats.count}`);
   * ```
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
      inputMean: Array.from(this.inputStats!.getMean()),
      inputStd: Array.from(this.inputStats!.getStd(this.config.epsilon)),
      outputMean: Array.from(this.outputStats!.getMean()),
      outputStd: Array.from(this.outputStats!.getStd(this.config.epsilon)),
      count: this.inputStats!.getCount(),
    };
  }

  /**
   * Reset model to initial state
   *
   * Clears all weights, statistics, and optimizer state.
   * Model will need to be re-initialized with new data.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now empty and ready for fresh training
   * ```
   */
  reset(): void {
    this.isInitialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.seqLen = 0;
    this.headDim = 0;
    this.ffnDim = 0;
    this.sampleCount = 0;
    this.updateCount = 0;
    this.runningLossSum = 0;
    this.runningLossCount = 0;
    this.converged = false;
    this.driftCount = 0;

    this.featureWeights = [];
    this.featureBiases = [];
    this.clsToken = null;
    this.blocks = [];
    this.wout = null;
    this.bout = null;

    this.featureWeightsM = [];
    this.featureWeightsV = [];
    this.featureBiasesM = [];
    this.featureBiasesV = [];
    this.clsTokenM = null;
    this.clsTokenV = null;
    this.woutM = null;
    this.woutV = null;
    this.boutM = null;
    this.boutV = null;

    this.inputStats = null;
    this.outputStats = null;
    this.predictionVariance = null;
    this.lastInput = null;

    this.forwardCache = null;
    this.gradients = null;
    this.tempInput = null;
    this.tempOutput = null;
    this.tempSeqGrad = null;

    this.adwin.reset();
    this.pool.clear();
  }

  /**
   * Serialize model state to JSON string
   *
   * Includes all weights, optimizer state, normalization statistics,
   * and configuration for complete model restoration.
   *
   * @returns JSON string containing complete model state
   *
   * @example
   * ```typescript
   * const state = model.save();
   * localStorage.setItem('model', state);
   * ```
   */
  save(): string {
    if (!this.isInitialized) {
      return JSON.stringify({ initialized: false, config: this.config });
    }

    const state: SerializedState = {
      config: this.config,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      sampleCount: this.sampleCount,
      updateCount: this.updateCount,
      runningLossSum: this.runningLossSum,
      runningLossCount: this.runningLossCount,
      converged: this.converged,
      driftCount: this.driftCount,
      lastInput: this.lastInput ? Array.from(this.lastInput) : null,
      weights: {
        featureWeights: this.featureWeights.map((w) => Array.from(w)),
        featureBiases: this.featureBiases.map((b) => Array.from(b)),
        clsToken: Array.from(this.clsToken!),
        blocks: this.blocks.map((block) => ({
          ln1Gamma: Array.from(block.ln1Gamma),
          ln1Beta: Array.from(block.ln1Beta),
          wq: Array.from(block.wq),
          wk: Array.from(block.wk),
          wv: Array.from(block.wv),
          wo: Array.from(block.wo),
          bo: Array.from(block.bo),
          ln2Gamma: Array.from(block.ln2Gamma),
          ln2Beta: Array.from(block.ln2Beta),
          w1: Array.from(block.w1),
          b1: Array.from(block.b1),
          w2: Array.from(block.w2),
          b2: Array.from(block.b2),
        })),
        wout: Array.from(this.wout!),
        bout: Array.from(this.bout!),
      },
      adam: {
        featureWeightsM: this.featureWeightsM.map((m) => Array.from(m)),
        featureWeightsV: this.featureWeightsV.map((v) => Array.from(v)),
        featureBiasesM: this.featureBiasesM.map((m) => Array.from(m)),
        featureBiasesV: this.featureBiasesV.map((v) => Array.from(v)),
        clsTokenM: Array.from(this.clsTokenM!),
        clsTokenV: Array.from(this.clsTokenV!),
        blocks: this.blocks.map((block) => ({
          ln1GammaM: Array.from(block.ln1GammaM),
          ln1GammaV: Array.from(block.ln1GammaV),
          ln1BetaM: Array.from(block.ln1BetaM),
          ln1BetaV: Array.from(block.ln1BetaV),
          wqM: Array.from(block.wqM),
          wqV: Array.from(block.wqV),
          wkM: Array.from(block.wkM),
          wkV: Array.from(block.wkV),
          wvM: Array.from(block.wvM),
          wvV: Array.from(block.wvV),
          woM: Array.from(block.woM),
          woV: Array.from(block.woV),
          boM: Array.from(block.boM),
          boV: Array.from(block.boV),
          ln2GammaM: Array.from(block.ln2GammaM),
          ln2GammaV: Array.from(block.ln2GammaV),
          ln2BetaM: Array.from(block.ln2BetaM),
          ln2BetaV: Array.from(block.ln2BetaV),
          w1M: Array.from(block.w1M),
          w1V: Array.from(block.w1V),
          b1M: Array.from(block.b1M),
          b1V: Array.from(block.b1V),
          w2M: Array.from(block.w2M),
          w2V: Array.from(block.w2V),
          b2M: Array.from(block.b2M),
          b2V: Array.from(block.b2V),
        })),
        woutM: Array.from(this.woutM!),
        woutV: Array.from(this.woutV!),
        boutM: Array.from(this.boutM!),
        boutV: Array.from(this.boutV!),
      },
      normalization: {
        inputMean: Array.from(this.inputStats!.getMean()),
        inputM2: Array.from(this.inputStats!.getM2()),
        outputMean: Array.from(this.outputStats!.getMean()),
        outputM2: Array.from(this.outputStats!.getM2()),
        count: this.inputStats!.getCount(),
      },
      adwin: this.adwin.getState(),
      predictionVariance: Array.from(this.predictionVariance!),
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   *
   * Restores complete model including weights, optimizer state,
   * and normalization statistics.
   *
   * @param jsonString - JSON string from save()
   * @throws Error if JSON is invalid or incompatible
   *
   * @example
   * ```typescript
   * const state = localStorage.getItem('model');
   * if (state) model.load(state);
   * ```
   */
  load(jsonString: string): void {
    const parsed = JSON.parse(jsonString);

    if (!parsed.initialized && parsed.initialized !== undefined) {
      this.reset();
      return;
    }

    const state = parsed as SerializedState;

    // Initialize with stored dimensions
    this.initialize(state.inputDim, state.outputDim);

    // Restore state
    this.sampleCount = state.sampleCount;
    this.updateCount = state.updateCount;
    this.runningLossSum = state.runningLossSum;
    this.runningLossCount = state.runningLossCount;
    this.converged = state.converged;
    this.driftCount = state.driftCount;

    if (state.lastInput) {
      for (let i = 0; i < state.lastInput.length; i++) {
        this.lastInput![i] = state.lastInput[i];
      }
    }

    // Restore weights
    for (let i = 0; i < this.inputDim; i++) {
      this.copyArray(state.weights.featureWeights[i], this.featureWeights[i]);
      this.copyArray(state.weights.featureBiases[i], this.featureBiases[i]);
    }
    this.copyArray(state.weights.clsToken, this.clsToken!);

    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      const savedBlock = state.weights.blocks[b];
      this.copyArray(savedBlock.ln1Gamma, block.ln1Gamma);
      this.copyArray(savedBlock.ln1Beta, block.ln1Beta);
      this.copyArray(savedBlock.wq, block.wq);
      this.copyArray(savedBlock.wk, block.wk);
      this.copyArray(savedBlock.wv, block.wv);
      this.copyArray(savedBlock.wo, block.wo);
      this.copyArray(savedBlock.bo, block.bo);
      this.copyArray(savedBlock.ln2Gamma, block.ln2Gamma);
      this.copyArray(savedBlock.ln2Beta, block.ln2Beta);
      this.copyArray(savedBlock.w1, block.w1);
      this.copyArray(savedBlock.b1, block.b1);
      this.copyArray(savedBlock.w2, block.w2);
      this.copyArray(savedBlock.b2, block.b2);
    }

    this.copyArray(state.weights.wout, this.wout!);
    this.copyArray(state.weights.bout, this.bout!);

    // Restore Adam state
    for (let i = 0; i < this.inputDim; i++) {
      this.copyArray(state.adam.featureWeightsM[i], this.featureWeightsM[i]);
      this.copyArray(state.adam.featureWeightsV[i], this.featureWeightsV[i]);
      this.copyArray(state.adam.featureBiasesM[i], this.featureBiasesM[i]);
      this.copyArray(state.adam.featureBiasesV[i], this.featureBiasesV[i]);
    }
    this.copyArray(state.adam.clsTokenM, this.clsTokenM!);
    this.copyArray(state.adam.clsTokenV, this.clsTokenV!);

    for (let b = 0; b < this.config.numBlocks; b++) {
      const block = this.blocks[b];
      const savedAdam = state.adam.blocks[b];
      this.copyArray(savedAdam.ln1GammaM, block.ln1GammaM);
      this.copyArray(savedAdam.ln1GammaV, block.ln1GammaV);
      this.copyArray(savedAdam.ln1BetaM, block.ln1BetaM);
      this.copyArray(savedAdam.ln1BetaV, block.ln1BetaV);
      this.copyArray(savedAdam.wqM, block.wqM);
      this.copyArray(savedAdam.wqV, block.wqV);
      this.copyArray(savedAdam.wkM, block.wkM);
      this.copyArray(savedAdam.wkV, block.wkV);
      this.copyArray(savedAdam.wvM, block.wvM);
      this.copyArray(savedAdam.wvV, block.wvV);
      this.copyArray(savedAdam.woM, block.woM);
      this.copyArray(savedAdam.woV, block.woV);
      this.copyArray(savedAdam.boM, block.boM);
      this.copyArray(savedAdam.boV, block.boV);
      this.copyArray(savedAdam.ln2GammaM, block.ln2GammaM);
      this.copyArray(savedAdam.ln2GammaV, block.ln2GammaV);
      this.copyArray(savedAdam.ln2BetaM, block.ln2BetaM);
      this.copyArray(savedAdam.ln2BetaV, block.ln2BetaV);
      this.copyArray(savedAdam.w1M, block.w1M);
      this.copyArray(savedAdam.w1V, block.w1V);
      this.copyArray(savedAdam.b1M, block.b1M);
      this.copyArray(savedAdam.b1V, block.b1V);
      this.copyArray(savedAdam.w2M, block.w2M);
      this.copyArray(savedAdam.w2V, block.w2V);
      this.copyArray(savedAdam.b2M, block.b2M);
      this.copyArray(savedAdam.b2V, block.b2V);
    }

    this.copyArray(state.adam.woutM, this.woutM!);
    this.copyArray(state.adam.woutV, this.woutV!);
    this.copyArray(state.adam.boutM, this.boutM!);
    this.copyArray(state.adam.boutV, this.boutV!);

    // Restore normalization statistics
    this.inputStats!.loadState(
      state.normalization.inputMean,
      state.normalization.inputM2,
      state.normalization.count,
    );
    this.outputStats!.loadState(
      state.normalization.outputMean,
      state.normalization.outputM2,
      state.normalization.count,
    );

    // Restore ADWIN
    this.adwin.loadState(state.adwin.window, state.adwin.windowSum);

    // Restore prediction variance
    this.copyArray(state.predictionVariance, this.predictionVariance!);
  }

  /**
   * Copy array values
   * @private
   */
  private copyArray(source: number[], target: Float64Array): void {
    for (let i = 0; i < source.length && i < target.length; i++) {
      target[i] = source[i];
    }
  }
}
