/**
 * FusionTemporalTransformerRegression
 *
 * A Fusion Temporal Transformer neural network for multivariate time series regression
 * with incremental online learning capabilities.
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

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
  gradientClipNorm?: number;
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

export interface WeightInfo {
  temporalConvWeights: number[][];
  temporalConvBiases: number[][];
  scaleEmbeddings: number[][];
  positionalEncoding: number[][];
  fusionGateWeights: number[];
  fusionGateBiases: number[];
  attentionQWeights: number[][][];
  attentionKWeights: number[][][];
  attentionVWeights: number[][][];
  attentionOutWeights: number[][];
  attentionOutBiases: number[][];
  ffnW1: number[][];
  ffnB1: number[][];
  ffnW2: number[][];
  ffnB2: number[][];
  ln1Gamma: number[][];
  ln1Beta: number[][];
  ln2Gamma: number[][];
  ln2Beta: number[][];
  poolWeights: number[];
  outputWeights: number[];
  outputBiases: number[];
  firstMoment: number[][][];
  secondMoment: number[][][];
  updateCount: number;
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

// ============================================================================
// WELFORD ONLINE STATISTICS
// ============================================================================

/**
 * Welford's online algorithm for computing running mean and variance
 * Memory efficient - O(n) space for n dimensions
 */
class WelfordAccumulator {
  private readonly dim: number;
  public mean: Float64Array;
  public m2: Float64Array;
  public count: number;

  constructor(dim: number) {
    this.dim = dim;
    this.mean = new Float64Array(dim);
    this.m2 = new Float64Array(dim);
    this.count = 0;
  }

  /**
   * Update statistics with new sample using Welford's algorithm
   * mean_new = mean_old + (x - mean_old) / n
   * m2_new = m2_old + (x - mean_old) * (x - mean_new)
   * @param x - New data point
   */
  update(x: Float64Array): void {
    this.count++;
    for (let i = 0; i < this.dim; i++) {
      const delta = x[i] - this.mean[i];
      this.mean[i] += delta / this.count;
      const delta2 = x[i] - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  /**
   * Get standard deviation with numerical stability
   * std = sqrt(variance + epsilon)
   * @param epsilon - Small constant for numerical stability
   * @returns Standard deviation array
   */
  getStd(epsilon: number): Float64Array {
    const std = new Float64Array(this.dim);
    if (this.count > 1) {
      for (let i = 0; i < this.dim; i++) {
        const variance = this.m2[i] / (this.count - 1);
        std[i] = Math.sqrt(variance + epsilon);
      }
    } else {
      std.fill(1);
    }
    return std;
  }

  /**
   * Get standard deviation into preallocated buffer
   * @param epsilon - Small constant for numerical stability
   * @param out - Output buffer
   */
  getStdInto(epsilon: number, out: Float64Array): void {
    if (this.count > 1) {
      for (let i = 0; i < this.dim; i++) {
        const variance = this.m2[i] / (this.count - 1);
        out[i] = Math.sqrt(variance + epsilon);
      }
    } else {
      out.fill(1);
    }
  }

  reset(): void {
    this.mean.fill(0);
    this.m2.fill(0);
    this.count = 0;
  }

  serialize(): { mean: number[]; m2: number[]; count: number } {
    return {
      mean: Array.from(this.mean),
      m2: Array.from(this.m2),
      count: this.count,
    };
  }

  deserialize(data: { mean: number[]; m2: number[]; count: number }): void {
    const len = Math.min(data.mean.length, this.dim);
    for (let i = 0; i < len; i++) {
      this.mean[i] = data.mean[i];
      this.m2[i] = data.m2[i];
    }
    this.count = data.count;
  }
}

// ============================================================================
// ADWIN DRIFT DETECTOR
// ============================================================================

/**
 * ADWIN (ADaptive WINdowing) algorithm for concept drift detection
 * Detects distributional changes in streaming data
 */
class ADWINDetector {
  private readonly capacity: number;
  private readonly delta: number;
  private values: Float64Array;
  private size: number;
  private sum: number;

  constructor(capacity: number, delta: number) {
    this.capacity = capacity;
    this.delta = delta;
    this.values = new Float64Array(capacity);
    this.size = 0;
    this.sum = 0;
  }

  /**
   * Add error value and check for drift
   * Uses statistical test: |μ_left - μ_right| >= ε_cut
   * where ε_cut = sqrt((1/n0 + 1/n1) * ln(4/δ) / 2)
   * @param error - Error value to add
   * @returns true if drift detected
   */
  addAndCheck(error: number): boolean {
    // Add to circular buffer
    if (this.size < this.capacity) {
      this.values[this.size] = error;
      this.sum += error;
      this.size++;
    } else {
      this.sum -= this.values[0];
      for (let i = 0; i < this.size - 1; i++) {
        this.values[i] = this.values[i + 1];
      }
      this.values[this.size - 1] = error;
      this.sum += error;
    }

    if (this.size < 10) return false;

    // Check for drift at each split point
    let leftSum = 0;
    const minWindow = 5;
    for (let cut = minWindow; cut < this.size - minWindow; cut++) {
      leftSum += this.values[cut - 1];
      const rightSum = this.sum - leftSum;
      const n0 = cut;
      const n1 = this.size - cut;
      const leftMean = leftSum / n0;
      const rightMean = rightSum / n1;
      const bound = Math.sqrt(
        ((1 / n0) + (1 / n1)) * Math.log(4 / this.delta) / 2,
      );

      if (Math.abs(leftMean - rightMean) >= bound) {
        // Drift detected - remove older portion
        for (let i = 0; i < n1; i++) {
          this.values[i] = this.values[cut + i];
        }
        this.size = n1;
        this.sum = rightSum;
        return true;
      }
    }
    return false;
  }

  reset(): void {
    this.size = 0;
    this.sum = 0;
    this.values.fill(0);
  }

  serialize(): { values: number[]; size: number; sum: number } {
    return {
      values: Array.from(this.values.subarray(0, this.size)),
      size: this.size,
      sum: this.sum,
    };
  }

  deserialize(data: { values: number[]; size: number; sum: number }): void {
    this.values.fill(0);
    const len = Math.min(data.values.length, this.capacity);
    for (let i = 0; i < len; i++) {
      this.values[i] = data.values[i];
    }
    this.size = Math.min(data.size, this.capacity);
    this.sum = data.sum;
  }
}

// ============================================================================
// PARAMETER CLASS WITH ADAM OPTIMIZER STATE
// ============================================================================

/**
 * Parameter container with Adam optimizer momentum terms
 * Encapsulates weights, gradients, and optimizer state
 */
class Param {
  public readonly size: number;
  public data: Float64Array;
  public grad: Float64Array;
  public m: Float64Array; // First moment (mean of gradients)
  public v: Float64Array; // Second moment (mean of squared gradients)

  constructor(size: number) {
    this.size = size;
    this.data = new Float64Array(size);
    this.grad = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  /**
   * Xavier/Glorot initialization
   * W ~ N(0, sqrt(2 / (fan_in + fan_out)))
   * @param fanIn - Number of input units
   * @param fanOut - Number of output units
   */
  initXavier(fanIn: number, fanOut: number): void {
    const std = Math.sqrt(2.0 / (fanIn + fanOut));
    for (let i = 0; i < this.size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      this.data[i] = std * Math.sqrt(-2 * Math.log(u1)) *
        Math.cos(2 * Math.PI * u2);
    }
  }

  initZero(): void {
    this.data.fill(0);
  }

  initOne(): void {
    this.data.fill(1);
  }

  initSmall(scale: number): void {
    for (let i = 0; i < this.size; i++) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      this.data[i] = scale * Math.sqrt(-2 * Math.log(u1)) *
        Math.cos(2 * Math.PI * u2);
    }
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /**
   * Adam optimizer step with L2 regularization
   * m_t = β1 * m_{t-1} + (1 - β1) * g_t
   * v_t = β2 * v_{t-1} + (1 - β2) * g_t²
   * m̂_t = m_t / (1 - β1^t)
   * v̂_t = v_t / (1 - β2^t)
   * θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
   */
  adamStep(
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    t: number,
    lambda: number,
  ): void {
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    for (let i = 0; i < this.size; i++) {
      // Add L2 regularization to gradient
      const g = this.grad[i] + lambda * this.data[i];

      // Update biased first moment estimate
      this.m[i] = beta1 * this.m[i] + (1 - beta1) * g;

      // Update biased second raw moment estimate
      this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g;

      // Compute bias-corrected estimates
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;

      // Update parameters
      this.data[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  fromArray(arr: number[]): void {
    const len = Math.min(arr.length, this.size);
    for (let i = 0; i < len; i++) {
      this.data[i] = arr[i];
    }
  }
}

// ============================================================================
// MATH CONSTANTS AND UTILITY FUNCTIONS
// ============================================================================

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_COEF = 0.044715;

/**
 * GELU activation function
 * gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * @param x - Input value
 * @returns Activated value
 */
function gelu(x: number): number {
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

/**
 * GELU derivative
 * @param x - Input value (pre-activation)
 * @returns Gradient value
 */
function geluGrad(x: number): number {
  const x3 = x * x * x;
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
  const tanhInner = Math.tanh(inner);
  const sech2 = 1 - tanhInner * tanhInner;
  const innerGrad = SQRT_2_OVER_PI * (1 + 3 * GELU_COEF * x * x);
  return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerGrad;
}

/**
 * Numerically stable sigmoid
 * σ(x) = 1 / (1 + e^(-x)) for x >= 0
 * σ(x) = e^x / (1 + e^x) for x < 0
 * @param x - Input value
 * @returns Sigmoid output in (0, 1)
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Sigmoid derivative given sigmoid output
 * σ'(x) = σ(x) * (1 - σ(x))
 * @param sig - Sigmoid output value
 * @returns Gradient value
 */
function sigmoidGrad(sig: number): number {
  return sig * (1 - sig);
}

/**
 * Numerically stable softmax (in-place)
 * softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 * @param scores - Input logits
 * @param out - Output probabilities
 */
function softmax(scores: Float64Array, out: Float64Array): void {
  const len = scores.length;
  if (len === 0) return;

  let max = scores[0];
  for (let i = 1; i < len; i++) {
    if (scores[i] > max) max = scores[i];
  }

  let sum = 0;
  for (let i = 0; i < len; i++) {
    const exp = Math.exp(scores[i] - max);
    out[i] = exp;
    sum += exp;
  }

  const invSum = 1 / (sum + 1e-10);
  for (let i = 0; i < len; i++) {
    out[i] *= invSum;
  }
}

/**
 * Clip value to range
 * @param x - Input value
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clipped value
 */
function clip(x: number, min: number, max: number): number {
  return x < min ? min : (x > max ? max : x);
}

// ============================================================================
// BUFFER POOL FOR MEMORY REUSE
// ============================================================================

/**
 * Object pool for Float64Array buffers to minimize GC pressure
 */
class BufferPool {
  private pools: Map<number, Float64Array[]> = new Map();

  /**
   * Get a buffer of specified size
   * Reuses existing buffer if available, creates new one otherwise
   * @param size - Required buffer size
   * @returns Float64Array buffer
   */
  get(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      const buf = pool.pop()!;
      buf.fill(0);
      return buf;
    }
    return new Float64Array(size);
  }

  /**
   * Return buffer to pool for reuse
   * @param buf - Buffer to return
   */
  release(buf: Float64Array): void {
    const size = buf.length;
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < 100) { // Limit pool size
      pool.push(buf);
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
// MAIN CLASS
// ============================================================================

/**
 * FusionTemporalTransformerRegression
 *
 * A Fusion Temporal Transformer neural network for multivariate time series
 * regression with incremental online learning capabilities.
 *
 * Architecture:
 * 1. Multi-scale temporal convolutions extract features at different time scales
 * 2. Gated fusion combines scale-specific representations
 * 3. Transformer blocks with multi-head self-attention model temporal dependencies
 * 4. Attention pooling aggregates sequence for final prediction
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({
 *   numBlocks: 2,
 *   embeddingDim: 32,
 *   numHeads: 4,
 *   learningRate: 0.001
 * });
 *
 * // Online training
 * for (const batch of data) {
 *   const result = model.fitOnline({
 *     xCoordinates: batch.inputs,
 *     yCoordinates: batch.targets
 *   });
 *   console.log(`Loss: ${result.loss}`);
 * }
 *
 * // Prediction
 * const predictions = model.predict(5);
 * ```
 */
export class FusionTemporalTransformerRegression {
  // Configuration (readonly after construction)
  private readonly numBlocks: number;
  private readonly embeddingDim: number;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly ffnHiddenDim: number;
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
  private readonly temporalKernelSize: number;
  private readonly maxSequenceLength: number;
  private readonly gradientClipNorm: number;

  // Model state
  private initialized: boolean = false;
  private inputDim: number = 0;
  private outputDim: number = 0;
  private sampleCount: number = 0;
  private updateCount: number = 0;
  private totalLoss: number = 0;
  private prevLoss: number = Infinity;
  private converged: boolean = false;
  private driftCount: number = 0;
  private recentLosses: Float64Array;
  private recentLossIdx: number = 0;
  private recentLossCount: number = 0;

  // Normalization statistics
  private inputWelford: WelfordAccumulator | null = null;
  private outputWelford: WelfordAccumulator | null = null;
  private adwin: ADWINDetector | null = null;

  // Model parameters (organized by layer)
  private params: Param[] = [];

  // Temporal convolution parameters
  private temporalConvW: Param[] = [];
  private temporalConvB: Param[] = [];
  private scaleEmbed: Param[] = [];

  // Positional encoding (fixed, not learnable)
  private posEncoding: Float64Array[] = [];

  // Fusion gate parameters
  private fusionGateW: Param | null = null;
  private fusionGateB: Param | null = null;

  // Transformer block parameters
  private ln1Gamma: Param[] = [];
  private ln1Beta: Param[] = [];
  private ln2Gamma: Param[] = [];
  private ln2Beta: Param[] = [];
  private attnWq: Param[][] = [];
  private attnWk: Param[][] = [];
  private attnWv: Param[][] = [];
  private attnWo: Param[] = [];
  private attnBo: Param[] = [];
  private ffnW1: Param[] = [];
  private ffnB1: Param[] = [];
  private ffnW2: Param[] = [];
  private ffnB2: Param[] = [];

  // Output layer parameters
  private poolW: Param | null = null;
  private outW: Param | null = null;
  private outB: Param | null = null;

  // Preallocated buffers for forward/backward passes
  private bufferPool: BufferPool = new BufferPool();
  private inputStdBuf: Float64Array | null = null;
  private outputStdBuf: Float64Array | null = null;

  // Cache for backpropagation (cleared each forward pass)
  private fwdCache: {
    normalizedInput: Float64Array[];
    convPre: Float64Array[][];
    convOut: Float64Array[][];
    scaleOutputs: Float64Array[][];
    fusionConcats: Float64Array[];
    fusionGatesPre: Float64Array[];
    fusionGates: Float64Array[];
    fused: Float64Array[];
    blockInputs: Float64Array[][];
    ln1Inputs: Float64Array[][];
    ln1Outputs: Float64Array[][];
    ln1Mean: Float64Array[];
    ln1InvStd: Float64Array[];
    attnQ: Float64Array[][][];
    attnK: Float64Array[][][];
    attnV: Float64Array[][][];
    attnScores: Float64Array[][][];
    attnWeights: Float64Array[][][];
    attnHeadOut: Float64Array[][][];
    attnConcat: Float64Array[][];
    attnOut: Float64Array[][];
    residual1: Float64Array[][];
    ln2Inputs: Float64Array[][];
    ln2Outputs: Float64Array[][];
    ln2Mean: Float64Array[];
    ln2InvStd: Float64Array[];
    ffnPre: Float64Array[][];
    ffnHidden: Float64Array[][];
    ffnOut: Float64Array[][];
    finalHidden: Float64Array[];
    poolScores: Float64Array;
    poolWeights: Float64Array;
    aggregated: Float64Array;
    output: Float64Array;
  } | null = null;

  // Stored input for prediction
  private lastRawInput: Float64Array[] | null = null;

  /**
   * Create a new FusionTemporalTransformerRegression model
   * @param config - Configuration options
   */
  constructor(config: FusionTemporalTransformerConfig = {}) {
    this.numBlocks = config.numBlocks ?? 3;
    this.embeddingDim = config.embeddingDim ?? 64;
    this.numHeads = config.numHeads ?? 8;
    this.headDim = Math.floor(this.embeddingDim / this.numHeads);
    this.ffnHiddenDim = this.embeddingDim * (config.ffnMultiplier ?? 4);
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
    this.temporalKernelSize = config.temporalKernelSize ?? 3;
    this.maxSequenceLength = config.maxSequenceLength ?? 512;
    this.gradientClipNorm = config.gradientClipNorm ?? 1.0;

    // Preallocate recent losses buffer
    this.recentLosses = new Float64Array(100);
  }

  /**
   * Initialize model parameters and buffers
   * Called automatically on first fitOnline call
   * @param inputDim - Input feature dimension
   * @param outputDim - Output dimension
   */
  private init(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.params = [];

    // Initialize normalization accumulators
    this.inputWelford = new WelfordAccumulator(inputDim);
    this.outputWelford = new WelfordAccumulator(outputDim);
    this.adwin = new ADWINDetector(1000, this.adwinDelta);

    // Preallocate std buffers
    this.inputStdBuf = new Float64Array(inputDim);
    this.outputStdBuf = new Float64Array(outputDim);

    // Generate sinusoidal positional encodings
    // PE(pos, 2i) = sin(pos / 10000^(2i/d))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    this.posEncoding = [];
    for (let pos = 0; pos < this.maxSequenceLength; pos++) {
      const pe = new Float64Array(this.embeddingDim);
      for (let i = 0; i < this.embeddingDim / 2; i++) {
        const angle = pos / Math.pow(10000, (2 * i) / this.embeddingDim);
        pe[2 * i] = Math.sin(angle);
        pe[2 * i + 1] = Math.cos(angle);
      }
      this.posEncoding.push(pe);
    }

    // Initialize temporal convolution parameters for each scale
    this.temporalConvW = [];
    this.temporalConvB = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const wSize = this.embeddingDim * inputDim * this.temporalKernelSize;
      const w = new Param(wSize);
      w.initXavier(inputDim * this.temporalKernelSize, this.embeddingDim);
      this.temporalConvW.push(w);
      this.params.push(w);

      const b = new Param(this.embeddingDim);
      b.initZero();
      this.temporalConvB.push(b);
      this.params.push(b);
    }

    // Initialize scale embeddings (learned per-scale bias)
    this.scaleEmbed = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const se = new Param(this.embeddingDim);
      se.initSmall(0.02);
      this.scaleEmbed.push(se);
      this.params.push(se);
    }

    // Initialize fusion gate parameters
    const fusionInDim = this.embeddingDim * this.temporalScales.length;
    this.fusionGateW = new Param(fusionInDim * this.temporalScales.length);
    this.fusionGateW.initXavier(fusionInDim, this.temporalScales.length);
    this.params.push(this.fusionGateW);

    this.fusionGateB = new Param(this.temporalScales.length);
    this.fusionGateB.initZero();
    this.params.push(this.fusionGateB);

    // Initialize transformer block parameters
    this.ln1Gamma = [];
    this.ln1Beta = [];
    this.ln2Gamma = [];
    this.ln2Beta = [];
    this.attnWq = [];
    this.attnWk = [];
    this.attnWv = [];
    this.attnWo = [];
    this.attnBo = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];

    for (let b = 0; b < this.numBlocks; b++) {
      // Layer norm 1 parameters (pre-attention)
      const g1 = new Param(this.embeddingDim);
      g1.initOne();
      this.ln1Gamma.push(g1);
      this.params.push(g1);

      const b1 = new Param(this.embeddingDim);
      b1.initZero();
      this.ln1Beta.push(b1);
      this.params.push(b1);

      // Layer norm 2 parameters (pre-FFN)
      const g2 = new Param(this.embeddingDim);
      g2.initOne();
      this.ln2Gamma.push(g2);
      this.params.push(g2);

      const b2Param = new Param(this.embeddingDim);
      b2Param.initZero();
      this.ln2Beta.push(b2Param);
      this.params.push(b2Param);

      // Multi-head attention parameters
      const wqHeads: Param[] = [];
      const wkHeads: Param[] = [];
      const wvHeads: Param[] = [];

      for (let h = 0; h < this.numHeads; h++) {
        const wq = new Param(this.headDim * this.embeddingDim);
        wq.initXavier(this.embeddingDim, this.headDim);
        wqHeads.push(wq);
        this.params.push(wq);

        const wk = new Param(this.headDim * this.embeddingDim);
        wk.initXavier(this.embeddingDim, this.headDim);
        wkHeads.push(wk);
        this.params.push(wk);

        const wv = new Param(this.headDim * this.embeddingDim);
        wv.initXavier(this.embeddingDim, this.headDim);
        wvHeads.push(wv);
        this.params.push(wv);
      }

      this.attnWq.push(wqHeads);
      this.attnWk.push(wkHeads);
      this.attnWv.push(wvHeads);

      // Output projection
      const wo = new Param(this.embeddingDim * this.embeddingDim);
      wo.initXavier(this.embeddingDim, this.embeddingDim);
      this.attnWo.push(wo);
      this.params.push(wo);

      const bo = new Param(this.embeddingDim);
      bo.initZero();
      this.attnBo.push(bo);
      this.params.push(bo);

      // Feed-forward network parameters
      const w1 = new Param(this.ffnHiddenDim * this.embeddingDim);
      w1.initXavier(this.embeddingDim, this.ffnHiddenDim);
      this.ffnW1.push(w1);
      this.params.push(w1);

      const fb1 = new Param(this.ffnHiddenDim);
      fb1.initZero();
      this.ffnB1.push(fb1);
      this.params.push(fb1);

      const w2 = new Param(this.embeddingDim * this.ffnHiddenDim);
      w2.initXavier(this.ffnHiddenDim, this.embeddingDim);
      this.ffnW2.push(w2);
      this.params.push(w2);

      const fb2 = new Param(this.embeddingDim);
      fb2.initZero();
      this.ffnB2.push(fb2);
      this.params.push(fb2);
    }

    // Attention pooling weight
    this.poolW = new Param(this.embeddingDim);
    this.poolW.initSmall(0.1);
    this.params.push(this.poolW);

    // Output projection
    this.outW = new Param(outputDim * this.embeddingDim);
    this.outW.initXavier(this.embeddingDim, outputDim);
    this.params.push(this.outW);

    this.outB = new Param(outputDim);
    this.outB.initZero();
    this.params.push(this.outB);

    this.initialized = true;
  }

  /**
   * Layer normalization forward pass
   * output = γ * (x - μ) / √(σ² + ε) + β
   * @param x - Input vector
   * @param gamma - Scale parameter
   * @param beta - Shift parameter
   * @param outNorm - Output buffer for normalized values (for backprop)
   * @param out - Output buffer
   * @returns [mean, invStd] for backprop
   */
  private layerNormForward(
    x: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    outNorm: Float64Array,
    out: Float64Array,
  ): [number, number] {
    const len = x.length;

    // Compute mean
    let mean = 0;
    for (let i = 0; i < len; i++) {
      mean += x[i];
    }
    mean /= len;

    // Compute variance
    let variance = 0;
    for (let i = 0; i < len; i++) {
      const d = x[i] - mean;
      variance += d * d;
    }
    variance /= len;

    const invStd = 1 / Math.sqrt(variance + this.epsilon);

    // Normalize and apply affine transform
    for (let i = 0; i < len; i++) {
      outNorm[i] = (x[i] - mean) * invStd;
      out[i] = gamma[i] * outNorm[i] + beta[i];
    }

    return [mean, invStd];
  }

  /**
   * Layer normalization backward pass
   * Computes gradients for gamma, beta, and propagates to input
   */
  private layerNormBackward(
    dOut: Float64Array,
    normalized: Float64Array,
    invStd: number,
    gamma: Param,
    beta: Param,
    dInput: Float64Array,
  ): void {
    const len = dOut.length;

    // Gradient for gamma and beta
    for (let i = 0; i < len; i++) {
      gamma.grad[i] += dOut[i] * normalized[i];
      beta.grad[i] += dOut[i];
    }

    // Gradient for input (simplified - full formula is more complex)
    // dL/dx ≈ γ * invStd * dL/dy (ignoring mean/var gradients for stability)
    for (let i = 0; i < len; i++) {
      dInput[i] += dOut[i] * gamma.data[i] * invStd;
    }
  }

  /**
   * Temporal convolution forward pass for one scale
   * Applies 1D convolution followed by GELU activation
   */
  private temporalConvForward(
    input: Float64Array[],
    scaleIdx: number,
    stride: number,
    outPre: Float64Array[],
    outAct: Float64Array[],
  ): void {
    const seqLen = input.length;
    const outLen = Math.max(
      1,
      Math.floor((seqLen - this.temporalKernelSize) / stride) + 1,
    );
    const W = this.temporalConvW[scaleIdx].data;
    const B = this.temporalConvB[scaleIdx].data;

    for (let t = 0; t < outLen; t++) {
      const pre = outPre[t];
      const act = outAct[t];
      const startPos = t * stride;

      for (let o = 0; o < this.embeddingDim; o++) {
        let sum = B[o];
        for (let k = 0; k < this.temporalKernelSize; k++) {
          const pos = startPos + k;
          if (pos < seqLen) {
            for (let i = 0; i < this.inputDim; i++) {
              const wIdx = o * (this.inputDim * this.temporalKernelSize) +
                i * this.temporalKernelSize + k;
              sum += input[pos][i] * W[wIdx];
            }
          }
        }
        pre[o] = sum;
        act[o] = gelu(sum);
      }
    }
  }

  /**
   * Multi-head self-attention forward pass
   */
  private multiHeadAttentionForward(
    input: Float64Array[],
    blockIdx: number,
    outQ: Float64Array[][],
    outK: Float64Array[][],
    outV: Float64Array[][],
    outScores: Float64Array[][],
    outWeights: Float64Array[][],
    outHeadOut: Float64Array[][],
    outConcat: Float64Array[],
    output: Float64Array[],
  ): void {
    const seqLen = input.length;
    const scale = 1 / Math.sqrt(this.headDim);

    // Process each head
    for (let h = 0; h < this.numHeads; h++) {
      const Wq = this.attnWq[blockIdx][h].data;
      const Wk = this.attnWk[blockIdx][h].data;
      const Wv = this.attnWv[blockIdx][h].data;

      // Compute Q, K, V projections
      for (let t = 0; t < seqLen; t++) {
        const q = outQ[h][t];
        const k = outK[h][t];
        const v = outV[h][t];

        for (let d = 0; d < this.headDim; d++) {
          let qSum = 0, kSum = 0, vSum = 0;
          for (let e = 0; e < this.embeddingDim; e++) {
            const idx = d * this.embeddingDim + e;
            const inputVal = input[t][e];
            qSum += inputVal * Wq[idx];
            kSum += inputVal * Wk[idx];
            vSum += inputVal * Wv[idx];
          }
          q[d] = qSum;
          k[d] = kSum;
          v[d] = vSum;
        }
      }

      // Compute attention scores and weights
      for (let i = 0; i < seqLen; i++) {
        const scores = outScores[h][i];
        const weights = outWeights[h][i];
        const Q_i = outQ[h][i];

        // Compute scaled dot-product attention scores
        for (let j = 0; j < seqLen; j++) {
          let s = 0;
          for (let d = 0; d < this.headDim; d++) {
            s += Q_i[d] * outK[h][j][d];
          }
          scores[j] = s * scale;
        }

        // Softmax
        softmax(scores, weights);

        // Compute weighted sum of values
        const headOut = outHeadOut[h][i];
        headOut.fill(0);
        for (let j = 0; j < seqLen; j++) {
          const w = weights[j];
          const V_j = outV[h][j];
          for (let d = 0; d < this.headDim; d++) {
            headOut[d] += w * V_j[d];
          }
        }
      }
    }

    // Concatenate heads and apply output projection
    const Wo = this.attnWo[blockIdx].data;
    const Bo = this.attnBo[blockIdx].data;

    for (let t = 0; t < seqLen; t++) {
      const concat = outConcat[t];

      // Concatenate all heads
      for (let h = 0; h < this.numHeads; h++) {
        const headOut = outHeadOut[h][t];
        for (let d = 0; d < this.headDim; d++) {
          concat[h * this.headDim + d] = headOut[d];
        }
      }

      // Output projection
      const out = output[t];
      for (let o = 0; o < this.embeddingDim; o++) {
        let sum = Bo[o];
        for (let c = 0; c < this.embeddingDim; c++) {
          sum += concat[c] * Wo[o * this.embeddingDim + c];
        }
        out[o] = sum;
      }
    }
  }

  /**
   * Feed-forward network forward pass
   * FFN(x) = GELU(xW1 + b1)W2 + b2
   */
  private ffnForward(
    input: Float64Array[],
    blockIdx: number,
    outPre: Float64Array[],
    outHidden: Float64Array[],
    output: Float64Array[],
  ): void {
    const W1 = this.ffnW1[blockIdx].data;
    const B1 = this.ffnB1[blockIdx].data;
    const W2 = this.ffnW2[blockIdx].data;
    const B2 = this.ffnB2[blockIdx].data;
    const seqLen = input.length;

    for (let t = 0; t < seqLen; t++) {
      const x = input[t];
      const pre = outPre[t];
      const h = outHidden[t];
      const out = output[t];

      // First linear layer + GELU
      for (let i = 0; i < this.ffnHiddenDim; i++) {
        let sum = B1[i];
        for (let j = 0; j < this.embeddingDim; j++) {
          sum += x[j] * W1[i * this.embeddingDim + j];
        }
        pre[i] = sum;
        h[i] = gelu(sum);
      }

      // Second linear layer
      for (let i = 0; i < this.embeddingDim; i++) {
        let sum = B2[i];
        for (let j = 0; j < this.ffnHiddenDim; j++) {
          sum += h[j] * W2[i * this.ffnHiddenDim + j];
        }
        out[i] = sum;
      }
    }
  }

  /**
   * Complete forward pass through the network
   * Caches intermediate values for backpropagation
   * @param normalizedInput - Z-score normalized input sequence
   * @returns Output prediction (normalized)
   */
  private forward(normalizedInput: Float64Array[]): Float64Array {
    const seqLen = normalizedInput.length;
    const numScales = this.temporalScales.length;

    // Calculate output lengths for each scale
    const scaleLens: number[] = [];
    for (let s = 0; s < numScales; s++) {
      const stride = this.temporalScales[s];
      const outLen = Math.max(
        1,
        Math.floor((seqLen - this.temporalKernelSize) / stride) + 1,
      );
      scaleLens.push(outLen);
    }
    const minLen = Math.max(1, Math.min(...scaleLens));

    // Initialize forward cache with preallocated buffers
    this.fwdCache = {
      normalizedInput: normalizedInput,
      convPre: [],
      convOut: [],
      scaleOutputs: [],
      fusionConcats: [],
      fusionGatesPre: [],
      fusionGates: [],
      fused: [],
      blockInputs: [],
      ln1Inputs: [],
      ln1Outputs: [],
      ln1Mean: [],
      ln1InvStd: [],
      attnQ: [],
      attnK: [],
      attnV: [],
      attnScores: [],
      attnWeights: [],
      attnHeadOut: [],
      attnConcat: [],
      attnOut: [],
      residual1: [],
      ln2Inputs: [],
      ln2Outputs: [],
      ln2Mean: [],
      ln2InvStd: [],
      ffnPre: [],
      ffnHidden: [],
      ffnOut: [],
      finalHidden: [],
      poolScores: new Float64Array(minLen),
      poolWeights: new Float64Array(minLen),
      aggregated: new Float64Array(this.embeddingDim),
      output: new Float64Array(this.outputDim),
    };

    // Multi-scale temporal convolutions
    for (let s = 0; s < numScales; s++) {
      const stride = this.temporalScales[s];
      const outLen = scaleLens[s];

      // Allocate buffers for this scale
      const convPre: Float64Array[] = [];
      const convOut: Float64Array[] = [];
      for (let t = 0; t < outLen; t++) {
        convPre.push(new Float64Array(this.embeddingDim));
        convOut.push(new Float64Array(this.embeddingDim));
      }

      this.temporalConvForward(normalizedInput, s, stride, convPre, convOut);
      this.fwdCache.convPre.push(convPre);
      this.fwdCache.convOut.push(convOut);

      // Add positional encoding and scale embedding
      const scaleOut: Float64Array[] = [];
      for (let t = 0; t < outLen; t++) {
        const emb = new Float64Array(this.embeddingDim);
        const posIdx = Math.min(t, this.maxSequenceLength - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          emb[e] = convOut[t][e] + this.posEncoding[posIdx][e] +
            this.scaleEmbed[s].data[e];
        }
        scaleOut.push(emb);
      }
      this.fwdCache.scaleOutputs.push(scaleOut);
    }

    // Gated fusion of multi-scale features
    const fusionInDim = this.embeddingDim * numScales;
    const fusionGateW = this.fusionGateW!.data;
    const fusionGateB = this.fusionGateB!.data;

    for (let t = 0; t < minLen; t++) {
      // Concatenate features from all scales
      const concat = new Float64Array(fusionInDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, this.fwdCache.scaleOutputs[s].length - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          concat[s * this.embeddingDim + e] =
            this.fwdCache.scaleOutputs[s][idx][e];
        }
      }
      this.fwdCache.fusionConcats.push(concat);

      // Compute fusion gates (sigmoid of linear projection)
      const gatesPre = new Float64Array(numScales);
      const gates = new Float64Array(numScales);
      for (let g = 0; g < numScales; g++) {
        let sum = fusionGateB[g];
        for (let i = 0; i < fusionInDim; i++) {
          sum += concat[i] * fusionGateW[g * fusionInDim + i];
        }
        gatesPre[g] = sum;
        gates[g] = sigmoid(sum);
      }
      this.fwdCache.fusionGatesPre.push(gatesPre);
      this.fwdCache.fusionGates.push(gates);

      // Apply gated fusion
      const fusedT = new Float64Array(this.embeddingDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, this.fwdCache.scaleOutputs[s].length - 1);
        const gate = gates[s];
        for (let e = 0; e < this.embeddingDim; e++) {
          fusedT[e] += gate * this.fwdCache.scaleOutputs[s][idx][e];
        }
      }
      this.fwdCache.fused.push(fusedT);
    }

    // Transformer blocks
    let current = this.fwdCache.fused;
    const curSeqLen = current.length;

    for (let b = 0; b < this.numBlocks; b++) {
      // Store block input
      const blockInput: Float64Array[] = [];
      for (let t = 0; t < curSeqLen; t++) {
        blockInput.push(new Float64Array(current[t]));
      }
      this.fwdCache.blockInputs.push(blockInput);

      // Layer norm 1 (pre-attention)
      const ln1In: Float64Array[] = [];
      const ln1Out: Float64Array[] = [];
      const ln1Norm: Float64Array[] = [];
      const ln1Mean: Float64Array = new Float64Array(curSeqLen);
      const ln1InvStd: Float64Array = new Float64Array(curSeqLen);

      for (let t = 0; t < curSeqLen; t++) {
        ln1In.push(new Float64Array(current[t]));
        const norm = new Float64Array(this.embeddingDim);
        const out = new Float64Array(this.embeddingDim);
        const [mean, invStd] = this.layerNormForward(
          current[t],
          this.ln1Gamma[b].data,
          this.ln1Beta[b].data,
          norm,
          out,
        );
        ln1Norm.push(norm);
        ln1Out.push(out);
        ln1Mean[t] = mean;
        ln1InvStd[t] = invStd;
      }
      this.fwdCache.ln1Inputs.push(ln1In);
      this.fwdCache.ln1Outputs.push(ln1Out);
      this.fwdCache.ln1Mean.push(ln1Mean);
      this.fwdCache.ln1InvStd.push(ln1InvStd);

      // Multi-head self-attention
      const attnQ: Float64Array[][] = [];
      const attnK: Float64Array[][] = [];
      const attnV: Float64Array[][] = [];
      const attnScores: Float64Array[][] = [];
      const attnWeights: Float64Array[][] = [];
      const attnHeadOut: Float64Array[][] = [];

      for (let h = 0; h < this.numHeads; h++) {
        const Q: Float64Array[] = [];
        const K: Float64Array[] = [];
        const V: Float64Array[] = [];
        const scores: Float64Array[] = [];
        const weights: Float64Array[] = [];
        const headOut: Float64Array[] = [];

        for (let t = 0; t < curSeqLen; t++) {
          Q.push(new Float64Array(this.headDim));
          K.push(new Float64Array(this.headDim));
          V.push(new Float64Array(this.headDim));
          scores.push(new Float64Array(curSeqLen));
          weights.push(new Float64Array(curSeqLen));
          headOut.push(new Float64Array(this.headDim));
        }

        attnQ.push(Q);
        attnK.push(K);
        attnV.push(V);
        attnScores.push(scores);
        attnWeights.push(weights);
        attnHeadOut.push(headOut);
      }

      const attnConcat: Float64Array[] = [];
      const attnOut: Float64Array[] = [];
      for (let t = 0; t < curSeqLen; t++) {
        attnConcat.push(new Float64Array(this.embeddingDim));
        attnOut.push(new Float64Array(this.embeddingDim));
      }

      this.multiHeadAttentionForward(
        ln1Out,
        b,
        attnQ,
        attnK,
        attnV,
        attnScores,
        attnWeights,
        attnHeadOut,
        attnConcat,
        attnOut,
      );

      this.fwdCache.attnQ.push(attnQ);
      this.fwdCache.attnK.push(attnK);
      this.fwdCache.attnV.push(attnV);
      this.fwdCache.attnScores.push(attnScores);
      this.fwdCache.attnWeights.push(attnWeights);
      this.fwdCache.attnHeadOut.push(attnHeadOut);
      this.fwdCache.attnConcat.push(attnConcat);
      this.fwdCache.attnOut.push(attnOut);

      // Residual connection 1
      const residual1: Float64Array[] = [];
      for (let t = 0; t < curSeqLen; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = current[t][e] + attnOut[t][e];
        }
        residual1.push(r);
      }
      this.fwdCache.residual1.push(residual1);

      // Layer norm 2 (pre-FFN)
      const ln2In: Float64Array[] = [];
      const ln2Out: Float64Array[] = [];
      const ln2Norm: Float64Array[] = [];
      const ln2Mean: Float64Array = new Float64Array(curSeqLen);
      const ln2InvStd: Float64Array = new Float64Array(curSeqLen);

      for (let t = 0; t < curSeqLen; t++) {
        ln2In.push(new Float64Array(residual1[t]));
        const norm = new Float64Array(this.embeddingDim);
        const out = new Float64Array(this.embeddingDim);
        const [mean, invStd] = this.layerNormForward(
          residual1[t],
          this.ln2Gamma[b].data,
          this.ln2Beta[b].data,
          norm,
          out,
        );
        ln2Norm.push(norm);
        ln2Out.push(out);
        ln2Mean[t] = mean;
        ln2InvStd[t] = invStd;
      }
      this.fwdCache.ln2Inputs.push(ln2In);
      this.fwdCache.ln2Outputs.push(ln2Out);
      this.fwdCache.ln2Mean.push(ln2Mean);
      this.fwdCache.ln2InvStd.push(ln2InvStd);

      // Feed-forward network
      const ffnPre: Float64Array[] = [];
      const ffnHidden: Float64Array[] = [];
      const ffnOut: Float64Array[] = [];
      for (let t = 0; t < curSeqLen; t++) {
        ffnPre.push(new Float64Array(this.ffnHiddenDim));
        ffnHidden.push(new Float64Array(this.ffnHiddenDim));
        ffnOut.push(new Float64Array(this.embeddingDim));
      }

      this.ffnForward(ln2Out, b, ffnPre, ffnHidden, ffnOut);
      this.fwdCache.ffnPre.push(ffnPre);
      this.fwdCache.ffnHidden.push(ffnHidden);
      this.fwdCache.ffnOut.push(ffnOut);

      // Residual connection 2
      current = [];
      for (let t = 0; t < curSeqLen; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = residual1[t][e] + ffnOut[t][e];
        }
        current.push(r);
      }
    }

    // Store final hidden states
    this.fwdCache.finalHidden = current;
    const finalSeqLen = current.length;

    // Attention pooling
    const poolWData = this.poolW!.data;
    const poolScores = this.fwdCache.poolScores;
    const poolWeights = this.fwdCache.poolWeights;

    for (let t = 0; t < finalSeqLen; t++) {
      let score = 0;
      for (let e = 0; e < this.embeddingDim; e++) {
        score += current[t][e] * poolWData[e];
      }
      poolScores[t] = score;
    }
    softmax(poolScores, poolWeights);

    // Weighted aggregation
    const aggregated = this.fwdCache.aggregated;
    aggregated.fill(0);
    for (let t = 0; t < finalSeqLen; t++) {
      const w = poolWeights[t];
      for (let e = 0; e < this.embeddingDim; e++) {
        aggregated[e] += w * current[t][e];
      }
    }

    // Output projection
    const outWData = this.outW!.data;
    const outBData = this.outB!.data;
    const output = this.fwdCache.output;

    for (let o = 0; o < this.outputDim; o++) {
      let sum = outBData[o];
      for (let e = 0; e < this.embeddingDim; e++) {
        sum += aggregated[e] * outWData[o * this.embeddingDim + e];
      }
      output[o] = sum;
    }

    return output;
  }

  /**
   * Complete backward pass through the network
   * Computes gradients for all parameters
   * @param target - Target values (normalized)
   * @param predicted - Predicted values (normalized)
   * @param sampleWeight - Weight for this sample (for outlier downweighting)
   * @returns Gradient L2 norm
   */
  private backward(
    target: Float64Array,
    predicted: Float64Array,
    sampleWeight: number,
  ): number {
    if (!this.fwdCache) return 0;

    // Zero all gradients
    for (const p of this.params) {
      p.zeroGrad();
    }

    const cache = this.fwdCache;
    const finalHidden = cache.finalHidden;
    const seqLen = finalHidden.length;

    // MSE gradient: dL/dy = (y - t) * weight / n
    const dOutput = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      dOutput[o] = (predicted[o] - target[o]) * sampleWeight / this.outputDim;
    }

    // Output layer backward
    const aggregated = cache.aggregated;
    const dAggregated = new Float64Array(this.embeddingDim);

    for (let o = 0; o < this.outputDim; o++) {
      this.outB!.grad[o] += dOutput[o];
      for (let e = 0; e < this.embeddingDim; e++) {
        this.outW!.grad[o * this.embeddingDim + e] += dOutput[o] *
          aggregated[e];
        dAggregated[e] += dOutput[o] *
          this.outW!.data[o * this.embeddingDim + e];
      }
    }

    // Attention pooling backward
    const poolWeights = cache.poolWeights;
    const dFinalHidden: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      dFinalHidden.push(new Float64Array(this.embeddingDim));
    }

    // Gradient through weighted sum: d(sum w_t * h_t) / dh_t = w_t * dL/dagg
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        dFinalHidden[t][e] = poolWeights[t] * dAggregated[e];
      }
    }

    // Gradient through softmax for pool weights
    // dL/ds_i = sum_j (dL/dw_j * w_j * (delta_ij - w_i))
    const dPoolScores = new Float64Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      let dScore = 0;
      for (let j = 0; j < seqLen; j++) {
        // dw_j/ds_i = w_j * (delta_ij - w_i)
        const jacobian = poolWeights[j] * ((i === j ? 1 : 0) - poolWeights[i]);
        // Upstream gradient for w_j is sum_e(dAggregated[e] * finalHidden[j][e])
        let upstream = 0;
        for (let e = 0; e < this.embeddingDim; e++) {
          upstream += dAggregated[e] * finalHidden[j][e];
        }
        dScore += upstream * jacobian;
      }
      dPoolScores[i] = dScore;
    }

    // Gradient for poolW and add to dFinalHidden
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        this.poolW!.grad[e] += dPoolScores[t] * finalHidden[t][e];
        dFinalHidden[t][e] += dPoolScores[t] * this.poolW!.data[e];
      }
    }

    // Backward through transformer blocks (in reverse order)
    let dCurrent = dFinalHidden;

    for (let b = this.numBlocks - 1; b >= 0; b--) {
      const residual1 = cache.residual1[b];
      const ln2Out = cache.ln2Outputs[b];
      const ln2InvStd = cache.ln2InvStd[b];
      const ffnHidden = cache.ffnHidden[b];
      const ffnPre = cache.ffnPre[b];
      const ln1Out = cache.ln1Outputs[b];
      const ln1InvStd = cache.ln1InvStd[b];
      const blockInput = cache.blockInputs[b];
      const attnConcat = cache.attnConcat[b];
      const attnOut = cache.attnOut[b];
      const attnWeights = cache.attnWeights[b];
      const attnV = cache.attnV[b];
      const attnQ = cache.attnQ[b];
      const attnK = cache.attnK[b];
      const curLen = dCurrent.length;

      // Split gradient through residual connection 2
      const dResidual1: Float64Array[] = [];
      const dFFNOut: Float64Array[] = [];
      for (let t = 0; t < curLen; t++) {
        dResidual1.push(new Float64Array(dCurrent[t]));
        dFFNOut.push(new Float64Array(dCurrent[t]));
      }

      // FFN backward
      for (let t = 0; t < curLen; t++) {
        const dHidden = new Float64Array(this.ffnHiddenDim);

        // W2 backward
        for (let o = 0; o < this.embeddingDim; o++) {
          this.ffnB2[b].grad[o] += dFFNOut[t][o];
          for (let h = 0; h < this.ffnHiddenDim; h++) {
            this.ffnW2[b].grad[o * this.ffnHiddenDim + h] += dFFNOut[t][o] *
              ffnHidden[t][h];
            dHidden[h] += dFFNOut[t][o] *
              this.ffnW2[b].data[o * this.ffnHiddenDim + h];
          }
        }

        // GELU backward
        const dPreGelu = new Float64Array(this.ffnHiddenDim);
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          dPreGelu[h] = dHidden[h] * geluGrad(ffnPre[t][h]);
        }

        // W1 backward
        const dLn2 = new Float64Array(this.embeddingDim);
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          this.ffnB1[b].grad[h] += dPreGelu[h];
          for (let e = 0; e < this.embeddingDim; e++) {
            this.ffnW1[b].grad[h * this.embeddingDim + e] += dPreGelu[h] *
              ln2Out[t][e];
            dLn2[e] += dPreGelu[h] *
              this.ffnW1[b].data[h * this.embeddingDim + e];
          }
        }

        // Layer norm 2 backward (simplified)
        for (let e = 0; e < this.embeddingDim; e++) {
          const normalized = (residual1[t][e] - cache.ln2Mean[b][t]) *
            ln2InvStd[t];
          this.ln2Gamma[b].grad[e] += dLn2[e] * normalized;
          this.ln2Beta[b].grad[e] += dLn2[e];
          dResidual1[t][e] += dLn2[e] * this.ln2Gamma[b].data[e] * ln2InvStd[t];
        }
      }

      // Split gradient through residual connection 1
      const dBlockInput: Float64Array[] = [];
      const dAttnOut: Float64Array[] = [];
      for (let t = 0; t < curLen; t++) {
        dBlockInput.push(new Float64Array(dResidual1[t]));
        dAttnOut.push(new Float64Array(dResidual1[t]));
      }

      // Attention backward
      for (let t = 0; t < curLen; t++) {
        // Output projection backward
        const dConcat = new Float64Array(this.embeddingDim);
        for (let o = 0; o < this.embeddingDim; o++) {
          this.attnBo[b].grad[o] += dAttnOut[t][o];
          for (let c = 0; c < this.embeddingDim; c++) {
            this.attnWo[b].grad[o * this.embeddingDim + c] += dAttnOut[t][o] *
              attnConcat[t][c];
            dConcat[c] += dAttnOut[t][o] *
              this.attnWo[b].data[o * this.embeddingDim + c];
          }
        }

        // Backward through each head
        for (let h = 0; h < this.numHeads; h++) {
          // Gradient for head output
          const dHeadOut = new Float64Array(this.headDim);
          for (let d = 0; d < this.headDim; d++) {
            dHeadOut[d] = dConcat[h * this.headDim + d];
          }

          // Backward through attention: headOut = sum_j(w_j * V_j)
          // dV_j += w_j * dHeadOut
          // dw_j += V_j . dHeadOut
          const weights = attnWeights[h][t];
          const dWeights = new Float64Array(curLen);

          for (let j = 0; j < curLen; j++) {
            let dwj = 0;
            for (let d = 0; d < this.headDim; d++) {
              dwj += attnV[h][j][d] * dHeadOut[d];
            }
            dWeights[j] = dwj;
          }

          // Backward through softmax
          // dScores[i] = sum_j(dWeights[j] * weights[j] * (delta_ij - weights[i]))
          const dScores = new Float64Array(curLen);
          for (let i = 0; i < curLen; i++) {
            let ds = 0;
            for (let j = 0; j < curLen; j++) {
              const jacobian = weights[j] * ((i === j ? 1 : 0) - weights[i]);
              ds += dWeights[j] * jacobian;
            }
            dScores[i] = ds;
          }

          // Backward through scaled dot product
          const scale = 1 / Math.sqrt(this.headDim);
          const Q_t = attnQ[h][t];

          // dQ[t] += sum_j(dScores[j] * scale * K[j])
          // dK[j] += dScores[j] * scale * Q[t]
          const dQ = new Float64Array(this.headDim);
          for (let j = 0; j < curLen; j++) {
            const ds = dScores[j] * scale;
            for (let d = 0; d < this.headDim; d++) {
              dQ[d] += ds * attnK[h][j][d];
            }
          }

          // Backward through Q, K, V projections (simplified - just update weights)
          // This is an approximation that still allows learning
          for (let d = 0; d < this.headDim; d++) {
            for (let e = 0; e < this.embeddingDim; e++) {
              const wIdx = d * this.embeddingDim + e;
              const inputVal = ln1Out[t][e];
              // Q gradient
              this.attnWq[b][h].grad[wIdx] += dQ[d] * inputVal;
              // V gradient (from weighted sum)
              for (let j = 0; j < curLen; j++) {
                this.attnWv[b][h].grad[wIdx] += weights[j] * dHeadOut[d] *
                  ln1Out[j][e];
              }
              // K gradient
              for (let j = 0; j < curLen; j++) {
                this.attnWk[b][h].grad[wIdx] += dScores[j] * scale * Q_t[d] *
                  ln1Out[j][e];
              }
            }
          }
        }

        // Layer norm 1 backward (simplified)
        const dLn1 = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          // Sum up gradients from all attention heads going to this position
          let totalGrad = 0;
          for (let h = 0; h < this.numHeads; h++) {
            totalGrad += dConcat[h * this.headDim + (e % this.headDim)] /
              this.numHeads;
          }
          dLn1[e] = totalGrad;
        }

        for (let e = 0; e < this.embeddingDim; e++) {
          const normalized = (blockInput[t][e] - cache.ln1Mean[b][t]) *
            ln1InvStd[t];
          this.ln1Gamma[b].grad[e] += dLn1[e] * normalized;
          this.ln1Beta[b].grad[e] += dLn1[e];
          dBlockInput[t][e] += dLn1[e] * this.ln1Gamma[b].data[e] *
            ln1InvStd[t];
        }
      }

      dCurrent = dBlockInput;
    }

    // Fusion backward
    const fusionGates = cache.fusionGates;
    const fusionConcats = cache.fusionConcats;
    const scaleOutputs = cache.scaleOutputs;
    const numScales = this.temporalScales.length;
    const fusionInDim = this.embeddingDim * numScales;

    for (let t = 0; t < Math.min(dCurrent.length, fusionGates.length); t++) {
      const gates = fusionGates[t];
      const concat = fusionConcats[t];
      const gatesPre = cache.fusionGatesPre[t];

      for (let s = 0; s < numScales; s++) {
        const scaleIdx = Math.min(t, scaleOutputs[s].length - 1);
        const gateGrad = sigmoidGrad(gates[s]);

        for (let e = 0; e < this.embeddingDim; e++) {
          const dFused = dCurrent[t][e];
          const scaleVal = scaleOutputs[s][scaleIdx][e];

          // Gradient through gate
          const dGatePre = dFused * scaleVal * gateGrad;

          // Fusion gate weight gradient
          for (let i = 0; i < fusionInDim; i++) {
            this.fusionGateW!.grad[s * fusionInDim + i] += dGatePre * concat[i];
          }
          this.fusionGateB!.grad[s] += dGatePre;

          // Scale embedding gradient
          this.scaleEmbed[s].grad[e] += dFused * gates[s];
        }
      }
    }

    // Temporal conv backward
    const normalizedInput = cache.normalizedInput;
    for (let s = 0; s < numScales; s++) {
      const convPre = cache.convPre[s];
      const stride = this.temporalScales[s];

      for (
        let t = 0;
        t < Math.min(convPre.length, dCurrent.length, fusionGates.length);
        t++
      ) {
        const gate = fusionGates[t][s];

        for (let o = 0; o < this.embeddingDim; o++) {
          const dConvOut = dCurrent[t][o] * gate;
          const dPre = dConvOut * geluGrad(convPre[t][o]);

          this.temporalConvB[s].grad[o] += dPre;

          const startPos = t * stride;
          for (let k = 0; k < this.temporalKernelSize; k++) {
            const pos = startPos + k;
            if (pos < normalizedInput.length) {
              for (let i = 0; i < this.inputDim; i++) {
                const wIdx = o * (this.inputDim * this.temporalKernelSize) +
                  i * this.temporalKernelSize + k;
                this.temporalConvW[s].grad[wIdx] += dPre *
                  normalizedInput[pos][i];
              }
            }
          }
        }
      }
    }

    // Compute gradient norm
    let gradNorm = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.size; i++) {
        gradNorm += p.grad[i] * p.grad[i];
      }
    }
    gradNorm = Math.sqrt(gradNorm);

    // Gradient clipping
    if (gradNorm > this.gradientClipNorm && gradNorm > 0) {
      const scale = this.gradientClipNorm / gradNorm;
      for (const p of this.params) {
        for (let i = 0; i < p.size; i++) {
          p.grad[i] *= scale;
        }
      }
      gradNorm = this.gradientClipNorm;
    }

    return gradNorm;
  }

  /**
   * Get effective learning rate with cosine warmup schedule
   * lr = lr_base * (step / warmup_steps) for step < warmup_steps
   * lr = lr_base * 0.5 * (1 + cos(π * progress)) otherwise
   * @returns Current learning rate
   */
  private getEffectiveLR(): number {
    if (this.updateCount < this.warmupSteps) {
      return this.learningRate * ((this.updateCount + 1) / this.warmupSteps);
    }
    const progress = (this.updateCount - this.warmupSteps) /
      Math.max(1, this.totalSteps - this.warmupSteps);
    return this.learningRate * 0.5 *
      (1 + Math.cos(Math.PI * Math.min(progress, 1)));
  }

  /**
   * Perform optimizer step on all parameters
   */
  private optimizerStep(): void {
    this.updateCount++;
    const lr = this.getEffectiveLR();

    for (const p of this.params) {
      p.adamStep(
        lr,
        this.beta1,
        this.beta2,
        this.epsilon,
        this.updateCount,
        this.regularizationStrength,
      );
    }
  }

  /**
   * Perform incremental online learning with one batch of data
   *
   * @param data - Input data with xCoordinates (sequence) and yCoordinates (targets)
   * @returns FitResult with loss, gradient norm, and training status
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],
   *   yCoordinates: [[7], [8], [9]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Input validation
    if (!xCoordinates || xCoordinates.length === 0) {
      throw new Error("xCoordinates cannot be empty");
    }
    if (!yCoordinates || yCoordinates.length === 0) {
      throw new Error("yCoordinates cannot be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "xCoordinates and yCoordinates must have the same length",
      );
    }

    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;
    const seqLen = xCoordinates.length;

    // Lazy initialization
    if (!this.initialized) {
      this.init(inputDim, outputDim);
    }

    // Convert to typed arrays and store for prediction
    const xInput: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      const x = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        x[i] = xCoordinates[t][i];
      }
      xInput.push(x);
    }
    this.lastRawInput = xInput;

    // Get target (last timestep)
    const yTarget = new Float64Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      yTarget[i] = yCoordinates[seqLen - 1][i];
    }

    // Update normalization statistics BEFORE forward pass
    // Only update with the last input to avoid overweighting
    this.inputWelford!.update(xInput[seqLen - 1]);
    this.outputWelford!.update(yTarget);

    // Get normalization statistics
    const inputMean = this.inputWelford!.mean;
    this.inputWelford!.getStdInto(this.epsilon, this.inputStdBuf!);
    const inputStd = this.inputStdBuf!;

    const outputMean = this.outputWelford!.mean;
    this.outputWelford!.getStdInto(this.epsilon, this.outputStdBuf!);
    const outputStd = this.outputStdBuf!;

    // Normalize input
    const normalizedInput: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      const norm = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        norm[i] = (xInput[t][i] - inputMean[i]) / inputStd[i];
        // Clip to prevent extreme values
        norm[i] = clip(norm[i], -10, 10);
      }
      normalizedInput.push(norm);
    }

    // Forward pass (model outputs in normalized space)
    const predictedNorm = this.forward(normalizedInput);

    // Normalize target for comparison
    const normalizedTarget = new Float64Array(outputDim);
    for (let o = 0; o < outputDim; o++) {
      normalizedTarget[o] = (yTarget[o] - outputMean[o]) / outputStd[o];
      normalizedTarget[o] = clip(normalizedTarget[o], -10, 10);
    }

    // Compute MSE loss in normalized space
    let mseLoss = 0;
    for (let o = 0; o < outputDim; o++) {
      const diff = predictedNorm[o] - normalizedTarget[o];
      mseLoss += diff * diff;
    }
    mseLoss /= 2 * outputDim;

    // L2 regularization
    let l2Loss = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.size; i++) {
        l2Loss += p.data[i] * p.data[i];
      }
    }
    l2Loss *= this.regularizationStrength / 2;

    const totalLoss = mseLoss + l2Loss;

    // Outlier detection based on standardized residuals
    let isOutlier = false;
    let sampleWeight = 1.0;

    if (this.sampleCount > 10) {
      let residualNorm = 0;
      for (let o = 0; o < outputDim; o++) {
        const residual = predictedNorm[o] - normalizedTarget[o];
        residualNorm += residual * residual;
      }
      residualNorm = Math.sqrt(residualNorm / outputDim);

      if (residualNorm > this.outlierThreshold) {
        isOutlier = true;
        sampleWeight = 0.1; // Downweight outliers
      }
    }

    // Drift detection
    const driftDetected = this.adwin!.addAndCheck(mseLoss);
    if (driftDetected) {
      this.driftCount++;
    }

    // Backward pass
    const gradNorm = this.backward(
      normalizedTarget,
      predictedNorm,
      sampleWeight,
    );

    // Optimizer step
    this.optimizerStep();

    // Update tracking statistics
    this.sampleCount++;
    this.totalLoss += totalLoss;

    // Circular buffer for recent losses
    this.recentLosses[this.recentLossIdx] = totalLoss;
    this.recentLossIdx = (this.recentLossIdx + 1) % this.recentLosses.length;
    if (this.recentLossCount < this.recentLosses.length) {
      this.recentLossCount++;
    }

    // Convergence check
    const lossDiff = Math.abs(this.prevLoss - totalLoss);
    this.converged = lossDiff < this.convergenceThreshold &&
      this.sampleCount > 100;
    this.prevLoss = totalLoss;

    // Clear forward cache to free memory
    this.fwdCache = null;

    return {
      loss: totalLoss,
      gradientNorm: gradNorm,
      effectiveLearningRate: this.getEffectiveLR(),
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Generate predictions for future timesteps
   *
   * @param futureSteps - Number of future timesteps to predict
   * @returns PredictionResult with predictions and confidence intervals
   * @example
   * ```typescript
   * const result = model.predict(5);
   * for (const pred of result.predictions) {
   *   console.log(`Predicted: ${pred.predicted}, CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   * }
   * ```
   */
  public predict(futureSteps: number): PredictionResult {
    if (!this.initialized || !this.lastRawInput || this.sampleCount < 1) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const zScore95 = 1.96;

    // Copy raw input for autoregressive prediction
    let currentRawInput: Float64Array[] = [];
    for (const x of this.lastRawInput) {
      currentRawInput.push(new Float64Array(x));
    }

    // Get normalization statistics
    const inputMean = this.inputWelford!.mean;
    this.inputWelford!.getStdInto(this.epsilon, this.inputStdBuf!);
    const inputStd = this.inputStdBuf!;

    const outputMean = this.outputWelford!.mean;
    this.outputWelford!.getStdInto(this.epsilon, this.outputStdBuf!);
    const outputStd = this.outputStdBuf!;

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current input
      const normalizedInput: Float64Array[] = [];
      for (let t = 0; t < currentRawInput.length; t++) {
        const norm = new Float64Array(this.inputDim);
        for (let i = 0; i < this.inputDim; i++) {
          norm[i] = (currentRawInput[t][i] - inputMean[i]) / inputStd[i];
          norm[i] = clip(norm[i], -10, 10);
        }
        normalizedInput.push(norm);
      }

      // Forward pass
      const predictedNorm = this.forward(normalizedInput);

      // Denormalize prediction
      const denormalized: number[] = [];
      const standardError: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];

      for (let o = 0; o < this.outputDim; o++) {
        // Denormalize
        const pred = predictedNorm[o] * outputStd[o] + outputMean[o];

        // Standard error with step-dependent uncertainty
        const se = outputStd[o] / Math.sqrt(Math.max(this.sampleCount, 10));
        const stepMultiplier = 1 + step * 0.1; // Increase uncertainty for further predictions

        denormalized.push(pred);
        standardError.push(se * stepMultiplier);
        lowerBound.push(pred - zScore95 * se * stepMultiplier);
        upperBound.push(pred + zScore95 * se * stepMultiplier);
      }

      predictions.push({
        predicted: denormalized,
        lowerBound,
        upperBound,
        standardError,
      });

      // Autoregressive: shift input and append prediction
      if (currentRawInput.length > 1 && this.inputDim === this.outputDim) {
        const newInput: Float64Array[] = [];
        for (let t = 1; t < currentRawInput.length; t++) {
          newInput.push(currentRawInput[t]);
        }
        const rawPred = new Float64Array(this.outputDim);
        for (let o = 0; o < this.outputDim; o++) {
          rawPred[o] = denormalized[o];
        }
        newInput.push(rawPred);
        currentRawInput = newInput;
      }
    }

    // Clear cache after prediction
    this.fwdCache = null;

    // Compute accuracy from recent losses
    let avgLoss = 0;
    if (this.recentLossCount > 0) {
      for (let i = 0; i < this.recentLossCount; i++) {
        avgLoss += this.recentLosses[i];
      }
      avgLoss /= this.recentLossCount;
    } else {
      avgLoss = this.sampleCount > 0 ? this.totalLoss / this.sampleCount : 1;
    }

    // Convert loss to accuracy using exponential transform
    // accuracy = exp(-loss) gives values in (0, 1)
    const accuracy = Math.exp(-avgLoss);

    return {
      predictions,
      accuracy: Math.min(Math.max(accuracy, 0), 1),
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get summary of model architecture and training state
   * @returns ModelSummary with configuration and statistics
   */
  public getModelSummary(): ModelSummary {
    let avgLoss = 0;
    if (this.recentLossCount > 0) {
      for (let i = 0; i < this.recentLossCount; i++) {
        avgLoss += this.recentLosses[i];
      }
      avgLoss /= this.recentLossCount;
    }
    const accuracy = Math.exp(-avgLoss);
    const totalParams = this.params.reduce((sum, p) => sum + p.size, 0);

    return {
      isInitialized: this.initialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.numBlocks,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      temporalScales: [...this.temporalScales],
      totalParameters: totalParams,
      sampleCount: this.sampleCount,
      accuracy: Math.min(Math.max(accuracy, 0), 1),
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLR(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Get all model weights and optimizer state
   * @returns WeightInfo with all parameters
   */
  public getWeights(): WeightInfo {
    if (!this.initialized) {
      return {
        temporalConvWeights: [],
        temporalConvBiases: [],
        scaleEmbeddings: [],
        positionalEncoding: [],
        fusionGateWeights: [],
        fusionGateBiases: [],
        attentionQWeights: [],
        attentionKWeights: [],
        attentionVWeights: [],
        attentionOutWeights: [],
        attentionOutBiases: [],
        ffnW1: [],
        ffnB1: [],
        ffnW2: [],
        ffnB2: [],
        ln1Gamma: [],
        ln1Beta: [],
        ln2Gamma: [],
        ln2Beta: [],
        poolWeights: [],
        outputWeights: [],
        outputBiases: [],
        firstMoment: [],
        secondMoment: [],
        updateCount: 0,
      };
    }

    return {
      temporalConvWeights: this.temporalConvW.map((p) => p.toArray()),
      temporalConvBiases: this.temporalConvB.map((p) => p.toArray()),
      scaleEmbeddings: this.scaleEmbed.map((p) => p.toArray()),
      positionalEncoding: this.posEncoding.map((pe) => Array.from(pe)),
      fusionGateWeights: this.fusionGateW!.toArray(),
      fusionGateBiases: this.fusionGateB!.toArray(),
      attentionQWeights: this.attnWq.map((block) =>
        block.map((h) => h.toArray())
      ),
      attentionKWeights: this.attnWk.map((block) =>
        block.map((h) => h.toArray())
      ),
      attentionVWeights: this.attnWv.map((block) =>
        block.map((h) => h.toArray())
      ),
      attentionOutWeights: this.attnWo.map((p) => p.toArray()),
      attentionOutBiases: this.attnBo.map((p) => p.toArray()),
      ffnW1: this.ffnW1.map((p) => p.toArray()),
      ffnB1: this.ffnB1.map((p) => p.toArray()),
      ffnW2: this.ffnW2.map((p) => p.toArray()),
      ffnB2: this.ffnB2.map((p) => p.toArray()),
      ln1Gamma: this.ln1Gamma.map((p) => p.toArray()),
      ln1Beta: this.ln1Beta.map((p) => p.toArray()),
      ln2Gamma: this.ln2Gamma.map((p) => p.toArray()),
      ln2Beta: this.ln2Beta.map((p) => p.toArray()),
      poolWeights: this.poolW!.toArray(),
      outputWeights: this.outW!.toArray(),
      outputBiases: this.outB!.toArray(),
      firstMoment: this.params.map((p) => [Array.from(p.m)]),
      secondMoment: this.params.map((p) => [Array.from(p.v)]),
      updateCount: this.updateCount,
    };
  }

  /**
   * Get normalization statistics for input and output
   * @returns NormalizationStats with mean and std
   */
  public getNormalizationStats(): NormalizationStats {
    if (!this.initialized || !this.inputWelford || !this.outputWelford) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    return {
      inputMean: Array.from(this.inputWelford.mean),
      inputStd: Array.from(this.inputWelford.getStd(this.epsilon)),
      outputMean: Array.from(this.outputWelford.mean),
      outputStd: Array.from(this.outputWelford.getStd(this.epsilon)),
      count: this.inputWelford.count,
    };
  }

  /**
   * Reset model to initial state
   */
  public reset(): void {
    this.initialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.sampleCount = 0;
    this.updateCount = 0;
    this.totalLoss = 0;
    this.prevLoss = Infinity;
    this.converged = false;
    this.driftCount = 0;
    this.recentLosses.fill(0);
    this.recentLossIdx = 0;
    this.recentLossCount = 0;
    this.params = [];
    this.fwdCache = null;
    this.lastRawInput = null;
    this.inputWelford = null;
    this.outputWelford = null;
    this.adwin = null;
    this.inputStdBuf = null;
    this.outputStdBuf = null;
    this.bufferPool.clear();

    // Clear parameter arrays
    this.temporalConvW = [];
    this.temporalConvB = [];
    this.scaleEmbed = [];
    this.posEncoding = [];
    this.fusionGateW = null;
    this.fusionGateB = null;
    this.ln1Gamma = [];
    this.ln1Beta = [];
    this.ln2Gamma = [];
    this.ln2Beta = [];
    this.attnWq = [];
    this.attnWk = [];
    this.attnWv = [];
    this.attnWo = [];
    this.attnBo = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    this.poolW = null;
    this.outW = null;
    this.outB = null;
  }

  /**
   * Serialize model state to JSON string
   * @returns JSON string containing all model state
   */
  public save(): string {
    const state = {
      config: {
        numBlocks: this.numBlocks,
        embeddingDim: this.embeddingDim,
        numHeads: this.numHeads,
        ffnHiddenDim: this.ffnHiddenDim,
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
        gradientClipNorm: this.gradientClipNorm,
      },
      state: {
        initialized: this.initialized,
        inputDim: this.inputDim,
        outputDim: this.outputDim,
        sampleCount: this.sampleCount,
        updateCount: this.updateCount,
        totalLoss: this.totalLoss,
        prevLoss: this.prevLoss,
        converged: this.converged,
        driftCount: this.driftCount,
        recentLosses: Array.from(this.recentLosses),
        recentLossIdx: this.recentLossIdx,
        recentLossCount: this.recentLossCount,
      },
      welford: this.initialized
        ? {
          input: this.inputWelford!.serialize(),
          output: this.outputWelford!.serialize(),
        }
        : null,
      adwin: this.initialized ? this.adwin!.serialize() : null,
      params: this.params.map((p) => ({
        data: p.toArray(),
        m: Array.from(p.m),
        v: Array.from(p.v),
      })),
      lastRawInput: this.lastRawInput
        ? this.lastRawInput.map((x) => Array.from(x))
        : null,
    };

    return JSON.stringify(state);
  }

  /**
   * Load model state from JSON string
   * @param jsonString - JSON string from save()
   */
  public load(jsonString: string): void {
    const state = JSON.parse(jsonString);

    if (!state.state.initialized) {
      this.reset();
      return;
    }

    // Reinitialize with saved dimensions
    this.init(state.state.inputDim, state.state.outputDim);

    // Restore normalization statistics
    if (state.welford) {
      this.inputWelford!.deserialize(state.welford.input);
      this.outputWelford!.deserialize(state.welford.output);
    }

    // Restore drift detector
    if (state.adwin) {
      this.adwin!.deserialize(state.adwin);
    }

    // Restore parameters and optimizer state
    if (state.params && state.params.length === this.params.length) {
      for (let i = 0; i < this.params.length; i++) {
        this.params[i].fromArray(state.params[i].data);
        const mLen = Math.min(
          state.params[i].m.length,
          this.params[i].m.length,
        );
        const vLen = Math.min(
          state.params[i].v.length,
          this.params[i].v.length,
        );
        for (let j = 0; j < mLen; j++) {
          this.params[i].m[j] = state.params[i].m[j];
        }
        for (let j = 0; j < vLen; j++) {
          this.params[i].v[j] = state.params[i].v[j];
        }
      }
    }

    // Restore last input
    if (state.lastRawInput) {
      this.lastRawInput = state.lastRawInput.map((x: number[]) =>
        new Float64Array(x)
      );
    }

    // Restore training state
    this.sampleCount = state.state.sampleCount;
    this.updateCount = state.state.updateCount;
    this.totalLoss = state.state.totalLoss;
    this.prevLoss = state.state.prevLoss;
    this.converged = state.state.converged;
    this.driftCount = state.state.driftCount;

    if (state.state.recentLosses) {
      const len = Math.min(
        state.state.recentLosses.length,
        this.recentLosses.length,
      );
      for (let i = 0; i < len; i++) {
        this.recentLosses[i] = state.state.recentLosses[i];
      }
    }
    this.recentLossIdx = state.state.recentLossIdx || 0;
    this.recentLossCount = state.state.recentLossCount || 0;
  }
}

export default FusionTemporalTransformerRegression;
