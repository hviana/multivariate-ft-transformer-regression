/**
 * FusionTemporalTransformerRegression
 *
 * A Fusion Temporal Transformer neural network for multivariate time series regression
 * with incremental online learning capabilities.
 *
 * Features:
 * - Multi-scale temporal convolutions for capturing patterns at different time scales
 * - Transformer blocks with multi-head self-attention
 * - Gated fusion mechanism for combining multi-scale representations
 * - Attention-based pooling for sequence aggregation
 * - Adam optimizer with cosine warmup learning rate schedule
 * - Welford's algorithm for online z-score normalization
 * - ADWIN algorithm for concept drift detection
 * - Outlier detection with sample downweighting
 *
 * @module FusionTemporalTransformerRegression
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Configuration options for the FusionTemporalTransformerRegression model.
 */
export interface FusionTemporalTransformerConfig {
  /** Number of transformer blocks. Default: 3 */
  numBlocks?: number;
  /** Dimension of embeddings. Default: 64 */
  embeddingDim?: number;
  /** Number of attention heads. Default: 8 */
  numHeads?: number;
  /** FFN hidden dimension multiplier. Default: 4 */
  ffnMultiplier?: number;
  /** Attention dropout rate (currently unused). Default: 0.0 */
  attentionDropout?: number;
  /** Base learning rate. Default: 0.001 */
  learningRate?: number;
  /** Number of warmup steps. Default: 100 */
  warmupSteps?: number;
  /** Total training steps for LR schedule. Default: 10000 */
  totalSteps?: number;
  /** Adam beta1. Default: 0.9 */
  beta1?: number;
  /** Adam beta2. Default: 0.999 */
  beta2?: number;
  /** Adam epsilon. Default: 1e-8 */
  epsilon?: number;
  /** L2 regularization strength. Default: 1e-4 */
  regularizationStrength?: number;
  /** Convergence threshold for loss change. Default: 1e-6 */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection. Default: 3.0 */
  outlierThreshold?: number;
  /** ADWIN delta parameter. Default: 0.002 */
  adwinDelta?: number;
  /** Temporal convolution scales (strides). Default: [1, 2, 4] */
  temporalScales?: number[];
  /** Temporal convolution kernel size. Default: 3 */
  temporalKernelSize?: number;
  /** Maximum sequence length for positional encoding. Default: 512 */
  maxSequenceLength?: number;
  /** Fusion dropout rate (currently unused). Default: 0.0 */
  fusionDropout?: number;
}

/**
 * Result from a single online training step.
 */
export interface FitResult {
  /** Total loss (MSE + L2 regularization) */
  loss: number;
  /** L2 norm of gradients */
  gradientNorm: number;
  /** Current effective learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether this sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether the model has converged */
  converged: boolean;
  /** Total number of samples seen */
  sampleIndex: number;
  /** Whether concept drift was detected on this step */
  driftDetected: boolean;
}

/**
 * A single prediction with uncertainty estimates.
 */
export interface SinglePrediction {
  /** Predicted values */
  predicted: number[];
  /** Lower bound of 95% confidence interval */
  lowerBound: number[];
  /** Upper bound of 95% confidence interval */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result from prediction.
 */
export interface PredictionResult {
  /** Array of predictions for each future step */
  predictions: SinglePrediction[];
  /** Model accuracy estimate (1 / (1 + avgLoss)) */
  accuracy: number;
  /** Total number of training samples seen */
  sampleCount: number;
  /** Whether the model is ready for prediction */
  isModelReady: boolean;
}

/**
 * Serialized model weights structure.
 */
export interface WeightInfo {
  /** Temporal convolution weights per scale */
  temporalConvWeights: number[][];
  /** Temporal convolution biases per scale */
  temporalConvBiases: number[][];
  /** Scale embedding vectors */
  scaleEmbeddings: number[][];
  /** Positional encoding matrix */
  positionalEncoding: number[][];
  /** Fusion gate weights */
  fusionGateWeights: number[];
  /** Fusion gate biases */
  fusionGateBiases: number[];
  /** Attention Q weights per block per head */
  attentionQWeights: number[][][];
  /** Attention K weights per block per head */
  attentionKWeights: number[][][];
  /** Attention V weights per block per head */
  attentionVWeights: number[][][];
  /** Attention output projection weights per block */
  attentionOutWeights: number[][];
  /** Attention output projection biases per block */
  attentionOutBiases: number[][];
  /** FFN first layer weights per block */
  ffnW1: number[][];
  /** FFN first layer biases per block */
  ffnB1: number[][];
  /** FFN second layer weights per block */
  ffnW2: number[][];
  /** FFN second layer biases per block */
  ffnB2: number[][];
  /** Layer norm 1 gamma per block */
  ln1Gamma: number[][];
  /** Layer norm 1 beta per block */
  ln1Beta: number[][];
  /** Layer norm 2 gamma per block */
  ln2Gamma: number[][];
  /** Layer norm 2 beta per block */
  ln2Beta: number[][];
  /** Attention pooling weights */
  poolWeights: number[];
  /** Output layer weights */
  outputWeights: number[];
  /** Output layer biases */
  outputBiases: number[];
  /** Adam update count */
  updateCount: number;
}

/**
 * Online normalization statistics.
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
  /** Number of samples seen */
  count: number;
}

/**
 * Summary of model configuration and state.
 */
export interface ModelSummary {
  /** Whether the model has been initialized */
  isInitialized: boolean;
  /** Input feature dimension */
  inputDimension: number;
  /** Output target dimension */
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
  /** Number of training samples seen */
  sampleCount: number;
  /** Current accuracy estimate */
  accuracy: number;
  /** Whether the model has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

// ============================================================================
// INTERNAL HELPER CLASSES
// ============================================================================

/**
 * Welford's online algorithm for computing running mean and variance.
 * Numerically stable single-pass algorithm.
 *
 * Formulas:
 *   delta = x - mean
 *   mean += delta / n
 *   delta2 = x - mean
 *   M2 += delta * delta2
 *   variance = M2 / (n - 1)
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

  update(x: Float64Array): void {
    this.count++;
    for (let i = 0; i < this.dim; i++) {
      const delta = x[i] - this.mean[i];
      this.mean[i] += delta / this.count;
      const delta2 = x[i] - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  getStd(epsilon: number): Float64Array {
    const std = new Float64Array(this.dim);
    if (this.count > 1) {
      for (let i = 0; i < this.dim; i++) {
        const variance = this.m2[i] / (this.count - 1);
        std[i] = Math.sqrt(variance);
        if (std[i] < epsilon) std[i] = 1;
      }
    } else {
      std.fill(1);
    }
    return std;
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
    for (let i = 0; i < Math.min(data.mean.length, this.dim); i++) {
      this.mean[i] = data.mean[i];
      this.m2[i] = data.m2[i];
    }
    this.count = data.count;
  }
}

/**
 * ADWIN (ADaptive WINdowing) algorithm for concept drift detection.
 *
 * Maintains a sliding window and detects when the distribution of errors
 * changes significantly between two sub-windows.
 *
 * Formula for cut detection:
 *   |mean_left - mean_right| >= sqrt((1/n0 + 1/n1) * ln(4/delta) / 2)
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

  addAndCheck(error: number): boolean {
    // Add to sliding window
    if (this.size < this.capacity) {
      this.values[this.size] = error;
      this.sum += error;
      this.size++;
    } else {
      // Shift window
      this.sum -= this.values[0];
      for (let i = 0; i < this.size - 1; i++) {
        this.values[i] = this.values[i + 1];
      }
      this.values[this.size - 1] = error;
      this.sum += error;
    }

    if (this.size < 10) return false;

    // Check for drift at each possible cut point
    let leftSum = 0;
    for (let cut = 5; cut < this.size - 5; cut++) {
      leftSum += this.values[cut - 1];
      const rightSum = this.sum - leftSum;
      const n0 = cut;
      const n1 = this.size - cut;
      const leftMean = leftSum / n0;
      const rightMean = rightSum / n1;
      const bound = Math.sqrt((1 / n0 + 1 / n1) * Math.log(4 / this.delta) / 2);

      if (Math.abs(leftMean - rightMean) >= bound) {
        // Drift detected - keep only the right portion
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
    for (let i = 0; i < Math.min(data.values.length, this.capacity); i++) {
      this.values[i] = data.values[i];
    }
    this.size = data.size;
    this.sum = data.sum;
  }
}

/**
 * A trainable parameter tensor with Adam optimizer state.
 */
class Param {
  public readonly size: number;
  public data: Float64Array;
  public grad: Float64Array;
  public m: Float64Array; // First moment
  public v: Float64Array; // Second moment

  constructor(size: number) {
    this.size = size;
    this.data = new Float64Array(size);
    this.grad = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  /**
   * Xavier/Glorot uniform initialization.
   * Formula: U(-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut)))
   */
  initXavier(fanIn: number, fanOut: number): void {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    for (let i = 0; i < this.size; i++) {
      this.data[i] = (Math.random() * 2 - 1) * limit;
    }
  }

  initZero(): void {
    this.data.fill(0);
  }

  initOne(): void {
    this.data.fill(1);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /**
   * Adam optimizer update step.
   *
   * Formulas:
   *   g = grad + lambda * data  (with L2 regularization)
   *   m = beta1 * m + (1 - beta1) * g
   *   v = beta2 * v + (1 - beta2) * g^2
   *   m_hat = m / (1 - beta1^t)
   *   v_hat = v / (1 - beta2^t)
   *   data -= lr * m_hat / (sqrt(v_hat) + epsilon)
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
      const g = this.grad[i] + lambda * this.data[i];
      this.m[i] = beta1 * this.m[i] + (1 - beta1) * g;
      this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g;
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
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
// MATH UTILITIES
// ============================================================================

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_COEF = 0.044715;

/**
 * GELU activation function.
 * Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
function gelu(x: number): number {
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

/**
 * Derivative of GELU for backpropagation.
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
 * Numerically stable sigmoid.
 * Formula: 1 / (1 + exp(-x))
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Derivative of sigmoid given sigmoid output.
 * Formula: sig * (1 - sig)
 */
function sigmoidGrad(sig: number): number {
  return sig * (1 - sig);
}

/**
 * Numerically stable softmax.
 */
function softmax(scores: Float64Array, out: Float64Array): void {
  const len = scores.length;
  let max = scores[0];
  for (let i = 1; i < len; i++) {
    if (scores[i] > max) max = scores[i];
  }

  let sum = 0;
  for (let i = 0; i < len; i++) {
    out[i] = Math.exp(scores[i] - max);
    sum += out[i];
  }

  const invSum = 1 / (sum + 1e-10);
  for (let i = 0; i < len; i++) {
    out[i] *= invSum;
  }
}

// ============================================================================
// MAIN CLASS
// ============================================================================

/**
 * FusionTemporalTransformerRegression
 *
 * A neural network for multivariate time series regression with online learning.
 *
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({
 *   numBlocks: 3,
 *   embeddingDim: 64,
 *   numHeads: 8
 * });
 *
 * // Online training
 * for (const batch of dataStream) {
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

  // Normalization components
  private inputWelford: WelfordAccumulator | null = null;
  private outputWelford: WelfordAccumulator | null = null;
  private adwin: ADWINDetector | null = null;

  // Model parameters
  private params: Param[] = [];
  private temporalConvW: Param[] = [];
  private temporalConvB: Param[] = [];
  private scaleEmbed: Param[] = [];
  private posEncoding: Float64Array[] = [];
  private fusionGateW: Param | null = null;
  private fusionGateB: Param | null = null;
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
  private poolW: Param | null = null;
  private outW: Param | null = null;
  private outB: Param | null = null;

  // Forward pass cache for backpropagation
  private fwdCache: Map<string, Float64Array | Float64Array[]> = new Map();
  private lastNormalizedInput: Float64Array[] | null = null;

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
  }

  /**
   * Initialize model parameters based on input/output dimensions.
   */
  private init(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.params = [];

    // Initialize normalization components
    this.inputWelford = new WelfordAccumulator(inputDim);
    this.outputWelford = new WelfordAccumulator(outputDim);
    this.adwin = new ADWINDetector(1000, this.adwinDelta);

    // Sinusoidal positional encoding
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

    // Temporal convolution parameters for each scale
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

    // Learnable scale embeddings
    this.scaleEmbed = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const se = new Param(this.embeddingDim);
      se.initXavier(this.embeddingDim, this.embeddingDim);
      this.scaleEmbed.push(se);
      this.params.push(se);
    }

    // Gated fusion parameters
    const fusionInDim = this.embeddingDim * this.temporalScales.length;
    this.fusionGateW = new Param(fusionInDim * this.temporalScales.length);
    this.fusionGateW.initXavier(fusionInDim, this.temporalScales.length);
    this.params.push(this.fusionGateW);

    this.fusionGateB = new Param(this.temporalScales.length);
    this.fusionGateB.initZero();
    this.params.push(this.fusionGateB);

    // Transformer blocks
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
      // Layer norm 1 (before attention)
      const g1 = new Param(this.embeddingDim);
      g1.initOne();
      this.ln1Gamma.push(g1);
      this.params.push(g1);

      const b1 = new Param(this.embeddingDim);
      b1.initZero();
      this.ln1Beta.push(b1);
      this.params.push(b1);

      // Layer norm 2 (before FFN)
      const g2 = new Param(this.embeddingDim);
      g2.initOne();
      this.ln2Gamma.push(g2);
      this.params.push(g2);

      const b2 = new Param(this.embeddingDim);
      b2.initZero();
      this.ln2Beta.push(b2);
      this.params.push(b2);

      // Multi-head attention weights
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

      // Attention output projection
      const wo = new Param(this.embeddingDim * this.embeddingDim);
      wo.initXavier(this.embeddingDim, this.embeddingDim);
      this.attnWo.push(wo);
      this.params.push(wo);

      const bo = new Param(this.embeddingDim);
      bo.initZero();
      this.attnBo.push(bo);
      this.params.push(bo);

      // FFN weights
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

    // Attention pooling
    this.poolW = new Param(this.embeddingDim);
    this.poolW.initXavier(this.embeddingDim, 1);
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
   * Apply layer normalization.
   * Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
   */
  private layerNorm(
    x: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    cacheKey: string | null = null,
  ): Float64Array {
    const len = x.length;
    let mean = 0;
    for (let i = 0; i < len; i++) {
      mean += x[i];
    }
    mean /= len;

    let variance = 0;
    for (let i = 0; i < len; i++) {
      const d = x[i] - mean;
      variance += d * d;
    }
    variance /= len;

    const invStd = 1 / Math.sqrt(variance + this.epsilon);
    const out = new Float64Array(len);
    const normalized = new Float64Array(len);

    for (let i = 0; i < len; i++) {
      normalized[i] = (x[i] - mean) * invStd;
      out[i] = gamma[i] * normalized[i] + beta[i];
    }

    if (cacheKey) {
      this.fwdCache.set(cacheKey + "_norm", normalized);
      this.fwdCache.set(cacheKey + "_invStd", new Float64Array([invStd]));
    }

    return out;
  }

  /**
   * Apply temporal 1D convolution with GELU activation.
   */
  private temporalConv(
    input: Float64Array[],
    scaleIdx: number,
    stride: number,
  ): Float64Array[] {
    const seqLen = input.length;
    const outLen = Math.max(
      1,
      Math.floor((seqLen - this.temporalKernelSize) / stride) + 1,
    );
    const W = this.temporalConvW[scaleIdx].data;
    const B = this.temporalConvB[scaleIdx].data;

    const output: Float64Array[] = [];
    const preActivations: Float64Array[] = [];

    for (let t = 0; t < outLen; t++) {
      const pre = new Float64Array(this.embeddingDim);
      const out = new Float64Array(this.embeddingDim);
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
        out[o] = gelu(sum);
      }

      preActivations.push(pre);
      output.push(out);
    }

    this.fwdCache.set(`conv_pre_${scaleIdx}`, preActivations);
    return output;
  }

  /**
   * Multi-head self-attention.
   */
  private multiHeadAttention(
    input: Float64Array[],
    blockIdx: number,
  ): Float64Array[] {
    const seqLen = input.length;
    const scale = 1 / Math.sqrt(this.headDim);

    const allHeadOutputs: Float64Array[][] = [];
    const allQ: Float64Array[][] = [];
    const allK: Float64Array[][] = [];
    const allV: Float64Array[][] = [];
    const allWeights: Float64Array[][] = [];

    // Compute attention for each head
    for (let h = 0; h < this.numHeads; h++) {
      const Wq = this.attnWq[blockIdx][h].data;
      const Wk = this.attnWk[blockIdx][h].data;
      const Wv = this.attnWv[blockIdx][h].data;

      const Q: Float64Array[] = [];
      const K: Float64Array[] = [];
      const V: Float64Array[] = [];

      // Linear projections
      for (let t = 0; t < seqLen; t++) {
        const q = new Float64Array(this.headDim);
        const k = new Float64Array(this.headDim);
        const v = new Float64Array(this.headDim);

        for (let d = 0; d < this.headDim; d++) {
          let qSum = 0, kSum = 0, vSum = 0;
          for (let e = 0; e < this.embeddingDim; e++) {
            const idx = d * this.embeddingDim + e;
            qSum += input[t][e] * Wq[idx];
            kSum += input[t][e] * Wk[idx];
            vSum += input[t][e] * Wv[idx];
          }
          q[d] = qSum;
          k[d] = kSum;
          v[d] = vSum;
        }

        Q.push(q);
        K.push(k);
        V.push(v);
      }

      allQ.push(Q);
      allK.push(K);
      allV.push(V);

      // Attention scores and weights
      const headWeights: Float64Array[] = [];
      const headOutput: Float64Array[] = [];

      for (let i = 0; i < seqLen; i++) {
        const scores = new Float64Array(seqLen);
        for (let j = 0; j < seqLen; j++) {
          let s = 0;
          for (let d = 0; d < this.headDim; d++) {
            s += Q[i][d] * K[j][d];
          }
          scores[j] = s * scale;
        }

        const weights = new Float64Array(seqLen);
        softmax(scores, weights);
        headWeights.push(weights);

        // Weighted sum of values
        const out = new Float64Array(this.headDim);
        for (let d = 0; d < this.headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += weights[j] * V[j][d];
          }
          out[d] = sum;
        }
        headOutput.push(out);
      }

      allWeights.push(headWeights);
      allHeadOutputs.push(headOutput);
    }

    // Cache for backprop
    this.fwdCache.set(`attn_Q_${blockIdx}`, allQ.flat());
    this.fwdCache.set(`attn_K_${blockIdx}`, allK.flat());
    this.fwdCache.set(`attn_V_${blockIdx}`, allV.flat());
    this.fwdCache.set(`attn_W_${blockIdx}`, allWeights.flat());

    // Concatenate heads and project
    const Wo = this.attnWo[blockIdx].data;
    const Bo = this.attnBo[blockIdx].data;
    const output: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      // Concatenate all heads
      const concat = new Float64Array(this.embeddingDim);
      for (let h = 0; h < this.numHeads; h++) {
        for (let d = 0; d < this.headDim; d++) {
          concat[h * this.headDim + d] = allHeadOutputs[h][t][d];
        }
      }

      // Output projection
      const out = new Float64Array(this.embeddingDim);
      for (let o = 0; o < this.embeddingDim; o++) {
        let sum = Bo[o];
        for (let c = 0; c < this.embeddingDim; c++) {
          sum += concat[c] * Wo[o * this.embeddingDim + c];
        }
        out[o] = sum;
      }
      output.push(out);
    }

    return output;
  }

  /**
   * Feed-forward network with GELU activation.
   * Formula: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
   */
  private feedForward(
    input: Float64Array[],
    blockIdx: number,
  ): {
    output: Float64Array[];
    preGelu: Float64Array[];
    hidden: Float64Array[];
  } {
    const W1 = this.ffnW1[blockIdx].data;
    const B1 = this.ffnB1[blockIdx].data;
    const W2 = this.ffnW2[blockIdx].data;
    const B2 = this.ffnB2[blockIdx].data;

    const output: Float64Array[] = [];
    const preGelu: Float64Array[] = [];
    const hidden: Float64Array[] = [];

    for (let t = 0; t < input.length; t++) {
      const x = input[t];

      // First layer
      const pre = new Float64Array(this.ffnHiddenDim);
      const h = new Float64Array(this.ffnHiddenDim);
      for (let i = 0; i < this.ffnHiddenDim; i++) {
        let sum = B1[i];
        for (let j = 0; j < this.embeddingDim; j++) {
          sum += x[j] * W1[i * this.embeddingDim + j];
        }
        pre[i] = sum;
        h[i] = gelu(sum);
      }
      preGelu.push(pre);
      hidden.push(h);

      // Second layer
      const out = new Float64Array(this.embeddingDim);
      for (let i = 0; i < this.embeddingDim; i++) {
        let sum = B2[i];
        for (let j = 0; j < this.ffnHiddenDim; j++) {
          sum += h[j] * W2[i * this.ffnHiddenDim + j];
        }
        out[i] = sum;
      }
      output.push(out);
    }

    return { output, preGelu, hidden };
  }

  /**
   * Full forward pass through the network.
   */
  private forward(normalizedInput: Float64Array[]): Float64Array {
    this.fwdCache.clear();
    this.fwdCache.set("input", normalizedInput);

    // Step 1: Multi-scale temporal convolutions
    const scaleOutputs: Float64Array[][] = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const stride = this.temporalScales[s];
      const convOut = this.temporalConv(normalizedInput, s, stride);

      // Add positional encoding and scale embedding
      const embedded: Float64Array[] = [];
      for (let t = 0; t < convOut.length; t++) {
        const emb = new Float64Array(this.embeddingDim);
        const posIdx = Math.min(t, this.maxSequenceLength - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          emb[e] = convOut[t][e] + this.posEncoding[posIdx][e] +
            this.scaleEmbed[s].data[e];
        }
        embedded.push(emb);
      }
      scaleOutputs.push(embedded);
    }

    // Step 2: Gated fusion
    let minLen = scaleOutputs[0].length;
    for (let s = 1; s < scaleOutputs.length; s++) {
      if (scaleOutputs[s].length < minLen) {
        minLen = scaleOutputs[s].length;
      }
    }
    if (minLen === 0) minLen = 1;

    const fused: Float64Array[] = [];
    const fusionGates: Float64Array[] = [];
    const fusionGateW = this.fusionGateW!.data;
    const fusionGateB = this.fusionGateB!.data;
    const numScales = this.temporalScales.length;
    const fusionInDim = this.embeddingDim * numScales;

    for (let t = 0; t < minLen; t++) {
      // Concatenate all scales at this time step
      const concat = new Float64Array(fusionInDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, scaleOutputs[s].length - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          concat[s * this.embeddingDim + e] = scaleOutputs[s][idx][e];
        }
      }

      // Compute gate values
      const gates = new Float64Array(numScales);
      for (let g = 0; g < numScales; g++) {
        let sum = fusionGateB[g];
        for (let i = 0; i < fusionInDim; i++) {
          sum += concat[i] * fusionGateW[g * fusionInDim + i];
        }
        gates[g] = sigmoid(sum);
      }
      fusionGates.push(gates);

      // Gated weighted sum
      const fusedT = new Float64Array(this.embeddingDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, scaleOutputs[s].length - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          fusedT[e] += gates[s] * scaleOutputs[s][idx][e];
        }
      }
      fused.push(fusedT);
    }

    this.fwdCache.set("fusionGates", fusionGates);
    this.fwdCache.set("scaleOutputs", scaleOutputs.flat());

    // Step 3: Transformer blocks
    let current = fused;

    for (let b = 0; b < this.numBlocks; b++) {
      this.fwdCache.set(
        `block_${b}_input`,
        current.map((x) => new Float64Array(x)),
      );

      // Pre-norm attention
      const ln1Out: Float64Array[] = [];
      for (let t = 0; t < current.length; t++) {
        ln1Out.push(this.layerNorm(
          current[t],
          this.ln1Gamma[b].data,
          this.ln1Beta[b].data,
          `ln1_${b}_${t}`,
        ));
      }
      this.fwdCache.set(`ln1_out_${b}`, ln1Out);

      const attnOut = this.multiHeadAttention(ln1Out, b);

      // Residual connection
      const residual1: Float64Array[] = [];
      for (let t = 0; t < current.length; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = current[t][e] + attnOut[t][e];
        }
        residual1.push(r);
      }
      this.fwdCache.set(`residual1_${b}`, residual1);

      // Pre-norm FFN
      const ln2Out: Float64Array[] = [];
      for (let t = 0; t < residual1.length; t++) {
        ln2Out.push(this.layerNorm(
          residual1[t],
          this.ln2Gamma[b].data,
          this.ln2Beta[b].data,
          `ln2_${b}_${t}`,
        ));
      }
      this.fwdCache.set(`ln2_out_${b}`, ln2Out);

      const { output: ffnOut, preGelu, hidden } = this.feedForward(ln2Out, b);
      this.fwdCache.set(`ffn_pre_${b}`, preGelu);
      this.fwdCache.set(`ffn_hidden_${b}`, hidden);

      // Residual connection
      current = [];
      for (let t = 0; t < residual1.length; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = residual1[t][e] + ffnOut[t][e];
        }
        current.push(r);
      }
    }

    this.fwdCache.set("final_hidden", current);

    // Step 4: Attention pooling
    const poolWeights = new Float64Array(current.length);
    const poolScores = new Float64Array(current.length);
    const poolWData = this.poolW!.data;

    for (let t = 0; t < current.length; t++) {
      let score = 0;
      for (let e = 0; e < this.embeddingDim; e++) {
        score += current[t][e] * poolWData[e];
      }
      poolScores[t] = score;
    }
    softmax(poolScores, poolWeights);
    this.fwdCache.set("poolWeights", poolWeights);

    // Weighted aggregation
    const aggregated = new Float64Array(this.embeddingDim);
    for (let t = 0; t < current.length; t++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        aggregated[e] += poolWeights[t] * current[t][e];
      }
    }
    this.fwdCache.set("aggregated", aggregated);

    // Step 5: Output projection
    const outWData = this.outW!.data;
    const outBData = this.outB!.data;
    const output = new Float64Array(this.outputDim);

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
   * Backward pass to compute gradients.
   */
  private backward(
    target: Float64Array,
    predicted: Float64Array,
    sampleWeight: number,
  ): number {
    // Zero all gradients
    for (const p of this.params) {
      p.zeroGrad();
    }

    const finalHidden = this.fwdCache.get("final_hidden") as Float64Array[];
    const seqLen = finalHidden.length;

    // Output layer gradients
    // dL/dOutput = (predicted - target) * sampleWeight / outputDim
    const dOutput = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      dOutput[o] = (predicted[o] - target[o]) * sampleWeight / this.outputDim;
    }

    const aggregated = this.fwdCache.get("aggregated") as Float64Array;
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

    // Backprop through attention pooling
    const poolWeights = this.fwdCache.get("poolWeights") as Float64Array;
    const dFinalHidden: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const dh = new Float64Array(this.embeddingDim);
      for (let e = 0; e < this.embeddingDim; e++) {
        dh[e] = poolWeights[t] * dAggregated[e];
      }
      dFinalHidden.push(dh);
    }

    // Gradient for poolW through softmax
    for (let t = 0; t < seqLen; t++) {
      let dotProd = 0;
      for (let e = 0; e < this.embeddingDim; e++) {
        dotProd += dAggregated[e] * finalHidden[t][e];
      }
      for (let e = 0; e < this.embeddingDim; e++) {
        this.poolW!.grad[e] += dotProd * poolWeights[t] * finalHidden[t][e];
        dFinalHidden[t][e] += dotProd * poolWeights[t] * this.poolW!.data[e];
      }
    }

    // Backprop through transformer blocks
    let dCurrent = dFinalHidden;

    for (let b = this.numBlocks - 1; b >= 0; b--) {
      const ln2Out = this.fwdCache.get(`ln2_out_${b}`) as Float64Array[];
      const ffnHidden = this.fwdCache.get(`ffn_hidden_${b}`) as Float64Array[];
      const ffnPreGelu = this.fwdCache.get(`ffn_pre_${b}`) as Float64Array[];
      const ln1Out = this.fwdCache.get(`ln1_out_${b}`) as Float64Array[];
      const curSeqLen = dCurrent.length;

      // Through residual2 = residual1 + ffnOut
      const dResidual1 = dCurrent.map((d) => new Float64Array(d));
      const dFFNOut = dCurrent.map((d) => new Float64Array(d));

      // Backprop through FFN
      for (let t = 0; t < curSeqLen; t++) {
        const dHidden = new Float64Array(this.ffnHiddenDim);

        // Through W2
        for (let o = 0; o < this.embeddingDim; o++) {
          this.ffnB2[b].grad[o] += dFFNOut[t][o];
          for (let h = 0; h < this.ffnHiddenDim; h++) {
            this.ffnW2[b].grad[o * this.ffnHiddenDim + h] += dFFNOut[t][o] *
              ffnHidden[t][h];
            dHidden[h] += dFFNOut[t][o] *
              this.ffnW2[b].data[o * this.ffnHiddenDim + h];
          }
        }

        // Through GELU
        const dPreGelu = new Float64Array(this.ffnHiddenDim);
        for (let h = 0; h < this.ffnHiddenDim; h++) {
          dPreGelu[h] = dHidden[h] * geluGrad(ffnPreGelu[t][h]);
        }

        // Through W1
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

        // Through layer norm 2
        const norm2 = this.fwdCache.get(`ln2_${b}_${t}_norm`) as Float64Array;
        if (norm2) {
          for (let e = 0; e < this.embeddingDim; e++) {
            this.ln2Gamma[b].grad[e] += dLn2[e] * norm2[e];
            this.ln2Beta[b].grad[e] += dLn2[e];
            dResidual1[t][e] += dLn2[e] * this.ln2Gamma[b].data[e];
          }
        }
      }

      // Backprop through attention (simplified)
      const dAttnOut = dResidual1.map((d) => new Float64Array(d));
      const dBlockInput = dResidual1.map((d) => new Float64Array(d));

      for (let t = 0; t < curSeqLen; t++) {
        // Through attention output projection
        const dConcat = new Float64Array(this.embeddingDim);
        for (let o = 0; o < this.embeddingDim; o++) {
          this.attnBo[b].grad[o] += dAttnOut[t][o];
          for (let c = 0; c < this.embeddingDim; c++) {
            this.attnWo[b].grad[o * this.embeddingDim + c] += dAttnOut[t][o] *
              ln1Out[t][c];
            dConcat[c] += dAttnOut[t][o] *
              this.attnWo[b].data[o * this.embeddingDim + c];
          }
        }

        // Distribute to Q, K, V weights (simplified)
        const gradScale = 0.1;
        for (let h = 0; h < this.numHeads; h++) {
          for (let d = 0; d < this.headDim; d++) {
            const cIdx = h * this.headDim + d;
            for (let e = 0; e < this.embeddingDim; e++) {
              const wIdx = d * this.embeddingDim + e;
              const grad = dConcat[cIdx] * ln1Out[t][e] * gradScale;
              this.attnWq[b][h].grad[wIdx] += grad;
              this.attnWk[b][h].grad[wIdx] += grad;
              this.attnWv[b][h].grad[wIdx] += grad;
            }
          }
        }

        // Through layer norm 1
        const norm1 = this.fwdCache.get(`ln1_${b}_${t}_norm`) as Float64Array;
        if (norm1) {
          for (let e = 0; e < this.embeddingDim; e++) {
            this.ln1Gamma[b].grad[e] += dAttnOut[t][e] * norm1[e] * gradScale;
            this.ln1Beta[b].grad[e] += dAttnOut[t][e] * gradScale;
            dBlockInput[t][e] += dAttnOut[t][e] * this.ln1Gamma[b].data[e] *
              gradScale;
          }
        }
      }

      dCurrent = dBlockInput;
    }

    // Backprop through fusion and temporal convolutions
    const fusionGates = this.fwdCache.get("fusionGates") as Float64Array[];
    const input = this.fwdCache.get("input") as Float64Array[];

    if (fusionGates) {
      // Fusion gate gradients
      for (let t = 0; t < Math.min(fusionGates.length, dCurrent.length); t++) {
        for (let s = 0; s < this.temporalScales.length; s++) {
          const gate = fusionGates[t][s];
          const dGate = sigmoidGrad(gate);
          for (let e = 0; e < this.embeddingDim; e++) {
            this.fusionGateB!.grad[s] += dCurrent[t][e] * dGate * 0.01;
          }
        }
      }

      // Scale embedding gradients
      for (let s = 0; s < this.temporalScales.length; s++) {
        for (let e = 0; e < this.embeddingDim; e++) {
          let grad = 0;
          for (
            let t = 0;
            t < Math.min(dCurrent.length, fusionGates.length);
            t++
          ) {
            grad += dCurrent[t][e] * fusionGates[t][s] * 0.01;
          }
          this.scaleEmbed[s].grad[e] += grad;
        }
      }

      // Temporal convolution gradients
      for (let s = 0; s < this.temporalScales.length; s++) {
        const preAct = this.fwdCache.get(`conv_pre_${s}`) as Float64Array[];
        if (!preAct) continue;

        const stride = this.temporalScales[s];

        for (let t = 0; t < Math.min(preAct.length, dCurrent.length); t++) {
          const gate = fusionGates[Math.min(t, fusionGates.length - 1)][s];
          const startPos = t * stride;

          for (let o = 0; o < this.embeddingDim; o++) {
            const dPre = dCurrent[t][o] * geluGrad(preAct[t][o]) * gate * 0.01;
            this.temporalConvB[s].grad[o] += dPre;

            for (let k = 0; k < this.temporalKernelSize; k++) {
              const pos = startPos + k;
              if (pos < input.length) {
                for (let i = 0; i < this.inputDim; i++) {
                  const wIdx = o * (this.inputDim * this.temporalKernelSize) +
                    i * this.temporalKernelSize + k;
                  this.temporalConvW[s].grad[wIdx] += dPre * input[pos][i];
                }
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
    return Math.sqrt(gradNorm);
  }

  /**
   * Get effective learning rate with cosine warmup schedule.
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
   * Perform Adam optimizer step on all parameters.
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
   * Performs one step of incremental online learning.
   *
   * @param data - Training data with xCoordinates (input sequences) and yCoordinates (targets)
   * @returns FitResult with training metrics
   * @throws Error if input data is invalid
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4], [5, 6]],  // 3-step sequence, 2 features
   *   yCoordinates: [[7], [8], [9]]             // Targets for each step
   * });
   * console.log(`Loss: ${result.loss}`);
   * ```
   */
  public fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

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

    // Initialize on first call
    if (!this.initialized) {
      this.init(inputDim, outputDim);
    }

    // Convert to Float64Array and update normalization stats
    const xInput: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      const x = new Float64Array(xCoordinates[t]);
      xInput.push(x);
      this.inputWelford!.update(x);
    }

    const yTarget = new Float64Array(yCoordinates[seqLen - 1]);
    this.outputWelford!.update(yTarget);

    // Normalize inputs using running statistics
    const inputMean = this.inputWelford!.mean;
    const inputStd = this.inputWelford!.getStd(this.epsilon);
    const normalizedInput: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const norm = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        norm[i] = (xInput[t][i] - inputMean[i]) / (inputStd[i] + this.epsilon);
      }
      normalizedInput.push(norm);
    }

    this.lastNormalizedInput = normalizedInput;

    // Forward pass
    const predicted = this.forward(normalizedInput);

    // Denormalize prediction for loss calculation
    const outputStd = this.outputWelford!.getStd(this.epsilon);
    const outputMean = this.outputWelford!.mean;
    const denormPred = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      denormPred[o] = predicted[o] * outputStd[o] + outputMean[o];
    }

    // Compute MSE loss
    let mseLoss = 0;
    for (let o = 0; o < this.outputDim; o++) {
      const diff = yTarget[o] - denormPred[o];
      mseLoss += diff * diff;
    }
    mseLoss /= 2 * this.outputDim;

    // L2 regularization loss
    let l2Loss = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.size; i++) {
        l2Loss += p.data[i] * p.data[i];
      }
    }
    l2Loss *= this.regularizationStrength / 2;

    const totalLoss = mseLoss + l2Loss;

    // Outlier detection
    let isOutlier = false;
    let sampleWeight = 1.0;

    if (this.outputWelford!.count > 10) {
      let residualNorm = 0;
      for (let o = 0; o < this.outputDim; o++) {
        const zScore = (yTarget[o] - denormPred[o]) /
          (outputStd[o] + this.epsilon);
        residualNorm += zScore * zScore;
      }
      residualNorm = Math.sqrt(residualNorm);

      if (residualNorm > this.outlierThreshold) {
        isOutlier = true;
        sampleWeight = 0.1;
      }
    }

    // Drift detection
    const driftDetected = this.adwin!.addAndCheck(mseLoss);
    if (driftDetected) {
      this.driftCount++;
    }

    // Normalize target for backprop
    const normalizedTarget = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      normalizedTarget[o] = (yTarget[o] - outputMean[o]) /
        (outputStd[o] + this.epsilon);
    }

    // Backward pass
    const gradNorm = this.backward(normalizedTarget, predicted, sampleWeight);

    // Optimizer step
    this.optimizerStep();

    // Update tracking
    this.sampleCount++;
    this.totalLoss += totalLoss;

    // Convergence check
    const lossDiff = Math.abs(this.prevLoss - totalLoss);
    this.converged = lossDiff < this.convergenceThreshold &&
      this.sampleCount > 100;
    this.prevLoss = totalLoss;

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
   * Generates predictions for future time steps.
   *
   * @param futureSteps - Number of future steps to predict
   * @returns PredictionResult with predictions and uncertainty bounds
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
    if (!this.initialized || !this.lastNormalizedInput) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const outputStd = this.outputWelford!.getStd(this.epsilon);
    const outputMean = this.outputWelford!.mean;
    const zScore95 = 1.96;

    let currentInput = this.lastNormalizedInput;

    for (let step = 0; step < futureSteps; step++) {
      const predicted = this.forward(currentInput);

      const denormalized: number[] = [];
      const standardError: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];

      for (let o = 0; o < this.outputDim; o++) {
        const denorm = predicted[o] * outputStd[o] + outputMean[o];
        const se = outputStd[o] / Math.sqrt(Math.max(this.sampleCount, 1));

        denormalized.push(denorm);
        standardError.push(se);
        lowerBound.push(denorm - zScore95 * se);
        upperBound.push(denorm + zScore95 * se);
      }

      predictions.push({
        predicted: denormalized,
        lowerBound,
        upperBound,
        standardError,
      });

      // Autoregressive: shift input window
      if (currentInput.length > 1 && this.inputDim === this.outputDim) {
        const newInput: Float64Array[] = [];
        for (let t = 1; t < currentInput.length; t++) {
          newInput.push(currentInput[t]);
        }
        newInput.push(predicted);
        currentInput = newInput;
      }
    }

    const avgLoss = this.sampleCount > 0
      ? this.totalLoss / this.sampleCount
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
   * Returns a summary of the model's configuration and current state.
   */
  public getModelSummary(): ModelSummary {
    const avgLoss = this.sampleCount > 0
      ? this.totalLoss / this.sampleCount
      : 0;
    const accuracy = 1 / (1 + avgLoss);
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
      accuracy,
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLR(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Returns all model weights for inspection or serialization.
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
      updateCount: this.updateCount,
    };
  }

  /**
   * Returns the current normalization statistics.
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
   * Resets the model to its initial state.
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
    this.params = [];
    this.fwdCache.clear();
    this.lastNormalizedInput = null;
    this.inputWelford = null;
    this.outputWelford = null;
    this.adwin = null;
  }

  /**
   * Serializes the model to a JSON string.
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
    };

    return JSON.stringify(state);
  }

  /**
   * Loads the model from a JSON string.
   */
  public load(jsonString: string): void {
    const state = JSON.parse(jsonString);

    if (!state.state.initialized) {
      this.reset();
      return;
    }

    // Re-initialize with saved dimensions
    this.init(state.state.inputDim, state.state.outputDim);

    // Restore Welford accumulators
    if (state.welford) {
      this.inputWelford!.deserialize(state.welford.input);
      this.outputWelford!.deserialize(state.welford.output);
    }

    // Restore ADWIN
    if (state.adwin) {
      this.adwin!.deserialize(state.adwin);
    }

    // Restore parameters
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

    // Restore state
    this.sampleCount = state.state.sampleCount;
    this.updateCount = state.state.updateCount;
    this.totalLoss = state.state.totalLoss;
    this.prevLoss = state.state.prevLoss;
    this.converged = state.state.converged;
    this.driftCount = state.state.driftCount;
  }
}

export default FusionTemporalTransformerRegression;
