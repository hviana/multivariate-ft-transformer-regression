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
// HELPER CLASSES
// ============================================================================

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
        std[i] = Math.sqrt(variance + epsilon);
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

class Param {
  public readonly size: number;
  public data: Float64Array;
  public grad: Float64Array;
  public m: Float64Array;
  public v: Float64Array;

  constructor(size: number) {
    this.size = size;
    this.data = new Float64Array(size);
    this.grad = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  initXavier(fanIn: number, fanOut: number): void {
    const std = Math.sqrt(2.0 / (fanIn + fanOut));
    for (let i = 0; i < this.size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      this.data[i] = std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) *
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
      this.data[i] = (Math.random() - 0.5) * scale;
    }
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

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

function gelu(x: number): number {
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function geluGrad(x: number): number {
  const x3 = x * x * x;
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
  const tanhInner = Math.tanh(inner);
  const sech2 = 1 - tanhInner * tanhInner;
  const innerGrad = SQRT_2_OVER_PI * (1 + 3 * GELU_COEF * x * x);
  return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerGrad;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function sigmoidGrad(sig: number): number {
  return sig * (1 - sig);
}

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

export class FusionTemporalTransformerRegression {
  // Configuration
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

  // State
  private initialized: boolean = false;
  private inputDim: number = 0;
  private outputDim: number = 0;
  private sampleCount: number = 0;
  private updateCount: number = 0;
  private totalLoss: number = 0;
  private prevLoss: number = Infinity;
  private converged: boolean = false;
  private driftCount: number = 0;
  private recentLosses: number[] = [];
  private predictionVariance: Float64Array | null = null;

  // Normalization
  private inputWelford: WelfordAccumulator | null = null;
  private outputWelford: WelfordAccumulator | null = null;
  private adwin: ADWINDetector | null = null;

  // Parameters
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

  // Cache for backprop - using any to allow nested array types
  private cache: Map<string, any> = new Map();
  private lastRawInput: Float64Array[] | null = null;

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
    this.regularizationStrength = config.regularizationStrength ?? 1e-5;
    this.convergenceThreshold = config.convergenceThreshold ?? 1e-6;
    this.outlierThreshold = config.outlierThreshold ?? 3.0;
    this.adwinDelta = config.adwinDelta ?? 0.002;
    this.temporalScales = config.temporalScales ?? [1, 2, 4];
    this.temporalKernelSize = config.temporalKernelSize ?? 3;
    this.maxSequenceLength = config.maxSequenceLength ?? 512;
    this.gradientClipNorm = config.gradientClipNorm ?? 1.0;
  }

  private init(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.params = [];

    this.inputWelford = new WelfordAccumulator(inputDim);
    this.outputWelford = new WelfordAccumulator(outputDim);
    this.adwin = new ADWINDetector(1000, this.adwinDelta);
    this.predictionVariance = new Float64Array(outputDim);

    // Positional encoding
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

    // Temporal convolutions
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

    // Scale embeddings
    this.scaleEmbed = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const se = new Param(this.embeddingDim);
      se.initSmall(0.02);
      this.scaleEmbed.push(se);
      this.params.push(se);
    }

    // Fusion gate
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
      const g1 = new Param(this.embeddingDim);
      g1.initOne();
      this.ln1Gamma.push(g1);
      this.params.push(g1);

      const b1 = new Param(this.embeddingDim);
      b1.initZero();
      this.ln1Beta.push(b1);
      this.params.push(b1);

      const g2 = new Param(this.embeddingDim);
      g2.initOne();
      this.ln2Gamma.push(g2);
      this.params.push(g2);

      const b2 = new Param(this.embeddingDim);
      b2.initZero();
      this.ln2Beta.push(b2);
      this.params.push(b2);

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

      const wo = new Param(this.embeddingDim * this.embeddingDim);
      wo.initXavier(this.embeddingDim, this.embeddingDim);
      this.attnWo.push(wo);
      this.params.push(wo);

      const bo = new Param(this.embeddingDim);
      bo.initZero();
      this.attnBo.push(bo);
      this.params.push(bo);

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

    this.poolW = new Param(this.embeddingDim);
    this.poolW.initXavier(this.embeddingDim, 1);
    this.params.push(this.poolW);

    this.outW = new Param(outputDim * this.embeddingDim);
    this.outW.initSmall(0.01); // Small initialization for output layer
    this.params.push(this.outW);

    this.outB = new Param(outputDim);
    this.outB.initZero();
    this.params.push(this.outB);

    this.initialized = true;
  }

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
      this.cache.set(cacheKey + "_norm", normalized);
      this.cache.set(cacheKey + "_mean", new Float64Array([mean]));
      this.cache.set(cacheKey + "_invStd", new Float64Array([invStd]));
      this.cache.set(cacheKey + "_input", new Float64Array(x));
    }

    return out;
  }

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

    this.cache.set(`conv_pre_${scaleIdx}`, preActivations);
    this.cache.set(`conv_out_${scaleIdx}`, output);
    return output;
  }

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

    for (let h = 0; h < this.numHeads; h++) {
      const Wq = this.attnWq[blockIdx][h].data;
      const Wk = this.attnWk[blockIdx][h].data;
      const Wv = this.attnWv[blockIdx][h].data;

      const Q: Float64Array[] = [];
      const K: Float64Array[] = [];
      const V: Float64Array[] = [];

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

    this.cache.set(`attn_Q_${blockIdx}`, allQ);
    this.cache.set(`attn_K_${blockIdx}`, allK);
    this.cache.set(`attn_V_${blockIdx}`, allV);
    this.cache.set(`attn_W_${blockIdx}`, allWeights);
    this.cache.set(`attn_headOut_${blockIdx}`, allHeadOutputs);

    const Wo = this.attnWo[blockIdx].data;
    const Bo = this.attnBo[blockIdx].data;
    const output: Float64Array[] = [];
    const concats: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const concat = new Float64Array(this.embeddingDim);
      for (let h = 0; h < this.numHeads; h++) {
        for (let d = 0; d < this.headDim; d++) {
          concat[h * this.headDim + d] = allHeadOutputs[h][t][d];
        }
      }
      concats.push(concat);

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

    this.cache.set(`attn_concat_${blockIdx}`, concats);
    return output;
  }

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

  private forward(normalizedInput: Float64Array[]): Float64Array {
    this.cache.clear();
    this.cache.set("input", normalizedInput);

    // Multi-scale temporal convolutions
    const scaleOutputs: Float64Array[][] = [];
    for (let s = 0; s < this.temporalScales.length; s++) {
      const stride = this.temporalScales[s];
      const convOut = this.temporalConv(normalizedInput, s, stride);

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

    this.cache.set("scaleOutputs", scaleOutputs);

    // Gated fusion
    let minLen = scaleOutputs[0].length;
    for (let s = 1; s < scaleOutputs.length; s++) {
      if (scaleOutputs[s].length < minLen) {
        minLen = scaleOutputs[s].length;
      }
    }
    if (minLen === 0) minLen = 1;

    const fused: Float64Array[] = [];
    const fusionGates: Float64Array[] = [];
    const fusionConcats: Float64Array[] = [];
    const fusionGateW = this.fusionGateW!.data;
    const fusionGateB = this.fusionGateB!.data;
    const numScales = this.temporalScales.length;
    const fusionInDim = this.embeddingDim * numScales;

    for (let t = 0; t < minLen; t++) {
      const concat = new Float64Array(fusionInDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, scaleOutputs[s].length - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          concat[s * this.embeddingDim + e] = scaleOutputs[s][idx][e];
        }
      }
      fusionConcats.push(concat);

      const gatesPreSig = new Float64Array(numScales);
      const gates = new Float64Array(numScales);
      for (let g = 0; g < numScales; g++) {
        let sum = fusionGateB[g];
        for (let i = 0; i < fusionInDim; i++) {
          sum += concat[i] * fusionGateW[g * fusionInDim + i];
        }
        gatesPreSig[g] = sum;
        gates[g] = sigmoid(sum);
      }
      fusionGates.push(gates);

      const fusedT = new Float64Array(this.embeddingDim);
      for (let s = 0; s < numScales; s++) {
        const idx = Math.min(t, scaleOutputs[s].length - 1);
        for (let e = 0; e < this.embeddingDim; e++) {
          fusedT[e] += gates[s] * scaleOutputs[s][idx][e];
        }
      }
      fused.push(fusedT);
    }

    this.cache.set("fusionGates", fusionGates);
    this.cache.set("fusionConcats", fusionConcats);
    this.cache.set("fused", fused);

    // Transformer blocks
    let current = fused;

    for (let b = 0; b < this.numBlocks; b++) {
      this.cache.set(
        `block_${b}_input`,
        current.map((x) => new Float64Array(x)),
      );

      const ln1Out: Float64Array[] = [];
      for (let t = 0; t < current.length; t++) {
        ln1Out.push(this.layerNorm(
          current[t],
          this.ln1Gamma[b].data,
          this.ln1Beta[b].data,
          `ln1_${b}_${t}`,
        ));
      }
      this.cache.set(`ln1_out_${b}`, ln1Out);

      const attnOut = this.multiHeadAttention(ln1Out, b);
      this.cache.set(`attn_out_${b}`, attnOut);

      const residual1: Float64Array[] = [];
      for (let t = 0; t < current.length; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = current[t][e] + attnOut[t][e];
        }
        residual1.push(r);
      }
      this.cache.set(`residual1_${b}`, residual1);

      const ln2Out: Float64Array[] = [];
      for (let t = 0; t < residual1.length; t++) {
        ln2Out.push(this.layerNorm(
          residual1[t],
          this.ln2Gamma[b].data,
          this.ln2Beta[b].data,
          `ln2_${b}_${t}`,
        ));
      }
      this.cache.set(`ln2_out_${b}`, ln2Out);

      const { output: ffnOut, preGelu, hidden } = this.feedForward(ln2Out, b);
      this.cache.set(`ffn_pre_${b}`, preGelu);
      this.cache.set(`ffn_hidden_${b}`, hidden);
      this.cache.set(`ffn_out_${b}`, ffnOut);

      current = [];
      for (let t = 0; t < residual1.length; t++) {
        const r = new Float64Array(this.embeddingDim);
        for (let e = 0; e < this.embeddingDim; e++) {
          r[e] = residual1[t][e] + ffnOut[t][e];
        }
        current.push(r);
      }
    }

    this.cache.set("final_hidden", current);

    // Attention pooling
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
    this.cache.set("poolWeights", poolWeights);
    this.cache.set("poolScores", poolScores);

    const aggregated = new Float64Array(this.embeddingDim);
    for (let t = 0; t < current.length; t++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        aggregated[e] += poolWeights[t] * current[t][e];
      }
    }
    this.cache.set("aggregated", aggregated);

    // Output projection
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

    this.cache.set("output", output);
    return output;
  }

  private backward(
    target: Float64Array,
    predicted: Float64Array,
    sampleWeight: number,
  ): number {
    for (const p of this.params) {
      p.zeroGrad();
    }

    const finalHidden = this.cache.get("final_hidden") as Float64Array[];
    const seqLen = finalHidden.length;

    // MSE gradient: dL/dy = (y - t) / n
    const dOutput = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      dOutput[o] = (predicted[o] - target[o]) * sampleWeight / this.outputDim;
    }

    // Output layer
    const aggregated = this.cache.get("aggregated") as Float64Array;
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

    // Pooling backprop
    const poolWeights = this.cache.get("poolWeights") as Float64Array;
    const dFinalHidden: Float64Array[] = [];

    // d(sum_t w_t * h_t) / dh_t = w_t
    for (let t = 0; t < seqLen; t++) {
      const dh = new Float64Array(this.embeddingDim);
      for (let e = 0; e < this.embeddingDim; e++) {
        dh[e] = poolWeights[t] * dAggregated[e];
      }
      dFinalHidden.push(dh);
    }

    // Gradient through softmax for poolW
    // dL/ds_i = sum_j dL/dw_j * dw_j/ds_i
    // where dw_j/ds_i = w_j * (delta_ij - w_i)
    const dPoolScores = new Float64Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      let upstream = 0;
      for (let e = 0; e < this.embeddingDim; e++) {
        upstream += dAggregated[e] * finalHidden[i][e];
      }
      for (let j = 0; j < seqLen; j++) {
        const jacobian = poolWeights[j] * ((i === j ? 1 : 0) - poolWeights[i]);
        dPoolScores[i] += upstream * jacobian;
      }
    }

    // poolW gradient
    for (let t = 0; t < seqLen; t++) {
      for (let e = 0; e < this.embeddingDim; e++) {
        this.poolW!.grad[e] += dPoolScores[t] * finalHidden[t][e];
        dFinalHidden[t][e] += dPoolScores[t] * this.poolW!.data[e];
      }
    }

    // Transformer blocks (reverse)
    let dCurrent = dFinalHidden;

    for (let b = this.numBlocks - 1; b >= 0; b--) {
      const residual1 = this.cache.get(`residual1_${b}`) as Float64Array[];
      const ln2Out = this.cache.get(`ln2_out_${b}`) as Float64Array[];
      const ffnHidden = this.cache.get(`ffn_hidden_${b}`) as Float64Array[];
      const ffnPreGelu = this.cache.get(`ffn_pre_${b}`) as Float64Array[];
      const ln1Out = this.cache.get(`ln1_out_${b}`) as Float64Array[];
      const blockInput = this.cache.get(`block_${b}_input`) as Float64Array[];
      const attnOut = this.cache.get(`attn_out_${b}`) as Float64Array[];
      const attnConcat = this.cache.get(`attn_concat_${b}`) as Float64Array[];
      const curSeqLen = dCurrent.length;

      // Through residual: d(x + ffn) = dx for both paths
      const dResidual1 = dCurrent.map((d) => new Float64Array(d));
      const dFFNOut = dCurrent.map((d) => new Float64Array(d));

      // FFN backward
      for (let t = 0; t < curSeqLen; t++) {
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
          dPreGelu[h] = dHidden[h] * geluGrad(ffnPreGelu[t][h]);
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

        // LayerNorm2 backward
        const norm2 = this.cache.get(`ln2_${b}_${t}_norm`) as Float64Array;
        const invStd2 =
          (this.cache.get(`ln2_${b}_${t}_invStd`) as Float64Array)[0];

        for (let e = 0; e < this.embeddingDim; e++) {
          this.ln2Gamma[b].grad[e] += dLn2[e] * norm2[e];
          this.ln2Beta[b].grad[e] += dLn2[e];
        }

        // Simplified LN backprop: just scale by gamma
        for (let e = 0; e < this.embeddingDim; e++) {
          dResidual1[t][e] += dLn2[e] * this.ln2Gamma[b].data[e] * invStd2;
        }
      }

      // Attention backward
      const dAttnOut = dResidual1.map((d) => new Float64Array(d));
      const dBlockInput = dResidual1.map((d) => new Float64Array(d));

      for (let t = 0; t < curSeqLen; t++) {
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

        // Q, K, V weight gradients (simplified - equal contribution)
        for (let h = 0; h < this.numHeads; h++) {
          for (let d = 0; d < this.headDim; d++) {
            const cIdx = h * this.headDim + d;
            const dHead = dConcat[cIdx];
            for (let e = 0; e < this.embeddingDim; e++) {
              const wIdx = d * this.embeddingDim + e;
              const inputVal = ln1Out[t][e];
              this.attnWq[b][h].grad[wIdx] += dHead * inputVal * 0.33;
              this.attnWk[b][h].grad[wIdx] += dHead * inputVal * 0.33;
              this.attnWv[b][h].grad[wIdx] += dHead * inputVal * 0.33;
            }
          }
        }

        // LayerNorm1 backward
        const norm1 = this.cache.get(`ln1_${b}_${t}_norm`) as Float64Array;
        const invStd1 =
          (this.cache.get(`ln1_${b}_${t}_invStd`) as Float64Array)[0];

        let dLn1Sum = 0;
        for (let e = 0; e < this.embeddingDim; e++) {
          dLn1Sum += dAttnOut[t][e];
        }

        for (let e = 0; e < this.embeddingDim; e++) {
          const dLn1 = dAttnOut[t][e] * 0.5; // Scale down attention path
          this.ln1Gamma[b].grad[e] += dLn1 * norm1[e];
          this.ln1Beta[b].grad[e] += dLn1;
          dBlockInput[t][e] += dLn1 * this.ln1Gamma[b].data[e] * invStd1;
        }
      }

      dCurrent = dBlockInput;
    }

    // Fusion backward
    const fusionGates = this.cache.get("fusionGates") as Float64Array[];
    const fusionConcats = this.cache.get("fusionConcats") as Float64Array[];
    const scaleOutputs = this.cache.get("scaleOutputs") as Float64Array[][];
    const numScales = this.temporalScales.length;
    const fusionInDim = this.embeddingDim * numScales;

    if (fusionGates && dCurrent.length > 0) {
      for (let t = 0; t < Math.min(fusionGates.length, dCurrent.length); t++) {
        const gates = fusionGates[t];
        const concat = fusionConcats[t];

        for (let s = 0; s < numScales; s++) {
          // Gradient through gate
          const dGate = sigmoidGrad(gates[s]);

          // Scale embedding gradients
          for (let e = 0; e < this.embeddingDim; e++) {
            const scaleIdx = Math.min(t, scaleOutputs[s].length - 1);
            const dFused = dCurrent[t][e];

            // d/dgate * dgate/dpreSig
            const gradGate = dFused * scaleOutputs[s][scaleIdx][e] * dGate;

            // Fusion gate weight gradient
            for (let i = 0; i < fusionInDim; i++) {
              this.fusionGateW!.grad[s * fusionInDim + i] += gradGate *
                concat[i];
            }
            this.fusionGateB!.grad[s] += gradGate;

            // Scale embedding gradient
            this.scaleEmbed[s].grad[e] += dFused * gates[s];
          }
        }
      }

      // Temporal conv backward
      const input = this.cache.get("input") as Float64Array[];
      for (let s = 0; s < numScales; s++) {
        const preAct = this.cache.get(`conv_pre_${s}`) as Float64Array[];
        if (!preAct) continue;

        const stride = this.temporalScales[s];

        for (
          let t = 0;
          t < Math.min(preAct.length, dCurrent.length, fusionGates.length);
          t++
        ) {
          const gate = fusionGates[t][s];

          for (let o = 0; o < this.embeddingDim; o++) {
            const dConvOut = dCurrent[t][o] * gate;
            const dPre = dConvOut * geluGrad(preAct[t][o]);

            this.temporalConvB[s].grad[o] += dPre;

            const startPos = t * stride;
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

    // Compute and clip gradient norm
    let gradNorm = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.size; i++) {
        gradNorm += p.grad[i] * p.grad[i];
      }
    }
    gradNorm = Math.sqrt(gradNorm);

    // Gradient clipping
    if (gradNorm > this.gradientClipNorm) {
      const scale = this.gradientClipNorm / gradNorm;
      for (const p of this.params) {
        for (let i = 0; i < p.size; i++) {
          p.grad[i] *= scale;
        }
      }
    }

    return gradNorm;
  }

  private getEffectiveLR(): number {
    if (this.updateCount < this.warmupSteps) {
      return this.learningRate * ((this.updateCount + 1) / this.warmupSteps);
    }
    const progress = (this.updateCount - this.warmupSteps) /
      Math.max(1, this.totalSteps - this.warmupSteps);
    return this.learningRate * 0.5 *
      (1 + Math.cos(Math.PI * Math.min(progress, 1)));
  }

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

    if (!this.initialized) {
      this.init(inputDim, outputDim);
    }

    // Convert and store raw input for prediction
    const xInput: Float64Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      xInput.push(new Float64Array(xCoordinates[t]));
    }
    this.lastRawInput = xInput;

    // Update input statistics (only on last timestep to avoid overweighting)
    this.inputWelford!.update(xInput[seqLen - 1]);

    // Get target
    const yTarget = new Float64Array(yCoordinates[seqLen - 1]);

    // Get output statistics BEFORE updating (so we have previous distribution)
    // This ensures the normalized target is relative to what the model has seen
    const outputMeanPrev = new Float64Array(this.outputWelford!.mean);
    const outputStdPrev = this.outputWelford!.getStd(this.epsilon);

    // Now update with current target
    this.outputWelford!.update(yTarget);

    // Normalize inputs
    const inputMean = this.inputWelford!.mean;
    const inputStd = this.inputWelford!.getStd(this.epsilon);
    const normalizedInput: Float64Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const norm = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        norm[i] = (xInput[t][i] - inputMean[i]) / inputStd[i];
      }
      normalizedInput.push(norm);
    }

    // Forward pass - model predicts in NORMALIZED space
    const predictedNorm = this.forward(normalizedInput);

    // Use current (updated) stats for denormalization
    const outputMean = this.outputWelford!.mean;
    const outputStd = this.outputWelford!.getStd(this.epsilon);

    // Denormalize prediction to raw scale for loss and reporting
    const predicted = new Float64Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      predicted[o] = predictedNorm[o] * outputStd[o] + outputMean[o];
    }

    // Compute MSE loss in raw scale
    let mseLoss = 0;
    for (let o = 0; o < this.outputDim; o++) {
      const diff = predicted[o] - yTarget[o];
      mseLoss += diff * diff;
    }
    mseLoss /= 2 * this.outputDim;

    // Update prediction variance for confidence intervals
    for (let o = 0; o < this.outputDim; o++) {
      const diff = predicted[o] - yTarget[o];
      const oldVar = this.predictionVariance![o];
      this.predictionVariance![o] = oldVar +
        (diff * diff - oldVar) / (this.sampleCount + 1);
    }

    // L2 regularization
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

    if (this.sampleCount > 10) {
      const outputStd = this.outputWelford!.getStd(this.epsilon);
      let residualNorm = 0;
      for (let o = 0; o < this.outputDim; o++) {
        const zScore = (predicted[o] - yTarget[o]) / outputStd[o];
        residualNorm += zScore * zScore;
      }
      residualNorm = Math.sqrt(residualNorm / this.outputDim);

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

    // Normalize target for backprop using PREVIOUS stats (before we updated)
    // This ensures meaningful gradients even on early samples
    const normalizedTarget = new Float64Array(this.outputDim);
    if (this.sampleCount === 0) {
      // First sample: use raw target as normalized (will learn bias)
      for (let o = 0; o < this.outputDim; o++) {
        normalizedTarget[o] = 0; // Predict mean
      }
    } else {
      for (let o = 0; o < this.outputDim; o++) {
        normalizedTarget[o] = (yTarget[o] - outputMeanPrev[o]) /
          outputStdPrev[o];
      }
    }

    // Backward pass with normalized values
    const gradNorm = this.backward(
      normalizedTarget,
      predictedNorm,
      sampleWeight,
    );

    // Optimizer step
    this.optimizerStep();

    // Update tracking
    this.sampleCount++;
    this.totalLoss += totalLoss;
    this.recentLosses.push(totalLoss);
    if (this.recentLosses.length > 100) {
      this.recentLosses.shift();
    }

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

  public predict(futureSteps: number): PredictionResult {
    if (!this.initialized || !this.lastRawInput) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const zScore95 = 1.96;

    // Start from raw input - explicit type to avoid inference issues
    let currentRawInput: Float64Array[] = [];
    for (const x of this.lastRawInput) {
      currentRawInput.push(new Float64Array(x));
    }

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current input
      const inputMean = this.inputWelford!.mean;
      const inputStd = this.inputWelford!.getStd(this.epsilon);

      const normalizedInput: Float64Array[] = [];
      for (let t = 0; t < currentRawInput.length; t++) {
        const norm = new Float64Array(this.inputDim);
        for (let i = 0; i < this.inputDim; i++) {
          norm[i] = (currentRawInput[t][i] - inputMean[i]) / inputStd[i];
        }
        normalizedInput.push(norm);
      }

      // Forward pass - model outputs normalized predictions
      const predictedNorm = this.forward(normalizedInput);

      // Denormalize to raw scale
      const outputMean = this.outputWelford!.mean;
      const outputStd = this.outputWelford!.getStd(this.epsilon);

      const denormalized: number[] = [];
      const standardError: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];

      for (let o = 0; o < this.outputDim; o++) {
        // Denormalize the prediction
        const pred = predictedNorm[o] * outputStd[o] + outputMean[o];

        // Use output standard deviation from training data for confidence intervals
        const se = outputStd[o] / Math.sqrt(Math.max(this.sampleCount, 10));

        // Widen CI slightly for future steps to account for compounding uncertainty
        const stepMultiplier = 1 + step * 0.1;

        denormalized.push(pred);
        standardError.push(se);
        lowerBound.push(pred - zScore95 * se * stepMultiplier);
        upperBound.push(pred + zScore95 * se * stepMultiplier);
      }

      predictions.push({
        predicted: denormalized,
        lowerBound,
        upperBound,
        standardError,
      });

      // Autoregressive: use normalized prediction as next input if dimensions match
      if (currentRawInput.length > 1 && this.inputDim === this.outputDim) {
        const newInput: Float64Array[] = [];
        for (let t = 1; t < currentRawInput.length; t++) {
          newInput.push(currentRawInput[t]);
        }
        // Append the raw (denormalized) prediction
        const rawPred = new Float64Array(this.outputDim);
        for (let o = 0; o < this.outputDim; o++) {
          rawPred[o] = denormalized[o];
        }
        newInput.push(rawPred);
        currentRawInput = newInput;
      }
    }

    // Compute accuracy from recent losses (skip first few high losses)
    let avgLoss = 0;
    const stableLosses = this.recentLosses.filter((_, i) =>
      i >= Math.min(10, this.recentLosses.length - 1)
    );
    if (stableLosses.length > 0) {
      for (const loss of stableLosses) {
        avgLoss += loss;
      }
      avgLoss /= stableLosses.length;
    } else if (this.recentLosses.length > 0) {
      avgLoss = this.recentLosses[this.recentLosses.length - 1];
    } else {
      avgLoss = this.sampleCount > 0 ? this.totalLoss / this.sampleCount : 1;
    }
    // Use log transform to get reasonable accuracy values
    const accuracy = Math.exp(-avgLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  public getModelSummary(): ModelSummary {
    let avgLoss = 0;
    const stableLosses = this.recentLosses.filter((_, i) =>
      i >= Math.min(10, this.recentLosses.length - 1)
    );
    if (stableLosses.length > 0) {
      for (const loss of stableLosses) {
        avgLoss += loss;
      }
      avgLoss /= stableLosses.length;
    } else if (this.recentLosses.length > 0) {
      avgLoss = this.recentLosses[this.recentLosses.length - 1];
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
      accuracy,
      converged: this.converged,
      effectiveLearningRate: this.getEffectiveLR(),
      driftCount: this.driftCount,
    };
  }

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
    this.recentLosses = [];
    this.params = [];
    this.cache.clear();
    this.lastRawInput = null;
    this.inputWelford = null;
    this.outputWelford = null;
    this.adwin = null;
    this.predictionVariance = null;
  }

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
        recentLosses: this.recentLosses,
        predictionVariance: this.predictionVariance
          ? Array.from(this.predictionVariance)
          : [],
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

  public load(jsonString: string): void {
    const state = JSON.parse(jsonString);

    if (!state.state.initialized) {
      this.reset();
      return;
    }

    this.init(state.state.inputDim, state.state.outputDim);

    if (state.welford) {
      this.inputWelford!.deserialize(state.welford.input);
      this.outputWelford!.deserialize(state.welford.output);
    }

    if (state.adwin) {
      this.adwin!.deserialize(state.adwin);
    }

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

    if (state.lastRawInput) {
      this.lastRawInput = state.lastRawInput.map((x: number[]) =>
        new Float64Array(x)
      );
    }

    if (state.state.predictionVariance) {
      this.predictionVariance = new Float64Array(
        state.state.predictionVariance,
      );
    }

    this.sampleCount = state.state.sampleCount;
    this.updateCount = state.state.updateCount;
    this.totalLoss = state.state.totalLoss;
    this.prevLoss = state.state.prevLoss;
    this.converged = state.state.converged;
    this.driftCount = state.state.driftCount;
    this.recentLosses = state.state.recentLosses || [];
  }
}

export default FusionTemporalTransformerRegression;
