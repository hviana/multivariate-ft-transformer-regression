/* FusionTemporalTransformerRegression.ts
 *
 * A CPU-optimized, allocation-minimized TypeScript implementation of a
 * Fusion Temporal Transformer-style network for multivariate sequence-to-sequence
 * regression with incremental online learning (Adam), Welford z-score normalization,
 * L2 regularization, outlier down-weighting, and ADWIN drift detection.
 *
 * Notes:
 * - Uses Float64Array for all numeric tensors.
 * - Preallocates and reuses buffers aggressively.
 * - Implements full backprop through: temporal conv (multi-scale), gated fusion,
 *   transformer blocks (LN + MHA + FFN), and per-timestep output head.
 * - Trains on ALL time steps: yCoordinates must be seqLen x outputDim.
 *
 * Export: FusionTemporalTransformerRegression
 */

export interface FitInput {
  xCoordinates: number[][];
  yCoordinates: number[][];
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
  temporalConvWeights: number[][][]; // [scale][flat...]
  scaleEmbeddings: number[][];
  positionalEncoding: number[][]; // [maxSeqLen][embDim]
  fusionWeights: number[][][]; // [Wg, bg]
  attentionWeights: number[][][]; // per block flattened
  ffnWeights: number[][][]; // per block flattened
  layerNormParams: number[][][]; // per block (gamma/beta)
  outputWeights: number[][][]; // [Wy, by]
  firstMoment: number[][][]; // large, grouped
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

export interface FusionTemporalTransformerConfig {
  numBlocks: number;
  embeddingDim: number;
  numHeads: number;
  ffnMultiplier: number;
  attentionDropout: number; // (kept for API; dropout disabled for determinism/perf by default)
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
  fusionDropout: number; // (kept for API; dropout disabled by default)
}

const DEFAULT_CONFIG: FusionTemporalTransformerConfig = {
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
};

// -----------------------------
// Low-level utilities (no alloc)
// -----------------------------

function clampInt(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function gelu(x: number): number {
  // GELU approximation:
  // 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
  const c = 0.7978845608028654;
  const x3 = x * x * x;
  const t = c * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  return 0.5 * x * (1.0 + th);
}

function dGelu(x: number): number {
  // derivative of approximate GELU used above
  // Let u = c*(x + a*x^3), y=0.5*x*(1+tanh(u))
  // dy/dx = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*du/dx
  const c = 0.7978845608028654;
  const a = 0.044715;
  const x2 = x * x;
  const x3 = x2 * x;
  const u = c * (x + a * x3);
  const th = Math.tanh(u);
  const sech2 = 1.0 - th * th;
  const du = c * (1.0 + 3.0 * a * x2);
  return 0.5 * (1.0 + th) + 0.5 * x * sech2 * du;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  } else {
    const z = Math.exp(x);
    return z / (1 + z);
  }
}

function softmaxRowInPlace(
  scores: Float64Array,
  offset: number,
  len: number,
): void {
  // stable softmax: subtract max
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = scores[offset + i];
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(scores[offset + i] - max);
    scores[offset + i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 1;
  for (let i = 0; i < len; i++) scores[offset + i] *= inv;
}

function dot(
  a: Float64Array,
  aOff: number,
  b: Float64Array,
  bOff: number,
  len: number,
): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}

function addInPlace(a: Float64Array, b: Float64Array): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

function scaleInPlace(a: Float64Array, s: number): void {
  for (let i = 0; i < a.length; i++) a[i] *= s;
}

function setZero(a: Float64Array): void {
  a.fill(0);
}

function l2Norm(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    s += v * v;
  }
  return Math.sqrt(s);
}

// Matrix multiply: C[m x p] = A[m x n] * B[n x p]
// row-major, no allocations.
function matmul(
  A: Float64Array,
  B: Float64Array,
  C: Float64Array,
  m: number,
  n: number,
  p: number,
): void {
  // assumes C length = m*p
  let cIdx = 0;
  for (let i = 0; i < m; i++) {
    const aRow = i * n;
    for (let j = 0; j < p; j++) {
      let sum = 0;
      // B col j stride p
      for (let k = 0; k < n; k++) sum += A[aRow + k] * B[k * p + j];
      C[cIdx++] = sum;
    }
  }
}

function matmulAddBias(
  A: Float64Array,
  B: Float64Array,
  bias: Float64Array,
  C: Float64Array,
  m: number,
  n: number,
  p: number,
): void {
  let cIdx = 0;
  for (let i = 0; i < m; i++) {
    const aRow = i * n;
    for (let j = 0; j < p; j++) {
      let sum = bias[j];
      for (let k = 0; k < n; k++) sum += A[aRow + k] * B[k * p + j];
      C[cIdx++] = sum;
    }
  }
}

function addPosEncInPlace(
  X: Float64Array,
  posEnc: Float64Array,
  seqLen: number,
  dim: number,
): void {
  // X: [seqLen, dim], posEnc: [maxSeqLen, dim]
  let x = 0;
  let p = 0;
  for (let t = 0; t < seqLen; t++) {
    for (let d = 0; d < dim; d++) X[x++] += posEnc[p++];
  }
}

// ---------------------------------------
// Float64Array pool (frequently used buf)
// ---------------------------------------
class Float64ArrayPool {
  private freeBySize: Map<number, Float64Array[]> = new Map();

  get(size: number): Float64Array {
    const stack = this.freeBySize.get(size);
    if (stack && stack.length > 0) {
      const arr = stack.pop()!;
      arr.fill(0);
      return arr;
    }
    return new Float64Array(size);
  }

  release(arr: Float64Array): void {
    const size = arr.length;
    let stack = this.freeBySize.get(size);
    if (!stack) {
      stack = [];
      this.freeBySize.set(size, stack);
    }
    stack.push(arr);
  }

  reset(): void {
    this.freeBySize.clear();
  }
}

// --------------------------
// Welford normalization stats
// --------------------------
class Welford {
  readonly dim: number;
  count: number;
  mean: Float64Array;
  m2: Float64Array;

  constructor(dim: number) {
    this.dim = dim;
    this.count = 0;
    this.mean = new Float64Array(dim);
    this.m2 = new Float64Array(dim);
  }

  reset(): void {
    this.count = 0;
    this.mean.fill(0);
    this.m2.fill(0);
  }

  // Update with one observation vector x[dim]
  update(x: Float64Array, off = 0): void {
    this.count++;
    const n = this.count;
    for (let i = 0; i < this.dim; i++) {
      const xi = x[off + i];
      const delta = xi - this.mean[i];
      this.mean[i] += delta / n;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  // Std with epsilon floor
  std(out: Float64Array, eps: number): void {
    if (this.count < 2) {
      for (let i = 0; i < this.dim; i++) out[i] = 1.0;
      return;
    }
    const inv = 1.0 / (this.count - 1);
    for (let i = 0; i < this.dim; i++) {
      const v = this.m2[i] * inv;
      out[i] = Math.sqrt(v + eps);
    }
  }
}

// --------------------------
// ADWIN drift detection (lite)
// --------------------------
class AdwinLite {
  // Ring buffer of errors; evaluate best cut.
  private readonly maxWindow: number;
  private readonly delta: number;
  private buf: Float64Array;
  private start = 0;
  private len = 0;

  constructor(delta: number, maxWindow = 256) {
    this.delta = delta;
    this.maxWindow = maxWindow;
    this.buf = new Float64Array(maxWindow);
  }

  reset(): void {
    this.start = 0;
    this.len = 0;
    this.buf.fill(0);
  }

  push(err: number): boolean {
    // insert
    if (this.len < this.maxWindow) {
      this.buf[(this.start + this.len) % this.maxWindow] = err;
      this.len++;
    } else {
      // overwrite oldest
      this.buf[this.start] = err;
      this.start = (this.start + 1) % this.maxWindow;
    }

    if (this.len < 32) return false;

    // compute prefix sums for best split (O(n^2) avoided: O(n) with rolling)
    // We try a handful of cuts for perf: powers-of-two-ish + midpoints.
    const n = this.len;
    const cuts = new Int32Array(12);
    let cCount = 0;
    cuts[cCount++] = (n / 2) | 0;
    cuts[cCount++] = (n / 3) | 0;
    cuts[cCount++] = ((2 * n) / 3) | 0;
    for (let k = 16; k < n - 16 && cCount < cuts.length; k *= 2) {
      cuts[cCount++] = k;
    }
    for (let k = n - 16; k > 16 && cCount < cuts.length; k -= 32) {
      cuts[cCount++] = k;
    }

    // compute global mean/var quickly for epsilon cut
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < n; i++) {
      const v = this.buf[(this.start + i) % this.maxWindow];
      sum += v;
      sum2 += v * v;
    }
    const mean = sum / n;
    const varPop = sum2 / n - mean * mean;
    const variance = varPop > 1e-12 ? varPop : 1e-12;

    // Hoeffding-like bound
    // epsCut = sqrt( (2*variance*ln(2/delta))/n ) + (2/3)*ln(2/delta)/n
    const ln = Math.log(2 / this.delta);
    const epsCut = Math.sqrt((2 * variance * ln) / n) + (2 / 3) * (ln / n);

    for (let ci = 0; ci < cCount; ci++) {
      const cut = clampInt(cuts[ci], 8, n - 8);
      let s0 = 0,
        s1 = 0;
      for (let i = 0; i < cut; i++) {
        s0 += this.buf[(this.start + i) % this.maxWindow];
      }
      for (let i = cut; i < n; i++) {
        s1 += this.buf[(this.start + i) % this.maxWindow];
      }
      const m0 = s0 / cut;
      const m1 = s1 / (n - cut);
      if (Math.abs(m0 - m1) >= epsCut) {
        // drift detected -> shrink window to newer portion
        const newLen = n - cut;
        for (let i = 0; i < newLen; i++) {
          this.buf[i] = this.buf[(this.start + cut + i) % this.maxWindow];
        }
        this.start = 0;
        this.len = newLen;
        return true;
      }
    }

    return false;
  }
}

// --------------------------
// Parameter + Adam optimizer
// --------------------------
class Parameter {
  readonly w: Float64Array;
  readonly g: Float64Array;
  readonly m: Float64Array;
  readonly v: Float64Array;
  readonly size: number;

  constructor(size: number, initScale: number, rng: () => number) {
    this.size = size;
    this.w = new Float64Array(size);
    this.g = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
    // Xavier-ish init: N(0, initScale)
    for (let i = 0; i < size; i++) this.w[i] = (rng() * 2 - 1) * initScale;
  }

  zeroGrad(): void {
    this.g.fill(0);
  }
}

class Adam {
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly eps: number;
  stepCount = 0;

  constructor(beta1: number, beta2: number, eps: number) {
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  reset(): void {
    this.stepCount = 0;
  }

  apply(param: Parameter, lr: number, weightDecay: number): void {
    // AdamW-like: add L2 grad = wd * w
    const b1 = this.beta1;
    const b2 = this.beta2;
    const t = ++this.stepCount;

    const b1t = 1.0 - Math.pow(b1, t);
    const b2t = 1.0 - Math.pow(b2, t);

    const w = param.w;
    const g = param.g;
    const m = param.m;
    const v = param.v;

    for (let i = 0; i < param.size; i++) {
      const gi = g[i] + weightDecay * w[i];
      const mi = m[i] = b1 * m[i] + (1 - b1) * gi;
      const vi = v[i] = b2 * v[i] + (1 - b2) * (gi * gi);
      const mHat = mi / b1t;
      const vHat = vi / b2t;
      w[i] -= lr * (mHat / (Math.sqrt(vHat) + this.eps));
      g[i] = 0; // clear in same pass (no extra fill)
    }
  }
}

// --------------------------
// LayerNorm (per time step)
// --------------------------
class LayerNorm {
  readonly dim: number;
  readonly gamma: Parameter;
  readonly beta: Parameter;

  // cache
  private mean: Float64Array;
  private invStd: Float64Array;
  private xHat: Float64Array;

  constructor(dim: number, rng: () => number) {
    this.dim = dim;
    this.gamma = new Parameter(dim, 0.02, rng);
    this.beta = new Parameter(dim, 0.0, () => 0.5); // start at 0
    // initialize gamma ~ 1
    for (let i = 0; i < dim; i++) this.gamma.w[i] = 1.0;

    this.mean = new Float64Array(0);
    this.invStd = new Float64Array(0);
    this.xHat = new Float64Array(0);
  }

  ensure(seqLen: number): void {
    const n = seqLen;
    if (this.mean.length !== n) {
      this.mean = new Float64Array(n);
      this.invStd = new Float64Array(n);
      this.xHat = new Float64Array(n * this.dim);
    }
  }

  /**
   * Forward LayerNorm over last dimension.
   * y = gamma ⊙ (x-μ)/sqrt(σ²+eps) + beta
   */
  forward(
    x: Float64Array,
    y: Float64Array,
    seqLen: number,
    eps: number,
  ): void {
    this.ensure(seqLen);
    const d = this.dim;
    let idx = 0;
    for (let t = 0; t < seqLen; t++) {
      // mean
      let mu = 0;
      for (let i = 0; i < d; i++) mu += x[idx + i];
      mu /= d;
      this.mean[t] = mu;
      // var
      let v = 0;
      for (let i = 0; i < d; i++) {
        const z = x[idx + i] - mu;
        v += z * z;
      }
      v /= d;
      const inv = 1.0 / Math.sqrt(v + eps);
      this.invStd[t] = inv;

      // normalize & affine
      const base = t * d;
      for (let i = 0; i < d; i++) {
        const xh = (x[idx + i] - mu) * inv;
        this.xHat[base + i] = xh;
        y[idx + i] = xh * this.gamma.w[i] + this.beta.w[i];
      }
      idx += d;
    }
  }

  /**
   * Backward: given dY, compute dX and accumulate gamma/beta grads.
   */
  backward(
    x: Float64Array,
    dY: Float64Array,
    dX: Float64Array,
    seqLen: number,
  ): void {
    const d = this.dim;
    const gamma = this.gamma.w;
    const dGamma = this.gamma.g;
    const dBeta = this.beta.g;

    // accumulate dGamma/dBeta
    for (let i = 0; i < d; i++) {
      dGamma[i] += 0; // touch (no-op)
      dBeta[i] += 0;
    }

    let idx = 0;
    for (let t = 0; t < seqLen; t++) {
      const mu = this.mean[t];
      const inv = this.invStd[t];
      const base = t * d;

      // dBeta and dGamma
      for (let i = 0; i < d; i++) {
        const dy = dY[idx + i];
        dBeta[i] += dy;
        dGamma[i] += dy * this.xHat[base + i];
      }

      // dx: LayerNorm backprop (per row)
      // Let xhat=(x-mu)*inv, y=gamma*xhat+beta
      // dx = (1/d)*inv * (dY*gamma*d - sum(dY*gamma) - xhat*sum(dY*gamma*xhat))
      let sum1 = 0;
      let sum2 = 0;
      for (let i = 0; i < d; i++) {
        const dyG = dY[idx + i] * gamma[i];
        sum1 += dyG;
        sum2 += dyG * this.xHat[base + i];
      }
      const invD = 1.0 / d;
      for (let i = 0; i < d; i++) {
        const dyG = dY[idx + i] * gamma[i];
        const xh = this.xHat[base + i];
        dX[idx + i] = inv * invD * (d * dyG - sum1 - xh * sum2);
      }

      idx += d;
    }
  }
}

// --------------------------
// Temporal Conv1D (causal-ish)
// --------------------------
class TemporalConv1D {
  readonly inDim: number;
  readonly outDim: number;
  readonly kernel: number;
  readonly stride: number;

  readonly W: Parameter; // [kernel, inDim, outDim] flattened as ((k*inDim + c)*outDim + o)
  readonly b: Parameter; // [outDim]

  // caches for backprop
  private preAct: Float64Array = new Float64Array(0); // [outLen*outDim]
  private xNormRef: Float64Array | null = null;
  private lastInSeqLen = 0;
  private lastOutLen = 0;

  constructor(
    inDim: number,
    outDim: number,
    kernel: number,
    stride: number,
    rng: () => number,
  ) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.kernel = kernel;
    this.stride = stride;
    const scale = Math.sqrt(2 / (inDim * kernel));
    this.W = new Parameter(kernel * inDim * outDim, scale, rng);
    this.b = new Parameter(outDim, 0.0, () => 0.5);
    this.b.w.fill(0);
  }

  outLen(seqLen: number): number {
    // causal left padding: output at positions 0, stride, 2*stride...
    return ((seqLen + this.stride - 1) / this.stride) | 0;
  }

  ensure(outLen: number): void {
    const need = outLen * this.outDim;
    if (this.preAct.length !== need) this.preAct = new Float64Array(need);
  }

  /**
   * Forward:
   * For each output time tOut, center corresponds to tIn = tOut*stride,
   * kernel taps look back: x[tIn - k] for k=0..kernel-1 (causal).
   * y = GELU(conv + b)
   */
  forward(x: Float64Array, seqLen: number, y: Float64Array): number {
    const outLen = this.outLen(seqLen);
    this.ensure(outLen);
    this.xNormRef = x;
    this.lastInSeqLen = seqLen;
    this.lastOutLen = outLen;

    const inD = this.inDim;
    const outD = this.outDim;
    const K = this.kernel;
    const stride = this.stride;
    const W = this.W.w;
    const b = this.b.w;

    let yIdx = 0;
    for (let t = 0; t < outLen; t++) {
      const tIn = t * stride;
      for (let o = 0; o < outD; o++) {
        let sum = b[o];
        // kernel taps (causal)
        for (let k = 0; k < K; k++) {
          const ti = tIn - k;
          if (ti < 0 || ti >= seqLen) continue;
          const xBase = ti * inD;
          const wBase = (k * inD) * outD + o;
          for (let c = 0; c < inD; c++) {
            sum += x[xBase + c] * W[wBase + c * outD];
          }
        }
        this.preAct[yIdx] = sum;
        y[yIdx] = gelu(sum);
        yIdx++;
      }
    }
    return outLen;
  }

  backward(dY: Float64Array, dX: Float64Array): void {
    // dY is gradient wrt output activation (after GELU) sized [outLen*outDim]
    // dX sized [inSeqLen*inDim]
    const x = this.xNormRef!;
    const seqLen = this.lastInSeqLen;
    const outLen = this.lastOutLen;
    const inD = this.inDim;
    const outD = this.outDim;
    const K = this.kernel;
    const stride = this.stride;

    const W = this.W.w;
    const dW = this.W.g;
    const db = this.b.g;

    // accumulate
    for (let i = 0; i < db.length; i++) db[i] += 0;

    let yIdx = 0;
    for (let t = 0; t < outLen; t++) {
      const tIn = t * stride;
      for (let o = 0; o < outD; o++) {
        const pre = this.preAct[yIdx];
        const dy = dY[yIdx] * dGelu(pre);
        db[o] += dy;

        for (let k = 0; k < K; k++) {
          const ti = tIn - k;
          if (ti < 0 || ti >= seqLen) continue;
          const xBase = ti * inD;
          const wBase = (k * inD) * outD + o;
          for (let c = 0; c < inD; c++) {
            const xi = x[xBase + c];
            dW[wBase + c * outD] += xi * dy;
            dX[xBase + c] += W[wBase + c * outD] * dy;
          }
        }

        yIdx++;
      }
    }
  }
}

// --------------------------
// Multi-Head Self Attention
// --------------------------
class MultiHeadSelfAttention {
  readonly embDim: number;
  readonly numHeads: number;
  readonly headDim: number;

  // parameters
  readonly Wq: Parameter;
  readonly Wk: Parameter;
  readonly Wv: Parameter;
  readonly Wo: Parameter;
  readonly bq: Parameter;
  readonly bk: Parameter;
  readonly bv: Parameter;
  readonly bo: Parameter;

  // caches (per forward)
  private Q: Float64Array = new Float64Array(0); // [T, D]
  private K: Float64Array = new Float64Array(0);
  private V: Float64Array = new Float64Array(0);
  private attn: Float64Array = new Float64Array(0); // [H, T, T] flattened
  private context: Float64Array = new Float64Array(0); // [T, D]
  private lastT = 0;

  // local buffers reused
  private dQ: Float64Array = new Float64Array(0);
  private dK: Float64Array = new Float64Array(0);
  private dV: Float64Array = new Float64Array(0);
  private dContext: Float64Array = new Float64Array(0);

  constructor(embDim: number, numHeads: number, rng: () => number) {
    this.embDim = embDim;
    this.numHeads = numHeads;
    this.headDim = (embDim / numHeads) | 0;
    if (this.headDim * numHeads !== embDim) {
      throw new Error("embeddingDim must be divisible by numHeads");
    }

    const scale = Math.sqrt(2 / embDim);
    // Dense: X[T,D] * W[D,D]
    this.Wq = new Parameter(embDim * embDim, scale, rng);
    this.Wk = new Parameter(embDim * embDim, scale, rng);
    this.Wv = new Parameter(embDim * embDim, scale, rng);
    this.Wo = new Parameter(embDim * embDim, scale, rng);
    this.bq = new Parameter(embDim, 0.0, () => 0.5);
    this.bk = new Parameter(embDim, 0.0, () => 0.5);
    this.bv = new Parameter(embDim, 0.0, () => 0.5);
    this.bo = new Parameter(embDim, 0.0, () => 0.5);
    this.bq.w.fill(0);
    this.bk.w.fill(0);
    this.bv.w.fill(0);
    this.bo.w.fill(0);
  }

  ensure(T: number): void {
    const D = this.embDim;
    const H = this.numHeads;
    const needTD = T * D;
    const needAttn = H * T * T;
    if (this.Q.length !== needTD) {
      this.Q = new Float64Array(needTD);
      this.K = new Float64Array(needTD);
      this.V = new Float64Array(needTD);
      this.context = new Float64Array(needTD);
      this.dQ = new Float64Array(needTD);
      this.dK = new Float64Array(needTD);
      this.dV = new Float64Array(needTD);
      this.dContext = new Float64Array(needTD);
    }
    if (this.attn.length !== needAttn) this.attn = new Float64Array(needAttn);
    this.lastT = T;
  }

  /**
   * Forward self-attention with causal mask:
   * A = softmax(QK^T / sqrt(dk) + mask), mask = -inf for j>i
   * context = A V, out = context Wo + bo
   */
  forward(x: Float64Array, out: Float64Array, T: number): void {
    this.ensure(T);
    const D = this.embDim;
    const dk = this.headDim;
    const invSqrtDk = 1.0 / Math.sqrt(dk);

    // Q,K,V = xW + b
    matmulAddBias(x, this.Wq.w, this.bq.w, this.Q, T, D, D);
    matmulAddBias(x, this.Wk.w, this.bk.w, this.K, T, D, D);
    matmulAddBias(x, this.Wv.w, this.bv.w, this.V, T, D, D);

    // attention per head
    const H = this.numHeads;
    const attn = this.attn;
    attn.fill(0);

    // context = 0
    this.context.fill(0);

    // Compute A[h, i, j] and context
    for (let h = 0; h < H; h++) {
      const qOffH = h * dk;
      const kOffH = h * dk;
      const vOffH = h * dk;
      const aBase = h * T * T;

      // scores in attn buffer for each i: row aBase + i*T
      for (let i = 0; i < T; i++) {
        const qi = i * D + qOffH;
        const rowOff = aBase + i * T;

        // causal: j<=i
        for (let j = 0; j <= i; j++) {
          const kj = j * D + kOffH;
          const s = dot(this.Q, qi, this.K, kj, dk) * invSqrtDk;
          attn[rowOff + j] = s;
        }
        // masked positions left at 0 but must be -inf; handle by setting to -1e30
        for (let j = i + 1; j < T; j++) attn[rowOff + j] = -1e30;

        softmaxRowInPlace(attn, rowOff, T);

        // context[i, head] = sum_j A * V
        const ci = i * D + vOffH;
        for (let u = 0; u < dk; u++) {
          let sum = 0;
          for (let j = 0; j < T; j++) {
            const aij = attn[rowOff + j];
            const vj = this.V[j * D + vOffH + u];
            sum += aij * vj;
          }
          this.context[ci + u] = sum;
        }
      }
    }

    // out = context Wo + bo
    matmulAddBias(this.context, this.Wo.w, this.bo.w, out, T, D, D);
  }

  backward(x: Float64Array, dOut: Float64Array, dX: Float64Array): void {
    const T = this.lastT;
    const D = this.embDim;
    const H = this.numHeads;
    const dk = this.headDim;
    const invSqrtDk = 1.0 / Math.sqrt(dk);

    // dContext = dOut * Wo^T
    // and dWo, dbo
    this.dContext.fill(0);
    const dWo = this.Wo.g;
    const dbo = this.bo.g;

    // dbo
    for (let j = 0; j < D; j++) dbo[j] += 0;
    let idx = 0;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < D; j++) dbo[j] += dOut[idx++];

      // dWo += context^T * dOut
      const cRow = i * D;
      const dRow = i * D;
      for (let k = 0; k < D; k++) {
        const ck = this.context[cRow + k];
        const wBase = k * D;
        for (let j = 0; j < D; j++) dWo[wBase + j] += ck * dOut[dRow + j];
      }

      // dContext = dOut * Wo^T
      for (let k = 0; k < D; k++) {
        let s = 0;
        const wBase = k * D;
        for (let j = 0; j < D; j++) s += dOut[dRow + j] * this.Wo.w[wBase + j];
        this.dContext[dRow + k] = s;
      }
    }

    // Backprop attention
    this.dQ.fill(0);
    this.dK.fill(0);
    this.dV.fill(0);

    const dWq = this.Wq.g, dWk = this.Wk.g, dWv = this.Wv.g;
    const dbq = this.bq.g, dbk = this.bk.g, dbv = this.bv.g;

    // For each head:
    // context_i = sum_j A_ij V_j
    // dV_j += sum_i A_ij * dContext_i
    // dA_ij += dot(dContext_i, V_j)
    // A = softmax(scores); dScores = A*(dA - sum(dA*A))
    // scores_ij = (Q_i · K_j)/sqrt(dk)
    // dQ_i += sum_j dScores_ij * K_j /sqrt(dk)
    // dK_j += sum_i dScores_ij * Q_i /sqrt(dk)

    const dArow = new Float64Array(T); // small, but allocation; avoid by reusing? keep single and reuse per row
    const dSrow = new Float64Array(T);

    for (let h = 0; h < H; h++) {
      const qOffH = h * dk;
      const kOffH = h * dk;
      const vOffH = h * dk;
      const aBase = h * T * T;

      // dV
      for (let j = 0; j < T; j++) {
        const vj = j * D + vOffH;
        for (let u = 0; u < dk; u++) {
          let sum = 0;
          for (let i = 0; i < T; i++) {
            const rowOff = aBase + i * T;
            const aij = this.attn[rowOff + j];
            const dci = this.dContext[i * D + vOffH + u];
            sum += aij * dci;
          }
          this.dV[vj + u] += sum;
        }
      }

      // dScores, then dQ/dK
      for (let i = 0; i < T; i++) {
        const rowOff = aBase + i * T;
        // compute dA for row i
        for (let j = 0; j < T; j++) {
          // masked j>i -> A ~ 0; still compute safe
          // dA_ij = dot(dContext_i_head, V_j_head)
          let s = 0;
          const dci = i * D + vOffH;
          const vj = j * D + vOffH;
          for (let u = 0; u < dk; u++) {
            s += this.dContext[dci + u] * this.V[vj + u];
          }
          dArow[j] = s;
        }

        // dScores = softmax backward: A*(dA - sum(dA*A))
        let sum = 0;
        for (let j = 0; j < T; j++) sum += dArow[j] * this.attn[rowOff + j];
        for (let j = 0; j < T; j++) {
          dSrow[j] = this.attn[rowOff + j] * (dArow[j] - sum);
        }

        // causal mask: enforce j>i have zero grad
        for (let j = i + 1; j < T; j++) dSrow[j] = 0;

        // dQ_i += sum_j dS_ij * K_j /sqrt(dk)
        const qi = i * D + qOffH;
        for (let u = 0; u < dk; u++) {
          let s = 0;
          for (let j = 0; j < T; j++) s += dSrow[j] * this.K[j * D + kOffH + u];
          this.dQ[qi + u] += s * invSqrtDk;
        }

        // dK_j += dS_ij * Q_i /sqrt(dk)
        for (let j = 0; j < T; j++) {
          const kj = j * D + kOffH;
          const dS = dSrow[j] * invSqrtDk;
          for (let u = 0; u < dk; u++) this.dK[kj + u] += dS * this.Q[qi + u];
        }
      }
    }

    // Backprop projections: Q = xWq + bq etc.
    // dWq += x^T dQ, dbq += sum dQ, dX += dQ Wq^T ... similarly for K,V.
    dX.fill(0);

    // helper: accumulate dense backward
    const denseBackward = (
      W: Float64Array,
      dW: Float64Array,
      db: Float64Array,
      dProj: Float64Array,
    ) => {
      // db
      for (let j = 0; j < D; j++) db[j] += 0;
      for (let i = 0; i < T; i++) {
        const dRow = i * D;
        for (let j = 0; j < D; j++) db[j] += dProj[dRow + j];
      }
      // dW += x^T dProj
      for (let k = 0; k < D; k++) {
        const wBase = k * D;
        for (let j = 0; j < D; j++) {
          let sum = 0;
          for (let i = 0; i < T; i++) sum += x[i * D + k] * dProj[i * D + j];
          dW[wBase + j] += sum;
        }
      }
      // dX += dProj W^T
      for (let i = 0; i < T; i++) {
        const dRow = i * D;
        const xRow = i * D;
        for (let k = 0; k < D; k++) {
          let sum = 0;
          const wBase = k * D;
          for (let j = 0; j < D; j++) sum += dProj[dRow + j] * W[wBase + j];
          dX[xRow + k] += sum;
        }
      }
    };

    denseBackward(this.Wq.w, dWq, dbq, this.dQ);
    denseBackward(this.Wk.w, dWk, dbk, this.dK);
    denseBackward(this.Wv.w, dWv, dbv, this.dV);
  }
}

// --------------------------
// Feed-forward network (FFN)
// --------------------------
class FFN {
  readonly embDim: number;
  readonly hiddenDim: number;

  readonly W1: Parameter;
  readonly b1: Parameter;
  readonly W2: Parameter;
  readonly b2: Parameter;

  // caches
  private preAct: Float64Array = new Float64Array(0); // [T, hidden]
  private act: Float64Array = new Float64Array(0); // [T, hidden]
  private lastT = 0;

  constructor(embDim: number, multiplier: number, rng: () => number) {
    this.embDim = embDim;
    this.hiddenDim = embDim * multiplier;

    const scale1 = Math.sqrt(2 / embDim);
    const scale2 = Math.sqrt(2 / this.hiddenDim);
    this.W1 = new Parameter(embDim * this.hiddenDim, scale1, rng);
    this.b1 = new Parameter(this.hiddenDim, 0.0, () => 0.5);
    this.W2 = new Parameter(this.hiddenDim * embDim, scale2, rng);
    this.b2 = new Parameter(embDim, 0.0, () => 0.5);
    this.b1.w.fill(0);
    this.b2.w.fill(0);
  }

  ensure(T: number): void {
    const need = T * this.hiddenDim;
    if (this.preAct.length !== need) {
      this.preAct = new Float64Array(need);
      this.act = new Float64Array(need);
    }
    this.lastT = T;
  }

  forward(x: Float64Array, out: Float64Array, T: number): void {
    this.ensure(T);
    const D = this.embDim;
    const H = this.hiddenDim;

    // preAct = xW1 + b1
    matmulAddBias(x, this.W1.w, this.b1.w, this.preAct, T, D, H);

    // act = GELU(preAct)
    for (let i = 0; i < this.preAct.length; i++) {
      const z = this.preAct[i];
      const a = gelu(z);
      this.act[i] = a;
    }

    // out = act W2 + b2
    matmulAddBias(this.act, this.W2.w, this.b2.w, out, T, H, D);
  }

  backward(x: Float64Array, dOut: Float64Array, dX: Float64Array): void {
    const T = this.lastT;
    const D = this.embDim;
    const H = this.hiddenDim;

    // dAct = dOut * W2^T ; dW2 = act^T dOut ; db2 = sum dOut
    const dAct = new Float64Array(T * H); // could pool; kept simple due to size variability
    const dW2 = this.W2.g;
    const db2 = this.b2.g;

    for (let j = 0; j < D; j++) db2[j] += 0;
    for (let i = 0; i < T; i++) {
      const dRow = i * D;
      for (let j = 0; j < D; j++) db2[j] += dOut[dRow + j];
    }

    // dW2
    for (let k = 0; k < H; k++) {
      const wBase = k * D;
      for (let j = 0; j < D; j++) {
        let sum = 0;
        for (let i = 0; i < T; i++) {
          sum += this.act[i * H + k] * dOut[i * D + j];
        }
        dW2[wBase + j] += sum;
      }
    }

    // dAct
    for (let i = 0; i < T; i++) {
      const dRow = i * D;
      const aRow = i * H;
      for (let k = 0; k < H; k++) {
        let sum = 0;
        const wBase = k * D;
        for (let j = 0; j < D; j++) {
          sum += dOut[dRow + j] * this.W2.w[wBase + j];
        }
        dAct[aRow + k] = sum;
      }
    }

    // dPre = dAct * dGELU(preAct)
    for (let i = 0; i < dAct.length; i++) dAct[i] *= dGelu(this.preAct[i]);

    // Backprop first dense: pre = xW1 + b1
    // dW1 += x^T dPre ; db1 += sum dPre ; dX = dPre W1^T
    const dW1 = this.W1.g;
    const db1 = this.b1.g;

    for (let j = 0; j < H; j++) db1[j] += 0;
    for (let i = 0; i < T; i++) {
      const pRow = i * H;
      for (let j = 0; j < H; j++) db1[j] += dAct[pRow + j];
    }

    for (let k = 0; k < D; k++) {
      const wBase = k * H;
      for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let i = 0; i < T; i++) sum += x[i * D + k] * dAct[i * H + j];
        dW1[wBase + j] += sum;
      }
    }

    dX.fill(0);
    for (let i = 0; i < T; i++) {
      const xRow = i * D;
      const pRow = i * H;
      for (let k = 0; k < D; k++) {
        let sum = 0;
        const wBase = k * H;
        for (let j = 0; j < H; j++) {
          sum += dAct[pRow + j] * this.W1.w[wBase + j];
        }
        dX[xRow + k] = sum;
      }
    }
  }
}

// --------------------------
// Transformer block
// --------------------------
class TransformerBlock {
  readonly ln1: LayerNorm;
  readonly attn: MultiHeadSelfAttention;
  readonly ln2: LayerNorm;
  readonly ffn: FFN;

  // buffers
  private x1: Float64Array = new Float64Array(0); // ln1 out
  private attnOut: Float64Array = new Float64Array(0);
  private x2: Float64Array = new Float64Array(0); // ln2 out
  private ffnOut: Float64Array = new Float64Array(0);

  // grads buffers
  private dX1: Float64Array = new Float64Array(0);
  private dAttnIn: Float64Array = new Float64Array(0);
  private dX2: Float64Array = new Float64Array(0);
  private dFfnIn: Float64Array = new Float64Array(0);

  private lastT = 0;
  private readonly D: number;
  private readonly eps: number;

  constructor(
    embDim: number,
    numHeads: number,
    ffnMultiplier: number,
    eps: number,
    rng: () => number,
  ) {
    this.D = embDim;
    this.eps = eps;
    this.ln1 = new LayerNorm(embDim, rng);
    this.attn = new MultiHeadSelfAttention(embDim, numHeads, rng);
    this.ln2 = new LayerNorm(embDim, rng);
    this.ffn = new FFN(embDim, ffnMultiplier, rng);
  }

  ensure(T: number): void {
    const need = T * this.D;
    if (this.x1.length !== need) {
      this.x1 = new Float64Array(need);
      this.attnOut = new Float64Array(need);
      this.x2 = new Float64Array(need);
      this.ffnOut = new Float64Array(need);

      this.dX1 = new Float64Array(need);
      this.dAttnIn = new Float64Array(need);
      this.dX2 = new Float64Array(need);
      this.dFfnIn = new Float64Array(need);
    }
    this.lastT = T;
  }

  forward(x: Float64Array, y: Float64Array, T: number): void {
    this.ensure(T);

    // LN1
    this.ln1.forward(x, this.x1, T, this.eps);

    // Attention
    this.attn.forward(this.x1, this.attnOut, T);

    // Residual: r1 = x + attnOut -> store in y for now
    for (let i = 0; i < y.length; i++) y[i] = x[i] + this.attnOut[i];

    // LN2 on residual
    this.ln2.forward(y, this.x2, T, this.eps);

    // FFN
    this.ffn.forward(this.x2, this.ffnOut, T);

    // Residual2: y = residual + ffnOut
    for (let i = 0; i < y.length; i++) y[i] += this.ffnOut[i];
  }

  backward(
    x: Float64Array,
    yResidual1: Float64Array,
    dY: Float64Array,
    dX: Float64Array,
  ): void {
    // x is input to block; yResidual1 is x + attnOut (the vector used as ln2 input)
    const T = this.lastT;
    const need = T * this.D;

    // dY -> split through residual2:
    // y = yResidual1 + ffnOut
    // d(yResidual1) += dY, d(ffnOut) += dY
    // backprop FFN through ln2 etc
    // dFfnOut = dY
    // dLN2Out = dFFNIn = dFFN input? We'll compute.

    // d through FFN: inputs x2 (ln2 output)
    // We need gradient wrt x2 from ffn.backward, then back through ln2 to yResidual1
    this.dFfnIn.set(dY); // gradient wrt ffn output is dY
    this.ffn.backward(this.x2, this.dFfnIn, this.dX2);

    // Back through LN2: ln2 input is yResidual1, output is x2
    this.ln2.backward(yResidual1, this.dX2, this.dAttnIn, T);

    // Add residual path from residual2: d(yResidual1) also gets direct dY
    for (let i = 0; i < need; i++) this.dAttnIn[i] += dY[i];

    // Back through residual1: yResidual1 = x + attnOut
    // so d(attnOut)=d(yResidual1), d(x) gets also d(yResidual1)
    // Backprop attention: input was x1=ln1(x), output attnOut
    // Need gradient wrt x1 from attention and add into path to LN1
    this.dX1.fill(0);
    this.attn.backward(this.x1, this.dAttnIn, this.dX1);

    // Back through LN1: ln1 input x, output x1
    this.ln1.backward(x, this.dX1, this.dX1, /* reuse as dX_ln1 */ T);

    // Now combine with residual direct: dX = d(yResidual1) + dX_ln1
    for (let i = 0; i < need; i++) dX[i] = this.dAttnIn[i] + this.dX1[i];
  }
}

// --------------------------
// Main model
// --------------------------
export class FusionTemporalTransformerRegression {
  private readonly cfg: FusionTemporalTransformerConfig;
  private readonly pool = new Float64ArrayPool();

  // init state
  #isInitialized = false;
  #inDim = 0;
  #outDim = 0;

  // normalization
  private xStats: Welford | null = null;
  private yStats: Welford | null = null;
  private xStd: Float64Array = new Float64Array(0);
  private yStd: Float64Array = new Float64Array(0);

  // positional encoding
  private posEnc: Float64Array = new Float64Array(0); // [maxSeqLen, embDim]

  // multi-scale conv
  private convs: TemporalConv1D[] = [];
  private scaleEmb: Parameter[] = []; // one vector per scale [embDim]

  // fusion gate: input = concat(scaleEmbeddings per t) => dim = S*D ; output = S gates
  private Wg: Parameter | null = null; // [S*D, S]
  private bg: Parameter | null = null; // [S]

  // transformer blocks
  private blocks: TransformerBlock[] = [];

  // output head (per timestep)
  private Wy: Parameter | null = null; // [D, outDim]
  private by: Parameter | null = null; // [outDim]

  // optimizer
  private adam: Adam;

  // training state
  #sampleCount = 0;
  #runningLoss = 0; // EMA-ish (running average)
  #prevLoss = Infinity;
  #converged = false;
  #driftCount = 0;
  private adwin: AdwinLite;

  // caches / buffers (reused)
  private xNorm: Float64Array = new Float64Array(0); // [T,inDim]
  private yNorm: Float64Array = new Float64Array(0); // [T,outDim]
  private fused: Float64Array = new Float64Array(0); // [T,D]
  private fusedIn: Float64Array = new Float64Array(0); // [T,D] input to first block
  private blockOut: Float64Array[] = []; // [numBlocks] each [T,D]
  private convOuts: Float64Array[] = []; // per scale [outLen,D]
  private convUps: Float64Array[] = []; // per scale upsampled to [T,D]
  private gates: Float64Array = new Float64Array(0); // [T,S]
  private yHatNorm: Float64Array = new Float64Array(0); // [T,outDim]
  private dYHatNorm: Float64Array = new Float64Array(0); // [T,outDim]
  private dFused: Float64Array = new Float64Array(0); // [T,D]
  private dBlock: Float64Array[] = []; // grads per block [T,D]
  private dConvUps: Float64Array[] = []; // per scale [T,D]
  private dConvOuts: Float64Array[] = []; // per scale [outLen,D]
  private dXNorm: Float64Array = new Float64Array(0); // [T,inDim]
  private tmpTD: Float64Array = new Float64Array(0); // [T,D] scratch

  // last seen input for forecasting
  private lastXRaw: Float64Array = new Float64Array(0); // [T,inDim]
  private lastT = 0;

  // deterministic RNG
  private rngState = 123456789;
  private rng = (): number => {
    // xorshift32
    let x = this.rngState | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x | 0;
    // [0,1)
    return ((x >>> 0) / 4294967296);
  };

  constructor(config: Partial<FusionTemporalTransformerConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
    this.adam = new Adam(this.cfg.beta1, this.cfg.beta2, this.cfg.epsilon);
    this.adwin = new AdwinLite(this.cfg.adwinDelta, 256);
  }

  reset(): void {
    this.#isInitialized = false;
    this.#inDim = 0;
    this.#outDim = 0;

    this.xStats = null;
    this.yStats = null;
    this.xStd = new Float64Array(0);
    this.yStd = new Float64Array(0);

    this.posEnc = new Float64Array(0);

    this.convs = [];
    this.scaleEmb = [];
    this.Wg = null;
    this.bg = null;

    this.blocks = [];

    this.Wy = null;
    this.by = null;

    this.adam.reset();
    this.#sampleCount = 0;
    this.#runningLoss = 0;
    this.#prevLoss = Infinity;
    this.#converged = false;
    this.#driftCount = 0;
    this.adwin.reset();

    this.xNorm = new Float64Array(0);
    this.yNorm = new Float64Array(0);
    this.fused = new Float64Array(0);
    this.fusedIn = new Float64Array(0);
    this.blockOut = [];
    this.convOuts = [];
    this.convUps = [];
    this.gates = new Float64Array(0);
    this.yHatNorm = new Float64Array(0);
    this.dYHatNorm = new Float64Array(0);
    this.dFused = new Float64Array(0);
    this.dBlock = [];
    this.dConvUps = [];
    this.dConvOuts = [];
    this.dXNorm = new Float64Array(0);
    this.tmpTD = new Float64Array(0);

    this.lastXRaw = new Float64Array(0);
    this.lastT = 0;

    this.pool.reset();
  }

  private initIfNeeded(x: number[][], y: number[][]): void {
    if (this.#isInitialized) return;

    const T = x.length;
    if (T <= 0) throw new Error("xCoordinates must be non-empty");
    const inDim = x[0].length;
    const outDim = y[0].length;
    if (inDim <= 0 || outDim <= 0) throw new Error("Invalid dimensions");
    if (y.length !== T) {
      throw new Error(
        "yCoordinates must have same seqLen as xCoordinates (seq-to-seq)",
      );
    }

    this.#inDim = inDim;
    this.#outDim = outDim;

    // stats
    this.xStats = new Welford(inDim);
    this.yStats = new Welford(outDim);
    this.xStd = new Float64Array(inDim);
    this.yStd = new Float64Array(outDim);

    // pos enc [maxSeqLen, D]
    this.buildPositionalEncoding();

    // multi-scale convs
    const D = this.cfg.embeddingDim;
    const K = this.cfg.temporalKernelSize;
    const scales = this.cfg.temporalScales;
    this.convs = new Array(scales.length);
    this.scaleEmb = new Array(scales.length);
    for (let si = 0; si < scales.length; si++) {
      this.convs[si] = new TemporalConv1D(inDim, D, K, scales[si], this.rng);
      this.scaleEmb[si] = new Parameter(D, 0.02, this.rng);
    }

    // fusion gating
    const S = scales.length;
    this.Wg = new Parameter((S * D) * S, Math.sqrt(2 / (S * D)), this.rng);
    this.bg = new Parameter(S, 0.0, () => 0.5);
    this.bg.w.fill(0);

    // blocks
    this.blocks = new Array(this.cfg.numBlocks);
    for (let b = 0; b < this.cfg.numBlocks; b++) {
      this.blocks[b] = new TransformerBlock(
        D,
        this.cfg.numHeads,
        this.cfg.ffnMultiplier,
        this.cfg.epsilon,
        this.rng,
      );
    }

    // output head per timestep
    this.Wy = new Parameter(D * outDim, Math.sqrt(2 / D), this.rng);
    this.by = new Parameter(outDim, 0.0, () => 0.5);
    this.by.w.fill(0);

    // buffers sized lazily per observed T; but allocate for max T to reduce churn
    this.ensureBuffers(Math.min(this.cfg.maxSequenceLength, T));

    this.#isInitialized = true;
  }

  private buildPositionalEncoding(): void {
    const maxT = this.cfg.maxSequenceLength;
    const D = this.cfg.embeddingDim;
    const pe = new Float64Array(maxT * D);

    // PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
    for (let pos = 0; pos < maxT; pos++) {
      const base = pos * D;
      for (let i = 0; i < D; i += 2) {
        const div = Math.pow(10000, i / D);
        pe[base + i] = Math.sin(pos / div);
        if (i + 1 < D) pe[base + i + 1] = Math.cos(pos / div);
      }
    }
    this.posEnc = pe;
  }

  private ensureBuffers(T: number): void {
    const inD = this.#inDim;
    const outD = this.#outDim;
    const D = this.cfg.embeddingDim;
    const S = this.cfg.temporalScales.length;

    const xNeed = T * inD;
    if (this.xNorm.length !== xNeed) {
      this.xNorm = new Float64Array(xNeed);
      this.dXNorm = new Float64Array(xNeed);
      this.lastXRaw = new Float64Array(xNeed);
    }

    const yNeed = T * outD;
    if (this.yNorm.length !== yNeed) {
      this.yNorm = new Float64Array(yNeed);
      this.yHatNorm = new Float64Array(yNeed);
      this.dYHatNorm = new Float64Array(yNeed);
    }

    const tdNeed = T * D;
    if (this.fused.length !== tdNeed) {
      this.fused = new Float64Array(tdNeed);
      this.fusedIn = new Float64Array(tdNeed);
      this.dFused = new Float64Array(tdNeed);
      this.tmpTD = new Float64Array(tdNeed);

      this.blockOut = new Array(this.cfg.numBlocks);
      this.dBlock = new Array(this.cfg.numBlocks);
      for (let b = 0; b < this.cfg.numBlocks; b++) {
        this.blockOut[b] = new Float64Array(tdNeed);
        this.dBlock[b] = new Float64Array(tdNeed);
      }
    }

    const gatesNeed = T * S;
    if (this.gates.length !== gatesNeed) {
      this.gates = new Float64Array(gatesNeed);
    }

    // conv outputs per scale: variable length; allocate upsample + grad as [T,D]
    if (this.convUps.length !== S) {
      this.convUps = new Array(S);
      this.dConvUps = new Array(S);
      this.convOuts = new Array(S);
      this.dConvOuts = new Array(S);
    }
    for (let si = 0; si < S; si++) {
      if (!this.convUps[si] || this.convUps[si].length !== tdNeed) {
        this.convUps[si] = new Float64Array(tdNeed);
        this.dConvUps[si] = new Float64Array(tdNeed);
      }
      // convOuts allocated per forward based on outLen
      this.convOuts[si] = this.convOuts[si] || new Float64Array(0);
      this.dConvOuts[si] = this.dConvOuts[si] || new Float64Array(0);
    }
  }

  private computeLearningRate(): number {
    const base = this.cfg.learningRate;
    const t = this.adam.stepCount + 1;
    const warm = this.cfg.warmupSteps;
    const total = this.cfg.totalSteps;

    // warmup then cosine decay:
    // lr = base * min(1, t/warm) * 0.5*(1+cos(pi*(t-warm)/(total-warm)))
    const warmFactor = t < warm ? (t / warm) : 1.0;
    let cosine = 1.0;
    if (t > warm) {
      const denom = Math.max(1, total - warm);
      const p = clampInt(t - warm, 0, denom) / denom;
      cosine = 0.5 * (1.0 + Math.cos(Math.PI * p));
    }
    return base * warmFactor * cosine;
  }

  private normalizeSequence(
    xRaw: Float64Array,
    yRaw: Float64Array,
    T: number,
  ): void {
    // Use current stats to normalize (no leakage); if count<2 use std=1.
    const xStats = this.xStats!;
    const yStats = this.yStats!;
    xStats.std(this.xStd, this.cfg.epsilon);
    yStats.std(this.yStd, this.cfg.epsilon);

    // xNorm = (x - mean)/std
    const inD = this.#inDim;
    for (let t = 0; t < T; t++) {
      const base = t * inD;
      for (let i = 0; i < inD; i++) {
        this.xNorm[base + i] = (xRaw[base + i] - xStats.mean[i]) / this.xStd[i];
      }
    }

    // yNorm
    const outD = this.#outDim;
    for (let t = 0; t < T; t++) {
      const base = t * outD;
      for (let i = 0; i < outD; i++) {
        this.yNorm[base + i] = (yRaw[base + i] - yStats.mean[i]) / this.yStd[i];
      }
    }
  }

  private updateStatsFromRaw(
    xRaw: Float64Array,
    yRaw: Float64Array,
    T: number,
  ): void {
    const inD = this.#inDim;
    const outD = this.#outDim;
    const xStats = this.xStats!;
    const yStats = this.yStats!;
    for (let t = 0; t < T; t++) {
      xStats.update(xRaw, t * inD);
      yStats.update(yRaw, t * outD);
    }
  }

  private forward(T: number): void {
    const D = this.cfg.embeddingDim;
    const S = this.cfg.temporalScales.length;

    // Multi-scale conv -> convOuts[si], then upsample to [T,D] with scale embedding + pos encoding
    for (let si = 0; si < S; si++) {
      const conv = this.convs[si];
      const outLen = conv.outLen(T);
      const need = outLen * D;
      if (this.convOuts[si].length !== need) {
        this.convOuts[si] = new Float64Array(need);
        this.dConvOuts[si] = new Float64Array(need);
      } else {
        this.convOuts[si].fill(0);
        this.dConvOuts[si].fill(0);
      }

      conv.forward(this.xNorm, T, this.convOuts[si]);

      // Add scale embedding & pos enc at that scale and upsample to T
      const up = this.convUps[si];
      up.fill(0);

      const stride = this.cfg.temporalScales[si];
      const scaleE = this.scaleEmb[si].w;

      // For each t in 0..T-1, map to idx = floor(t/stride)
      for (let t = 0; t < T; t++) {
        const idx = (t / stride) | 0;
        const src = idx * D;
        const dst = t * D;
        // up[t] = convOut[idx] + scaleEmb + posEnc[t]
        const peBase = t * D;
        for (let d = 0; d < D; d++) {
          up[dst + d] = this.convOuts[si][src + d] + scaleE[d] +
            this.posEnc[peBase + d];
        }
      }
    }

    // Gated fusion:
    // gates = sigmoid( concat(E_s) * Wg + bg ) => [T,S]
    // fused[t] = sum_s gates[t,s] * E_s[t]
    const Wg = this.Wg!.w;
    const bg = this.bg!.w;

    this.fused.fill(0);
    this.gates.fill(0);

    // concat vector length S*D computed on the fly to avoid allocating concat buffer:
    // gate_s = sigmoid( sum_{k=0..S*D-1} concat[k]*Wg[k,S] + bg[s] )
    // fused_d = sum_s gate_s * E_s_d
    for (let t = 0; t < T; t++) {
      const dst = t * D;
      const gateBase = t * S;

      // compute each gate s
      for (let s = 0; s < S; s++) {
        let sum = bg[s];
        // concat order: [E0, E1, ...] each D
        let colBase = s; // Wg[(k)*S + s]
        for (let si = 0; si < S; si++) {
          const src = t * D;
          const E = this.convUps[si];
          for (let d = 0; d < D; d++) {
            const k = si * D + d;
            sum += E[src + d] * Wg[k * S + s];
          }
        }
        const g = sigmoid(sum);
        this.gates[gateBase + s] = g;
      }

      // fused = Σ_s g_s ⊙ E_s
      for (let d = 0; d < D; d++) {
        let sum = 0;
        for (let si = 0; si < S; si++) {
          sum += this.gates[gateBase + si] * this.convUps[si][dst + d];
        }
        this.fused[dst + d] = sum;
      }
    }

    // Transformer blocks
    this.fusedIn.set(this.fused);
    let inp = this.fusedIn;
    for (let b = 0; b < this.blocks.length; b++) {
      const out = this.blockOut[b];
      this.blocks[b].forward(inp, out, T);
      inp = out;
    }

    // Output head per timestep: yHatNorm = H * Wy + by
    matmulAddBias(
      inp,
      this.Wy!.w,
      this.by!.w,
      this.yHatNorm,
      T,
      D,
      this.#outDim,
    );
  }

  private computeLossAndGrad(T: number, outlierDownWeight: number): number {
    // MSE across all time steps and outputs:
    // L = (1/(2*N)) Σ (y - yhat)^2
    // dYhat = (1/N) * (yhat - y) * w_outlier
    const outD = this.#outDim;
    const N = T * outD;
    const invN = 1.0 / Math.max(1, N);
    let loss = 0;

    const dY = this.dYHatNorm;
    const y = this.yNorm;
    const yh = this.yHatNorm;

    for (let i = 0; i < N; i++) {
      const r = yh[i] - y[i];
      loss += 0.5 * r * r * invN;
      dY[i] = r * invN * outlierDownWeight;
    }
    return loss;
  }

  private backward(T: number): number {
    // Backprop output head -> transformer -> fusion -> convs
    const D = this.cfg.embeddingDim;
    const outD = this.#outDim;

    // output head grads:
    // yHat = H*Wy + by
    // dWy = H^T dY, dby = sum dY, dH = dY * Wy^T
    const dWy = this.Wy!.g;
    const dby = this.by!.g;
    const Wy = this.Wy!.w;

    // dby
    for (let j = 0; j < outD; j++) dby[j] += 0;
    for (let t = 0; t < T; t++) {
      const base = t * outD;
      for (let j = 0; j < outD; j++) dby[j] += this.dYHatNorm[base + j];
    }

    // dWy
    const Hlast = this.blockOut[this.blocks.length - 1];
    for (let k = 0; k < D; k++) {
      const wBase = k * outD;
      for (let j = 0; j < outD; j++) {
        let sum = 0;
        for (let t = 0; t < T; t++) {
          sum += Hlast[t * D + k] * this.dYHatNorm[t * outD + j];
        }
        dWy[wBase + j] += sum;
      }
    }

    // dHlast
    let dH = this.dBlock[this.blocks.length - 1];
    dH.fill(0);
    for (let t = 0; t < T; t++) {
      const dyBase = t * outD;
      const dhBase = t * D;
      for (let k = 0; k < D; k++) {
        let sum = 0;
        const wBase = k * outD;
        for (let j = 0; j < outD; j++) {
          sum += this.dYHatNorm[dyBase + j] * Wy[wBase + j];
        }
        dH[dhBase + k] = sum;
      }
    }

    // transformer blocks backward
    for (let b = this.blocks.length - 1; b >= 0; b--) {
      const x = b === 0 ? this.fusedIn : this.blockOut[b - 1];
      const yResidual1 = this.blockOut[b]; // used as ln2 input in block, but our block expects it
      // Our TransformerBlock.backward expects yResidual1 = x + attnOut (residual1).
      // We didn't store it explicitly. Reconstruct yResidual1 approximately:
      // In forward, blockOut[b] is final output. But ln2 input is x + attnOut.
      // For correctness we need residual1; we can recompute by doing a cheap re-forward of ln1+attn,
      // but that allocates less than full caching. Here we recompute residual1 in tmpTD.
      this.tmpTD.set(x);
      // recompute residual1: tmpTD = x + attn(ln1(x))
      // Use block internals: forward LN1 into block.x1, then attn into block.attnOut, sum
      const blk = this.blocks[b] as any;
      blk.ln1.forward(x, blk.x1, T, this.cfg.epsilon);
      blk.attn.forward(blk.x1, blk.attnOut, T);
      for (let i = 0; i < this.tmpTD.length; i++) {
        this.tmpTD[i] = x[i] + blk.attnOut[i];
      }

      const dIn = b === 0 ? this.dFused : this.dBlock[b - 1];
      blk.backward(x, this.tmpTD, this.dBlock[b], dIn);
    }

    // Backprop fusion gates and conv upsamples
    const S = this.cfg.temporalScales.length;
    const Wg = this.Wg!.w;
    const dWg = this.Wg!.g;
    const bg = this.bg!.w;
    const dbg = this.bg!.g;

    // dConvUps[si] and fusion params grads
    for (let si = 0; si < S; si++) this.dConvUps[si].fill(0);

    // fused[t,d] = Σ_s g[t,s] * E_s[t,d]
    // g[t,s] = sigmoid(z[t,s]); z[t,s] = bg[s] + Σ_k concat[k]*Wg[k,s]
    // dE_s += dFused * g
    // dG_s += Σ_d dFused_d * E_s_d
    // dZ_s = dG_s * g*(1-g)
    // dWg += concat^T dZ ; dbg += Σ dZ ; dE also contributes via concat into dWg path? (yes)
    // concat is [E0..ES-1], so dE from gate linear too:
    // dConcat_k += Σ_s dZ_s * Wg[k,s]
    // then distribute dConcat to each E_s.

    // We'll do per time-step to keep locality.
    const Ddim = D;
    const dZ = new Float64Array(S); // small; acceptable
    const dG = new Float64Array(S);

    for (let t = 0; t < T; t++) {
      const td = t * Ddim;
      const gs = t * S;

      // dE from fused direct + initialize dG
      for (let si = 0; si < S; si++) dG[si] = 0;

      for (let si = 0; si < S; si++) {
        const g = this.gates[gs + si];
        const E = this.convUps[si];
        const dE = this.dConvUps[si];
        for (let d = 0; d < Ddim; d++) {
          const df = this.dFused[td + d];
          dE[td + d] += df * g;
          dG[si] += df * E[td + d];
        }
      }

      // dZ = dG * g*(1-g)
      for (let si = 0; si < S; si++) {
        const g = this.gates[gs + si];
        dZ[si] = dG[si] * g * (1.0 - g);
        dbg[si] += dZ[si];
      }

      // dWg and dConcat
      // dWg[k,s] += concat[k]*dZ[s]
      // dConcat[k] += Σ_s dZ[s]*Wg[k,s]
      // We avoid allocating dConcat array; directly add to corresponding dConvUps via concat mapping.
      for (let si = 0; si < S; si++) {
        // concat chunk for scale si: E_si[t, d]
        const E = this.convUps[si];
        const dE = this.dConvUps[si];

        for (let d = 0; d < Ddim; d++) {
          const k = si * Ddim + d;
          const concatVal = E[td + d];
          const wRow = k * S;

          // dWg row
          for (let s = 0; s < S; s++) dWg[wRow + s] += concatVal * dZ[s];

          // dConcat contribution -> dE += Σ_s dZ[s]*Wg[k,s]
          let sum = 0;
          for (let s = 0; s < S; s++) sum += dZ[s] * Wg[wRow + s];
          dE[td + d] += sum;
        }
      }
    }

    // Backprop into scale embeddings and conv outputs (upsample inverse)
    // up[t] = convOut[idx] + scaleEmb + posEnc => dScaleEmb += sum_t dUp[t]
    // dConvOut[idx] accumulates from all t mapping to idx
    // conv backward to xNorm
    this.dXNorm.fill(0);
    for (let si = 0; si < S; si++) {
      const stride = this.cfg.temporalScales[si];
      const outLen = this.convs[si].outLen(T);
      const need = outLen * Ddim;
      const dConvOut = this.dConvOuts[si];
      if (dConvOut.length !== need) this.dConvOuts[si] = new Float64Array(need);
      else dConvOut.fill(0);

      const dUp = this.dConvUps[si];
      const dScale = this.scaleEmb[si].g;

      // dScaleEmb
      for (let d = 0; d < Ddim; d++) dScale[d] += 0;
      for (let t = 0; t < T; t++) {
        const td = t * Ddim;
        const idx = (t / stride) | 0;
        const od = idx * Ddim;
        for (let d = 0; d < Ddim; d++) {
          const g = dUp[td + d];
          dScale[d] += g;
          dConvOut[od + d] += g;
          // posEnc has no params
        }
      }

      // backprop conv: dConvOut -> dXNorm via conv.backward
      this.convs[si].backward(dConvOut, this.dXNorm);
    }

    // Gradient norm (approx): sum norms of key grads
    let gnorm = 0;
    gnorm += l2Norm(this.Wy!.g);
    gnorm += l2Norm(this.Wg!.g);
    for (let si = 0; si < S; si++) gnorm += l2Norm(this.convs[si].W.g);
    return gnorm;
  }

  private applyAdam(lr: number): void {
    const wd = this.cfg.regularizationStrength;

    // conv params
    for (let si = 0; si < this.convs.length; si++) {
      this.adam.apply(this.convs[si].W, lr, wd);
      this.adam.apply(this.convs[si].b, lr, 0.0);
      this.adam.apply(this.scaleEmb[si], lr, wd);
    }

    // fusion
    this.adam.apply(this.Wg!, lr, wd);
    this.adam.apply(this.bg!, lr, 0.0);

    // blocks params
    for (let b = 0; b < this.blocks.length; b++) {
      const blk = this.blocks[b];
      this.adam.apply(blk.ln1.gamma, lr, 0.0);
      this.adam.apply(blk.ln1.beta, lr, 0.0);
      this.adam.apply(blk.attn.Wq, lr, wd);
      this.adam.apply(blk.attn.Wk, lr, wd);
      this.adam.apply(blk.attn.Wv, lr, wd);
      this.adam.apply(blk.attn.Wo, lr, wd);
      this.adam.apply(blk.attn.bq, lr, 0.0);
      this.adam.apply(blk.attn.bk, lr, 0.0);
      this.adam.apply(blk.attn.bv, lr, 0.0);
      this.adam.apply(blk.attn.bo, lr, 0.0);
      this.adam.apply(blk.ln2.gamma, lr, 0.0);
      this.adam.apply(blk.ln2.beta, lr, 0.0);
      this.adam.apply(blk.ffn.W1, lr, wd);
      this.adam.apply(blk.ffn.b1, lr, 0.0);
      this.adam.apply(blk.ffn.W2, lr, wd);
      this.adam.apply(blk.ffn.b2, lr, 0.0);
    }

    // output
    this.adam.apply(this.Wy!, lr, wd);
    this.adam.apply(this.by!, lr, 0.0);
  }

  private computeOutlierWeight(
    T: number,
  ): { isOutlier: boolean; weight: number } {
    // r = (y - yhat) / yStd ; mark outlier if any |r| > threshold
    const outD = this.#outDim;
    const N = T * outD;
    const thr = this.cfg.outlierThreshold;

    // yStd for normalized space is ~1, but we detect using normalized residual directly.
    // Since both are normalized, residual ~ z-score.
    let isOutlier = false;
    for (let i = 0; i < N; i++) {
      const r = this.yNorm[i] - this.yHatNorm[i]; // in normalized units
      if (Math.abs(r) > thr) {
        isOutlier = true;
        break;
      }
    }
    return { isOutlier, weight: isOutlier ? 0.1 : 1.0 };
  }

  /**
   * Incremental online fit for one sequence sample.
   *
   * @param input {FitInput}
   * @returns {FitResult}
   *
   * @example
   * const m = new FusionTemporalTransformerRegression();
   * const r = m.fitOnline({ xCoordinates: [[1,2],[2,3]], yCoordinates: [[0.1],[0.2]] });
   * console.log(r.loss);
   */
  fitOnline(input: FitInput): FitResult {
    const x = input.xCoordinates;
    const y = input.yCoordinates;
    this.initIfNeeded(x, y);

    const Traw = x.length;
    const T = clampInt(Traw, 1, this.cfg.maxSequenceLength);
    if (y.length !== x.length) {
      throw new Error(
        "Sequence-to-sequence training requires yCoordinates.length === xCoordinates.length",
      );
    }
    if (T !== Traw) {
      throw new Error(
        `seqLen exceeds maxSequenceLength (${this.cfg.maxSequenceLength})`,
      );
    }

    this.ensureBuffers(T);
    this.lastT = T;

    // pack raw into typed arrays (one unavoidable allocation avoided by reusing lastXRaw? we reuse lastXRaw and a pooled y buffer)
    const inD = this.#inDim;
    const outD = this.#outDim;

    // Reuse lastXRaw for raw X; get pooled buffer for y raw
    const xRaw = this.lastXRaw;
    const yRaw = this.pool.get(T * outD);

    for (let t = 0; t < T; t++) {
      const xb = t * inD;
      const yb = t * outD;
      const xr = x[t];
      const yr = y[t];
      for (let i = 0; i < inD; i++) xRaw[xb + i] = xr[i];
      for (let i = 0; i < outD; i++) yRaw[yb + i] = yr[i];
    }

    // Normalize using current stats (no leakage), then forward/backward/update, then update stats
    this.normalizeSequence(xRaw, yRaw, T);

    // Forward
    this.forward(T);

    // Outlier detection (based on normalized residual)
    const ow = this.computeOutlierWeight(T);

    // Loss + grad wrt yhat
    const loss = this.computeLossAndGrad(T, ow.weight);

    // Drift detection on loss
    const driftDetected = this.adwin.push(loss);
    if (driftDetected) {
      this.#driftCount++;
      // Reset optimizer moments & running loss on drift for stability
      this.adam.reset();
      this.#runningLoss = 0;
      this.#prevLoss = Infinity;
      this.#converged = false;
      // (Optional) reset normalization stats as requested:
      this.xStats!.reset();
      this.yStats!.reset();
    }

    // Backward
    const gnorm = this.backward(T);

    // LR schedule and update
    const lr = this.computeLearningRate();
    this.applyAdam(lr);

    // Update running accuracy/loss
    this.#sampleCount++;
    // running average loss
    const n = this.#sampleCount;
    this.#runningLoss += (loss - this.#runningLoss) / n;
    const converged =
      Math.abs(this.#prevLoss - loss) < this.cfg.convergenceThreshold;
    this.#converged = converged;
    this.#prevLoss = loss;

    // Update stats after training step
    this.updateStatsFromRaw(xRaw, yRaw, T);

    this.pool.release(yRaw);

    return {
      loss,
      gradientNorm: gnorm,
      effectiveLearningRate: lr,
      isOutlier: ow.isOutlier,
      converged,
      sampleIndex: this.#sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict next futureSteps using a simple linear extrapolation of x features based on last sequence.
   * Uses current model in normalized space, then denormalizes outputs.
   *
   * @param futureSteps number
   * @returns PredictionResult
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.#isInitialized || this.#sampleCount === 0 || this.lastT === 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.#sampleCount,
        isModelReady: false,
      };
    }
    const steps = clampInt(futureSteps | 0, 1, this.cfg.maxSequenceLength);
    const T = this.lastT;
    const inD = this.#inDim;
    const outD = this.#outDim;

    // Build future xRaw in a pooled buffer: [steps, inD]
    const xFuture = this.pool.get(steps * inD);

    // Linear extrapolation using last two points (per feature):
    // x_{t+k} = x_{t} + k*(x_{t} - x_{t-1})
    const xRaw = this.lastXRaw;
    const lastBase = (T - 1) * inD;
    const prevBase = (T - 2 >= 0 ? (T - 2) : (T - 1)) * inD;

    for (let k = 1; k <= steps; k++) {
      const dst = (k - 1) * inD;
      for (let i = 0; i < inD; i++) {
        const last = xRaw[lastBase + i];
        const prev = xRaw[prevBase + i];
        xFuture[dst + i] = last + k * (last - prev);
      }
    }

    // Normalize future x using current stats
    const xStats = this.xStats!;
    xStats.std(this.xStd, this.cfg.epsilon);

    const xNormFuture = this.pool.get(steps * inD);
    for (let t = 0; t < steps; t++) {
      const base = t * inD;
      for (let i = 0; i < inD; i++) {
        xNormFuture[base + i] = (xFuture[base + i] - xStats.mean[i]) /
          this.xStd[i];
      }
    }

    // Temporarily swap xNorm buffer for forward call: reuse model buffers by copying into xNorm[0..steps)
    this.ensureBuffers(steps);
    this.xNorm.set(xNormFuture);

    // Forward
    this.forward(steps);

    // Denormalize predictions
    const yStats = this.yStats!;
    yStats.std(this.yStd, this.cfg.epsilon);

    const preds: SinglePrediction[] = new Array(steps);
    const se = Math.sqrt(Math.max(1e-12, this.#runningLoss)); // normalized MSE-ish -> SE approx
    for (let t = 0; t < steps; t++) {
      const base = t * outD;
      const predicted = new Array(outD);
      const lower = new Array(outD);
      const upper = new Array(outD);
      const stderr = new Array(outD);

      for (let i = 0; i < outD; i++) {
        // y = yhatNorm * std + mean
        const yhat = this.yHatNorm[base + i] * this.yStd[i] + yStats.mean[i];
        const err = se * this.yStd[i];
        predicted[i] = yhat;
        stderr[i] = err;
        lower[i] = yhat - 1.96 * err;
        upper[i] = yhat + 1.96 * err;
      }

      preds[t] = {
        predicted,
        lowerBound: lower,
        upperBound: upper,
        standardError: stderr,
      };
    }

    // restore buffers for lastT (optional); keep as steps for next call? We keep lastXRaw unchanged.
    this.pool.release(xFuture);
    this.pool.release(xNormFuture);

    return {
      predictions: preds,
      accuracy: 1 / (1 + this.#runningLoss),
      sampleCount: this.#sampleCount,
      isModelReady: true,
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this.#isInitialized) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }
    const xs = this.xStats!;
    const ys = this.yStats!;
    xs.std(this.xStd, this.cfg.epsilon);
    ys.std(this.yStd, this.cfg.epsilon);

    return {
      inputMean: Array.from(xs.mean),
      inputStd: Array.from(this.xStd),
      outputMean: Array.from(ys.mean),
      outputStd: Array.from(this.yStd),
      count: xs.count,
    };
  }

  getModelSummary(): ModelSummary {
    const totalParams = this.#isInitialized ? this.countParams() : 0;
    return {
      isInitialized: this.#isInitialized,
      inputDimension: this.#inDim,
      outputDimension: this.#outDim,
      numBlocks: this.cfg.numBlocks,
      embeddingDim: this.cfg.embeddingDim,
      numHeads: this.cfg.numHeads,
      temporalScales: this.cfg.temporalScales.slice(),
      totalParameters: totalParams,
      sampleCount: this.#sampleCount,
      accuracy: 1 / (1 + this.#runningLoss),
      converged: this.#converged,
      effectiveLearningRate: this.computeLearningRate(),
      driftCount: this.#driftCount,
    };
  }

  private countParams(): number {
    let n = 0;
    for (let si = 0; si < this.convs.length; si++) {
      n += this.convs[si].W.size + this.convs[si].b.size;
      n += this.scaleEmb[si].size;
    }
    n += this.Wg!.size + this.bg!.size;
    for (let b = 0; b < this.blocks.length; b++) {
      const blk = this.blocks[b];
      n += blk.ln1.gamma.size + blk.ln1.beta.size;
      n += blk.attn.Wq.size + blk.attn.Wk.size + blk.attn.Wv.size +
        blk.attn.Wo.size;
      n += blk.attn.bq.size + blk.attn.bk.size + blk.attn.bv.size +
        blk.attn.bo.size;
      n += blk.ln2.gamma.size + blk.ln2.beta.size;
      n += blk.ffn.W1.size + blk.ffn.b1.size + blk.ffn.W2.size +
        blk.ffn.b2.size;
    }
    n += this.Wy!.size + this.by!.size;
    return n;
  }

  getWeights(): WeightInfo {
    if (!this.#isInitialized) throw new Error("Model not initialized");

    const S = this.cfg.temporalScales.length;

    const temporalConvWeights: number[][][] = new Array(S);
    for (let si = 0; si < S; si++) {
      temporalConvWeights[si] = [
        Array.from(this.convs[si].W.w),
        Array.from(this.convs[si].b.w),
      ];
    }

    const scaleEmbeddings: number[][] = new Array(S);
    for (let si = 0; si < S; si++) {
      scaleEmbeddings[si] = Array.from(this.scaleEmb[si].w);
    }

    const positionalEncoding: number[][] = [];
    const D = this.cfg.embeddingDim;
    const maxT = this.cfg.maxSequenceLength;
    for (let t = 0; t < maxT; t++) {
      const row = new Array(D);
      const base = t * D;
      for (let d = 0; d < D; d++) row[d] = this.posEnc[base + d];
      positionalEncoding.push(row);
    }

    const fusionWeights: number[][][] = [[Array.from(this.Wg!.w)], [
      Array.from(this.bg!.w),
    ]];

    const attentionWeights: number[][][] = [];
    const ffnWeights: number[][][] = [];
    const layerNormParams: number[][][] = [];

    for (let b = 0; b < this.blocks.length; b++) {
      const blk = this.blocks[b];
      attentionWeights.push([
        Array.from(blk.attn.Wq.w),
        Array.from(blk.attn.Wk.w),
        Array.from(blk.attn.Wv.w),
        Array.from(blk.attn.Wo.w),
        Array.from(blk.attn.bq.w),
        Array.from(blk.attn.bk.w),
        Array.from(blk.attn.bv.w),
        Array.from(blk.attn.bo.w),
      ]);
      ffnWeights.push([
        Array.from(blk.ffn.W1.w),
        Array.from(blk.ffn.b1.w),
        Array.from(blk.ffn.W2.w),
        Array.from(blk.ffn.b2.w),
      ]);
      layerNormParams.push([
        Array.from(blk.ln1.gamma.w),
        Array.from(blk.ln1.beta.w),
        Array.from(blk.ln2.gamma.w),
        Array.from(blk.ln2.beta.w),
      ]);
    }

    const outputWeights: number[][][] = [[Array.from(this.Wy!.w)], [
      Array.from(this.by!.w),
    ]];

    // Moments (grouped, not exhaustive naming)
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    // collect a subset; full collection would be large but included as requested
    const collectParam = (p: Parameter) => {
      firstMoment.push([Array.from(p.m)]);
      secondMoment.push([Array.from(p.v)]);
    };
    for (let si = 0; si < S; si++) {
      collectParam(this.convs[si].W);
      collectParam(this.convs[si].b);
      collectParam(this.scaleEmb[si]);
    }
    collectParam(this.Wg!);
    collectParam(this.bg!);
    for (let b = 0; b < this.blocks.length; b++) {
      const blk = this.blocks[b];
      collectParam(blk.ln1.gamma);
      collectParam(blk.ln1.beta);
      collectParam(blk.attn.Wq);
      collectParam(blk.attn.Wk);
      collectParam(blk.attn.Wv);
      collectParam(blk.attn.Wo);
      collectParam(blk.attn.bq);
      collectParam(blk.attn.bk);
      collectParam(blk.attn.bv);
      collectParam(blk.attn.bo);
      collectParam(blk.ln2.gamma);
      collectParam(blk.ln2.beta);
      collectParam(blk.ffn.W1);
      collectParam(blk.ffn.b1);
      collectParam(blk.ffn.W2);
      collectParam(blk.ffn.b2);
    }
    collectParam(this.Wy!);
    collectParam(this.by!);

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
      updateCount: this.adam.stepCount,
    };
  }

  save(): string {
    const obj: any = {
      cfg: this.cfg,
      isInitialized: this.#isInitialized,
      inDim: this.#inDim,
      outDim: this.#outDim,
      sampleCount: this.#sampleCount,
      runningLoss: this.#runningLoss,
      prevLoss: this.#prevLoss,
      converged: this.#converged,
      driftCount: this.#driftCount,
      adamStep: this.adam.stepCount,
      rngState: this.rngState,

      xStats: this.xStats
        ? {
          count: this.xStats.count,
          mean: Array.from(this.xStats.mean),
          m2: Array.from(this.xStats.m2),
        }
        : null,
      yStats: this.yStats
        ? {
          count: this.yStats.count,
          mean: Array.from(this.yStats.mean),
          m2: Array.from(this.yStats.m2),
        }
        : null,

      posEnc: Array.from(this.posEnc),

      // weights
      convs: this.convs.map((c) => ({
        W: Array.from(c.W.w),
        b: Array.from(c.b.w),
        mW: Array.from(c.W.m),
        vW: Array.from(c.W.v),
        mb: Array.from(c.b.m),
        vb: Array.from(c.b.v),
        stride: c.stride,
        kernel: c.kernel,
      })),
      scaleEmb: this.scaleEmb.map((p) => ({
        w: Array.from(p.w),
        m: Array.from(p.m),
        v: Array.from(p.v),
      })),
      fusion: this.Wg && this.bg
        ? ({
          Wg: Array.from(this.Wg.w),
          bg: Array.from(this.bg.w),
          mWg: Array.from(this.Wg.m),
          vWg: Array.from(this.Wg.v),
          mbg: Array.from(this.bg.m),
          vbg: Array.from(this.bg.v),
        })
        : null,
      blocks: this.blocks.map((blk) => ({
        ln1g: Array.from(blk.ln1.gamma.w),
        ln1b: Array.from(blk.ln1.beta.w),
        ln2g: Array.from(blk.ln2.gamma.w),
        ln2b: Array.from(blk.ln2.beta.w),
        attn: {
          Wq: Array.from(blk.attn.Wq.w),
          Wk: Array.from(blk.attn.Wk.w),
          Wv: Array.from(blk.attn.Wv.w),
          Wo: Array.from(blk.attn.Wo.w),
          bq: Array.from(blk.attn.bq.w),
          bk: Array.from(blk.attn.bk.w),
          bv: Array.from(blk.attn.bv.w),
          bo: Array.from(blk.attn.bo.w),
          mWq: Array.from(blk.attn.Wq.m),
          vWq: Array.from(blk.attn.Wq.v),
          mWk: Array.from(blk.attn.Wk.m),
          vWk: Array.from(blk.attn.Wk.v),
          mWv: Array.from(blk.attn.Wv.m),
          vWv: Array.from(blk.attn.Wv.v),
          mWo: Array.from(blk.attn.Wo.m),
          vWo: Array.from(blk.attn.Wo.v),
        },
        ffn: {
          W1: Array.from(blk.ffn.W1.w),
          b1: Array.from(blk.ffn.b1.w),
          W2: Array.from(blk.ffn.W2.w),
          b2: Array.from(blk.ffn.b2.w),
          mW1: Array.from(blk.ffn.W1.m),
          vW1: Array.from(blk.ffn.W1.v),
          mW2: Array.from(blk.ffn.W2.m),
          vW2: Array.from(blk.ffn.W2.v),
        },
      })),
      out: this.Wy && this.by
        ? ({
          Wy: Array.from(this.Wy.w),
          by: Array.from(this.by.w),
          mWy: Array.from(this.Wy.m),
          vWy: Array.from(this.Wy.v),
          mby: Array.from(this.by.m),
          vby: Array.from(this.by.v),
        })
        : null,
    };

    return JSON.stringify(obj);
  }

  load(w: string): void {
    const obj = JSON.parse(w);

    this.reset();
    this.rngState = obj.rngState | 0;

    // restore cfg (but keep defaults for missing)
    (this as any).cfg = { ...DEFAULT_CONFIG, ...(obj.cfg || {}) };

    this.#isInitialized = !!obj.isInitialized;
    this.#inDim = obj.inDim | 0;
    this.#outDim = obj.outDim | 0;

    this.#sampleCount = obj.sampleCount | 0;
    this.#runningLoss = +obj.runningLoss;
    this.#prevLoss = +obj.prevLoss;
    this.#converged = !!obj.converged;
    this.#driftCount = obj.driftCount | 0;

    this.adam = new Adam(this.cfg.beta1, this.cfg.beta2, this.cfg.epsilon);
    this.adam.stepCount = obj.adamStep | 0;

    // stats
    this.xStats = new Welford(this.#inDim);
    this.yStats = new Welford(this.#outDim);
    if (obj.xStats) {
      this.xStats.count = obj.xStats.count | 0;
      this.xStats.mean.set(obj.xStats.mean);
      this.xStats.m2.set(obj.xStats.m2);
    }
    if (obj.yStats) {
      this.yStats.count = obj.yStats.count | 0;
      this.yStats.mean.set(obj.yStats.mean);
      this.yStats.m2.set(obj.yStats.m2);
    }
    this.xStd = new Float64Array(this.#inDim);
    this.yStd = new Float64Array(this.#outDim);

    // pos enc
    this.posEnc = new Float64Array(obj.posEnc);

    // rebuild modules and load weights
    const D = this.cfg.embeddingDim;
    const K = this.cfg.temporalKernelSize;
    const scales = this.cfg.temporalScales;
    const S = scales.length;

    this.convs = new Array(S);
    this.scaleEmb = new Array(S);
    for (let si = 0; si < S; si++) {
      const stride = obj.convs[si].stride | 0;
      this.convs[si] = new TemporalConv1D(this.#inDim, D, K, stride, this.rng);
      this.convs[si].W.w.set(obj.convs[si].W);
      this.convs[si].b.w.set(obj.convs[si].b);
      this.convs[si].W.m.set(obj.convs[si].mW);
      this.convs[si].W.v.set(obj.convs[si].vW);
      this.convs[si].b.m.set(obj.convs[si].mb);
      this.convs[si].b.v.set(obj.convs[si].vb);

      this.scaleEmb[si] = new Parameter(D, 0.0, () => 0.5);
      this.scaleEmb[si].w.set(obj.scaleEmb[si].w);
      this.scaleEmb[si].m.set(obj.scaleEmb[si].m);
      this.scaleEmb[si].v.set(obj.scaleEmb[si].v);
    }

    this.Wg = new Parameter((S * D) * S, 0.0, () => 0.5);
    this.bg = new Parameter(S, 0.0, () => 0.5);
    this.Wg.w.set(obj.fusion.Wg);
    this.bg.w.set(obj.fusion.bg);
    this.Wg.m.set(obj.fusion.mWg);
    this.Wg.v.set(obj.fusion.vWg);
    this.bg.m.set(obj.fusion.mbg);
    this.bg.v.set(obj.fusion.vbg);

    this.blocks = new Array(this.cfg.numBlocks);
    for (let b = 0; b < this.blocks.length; b++) {
      this.blocks[b] = new TransformerBlock(
        D,
        this.cfg.numHeads,
        this.cfg.ffnMultiplier,
        this.cfg.epsilon,
        this.rng,
      );
      const blkObj = obj.blocks[b];
      const blk = this.blocks[b];
      blk.ln1.gamma.w.set(blkObj.ln1g);
      blk.ln1.beta.w.set(blkObj.ln1b);
      blk.ln2.gamma.w.set(blkObj.ln2g);
      blk.ln2.beta.w.set(blkObj.ln2b);

      blk.attn.Wq.w.set(blkObj.attn.Wq);
      blk.attn.Wk.w.set(blkObj.attn.Wk);
      blk.attn.Wv.w.set(blkObj.attn.Wv);
      blk.attn.Wo.w.set(blkObj.attn.Wo);
      blk.attn.bq.w.set(blkObj.attn.bq);
      blk.attn.bk.w.set(blkObj.attn.bk);
      blk.attn.bv.w.set(blkObj.attn.bv);
      blk.attn.bo.w.set(blkObj.attn.bo);

      blk.attn.Wq.m.set(blkObj.attn.mWq);
      blk.attn.Wq.v.set(blkObj.attn.vWq);
      blk.attn.Wk.m.set(blkObj.attn.mWk);
      blk.attn.Wk.v.set(blkObj.attn.vWk);
      blk.attn.Wv.m.set(blkObj.attn.mWv);
      blk.attn.Wv.v.set(blkObj.attn.vWv);
      blk.attn.Wo.m.set(blkObj.attn.mWo);
      blk.attn.Wo.v.set(blkObj.attn.vWo);

      blk.ffn.W1.w.set(blkObj.ffn.W1);
      blk.ffn.b1.w.set(blkObj.ffn.b1);
      blk.ffn.W2.w.set(blkObj.ffn.W2);
      blk.ffn.b2.w.set(blkObj.ffn.b2);

      blk.ffn.W1.m.set(blkObj.ffn.mW1);
      blk.ffn.W1.v.set(blkObj.ffn.vW1);
      blk.ffn.W2.m.set(blkObj.ffn.mW2);
      blk.ffn.W2.v.set(blkObj.ffn.vW2);
    }

    this.Wy = new Parameter(D * this.#outDim, 0.0, () => 0.5);
    this.by = new Parameter(this.#outDim, 0.0, () => 0.5);
    this.Wy.w.set(obj.out.Wy);
    this.by.w.set(obj.out.by);
    this.Wy.m.set(obj.out.mWy);
    this.Wy.v.set(obj.out.vWy);
    this.by.m.set(obj.out.mby);
    this.by.v.set(obj.out.vby);

    this.adwin = new AdwinLite(this.cfg.adwinDelta, 256);

    this.ensureBuffers(1);
    this.#isInitialized = true;
  }
}
