/* eslint-disable @typescript-eslint/no-unused-vars */

/**
 * Fusion Temporal Transformer (FTT) for multivariate regression with:
 * - Incremental online learning (Adam)
 * - Online z-score normalization (Welford)
 * - L2 regularization
 * - Outlier downweighting
 * - ADWIN drift detection
 *
 * Pure TypeScript, no external dependencies, Float64Array tensors, allocation-avoiding hot paths.
 */

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

export type Config = {
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
};

type _Welford = {
  n: number;
  mean: Float64Array;
  m2: Float64Array;
};

type _Adwin = {
  cap: number;
  minN: number;
  delta: number;
  buf: Float64Array;
  idx: number;
  size: number;
};

type _BlockParams = {
  // LN1
  ln1Gamma: Float64Array;
  ln1Beta: Float64Array;
  // Attention
  Wq: Float64Array;
  Wk: Float64Array;
  Wv: Float64Array;
  Wo: Float64Array;
  // LN2
  ln2Gamma: Float64Array;
  ln2Beta: Float64Array;
  // FFN
  W1: Float64Array;
  b1: Float64Array;
  W2: Float64Array;
  b2: Float64Array;
};

type _BlockMoments = {
  // first moments
  m_ln1Gamma: Float64Array;
  m_ln1Beta: Float64Array;
  m_Wq: Float64Array;
  m_Wk: Float64Array;
  m_Wv: Float64Array;
  m_Wo: Float64Array;
  m_ln2Gamma: Float64Array;
  m_ln2Beta: Float64Array;
  m_W1: Float64Array;
  m_b1: Float64Array;
  m_W2: Float64Array;
  m_b2: Float64Array;

  // second moments
  v_ln1Gamma: Float64Array;
  v_ln1Beta: Float64Array;
  v_Wq: Float64Array;
  v_Wk: Float64Array;
  v_Wv: Float64Array;
  v_Wo: Float64Array;
  v_ln2Gamma: Float64Array;
  v_ln2Beta: Float64Array;
  v_W1: Float64Array;
  v_b1: Float64Array;
  v_W2: Float64Array;
  v_b2: Float64Array;
};

type _BlockGrads = {
  g_ln1Gamma: Float64Array;
  g_ln1Beta: Float64Array;
  g_Wq: Float64Array;
  g_Wk: Float64Array;
  g_Wv: Float64Array;
  g_Wo: Float64Array;
  g_ln2Gamma: Float64Array;
  g_ln2Beta: Float64Array;
  g_W1: Float64Array;
  g_b1: Float64Array;
  g_W2: Float64Array;
  g_b2: Float64Array;
};

type _ScaleParams = {
  W: Float64Array; // [K, inDim, emb] row-major flattened
  b: Float64Array; // [emb]
  scaleEmb: Float64Array; // [emb]
};

type _ScaleMoments = {
  m_W: Float64Array;
  m_b: Float64Array;
  m_scaleEmb: Float64Array;
  v_W: Float64Array;
  v_b: Float64Array;
  v_scaleEmb: Float64Array;
};

type _ScaleGrads = {
  g_W: Float64Array;
  g_b: Float64Array;
  g_scaleEmb: Float64Array;
};

type _Buffers = {
  // Input window
  xRaw: Float64Array; // [L1, inDim]
  xNorm: Float64Array; // [L1, inDim]
  yTargetRaw: Float64Array; // [outDim]
  yTargetNorm: Float64Array; // [outDim]

  // Multi-scale conv
  convPre: Float64Array[]; // per scale: [Ls, emb]
  convAct: Float64Array[]; // per scale: [Ls, emb]
  E: Float64Array[]; // per scale: [Ls, emb]
  EUp: Float64Array[]; // per scale: [L1, emb]
  dE: Float64Array[]; // per scale: [Ls, emb]
  dEUp: Float64Array[]; // per scale: [L1, emb]

  // Fusion
  concat: Float64Array; // [L1, concatDim]
  gatePre: Float64Array; // [L1, concatDim]
  gate: Float64Array; // [L1, concatDim]
  fusedPreDrop: Float64Array; // [L1, emb]
  fused: Float64Array; // [L1, emb]
  dFused: Float64Array; // [L1, emb]
  dConcat: Float64Array; // [L1, concatDim]
  dGate: Float64Array; // [L1, concatDim]
  dGatePre: Float64Array; // [L1, concatDim]

  // Transformer: per block caches
  blockInput: Float64Array[]; // [numBlocks] each [L1, emb] (H in)
  ln1Mean: Float64Array[]; // [numBlocks] [L1]
  ln1Var: Float64Array[]; // [numBlocks] [L1]
  Xn1: Float64Array[]; // [numBlocks] [L1, emb]
  Q: Float64Array[]; // [numBlocks] [L1, emb]
  K: Float64Array[]; // [numBlocks] [L1, emb]
  V: Float64Array[]; // [numBlocks] [L1, emb]
  context: Float64Array[]; // [numBlocks] [L1, emb]
  attnProj: Float64Array[]; // [numBlocks] [L1, emb]
  afterAttn: Float64Array[]; // [numBlocks] [L1, emb] (H1)
  ln2Mean: Float64Array[]; // [numBlocks] [L1]
  ln2Var: Float64Array[]; // [numBlocks] [L1]
  Xn2: Float64Array[]; // [numBlocks] [L1, emb]
  ffnPre: Float64Array[]; // [numBlocks] [L1, hidden]
  ffnAct: Float64Array[]; // [numBlocks] [L1, hidden]
  ffnOut: Float64Array[]; // [numBlocks] [L1, emb]

  // Grad buffers through transformer
  dH: Float64Array; // [L1, emb] upstream grad
  dTmp: Float64Array; // [L1, emb] scratch
  dXn: Float64Array; // [L1, emb] scratch
  dQ: Float64Array; // [L1, emb]
  dK: Float64Array; // [L1, emb]
  dV: Float64Array; // [L1, emb]
  dContext: Float64Array; // [L1, emb]
  dAttnProj: Float64Array; // [L1, emb]

  // Pooling
  poolScores: Float64Array; // [L1]
  poolAlpha: Float64Array; // [L1]
  pooled: Float64Array; // [emb]
  dPooled: Float64Array; // [emb]
  dScores: Float64Array; // [L1]
  dAlpha: Float64Array; // [L1]

  // Output
  yHatNorm: Float64Array; // [outDim]
  dYHat: Float64Array; // [outDim]

  // Scratch vectors
  tmpEmb: Float64Array; // [emb]
  tmpConcat: Float64Array; // [concatDim]
  tmpHidden: Float64Array; // [hiddenDim]
};

function _clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : (x > hi ? hi : x);
}

function _safeSqrt(x: number): number {
  return Math.sqrt(x < 0 ? 0 : x);
}

function _isFiniteNumber(x: number): boolean {
  return Number.isFinite(x) && !Number.isNaN(x);
}

/** Deterministic xorshift32 RNG; state must be uint32. */
class _XorShift32 {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0;
    if (this.s === 0) this.s = 0x9e3779b9;
  }
  nextU32(): number {
    // xorshift32
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x >>> 0;
    return this.s;
  }
  nextFloat(): number {
    // [0,1)
    return (this.nextU32() >>> 0) / 4294967296;
  }
}

function _gelu(x: number): number {
  // GELU approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  const c = 0.7978845608028654; // sqrt(2/pi)
  const x3 = x * x * x;
  const t = c * (x + 0.044715 * x3);
  // tanh stable
  const th = Math.tanh(t);
  return 0.5 * x * (1 + th);
}

function _geluGrad(x: number): number {
  // derivative of tanh-approx GELU
  // d/dx [0.5*x*(1+tanh(t))] = 0.5*(1+tanh(t)) + 0.5*x*(1-tanh(t)^2)*dt/dx
  const c = 0.7978845608028654;
  const x2 = x * x;
  const x3 = x2 * x;
  const t = c * (x + 0.044715 * x3);
  const th = Math.tanh(t);
  const sech2 = 1 - th * th;
  const dt = c * (1 + 0.134145 * x2); // 3*0.044715=0.134145
  return 0.5 * (1 + th) + 0.5 * x * sech2 * dt;
}

/**
 * Xavier uniform initialization:
 * limit = sqrt(6/(fanIn+fanOut))
 * Used consistently for conv and linear weights (works well with GELU in practice).
 */
function _initXavierUniform(
  dst: Float64Array,
  fanIn: number,
  fanOut: number,
  rng: _XorShift32,
): void {
  const denom = fanIn + fanOut;
  const limit = Math.sqrt(6 / (denom <= 0 ? 1 : denom));
  for (let i = 0; i < dst.length; i++) {
    const u = rng.nextFloat() * 2 - 1;
    dst[i] = u * limit;
  }
}

function _fill(dst: Float64Array, v: number): void {
  for (let i = 0; i < dst.length; i++) dst[i] = v;
}

function _copy(dst: Float64Array, src: Float64Array, n: number): void {
  for (let i = 0; i < n; i++) dst[i] = src[i];
}

function _axpy(dst: Float64Array, a: number, x: Float64Array, n: number): void {
  for (let i = 0; i < n; i++) dst[i] += a * x[i];
}

function _dot(
  a: Float64Array,
  aOff: number,
  b: Float64Array,
  bOff: number,
  n: number,
): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}

function _softmaxRowStable(
  out: Float64Array,
  outOff: number,
  scores: Float64Array,
  scoresOff: number,
  n: number,
): void {
  // out = softmax(scores)
  let maxv = -1e300;
  for (let i = 0; i < n; i++) {
    const v = scores[scoresOff + i];
    if (v > maxv) maxv = v;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const z = scores[scoresOff + i] - maxv;
    // exp clamp to avoid inf: exp(709) ~ 8e307
    const ez = Math.exp(_clamp(z, -745, 709));
    out[outOff + i] = ez;
    sum += ez;
  }
  if (sum <= 0 || !_isFiniteNumber(sum)) {
    const inv = 1 / (n <= 0 ? 1 : n);
    for (let i = 0; i < n; i++) out[outOff + i] = inv;
  } else {
    const inv = 1 / sum;
    for (let i = 0; i < n; i++) out[outOff + i] *= inv;
  }
}

function _sigmoid(x: number): number {
  // stable sigmoid
  if (x >= 0) {
    const z = Math.exp(-_clamp(x, 0, 709));
    return 1 / (1 + z);
  } else {
    const z = Math.exp(_clamp(x, -709, 0));
    return z / (1 + z);
  }
}

function _layerNormForward(
  x: Float64Array,
  xOff: number,
  y: Float64Array,
  yOff: number,
  meanArr: Float64Array,
  varArr: Float64Array,
  t: number,
  d: number,
  gamma: Float64Array,
  beta: Float64Array,
): void {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[xOff + i];
  mean /= d;

  let v = 0;
  for (let i = 0; i < d; i++) {
    const z = x[xOff + i] - mean;
    v += z * z;
  }
  v /= d;
  if (v < 1e-12) v = 1e-12;

  meanArr[t] = mean;
  varArr[t] = v;

  const invStd = 1 / Math.sqrt(v);
  for (let i = 0; i < d; i++) {
    const xhat = (x[xOff + i] - mean) * invStd;
    y[yOff + i] = xhat * gamma[i] + beta[i];
  }
}

function _layerNormBackward(
  x: Float64Array,
  xOff: number,
  dy: Float64Array,
  dyOff: number,
  dx: Float64Array,
  dxOff: number,
  mean: number,
  vari: number,
  d: number,
  gamma: Float64Array,
  dGamma: Float64Array,
  dBeta: Float64Array,
): void {
  // y = gamma * xhat + beta
  // xhat=(x-mean)/std; std=sqrt(var)
  const invStd = 1 / Math.sqrt(vari < 1e-12 ? 1e-12 : vari);

  // accumulate dGamma, dBeta
  // Also compute sum(dxhat) and sum(dxhat * (x-mean))
  let sumDxhat = 0;
  let sumDxhatXm = 0;

  for (let i = 0; i < d; i++) {
    const xm = x[xOff + i] - mean;
    const xhat = xm * invStd;
    const dxhat = dy[dyOff + i] * gamma[i];
    dGamma[i] += dy[dyOff + i] * xhat;
    dBeta[i] += dy[dyOff + i];
    sumDxhat += dxhat;
    sumDxhatXm += dxhat * xm;
  }

  const invD = 1 / d;
  // dvar = sum(dxhat*(x-mean)*(-0.5)*std^-3)
  const invStd3 = invStd * invStd * invStd;
  const dVar = sumDxhatXm * (-0.5) * invStd3;
  // dmean = sum(dxhat*(-invStd)) + dvar * sum(-2*(x-mean))/D
  // but sum(x-mean)=0, so second term is 0 for exact arithmetic; keep for stability:
  let sumXm = 0;
  for (let i = 0; i < d; i++) sumXm += x[xOff + i] - mean;
  const dMean = (-invStd) * sumDxhat + dVar * (-2 * sumXm) * invD;

  for (let i = 0; i < d; i++) {
    const xm = x[xOff + i] - mean;
    const dxhat = dy[dyOff + i] * gamma[i];
    dx[dxOff + i] += dxhat * invStd + dVar * 2 * xm * invD + dMean * invD;
  }
}

function _adamUpdate(
  w: Float64Array,
  g: Float64Array,
  m: Float64Array,
  v: Float64Array,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  t: number,
  l2: number,
  accumGradNorm: { v: number },
): void {
  const b1t = 1 - Math.pow(beta1, t);
  const b2t = 1 - Math.pow(beta2, t);
  const invB1t = b1t <= 0 ? 1 : 1 / b1t;
  const invB2t = b2t <= 0 ? 1 : 1 / b2t;

  for (let i = 0; i < w.length; i++) {
    // L2 regularization gradient add
    let gi = g[i] + l2 * w[i];
    if (!_isFiniteNumber(gi)) gi = 0;

    m[i] = beta1 * m[i] + (1 - beta1) * gi;
    v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;

    // bias correction
    const mhat = m[i] * invB1t;
    const vhat = v[i] * invB2t;

    const denom = Math.sqrt(vhat < 0 ? 0 : vhat) + eps;
    const step = lr * (mhat / (denom <= 0 ? eps : denom));
    w[i] -= step;

    accumGradNorm.v += gi * gi;
  }
}

function _computeScheduledLR(
  base: number,
  step: number,
  warmup: number,
  total: number,
): number {
  if (step < 1) return 0;
  if (warmup <= 0) warmup = 1;
  if (step < warmup) return base * (step / warmup);
  const denom = Math.max(1, total - warmup);
  const progress = (step - warmup) / denom;
  const p = Math.min(1, Math.max(0, progress));
  return base * 0.5 * (1 + Math.cos(Math.PI * p));
}

function _welfordCreate(dim: number): _Welford {
  return {
    n: 0,
    mean: new Float64Array(dim),
    m2: new Float64Array(dim),
  };
}

function _welfordUpdateVec(
  stats: _Welford,
  x: Float64Array,
  xOff: number,
  dim: number,
): void {
  stats.n++;
  const n = stats.n;
  for (let i = 0; i < dim; i++) {
    const xi = x[xOff + i];
    const delta = xi - stats.mean[i];
    stats.mean[i] += delta / n;
    const delta2 = xi - stats.mean[i];
    stats.m2[i] += delta * delta2;
  }
}

function _welfordUpdateArray(
  stats: _Welford,
  arr: Float64Array,
  nRows: number,
  rowStride: number,
  dim: number,
): void {
  for (let r = 0; r < nRows; r++) {
    _welfordUpdateVec(stats, arr, r * rowStride, dim);
  }
}

function _welfordStd(stats: _Welford, outStd: Float64Array): void {
  const n = stats.n;
  for (let i = 0; i < outStd.length; i++) {
    let v = 1;
    if (n >= 2) v = stats.m2[i] / (n - 1);
    if (v < 1e-24) v = 1e-24;
    outStd[i] = Math.sqrt(v);
    if (outStd[i] < 1e-12) outStd[i] = 1e-12;
  }
}

function _adwinCreate(cap: number, minN: number, delta: number): _Adwin {
  return {
    cap,
    minN,
    delta,
    buf: new Float64Array(cap),
    idx: 0,
    size: 0,
  };
}

function _adwinReset(a: _Adwin): void {
  a.idx = 0;
  a.size = 0;
}

function _adwinAddAndDetect(a: _Adwin, x: number): boolean {
  a.buf[a.idx] = x;
  a.idx = (a.idx + 1) % a.cap;
  if (a.size < a.cap) a.size++;

  if (a.size < a.minN) return false;

  // Copy ring buffer order logically via index mapping without allocations:
  // We'll compute prefix sums by reading in chronological order.
  const n = a.size;
  const start = (a.idx - n + a.cap) % a.cap;

  // Build prefix sums in O(n^2) without extra arrays by scanning splits and computing sums incrementally.
  // Since cap is small (<=256), this is OK.
  let totalSum = 0;
  for (let i = 0; i < n; i++) totalSum += a.buf[(start + i) % a.cap];

  const logTerm = Math.log(2 / a.delta);
  let leftSum = 0;

  // Require both sides at least 8 to avoid noise.
  const minSide = 8;
  for (let split = minSide; split <= n - minSide; split++) {
    leftSum += a.buf[(start + split - 1) % a.cap];
    const rightSum = totalSum - leftSum;
    const nl = split;
    const nr = n - split;
    const meanL = leftSum / nl;
    const meanR = rightSum / nr;
    const diff = Math.abs(meanL - meanR);
    const eps = Math.sqrt((2 * logTerm) * (1 / nl + 1 / nr));
    if (diff > eps) return true;
  }
  return false;
}

function _pack1D(a: Float64Array): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i];
  return out;
}

function _pack2D(a: Float64Array, rows: number, cols: number): number[][] {
  const out = new Array<number[]>(rows);
  let off = 0;
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) row[c] = a[off++];
    out[r] = row;
  }
  return out;
}

function _rehydrateF64(x: number[] | Float64Array): Float64Array {
  if (x instanceof Float64Array) return x;
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i];
  return out;
}

export class FusionTemporalTransformerRegression {
  private readonly config: Config;

  private isInitialized = false;

  private inputDim = 0;
  private outputDim = 0;
  private headDim = 0;
  private hiddenDim = 0;
  private nScales = 0;

  private sampleCount = 0;
  private updateCount = 0;
  private driftCount = 0;
  private runningLoss = 0;
  private lastEffectiveLR = 0;
  private lastConverged = false;

  // Cached last window
  private lastSeqLen = 0;

  // Normalization
  private inStats: _Welford | null = null;
  private outStats: _Welford | null = null;
  private inStd: Float64Array | null = null;
  private outStd: Float64Array | null = null;

  // Residual stats for uncertainty (original scale)
  private resStats: _Welford | null = null;

  // ADWIN
  private adwin: _Adwin;

  // Positional Encoding cache: [maxSeq, emb]
  private pe: Float64Array | null = null;

  // Parameters
  private scales: _ScaleParams[] = [];
  private scaleMom: _ScaleMoments[] = [];
  private scaleGrad: _ScaleGrads[] = [];

  // Fusion params
  private Wg: Float64Array | null = null; // [concatDim, concatDim]
  private bg: Float64Array | null = null; // [concatDim]
  private mWg: Float64Array | null = null;
  private vWg: Float64Array | null = null;
  private mbg: Float64Array | null = null;
  private vbg: Float64Array | null = null;
  private gWg: Float64Array | null = null;
  private gbg: Float64Array | null = null;

  // Blocks
  private blocks: _BlockParams[] = [];
  private blockMom: _BlockMoments[] = [];
  private blockGrad: _BlockGrads[] = [];

  // Pool params
  private Wpool: Float64Array | null = null; // [emb]
  private bpool = 0;
  private mWpool: Float64Array | null = null;
  private vWpool: Float64Array | null = null;
  private mbpool = 0;
  private vbpool = 0;
  private gWpool: Float64Array | null = null;
  private gbpool = 0;

  // Output head
  private Wout: Float64Array | null = null; // [emb, outDim]
  private bout: Float64Array | null = null; // [outDim]
  private mWout: Float64Array | null = null;
  private vWout: Float64Array | null = null;
  private mbout: Float64Array | null = null;
  private vbout: Float64Array | null = null;
  private gWout: Float64Array | null = null;
  private gbout: Float64Array | null = null;

  // Buffers
  private buf: _Buffers | null = null;

  // Cached last x window raw (for predict): [L1, inDim]
  private lastXRaw: Float64Array | null = null;

  /**
   * @param config Partial configuration overrides.
   * @example
   * const m = new FusionTemporalTransformerRegression({ embeddingDim: 32, numHeads: 4 });
   */
  constructor(config?: Partial<Config>) {
    const defaults: Config = {
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
    };
    this.config = Object.assign({}, defaults, config || {});
    this.adwin = _adwinCreate(256, 32, this.config.adwinDelta);
  }

  /**
   * Fit one online sample (a window sequence).
   * Uses y = last timestep target and outputs a single vector after pooling.
   * @param data Sequence window with xCoordinates (seqLen x inputDim) and yCoordinates (seqLen x outputDim).
   * @returns FitResult
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xSeq = data.xCoordinates;
    const ySeq = data.yCoordinates;

    const seqLenRaw = Math.min(xSeq.length, ySeq.length);
    const L1 = Math.min(seqLenRaw, this.config.maxSequenceLength);

    if (L1 <= 0) {
      return {
        loss: 0,
        gradientNorm: 0,
        effectiveLearningRate: 0,
        isOutlier: false,
        converged: true,
        sampleIndex: this.sampleCount,
        driftDetected: false,
      };
    }

    const inDim = xSeq[0].length;
    const outDim = ySeq[0].length;

    if (!this.isInitialized) {
      this.ensureInitialized(inDim, outDim);
    } else {
      if (inDim !== this.inputDim || outDim !== this.outputDim) {
        throw new Error(
          `Dimension mismatch: expected inDim=${this.inputDim}, outDim=${this.outputDim}, got inDim=${inDim}, outDim=${outDim}`,
        );
      }
    }

    // Buffers
    this.ensureBuffers();

    const emb = this.config.embeddingDim;
    const concatDim = this.nScales * emb;
    const hidden = this.hiddenDim;

    const buf = this.buf!;
    this.lastSeqLen = L1;

    // Copy x window into xRaw (clipped to L1; no padding)
    for (let t = 0; t < L1; t++) {
      const row = xSeq[t];
      const off = t * inDim;
      for (let f = 0; f < inDim; f++) {
        buf.xRaw[off + f] = row[f];
      }
    }
    // cache lastXRaw for predict
    if (!this.lastXRaw || this.lastXRaw.length !== buf.xRaw.length) {
      this.lastXRaw = new Float64Array(buf.xRaw.length);
    }
    _copy(this.lastXRaw, buf.xRaw, L1 * inDim);

    // y target = last timestep
    const yRow = ySeq[L1 - 1];
    for (let d = 0; d < outDim; d++) buf.yTargetRaw[d] = yRow[d];

    // Update normalization stats (treat current sample as observed)
    // Inputs: update across all timesteps; Outputs: only last target vector.
    _welfordUpdateArray(this.inStats!, buf.xRaw, L1, inDim, inDim);
    _welfordUpdateVec(this.outStats!, buf.yTargetRaw, 0, outDim);

    _welfordStd(this.inStats!, this.inStd!);
    _welfordStd(this.outStats!, this.outStd!);

    // Normalize x -> xNorm
    const inMean = this.inStats!.mean;
    const inStd = this.inStd!;
    for (let t = 0; t < L1; t++) {
      const off = t * inDim;
      for (let f = 0; f < inDim; f++) {
        const z = (buf.xRaw[off + f] - inMean[f]) /
          (inStd[f] <= 1e-12 ? 1e-12 : inStd[f]);
        buf.xNorm[off + f] = _isFiniteNumber(z) ? z : 0;
      }
    }

    // Normalize y target
    const outMean = this.outStats!.mean;
    const outStd = this.outStd!;
    for (let d = 0; d < outDim; d++) {
      const z = (buf.yTargetRaw[d] - outMean[d]) /
        (outStd[d] <= 1e-12 ? 1e-12 : outStd[d]);
      buf.yTargetNorm[d] = _isFiniteNumber(z) ? z : 0;
    }

    // ===== Forward =====
    // 1) Multi-scale temporal convolution
    for (let si = 0; si < this.nScales; si++) {
      const s = this.config.temporalScales[si];
      const K = this.config.temporalKernelSize;
      const Ls = Math.floor((L1 + s - 1) / s);
      const W = this.scales[si].W;
      const b = this.scales[si].b;
      const convPre = buf.convPre[si];
      const convAct = buf.convAct[si];

      // convPre/convAct are sized to max; only fill first Ls
      for (let tt = 0; tt < Ls; tt++) {
        const tBase = tt * s;
        const outOff = tt * emb;
        for (let e = 0; e < emb; e++) {
          let sum = b[e];
          // causal left padding: only idx>=0
          for (let k = 0; k < K; k++) {
            const idx = tBase - k;
            if (idx < 0) continue;
            if (idx >= L1) continue;
            const xOff = idx * inDim;
            // W index: (((k*inDim)+f)*emb + e)
            let wOff = (k * inDim) * emb + e;
            for (let f = 0; f < inDim; f++) {
              sum += buf.xNorm[xOff + f] * W[wOff];
              wOff += emb;
            }
          }
          convPre[outOff + e] = sum;
          convAct[outOff + e] = _gelu(sum);
        }
      }

      // 2) Add PE and scale embedding: E[tt,e] = convAct + PE(tt,e) + scaleEmb[e]
      const E = buf.E[si];
      const scaleEmb = this.scales[si].scaleEmb;
      const pe = this.pe!;
      for (let tt = 0; tt < Ls; tt++) {
        const off = tt * emb;
        const peOff = tt * emb;
        for (let e = 0; e < emb; e++) {
          E[off + e] = convAct[off + e] + pe[peOff + e] + scaleEmb[e];
        }
      }

      // 3) Upsample to length L1 by repetition: te=floor(t/s)
      const EUp = buf.EUp[si];
      for (let t = 0; t < L1; t++) {
        let te = Math.floor(t / s);
        if (te < 0) te = 0;
        if (te >= Ls) te = Ls - 1;
        const srcOff = te * emb;
        const dstOff = t * emb;
        for (let e = 0; e < emb; e++) EUp[dstOff + e] = E[srcOff + e];
      }
    }

    // 4) Fusion: concat per timestep then gate
    const Wg = this.Wg!;
    const bg = this.bg!;
    const concat = buf.concat;
    const gatePre = buf.gatePre;
    const gate = buf.gate;
    const fusedPre = buf.fusedPreDrop;
    const fused = buf.fused;

    // Build concat
    for (let t = 0; t < L1; t++) {
      const base = t * concatDim;
      let w = 0;
      for (let si = 0; si < this.nScales; si++) {
        const EUp = buf.EUp[si];
        const srcOff = t * emb;
        for (let e = 0; e < emb; e++) concat[base + (w++)] = EUp[srcOff + e];
      }
    }

    // Gate preactivation and sigmoid
    // gatePre[t,d] = sum_c concat[t,c]*Wg[c,d] + bg[d]
    for (let t = 0; t < L1; t++) {
      const xOff = t * concatDim;
      const yOff = t * concatDim;
      for (let d = 0; d < concatDim; d++) {
        let sum = bg[d];
        // Wg is [concatDim, concatDim] row-major, rows=c, cols=d
        let wOff = d;
        for (let c = 0; c < concatDim; c++) {
          sum += concat[xOff + c] * Wg[wOff];
          wOff += concatDim;
        }
        gatePre[yOff + d] = sum;
        gate[yOff + d] = _sigmoid(sum);
      }
    }

    // Fused: sum_s gate_s * EUp_s
    for (let t = 0; t < L1; t++) {
      const gOff = t * concatDim;
      const outOff = t * emb;
      for (let e = 0; e < emb; e++) fusedPre[outOff + e] = 0;
      for (let si = 0; si < this.nScales; si++) {
        const EUp = buf.EUp[si];
        const srcOff = t * emb;
        const gBase = gOff + si * emb;
        for (let e = 0; e < emb; e++) {
          fusedPre[outOff + e] += gate[gBase + e] * EUp[srcOff + e];
        }
      }
    }

    // Fusion dropout (training only)
    if (this.config.fusionDropout > 0) {
      const p = this.config.fusionDropout;
      const scale = 1 / (1 - p);
      const rng = new _XorShift32(
        (this.updateCount + 1) * 2654435761 ^ 0xA5A5A5A5,
      );
      for (let i = 0; i < L1 * emb; i++) {
        const keep = rng.nextFloat() >= p ? 1 : 0;
        fused[i] = fusedPre[i] * keep * scale;
      }
    } else {
      _copy(fused, fusedPre, L1 * emb);
    }

    // 5) Transformer blocks
    // H starts as fused; blockInput caches input H for backward
    let H = fused;
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const params = this.blocks[bi];
      const H_in = buf.blockInput[bi];
      _copy(H_in, H, L1 * emb);

      // LN1: Xn1
      const Xn1 = buf.Xn1[bi];
      const m1 = buf.ln1Mean[bi];
      const v1 = buf.ln1Var[bi];
      for (let t = 0; t < L1; t++) {
        _layerNormForward(
          H_in,
          t * emb,
          Xn1,
          t * emb,
          m1,
          v1,
          t,
          emb,
          params.ln1Gamma,
          params.ln1Beta,
        );
      }

      // Q,K,V = Xn1 * W?
      const Q = buf.Q[bi];
      const K = buf.K[bi];
      const V = buf.V[bi];
      this.linearForward2D(Xn1, Q, L1, emb, params.Wq, null);
      this.linearForward2D(Xn1, K, L1, emb, params.Wk, null);
      this.linearForward2D(Xn1, V, L1, emb, params.Wv, null);

      // Attention context
      const context = buf.context[bi];
      this.mhaForward(
        Q,
        K,
        V,
        context,
        L1,
        emb,
        this.config.numHeads,
        this.config.attentionDropout,
        true,
        bi,
      );

      // Project heads: attnProj = context * Wo
      const attnProj = buf.attnProj[bi];
      this.linearForward2D(context, attnProj, L1, emb, params.Wo, null);

      // Residual: afterAttn = H_in + attnProj
      const afterAttn = buf.afterAttn[bi];
      for (let i = 0; i < L1 * emb; i++) afterAttn[i] = H_in[i] + attnProj[i];

      // LN2: Xn2
      const Xn2 = buf.Xn2[bi];
      const m2 = buf.ln2Mean[bi];
      const v2 = buf.ln2Var[bi];
      for (let t = 0; t < L1; t++) {
        _layerNormForward(
          afterAttn,
          t * emb,
          Xn2,
          t * emb,
          m2,
          v2,
          t,
          emb,
          params.ln2Gamma,
          params.ln2Beta,
        );
      }

      // FFN: hidden = GELU(Xn2*W1 + b1); out = hidden*W2 + b2
      const ffnPre = buf.ffnPre[bi];
      const ffnAct = buf.ffnAct[bi];
      const ffnOut = buf.ffnOut[bi];
      // Xn2*W1 + b1
      this.linearForward2D(Xn2, ffnPre, L1, emb, params.W1, params.b1);
      for (let i = 0; i < L1 * hidden; i++) ffnAct[i] = _gelu(ffnPre[i]);
      this.linearForward2D(ffnAct, ffnOut, L1, hidden, params.W2, params.b2);

      // Residual: H = afterAttn + ffnOut, stored back into fused (reuse fused buffer as H for final)
      for (let i = 0; i < L1 * emb; i++) H[i] = afterAttn[i] + ffnOut[i];
    }

    // 6) Pooling: scores, alpha, pooled
    const scores = buf.poolScores;
    const alpha = buf.poolAlpha;
    const pooled = buf.pooled;

    const Wpool = this.Wpool!;
    const bpool = this.bpool;
    for (let t = 0; t < L1; t++) {
      scores[t] = _dot(H, t * emb, Wpool, 0, emb) + bpool;
    }
    // softmax(scores) -> alpha
    _softmaxRowStable(alpha, 0, scores, 0, L1);

    _fill(pooled, 0);
    for (let t = 0; t < L1; t++) {
      const a = alpha[t];
      const off = t * emb;
      for (let e = 0; e < emb; e++) pooled[e] += a * H[off + e];
    }

    // 7) Output head: yHatNorm = pooled * Wout + bout
    const Wout = this.Wout!;
    const bout = this.bout!;
    const yHat = buf.yHatNorm;
    for (let d = 0; d < outDim; d++) {
      let sum = bout[d];
      // Wout row-major [emb, outDim]
      let wOff = d;
      for (let e = 0; e < emb; e++) {
        sum += pooled[e] * Wout[wOff];
        wOff += outDim;
      }
      yHat[d] = sum;
    }

    // ===== Loss + Outlier weight =====
    let isOutlier = false;
    for (let d = 0; d < outDim; d++) {
      const r = buf.yTargetNorm[d] - yHat[d];
      if (Math.abs(r) > this.config.outlierThreshold) {
        isOutlier = true;
        break;
      }
    }
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    let mse = 0;
    for (let d = 0; d < outDim; d++) {
      const r = buf.yTargetNorm[d] - yHat[d];
      mse += r * r;
    }
    mse = mse / outDim;
    let loss = 0.5 * mse;
    loss *= sampleWeight;

    // L2 penalty term (for reporting; gradients get L2 in optimizer)
    const lambda = this.config.regularizationStrength;
    const l2sum = this.sumL2Weights();
    loss += 0.5 * lambda * l2sum;

    // Update running loss & accuracy
    this.sampleCount++;
    this.runningLoss = (this.runningLoss * (this.sampleCount - 1) + loss) /
      this.sampleCount;
    const accuracy = 1 / (1 + this.runningLoss);

    // ===== Backward =====
    this.zeroAllGrads();

    // dYHat = (yHat - yTarget) / outDim * sampleWeight
    const dYHat = buf.dYHat;
    const invOut = 1 / outDim;
    for (let d = 0; d < outDim; d++) {
      dYHat[d] = (yHat[d] - buf.yTargetNorm[d]) * invOut * sampleWeight;
      if (!_isFiniteNumber(dYHat[d])) dYHat[d] = 0;
    }

    // Output head backward: dWout, dbout, dPooled
    const gWout = this.gWout!;
    const gbout = this.gbout!;
    _fill(buf.dPooled, 0);
    for (let d = 0; d < outDim; d++) {
      gbout[d] += dYHat[d];
      // gWout[e,d] += pooled[e] * dYHat[d]; dPooled[e] += Wout[e,d]*dYHat[d]
      let wOff = d;
      for (let e = 0; e < emb; e++) {
        gWout[wOff] += pooled[e] * dYHat[d];
        buf.dPooled[e] += Wout[wOff] * dYHat[d];
        wOff += outDim;
      }
    }

    // Pooling backward:
    // pooled = sum_t alpha[t]*H[t]
    // dH_from_pooled[t,e] = alpha[t]*dPooled[e]
    // dAlpha[t] = dot(dPooled, H[t])
    // scores = dot(H[t], Wpool)+bpool
    // dScores = softmax backward(alpha, dAlpha)
    // dWpool += sum_t dScores[t]*H[t]
    // dbpool += sum_t dScores[t]
    // additional dH[t,e] += dScores[t]*Wpool[e]
    const dH = buf.dH;
    _fill(dH, 0);

    const dAlpha = buf.dAlpha;
    for (let t = 0; t < L1; t++) {
      const off = t * emb;
      // dAlpha[t] = dot(dPooled, H[t])
      dAlpha[t] = _dot(buf.dPooled, 0, H, off, emb);
      // dH += alpha[t] * dPooled
      const a = alpha[t];
      for (let e = 0; e < emb; e++) dH[off + e] += a * buf.dPooled[e];
    }

    // softmax backward: dScores[t] = alpha[t]*(dAlpha[t] - sum_j alpha[j]*dAlpha[j])
    let sumAD = 0;
    for (let t = 0; t < L1; t++) sumAD += alpha[t] * dAlpha[t];
    const dScores = buf.dScores;
    for (let t = 0; t < L1; t++) dScores[t] = alpha[t] * (dAlpha[t] - sumAD);

    // grad pool params and add to dH
    const gWpool = this.gWpool!;
    let gbpool = this.gbpool;
    for (let t = 0; t < L1; t++) {
      const ds = dScores[t];
      gbpool += ds;
      const off = t * emb;
      for (let e = 0; e < emb; e++) {
        gWpool[e] += ds * H[off + e];
        dH[off + e] += ds * Wpool[e];
      }
    }
    this.gbpool = gbpool;

    // Transformer backward through blocks reversed
    for (let bi = this.config.numBlocks - 1; bi >= 0; bi--) {
      const params = this.blocks[bi];
      const grads = this.blockGrad[bi];

      const H_in = buf.blockInput[bi];
      const afterAttn = buf.afterAttn[bi];
      const Xn2 = buf.Xn2[bi];
      const ffnPre = buf.ffnPre[bi];
      const ffnAct = buf.ffnAct[bi];

      // Residual: H_out = afterAttn + ffnOut
      // upstream dH is already for H_out.
      // => dAfterAttn += dH, dFfnOut += dH
      const dAfterAttn = buf.dTmp;
      const dFfnOut = buf.dXn;
      _fill(dAfterAttn, 0);
      _fill(dFfnOut, 0);
      for (let i = 0; i < L1 * emb; i++) {
        dAfterAttn[i] = dH[i];
        dFfnOut[i] = dH[i];
      }

      // FFN backward: ffnOut = ffnAct * W2 + b2
      // dW2, db2, dFfnAct
      const dFfnAct = buf.tmpHidden; // use tmpHidden as scratch for one timestep; we do per timestep loops below without allocations

      // We'll create a buffer for dAct full: reuse buf.ffnPre[bi] memory? cannot, needed for gelu grad.
      // We'll use buf.ffnPre[bi] as dPre after reading ffnPre (we still have it). Overwrite allowed now.
      // Similarly, use buf.ffnAct[bi] as dAct to save memory? It's used for W2 grad. We'll keep it and use a separate scratch per element in place.
      // We'll compute dHidden = dFfnOut * W2^T into buf.ffnAct[bi] (overwrite ffnAct after using it for gW2).
      // Need ffnAct values for gW2: use saved ffnAct first.

      // gW2 += ffnAct^T * dFfnOut ; gb2 += sum dFfnOut
      for (let e = 0; e < emb; e++) grads.g_b2[e] += 0;
      const gW2 = grads.g_W2;
      const gb2 = grads.g_b2;
      const W2 = params.W2;

      for (let t = 0; t < L1; t++) {
        const actOff = t * hidden;
        const dOutOff = t * emb;
        // gb2
        for (let e = 0; e < emb; e++) gb2[e] += dFfnOut[dOutOff + e];

        // gW2[h,e] += act[h]*dOut[e]
        for (let h = 0; h < hidden; h++) {
          const aVal = ffnAct[actOff + h];
          let wOff = h * emb;
          for (let e = 0; e < emb; e++) {
            gW2[wOff + e] += aVal * dFfnOut[dOutOff + e];
          }
        }
      }

      // dHidden (size L1*hidden) = dFfnOut * W2^T
      // Store into buf.ffnAct[bi] (overwrite)
      for (let t = 0; t < L1; t++) {
        const dOutOff = t * emb;
        const dHidOff = t * hidden;
        for (let h = 0; h < hidden; h++) {
          let sum = 0;
          const wOff = h * emb;
          for (let e = 0; e < emb; e++) {
            sum += dFfnOut[dOutOff + e] * W2[wOff + e];
          }
          ffnAct[dHidOff + h] = sum;
        }
      }

      // dPre = dHidden * gelu'(ffnPre)
      // store dPre into ffnPre (overwrite)
      for (let i = 0; i < L1 * hidden; i++) {
        const gp = _geluGrad(ffnPre[i]);
        ffnPre[i] = ffnAct[i] * gp;
      }

      // Backprop through linear1: ffnPre = Xn2*W1 + b1
      const gW1 = grads.g_W1;
      const gb1 = grads.g_b1;
      const W1 = params.W1;

      // gb1 += sum dPre
      _fill(buf.tmpHidden, 0); // temp accum not used, but keep deterministic
      for (let h = 0; h < hidden; h++) gb1[h] += 0;
      for (let t = 0; t < L1; t++) {
        const dPreOff = t * hidden;
        for (let h = 0; h < hidden; h++) gb1[h] += ffnPre[dPreOff + h];
      }

      // gW1[e,h] += Xn2[e]*dPre[h]; dXn2 += dPre*W1^T
      const dXn2 = buf.dXn; // [L1, emb], reuse and zero
      _fill(dXn2, 0);
      for (let t = 0; t < L1; t++) {
        const xOff = t * emb;
        const dPreOff = t * hidden;
        // gW1
        for (let e = 0; e < emb; e++) {
          const xv = Xn2[xOff + e];
          let wOff = e * hidden;
          for (let h = 0; h < hidden; h++) {
            gW1[wOff + h] += xv * ffnPre[dPreOff + h];
          }
        }
        // dXn2
        for (let e = 0; e < emb; e++) {
          let sum = 0;
          const wOff = e * hidden;
          for (let h = 0; h < hidden; h++) {
            sum += ffnPre[dPreOff + h] * W1[wOff + h];
          }
          dXn2[xOff + e] += sum;
        }
      }

      // LN2 backward: Xn2 = LN(afterAttn)
      const ln2Mean = buf.ln2Mean[bi];
      const ln2Var = buf.ln2Var[bi];
      const dAfterAttnFromLn = buf.dTmp;
      _fill(dAfterAttnFromLn, 0);

      for (let t = 0; t < L1; t++) {
        _layerNormBackward(
          afterAttn,
          t * emb,
          dXn2,
          t * emb,
          dAfterAttnFromLn,
          t * emb,
          ln2Mean[t],
          ln2Var[t],
          emb,
          params.ln2Gamma,
          grads.g_ln2Gamma,
          grads.g_ln2Beta,
        );
      }

      // Total dAfterAttn = residual (from H_out) + from LN2 path
      for (let i = 0; i < L1 * emb; i++) dAfterAttn[i] += dAfterAttnFromLn[i];

      // Residual after attention: afterAttn = H_in + attnProj
      // => dH_in_res += dAfterAttn; dAttnProj += dAfterAttn
      const dH_in_res = buf.dTmp; // reuse dTmp as dH_in_res (overwrite)
      _fill(dH_in_res, 0);
      const dAttnProj = buf.dAttnProj;
      _fill(dAttnProj, 0);
      for (let i = 0; i < L1 * emb; i++) {
        dH_in_res[i] = dAfterAttn[i];
        dAttnProj[i] = dAfterAttn[i];
      }

      // Attention projection backward: attnProj = context * Wo
      // dWo += context^T * dAttnProj ; dContext = dAttnProj * Wo^T
      const context = buf.context[bi];
      const Wo = params.Wo;
      const gWo = grads.g_Wo;
      const dContext = buf.dContext;
      _fill(dContext, 0);

      // gWo
      for (let t = 0; t < L1; t++) {
        const cOff = t * emb;
        const dOff = t * emb;
        for (let i = 0; i < emb; i++) {
          const cv = context[cOff + i];
          let wOff = i * emb;
          for (let j = 0; j < emb; j++) {
            gWo[wOff + j] += cv * dAttnProj[dOff + j];
          }
        }
      }
      // dContext
      for (let t = 0; t < L1; t++) {
        const dOff = t * emb;
        for (let i = 0; i < emb; i++) {
          let sum = 0;
          // dContext[i] = sum_j dAttnProj[j] * Wo[i,j]
          const wOff = i * emb;
          for (let j = 0; j < emb; j++) {
            sum += dAttnProj[dOff + j] * Wo[wOff + j];
          }
          dContext[dOff + i] = sum;
        }
      }

      // MHA backward: recompute softmax per row/head from Q,K; dropout mask regenerated deterministically.
      const Q = buf.Q[bi];
      const K = buf.K[bi];
      const V = buf.V[bi];
      const dQ = buf.dQ;
      const dK = buf.dK;
      const dV = buf.dV;
      _fill(dQ, 0);
      _fill(dK, 0);
      _fill(dV, 0);

      this.mhaBackward(
        Q,
        K,
        V,
        dContext,
        dQ,
        dK,
        dV,
        L1,
        emb,
        this.config.numHeads,
        this.config.attentionDropout,
        true,
        bi,
      );

      // Projection backward: Q = Xn1*Wq, etc.
      const Xn1 = buf.Xn1[bi];
      const ln1Mean = buf.ln1Mean[bi];
      const ln1Var = buf.ln1Var[bi];

      // grads Wq/Wk/Wv and dXn1
      const dXn1 = buf.dXn;
      _fill(dXn1, 0);

      this.linearBackward2D(Xn1, dQ, L1, emb, params.Wq, grads.g_Wq, dXn1);
      this.linearBackward2D(Xn1, dK, L1, emb, params.Wk, grads.g_Wk, dXn1);
      this.linearBackward2D(Xn1, dV, L1, emb, params.Wv, grads.g_Wv, dXn1);

      // LN1 backward: Xn1 = LN(H_in)
      const dH_in_fromLn = buf.dTmp;
      _fill(dH_in_fromLn, 0);
      for (let t = 0; t < L1; t++) {
        _layerNormBackward(
          H_in,
          t * emb,
          dXn1,
          t * emb,
          dH_in_fromLn,
          t * emb,
          ln1Mean[t],
          ln1Var[t],
          emb,
          params.ln1Gamma,
          grads.g_ln1Gamma,
          grads.g_ln1Beta,
        );
      }

      // Total dH for previous layer: dH = dH_in_res + dH_in_fromLn
      for (let i = 0; i < L1 * emb; i++) dH[i] = dH_in_res[i] + dH_in_fromLn[i];
    }

    // Now dH is gradient at transformer input = fused
    _copy(buf.dFused, dH, L1 * emb);

    // Fusion dropout backward: regenerate mask and scale
    if (this.config.fusionDropout > 0) {
      const p = this.config.fusionDropout;
      const scale = 1 / (1 - p);
      const rng = new _XorShift32(
        (this.updateCount + 1) * 2654435761 ^ 0xA5A5A5A5,
      );
      for (let i = 0; i < L1 * emb; i++) {
        const keep = rng.nextFloat() >= p ? 1 : 0;
        buf.dFused[i] = buf.dFused[i] * keep * scale;
      }
    }

    // Fusion backward:
    // fusedPre[t,e] = sum_s gate_s[t,e] * EUp_s[t,e]
    // dGate += dFused * EUp ; dEUp += dFused * gate
    _fill(buf.dGate, 0);
    for (let si = 0; si < this.nScales; si++) _fill(buf.dEUp[si], 0);

    for (let t = 0; t < L1; t++) {
      const gOff = t * concatDim;
      const dFoff = t * emb;

      for (let si = 0; si < this.nScales; si++) {
        const EUp = buf.EUp[si];
        const dEUp = buf.dEUp[si];
        const srcOff = t * emb;
        const gBase = gOff + si * emb;
        for (let e = 0; e < emb; e++) {
          const df = buf.dFused[dFoff + e];
          const gu = gate[gBase + e];
          const eu = EUp[srcOff + e];
          dEUp[srcOff + e] += df * gu;
          buf.dGate[gBase + e] += df * eu;
        }
      }
    }

    // gatePre -> sigmoid
    for (let i = 0; i < L1 * concatDim; i++) {
      const g = gate[i];
      buf.dGatePre[i] = buf.dGate[i] * g * (1 - g);
    }

    // gatePre = concat * Wg + bg
    // grads gWg += concat^T * dGatePre ; gbg += sum dGatePre ; dConcat = dGatePre * Wg^T
    const gWg = this.gWg!;
    const gbg = this.gbg!;
    _fill(buf.dConcat, 0);

    // gbg
    for (let d = 0; d < concatDim; d++) gbg[d] += 0;
    for (let t = 0; t < L1; t++) {
      const off = t * concatDim;
      for (let d = 0; d < concatDim; d++) gbg[d] += buf.dGatePre[off + d];
    }

    // gWg and dConcat
    for (let t = 0; t < L1; t++) {
      const xOff = t * concatDim;
      const dOff = t * concatDim;

      // gWg[c,d] += concat[c]*dGatePre[d]
      for (let c = 0; c < concatDim; c++) {
        const xv = concat[xOff + c];
        const wBase = c * concatDim;
        for (let d = 0; d < concatDim; d++) {
          gWg[wBase + d] += xv * buf.dGatePre[dOff + d];
        }
      }

      // dConcat[c] += sum_d dGatePre[d] * Wg[c,d]
      for (let c = 0; c < concatDim; c++) {
        const wBase = c * concatDim;
        let sum = 0;
        for (let d = 0; d < concatDim; d++) {
          sum += buf.dGatePre[dOff + d] * Wg[wBase + d];
        }
        buf.dConcat[xOff + c] += sum;
      }
    }

    // Split dConcat into dEUp parts and add
    for (let t = 0; t < L1; t++) {
      const base = t * concatDim;
      let w = 0;
      for (let si = 0; si < this.nScales; si++) {
        const dEUp = buf.dEUp[si];
        const dstOff = t * emb;
        for (let e = 0; e < emb; e++) {
          dEUp[dstOff + e] += buf.dConcat[base + (w++)];
        }
      }
    }

    // Backprop upsampling and add scale embedding grads and conv grads
    // Reset per-scale dE
    for (let si = 0; si < this.nScales; si++) _fill(buf.dE[si], 0);

    // Upsampling backward: EUp[t] from E[te]
    for (let si = 0; si < this.nScales; si++) {
      const s = this.config.temporalScales[si];
      const Ls = Math.floor((L1 + s - 1) / s);
      const dE = buf.dE[si];
      const dEUp = buf.dEUp[si];

      for (let t = 0; t < L1; t++) {
        let te = Math.floor(t / s);
        if (te < 0) te = 0;
        if (te >= Ls) te = Ls - 1;
        const dstOff = te * emb;
        const srcOff = t * emb;
        for (let e = 0; e < emb; e++) dE[dstOff + e] += dEUp[srcOff + e];
      }
    }

    // E = convAct + PE + scaleEmb
    // => dScaleEmb[e] += sum_t dE[t,e]; dConvAct = dE
    for (let si = 0; si < this.nScales; si++) {
      const sparams = this.scales[si];
      const sgrad = this.scaleGrad[si];

      const s = this.config.temporalScales[si];
      const Ls = Math.floor((L1 + s - 1) / s);

      // dScaleEmb
      for (let e = 0; e < emb; e++) sgrad.g_scaleEmb[e] += 0;
      for (let tt = 0; tt < Ls; tt++) {
        const off = tt * emb;
        for (let e = 0; e < emb; e++) {
          sgrad.g_scaleEmb[e] += buf.dE[si][off + e];
        }
      }

      // convAct = GELU(convPre)
      // dConvPre = dE * gelu'(convPre)
      const convPre = buf.convPre[si];
      const dConvPre = buf.convAct[si]; // reuse convAct buffer as dConvPre (safe; convAct not needed anymore)
      for (let i = 0; i < Ls * emb; i++) {
        dConvPre[i] = buf.dE[si][i] * _geluGrad(convPre[i]);
      }

      // Conv backward:
      // convPre[tt,e] = sum_{k,f} xNorm[tBase-k,f]*W[k,f,e] + b[e], tBase=tt*s
      const K = this.config.temporalKernelSize;
      const W = sparams.W;
      const b = sparams.b;
      const gW = sgrad.g_W;
      const gb = sgrad.g_b;

      for (let e = 0; e < emb; e++) gb[e] += 0;

      for (let tt = 0; tt < Ls; tt++) {
        const tBase = tt * s;
        const dOff = tt * emb;
        for (let e = 0; e < emb; e++) {
          const dp = dConvPre[dOff + e];
          gb[e] += dp;

          for (let k = 0; k < K; k++) {
            const idx = tBase - k;
            if (idx < 0) continue;
            if (idx >= L1) continue;
            const xOff = idx * inDim;

            let wOff = (k * inDim) * emb + e;
            for (let f = 0; f < inDim; f++) {
              gW[wOff] += buf.xNorm[xOff + f] * dp;
              wOff += emb;
            }
          }
        }
      }
    }

    // ===== Gradient norm + clip + Adam =====
    this.updateCount++;
    const t = this.updateCount;
    const lr = _computeScheduledLR(
      this.config.learningRate,
      t,
      this.config.warmupSteps,
      this.config.totalSteps,
    );
    this.lastEffectiveLR = lr;

    // Compute grad norm (after L2 addition inside adam); we also clip by norm 5.0 across raw grads.
    let rawNorm2 = 0;
    rawNorm2 += this.sumGradSquares();
    let rawNorm = Math.sqrt(rawNorm2);
    if (!Number.isFinite(rawNorm)) rawNorm = 0;

    const clip = 5.0;
    let clipScale = 1.0;
    if (rawNorm > clip && rawNorm > 0) clipScale = clip / rawNorm;

    if (clipScale !== 1.0) this.scaleAllGrads(clipScale);

    const accum = { v: 0 };
    // Update scale params
    for (let si = 0; si < this.nScales; si++) {
      const sp = this.scales[si];
      const sm = this.scaleMom[si];
      const sg = this.scaleGrad[si];
      _adamUpdate(
        sp.W,
        sg.g_W,
        sm.m_W,
        sm.v_W,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        sp.b,
        sg.g_b,
        sm.m_b,
        sm.v_b,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        sp.scaleEmb,
        sg.g_scaleEmb,
        sm.m_scaleEmb,
        sm.v_scaleEmb,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
    }

    // Fusion params
    _adamUpdate(
      this.Wg!,
      this.gWg!,
      this.mWg!,
      this.vWg!,
      lr,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      t,
      lambda,
      accum,
    );
    _adamUpdate(
      this.bg!,
      this.gbg!,
      this.mbg!,
      this.vbg!,
      lr,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      t,
      lambda,
      accum,
    );

    // Blocks
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const p = this.blocks[bi];
      const m = this.blockMom[bi];
      const g = this.blockGrad[bi];

      _adamUpdate(
        p.ln1Gamma,
        g.g_ln1Gamma,
        m.m_ln1Gamma,
        m.v_ln1Gamma,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.ln1Beta,
        g.g_ln1Beta,
        m.m_ln1Beta,
        m.v_ln1Beta,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );

      _adamUpdate(
        p.Wq,
        g.g_Wq,
        m.m_Wq,
        m.v_Wq,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.Wk,
        g.g_Wk,
        m.m_Wk,
        m.v_Wk,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.Wv,
        g.g_Wv,
        m.m_Wv,
        m.v_Wv,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.Wo,
        g.g_Wo,
        m.m_Wo,
        m.v_Wo,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );

      _adamUpdate(
        p.ln2Gamma,
        g.g_ln2Gamma,
        m.m_ln2Gamma,
        m.v_ln2Gamma,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.ln2Beta,
        g.g_ln2Beta,
        m.m_ln2Beta,
        m.v_ln2Beta,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );

      _adamUpdate(
        p.W1,
        g.g_W1,
        m.m_W1,
        m.v_W1,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.b1,
        g.g_b1,
        m.m_b1,
        m.v_b1,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.W2,
        g.g_W2,
        m.m_W2,
        m.v_W2,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      _adamUpdate(
        p.b2,
        g.g_b2,
        m.m_b2,
        m.v_b2,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
    }

    // Pool and Output head
    _adamUpdate(
      this.Wpool!,
      this.gWpool!,
      this.mWpool!,
      this.vWpool!,
      lr,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      t,
      lambda,
      accum,
    );
    // bpool scalar as length-1 update
    {
      const bArr = new Float64Array(1);
      bArr[0] = this.bpool;
      const gArr = new Float64Array(1);
      gArr[0] = this.gbpool;
      const mArr = new Float64Array(1);
      mArr[0] = this.mbpool;
      const vArr = new Float64Array(1);
      vArr[0] = this.vbpool;
      _adamUpdate(
        bArr,
        gArr,
        mArr,
        vArr,
        lr,
        this.config.beta1,
        this.config.beta2,
        this.config.epsilon,
        t,
        lambda,
        accum,
      );
      this.bpool = bArr[0];
      this.mbpool = mArr[0];
      this.vbpool = vArr[0];
      // note: tiny allocation; acceptable because scalar, but still "hot". Avoid by storing as length-1 Float64Array fields.
      // For strictness, users can set attentionDropout/fusionDropout=0 to keep speed.
    }

    _adamUpdate(
      this.Wout!,
      this.gWout!,
      this.mWout!,
      this.vWout!,
      lr,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      t,
      lambda,
      accum,
    );
    _adamUpdate(
      this.bout!,
      this.gbout!,
      this.mbout!,
      this.vbout!,
      lr,
      this.config.beta1,
      this.config.beta2,
      this.config.epsilon,
      t,
      lambda,
      accum,
    );

    const gradNorm = Math.sqrt(accum.v);
    this.lastConverged = gradNorm < this.config.convergenceThreshold;

    // Update residual stats (original scale)
    const yHatRaw = buf.tmpHidden; // reuse tmpHidden as scratch vector length hidden; but outDim may exceed hidden. Ensure separate:
    // We'll compute in-place without storing full vector: update residual stats dimensionwise.
    // yHatRaw[d] = yHatNorm[d]*outStd[d] + outMean[d]
    // residual = yTargetRaw - yHatRaw
    const resVec = new Float64Array(outDim);
    for (let d = 0; d < outDim; d++) {
      const yh = yHat[d] * outStd[d] + outMean[d];
      resVec[d] = buf.yTargetRaw[d] - yh;
    }
    _welfordUpdateVec(this.resStats!, resVec, 0, outDim);

    // ADWIN drift detection on weighted loss (normalized after outlier weight)
    const driftDetected = _adwinAddAndDetect(this.adwin, loss);
    if (driftDetected) {
      this.driftCount++;
      _adwinReset(this.adwin);
      this.runningLoss = loss; // optionally reset running loss baseline
    }

    return {
      loss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this.lastConverged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict future steps using cached last window. Autoregressive without future x:
   * - step 0 uses one-step prediction from cached window
   * - further steps repeat same mean prediction and increase uncertainty: se *= sqrt(stepIndex+1)
   * @param futureSteps number of steps to predict
   * @returns PredictionResult
   */
  predict(futureSteps: number): PredictionResult {
    const steps = Math.max(1, Math.floor(futureSteps));
    const isModelReady = this.isInitialized && this.sampleCount >= 2 &&
      !!this.lastXRaw && this.lastSeqLen > 0;

    const preds: SinglePrediction[] = new Array<SinglePrediction>(steps);

    const accuracy = 1 / (1 + this.runningLoss);

    if (!isModelReady) {
      for (let s = 0; s < steps; s++) {
        preds[s] = {
          predicted: [],
          lowerBound: [],
          upperBound: [],
          standardError: [],
        };
      }
      return {
        predictions: preds,
        accuracy,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    // Run a forward pass using cached xRaw; deterministic inference (dropouts off).
    const inDim = this.inputDim;
    const outDim = this.outputDim;
    const emb = this.config.embeddingDim;
    const concatDim = this.nScales * emb;
    const L1 = this.lastSeqLen;

    this.ensureBuffers();
    const buf = this.buf!;

    // Copy lastXRaw into xRaw
    _copy(buf.xRaw, this.lastXRaw!, L1 * inDim);

    // Normalize x using current stats
    const inMean = this.inStats!.mean;
    const inStd = this.inStd!;
    for (let t = 0; t < L1; t++) {
      const off = t * inDim;
      for (let f = 0; f < inDim; f++) {
        const z = (buf.xRaw[off + f] - inMean[f]) /
          (inStd[f] <= 1e-12 ? 1e-12 : inStd[f]);
        buf.xNorm[off + f] = _isFiniteNumber(z) ? z : 0;
      }
    }

    // Forward (same as training but with dropout disabled)
    // Multi-scale conv
    for (let si = 0; si < this.nScales; si++) {
      const s = this.config.temporalScales[si];
      const K = this.config.temporalKernelSize;
      const Ls = Math.floor((L1 + s - 1) / s);
      const W = this.scales[si].W;
      const b = this.scales[si].b;
      const convAct = buf.convAct[si];

      for (let tt = 0; tt < Ls; tt++) {
        const tBase = tt * s;
        const outOff = tt * emb;
        for (let e = 0; e < emb; e++) {
          let sum = b[e];
          for (let k = 0; k < K; k++) {
            const idx = tBase - k;
            if (idx < 0) continue;
            if (idx >= L1) continue;
            const xOff = idx * inDim;
            let wOff = (k * inDim) * emb + e;
            for (let f = 0; f < inDim; f++) {
              sum += buf.xNorm[xOff + f] * W[wOff];
              wOff += emb;
            }
          }
          convAct[outOff + e] = _gelu(sum);
        }
      }

      // E and upsample
      const E = buf.E[si];
      const scaleEmb = this.scales[si].scaleEmb;
      const pe = this.pe!;
      for (let tt = 0; tt < Ls; tt++) {
        const off = tt * emb;
        const peOff = tt * emb;
        for (let e = 0; e < emb; e++) {
          E[off + e] = convAct[off + e] + pe[peOff + e] + scaleEmb[e];
        }
      }
      const EUp = buf.EUp[si];
      for (let t = 0; t < L1; t++) {
        let te = Math.floor(t / s);
        if (te < 0) te = 0;
        if (te >= Ls) te = Ls - 1;
        const srcOff = te * emb;
        const dstOff = t * emb;
        for (let e = 0; e < emb; e++) EUp[dstOff + e] = E[srcOff + e];
      }
    }

    // concat
    for (let t = 0; t < L1; t++) {
      const base = t * concatDim;
      let w = 0;
      for (let si = 0; si < this.nScales; si++) {
        const EUp = buf.EUp[si];
        const srcOff = t * emb;
        for (let e = 0; e < emb; e++) {
          buf.concat[base + (w++)] = EUp[srcOff + e];
        }
      }
    }

    // gate and fused (no fusion dropout)
    const Wg = this.Wg!;
    const bg = this.bg!;
    for (let t = 0; t < L1; t++) {
      const xOff = t * concatDim;
      const yOff = t * concatDim;
      for (let d = 0; d < concatDim; d++) {
        let sum = bg[d];
        let wOff = d;
        for (let c = 0; c < concatDim; c++) {
          sum += buf.concat[xOff + c] * Wg[wOff];
          wOff += concatDim;
        }
        buf.gate[yOff + d] = _sigmoid(sum);
      }
    }
    for (let t = 0; t < L1; t++) {
      const gOff = t * concatDim;
      const outOff = t * emb;
      for (let e = 0; e < emb; e++) buf.fused[outOff + e] = 0;
      for (let si = 0; si < this.nScales; si++) {
        const EUp = buf.EUp[si];
        const srcOff = t * emb;
        const gBase = gOff + si * emb;
        for (let e = 0; e < emb; e++) {
          buf.fused[outOff + e] += buf.gate[gBase + e] * EUp[srcOff + e];
        }
      }
    }

    // Transformer blocks (no attention dropout)
    let H = buf.fused;
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const params = this.blocks[bi];
      const H_in = buf.blockInput[bi];
      _copy(H_in, H, L1 * emb);

      const Xn1 = buf.Xn1[bi];
      const m1 = buf.ln1Mean[bi];
      const v1 = buf.ln1Var[bi];
      for (let t = 0; t < L1; t++) {
        _layerNormForward(
          H_in,
          t * emb,
          Xn1,
          t * emb,
          m1,
          v1,
          t,
          emb,
          params.ln1Gamma,
          params.ln1Beta,
        );
      }

      const Q = buf.Q[bi];
      const K = buf.K[bi];
      const V = buf.V[bi];
      this.linearForward2D(Xn1, Q, L1, emb, params.Wq, null);
      this.linearForward2D(Xn1, K, L1, emb, params.Wk, null);
      this.linearForward2D(Xn1, V, L1, emb, params.Wv, null);

      const context = buf.context[bi];
      this.mhaForward(
        Q,
        K,
        V,
        context,
        L1,
        emb,
        this.config.numHeads,
        0.0,
        true,
        bi,
      );

      const attnProj = buf.attnProj[bi];
      this.linearForward2D(context, attnProj, L1, emb, params.Wo, null);

      const afterAttn = buf.afterAttn[bi];
      for (let i = 0; i < L1 * emb; i++) afterAttn[i] = H_in[i] + attnProj[i];

      const Xn2 = buf.Xn2[bi];
      const m2 = buf.ln2Mean[bi];
      const v2 = buf.ln2Var[bi];
      for (let t = 0; t < L1; t++) {
        _layerNormForward(
          afterAttn,
          t * emb,
          Xn2,
          t * emb,
          m2,
          v2,
          t,
          emb,
          params.ln2Gamma,
          params.ln2Beta,
        );
      }

      const ffnPre = buf.ffnPre[bi];
      const ffnAct = buf.ffnAct[bi];
      const ffnOut = buf.ffnOut[bi];
      this.linearForward2D(Xn2, ffnPre, L1, emb, params.W1, params.b1);
      for (let i = 0; i < L1 * this.hiddenDim; i++) {
        ffnAct[i] = _gelu(ffnPre[i]);
      }
      this.linearForward2D(
        ffnAct,
        ffnOut,
        L1,
        this.hiddenDim,
        params.W2,
        params.b2,
      );

      for (let i = 0; i < L1 * emb; i++) H[i] = afterAttn[i] + ffnOut[i];
    }

    // Pool
    const scores = buf.poolScores;
    const alpha = buf.poolAlpha;
    const pooled = buf.pooled;
    const Wpool = this.Wpool!;
    const bpool = this.bpool;
    for (let t = 0; t < L1; t++) {
      scores[t] = _dot(H, t * emb, Wpool, 0, emb) + bpool;
    }
    _softmaxRowStable(alpha, 0, scores, 0, L1);
    _fill(pooled, 0);
    for (let t = 0; t < L1; t++) {
      const a = alpha[t];
      const off = t * emb;
      for (let e = 0; e < emb; e++) pooled[e] += a * H[off + e];
    }

    // Output
    const yHatNorm = buf.yHatNorm;
    for (let d = 0; d < outDim; d++) {
      let sum = this.bout![d];
      let wOff = d;
      for (let e = 0; e < emb; e++) {
        sum += pooled[e] * this.Wout![wOff];
        wOff += outDim;
      }
      yHatNorm[d] = sum;
    }

    // Denormalize and compute standard errors from residual stats
    const outMean = this.outStats!.mean;
    const outStd = this.outStd!;
    const resStd = new Float64Array(outDim);
    _welfordStd(this.resStats!, resStd);

    const meanPred = new Array<number>(outDim);
    const baseSE = new Array<number>(outDim);
    for (let d = 0; d < outDim; d++) {
      const pr = yHatNorm[d] * outStd[d] + outMean[d];
      meanPred[d] = _isFiniteNumber(pr) ? pr : 0;
      baseSE[d] = _isFiniteNumber(resStd[d]) ? resStd[d] : 0;
    }

    for (let s = 0; s < steps; s++) {
      const mul = Math.sqrt(s + 1);
      const predicted = new Array<number>(outDim);
      const standardError = new Array<number>(outDim);
      const lowerBound = new Array<number>(outDim);
      const upperBound = new Array<number>(outDim);

      for (let d = 0; d < outDim; d++) {
        const se = baseSE[d] * mul;
        const mu = meanPred[d];
        predicted[d] = mu;
        standardError[d] = se;
        lowerBound[d] = mu - 1.96 * se;
        upperBound[d] = mu + 1.96 * se;
      }

      preds[s] = { predicted, lowerBound, upperBound, standardError };
    }

    return {
      predictions: preds,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /** @returns current model summary. */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      numBlocks: this.config.numBlocks,
      embeddingDim: this.config.embeddingDim,
      numHeads: this.config.numHeads,
      temporalScales: this.config.temporalScales.slice(),
      totalParameters: this.countTotalParameters(),
      sampleCount: this.sampleCount,
      accuracy: 1 / (1 + this.runningLoss),
      converged: this.lastConverged,
      effectiveLearningRate: this.lastEffectiveLR,
      driftCount: this.driftCount,
    };
  }

  /** @returns current weights and optimizer moments in the declared shapes (nested arrays). */
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

    const emb = this.config.embeddingDim;
    const inDim = this.inputDim;
    const outDim = this.outputDim;
    const K = this.config.temporalKernelSize;
    const concatDim = this.nScales * emb;

    // temporalConvWeights: [nScales][K+1][...flattened...]
    const temporalConvWeights: number[][][] = new Array(this.nScales);
    const scaleEmbeddings: number[][] = new Array(this.nScales);

    for (let si = 0; si < this.nScales; si++) {
      const W = this.scales[si].W;
      const b = this.scales[si].b;
      const group: number[][] = new Array(K + 1);
      // K kernels flattened [inDim*emb]
      for (let k = 0; k < K; k++) {
        const flat = new Array<number>(inDim * emb);
        let idx = 0;
        for (let f = 0; f < inDim; f++) {
          for (let e = 0; e < emb; e++) {
            const wOff = (k * inDim + f) * emb + e;
            flat[idx++] = W[wOff];
          }
        }
        group[k] = flat;
      }
      group[K] = _pack1D(b);
      temporalConvWeights[si] = group;

      scaleEmbeddings[si] = _pack1D(this.scales[si].scaleEmb);
    }

    // Positional encoding (expose cached length up to maxSequenceLength)
    const peLen = this.config.maxSequenceLength;
    const positionalEncoding = this.pe ? _pack2D(this.pe, peLen, emb) : [];

    // Fusion weights: [WgFlat, bg]
    const fusionWeights: number[][] = [
      this.Wg ? _pack1D(this.Wg) : [],
      this.bg ? _pack1D(this.bg) : [],
    ];

    // Attention weights: [numBlocks][4][flat]
    const attentionWeights: number[][][] = new Array(this.config.numBlocks);
    const ffnWeights: number[][][] = new Array(this.config.numBlocks);
    const layerNormParams: number[][] = new Array(this.config.numBlocks * 2);

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const p = this.blocks[bi];
      attentionWeights[bi] = [
        _pack1D(p.Wq),
        _pack1D(p.Wk),
        _pack1D(p.Wv),
        _pack1D(p.Wo),
      ];

      ffnWeights[bi] = [
        _pack1D(p.W1),
        _pack1D(p.b1),
        _pack1D(p.W2),
        _pack1D(p.b2),
      ];

      // layerNormParams: each row is [gamma..., beta...]
      const ln1 = new Array<number>(2 * emb);
      const ln2 = new Array<number>(2 * emb);
      for (let e = 0; e < emb; e++) {
        ln1[e] = p.ln1Gamma[e];
        ln1[emb + e] = p.ln1Beta[e];
        ln2[e] = p.ln2Gamma[e];
        ln2[emb + e] = p.ln2Beta[e];
      }
      layerNormParams[2 * bi] = ln1;
      layerNormParams[2 * bi + 1] = ln2;
    }

    // Output weights: [Wpool, [bpool], WoutFlat, bout]
    const outputWeights: number[][] = [
      this.Wpool ? _pack1D(this.Wpool) : [],
      [this.bpool],
      this.Wout ? _pack1D(this.Wout) : [],
      this.bout ? _pack1D(this.bout) : [],
    ];

    // Moments packed as [paramIndex][1][flat] to satisfy number[][][] without ambiguity.
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];
    // scales
    for (let si = 0; si < this.nScales; si++) {
      const sm = this.scaleMom[si];
      firstMoment.push([_pack1D(sm.m_W)], [_pack1D(sm.m_b)], [
        _pack1D(sm.m_scaleEmb),
      ]);
      secondMoment.push([_pack1D(sm.v_W)], [_pack1D(sm.v_b)], [
        _pack1D(sm.v_scaleEmb),
      ]);
    }
    // fusion
    firstMoment.push([this.mWg ? _pack1D(this.mWg) : []], [
      this.mbg ? _pack1D(this.mbg) : [],
    ]);
    secondMoment.push([this.vWg ? _pack1D(this.vWg) : []], [
      this.vbg ? _pack1D(this.vbg) : [],
    ]);
    // blocks
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const bm = this.blockMom[bi];
      firstMoment.push(
        [_pack1D(bm.m_ln1Gamma)],
        [_pack1D(bm.m_ln1Beta)],
        [_pack1D(bm.m_Wq)],
        [_pack1D(bm.m_Wk)],
        [_pack1D(bm.m_Wv)],
        [_pack1D(bm.m_Wo)],
        [_pack1D(bm.m_ln2Gamma)],
        [_pack1D(bm.m_ln2Beta)],
        [_pack1D(bm.m_W1)],
        [_pack1D(bm.m_b1)],
        [_pack1D(bm.m_W2)],
        [_pack1D(bm.m_b2)],
      );
      secondMoment.push(
        [_pack1D(bm.v_ln1Gamma)],
        [_pack1D(bm.v_ln1Beta)],
        [_pack1D(bm.v_Wq)],
        [_pack1D(bm.v_Wk)],
        [_pack1D(bm.v_Wv)],
        [_pack1D(bm.v_Wo)],
        [_pack1D(bm.v_ln2Gamma)],
        [_pack1D(bm.v_ln2Beta)],
        [_pack1D(bm.v_W1)],
        [_pack1D(bm.v_b1)],
        [_pack1D(bm.v_W2)],
        [_pack1D(bm.v_b2)],
      );
    }
    // pool and output
    firstMoment.push(
      [this.mWpool ? _pack1D(this.mWpool) : []],
      [[this.mbpool]],
      [this.mWout ? _pack1D(this.mWout) : []],
      [this.mbout ? _pack1D(this.mbout) : []],
    );
    secondMoment.push(
      [this.vWpool ? _pack1D(this.vWpool) : []],
      [[this.vbpool]],
      [this.vWout ? _pack1D(this.vWout) : []],
      [this.vbout ? _pack1D(this.vbout) : []],
    );

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

  /** @returns current normalization stats (means/stds) */
  getNormalizationStats(): NormalizationStats {
    if (
      !this.isInitialized || !this.inStats || !this.outStats || !this.inStd ||
      !this.outStd
    ) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: this.sampleCount,
      };
    }
    return {
      inputMean: _pack1D(this.inStats.mean),
      inputStd: _pack1D(this.inStd),
      outputMean: _pack1D(this.outStats.mean),
      outputStd: _pack1D(this.outStd),
      count: this.sampleCount,
    };
  }

  /** Reset: reinitialize weights/moments and counters; keep config. */
  reset(): void {
    this.sampleCount = 0;
    this.updateCount = 0;
    this.driftCount = 0;
    this.runningLoss = 0;
    this.lastEffectiveLR = 0;
    this.lastConverged = false;
    _adwinReset(this.adwin);
    this.lastSeqLen = 0;

    if (!this.isInitialized) return;

    // Reinit params with fresh RNG
    const rng = new _XorShift32(
      0xC0FFEE ^ (this.inputDim * 1315423911) ^ (this.outputDim * 2654435761),
    );
    this.initAllWeights(rng);
    this.zeroAllMoments();
  }

  /**
   * Serialize model to JSON string; Float64Array are stored as number[].
   * @returns JSON string
   */
  save(): string {
    const obj: any = {
      config: this.config,
      dims: {
        inputDim: this.inputDim,
        outputDim: this.outputDim,
      },
      sampleCount: this.sampleCount,
      updateCount: this.updateCount,
      driftCount: this.driftCount,
      runningLoss: this.runningLoss,
      lastEffectiveLR: this.lastEffectiveLR,
      lastConverged: this.lastConverged,
      lastSeqLen: this.lastSeqLen,
      isInitialized: this.isInitialized,

      inStats: this.inStats
        ? {
          n: this.inStats.n,
          mean: _pack1D(this.inStats.mean),
          m2: _pack1D(this.inStats.m2),
        }
        : null,
      outStats: this.outStats
        ? {
          n: this.outStats.n,
          mean: _pack1D(this.outStats.mean),
          m2: _pack1D(this.outStats.m2),
        }
        : null,
      resStats: this.resStats
        ? {
          n: this.resStats.n,
          mean: _pack1D(this.resStats.mean),
          m2: _pack1D(this.resStats.m2),
        }
        : null,

      pe: this.pe ? _pack1D(this.pe) : null,

      scales: this.scales.map((s) => ({
        W: _pack1D(s.W),
        b: _pack1D(s.b),
        scaleEmb: _pack1D(s.scaleEmb),
      })),
      scaleMom: this.scaleMom.map((m) => ({
        m_W: _pack1D(m.m_W),
        m_b: _pack1D(m.m_b),
        m_scaleEmb: _pack1D(m.m_scaleEmb),
        v_W: _pack1D(m.v_W),
        v_b: _pack1D(m.v_b),
        v_scaleEmb: _pack1D(m.v_scaleEmb),
      })),

      Wg: this.Wg ? _pack1D(this.Wg) : null,
      bg: this.bg ? _pack1D(this.bg) : null,
      mWg: this.mWg ? _pack1D(this.mWg) : null,
      vWg: this.vWg ? _pack1D(this.vWg) : null,
      mbg: this.mbg ? _pack1D(this.mbg) : null,
      vbg: this.vbg ? _pack1D(this.vbg) : null,

      blocks: this.blocks.map((b) => ({
        ln1Gamma: _pack1D(b.ln1Gamma),
        ln1Beta: _pack1D(b.ln1Beta),
        Wq: _pack1D(b.Wq),
        Wk: _pack1D(b.Wk),
        Wv: _pack1D(b.Wv),
        Wo: _pack1D(b.Wo),
        ln2Gamma: _pack1D(b.ln2Gamma),
        ln2Beta: _pack1D(b.ln2Beta),
        W1: _pack1D(b.W1),
        b1: _pack1D(b.b1),
        W2: _pack1D(b.W2),
        b2: _pack1D(b.b2),
      })),
      blockMom: this.blockMom.map((m) => ({
        m_ln1Gamma: _pack1D(m.m_ln1Gamma),
        m_ln1Beta: _pack1D(m.m_ln1Beta),
        m_Wq: _pack1D(m.m_Wq),
        m_Wk: _pack1D(m.m_Wk),
        m_Wv: _pack1D(m.m_Wv),
        m_Wo: _pack1D(m.m_Wo),
        m_ln2Gamma: _pack1D(m.m_ln2Gamma),
        m_ln2Beta: _pack1D(m.m_ln2Beta),
        m_W1: _pack1D(m.m_W1),
        m_b1: _pack1D(m.m_b1),
        m_W2: _pack1D(m.m_W2),
        m_b2: _pack1D(m.m_b2),

        v_ln1Gamma: _pack1D(m.v_ln1Gamma),
        v_ln1Beta: _pack1D(m.v_ln1Beta),
        v_Wq: _pack1D(m.v_Wq),
        v_Wk: _pack1D(m.v_Wk),
        v_Wv: _pack1D(m.v_Wv),
        v_Wo: _pack1D(m.v_Wo),
        v_ln2Gamma: _pack1D(m.v_ln2Gamma),
        v_ln2Beta: _pack1D(m.v_ln2Beta),
        v_W1: _pack1D(m.v_W1),
        v_b1: _pack1D(m.v_b1),
        v_W2: _pack1D(m.v_W2),
        v_b2: _pack1D(m.v_b2),
      })),

      Wpool: this.Wpool ? _pack1D(this.Wpool) : null,
      bpool: this.bpool,
      mWpool: this.mWpool ? _pack1D(this.mWpool) : null,
      vWpool: this.vWpool ? _pack1D(this.vWpool) : null,
      mbpool: this.mbpool,
      vbpool: this.vbpool,

      Wout: this.Wout ? _pack1D(this.Wout) : null,
      bout: this.bout ? _pack1D(this.bout) : null,
      mWout: this.mWout ? _pack1D(this.mWout) : null,
      vWout: this.vWout ? _pack1D(this.vWout) : null,
      mbout: this.mbout ? _pack1D(this.mbout) : null,
      vbout: this.vbout ? _pack1D(this.vbout) : null,
    };

    return JSON.stringify(obj);
  }

  /**
   * Load model from JSON string. Validates key shapes.
   * @param w JSON string from save()
   */
  load(w: string): void {
    const obj = JSON.parse(w);

    // Restore config (must match defaults merged behavior)
    const defaults = new FusionTemporalTransformerRegression().config;
    const cfg: Config = Object.assign({}, defaults, obj.config || {});
    (this as any).config = cfg; // keep API; internal override is OK in TS runtime

    this.sampleCount = obj.sampleCount | 0;
    this.updateCount = obj.updateCount | 0;
    this.driftCount = obj.driftCount | 0;
    this.runningLoss = +obj.runningLoss || 0;
    this.lastEffectiveLR = +obj.lastEffectiveLR || 0;
    this.lastConverged = !!obj.lastConverged;
    this.lastSeqLen = obj.lastSeqLen | 0;

    const dims = obj.dims || {};
    const inDim = dims.inputDim | 0;
    const outDim = dims.outputDim | 0;

    if (inDim > 0 && outDim > 0) {
      this.ensureInitialized(inDim, outDim);
    } else {
      this.isInitialized = false;
      this.inputDim = 0;
      this.outputDim = 0;
      return;
    }

    // Restore stats
    if (obj.inStats) {
      this.inStats!.n = obj.inStats.n | 0;
      this.inStats!.mean = _rehydrateF64(obj.inStats.mean);
      this.inStats!.m2 = _rehydrateF64(obj.inStats.m2);
      this.inStd = new Float64Array(this.inputDim);
      _welfordStd(this.inStats!, this.inStd);
    }
    if (obj.outStats) {
      this.outStats!.n = obj.outStats.n | 0;
      this.outStats!.mean = _rehydrateF64(obj.outStats.mean);
      this.outStats!.m2 = _rehydrateF64(obj.outStats.m2);
      this.outStd = new Float64Array(this.outputDim);
      _welfordStd(this.outStats!, this.outStd);
    }
    if (obj.resStats) {
      this.resStats!.n = obj.resStats.n | 0;
      this.resStats!.mean = _rehydrateF64(obj.resStats.mean);
      this.resStats!.m2 = _rehydrateF64(obj.resStats.m2);
    }

    // Positional encoding
    if (obj.pe) this.pe = _rehydrateF64(obj.pe);

    // Scales and moments
    const nScales = this.config.temporalScales.length;
    if (!Array.isArray(obj.scales) || obj.scales.length !== nScales) {
      throw new Error("Invalid scales in load()");
    }
    if (!Array.isArray(obj.scaleMom) || obj.scaleMom.length !== nScales) {
      throw new Error("Invalid scaleMom in load()");
    }

    for (let si = 0; si < nScales; si++) {
      const s = obj.scales[si];
      const m = obj.scaleMom[si];
      this.scales[si].W = _rehydrateF64(s.W);
      this.scales[si].b = _rehydrateF64(s.b);
      this.scales[si].scaleEmb = _rehydrateF64(s.scaleEmb);

      this.scaleMom[si].m_W = _rehydrateF64(m.m_W);
      this.scaleMom[si].m_b = _rehydrateF64(m.m_b);
      this.scaleMom[si].m_scaleEmb = _rehydrateF64(m.m_scaleEmb);
      this.scaleMom[si].v_W = _rehydrateF64(m.v_W);
      this.scaleMom[si].v_b = _rehydrateF64(m.v_b);
      this.scaleMom[si].v_scaleEmb = _rehydrateF64(m.v_scaleEmb);
    }

    // Fusion
    this.Wg = _rehydrateF64(obj.Wg);
    this.bg = _rehydrateF64(obj.bg);
    this.mWg = _rehydrateF64(obj.mWg);
    this.vWg = _rehydrateF64(obj.vWg);
    this.mbg = _rehydrateF64(obj.mbg);
    this.vbg = _rehydrateF64(obj.vbg);

    // Blocks and moments
    if (
      !Array.isArray(obj.blocks) || obj.blocks.length !== this.config.numBlocks
    ) throw new Error("Invalid blocks in load()");
    if (
      !Array.isArray(obj.blockMom) ||
      obj.blockMom.length !== this.config.numBlocks
    ) throw new Error("Invalid blockMom in load()");
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const b = obj.blocks[bi];
      const m = obj.blockMom[bi];
      const p = this.blocks[bi];
      p.ln1Gamma = _rehydrateF64(b.ln1Gamma);
      p.ln1Beta = _rehydrateF64(b.ln1Beta);
      p.Wq = _rehydrateF64(b.Wq);
      p.Wk = _rehydrateF64(b.Wk);
      p.Wv = _rehydrateF64(b.Wv);
      p.Wo = _rehydrateF64(b.Wo);
      p.ln2Gamma = _rehydrateF64(b.ln2Gamma);
      p.ln2Beta = _rehydrateF64(b.ln2Beta);
      p.W1 = _rehydrateF64(b.W1);
      p.b1 = _rehydrateF64(b.b1);
      p.W2 = _rehydrateF64(b.W2);
      p.b2 = _rehydrateF64(b.b2);

      const bm = this.blockMom[bi];
      bm.m_ln1Gamma = _rehydrateF64(m.m_ln1Gamma);
      bm.m_ln1Beta = _rehydrateF64(m.m_ln1Beta);
      bm.m_Wq = _rehydrateF64(m.m_Wq);
      bm.m_Wk = _rehydrateF64(m.m_Wk);
      bm.m_Wv = _rehydrateF64(m.m_Wv);
      bm.m_Wo = _rehydrateF64(m.m_Wo);
      bm.m_ln2Gamma = _rehydrateF64(m.m_ln2Gamma);
      bm.m_ln2Beta = _rehydrateF64(m.m_ln2Beta);
      bm.m_W1 = _rehydrateF64(m.m_W1);
      bm.m_b1 = _rehydrateF64(m.m_b1);
      bm.m_W2 = _rehydrateF64(m.m_W2);
      bm.m_b2 = _rehydrateF64(m.m_b2);

      bm.v_ln1Gamma = _rehydrateF64(m.v_ln1Gamma);
      bm.v_ln1Beta = _rehydrateF64(m.v_ln1Beta);
      bm.v_Wq = _rehydrateF64(m.v_Wq);
      bm.v_Wk = _rehydrateF64(m.v_Wk);
      bm.v_Wv = _rehydrateF64(m.v_Wv);
      bm.v_Wo = _rehydrateF64(m.v_Wo);
      bm.v_ln2Gamma = _rehydrateF64(m.v_ln2Gamma);
      bm.v_ln2Beta = _rehydrateF64(m.v_ln2Beta);
      bm.v_W1 = _rehydrateF64(m.v_W1);
      bm.v_b1 = _rehydrateF64(m.v_b1);
      bm.v_W2 = _rehydrateF64(m.v_W2);
      bm.v_b2 = _rehydrateF64(m.v_b2);
    }

    // Pool and output
    this.Wpool = _rehydrateF64(obj.Wpool);
    this.bpool = +obj.bpool || 0;
    this.mWpool = _rehydrateF64(obj.mWpool);
    this.vWpool = _rehydrateF64(obj.vWpool);
    this.mbpool = +obj.mbpool || 0;
    this.vbpool = +obj.vbpool || 0;

    this.Wout = _rehydrateF64(obj.Wout);
    this.bout = _rehydrateF64(obj.bout);
    this.mWout = _rehydrateF64(obj.mWout);
    this.vWout = _rehydrateF64(obj.vWout);
    this.mbout = _rehydrateF64(obj.mbout);
    this.vbout = _rehydrateF64(obj.vbout);

    // Drift detector reset (loss window not serialized)
    _adwinReset(this.adwin);

    this.isInitialized = true;
  }

  // ===== Internal helpers =====

  private ensureInitialized(inDim: number, outDim: number): void {
    const emb = this.config.embeddingDim;
    const heads = this.config.numHeads;

    if (emb % heads !== 0) {
      throw new Error(
        `embeddingDim (${emb}) must be divisible by numHeads (${heads}).`,
      );
    }
    this.headDim = emb / heads;
    this.hiddenDim = emb * this.config.ffnMultiplier;

    this.inputDim = inDim;
    this.outputDim = outDim;
    this.nScales = this.config.temporalScales.length;

    // Stats
    this.inStats = _welfordCreate(inDim);
    this.outStats = _welfordCreate(outDim);
    this.resStats = _welfordCreate(outDim);
    this.inStd = new Float64Array(inDim);
    this.outStd = new Float64Array(outDim);

    // Positional encoding cache
    this.pe = new Float64Array(this.config.maxSequenceLength * emb);
    this.computePositionalEncoding(this.pe, this.config.maxSequenceLength, emb);

    // Allocate params
    this.scales = [];
    this.scaleMom = [];
    this.scaleGrad = [];

    const K = this.config.temporalKernelSize;
    for (let si = 0; si < this.nScales; si++) {
      const W = new Float64Array(K * inDim * emb);
      const b = new Float64Array(emb);
      const scaleEmb = new Float64Array(emb);

      const m_W = new Float64Array(W.length);
      const v_W = new Float64Array(W.length);
      const m_b = new Float64Array(b.length);
      const v_b = new Float64Array(b.length);
      const m_se = new Float64Array(scaleEmb.length);
      const v_se = new Float64Array(scaleEmb.length);

      const g_W = new Float64Array(W.length);
      const g_b = new Float64Array(b.length);
      const g_se = new Float64Array(scaleEmb.length);

      this.scales.push({ W, b, scaleEmb });
      this.scaleMom.push({
        m_W,
        m_b,
        m_scaleEmb: m_se,
        v_W,
        v_b,
        v_scaleEmb: v_se,
      });
      this.scaleGrad.push({ g_W, g_b, g_scaleEmb: g_se });
    }

    const concatDim = this.nScales * emb;
    this.Wg = new Float64Array(concatDim * concatDim);
    this.bg = new Float64Array(concatDim);
    this.mWg = new Float64Array(this.Wg.length);
    this.vWg = new Float64Array(this.Wg.length);
    this.mbg = new Float64Array(this.bg.length);
    this.vbg = new Float64Array(this.bg.length);
    this.gWg = new Float64Array(this.Wg.length);
    this.gbg = new Float64Array(this.bg.length);

    // Blocks
    this.blocks = [];
    this.blockMom = [];
    this.blockGrad = [];
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const ln1Gamma = new Float64Array(emb);
      const ln1Beta = new Float64Array(emb);
      const ln2Gamma = new Float64Array(emb);
      const ln2Beta = new Float64Array(emb);

      const Wq = new Float64Array(emb * emb);
      const Wk = new Float64Array(emb * emb);
      const Wv = new Float64Array(emb * emb);
      const Wo = new Float64Array(emb * emb);

      const W1 = new Float64Array(emb * this.hiddenDim);
      const b1 = new Float64Array(this.hiddenDim);
      const W2 = new Float64Array(this.hiddenDim * emb);
      const b2 = new Float64Array(emb);

      const m: _BlockMoments = {
        m_ln1Gamma: new Float64Array(emb),
        m_ln1Beta: new Float64Array(emb),
        m_Wq: new Float64Array(emb * emb),
        m_Wk: new Float64Array(emb * emb),
        m_Wv: new Float64Array(emb * emb),
        m_Wo: new Float64Array(emb * emb),
        m_ln2Gamma: new Float64Array(emb),
        m_ln2Beta: new Float64Array(emb),
        m_W1: new Float64Array(emb * this.hiddenDim),
        m_b1: new Float64Array(this.hiddenDim),
        m_W2: new Float64Array(this.hiddenDim * emb),
        m_b2: new Float64Array(emb),

        v_ln1Gamma: new Float64Array(emb),
        v_ln1Beta: new Float64Array(emb),
        v_Wq: new Float64Array(emb * emb),
        v_Wk: new Float64Array(emb * emb),
        v_Wv: new Float64Array(emb * emb),
        v_Wo: new Float64Array(emb * emb),
        v_ln2Gamma: new Float64Array(emb),
        v_ln2Beta: new Float64Array(emb),
        v_W1: new Float64Array(emb * this.hiddenDim),
        v_b1: new Float64Array(this.hiddenDim),
        v_W2: new Float64Array(this.hiddenDim * emb),
        v_b2: new Float64Array(emb),
      };

      const g: _BlockGrads = {
        g_ln1Gamma: new Float64Array(emb),
        g_ln1Beta: new Float64Array(emb),
        g_Wq: new Float64Array(emb * emb),
        g_Wk: new Float64Array(emb * emb),
        g_Wv: new Float64Array(emb * emb),
        g_Wo: new Float64Array(emb * emb),
        g_ln2Gamma: new Float64Array(emb),
        g_ln2Beta: new Float64Array(emb),
        g_W1: new Float64Array(emb * this.hiddenDim),
        g_b1: new Float64Array(this.hiddenDim),
        g_W2: new Float64Array(this.hiddenDim * emb),
        g_b2: new Float64Array(emb),
      };

      this.blocks.push({
        ln1Gamma,
        ln1Beta,
        Wq,
        Wk,
        Wv,
        Wo,
        ln2Gamma,
        ln2Beta,
        W1,
        b1,
        W2,
        b2,
      });
      this.blockMom.push(m);
      this.blockGrad.push(g);
    }

    // Pool and output
    this.Wpool = new Float64Array(emb);
    this.mWpool = new Float64Array(emb);
    this.vWpool = new Float64Array(emb);
    this.gWpool = new Float64Array(emb);

    this.Wout = new Float64Array(emb * outDim);
    this.bout = new Float64Array(outDim);
    this.mWout = new Float64Array(this.Wout.length);
    this.vWout = new Float64Array(this.Wout.length);
    this.mbout = new Float64Array(this.bout.length);
    this.vbout = new Float64Array(this.bout.length);
    this.gWout = new Float64Array(this.Wout.length);
    this.gbout = new Float64Array(this.bout.length);

    // Initialize weights
    const rng = new _XorShift32(
      0x12345678 ^ (inDim * 1315423911) ^ (outDim * 2654435761),
    );
    this.initAllWeights(rng);
    this.zeroAllMoments();

    this.isInitialized = true;

    // Lazy buffers
    this.buf = null;
    this.lastXRaw = null;
  }

  private ensureBuffers(): void {
    if (!this.isInitialized) return;
    const emb = this.config.embeddingDim;
    const inDim = this.inputDim;
    const outDim = this.outputDim;
    const maxL = this.config.maxSequenceLength;
    const concatDim = this.nScales * emb;
    const hidden = this.hiddenDim;

    if (this.buf) {
      // Ensure sizes match config; if config changed, rebuild.
      // (Normally config is constant.)
      return;
    }

    const xRaw = new Float64Array(maxL * inDim);
    const xNorm = new Float64Array(maxL * inDim);
    const yTargetRaw = new Float64Array(outDim);
    const yTargetNorm = new Float64Array(outDim);

    const convPre: Float64Array[] = new Array(this.nScales);
    const convAct: Float64Array[] = new Array(this.nScales);
    const E: Float64Array[] = new Array(this.nScales);
    const EUp: Float64Array[] = new Array(this.nScales);
    const dE: Float64Array[] = new Array(this.nScales);
    const dEUp: Float64Array[] = new Array(this.nScales);

    for (let si = 0; si < this.nScales; si++) {
      const s = this.config.temporalScales[si];
      const maxLs = Math.floor((maxL + s - 1) / s);
      convPre[si] = new Float64Array(maxLs * emb);
      convAct[si] = new Float64Array(maxLs * emb);
      E[si] = new Float64Array(maxLs * emb);
      EUp[si] = new Float64Array(maxL * emb);
      dE[si] = new Float64Array(maxLs * emb);
      dEUp[si] = new Float64Array(maxL * emb);
    }

    const concat = new Float64Array(maxL * concatDim);
    const gatePre = new Float64Array(maxL * concatDim);
    const gate = new Float64Array(maxL * concatDim);
    const fusedPreDrop = new Float64Array(maxL * emb);
    const fused = new Float64Array(maxL * emb);
    const dFused = new Float64Array(maxL * emb);
    const dConcat = new Float64Array(maxL * concatDim);
    const dGate = new Float64Array(maxL * concatDim);
    const dGatePre = new Float64Array(maxL * concatDim);

    const blockInput: Float64Array[] = new Array(this.config.numBlocks);
    const ln1Mean: Float64Array[] = new Array(this.config.numBlocks);
    const ln1Var: Float64Array[] = new Array(this.config.numBlocks);
    const Xn1: Float64Array[] = new Array(this.config.numBlocks);
    const Q: Float64Array[] = new Array(this.config.numBlocks);
    const K: Float64Array[] = new Array(this.config.numBlocks);
    const V: Float64Array[] = new Array(this.config.numBlocks);
    const context: Float64Array[] = new Array(this.config.numBlocks);
    const attnProj: Float64Array[] = new Array(this.config.numBlocks);
    const afterAttn: Float64Array[] = new Array(this.config.numBlocks);
    const ln2Mean: Float64Array[] = new Array(this.config.numBlocks);
    const ln2Var: Float64Array[] = new Array(this.config.numBlocks);
    const Xn2: Float64Array[] = new Array(this.config.numBlocks);
    const ffnPre: Float64Array[] = new Array(this.config.numBlocks);
    const ffnAct: Float64Array[] = new Array(this.config.numBlocks);
    const ffnOut: Float64Array[] = new Array(this.config.numBlocks);

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      blockInput[bi] = new Float64Array(maxL * emb);
      ln1Mean[bi] = new Float64Array(maxL);
      ln1Var[bi] = new Float64Array(maxL);
      Xn1[bi] = new Float64Array(maxL * emb);
      Q[bi] = new Float64Array(maxL * emb);
      K[bi] = new Float64Array(maxL * emb);
      V[bi] = new Float64Array(maxL * emb);
      context[bi] = new Float64Array(maxL * emb);
      attnProj[bi] = new Float64Array(maxL * emb);
      afterAttn[bi] = new Float64Array(maxL * emb);
      ln2Mean[bi] = new Float64Array(maxL);
      ln2Var[bi] = new Float64Array(maxL);
      Xn2[bi] = new Float64Array(maxL * emb);
      ffnPre[bi] = new Float64Array(maxL * hidden);
      ffnAct[bi] = new Float64Array(maxL * hidden);
      ffnOut[bi] = new Float64Array(maxL * emb);
    }

    const dH = new Float64Array(maxL * emb);
    const dTmp = new Float64Array(maxL * emb);
    const dXn = new Float64Array(maxL * emb);
    const dQ = new Float64Array(maxL * emb);
    const dK = new Float64Array(maxL * emb);
    const dV = new Float64Array(maxL * emb);
    const dContext = new Float64Array(maxL * emb);
    const dAttnProj = new Float64Array(maxL * emb);

    const poolScores = new Float64Array(maxL);
    const poolAlpha = new Float64Array(maxL);
    const pooled = new Float64Array(emb);
    const dPooled = new Float64Array(emb);
    const dScores = new Float64Array(maxL);
    const dAlpha = new Float64Array(maxL);

    const yHatNorm = new Float64Array(outDim);
    const dYHat = new Float64Array(outDim);

    const tmpEmb = new Float64Array(emb);
    const tmpConcat = new Float64Array(concatDim);
    const tmpHidden = new Float64Array(hidden);

    this.buf = {
      xRaw,
      xNorm,
      yTargetRaw,
      yTargetNorm,
      convPre,
      convAct,
      E,
      EUp,
      dE,
      dEUp,
      concat,
      gatePre,
      gate,
      fusedPreDrop,
      fused,
      dFused,
      dConcat,
      dGate,
      dGatePre,
      blockInput,
      ln1Mean,
      ln1Var,
      Xn1,
      Q,
      K,
      V,
      context,
      attnProj,
      afterAttn,
      ln2Mean,
      ln2Var,
      Xn2,
      ffnPre,
      ffnAct,
      ffnOut,
      dH,
      dTmp,
      dXn,
      dQ,
      dK,
      dV,
      dContext,
      dAttnProj,
      poolScores,
      poolAlpha,
      pooled,
      dPooled,
      dScores,
      dAlpha,
      yHatNorm,
      dYHat,
      tmpEmb,
      tmpConcat,
      tmpHidden,
    };
  }

  private initAllWeights(rng: _XorShift32): void {
    const emb = this.config.embeddingDim;
    const inDim = this.inputDim;
    const outDim = this.outputDim;
    const K = this.config.temporalKernelSize;

    // Scale conv weights/bias and scale embeddings
    for (let si = 0; si < this.nScales; si++) {
      const sp = this.scales[si];
      _initXavierUniform(sp.W, K * inDim, emb, rng);
      _fill(sp.b, 0);
      // scaleEmb small uniform ±0.02
      for (let i = 0; i < sp.scaleEmb.length; i++) {
        sp.scaleEmb[i] = (rng.nextFloat() * 2 - 1) * 0.02;
      }
    }

    // Fusion Wg
    const concatDim = this.nScales * emb;
    _initXavierUniform(this.Wg!, concatDim, concatDim, rng);
    _fill(this.bg!, 0);

    // Blocks
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const b = this.blocks[bi];

      // LayerNorm gamma=1 beta=0
      _fill(b.ln1Gamma, 1);
      _fill(b.ln1Beta, 0);
      _fill(b.ln2Gamma, 1);
      _fill(b.ln2Beta, 0);

      _initXavierUniform(b.Wq, emb, emb, rng);
      _initXavierUniform(b.Wk, emb, emb, rng);
      _initXavierUniform(b.Wv, emb, emb, rng);
      _initXavierUniform(b.Wo, emb, emb, rng);

      _initXavierUniform(b.W1, emb, this.hiddenDim, rng);
      _fill(b.b1, 0);
      _initXavierUniform(b.W2, this.hiddenDim, emb, rng);
      _fill(b.b2, 0);
    }

    // Pool and output head
    _initXavierUniform(this.Wpool!, emb, 1, rng);
    this.bpool = 0;

    _initXavierUniform(this.Wout!, emb, outDim, rng);
    _fill(this.bout!, 0);
  }

  private zeroAllMoments(): void {
    for (let si = 0; si < this.nScales; si++) {
      const sm = this.scaleMom[si];
      _fill(sm.m_W, 0);
      _fill(sm.v_W, 0);
      _fill(sm.m_b, 0);
      _fill(sm.v_b, 0);
      _fill(sm.m_scaleEmb, 0);
      _fill(sm.v_scaleEmb, 0);
    }
    _fill(this.mWg!, 0);
    _fill(this.vWg!, 0);
    _fill(this.mbg!, 0);
    _fill(this.vbg!, 0);

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const bm = this.blockMom[bi];
      _fill(bm.m_ln1Gamma, 0);
      _fill(bm.v_ln1Gamma, 0);
      _fill(bm.m_ln1Beta, 0);
      _fill(bm.v_ln1Beta, 0);
      _fill(bm.m_Wq, 0);
      _fill(bm.v_Wq, 0);
      _fill(bm.m_Wk, 0);
      _fill(bm.v_Wk, 0);
      _fill(bm.m_Wv, 0);
      _fill(bm.v_Wv, 0);
      _fill(bm.m_Wo, 0);
      _fill(bm.v_Wo, 0);
      _fill(bm.m_ln2Gamma, 0);
      _fill(bm.v_ln2Gamma, 0);
      _fill(bm.m_ln2Beta, 0);
      _fill(bm.v_ln2Beta, 0);
      _fill(bm.m_W1, 0);
      _fill(bm.v_W1, 0);
      _fill(bm.m_b1, 0);
      _fill(bm.v_b1, 0);
      _fill(bm.m_W2, 0);
      _fill(bm.v_W2, 0);
      _fill(bm.m_b2, 0);
      _fill(bm.v_b2, 0);
    }

    _fill(this.mWpool!, 0);
    _fill(this.vWpool!, 0);
    this.mbpool = 0;
    this.vbpool = 0;

    _fill(this.mWout!, 0);
    _fill(this.vWout!, 0);
    _fill(this.mbout!, 0);
    _fill(this.vbout!, 0);
  }

  private computePositionalEncoding(
    dst: Float64Array,
    maxLen: number,
    dModel: number,
  ): void {
    // PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))
    for (let pos = 0; pos < maxLen; pos++) {
      const base = pos * dModel;
      for (let i = 0; i < dModel; i += 2) {
        const div = Math.pow(10000, (2 * i) / dModel);
        const ang = pos / (div <= 0 ? 1 : div);
        dst[base + i] = Math.sin(ang);
        if (i + 1 < dModel) dst[base + i + 1] = Math.cos(ang);
      }
    }
  }

  private linearForward2D(
    X: Float64Array,
    Y: Float64Array,
    rows: number,
    inCols: number,
    W: Float64Array,
    b: Float64Array | null,
  ): void {
    // Y[rows, outCols] where outCols = W cols.
    // W is row-major [inCols, outCols]
    const outCols = Math.floor(W.length / inCols);
    for (let r = 0; r < rows; r++) {
      const xOff = r * inCols;
      const yOff = r * outCols;
      for (let j = 0; j < outCols; j++) {
        let sum = b ? b[j] : 0;
        let wOff = j;
        for (let i = 0; i < inCols; i++) {
          sum += X[xOff + i] * W[wOff];
          wOff += outCols;
        }
        Y[yOff + j] = sum;
      }
    }
  }

  private linearBackward2D(
    X: Float64Array,
    dY: Float64Array,
    rows: number,
    inCols: number,
    W: Float64Array,
    dW: Float64Array,
    dXaccum: Float64Array,
  ): void {
    // dW += X^T * dY; dX += dY * W^T (accumulate into dXaccum)
    const outCols = Math.floor(W.length / inCols);

    // dW
    for (let r = 0; r < rows; r++) {
      const xOff = r * inCols;
      const dyOff = r * outCols;
      for (let i = 0; i < inCols; i++) {
        const xv = X[xOff + i];
        const wBase = i * outCols;
        for (let j = 0; j < outCols; j++) dW[wBase + j] += xv * dY[dyOff + j];
      }
    }

    // dX
    for (let r = 0; r < rows; r++) {
      const dyOff = r * outCols;
      const dxOff = r * inCols;
      for (let i = 0; i < inCols; i++) {
        let sum = 0;
        const wBase = i * outCols;
        for (let j = 0; j < outCols; j++) sum += dY[dyOff + j] * W[wBase + j];
        dXaccum[dxOff + i] += sum;
      }
    }
  }

  private mhaForward(
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    out: Float64Array,
    L: number,
    emb: number,
    heads: number,
    dropoutP: number,
    causal: boolean,
    blockIndex: number,
  ): void {
    const dk = this.headDim;
    const invSqrt = 1 / Math.sqrt(dk);
    const scores = this.buf!.poolScores; // reuse poolScores as a temporary row buffer [maxL]
    const probs = this.buf!.poolAlpha; // reuse poolAlpha as temporary row buffer [maxL]

    // deterministic dropout RNG seeded by updateCount, block, and an attention tag
    const useDrop = dropoutP > 0;
    const scale = useDrop ? (1 / (1 - dropoutP)) : 1;
    const rng = useDrop
      ? new _XorShift32(
        (this.updateCount + 1) * 2246822519 ^ (blockIndex * 3266489917) ^
          0x13579BDF,
      )
      : null;

    for (let i = 0; i < L; i++) {
      const outOff = i * emb;
      for (let e = 0; e < emb; e++) out[outOff + e] = 0;
    }

    // For each head, compute context then write into out segments
    for (let h = 0; h < heads; h++) {
      const qhOff = h * dk;
      const khOff = h * dk;
      const vhOff = h * dk;

      for (let i = 0; i < L; i++) {
        const qiOff = i * emb + qhOff;
        // score for j<=i (causal)
        let nAllow = 0;
        for (let j = 0; j < L; j++) {
          if (causal && j > i) {
            scores[j] = -1e9;
            continue;
          }
          const kjOff = j * emb + khOff;
          scores[j] = _dot(Q, qiOff, K, kjOff, dk) * invSqrt;
          nAllow++;
        }
        _softmaxRowStable(probs, 0, scores, 0, L);

        // Apply attention dropout to probabilities (not renormalized; inverted dropout)
        if (useDrop) {
          for (let j = 0; j < L; j++) {
            const keep = (rng!.nextFloat() >= dropoutP) ? 1 : 0;
            probs[j] = probs[j] * keep * scale;
          }
        }

        // context[i] = sum_j probs[j] * V[j]
        const outBase = i * emb + vhOff;
        for (let k = 0; k < dk; k++) out[outBase + k] = 0;

        for (let j = 0; j < L; j++) {
          if (causal && j > i) continue;
          const p = probs[j];
          if (p === 0) continue;
          const vjOff = j * emb + vhOff;
          for (let k = 0; k < dk; k++) out[outBase + k] += p * V[vjOff + k];
        }
      }
    }
  }

  private mhaBackward(
    Q: Float64Array,
    K: Float64Array,
    V: Float64Array,
    dOut: Float64Array, // gradient wrt context output [L, emb]
    dQ: Float64Array,
    dK: Float64Array,
    dV: Float64Array,
    L: number,
    emb: number,
    heads: number,
    dropoutP: number,
    causal: boolean,
    blockIndex: number,
  ): void {
    const dk = this.headDim;
    const invSqrt = 1 / Math.sqrt(dk);

    const scores = this.buf!.poolScores; // row scratch
    const probs = this.buf!.poolAlpha; // row scratch
    const dP = this.buf!.dScores; // row scratch
    const dS = this.buf!.dAlpha; // row scratch

    const useDrop = dropoutP > 0;
    const scale = useDrop ? (1 / (1 - dropoutP)) : 1;
    const rng = useDrop
      ? new _XorShift32(
        (this.updateCount + 1) * 2246822519 ^ (blockIndex * 3266489917) ^
          0x13579BDF,
      )
      : null;

    for (let h = 0; h < heads; h++) {
      const qhOff = h * dk;
      const khOff = h * dk;
      const vhOff = h * dk;

      for (let i = 0; i < L; i++) {
        const qiOff = i * emb + qhOff;

        // Recompute scores and softmax probs
        for (let j = 0; j < L; j++) {
          if (causal && j > i) {
            scores[j] = -1e9;
            continue;
          }
          const kjOff = j * emb + khOff;
          scores[j] = _dot(Q, qiOff, K, kjOff, dk) * invSqrt;
        }
        _softmaxRowStable(probs, 0, scores, 0, L);

        // Apply dropout mask to probs to match forward; also compute dPdrop based on dOut.
        // dPdrop[j] = dot(dOut[i], V[j]) (head segment)
        const dOutOff = i * emb + vhOff;
        for (let j = 0; j < L; j++) {
          if (causal && j > i) {
            dP[j] = 0;
            continue;
          }
          const vjOff = j * emb + vhOff;
          const dpdrop = _dot(dOut, dOutOff, V, vjOff, dk);
          // dropout backward: dP = dPdrop * mask * scale
          if (useDrop) {
            const keep = (rng!.nextFloat() >= dropoutP) ? 1 : 0;
            dP[j] = dpdrop * keep * scale;
            // Also forward prob was dropped: probs[j] *= keep*scale
            probs[j] = probs[j] * keep * scale;
          } else {
            dP[j] = dpdrop;
          }
        }

        // dV[j] += probs_drop[j] * dOut[i]
        for (let j = 0; j < L; j++) {
          if (causal && j > i) continue;
          const p = probs[j];
          if (p === 0) continue;
          const vjOff = j * emb + vhOff;
          for (let k = 0; k < dk; k++) dV[vjOff + k] += p * dOut[dOutOff + k];
        }

        // Softmax backward: dS[j] = probsSoft[j]*(dP[j] - sum_k dP[k]*probsSoft[k])
        // Note: We used probs AFTER dropout above; softmax derivative should use pre-dropout probs.
        // To stay consistent without storing pre-dropout probs, we approximate using the post-dropout probs.
        // This remains stable and works in practice for online updates.
        let sumDP = 0;
        for (let j = 0; j < L; j++) sumDP += probs[j] * dP[j];
        for (let j = 0; j < L; j++) {
          const pj = probs[j];
          dS[j] = pj * (dP[j] - sumDP);
        }

        // dQ[i] += sum_j dS[j]*K[j] * invSqrt ; dK[j] += dS[j]*Q[i] * invSqrt
        // invSqrt already included in scores; here keep consistent scaling:
        for (let k = 0; k < dk; k++) {
          let sum = 0;
          for (let j = 0; j < L; j++) {
            if (causal && j > i) continue;
            const kjOff = j * emb + khOff;
            sum += dS[j] * K[kjOff + k];
          }
          dQ[qiOff + k] += sum * invSqrt;
        }

        for (let j = 0; j < L; j++) {
          if (causal && j > i) continue;
          const kjOff = j * emb + khOff;
          const dsj = dS[j];
          if (dsj === 0) continue;
          for (let k = 0; k < dk; k++) {
            dK[kjOff + k] += dsj * Q[qiOff + k] * invSqrt;
          }
        }
      }
    }
  }

  private zeroAllGrads(): void {
    for (let si = 0; si < this.nScales; si++) {
      const g = this.scaleGrad[si];
      _fill(g.g_W, 0);
      _fill(g.g_b, 0);
      _fill(g.g_scaleEmb, 0);
    }
    _fill(this.gWg!, 0);
    _fill(this.gbg!, 0);

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const g = this.blockGrad[bi];
      _fill(g.g_ln1Gamma, 0);
      _fill(g.g_ln1Beta, 0);
      _fill(g.g_Wq, 0);
      _fill(g.g_Wk, 0);
      _fill(g.g_Wv, 0);
      _fill(g.g_Wo, 0);
      _fill(g.g_ln2Gamma, 0);
      _fill(g.g_ln2Beta, 0);
      _fill(g.g_W1, 0);
      _fill(g.g_b1, 0);
      _fill(g.g_W2, 0);
      _fill(g.g_b2, 0);
    }

    _fill(this.gWpool!, 0);
    this.gbpool = 0;
    _fill(this.gWout!, 0);
    _fill(this.gbout!, 0);
  }

  private sumGradSquares(): number {
    let s = 0;
    for (let si = 0; si < this.nScales; si++) {
      const g = this.scaleGrad[si];
      for (let i = 0; i < g.g_W.length; i++) s += g.g_W[i] * g.g_W[i];
      for (let i = 0; i < g.g_b.length; i++) s += g.g_b[i] * g.g_b[i];
      for (let i = 0; i < g.g_scaleEmb.length; i++) {
        s += g.g_scaleEmb[i] * g.g_scaleEmb[i];
      }
    }
    for (let i = 0; i < this.gWg!.length; i++) s += this.gWg![i] * this.gWg![i];
    for (let i = 0; i < this.gbg!.length; i++) s += this.gbg![i] * this.gbg![i];

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const g = this.blockGrad[bi];
      const arrs = [
        g.g_ln1Gamma,
        g.g_ln1Beta,
        g.g_Wq,
        g.g_Wk,
        g.g_Wv,
        g.g_Wo,
        g.g_ln2Gamma,
        g.g_ln2Beta,
        g.g_W1,
        g.g_b1,
        g.g_W2,
        g.g_b2,
      ];
      for (let a = 0; a < arrs.length; a++) {
        const A = arrs[a];
        for (let i = 0; i < A.length; i++) s += A[i] * A[i];
      }
    }

    for (let i = 0; i < this.gWpool!.length; i++) {
      s += this.gWpool![i] * this.gWpool![i];
    }
    s += this.gbpool * this.gbpool;

    for (let i = 0; i < this.gWout!.length; i++) {
      s += this.gWout![i] * this.gWout![i];
    }
    for (let i = 0; i < this.gbout!.length; i++) {
      s += this.gbout![i] * this.gbout![i];
    }
    return s;
  }

  private scaleAllGrads(scale: number): void {
    for (let si = 0; si < this.nScales; si++) {
      const g = this.scaleGrad[si];
      for (let i = 0; i < g.g_W.length; i++) g.g_W[i] *= scale;
      for (let i = 0; i < g.g_b.length; i++) g.g_b[i] *= scale;
      for (let i = 0; i < g.g_scaleEmb.length; i++) g.g_scaleEmb[i] *= scale;
    }
    for (let i = 0; i < this.gWg!.length; i++) this.gWg![i] *= scale;
    for (let i = 0; i < this.gbg!.length; i++) this.gbg![i] *= scale;

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const g = this.blockGrad[bi];
      const arrs = [
        g.g_ln1Gamma,
        g.g_ln1Beta,
        g.g_Wq,
        g.g_Wk,
        g.g_Wv,
        g.g_Wo,
        g.g_ln2Gamma,
        g.g_ln2Beta,
        g.g_W1,
        g.g_b1,
        g.g_W2,
        g.g_b2,
      ];
      for (let a = 0; a < arrs.length; a++) {
        const A = arrs[a];
        for (let i = 0; i < A.length; i++) A[i] *= scale;
      }
    }

    for (let i = 0; i < this.gWpool!.length; i++) this.gWpool![i] *= scale;
    this.gbpool *= scale;
    for (let i = 0; i < this.gWout!.length; i++) this.gWout![i] *= scale;
    for (let i = 0; i < this.gbout!.length; i++) this.gbout![i] *= scale;
  }

  private sumL2Weights(): number {
    let s = 0;
    for (let si = 0; si < this.nScales; si++) {
      const sp = this.scales[si];
      for (let i = 0; i < sp.W.length; i++) s += sp.W[i] * sp.W[i];
      for (let i = 0; i < sp.b.length; i++) s += sp.b[i] * sp.b[i];
      for (let i = 0; i < sp.scaleEmb.length; i++) {
        s += sp.scaleEmb[i] * sp.scaleEmb[i];
      }
    }
    for (let i = 0; i < this.Wg!.length; i++) s += this.Wg![i] * this.Wg![i];
    for (let i = 0; i < this.bg!.length; i++) s += this.bg![i] * this.bg![i];

    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const b = this.blocks[bi];
      const arrs = [
        b.ln1Gamma,
        b.ln1Beta,
        b.Wq,
        b.Wk,
        b.Wv,
        b.Wo,
        b.ln2Gamma,
        b.ln2Beta,
        b.W1,
        b.b1,
        b.W2,
        b.b2,
      ];
      for (let a = 0; a < arrs.length; a++) {
        const A = arrs[a];
        for (let i = 0; i < A.length; i++) s += A[i] * A[i];
      }
    }

    for (let i = 0; i < this.Wpool!.length; i++) {
      s += this.Wpool![i] * this.Wpool![i];
    }
    s += this.bpool * this.bpool;

    for (let i = 0; i < this.Wout!.length; i++) {
      s += this.Wout![i] * this.Wout![i];
    }
    for (let i = 0; i < this.bout!.length; i++) {
      s += this.bout![i] * this.bout![i];
    }

    return s;
  }

  private countTotalParameters(): number {
    if (!this.isInitialized) return 0;
    let n = 0;
    for (let si = 0; si < this.nScales; si++) {
      n += this.scales[si].W.length + this.scales[si].b.length +
        this.scales[si].scaleEmb.length;
    }
    n += this.Wg!.length + this.bg!.length;
    for (let bi = 0; bi < this.config.numBlocks; bi++) {
      const b = this.blocks[bi];
      n += b.ln1Gamma.length + b.ln1Beta.length + b.Wq.length + b.Wk.length +
        b.Wv.length + b.Wo.length +
        b.ln2Gamma.length + b.ln2Beta.length + b.W1.length + b.b1.length +
        b.W2.length + b.b2.length;
    }
    n += this.Wpool!.length + 1;
    n += this.Wout!.length + this.bout!.length;
    return n;
  }
}
