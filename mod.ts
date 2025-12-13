/**
 * Fusion Temporal Transformer for Multivariate Regression
 *
 * Architecture:
 * 1. Multi-scale temporal convolution with GELU activation
 * 2. Scale-specific embeddings + sinusoidal positional encoding
 * 3. Cross-scale gated fusion
 * 4. Transformer blocks with causal multi-head self-attention
 * 5. Attention-weighted temporal pooling
 * 6. Linear output head
 *
 * Weight initialization: Xavier uniform, limit = sqrt(6/(fanIn+fanOut))
 */

type FitResult = {
  loss: number;
  gradientNorm: number;
  effectiveLearningRate: number;
  isOutlier: boolean;
  converged: boolean;
  sampleIndex: number;
  driftDetected: boolean;
};

type SinglePrediction = {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
};

type PredictionResult = {
  predictions: SinglePrediction[];
  accuracy: number;
  sampleCount: number;
  isModelReady: boolean;
};

type WeightInfo = {
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

type NormalizationStats = {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
  count: number;
};

type ModelSummary = {
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

interface Config {
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

const DEFAULT_CONFIG: Config = {
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

/**
 * @example
 * ```typescript
 * const model = new FusionTemporalTransformerRegression({ numBlocks: 2 });
 * const result = model.fitOnline({ xCoordinates: [[1,2],[3,4]], yCoordinates: [[5],[6]] });
 * const pred = model.predict(3);
 * ```
 */
export class FusionTemporalTransformerRegression {
  private config: Config;
  private inputDim: number = 0;
  private outputDim: number = 0;
  private seqLen: number = 0;
  private isInitialized: boolean = false;
  private sampleCount: number = 0;
  private updateCount: number = 0;
  private driftCount: number = 0;
  private runningLoss: number = 0;
  private converged: boolean = false;
  private nScales: number;
  private headDim: number;
  private ffnDim: number;
  private totalParams: number = 0;

  // Welford normalization stats
  private inputMean!: Float64Array;
  private inputM2!: Float64Array;
  private outputMean!: Float64Array;
  private outputM2!: Float64Array;
  private residualM2!: Float64Array;

  // Positional encoding cache
  private posEnc!: Float64Array;

  // Cached window for prediction
  private cachedWindow!: Float64Array;
  private cachedWindowLen: number = 0;

  // ADWIN drift detection
  private readonly ADWIN_CAP = 256;
  private readonly ADWIN_MIN = 32;
  private adwinBuf!: Float64Array;
  private adwinHead: number = 0;
  private adwinSize: number = 0;

  // === WEIGHTS ===
  // Temporal conv per scale: W[K*inputDim*embDim], b[embDim]
  private convW!: Float64Array[];
  private convB!: Float64Array[];
  // Scale embeddings per scale: [embDim]
  private scaleEmb!: Float64Array[];
  // Fusion: Wg[(nScales*embDim)^2], bg[nScales*embDim]
  private fusionW!: Float64Array;
  private fusionB!: Float64Array;
  // Attention per block: Wq,Wk,Wv,Wo each [embDim^2]
  private attnWq!: Float64Array[];
  private attnWk!: Float64Array[];
  private attnWv!: Float64Array[];
  private attnWo!: Float64Array[];
  // FFN per block: W1[embDim*ffnDim], b1[ffnDim], W2[ffnDim*embDim], b2[embDim]
  private ffnW1!: Float64Array[];
  private ffnB1!: Float64Array[];
  private ffnW2!: Float64Array[];
  private ffnB2!: Float64Array[];
  // LayerNorm per block: gamma1,beta1,gamma2,beta2 each [embDim]
  private lnG1!: Float64Array[];
  private lnB1!: Float64Array[];
  private lnG2!: Float64Array[];
  private lnB2!: Float64Array[];
  // Pooling: Wpool[embDim], bpool[1]
  private poolW!: Float64Array;
  private poolB!: Float64Array;
  // Output: Wout[embDim*outDim], bout[outDim]
  private outW!: Float64Array;
  private outB!: Float64Array;

  // === ADAM MOMENTS ===
  private mConvW!: Float64Array[];
  private mConvB!: Float64Array[];
  private mScaleEmb!: Float64Array[];
  private mFusionW!: Float64Array;
  private mFusionB!: Float64Array;
  private mAttnWq!: Float64Array[];
  private mAttnWk!: Float64Array[];
  private mAttnWv!: Float64Array[];
  private mAttnWo!: Float64Array[];
  private mFfnW1!: Float64Array[];
  private mFfnB1!: Float64Array[];
  private mFfnW2!: Float64Array[];
  private mFfnB2!: Float64Array[];
  private mLnG1!: Float64Array[];
  private mLnB1!: Float64Array[];
  private mLnG2!: Float64Array[];
  private mLnB2!: Float64Array[];
  private mPoolW!: Float64Array;
  private mPoolB!: Float64Array;
  private mOutW!: Float64Array;
  private mOutB!: Float64Array;

  private vConvW!: Float64Array[];
  private vConvB!: Float64Array[];
  private vScaleEmb!: Float64Array[];
  private vFusionW!: Float64Array;
  private vFusionB!: Float64Array;
  private vAttnWq!: Float64Array[];
  private vAttnWk!: Float64Array[];
  private vAttnWv!: Float64Array[];
  private vAttnWo!: Float64Array[];
  private vFfnW1!: Float64Array[];
  private vFfnB1!: Float64Array[];
  private vFfnW2!: Float64Array[];
  private vFfnB2!: Float64Array[];
  private vLnG1!: Float64Array[];
  private vLnB1!: Float64Array[];
  private vLnG2!: Float64Array[];
  private vLnB2!: Float64Array[];
  private vPoolW!: Float64Array;
  private vPoolB!: Float64Array;
  private vOutW!: Float64Array;
  private vOutB!: Float64Array;

  // === GRADIENTS (reused each step) ===
  private gConvW!: Float64Array[];
  private gConvB!: Float64Array[];
  private gScaleEmb!: Float64Array[];
  private gFusionW!: Float64Array;
  private gFusionB!: Float64Array;
  private gAttnWq!: Float64Array[];
  private gAttnWk!: Float64Array[];
  private gAttnWv!: Float64Array[];
  private gAttnWo!: Float64Array[];
  private gFfnW1!: Float64Array[];
  private gFfnB1!: Float64Array[];
  private gFfnW2!: Float64Array[];
  private gFfnB2!: Float64Array[];
  private gLnG1!: Float64Array[];
  private gLnB1!: Float64Array[];
  private gLnG2!: Float64Array[];
  private gLnB2!: Float64Array[];
  private gPoolW!: Float64Array;
  private gPoolB!: Float64Array;
  private gOutW!: Float64Array;
  private gOutB!: Float64Array;

  // === FORWARD CACHE BUFFERS ===
  // Normalized input: [seqLen * inputDim]
  private xNorm!: Float64Array;
  // Conv outputs per scale: [Ls * embDim] where Ls = ceil(seqLen/scale)
  private convOut!: Float64Array[];
  private convOutLens!: number[];
  // Pre-GELU conv for backward
  private convPreGelu!: Float64Array[];
  // Scale embeddings added: [Ls * embDim]
  private scaleEmbOut!: Float64Array[];
  // Upsampled scale outputs: [seqLen * embDim] per scale
  private upsampled!: Float64Array[];
  // Concatenated for fusion: [seqLen * (nScales * embDim)]
  private fusionConcat!: Float64Array;
  // Pre-sigmoid fusion gate
  private fusionPreSig!: Float64Array;
  // Gate values: [seqLen * (nScales * embDim)]
  private fusionGate!: Float64Array;
  // Fused output: [seqLen * embDim]
  private fusedOut!: Float64Array;
  // Transformer block caches
  private blockInputs!: Float64Array[];
  private ln1Out!: Float64Array[];
  private ln1Mean!: Float64Array[];
  private ln1Var!: Float64Array[];
  private attnOut!: Float64Array[];
  private attnResid!: Float64Array[];
  private ln2Out!: Float64Array[];
  private ln2Mean!: Float64Array[];
  private ln2Var!: Float64Array[];
  private ffnOut!: Float64Array[];
  // MHA caches per block
  private mhaQ!: Float64Array[];
  private mhaK!: Float64Array[];
  private mhaV!: Float64Array[];
  private mhaScores!: Float64Array[];
  private mhaProbs!: Float64Array[];
  private mhaHeadOut!: Float64Array[];
  // FFN caches
  private ffnHid!: Float64Array[];
  private ffnPreGelu!: Float64Array[];
  // Pooling cache
  private poolScores!: Float64Array;
  private poolAlpha!: Float64Array;
  private poolOut!: Float64Array;
  // Output
  private yHat!: Float64Array;
  // Backward buffers
  private dOut!: Float64Array;
  private dPool!: Float64Array;
  private dBlockOut!: Float64Array;
  private dLn2Out!: Float64Array;
  private dFfnOut!: Float64Array;
  private dFfnHid!: Float64Array;
  private dAttnResid!: Float64Array;
  private dLn1Out!: Float64Array;
  private dAttnOut!: Float64Array;
  private dMhaConcat!: Float64Array;
  private dMhaV!: Float64Array;
  private dMhaK!: Float64Array;
  private dMhaQ!: Float64Array;
  private dFused!: Float64Array;
  private dFusionGate!: Float64Array;
  private dFusionConcat!: Float64Array;
  private dUpsampled!: Float64Array[];
  private dScaleEmbOut!: Float64Array[];
  private dConvOut!: Float64Array[];
  private dXNorm!: Float64Array;
  // RNG state
  private rngState: number = 1;

  /**
   * @param config - Partial configuration, merged with defaults
   */
  constructor(config?: Partial<Config>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.nScales = this.config.temporalScales.length;
    this.headDim = Math.floor(this.config.embeddingDim / this.config.numHeads);
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;
    if (this.config.embeddingDim % this.config.numHeads !== 0) {
      throw new Error("embeddingDim must be divisible by numHeads");
    }
    this.adwinBuf = new Float64Array(this.ADWIN_CAP);
  }

  /**
   * Deterministic xorshift32 RNG
   * @returns number in [0, 1)
   */
  private xorshift(): number {
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return (this.rngState >>> 0) / 4294967296;
  }

  /**
   * Xavier uniform initialization
   * @param fanIn - Input features
   * @param fanOut - Output features
   * @returns Initialized weight
   */
  private xavierUniform(fanIn: number, fanOut: number): number {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return (this.xorshift() * 2 - 1) * limit;
  }

  /**
   * Initialize all model parameters and buffers
   */
  private initialize(): void {
    const { numBlocks, embeddingDim, maxSequenceLength, temporalKernelSize } =
      this.config;
    const K = temporalKernelSize;
    const E = embeddingDim;
    const F = this.ffnDim;
    const nS = this.nScales;
    const L = this.seqLen;
    const inD = this.inputDim;
    const outD = this.outputDim;

    // Normalization stats
    this.inputMean = new Float64Array(inD);
    this.inputM2 = new Float64Array(inD);
    this.outputMean = new Float64Array(outD);
    this.outputM2 = new Float64Array(outD);
    this.residualM2 = new Float64Array(outD);

    // Positional encoding: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
    this.posEnc = new Float64Array(maxSequenceLength * E);
    for (let p = 0; p < maxSequenceLength; p++) {
      for (let i = 0; i < E; i++) {
        const angle = p / Math.pow(10000, (2 * Math.floor(i / 2)) / E);
        this.posEnc[p * E + i] = i % 2 === 0
          ? Math.sin(angle)
          : Math.cos(angle);
      }
    }

    // Cached window
    this.cachedWindow = new Float64Array(maxSequenceLength * inD);

    // ADWIN buffer already created in constructor

    // === WEIGHTS INIT ===
    this.rngState = 42;
    this.convW = [];
    this.convB = [];
    this.mConvW = [];
    this.mConvB = [];
    this.vConvW = [];
    this.vConvB = [];
    this.gConvW = [];
    this.gConvB = [];
    this.scaleEmb = [];
    this.mScaleEmb = [];
    this.vScaleEmb = [];
    this.gScaleEmb = [];
    this.convOut = [];
    this.convOutLens = [];
    this.convPreGelu = [];
    this.scaleEmbOut = [];
    this.upsampled = [];
    this.dUpsampled = [];
    this.dScaleEmbOut = [];
    this.dConvOut = [];

    for (let s = 0; s < nS; s++) {
      const scale = this.config.temporalScales[s];
      const Ls = Math.ceil(L / scale);
      this.convOutLens.push(Ls);

      // Conv weights: [K * inD * E]
      const wSize = K * inD * E;
      const cw = new Float64Array(wSize);
      for (let i = 0; i < wSize; i++) cw[i] = this.xavierUniform(K * inD, E);
      this.convW.push(cw);

      const cb = new Float64Array(E);
      this.convB.push(cb);

      this.mConvW.push(new Float64Array(wSize));
      this.vConvW.push(new Float64Array(wSize));
      this.gConvW.push(new Float64Array(wSize));
      this.mConvB.push(new Float64Array(E));
      this.vConvB.push(new Float64Array(E));
      this.gConvB.push(new Float64Array(E));

      // Scale embeddings
      const se = new Float64Array(E);
      for (let i = 0; i < E; i++) se[i] = (this.xorshift() * 2 - 1) * 0.02;
      this.scaleEmb.push(se);
      this.mScaleEmb.push(new Float64Array(E));
      this.vScaleEmb.push(new Float64Array(E));
      this.gScaleEmb.push(new Float64Array(E));

      this.convOut.push(new Float64Array(Ls * E));
      this.convPreGelu.push(new Float64Array(Ls * E));
      this.scaleEmbOut.push(new Float64Array(Ls * E));
      this.upsampled.push(new Float64Array(L * E));
      this.dUpsampled.push(new Float64Array(L * E));
      this.dScaleEmbOut.push(new Float64Array(Ls * E));
      this.dConvOut.push(new Float64Array(Ls * E));
    }

    // Fusion
    const fusionDim = nS * E;
    const fwSize = fusionDim * fusionDim;
    this.fusionW = new Float64Array(fwSize);
    for (let i = 0; i < fwSize; i++) {
      this.fusionW[i] = this.xavierUniform(fusionDim, fusionDim);
    }
    this.fusionB = new Float64Array(fusionDim);
    this.mFusionW = new Float64Array(fwSize);
    this.vFusionW = new Float64Array(fwSize);
    this.gFusionW = new Float64Array(fwSize);
    this.mFusionB = new Float64Array(fusionDim);
    this.vFusionB = new Float64Array(fusionDim);
    this.gFusionB = new Float64Array(fusionDim);

    this.fusionConcat = new Float64Array(L * fusionDim);
    this.fusionPreSig = new Float64Array(L * fusionDim);
    this.fusionGate = new Float64Array(L * fusionDim);
    this.fusedOut = new Float64Array(L * E);
    this.dFused = new Float64Array(L * E);
    this.dFusionGate = new Float64Array(L * fusionDim);
    this.dFusionConcat = new Float64Array(L * fusionDim);

    // Attention per block
    const aSize = E * E;
    this.attnWq = [];
    this.attnWk = [];
    this.attnWv = [];
    this.attnWo = [];
    this.mAttnWq = [];
    this.mAttnWk = [];
    this.mAttnWv = [];
    this.mAttnWo = [];
    this.vAttnWq = [];
    this.vAttnWk = [];
    this.vAttnWv = [];
    this.vAttnWo = [];
    this.gAttnWq = [];
    this.gAttnWk = [];
    this.gAttnWv = [];
    this.gAttnWo = [];
    this.ffnW1 = [];
    this.ffnB1 = [];
    this.ffnW2 = [];
    this.ffnB2 = [];
    this.mFfnW1 = [];
    this.mFfnB1 = [];
    this.mFfnW2 = [];
    this.mFfnB2 = [];
    this.vFfnW1 = [];
    this.vFfnB1 = [];
    this.vFfnW2 = [];
    this.vFfnB2 = [];
    this.gFfnW1 = [];
    this.gFfnB1 = [];
    this.gFfnW2 = [];
    this.gFfnB2 = [];
    this.lnG1 = [];
    this.lnB1 = [];
    this.lnG2 = [];
    this.lnB2 = [];
    this.mLnG1 = [];
    this.mLnB1 = [];
    this.mLnG2 = [];
    this.mLnB2 = [];
    this.vLnG1 = [];
    this.vLnB1 = [];
    this.vLnG2 = [];
    this.vLnB2 = [];
    this.gLnG1 = [];
    this.gLnB1 = [];
    this.gLnG2 = [];
    this.gLnB2 = [];

    this.blockInputs = [];
    this.ln1Out = [];
    this.ln1Mean = [];
    this.ln1Var = [];
    this.attnOut = [];
    this.attnResid = [];
    this.ln2Out = [];
    this.ln2Mean = [];
    this.ln2Var = [];
    this.ffnOut = [];
    this.mhaQ = [];
    this.mhaK = [];
    this.mhaV = [];
    this.mhaScores = [];
    this.mhaProbs = [];
    this.mhaHeadOut = [];
    this.ffnHid = [];
    this.ffnPreGelu = [];

    for (let b = 0; b < numBlocks; b++) {
      // Attention weights
      const wq = new Float64Array(aSize);
      const wk = new Float64Array(aSize);
      const wv = new Float64Array(aSize);
      const wo = new Float64Array(aSize);
      for (let i = 0; i < aSize; i++) {
        wq[i] = this.xavierUniform(E, E);
        wk[i] = this.xavierUniform(E, E);
        wv[i] = this.xavierUniform(E, E);
        wo[i] = this.xavierUniform(E, E);
      }
      this.attnWq.push(wq);
      this.attnWk.push(wk);
      this.attnWv.push(wv);
      this.attnWo.push(wo);
      this.mAttnWq.push(new Float64Array(aSize));
      this.mAttnWk.push(new Float64Array(aSize));
      this.mAttnWv.push(new Float64Array(aSize));
      this.mAttnWo.push(new Float64Array(aSize));
      this.vAttnWq.push(new Float64Array(aSize));
      this.vAttnWk.push(new Float64Array(aSize));
      this.vAttnWv.push(new Float64Array(aSize));
      this.vAttnWo.push(new Float64Array(aSize));
      this.gAttnWq.push(new Float64Array(aSize));
      this.gAttnWk.push(new Float64Array(aSize));
      this.gAttnWv.push(new Float64Array(aSize));
      this.gAttnWo.push(new Float64Array(aSize));

      // FFN weights
      const w1Size = E * F;
      const w2Size = F * E;
      const w1 = new Float64Array(w1Size);
      const w2 = new Float64Array(w2Size);
      for (let i = 0; i < w1Size; i++) w1[i] = this.xavierUniform(E, F);
      for (let i = 0; i < w2Size; i++) w2[i] = this.xavierUniform(F, E);
      this.ffnW1.push(w1);
      this.ffnB1.push(new Float64Array(F));
      this.ffnW2.push(w2);
      this.ffnB2.push(new Float64Array(E));
      this.mFfnW1.push(new Float64Array(w1Size));
      this.mFfnB1.push(new Float64Array(F));
      this.mFfnW2.push(new Float64Array(w2Size));
      this.mFfnB2.push(new Float64Array(E));
      this.vFfnW1.push(new Float64Array(w1Size));
      this.vFfnB1.push(new Float64Array(F));
      this.vFfnW2.push(new Float64Array(w2Size));
      this.vFfnB2.push(new Float64Array(E));
      this.gFfnW1.push(new Float64Array(w1Size));
      this.gFfnB1.push(new Float64Array(F));
      this.gFfnW2.push(new Float64Array(w2Size));
      this.gFfnB2.push(new Float64Array(E));

      // LayerNorm
      const g1 = new Float64Array(E);
      const b1 = new Float64Array(E);
      const g2 = new Float64Array(E);
      const b2 = new Float64Array(E);
      for (let i = 0; i < E; i++) {
        g1[i] = 1;
        g2[i] = 1;
      }
      this.lnG1.push(g1);
      this.lnB1.push(b1);
      this.lnG2.push(g2);
      this.lnB2.push(b2);
      this.mLnG1.push(new Float64Array(E));
      this.mLnB1.push(new Float64Array(E));
      this.mLnG2.push(new Float64Array(E));
      this.mLnB2.push(new Float64Array(E));
      this.vLnG1.push(new Float64Array(E));
      this.vLnB1.push(new Float64Array(E));
      this.vLnG2.push(new Float64Array(E));
      this.vLnB2.push(new Float64Array(E));
      this.gLnG1.push(new Float64Array(E));
      this.gLnB1.push(new Float64Array(E));
      this.gLnG2.push(new Float64Array(E));
      this.gLnB2.push(new Float64Array(E));

      // Caches
      this.blockInputs.push(new Float64Array(L * E));
      this.ln1Out.push(new Float64Array(L * E));
      this.ln1Mean.push(new Float64Array(L));
      this.ln1Var.push(new Float64Array(L));
      this.attnOut.push(new Float64Array(L * E));
      this.attnResid.push(new Float64Array(L * E));
      this.ln2Out.push(new Float64Array(L * E));
      this.ln2Mean.push(new Float64Array(L));
      this.ln2Var.push(new Float64Array(L));
      this.ffnOut.push(new Float64Array(L * E));

      this.mhaQ.push(new Float64Array(L * E));
      this.mhaK.push(new Float64Array(L * E));
      this.mhaV.push(new Float64Array(L * E));
      this.mhaScores.push(new Float64Array(this.config.numHeads * L * L));
      this.mhaProbs.push(new Float64Array(this.config.numHeads * L * L));
      this.mhaHeadOut.push(new Float64Array(L * E));
      this.ffnHid.push(new Float64Array(L * F));
      this.ffnPreGelu.push(new Float64Array(L * F));
    }

    // Pooling
    this.poolW = new Float64Array(E);
    for (let i = 0; i < E; i++) this.poolW[i] = this.xavierUniform(E, 1);
    this.poolB = new Float64Array(1);
    this.mPoolW = new Float64Array(E);
    this.vPoolW = new Float64Array(E);
    this.gPoolW = new Float64Array(E);
    this.mPoolB = new Float64Array(1);
    this.vPoolB = new Float64Array(1);
    this.gPoolB = new Float64Array(1);
    this.poolScores = new Float64Array(L);
    this.poolAlpha = new Float64Array(L);
    this.poolOut = new Float64Array(E);

    // Output
    const owSize = E * outD;
    this.outW = new Float64Array(owSize);
    for (let i = 0; i < owSize; i++) this.outW[i] = this.xavierUniform(E, outD);
    this.outB = new Float64Array(outD);
    this.mOutW = new Float64Array(owSize);
    this.vOutW = new Float64Array(owSize);
    this.gOutW = new Float64Array(owSize);
    this.mOutB = new Float64Array(outD);
    this.vOutB = new Float64Array(outD);
    this.gOutB = new Float64Array(outD);

    this.yHat = new Float64Array(outD);
    this.xNorm = new Float64Array(L * inD);

    // Backward buffers
    this.dOut = new Float64Array(outD);
    this.dPool = new Float64Array(E);
    this.dBlockOut = new Float64Array(L * E);
    this.dLn2Out = new Float64Array(L * E);
    this.dFfnOut = new Float64Array(L * E);
    this.dFfnHid = new Float64Array(L * F);
    this.dAttnResid = new Float64Array(L * E);
    this.dLn1Out = new Float64Array(L * E);
    this.dAttnOut = new Float64Array(L * E);
    this.dMhaConcat = new Float64Array(L * E);
    this.dMhaV = new Float64Array(L * E);
    this.dMhaK = new Float64Array(L * E);
    this.dMhaQ = new Float64Array(L * E);
    this.dXNorm = new Float64Array(L * inD);

    this.countTotalParams();
    this.isInitialized = true;
  }

  private countTotalParams(): void {
    const { numBlocks, embeddingDim, temporalKernelSize } = this.config;
    const K = temporalKernelSize;
    const E = embeddingDim;
    const F = this.ffnDim;
    const nS = this.nScales;
    let total = 0;
    // Conv
    total += nS * (K * this.inputDim * E + E);
    // Scale emb
    total += nS * E;
    // Fusion
    total += (nS * E) * (nS * E) + nS * E;
    // Per block
    total += numBlocks * (4 * E * E + E * F + F + F * E + E + 4 * E);
    // Pool
    total += E + 1;
    // Out
    total += E * this.outputDim + this.outputDim;
    this.totalParams = total;
  }

  /**
   * GELU activation: GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))
   */
  private gelu(x: number): number {
    const c = 0.7978845608028654;
    const x3 = x * x * x;
    const inner = c * (x + 0.044715 * x3);
    const t = Math.tanh(Math.max(-20, Math.min(20, inner)));
    return 0.5 * x * (1 + t);
  }

  /**
   * GELU derivative
   */
  private geluDeriv(x: number): number {
    const c = 0.7978845608028654;
    const x2 = x * x;
    const inner = c * (x + 0.044715 * x * x2);
    const t = Math.tanh(Math.max(-20, Math.min(20, inner)));
    const sech2 = 1 - t * t;
    const dinnerDx = c * (1 + 3 * 0.044715 * x2);
    return 0.5 * (1 + t) + 0.5 * x * sech2 * dinnerDx;
  }

  /**
   * Sigmoid: 1/(1+exp(-x))
   */
  private sigmoid(x: number): number {
    if (x >= 0) {
      const ex = Math.exp(-x);
      return 1 / (1 + ex);
    } else {
      const ex = Math.exp(x);
      return ex / (1 + ex);
    }
  }

  /**
   * Sigmoid derivative: s(1-s)
   */
  private sigmoidDeriv(s: number): number {
    return s * (1 - s);
  }

  /**
   * Update Welford statistics
   * mean, M2 arrays, count
   */
  private updateWelford(
    mean: Float64Array,
    m2: Float64Array,
    count: number,
    x: Float64Array | number[],
    start: number,
    len: number,
  ): void {
    const n = count;
    for (let i = 0; i < len; i++) {
      const val = Array.isArray(x) ? x[start + i] : x[start + i];
      const delta = val - mean[i];
      mean[i] += delta / n;
      const delta2 = val - mean[i];
      m2[i] += delta * delta2;
    }
  }

  /**
   * Get std from Welford M2, clamped >= 1e-12
   */
  private getStd(m2: Float64Array, count: number, out: Float64Array): void {
    for (let i = 0; i < m2.length; i++) {
      const variance = count > 1 ? m2[i] / (count - 1) : 0;
      out[i] = Math.max(1e-12, Math.sqrt(Math.max(0, variance)));
    }
  }

  /**
   * Compute learning rate with warmup and cosine decay
   */
  private getLearningRate(): number {
    const { learningRate, warmupSteps, totalSteps } = this.config;
    const step = this.updateCount;
    if (step < warmupSteps) {
      return learningRate * (step / Math.max(1, warmupSteps));
    }
    const progress = (step - warmupSteps) /
      Math.max(1, totalSteps - warmupSteps);
    return learningRate * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)));
  }

  /**
   * Forward pass for temporal convolution with GELU
   * Conv1D: F_s[t,e] = GELU( sum_{k,f} X[(t*s-k),f] * W[k,f,e] + b[e] )
   */
  private forwardConv(scaleIdx: number): void {
    const scale = this.config.temporalScales[scaleIdx];
    const K = this.config.temporalKernelSize;
    const E = this.config.embeddingDim;
    const inD = this.inputDim;
    const L = this.seqLen;
    const Ls = this.convOutLens[scaleIdx];
    const W = this.convW[scaleIdx];
    const b = this.convB[scaleIdx];
    const out = this.convOut[scaleIdx];
    const preGelu = this.convPreGelu[scaleIdx];

    for (let t = 0; t < Ls; t++) {
      const tOrig = t * scale;
      for (let e = 0; e < E; e++) {
        let sum = b[e];
        for (let k = 0; k < K; k++) {
          const tSrc = tOrig - k;
          if (tSrc >= 0 && tSrc < L) {
            for (let f = 0; f < inD; f++) {
              // W[k,f,e] = W[(k * inD + f) * E + e]
              sum += this.xNorm[tSrc * inD + f] * W[(k * inD + f) * E + e];
            }
          }
        }
        preGelu[t * E + e] = sum;
        out[t * E + e] = this.gelu(sum);
      }
    }
  }

  /**
   * Add positional encoding and scale embedding to conv output
   */
  private forwardScaleEmbed(scaleIdx: number): void {
    const E = this.config.embeddingDim;
    const Ls = this.convOutLens[scaleIdx];
    const convO = this.convOut[scaleIdx];
    const scaleE = this.scaleEmb[scaleIdx];
    const out = this.scaleEmbOut[scaleIdx];

    for (let t = 0; t < Ls; t++) {
      for (let e = 0; e < E; e++) {
        out[t * E + e] = convO[t * E + e] + this.posEnc[t * E + e] + scaleE[e];
      }
    }
  }

  /**
   * Upsample scale output to full sequence length by repetition
   */
  private forwardUpsample(scaleIdx: number): void {
    const scale = this.config.temporalScales[scaleIdx];
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const Ls = this.convOutLens[scaleIdx];
    const src = this.scaleEmbOut[scaleIdx];
    const dst = this.upsampled[scaleIdx];

    for (let t = 0; t < L; t++) {
      const tSrc = Math.min(Math.floor(t / scale), Ls - 1);
      for (let e = 0; e < E; e++) {
        dst[t * E + e] = src[tSrc * E + e];
      }
    }
  }

  /**
   * Cross-scale fusion with gating
   * Concat all scale outputs, apply gate, weighted sum
   */
  private forwardFusion(): void {
    const nS = this.nScales;
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const fusionDim = nS * E;

    // Concatenate
    for (let t = 0; t < L; t++) {
      for (let s = 0; s < nS; s++) {
        for (let e = 0; e < E; e++) {
          this.fusionConcat[t * fusionDim + s * E + e] =
            this.upsampled[s][t * E + e];
        }
      }
    }

    // Gate: G = sigmoid(Concat * Wg + bg)
    for (let t = 0; t < L; t++) {
      for (let j = 0; j < fusionDim; j++) {
        let sum = this.fusionB[j];
        for (let i = 0; i < fusionDim; i++) {
          sum += this.fusionConcat[t * fusionDim + i] *
            this.fusionW[i * fusionDim + j];
        }
        this.fusionPreSig[t * fusionDim + j] = sum;
        this.fusionGate[t * fusionDim + j] = this.sigmoid(sum);
      }
    }

    // Apply fusion dropout (deterministic)
    if (this.config.fusionDropout > 0) {
      this.rngState = (this.updateCount * 31337) >>> 0 || 1;
      const p = this.config.fusionDropout;
      const scale = 1 / (1 - p);
      for (let i = 0; i < L * fusionDim; i++) {
        if (this.xorshift() < p) {
          this.fusionGate[i] = 0;
        } else {
          this.fusionGate[i] *= scale;
        }
      }
    }

    // Weighted sum: Fused[t,e] = sum_s G_s[t,e] * E_s_up[t,e]
    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        let sum = 0;
        for (let s = 0; s < nS; s++) {
          const gateVal = this.fusionGate[t * fusionDim + s * E + e];
          const embVal = this.upsampled[s][t * E + e];
          sum += gateVal * embVal;
        }
        this.fusedOut[t * E + e] = sum;
      }
    }
  }

  /**
   * LayerNorm forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
   */
  private layerNormForward(
    input: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    output: Float64Array,
    meanBuf: Float64Array,
    varBuf: Float64Array,
  ): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const eps = 1e-12;

    for (let t = 0; t < L; t++) {
      let mean = 0;
      for (let e = 0; e < E; e++) {
        mean += input[t * E + e];
      }
      mean /= E;
      meanBuf[t] = mean;

      let variance = 0;
      for (let e = 0; e < E; e++) {
        const diff = input[t * E + e] - mean;
        variance += diff * diff;
      }
      variance /= E;
      varBuf[t] = variance;

      const invStd = 1 / Math.sqrt(Math.max(eps, variance));
      for (let e = 0; e < E; e++) {
        const norm = (input[t * E + e] - mean) * invStd;
        output[t * E + e] = gamma[e] * norm + beta[e];
      }
    }
  }

  /**
   * Multi-head self-attention forward (causal)
   */
  private mhaForward(blockIdx: number, input: Float64Array): void {
    const E = this.config.embeddingDim;
    const H = this.config.numHeads;
    const dk = this.headDim;
    const L = this.seqLen;
    const sqrtDk = Math.sqrt(dk);

    const Wq = this.attnWq[blockIdx];
    const Wk = this.attnWk[blockIdx];
    const Wv = this.attnWv[blockIdx];
    const Wo = this.attnWo[blockIdx];
    const Q = this.mhaQ[blockIdx];
    const K = this.mhaK[blockIdx];
    const V = this.mhaV[blockIdx];
    const scores = this.mhaScores[blockIdx];
    const probs = this.mhaProbs[blockIdx];
    const headOut = this.mhaHeadOut[blockIdx];
    const output = this.attnOut[blockIdx];

    // Q = input * Wq, K = input * Wk, V = input * Wv
    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        let qSum = 0, kSum = 0, vSum = 0;
        for (let i = 0; i < E; i++) {
          const inVal = input[t * E + i];
          qSum += inVal * Wq[i * E + e];
          kSum += inVal * Wk[i * E + e];
          vSum += inVal * Wv[i * E + e];
        }
        Q[t * E + e] = qSum;
        K[t * E + e] = kSum;
        V[t * E + e] = vSum;
      }
    }

    // Apply attention dropout seed
    this.rngState = ((this.updateCount * 7919 + blockIdx * 1031) >>> 0) || 1;
    const dropP = this.config.attentionDropout;
    const dropScale = dropP > 0 ? 1 / (1 - dropP) : 1;

    // Per head attention
    for (let h = 0; h < H; h++) {
      const hOff = h * dk;

      // Compute scores: score[i,j] = Q[i] · K[j] / sqrt(dk)
      for (let i = 0; i < L; i++) {
        // Find max for numerical stability
        let maxScore = -1e30;
        for (let j = 0; j <= i; j++) {
          let dot = 0;
          for (let d = 0; d < dk; d++) {
            dot += Q[i * E + hOff + d] * K[j * E + hOff + d];
          }
          const s = dot / sqrtDk;
          scores[h * L * L + i * L + j] = s;
          if (s > maxScore) maxScore = s;
        }
        // Mask future positions
        for (let j = i + 1; j < L; j++) {
          scores[h * L * L + i * L + j] = -1e9;
        }

        // Softmax
        let sumExp = 0;
        for (let j = 0; j <= i; j++) {
          const expVal = Math.exp(scores[h * L * L + i * L + j] - maxScore);
          probs[h * L * L + i * L + j] = expVal;
          sumExp += expVal;
        }
        for (let j = i + 1; j < L; j++) {
          probs[h * L * L + i * L + j] = 0;
        }
        if (sumExp < 1e-30) sumExp = 1e-30;
        for (let j = 0; j < L; j++) {
          probs[h * L * L + i * L + j] /= sumExp;
        }

        // Apply dropout
        if (dropP > 0) {
          for (let j = 0; j < L; j++) {
            if (this.xorshift() < dropP) {
              probs[h * L * L + i * L + j] = 0;
            } else {
              probs[h * L * L + i * L + j] *= dropScale;
            }
          }
        }

        // Compute head output: sum_j prob[i,j] * V[j]
        for (let d = 0; d < dk; d++) {
          let sum = 0;
          for (let j = 0; j < L; j++) {
            sum += probs[h * L * L + i * L + j] * V[j * E + hOff + d];
          }
          headOut[i * E + hOff + d] = sum;
        }
      }
    }

    // Output projection: output = headOut * Wo
    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        let sum = 0;
        for (let i = 0; i < E; i++) {
          sum += headOut[t * E + i] * Wo[i * E + e];
        }
        output[t * E + e] = sum;
      }
    }
  }

  /**
   * FFN forward: GELU(x * W1 + b1) * W2 + b2
   */
  private ffnForward(blockIdx: number, input: Float64Array): void {
    const E = this.config.embeddingDim;
    const F = this.ffnDim;
    const L = this.seqLen;
    const W1 = this.ffnW1[blockIdx];
    const b1 = this.ffnB1[blockIdx];
    const W2 = this.ffnW2[blockIdx];
    const b2 = this.ffnB2[blockIdx];
    const hid = this.ffnHid[blockIdx];
    const preGelu = this.ffnPreGelu[blockIdx];
    const output = this.ffnOut[blockIdx];

    // Hidden layer
    for (let t = 0; t < L; t++) {
      for (let f = 0; f < F; f++) {
        let sum = b1[f];
        for (let e = 0; e < E; e++) {
          sum += input[t * E + e] * W1[e * F + f];
        }
        preGelu[t * F + f] = sum;
        hid[t * F + f] = this.gelu(sum);
      }
    }

    // Output layer
    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        let sum = b2[e];
        for (let f = 0; f < F; f++) {
          sum += hid[t * F + f] * W2[f * E + e];
        }
        output[t * E + e] = sum;
      }
    }
  }

  /**
   * Transformer block forward
   */
  private transformerBlockForward(blockIdx: number): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const input = blockIdx === 0 ? this.fusedOut : this.blockInputs[blockIdx];

    // Save input for residual
    if (blockIdx > 0) {
      for (let i = 0; i < L * E; i++) {
        this.blockInputs[blockIdx][i] = input[i];
      }
    } else {
      for (let i = 0; i < L * E; i++) {
        this.blockInputs[0][i] = this.fusedOut[i];
      }
    }

    // LN1 -> MHA -> residual
    this.layerNormForward(
      this.blockInputs[blockIdx],
      this.lnG1[blockIdx],
      this.lnB1[blockIdx],
      this.ln1Out[blockIdx],
      this.ln1Mean[blockIdx],
      this.ln1Var[blockIdx],
    );

    this.mhaForward(blockIdx, this.ln1Out[blockIdx]);

    // Residual
    for (let i = 0; i < L * E; i++) {
      this.attnResid[blockIdx][i] = this.blockInputs[blockIdx][i] +
        this.attnOut[blockIdx][i];
    }

    // LN2 -> FFN -> residual
    this.layerNormForward(
      this.attnResid[blockIdx],
      this.lnG2[blockIdx],
      this.lnB2[blockIdx],
      this.ln2Out[blockIdx],
      this.ln2Mean[blockIdx],
      this.ln2Var[blockIdx],
    );

    this.ffnForward(blockIdx, this.ln2Out[blockIdx]);

    // Final residual - store in next block's input or use as output
    const outBuf = blockIdx < this.config.numBlocks - 1
      ? this.blockInputs[blockIdx + 1]
      : this.blockInputs[blockIdx]; // Reuse for final
    for (let i = 0; i < L * E; i++) {
      outBuf[i] = this.attnResid[blockIdx][i] + this.ffnOut[blockIdx][i];
    }
  }

  /**
   * Attention-weighted temporal pooling
   * score[t] = H[t] · Wpool + bpool, alpha = softmax(scores), out = sum alpha * H
   */
  private forwardPooling(input: Float64Array): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;

    // Compute scores
    let maxScore = -1e30;
    for (let t = 0; t < L; t++) {
      let score = this.poolB[0];
      for (let e = 0; e < E; e++) {
        score += input[t * E + e] * this.poolW[e];
      }
      this.poolScores[t] = score;
      if (score > maxScore) maxScore = score;
    }

    // Softmax
    let sumExp = 0;
    for (let t = 0; t < L; t++) {
      const expVal = Math.exp(this.poolScores[t] - maxScore);
      this.poolAlpha[t] = expVal;
      sumExp += expVal;
    }
    if (sumExp < 1e-30) sumExp = 1e-30;
    for (let t = 0; t < L; t++) {
      this.poolAlpha[t] /= sumExp;
    }

    // Weighted sum
    for (let e = 0; e < E; e++) {
      let sum = 0;
      for (let t = 0; t < L; t++) {
        sum += this.poolAlpha[t] * input[t * E + e];
      }
      this.poolOut[e] = sum;
    }
  }

  /**
   * Output head forward: yHat = poolOut * Wout + bout
   */
  private forwardOutput(): void {
    const E = this.config.embeddingDim;
    const outD = this.outputDim;

    for (let d = 0; d < outD; d++) {
      let sum = this.outB[d];
      for (let e = 0; e < E; e++) {
        sum += this.poolOut[e] * this.outW[e * outD + d];
      }
      this.yHat[d] = sum;
    }
  }

  /**
   * Complete forward pass
   */
  private forward(): void {
    // Temporal conv per scale
    for (let s = 0; s < this.nScales; s++) {
      this.forwardConv(s);
      this.forwardScaleEmbed(s);
      this.forwardUpsample(s);
    }

    // Fusion
    this.forwardFusion();

    // Transformer blocks
    for (let b = 0; b < this.config.numBlocks; b++) {
      this.transformerBlockForward(b);
    }

    // Get final transformer output
    const lastBlockIdx = this.config.numBlocks - 1;
    const L = this.seqLen;
    const E = this.config.embeddingDim;

    // Final output after last block is in attnResid[last] + ffnOut[last]
    // Already computed and stored
    const finalOut = this.blockInputs[lastBlockIdx]; // This was overwritten with final output

    // Actually, need to recompute or use correct buffer
    // The final residual was stored in blockInputs[lastBlockIdx+1] if it exists,
    // otherwise we need a separate buffer. Let me fix this:
    // For the last block, store in a dedicated buffer
    for (let i = 0; i < L * E; i++) {
      this.dBlockOut[i] = this.attnResid[lastBlockIdx][i] +
        this.ffnOut[lastBlockIdx][i];
    }

    // Pooling
    this.forwardPooling(this.dBlockOut);

    // Output
    this.forwardOutput();
  }

  /**
   * Zero all gradients
   */
  private zeroGradients(): void {
    for (let s = 0; s < this.nScales; s++) {
      this.gConvW[s].fill(0);
      this.gConvB[s].fill(0);
      this.gScaleEmb[s].fill(0);
    }
    this.gFusionW.fill(0);
    this.gFusionB.fill(0);
    for (let b = 0; b < this.config.numBlocks; b++) {
      this.gAttnWq[b].fill(0);
      this.gAttnWk[b].fill(0);
      this.gAttnWv[b].fill(0);
      this.gAttnWo[b].fill(0);
      this.gFfnW1[b].fill(0);
      this.gFfnB1[b].fill(0);
      this.gFfnW2[b].fill(0);
      this.gFfnB2[b].fill(0);
      this.gLnG1[b].fill(0);
      this.gLnB1[b].fill(0);
      this.gLnG2[b].fill(0);
      this.gLnB2[b].fill(0);
    }
    this.gPoolW.fill(0);
    this.gPoolB.fill(0);
    this.gOutW.fill(0);
    this.gOutB.fill(0);
  }

  /**
   * Backward through output layer
   */
  private backwardOutput(dLoss: Float64Array): void {
    const E = this.config.embeddingDim;
    const outD = this.outputDim;

    // dWout[e,d] = poolOut[e] * dLoss[d]
    // dbout[d] = dLoss[d]
    // dPoolOut[e] = sum_d dLoss[d] * Wout[e,d]
    this.dPool.fill(0);
    for (let d = 0; d < outD; d++) {
      this.gOutB[d] += dLoss[d];
      for (let e = 0; e < E; e++) {
        this.gOutW[e * outD + d] += this.poolOut[e] * dLoss[d];
        this.dPool[e] += dLoss[d] * this.outW[e * outD + d];
      }
    }
  }

  /**
   * Backward through pooling
   */
  private backwardPooling(): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const lastBlockIdx = this.config.numBlocks - 1;

    // Recompute final transformer output
    for (let i = 0; i < L * E; i++) {
      this.dBlockOut[i] = this.attnResid[lastBlockIdx][i] +
        this.ffnOut[lastBlockIdx][i];
    }

    // dWpool[e] = sum_t alpha[t] * dPool[e] * (H[t,e] - sum_t' alpha[t'] * H[t',e])
    // Actually more complex - need full softmax backward

    // Softmax backward: d_score[t] = alpha[t] * (dPool · H[t] - sum_t' alpha[t'] * dPool · H[t'])
    let dPoolDotOut = 0;
    for (let t = 0; t < L; t++) {
      let dot = 0;
      for (let e = 0; e < E; e++) {
        dot += this.dPool[e] * this.dBlockOut[t * E + e];
      }
      dPoolDotOut += this.poolAlpha[t] * dot;
    }

    // dH[t,e] = alpha[t] * dPool[e]
    // dScore[t] = alpha[t] * (dPool · H[t] - dPoolDotOut)
    for (let t = 0; t < L; t++) {
      let dotT = 0;
      for (let e = 0; e < E; e++) {
        dotT += this.dPool[e] * this.dBlockOut[t * E + e];
      }
      const dScore = this.poolAlpha[t] * (dotT - dPoolDotOut);

      // dWpool[e] += dScore * H[t,e]
      // dbpool += dScore
      this.gPoolB[0] += dScore;
      for (let e = 0; e < E; e++) {
        this.gPoolW[e] += dScore * this.dBlockOut[t * E + e];
      }
    }

    // dH[t] = alpha[t] * dPool
    // Plus contribution from softmax: dH through scores
    // dH[t,e] += sum_t2 dScore[t2] * d(score[t2])/d(H[t,e])
    // score[t2] = H[t2] · Wpool + bpool
    // So d(score[t2])/d(H[t,e]) = Wpool[e] if t2==t else 0
    // Combined: dBlockOut_grad[t,e] = alpha[t] * dPool[e] + dScore[t] * Wpool[e]
    for (let t = 0; t < L; t++) {
      let dotT = 0;
      for (let e = 0; e < E; e++) {
        dotT += this.dPool[e] * this.dBlockOut[t * E + e];
      }
      const dScore = this.poolAlpha[t] * (dotT - dPoolDotOut);
      for (let e = 0; e < E; e++) {
        // Store in dBlockOut as gradient
        // Need separate buffer - use dFfnOut temporarily
        this.dFfnOut[t * E + e] = this.poolAlpha[t] * this.dPool[e] +
          dScore * this.poolW[e];
      }
    }
    // Copy to actual dBlockOut
    for (let i = 0; i < L * E; i++) {
      this.dBlockOut[i] = this.dFfnOut[i];
    }
  }

  /**
   * LayerNorm backward
   */
  private layerNormBackward(
    dOut: Float64Array,
    input: Float64Array,
    gamma: Float64Array,
    dGamma: Float64Array,
    dBeta: Float64Array,
    meanBuf: Float64Array,
    varBuf: Float64Array,
    dInput: Float64Array,
  ): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const eps = 1e-12;

    for (let t = 0; t < L; t++) {
      const mean = meanBuf[t];
      const variance = varBuf[t];
      const invStd = 1 / Math.sqrt(Math.max(eps, variance));

      // Accumulate dGamma, dBeta
      for (let e = 0; e < E; e++) {
        const norm = (input[t * E + e] - mean) * invStd;
        dGamma[e] += dOut[t * E + e] * norm;
        dBeta[e] += dOut[t * E + e];
      }

      // dNorm = dOut * gamma
      // dVar = sum_e dNorm_e * (x_e - mean) * (-0.5) * (var + eps)^(-1.5)
      // dMean = sum_e dNorm_e * (-invStd) + dVar * sum_e (-2/E)*(x_e - mean)
      // dX = dNorm * invStd + dVar * 2/E * (x - mean) + dMean / E

      let dVar = 0;
      let dMeanPart1 = 0;
      let dMeanPart2 = 0;
      for (let e = 0; e < E; e++) {
        const dNorm = dOut[t * E + e] * gamma[e];
        const xMinusMean = input[t * E + e] - mean;
        dVar += dNorm * xMinusMean * (-0.5) * invStd * invStd * invStd;
        dMeanPart1 += dNorm * (-invStd);
        dMeanPart2 += xMinusMean;
      }
      const dMean = dMeanPart1 + dVar * (-2 / E) * dMeanPart2;

      for (let e = 0; e < E; e++) {
        const dNorm = dOut[t * E + e] * gamma[e];
        const xMinusMean = input[t * E + e] - mean;
        dInput[t * E + e] = dNorm * invStd + dVar * 2 / E * xMinusMean +
          dMean / E;
      }
    }
  }

  /**
   * MHA backward
   */
  private mhaBackward(
    blockIdx: number,
    dOut: Float64Array,
    input: Float64Array,
  ): void {
    const E = this.config.embeddingDim;
    const H = this.config.numHeads;
    const dk = this.headDim;
    const L = this.seqLen;
    const sqrtDk = Math.sqrt(dk);

    const Wq = this.attnWq[blockIdx];
    const Wk = this.attnWk[blockIdx];
    const Wv = this.attnWv[blockIdx];
    const Wo = this.attnWo[blockIdx];
    const Q = this.mhaQ[blockIdx];
    const K = this.mhaK[blockIdx];
    const V = this.mhaV[blockIdx];
    const probs = this.mhaProbs[blockIdx];
    const headOut = this.mhaHeadOut[blockIdx];

    // dWo[i,e] = sum_t headOut[t,i] * dOut[t,e]
    // dHeadOut[t,i] = sum_e dOut[t,e] * Wo[i,e]
    this.dMhaConcat.fill(0);
    for (let t = 0; t < L; t++) {
      for (let i = 0; i < E; i++) {
        let sum = 0;
        for (let e = 0; e < E; e++) {
          this.gAttnWo[blockIdx][i * E + e] += headOut[t * E + i] *
            dOut[t * E + e];
          sum += dOut[t * E + e] * Wo[i * E + e];
        }
        this.dMhaConcat[t * E + i] = sum;
      }
    }

    // Backward through attention per head
    this.dMhaQ.fill(0);
    this.dMhaK.fill(0);
    this.dMhaV.fill(0);

    for (let h = 0; h < H; h++) {
      const hOff = h * dk;

      for (let i = 0; i < L; i++) {
        // dHeadOut[i,hOff:hOff+dk] -> dProbs, dV
        // headOut[i] = sum_j probs[i,j] * V[j]
        // dProbs[i,j] = dHeadOut[i] · V[j]
        // dV[j] += probs[i,j] * dHeadOut[i]

        // Compute dProbs contribution
        const dProbsBuf = new Float64Array(L); // Small temp allocation per position
        for (let j = 0; j < L; j++) {
          let dot = 0;
          for (let d = 0; d < dk; d++) {
            dot += this.dMhaConcat[i * E + hOff + d] * V[j * E + hOff + d];
          }
          dProbsBuf[j] = dot;
        }

        // dV
        for (let j = 0; j < L; j++) {
          const p = probs[h * L * L + i * L + j];
          for (let d = 0; d < dk; d++) {
            this.dMhaV[j * E + hOff + d] += p *
              this.dMhaConcat[i * E + hOff + d];
          }
        }

        // Softmax backward: dScore[j] = probs[j] * (dProbs[j] - sum_k probs[k] * dProbs[k])
        let sumProbDProb = 0;
        for (let j = 0; j <= i; j++) {
          sumProbDProb += probs[h * L * L + i * L + j] * dProbsBuf[j];
        }

        // dScore -> dQ, dK
        // score[i,j] = Q[i] · K[j] / sqrt(dk)
        // dQ[i] += dScore[i,j] * K[j] / sqrt(dk)
        // dK[j] += dScore[i,j] * Q[i] / sqrt(dk)
        for (let j = 0; j <= i; j++) {
          const dScore = probs[h * L * L + i * L + j] *
            (dProbsBuf[j] - sumProbDProb);
          for (let d = 0; d < dk; d++) {
            this.dMhaQ[i * E + hOff + d] += dScore * K[j * E + hOff + d] /
              sqrtDk;
            this.dMhaK[j * E + hOff + d] += dScore * Q[i * E + hOff + d] /
              sqrtDk;
          }
        }
      }
    }

    // Backward through Q, K, V projections
    // Q = input * Wq => dWq[i,e] = sum_t input[t,i] * dQ[t,e], dInput += dQ * Wq^T
    this.dAttnOut.fill(0);
    for (let t = 0; t < L; t++) {
      for (let i = 0; i < E; i++) {
        for (let e = 0; e < E; e++) {
          this.gAttnWq[blockIdx][i * E + e] += input[t * E + i] *
            this.dMhaQ[t * E + e];
          this.gAttnWk[blockIdx][i * E + e] += input[t * E + i] *
            this.dMhaK[t * E + e];
          this.gAttnWv[blockIdx][i * E + e] += input[t * E + i] *
            this.dMhaV[t * E + e];
          this.dAttnOut[t * E + i] += this.dMhaQ[t * E + e] * Wq[i * E + e];
          this.dAttnOut[t * E + i] += this.dMhaK[t * E + e] * Wk[i * E + e];
          this.dAttnOut[t * E + i] += this.dMhaV[t * E + e] * Wv[i * E + e];
        }
      }
    }
  }

  /**
   * FFN backward
   */
  private ffnBackward(
    blockIdx: number,
    dOut: Float64Array,
    input: Float64Array,
  ): void {
    const E = this.config.embeddingDim;
    const F = this.ffnDim;
    const L = this.seqLen;
    const W1 = this.ffnW1[blockIdx];
    const W2 = this.ffnW2[blockIdx];
    const hid = this.ffnHid[blockIdx];
    const preGelu = this.ffnPreGelu[blockIdx];

    // dW2[f,e] = sum_t hid[t,f] * dOut[t,e]
    // db2[e] = sum_t dOut[t,e]
    // dHid[t,f] = sum_e dOut[t,e] * W2[f,e]
    this.dFfnHid.fill(0);
    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        this.gFfnB2[blockIdx][e] += dOut[t * E + e];
        for (let f = 0; f < F; f++) {
          this.gFfnW2[blockIdx][f * E + e] += hid[t * F + f] * dOut[t * E + e];
          this.dFfnHid[t * F + f] += dOut[t * E + e] * W2[f * E + e];
        }
      }
    }

    // GELU backward: dPreGelu = dHid * gelu'(preGelu)
    // dW1[e,f] = sum_t input[t,e] * dPreGelu[t,f]
    // db1[f] = sum_t dPreGelu[t,f]
    // dInput[t,e] = sum_f dPreGelu[t,f] * W1[e,f]
    this.dLn2Out.fill(0);
    for (let t = 0; t < L; t++) {
      for (let f = 0; f < F; f++) {
        const dPreGelu = this.dFfnHid[t * F + f] *
          this.geluDeriv(preGelu[t * F + f]);
        this.gFfnB1[blockIdx][f] += dPreGelu;
        for (let e = 0; e < E; e++) {
          this.gFfnW1[blockIdx][e * F + f] += input[t * E + e] * dPreGelu;
          this.dLn2Out[t * E + e] += dPreGelu * W1[e * F + f];
        }
      }
    }
  }

  /**
   * Transformer block backward
   */
  private transformerBlockBackward(blockIdx: number, dOut: Float64Array): void {
    const E = this.config.embeddingDim;
    const L = this.seqLen;

    // dOut = gradient flowing from next block or pooling
    // Block: LN1 -> MHA -> +resid -> LN2 -> FFN -> +resid
    // So backward: dResid2 -> dFFN + dResid2 -> dLN2 -> dResid1 -> dMHA + dResid1 -> dLN1 -> dInput

    // dResid2 = dOut (after final residual)
    // Split: dFFNOut = dResid2, dAttnResid = dResid2 (residual connection)
    for (let i = 0; i < L * E; i++) {
      this.dFfnOut[i] = dOut[i];
    }

    // FFN backward
    this.ffnBackward(blockIdx, this.dFfnOut, this.ln2Out[blockIdx]);

    // LN2 backward + residual from FFN path
    // dLn2Input = LN2_backward(dLn2Out)
    // dAttnResid = dLn2Input + dResid2 (from residual)
    this.dAttnResid.fill(0);
    this.layerNormBackward(
      this.dLn2Out,
      this.attnResid[blockIdx],
      this.lnG2[blockIdx],
      this.gLnG2[blockIdx],
      this.gLnB2[blockIdx],
      this.ln2Mean[blockIdx],
      this.ln2Var[blockIdx],
      this.dAttnResid,
    );
    // Add residual gradient
    for (let i = 0; i < L * E; i++) {
      this.dAttnResid[i] += dOut[i];
    }

    // Split for MHA residual: dMHAOut = dAttnResid, dBlockInput_part = dAttnResid
    for (let i = 0; i < L * E; i++) {
      this.dAttnOut[i] = this.dAttnResid[i];
    }

    // MHA backward
    this.mhaBackward(blockIdx, this.dAttnOut, this.ln1Out[blockIdx]);

    // LN1 backward
    this.dLn1Out.fill(0);
    this.layerNormBackward(
      this.dAttnOut, // dMHA flows into LN1
      this.blockInputs[blockIdx],
      this.lnG1[blockIdx],
      this.gLnG1[blockIdx],
      this.gLnB1[blockIdx],
      this.ln1Mean[blockIdx],
      this.ln1Var[blockIdx],
      this.dLn1Out,
    );

    // Combine: dBlockInput = dLn1Out + dAttnResid (residual from attention)
    for (let i = 0; i < L * E; i++) {
      this.dBlockOut[i] = this.dLn1Out[i] + this.dAttnResid[i];
    }
  }

  /**
   * Fusion backward
   */
  private backwardFusion(): void {
    const nS = this.nScales;
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const fusionDim = nS * E;

    // Forward was: Fused[t,e] = sum_s G_s[t,e] * E_s_up[t,e]
    // where G = sigmoid(Concat * Wg + bg)

    // dG[t,s,e] = dFused[t,e] * E_s_up[t,e]
    // dE_s_up[t,e] = dFused[t,e] * G_s[t,e]
    this.dFusionGate.fill(0);
    for (let s = 0; s < nS; s++) {
      this.dUpsampled[s].fill(0);
    }

    for (let t = 0; t < L; t++) {
      for (let e = 0; e < E; e++) {
        const dF = this.dFused[t * E + e];
        for (let s = 0; s < nS; s++) {
          const gIdx = t * fusionDim + s * E + e;
          const embVal = this.upsampled[s][t * E + e];
          const gateVal = this.fusionGate[gIdx];
          this.dFusionGate[gIdx] = dF * embVal;
          this.dUpsampled[s][t * E + e] = dF * gateVal;
        }
      }
    }

    // Sigmoid backward: dPreSig = dG * sigmoid'(preSig) = dG * G * (1-G)
    // dWg[i,j] = sum_t Concat[t,i] * dPreSig[t,j]
    // dbg[j] = sum_t dPreSig[t,j]
    // dConcat[t,i] = sum_j dPreSig[t,j] * Wg[i,j]
    this.dFusionConcat.fill(0);
    for (let t = 0; t < L; t++) {
      for (let j = 0; j < fusionDim; j++) {
        const g = this.fusionGate[t * fusionDim + j];
        const dPreSig = this.dFusionGate[t * fusionDim + j] * g * (1 - g);
        this.gFusionB[j] += dPreSig;
        for (let i = 0; i < fusionDim; i++) {
          this.gFusionW[i * fusionDim + j] +=
            this.fusionConcat[t * fusionDim + i] * dPreSig;
          this.dFusionConcat[t * fusionDim + i] += dPreSig *
            this.fusionW[i * fusionDim + j];
        }
      }
    }

    // Add gradient from concat to upsampled
    for (let t = 0; t < L; t++) {
      for (let s = 0; s < nS; s++) {
        for (let e = 0; e < E; e++) {
          this.dUpsampled[s][t * E + e] +=
            this.dFusionConcat[t * fusionDim + s * E + e];
        }
      }
    }
  }

  /**
   * Upsample backward - accumulate to downsampled positions
   */
  private backwardUpsample(scaleIdx: number): void {
    const scale = this.config.temporalScales[scaleIdx];
    const E = this.config.embeddingDim;
    const L = this.seqLen;
    const Ls = this.convOutLens[scaleIdx];

    this.dScaleEmbOut[scaleIdx].fill(0);

    for (let t = 0; t < L; t++) {
      const tSrc = Math.min(Math.floor(t / scale), Ls - 1);
      for (let e = 0; e < E; e++) {
        this.dScaleEmbOut[scaleIdx][tSrc * E + e] +=
          this.dUpsampled[scaleIdx][t * E + e];
      }
    }
  }

  /**
   * Scale embedding backward
   */
  private backwardScaleEmbed(scaleIdx: number): void {
    const E = this.config.embeddingDim;
    const Ls = this.convOutLens[scaleIdx];

    // dScaleEmb[e] = sum_t dScaleEmbOut[t,e]
    // dConvOut = dScaleEmbOut (positional encoding has no learnable params)
    for (let t = 0; t < Ls; t++) {
      for (let e = 0; e < E; e++) {
        this.gScaleEmb[scaleIdx][e] += this.dScaleEmbOut[scaleIdx][t * E + e];
        this.dConvOut[scaleIdx][t * E + e] =
          this.dScaleEmbOut[scaleIdx][t * E + e];
      }
    }
  }

  /**
   * Temporal conv backward
   */
  private backwardConv(scaleIdx: number): void {
    const scale = this.config.temporalScales[scaleIdx];
    const K = this.config.temporalKernelSize;
    const E = this.config.embeddingDim;
    const inD = this.inputDim;
    const L = this.seqLen;
    const Ls = this.convOutLens[scaleIdx];

    // GELU backward
    // dPreGelu = dConvOut * gelu'(preGelu)
    // dW[k,f,e] = sum_t X[(t*s-k),f] * dPreGelu[t,e]
    // db[e] = sum_t dPreGelu[t,e]
    // dX[(t*s-k),f] += sum_e dPreGelu[t,e] * W[k,f,e]

    for (let t = 0; t < Ls; t++) {
      const tOrig = t * scale;
      for (let e = 0; e < E; e++) {
        const dPreGelu = this.dConvOut[scaleIdx][t * E + e] *
          this.geluDeriv(this.convPreGelu[scaleIdx][t * E + e]);
        this.gConvB[scaleIdx][e] += dPreGelu;
        for (let k = 0; k < K; k++) {
          const tSrc = tOrig - k;
          if (tSrc >= 0 && tSrc < L) {
            for (let f = 0; f < inD; f++) {
              this.gConvW[scaleIdx][(k * inD + f) * E + e] +=
                this.xNorm[tSrc * inD + f] * dPreGelu;
              this.dXNorm[tSrc * inD + f] += dPreGelu *
                this.convW[scaleIdx][(k * inD + f) * E + e];
            }
          }
        }
      }
    }
  }

  /**
   * Complete backward pass
   */
  private backward(yNorm: Float64Array, sampleWeight: number): void {
    const outD = this.outputDim;
    const L = this.seqLen;
    const E = this.config.embeddingDim;

    this.zeroGradients();

    // Output loss gradient: dL/dyHat = (yHat - yNorm) * sampleWeight / outputDim
    for (let d = 0; d < outD; d++) {
      this.dOut[d] = (this.yHat[d] - yNorm[d]) * sampleWeight / outD;
    }

    // Backward through output
    this.backwardOutput(this.dOut);

    // Backward through pooling
    this.backwardPooling();

    // Backward through transformer blocks (reverse order)
    for (let b = this.config.numBlocks - 1; b >= 0; b--) {
      this.transformerBlockBackward(b, this.dBlockOut);
      // dBlockOut now contains gradient for previous block's output
    }

    // dBlockOut contains gradient for fusion output
    for (let i = 0; i < L * E; i++) {
      this.dFused[i] = this.dBlockOut[i];
    }

    // Backward through fusion
    this.backwardFusion();

    // Backward through each scale
    this.dXNorm.fill(0);
    for (let s = 0; s < this.nScales; s++) {
      this.backwardUpsample(s);
      this.backwardScaleEmbed(s);
      this.backwardConv(s);
    }
  }

  /**
   * Add L2 regularization to gradients
   */
  private addL2Regularization(): void {
    const lambda = this.config.regularizationStrength;

    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.gConvW[s].length; i++) {
        this.gConvW[s][i] += lambda * this.convW[s][i];
      }
      for (let i = 0; i < this.gScaleEmb[s].length; i++) {
        this.gScaleEmb[s][i] += lambda * this.scaleEmb[s][i];
      }
    }

    for (let i = 0; i < this.gFusionW.length; i++) {
      this.gFusionW[i] += lambda * this.fusionW[i];
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.gAttnWq[b].length; i++) {
        this.gAttnWq[b][i] += lambda * this.attnWq[b][i];
        this.gAttnWk[b][i] += lambda * this.attnWk[b][i];
        this.gAttnWv[b][i] += lambda * this.attnWv[b][i];
        this.gAttnWo[b][i] += lambda * this.attnWo[b][i];
      }
      for (let i = 0; i < this.gFfnW1[b].length; i++) {
        this.gFfnW1[b][i] += lambda * this.ffnW1[b][i];
      }
      for (let i = 0; i < this.gFfnW2[b].length; i++) {
        this.gFfnW2[b][i] += lambda * this.ffnW2[b][i];
      }
    }

    for (let i = 0; i < this.gOutW.length; i++) {
      this.gOutW[i] += lambda * this.outW[i];
    }
    for (let i = 0; i < this.gPoolW.length; i++) {
      this.gPoolW[i] += lambda * this.poolW[i];
    }
  }

  /**
   * Compute gradient norm and apply clipping
   */
  private clipGradients(): number {
    let normSq = 0;

    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.gConvW[s].length; i++) {
        normSq += this.gConvW[s][i] ** 2;
      }
      for (let i = 0; i < this.gConvB[s].length; i++) {
        normSq += this.gConvB[s][i] ** 2;
      }
      for (let i = 0; i < this.gScaleEmb[s].length; i++) {
        normSq += this.gScaleEmb[s][i] ** 2;
      }
    }
    for (let i = 0; i < this.gFusionW.length; i++) {
      normSq += this.gFusionW[i] ** 2;
    }
    for (let i = 0; i < this.gFusionB.length; i++) {
      normSq += this.gFusionB[i] ** 2;
    }

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.gAttnWq[b].length; i++) {
        normSq += this.gAttnWq[b][i] ** 2;
        normSq += this.gAttnWk[b][i] ** 2;
        normSq += this.gAttnWv[b][i] ** 2;
        normSq += this.gAttnWo[b][i] ** 2;
      }
      for (let i = 0; i < this.gFfnW1[b].length; i++) {
        normSq += this.gFfnW1[b][i] ** 2;
      }
      for (let i = 0; i < this.gFfnB1[b].length; i++) {
        normSq += this.gFfnB1[b][i] ** 2;
      }
      for (let i = 0; i < this.gFfnW2[b].length; i++) {
        normSq += this.gFfnW2[b][i] ** 2;
      }
      for (let i = 0; i < this.gFfnB2[b].length; i++) {
        normSq += this.gFfnB2[b][i] ** 2;
      }
      for (let i = 0; i < this.gLnG1[b].length; i++) {
        normSq += this.gLnG1[b][i] ** 2;
        normSq += this.gLnB1[b][i] ** 2;
        normSq += this.gLnG2[b][i] ** 2;
        normSq += this.gLnB2[b][i] ** 2;
      }
    }

    for (let i = 0; i < this.gPoolW.length; i++) normSq += this.gPoolW[i] ** 2;
    normSq += this.gPoolB[0] ** 2;
    for (let i = 0; i < this.gOutW.length; i++) normSq += this.gOutW[i] ** 2;
    for (let i = 0; i < this.gOutB.length; i++) normSq += this.gOutB[i] ** 2;

    const norm = Math.sqrt(normSq);
    const maxNorm = 5.0;

    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (let s = 0; s < this.nScales; s++) {
        for (let i = 0; i < this.gConvW[s].length; i++) {
          this.gConvW[s][i] *= scale;
        }
        for (let i = 0; i < this.gConvB[s].length; i++) {
          this.gConvB[s][i] *= scale;
        }
        for (let i = 0; i < this.gScaleEmb[s].length; i++) {
          this.gScaleEmb[s][i] *= scale;
        }
      }
      for (let i = 0; i < this.gFusionW.length; i++) this.gFusionW[i] *= scale;
      for (let i = 0; i < this.gFusionB.length; i++) this.gFusionB[i] *= scale;

      for (let b = 0; b < this.config.numBlocks; b++) {
        for (let i = 0; i < this.gAttnWq[b].length; i++) {
          this.gAttnWq[b][i] *= scale;
          this.gAttnWk[b][i] *= scale;
          this.gAttnWv[b][i] *= scale;
          this.gAttnWo[b][i] *= scale;
        }
        for (let i = 0; i < this.gFfnW1[b].length; i++) {
          this.gFfnW1[b][i] *= scale;
        }
        for (let i = 0; i < this.gFfnB1[b].length; i++) {
          this.gFfnB1[b][i] *= scale;
        }
        for (let i = 0; i < this.gFfnW2[b].length; i++) {
          this.gFfnW2[b][i] *= scale;
        }
        for (let i = 0; i < this.gFfnB2[b].length; i++) {
          this.gFfnB2[b][i] *= scale;
        }
        for (let i = 0; i < this.gLnG1[b].length; i++) {
          this.gLnG1[b][i] *= scale;
          this.gLnB1[b][i] *= scale;
          this.gLnG2[b][i] *= scale;
          this.gLnB2[b][i] *= scale;
        }
      }

      for (let i = 0; i < this.gPoolW.length; i++) this.gPoolW[i] *= scale;
      this.gPoolB[0] *= scale;
      for (let i = 0; i < this.gOutW.length; i++) this.gOutW[i] *= scale;
      for (let i = 0; i < this.gOutB.length; i++) this.gOutB[i] *= scale;
    }

    return norm;
  }

  /**
   * Adam update for a parameter array
   */
  private adamUpdate(
    w: Float64Array,
    g: Float64Array,
    m: Float64Array,
    v: Float64Array,
    lr: number,
  ): void {
    const { beta1, beta2, epsilon } = this.config;
    const t = this.updateCount;
    const biasCorr1 = 1 - Math.pow(beta1, t);
    const biasCorr2 = 1 - Math.pow(beta2, t);

    for (let i = 0; i < w.length; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
      const mHat = m[i] / biasCorr1;
      const vHat = v[i] / biasCorr2;
      w[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  /**
   * Apply Adam updates to all parameters
   */
  private applyAdam(lr: number): void {
    for (let s = 0; s < this.nScales; s++) {
      this.adamUpdate(
        this.convW[s],
        this.gConvW[s],
        this.mConvW[s],
        this.vConvW[s],
        lr,
      );
      this.adamUpdate(
        this.convB[s],
        this.gConvB[s],
        this.mConvB[s],
        this.vConvB[s],
        lr,
      );
      this.adamUpdate(
        this.scaleEmb[s],
        this.gScaleEmb[s],
        this.mScaleEmb[s],
        this.vScaleEmb[s],
        lr,
      );
    }

    this.adamUpdate(
      this.fusionW,
      this.gFusionW,
      this.mFusionW,
      this.vFusionW,
      lr,
    );
    this.adamUpdate(
      this.fusionB,
      this.gFusionB,
      this.mFusionB,
      this.vFusionB,
      lr,
    );

    for (let b = 0; b < this.config.numBlocks; b++) {
      this.adamUpdate(
        this.attnWq[b],
        this.gAttnWq[b],
        this.mAttnWq[b],
        this.vAttnWq[b],
        lr,
      );
      this.adamUpdate(
        this.attnWk[b],
        this.gAttnWk[b],
        this.mAttnWk[b],
        this.vAttnWk[b],
        lr,
      );
      this.adamUpdate(
        this.attnWv[b],
        this.gAttnWv[b],
        this.mAttnWv[b],
        this.vAttnWv[b],
        lr,
      );
      this.adamUpdate(
        this.attnWo[b],
        this.gAttnWo[b],
        this.mAttnWo[b],
        this.vAttnWo[b],
        lr,
      );
      this.adamUpdate(
        this.ffnW1[b],
        this.gFfnW1[b],
        this.mFfnW1[b],
        this.vFfnW1[b],
        lr,
      );
      this.adamUpdate(
        this.ffnB1[b],
        this.gFfnB1[b],
        this.mFfnB1[b],
        this.vFfnB1[b],
        lr,
      );
      this.adamUpdate(
        this.ffnW2[b],
        this.gFfnW2[b],
        this.mFfnW2[b],
        this.vFfnW2[b],
        lr,
      );
      this.adamUpdate(
        this.ffnB2[b],
        this.gFfnB2[b],
        this.mFfnB2[b],
        this.vFfnB2[b],
        lr,
      );
      this.adamUpdate(
        this.lnG1[b],
        this.gLnG1[b],
        this.mLnG1[b],
        this.vLnG1[b],
        lr,
      );
      this.adamUpdate(
        this.lnB1[b],
        this.gLnB1[b],
        this.mLnB1[b],
        this.vLnB1[b],
        lr,
      );
      this.adamUpdate(
        this.lnG2[b],
        this.gLnG2[b],
        this.mLnG2[b],
        this.vLnG2[b],
        lr,
      );
      this.adamUpdate(
        this.lnB2[b],
        this.gLnB2[b],
        this.mLnB2[b],
        this.vLnB2[b],
        lr,
      );
    }

    this.adamUpdate(this.poolW, this.gPoolW, this.mPoolW, this.vPoolW, lr);
    this.adamUpdate(this.poolB, this.gPoolB, this.mPoolB, this.vPoolB, lr);
    this.adamUpdate(this.outW, this.gOutW, this.mOutW, this.vOutW, lr);
    this.adamUpdate(this.outB, this.gOutB, this.mOutB, this.vOutB, lr);
  }

  /**
   * ADWIN drift detection
   * @returns true if drift detected
   */
  private adwinCheck(loss: number): boolean {
    // Add to ring buffer
    this.adwinBuf[this.adwinHead] = loss;
    this.adwinHead = (this.adwinHead + 1) % this.ADWIN_CAP;
    if (this.adwinSize < this.ADWIN_CAP) this.adwinSize++;

    if (this.adwinSize < this.ADWIN_MIN) return false;

    // Find best split point
    const delta = this.config.adwinDelta;
    let driftDetected = false;

    // Compute total sum
    let totalSum = 0;
    for (let i = 0; i < this.adwinSize; i++) {
      const idx = (this.adwinHead - this.adwinSize + i + this.ADWIN_CAP) %
        this.ADWIN_CAP;
      totalSum += this.adwinBuf[idx];
    }

    // Try different split points
    let leftSum = 0;
    for (
      let split = this.ADWIN_MIN;
      split <= this.adwinSize - this.ADWIN_MIN;
      split++
    ) {
      const idx =
        (this.adwinHead - this.adwinSize + split - 1 + this.ADWIN_CAP) %
        this.ADWIN_CAP;
      leftSum += this.adwinBuf[idx];
      const rightSum = totalSum - leftSum;

      const nLeft = split;
      const nRight = this.adwinSize - split;
      const meanLeft = leftSum / nLeft;
      const meanRight = rightSum / nRight;

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
      // Reset window
      this.adwinSize = 1;
      this.adwinHead = 1;
      this.adwinBuf[0] = loss;
      this.driftCount++;
      this.runningLoss = loss;
    }

    return driftDetected;
  }

  /**
   * Fit one sample online
   * @param data - Training sample with xCoordinates and yCoordinates
   * @returns FitResult with training metrics
   */
  fitOnline(
    data: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = data;

    // Auto-detect dimensions on first call
    if (!this.isInitialized) {
      this.inputDim = xCoordinates[0].length;
      this.outputDim = yCoordinates[0].length;
      this.seqLen = Math.min(
        xCoordinates.length,
        this.config.maxSequenceLength,
      );
      this.initialize();
    }

    this.sampleCount++;
    this.updateCount++;

    const L = this.seqLen;
    const inD = this.inputDim;
    const outD = this.outputDim;

    // Cache window for prediction (clipped/padded)
    const actualLen = Math.min(
      xCoordinates.length,
      this.config.maxSequenceLength,
    );
    this.cachedWindowLen = actualLen;
    for (let t = 0; t < actualLen; t++) {
      for (let f = 0; f < inD; f++) {
        this.cachedWindow[t * inD + f] = xCoordinates[t][f];
      }
    }

    // Update Welford stats
    for (let t = 0; t < L; t++) {
      const n = this.sampleCount;
      for (let f = 0; f < inD; f++) {
        const val = xCoordinates[t][f];
        const delta = val - this.inputMean[f];
        this.inputMean[f] += delta / n;
        const delta2 = val - this.inputMean[f];
        this.inputM2[f] += delta * delta2;
      }
    }

    const yTarget = yCoordinates[L - 1];
    for (let d = 0; d < outD; d++) {
      const val = yTarget[d];
      const delta = val - this.outputMean[d];
      this.outputMean[d] += delta / this.sampleCount;
      const delta2 = val - this.outputMean[d];
      this.outputM2[d] += delta * delta2;
    }

    // Compute std
    const inputStd = new Float64Array(inD);
    const outputStd = new Float64Array(outD);
    this.getStd(this.inputM2, this.sampleCount, inputStd);
    this.getStd(this.outputM2, this.sampleCount, outputStd);

    // Normalize input
    for (let t = 0; t < L; t++) {
      for (let f = 0; f < inD; f++) {
        this.xNorm[t * inD + f] = (xCoordinates[t][f] - this.inputMean[f]) /
          inputStd[f];
      }
    }

    // Normalize target
    const yNorm = new Float64Array(outD);
    for (let d = 0; d < outD; d++) {
      yNorm[d] = (yTarget[d] - this.outputMean[d]) / outputStd[d];
    }

    // Forward pass
    this.forward();

    // Compute loss (MSE on normalized)
    let mseLoss = 0;
    let isOutlier = false;
    for (let d = 0; d < outD; d++) {
      const residual = yNorm[d] - this.yHat[d];
      mseLoss += residual * residual;
      if (Math.abs(residual) > this.config.outlierThreshold) {
        isOutlier = true;
      }
    }
    mseLoss = mseLoss / (2 * outD);

    // Add L2 penalty to loss
    let l2Loss = 0;
    const lambda = this.config.regularizationStrength;
    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.convW[s].length; i++) {
        l2Loss += this.convW[s][i] ** 2;
      }
    }
    for (let i = 0; i < this.fusionW.length; i++) {
      l2Loss += this.fusionW[i] ** 2;
    }
    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.attnWq[b].length; i++) {
        l2Loss += this.attnWq[b][i] ** 2 + this.attnWk[b][i] ** 2 +
          this.attnWv[b][i] ** 2 + this.attnWo[b][i] ** 2;
      }
      for (let i = 0; i < this.ffnW1[b].length; i++) {
        l2Loss += this.ffnW1[b][i] ** 2;
      }
      for (let i = 0; i < this.ffnW2[b].length; i++) {
        l2Loss += this.ffnW2[b][i] ** 2;
      }
    }
    for (let i = 0; i < this.outW.length; i++) l2Loss += this.outW[i] ** 2;
    l2Loss = (lambda / 2) * l2Loss;

    const totalLoss = mseLoss + l2Loss;
    const sampleWeight = isOutlier ? 0.1 : 1.0;

    // Update running loss
    this.runningLoss =
      (this.runningLoss * (this.sampleCount - 1) + totalLoss * sampleWeight) /
      this.sampleCount;

    // Update residual M2 for prediction uncertainty
    for (let d = 0; d < outD; d++) {
      const residual = (yNorm[d] - this.yHat[d]) * outputStd[d];
      const delta = residual * residual -
        (this.residualM2[d] / Math.max(1, this.sampleCount - 1));
      this.residualM2[d] += delta;
    }

    // Backward pass
    this.backward(yNorm, sampleWeight);

    // Add L2 regularization to gradients
    this.addL2Regularization();

    // Clip gradients
    const gradNorm = this.clipGradients();

    // Get learning rate
    const lr = this.getLearningRate();

    // Apply Adam
    this.applyAdam(lr);

    // Check convergence
    this.converged = gradNorm < this.config.convergenceThreshold;

    // ADWIN drift detection
    const driftDetected = this.adwinCheck(totalLoss * sampleWeight);

    return {
      loss: totalLoss,
      gradientNorm: gradNorm,
      effectiveLearningRate: lr,
      isOutlier,
      converged: this.converged,
      sampleIndex: this.sampleCount,
      driftDetected,
    };
  }

  /**
   * Predict future steps
   * @param futureSteps - Number of steps to predict
   * @returns PredictionResult with predictions and uncertainty
   */
  predict(futureSteps: number): PredictionResult {
    const isModelReady = this.isInitialized && this.sampleCount >= 2;

    if (!isModelReady) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: this.sampleCount,
        isModelReady: false,
      };
    }

    const L = Math.min(this.cachedWindowLen, this.seqLen);
    const inD = this.inputDim;
    const outD = this.outputDim;

    // Compute std for normalization
    const inputStd = new Float64Array(inD);
    const outputStd = new Float64Array(outD);
    this.getStd(this.inputM2, this.sampleCount, inputStd);
    this.getStd(this.outputM2, this.sampleCount, outputStd);

    // Normalize cached window
    for (let t = 0; t < L; t++) {
      for (let f = 0; f < inD; f++) {
        this.xNorm[t * inD + f] =
          (this.cachedWindow[t * inD + f] - this.inputMean[f]) / inputStd[f];
      }
    }

    // Adjust seqLen temporarily if needed
    const originalSeqLen = this.seqLen;
    this.seqLen = L;

    // Recompute convOutLens for current seqLen
    for (let s = 0; s < this.nScales; s++) {
      this.convOutLens[s] = Math.ceil(L / this.config.temporalScales[s]);
    }

    // Forward pass
    this.forward();

    // Restore seqLen
    this.seqLen = originalSeqLen;
    for (let s = 0; s < this.nScales; s++) {
      this.convOutLens[s] = Math.ceil(
        originalSeqLen / this.config.temporalScales[s],
      );
    }

    // Denormalize prediction
    const basePrediction = new Float64Array(outD);
    for (let d = 0; d < outD; d++) {
      basePrediction[d] = this.yHat[d] * outputStd[d] + this.outputMean[d];
    }

    // Compute base standard error from residual variance
    const baseStdErr = new Float64Array(outD);
    for (let d = 0; d < outD; d++) {
      const variance = this.sampleCount > 1
        ? this.residualM2[d] / (this.sampleCount - 1)
        : 1;
      baseStdErr[d] = Math.max(1e-12, Math.sqrt(Math.max(0, variance)));
    }

    // Generate predictions with increasing uncertainty
    const predictions: SinglePrediction[] = [];
    for (let step = 0; step < futureSteps; step++) {
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      const uncertaintyScale = Math.sqrt(step + 1);

      for (let d = 0; d < outD; d++) {
        predicted.push(basePrediction[d]);
        const se = baseStdErr[d] * uncertaintyScale;
        standardError.push(se);
        lowerBound.push(basePrediction[d] - 1.96 * se);
        upperBound.push(basePrediction[d] + 1.96 * se);
      }

      predictions.push({ predicted, lowerBound, upperBound, standardError });
    }

    const accuracy = 1 / (1 + this.runningLoss);

    return {
      predictions,
      accuracy,
      sampleCount: this.sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Get model summary
   * @returns ModelSummary with current state
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
      totalParameters: this.totalParams,
      sampleCount: this.sampleCount,
      accuracy: 1 / (1 + this.runningLoss),
      converged: this.converged,
      effectiveLearningRate: this.getLearningRate(),
      driftCount: this.driftCount,
    };
  }

  /**
   * Convert Float64Array to number[]
   */
  private toArray(arr: Float64Array): number[] {
    const result: number[] = [];
    for (let i = 0; i < arr.length; i++) result.push(arr[i]);
    return result;
  }

  /**
   * Get current weights
   * @returns WeightInfo with all weights and moments
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
        updateCount: 0,
      };
    }

    const E = this.config.embeddingDim;
    const K = this.config.temporalKernelSize;
    const inD = this.inputDim;

    // Temporal conv weights: [scale][kernel position][flat weights + bias]
    const temporalConvWeights: number[][][] = [];
    for (let s = 0; s < this.nScales; s++) {
      const scaleWeights: number[][] = [];
      scaleWeights.push(this.toArray(this.convW[s]));
      scaleWeights.push(this.toArray(this.convB[s]));
      temporalConvWeights.push(scaleWeights);
    }

    // Scale embeddings
    const scaleEmbeddings: number[][] = [];
    for (let s = 0; s < this.nScales; s++) {
      scaleEmbeddings.push(this.toArray(this.scaleEmb[s]));
    }

    // Positional encoding
    const positionalEncoding: number[][] = [];
    const maxLen = this.config.maxSequenceLength;
    for (let t = 0; t < maxLen; t++) {
      const row: number[] = [];
      for (let e = 0; e < E; e++) {
        row.push(this.posEnc[t * E + e]);
      }
      positionalEncoding.push(row);
    }

    // Fusion weights
    const fusionWeights: number[][] = [
      this.toArray(this.fusionW),
      this.toArray(this.fusionB),
    ];

    // Attention weights: [block][Wq, Wk, Wv, Wo]
    const attentionWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      attentionWeights.push([
        this.toArray(this.attnWq[b]),
        this.toArray(this.attnWk[b]),
        this.toArray(this.attnWv[b]),
        this.toArray(this.attnWo[b]),
      ]);
    }

    // FFN weights: [block][W1, b1, W2, b2]
    const ffnWeights: number[][][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      ffnWeights.push([
        this.toArray(this.ffnW1[b]),
        this.toArray(this.ffnB1[b]),
        this.toArray(this.ffnW2[b]),
        this.toArray(this.ffnB2[b]),
      ]);
    }

    // LayerNorm params: [block][gamma1, beta1, gamma2, beta2]
    const layerNormParams: number[][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      layerNormParams.push(this.toArray(this.lnG1[b]));
      layerNormParams.push(this.toArray(this.lnB1[b]));
      layerNormParams.push(this.toArray(this.lnG2[b]));
      layerNormParams.push(this.toArray(this.lnB2[b]));
    }

    // Output weights: [poolW, poolB, outW, outB]
    const outputWeights: number[][] = [
      this.toArray(this.poolW),
      this.toArray(this.poolB),
      this.toArray(this.outW),
      this.toArray(this.outB),
    ];

    // First moments
    const firstMoment: number[][][] = [];
    const secondMoment: number[][][] = [];

    // Conv moments
    const convM1: number[][] = [];
    const convM2: number[][] = [];
    for (let s = 0; s < this.nScales; s++) {
      convM1.push(this.toArray(this.mConvW[s]));
      convM2.push(this.toArray(this.vConvW[s]));
    }
    firstMoment.push(convM1);
    secondMoment.push(convM2);

    // Attention moments
    const attnM1: number[][] = [];
    const attnM2: number[][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      attnM1.push(this.toArray(this.mAttnWq[b]));
      attnM2.push(this.toArray(this.vAttnWq[b]));
    }
    firstMoment.push(attnM1);
    secondMoment.push(attnM2);

    // FFN moments
    const ffnM1: number[][] = [];
    const ffnM2: number[][] = [];
    for (let b = 0; b < this.config.numBlocks; b++) {
      ffnM1.push(this.toArray(this.mFfnW1[b]));
      ffnM2.push(this.toArray(this.vFfnW1[b]));
    }
    firstMoment.push(ffnM1);
    secondMoment.push(ffnM2);

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
   * Get normalization statistics
   * @returns NormalizationStats with means and stds
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

    const inputStd = new Float64Array(this.inputDim);
    const outputStd = new Float64Array(this.outputDim);
    this.getStd(this.inputM2, this.sampleCount, inputStd);
    this.getStd(this.outputM2, this.sampleCount, outputStd);

    return {
      inputMean: this.toArray(this.inputMean),
      inputStd: this.toArray(inputStd),
      outputMean: this.toArray(this.outputMean),
      outputStd: this.toArray(outputStd),
      count: this.sampleCount,
    };
  }

  /**
   * Reset model to initial state
   */
  reset(): void {
    if (!this.isInitialized) return;

    this.sampleCount = 0;
    this.updateCount = 0;
    this.driftCount = 0;
    this.runningLoss = 0;
    this.converged = false;
    this.adwinHead = 0;
    this.adwinSize = 0;
    this.cachedWindowLen = 0;

    // Reset normalization stats
    this.inputMean.fill(0);
    this.inputM2.fill(0);
    this.outputMean.fill(0);
    this.outputM2.fill(0);
    this.residualM2.fill(0);

    // Reinitialize weights
    this.rngState = 42;
    const E = this.config.embeddingDim;
    const K = this.config.temporalKernelSize;
    const F = this.ffnDim;
    const inD = this.inputDim;
    const outD = this.outputDim;

    for (let s = 0; s < this.nScales; s++) {
      for (let i = 0; i < this.convW[s].length; i++) {
        this.convW[s][i] = this.xavierUniform(K * inD, E);
      }
      this.convB[s].fill(0);
      for (let i = 0; i < E; i++) {
        this.scaleEmb[s][i] = (this.xorshift() * 2 - 1) * 0.02;
      }
      this.mConvW[s].fill(0);
      this.vConvW[s].fill(0);
      this.mConvB[s].fill(0);
      this.vConvB[s].fill(0);
      this.mScaleEmb[s].fill(0);
      this.vScaleEmb[s].fill(0);
    }

    const fusionDim = this.nScales * E;
    for (let i = 0; i < this.fusionW.length; i++) {
      this.fusionW[i] = this.xavierUniform(fusionDim, fusionDim);
    }
    this.fusionB.fill(0);
    this.mFusionW.fill(0);
    this.vFusionW.fill(0);
    this.mFusionB.fill(0);
    this.vFusionB.fill(0);

    for (let b = 0; b < this.config.numBlocks; b++) {
      for (let i = 0; i < this.attnWq[b].length; i++) {
        this.attnWq[b][i] = this.xavierUniform(E, E);
        this.attnWk[b][i] = this.xavierUniform(E, E);
        this.attnWv[b][i] = this.xavierUniform(E, E);
        this.attnWo[b][i] = this.xavierUniform(E, E);
      }
      for (let i = 0; i < this.ffnW1[b].length; i++) {
        this.ffnW1[b][i] = this.xavierUniform(E, F);
      }
      this.ffnB1[b].fill(0);
      for (let i = 0; i < this.ffnW2[b].length; i++) {
        this.ffnW2[b][i] = this.xavierUniform(F, E);
      }
      this.ffnB2[b].fill(0);
      for (let i = 0; i < E; i++) {
        this.lnG1[b][i] = 1;
        this.lnB1[b][i] = 0;
        this.lnG2[b][i] = 1;
        this.lnB2[b][i] = 0;
      }
      this.mAttnWq[b].fill(0);
      this.mAttnWk[b].fill(0);
      this.mAttnWv[b].fill(0);
      this.mAttnWo[b].fill(0);
      this.vAttnWq[b].fill(0);
      this.vAttnWk[b].fill(0);
      this.vAttnWv[b].fill(0);
      this.vAttnWo[b].fill(0);
      this.mFfnW1[b].fill(0);
      this.mFfnB1[b].fill(0);
      this.mFfnW2[b].fill(0);
      this.mFfnB2[b].fill(0);
      this.vFfnW1[b].fill(0);
      this.vFfnB1[b].fill(0);
      this.vFfnW2[b].fill(0);
      this.vFfnB2[b].fill(0);
      this.mLnG1[b].fill(0);
      this.mLnB1[b].fill(0);
      this.mLnG2[b].fill(0);
      this.mLnB2[b].fill(0);
      this.vLnG1[b].fill(0);
      this.vLnB1[b].fill(0);
      this.vLnG2[b].fill(0);
      this.vLnB2[b].fill(0);
    }

    for (let i = 0; i < E; i++) {
      this.poolW[i] = this.xavierUniform(E, 1);
    }
    this.poolB[0] = 0;
    for (let i = 0; i < this.outW.length; i++) {
      this.outW[i] = this.xavierUniform(E, outD);
    }
    this.outB.fill(0);
    this.mPoolW.fill(0);
    this.vPoolW.fill(0);
    this.mPoolB.fill(0);
    this.vPoolB.fill(0);
    this.mOutW.fill(0);
    this.vOutW.fill(0);
    this.mOutB.fill(0);
    this.vOutB.fill(0);
  }

  /**
   * Serialize model to JSON string
   * @returns JSON string of model state
   */
  save(): string {
    const state = {
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
      rngState: this.rngState,
      adwinHead: this.adwinHead,
      adwinSize: this.adwinSize,
      cachedWindowLen: this.cachedWindowLen,
      weights: this.isInitialized
        ? {
          inputMean: this.toArray(this.inputMean),
          inputM2: this.toArray(this.inputM2),
          outputMean: this.toArray(this.outputMean),
          outputM2: this.toArray(this.outputM2),
          residualM2: this.toArray(this.residualM2),
          cachedWindow: this.toArray(this.cachedWindow),
          adwinBuf: this.toArray(this.adwinBuf),
          convW: this.convW.map((w) => this.toArray(w)),
          convB: this.convB.map((b) => this.toArray(b)),
          scaleEmb: this.scaleEmb.map((e) => this.toArray(e)),
          fusionW: this.toArray(this.fusionW),
          fusionB: this.toArray(this.fusionB),
          attnWq: this.attnWq.map((w) => this.toArray(w)),
          attnWk: this.attnWk.map((w) => this.toArray(w)),
          attnWv: this.attnWv.map((w) => this.toArray(w)),
          attnWo: this.attnWo.map((w) => this.toArray(w)),
          ffnW1: this.ffnW1.map((w) => this.toArray(w)),
          ffnB1: this.ffnB1.map((b) => this.toArray(b)),
          ffnW2: this.ffnW2.map((w) => this.toArray(w)),
          ffnB2: this.ffnB2.map((b) => this.toArray(b)),
          lnG1: this.lnG1.map((g) => this.toArray(g)),
          lnB1: this.lnB1.map((b) => this.toArray(b)),
          lnG2: this.lnG2.map((g) => this.toArray(g)),
          lnB2: this.lnB2.map((b) => this.toArray(b)),
          poolW: this.toArray(this.poolW),
          poolB: this.toArray(this.poolB),
          outW: this.toArray(this.outW),
          outB: this.toArray(this.outB),
          mConvW: this.mConvW.map((m) => this.toArray(m)),
          mConvB: this.mConvB.map((m) => this.toArray(m)),
          mScaleEmb: this.mScaleEmb.map((m) => this.toArray(m)),
          mFusionW: this.toArray(this.mFusionW),
          mFusionB: this.toArray(this.mFusionB),
          mAttnWq: this.mAttnWq.map((m) => this.toArray(m)),
          mAttnWk: this.mAttnWk.map((m) => this.toArray(m)),
          mAttnWv: this.mAttnWv.map((m) => this.toArray(m)),
          mAttnWo: this.mAttnWo.map((m) => this.toArray(m)),
          mFfnW1: this.mFfnW1.map((m) => this.toArray(m)),
          mFfnB1: this.mFfnB1.map((m) => this.toArray(m)),
          mFfnW2: this.mFfnW2.map((m) => this.toArray(m)),
          mFfnB2: this.mFfnB2.map((m) => this.toArray(m)),
          mLnG1: this.mLnG1.map((m) => this.toArray(m)),
          mLnB1: this.mLnB1.map((m) => this.toArray(m)),
          mLnG2: this.mLnG2.map((m) => this.toArray(m)),
          mLnB2: this.mLnB2.map((m) => this.toArray(m)),
          mPoolW: this.toArray(this.mPoolW),
          mPoolB: this.toArray(this.mPoolB),
          mOutW: this.toArray(this.mOutW),
          mOutB: this.toArray(this.mOutB),
          vConvW: this.vConvW.map((v) => this.toArray(v)),
          vConvB: this.vConvB.map((v) => this.toArray(v)),
          vScaleEmb: this.vScaleEmb.map((v) => this.toArray(v)),
          vFusionW: this.toArray(this.vFusionW),
          vFusionB: this.toArray(this.vFusionB),
          vAttnWq: this.vAttnWq.map((v) => this.toArray(v)),
          vAttnWk: this.vAttnWk.map((v) => this.toArray(v)),
          vAttnWv: this.vAttnWv.map((v) => this.toArray(v)),
          vAttnWo: this.vAttnWo.map((v) => this.toArray(v)),
          vFfnW1: this.vFfnW1.map((v) => this.toArray(v)),
          vFfnB1: this.vFfnB1.map((v) => this.toArray(v)),
          vFfnW2: this.vFfnW2.map((v) => this.toArray(v)),
          vFfnB2: this.vFfnB2.map((v) => this.toArray(v)),
          vLnG1: this.vLnG1.map((v) => this.toArray(v)),
          vLnB1: this.vLnB1.map((v) => this.toArray(v)),
          vLnG2: this.vLnG2.map((v) => this.toArray(v)),
          vLnB2: this.vLnB2.map((v) => this.toArray(v)),
          vPoolW: this.toArray(this.vPoolW),
          vPoolB: this.toArray(this.vPoolB),
          vOutW: this.toArray(this.vOutW),
          vOutB: this.toArray(this.vOutB),
        }
        : null,
    };
    return JSON.stringify(state);
  }

  /**
   * Load model from JSON string
   * @param w - JSON string of model state
   */
  load(w: string): void {
    const state = JSON.parse(w);

    this.config = { ...DEFAULT_CONFIG, ...state.config };
    this.inputDim = state.inputDim;
    this.outputDim = state.outputDim;
    this.seqLen = state.seqLen;
    this.sampleCount = state.sampleCount;
    this.updateCount = state.updateCount;
    this.driftCount = state.driftCount;
    this.runningLoss = state.runningLoss;
    this.converged = state.converged;
    this.rngState = state.rngState;
    this.adwinHead = state.adwinHead;
    this.adwinSize = state.adwinSize;
    this.cachedWindowLen = state.cachedWindowLen;
    this.nScales = this.config.temporalScales.length;
    this.headDim = Math.floor(this.config.embeddingDim / this.config.numHeads);
    this.ffnDim = this.config.embeddingDim * this.config.ffnMultiplier;

    if (state.isInitialized && state.weights) {
      this.initialize();
      const wts = state.weights;

      // Restore all arrays
      const copyTo = (src: number[], dst: Float64Array): void => {
        for (let i = 0; i < src.length; i++) dst[i] = src[i];
      };

      copyTo(wts.inputMean, this.inputMean);
      copyTo(wts.inputM2, this.inputM2);
      copyTo(wts.outputMean, this.outputMean);
      copyTo(wts.outputM2, this.outputM2);
      copyTo(wts.residualM2, this.residualM2);
      copyTo(wts.cachedWindow, this.cachedWindow);
      copyTo(wts.adwinBuf, this.adwinBuf);
      copyTo(wts.fusionW, this.fusionW);
      copyTo(wts.fusionB, this.fusionB);
      copyTo(wts.poolW, this.poolW);
      copyTo(wts.poolB, this.poolB);
      copyTo(wts.outW, this.outW);
      copyTo(wts.outB, this.outB);
      copyTo(wts.mFusionW, this.mFusionW);
      copyTo(wts.mFusionB, this.mFusionB);
      copyTo(wts.mPoolW, this.mPoolW);
      copyTo(wts.mPoolB, this.mPoolB);
      copyTo(wts.mOutW, this.mOutW);
      copyTo(wts.mOutB, this.mOutB);
      copyTo(wts.vFusionW, this.vFusionW);
      copyTo(wts.vFusionB, this.vFusionB);
      copyTo(wts.vPoolW, this.vPoolW);
      copyTo(wts.vPoolB, this.vPoolB);
      copyTo(wts.vOutW, this.vOutW);
      copyTo(wts.vOutB, this.vOutB);

      for (let s = 0; s < this.nScales; s++) {
        copyTo(wts.convW[s], this.convW[s]);
        copyTo(wts.convB[s], this.convB[s]);
        copyTo(wts.scaleEmb[s], this.scaleEmb[s]);
        copyTo(wts.mConvW[s], this.mConvW[s]);
        copyTo(wts.mConvB[s], this.mConvB[s]);
        copyTo(wts.mScaleEmb[s], this.mScaleEmb[s]);
        copyTo(wts.vConvW[s], this.vConvW[s]);
        copyTo(wts.vConvB[s], this.vConvB[s]);
        copyTo(wts.vScaleEmb[s], this.vScaleEmb[s]);
      }

      for (let b = 0; b < this.config.numBlocks; b++) {
        copyTo(wts.attnWq[b], this.attnWq[b]);
        copyTo(wts.attnWk[b], this.attnWk[b]);
        copyTo(wts.attnWv[b], this.attnWv[b]);
        copyTo(wts.attnWo[b], this.attnWo[b]);
        copyTo(wts.ffnW1[b], this.ffnW1[b]);
        copyTo(wts.ffnB1[b], this.ffnB1[b]);
        copyTo(wts.ffnW2[b], this.ffnW2[b]);
        copyTo(wts.ffnB2[b], this.ffnB2[b]);
        copyTo(wts.lnG1[b], this.lnG1[b]);
        copyTo(wts.lnB1[b], this.lnB1[b]);
        copyTo(wts.lnG2[b], this.lnG2[b]);
        copyTo(wts.lnB2[b], this.lnB2[b]);
        copyTo(wts.mAttnWq[b], this.mAttnWq[b]);
        copyTo(wts.mAttnWk[b], this.mAttnWk[b]);
        copyTo(wts.mAttnWv[b], this.mAttnWv[b]);
        copyTo(wts.mAttnWo[b], this.mAttnWo[b]);
        copyTo(wts.mFfnW1[b], this.mFfnW1[b]);
        copyTo(wts.mFfnB1[b], this.mFfnB1[b]);
        copyTo(wts.mFfnW2[b], this.mFfnW2[b]);
        copyTo(wts.mFfnB2[b], this.mFfnB2[b]);
        copyTo(wts.mLnG1[b], this.mLnG1[b]);
        copyTo(wts.mLnB1[b], this.mLnB1[b]);
        copyTo(wts.mLnG2[b], this.mLnG2[b]);
        copyTo(wts.mLnB2[b], this.mLnB2[b]);
        copyTo(wts.vAttnWq[b], this.vAttnWq[b]);
        copyTo(wts.vAttnWk[b], this.vAttnWk[b]);
        copyTo(wts.vAttnWv[b], this.vAttnWv[b]);
        copyTo(wts.vAttnWo[b], this.vAttnWo[b]);
        copyTo(wts.vFfnW1[b], this.vFfnW1[b]);
        copyTo(wts.vFfnB1[b], this.vFfnB1[b]);
        copyTo(wts.vFfnW2[b], this.vFfnW2[b]);
        copyTo(wts.vFfnB2[b], this.vFfnB2[b]);
        copyTo(wts.vLnG1[b], this.vLnG1[b]);
        copyTo(wts.vLnB1[b], this.vLnB1[b]);
        copyTo(wts.vLnG2[b], this.vLnG2[b]);
        copyTo(wts.vLnB2[b], this.vLnB2[b]);
      }
    }

    this.isInitialized = state.isInitialized;
  }
}
