📊 Fusion Temporal Transformer Regression

<div align="center">

**A powerful multivariate time-series regression library with incremental online
learning capabilities**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-ft-transformer-regression)
• [🐙 GitHub](https://github.com/hviana/multivariate-ft-transformer-regression)
• [📚 Documentation](#-api-reference)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 API Reference](#-api-reference)
- [⚙️ Configuration Guide](#️-configuration-guide)
- [💡 Examples](#-examples)
- [🎯 Best Practices](#-best-practices)
- [🔧 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Advanced Architecture

- **Multi-scale temporal convolution** for capturing patterns at different time
  scales
- **Gated cross-scale fusion** for intelligent feature combination
- **Transformer blocks** with self-attention mechanism
- **Attention-weighted temporal pooling** for sequence aggregation

</td>
<td width="50%">

### 📈 Online Learning

- **Incremental training** - learn from streaming data
- **Adam optimizer** with warmup & cosine decay
- **ADWIN-lite drift detection** for concept drift
- **Outlier downweighting** for robust training

</td>
</tr>
<tr>
<td width="50%">

### 🔒 Numerical Stability

- **Stable softmax** with max-subtraction
- **LayerNorm** with epsilon protection
- **Welford algorithm** for streaming statistics
- **Causal masking** preventing future leakage

</td>
<td width="50%">

### 📊 Predictions & Monitoring

- **Confidence intervals** with uncertainty estimation
- **Multi-step forecasting** with widening uncertainty
- **Real-time model metrics** and convergence tracking
- **Complete state serialization** for model persistence

</td>
</tr>
</table>

---

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUSION TEMPORAL TRANSFORMER (FTT)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐│
│  │   INPUT     │    │  MULTI-SCALE    │    │     TRANSFORMER STACK       ││
│  │  SEQUENCE   │───▶│  CONVOLUTION    │───▶│                              ││
│  │ [T × D_in]  │    │  + FUSION       │    │  ┌─────────────────────────┐ ││
│  └─────────────┘    └─────────────────┘    │  │ Block 1: LN→MHA→LN→FFN │ ││
│                                             │  ├─────────────────────────┤ ││
│                                             │  │ Block 2: LN→MHA→LN→FFN │ ││
│                                             │  ├─────────────────────────┤ ││
│                                             │  │ Block N: LN→MHA→LN→FFN │ ││
│                                             │  └─────────────────────────┘ ││
│                                             └──────────────────────────────┘│
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐│
│  │   OUTPUT    │    │    OUTPUT       │    │   ATTENTION POOLING          ││
│  │ PREDICTION  │◀───│    LAYER        │◀───│   [T × E] → [E]              ││
│  │  [D_out]    │    │  [E → D_out]    │    │                              ││
│  └─────────────┘    └─────────────────┘    └──────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Scale Temporal Processing

```
INPUT SEQUENCE                    MULTI-SCALE CONVOLUTION
     │                                    │
     │                           ┌────────┼────────┐
     ▼                           ▼        ▼        ▼
┌─────────────┐            ┌─────────┐┌─────────┐┌─────────┐
│ x₁ x₂ x₃...│            │Scale=1  ││Scale=2  ││Scale=4  │
│ [T × D_in] │            │(Fine)   ││(Medium) ││(Coarse) │
└─────────────┘            │stride=1 ││stride=2 ││stride=4 │
                           └────┬────┘└────┬────┘└────┬────┘
                                │          │          │
                                ▼          ▼          ▼
                           ┌──────────────────────────────┐
                           │    GATED CROSS-SCALE FUSION  │
                           │                              │
                           │  g_s = σ(W_g · concat(E) + b)│
                           │  fused = Σ g_s · E_s         │
                           └──────────────────────────────┘
                                         │
                                         ▼
                                   [T × Embed]
```

### Transformer Block Detail

```
          INPUT [T × E]
               │
┌──────────────┼──────────────┐
│              │              │
│         ┌────▼────┐         │
│         │LayerNorm│         │
│         └────┬────┘         │
│              │              │
│    ┌─────────▼─────────┐    │
│    │  Multi-Head Self  │    │
│    │    Attention      │    │
│    │   (Causal Mask)   │    │
│    └─────────┬─────────┘    │
│              │              │
└──────────────┼──────────────┘
          ADD (Residual)
               │
┌──────────────┼──────────────┐
│              │              │
│         ┌────▼────┐         │
│         │LayerNorm│         │
│         └────┬────┘         │
│              │              │
│    ┌─────────▼─────────┐    │
│    │   Feed-Forward    │    │
│    │  (GELU Activation)│    │
│    │  [E → 4E → E]     │    │
│    └─────────┬─────────┘    │
│              │              │
└──────────────┼──────────────┘
          ADD (Residual)
               │
               ▼
         OUTPUT [T × E]
```

### Data Flow Pipeline

```mermaid
graph LR
    A[Raw Input] --> B[Welford Normalization]
    B --> C[Multi-Scale Conv]
    C --> D[Positional Encoding]
    D --> E[Scale Embeddings]
    E --> F[Gated Fusion]
    F --> G[Transformer Blocks]
    G --> H[Attention Pooling]
    H --> I[Output Layer]
    I --> J[Denormalization]
    J --> K[Prediction]
    
    style A fill:#e1f5fe
    style K fill:#c8e6c9
    style F fill:#fff3e0
    style G fill:#fce4ec
```

---

## 📦 Installation

### Using JSR (Recommended)

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";
```

### Using Deno

```typescript
import { FusionTemporalTransformerRegression } from "https://jsr.io/@hviana/multivariate-ft-transformer-regression/mod.ts";
```

---

## 🚀 Quick Start

### Basic Example

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

// Create model with default configuration
const model = new FusionTemporalTransformerRegression();

// Training data: sequence of [feature1, feature2] pairs
const xCoordinates = [
  [1.0, 2.0], // t=0
  [2.0, 3.0], // t=1
  [3.0, 4.0], // t=2
  [4.0, 5.0], // t=3
];

// Target: what we want to predict (uses last row)
const yCoordinates = [
  [0.1], // t=0
  [0.2], // t=1
  [0.25], // t=2
  [0.3], // t=3 ← This is the actual target
];

// Train incrementally
const fitResult = model.fitOnline({ xCoordinates, yCoordinates });

console.log(`📉 Loss: ${fitResult.loss.toFixed(6)}`);
console.log(`📊 Gradient Norm: ${fitResult.gradientNorm.toFixed(6)}`);
console.log(`✅ Converged: ${fitResult.converged}`);

// Make predictions for next 3 steps
const predictions = model.predict(3);

predictions.predictions.forEach((pred, i) => {
  console.log(`Step ${i + 1}:`, {
    predicted: pred.predicted[0].toFixed(4),
    confidence: `[${pred.lowerBound[0].toFixed(4)}, ${
      pred.upperBound[0].toFixed(4)
    }]`,
  });
});
```

### Output

```
📉 Loss: 0.125432
📊 Gradient Norm: 0.034521
✅ Converged: false
Step 1: { predicted: '0.3245', confidence: '[-0.1234, 0.7724]' }
Step 2: { predicted: '0.3512', confidence: '[-0.2156, 0.9180]' }
Step 3: { predicted: '0.3801', confidence: '[-0.3012, 1.0614]' }
```

---

## 📖 API Reference

### Constructor

```typescript
const model = new FusionTemporalTransformerRegression(config?: Partial<FusionTemporalTransformerRegressionConfig>);
```

### Methods

| Method                    | Description                    | Returns              |
| ------------------------- | ------------------------------ | -------------------- |
| `fitOnline(data)`         | Train model on a single sample | `FitResult`          |
| `predict(steps)`          | Generate predictions           | `PredictionResult`   |
| `getModelSummary()`       | Get model information          | `ModelSummary`       |
| `getWeights()`            | Export all weights             | `WeightInfo`         |
| `getNormalizationStats()` | Get normalization statistics   | `NormalizationStats` |
| `reset()`                 | Reset model to initial state   | `void`               |
| `save()`                  | Serialize model to JSON string | `string`             |
| `load(json)`              | Load model from JSON string    | `void`               |

### Type Definitions

#### FitResult

```typescript
interface FitResult {
  loss: number; // Combined MSE + L2 loss
  gradientNorm: number; // L2 norm of all gradients
  effectiveLearningRate: number; // Current LR after scheduling
  isOutlier: boolean; // Whether sample was detected as outlier
  converged: boolean; // Whether gradient norm < threshold
  sampleIndex: number; // Total samples seen
  driftDetected: boolean; // Whether ADWIN detected drift
}
```

#### PredictionResult

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Array of predictions per step
  accuracy: number; // Model accuracy metric (0-1)
  sampleCount: number; // Total training samples
  isModelReady: boolean; // Whether model can predict
}

interface SinglePrediction {
  predicted: number[]; // Predicted values
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard error per output
}
```

#### ModelSummary

```typescript
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
```

---

## ⚙️ Configuration Guide

### Complete Configuration Reference

```typescript
interface FusionTemporalTransformerRegressionConfig {
  // ═══════════════════════════════════════════════════════════════════
  // 🏗️ ARCHITECTURE PARAMETERS
  // ═══════════════════════════════════════════════════════════════════

  numBlocks: number; // Number of transformer blocks
  embeddingDim: number; // Embedding dimension (must be divisible by numHeads)
  numHeads: number; // Number of attention heads
  ffnMultiplier: number; // FFN hidden size = embeddingDim × ffnMultiplier
  temporalScales: number[]; // Convolution stride scales
  temporalKernelSize: number; // Convolution kernel size
  maxSequenceLength: number; // Maximum input sequence length

  // ═══════════════════════════════════════════════════════════════════
  // 📉 OPTIMIZER PARAMETERS
  // ═══════════════════════════════════════════════════════════════════

  learningRate: number; // Base learning rate
  warmupSteps: number; // Steps for linear warmup
  totalSteps: number; // Total steps for cosine decay
  beta1: number; // Adam first moment decay
  beta2: number; // Adam second moment decay
  epsilon: number; // Numerical stability constant

  // ═══════════════════════════════════════════════════════════════════
  // 🎛️ REGULARIZATION & STABILITY
  // ═══════════════════════════════════════════════════════════════════

  regularizationStrength: number; // L2 regularization coefficient
  convergenceThreshold: number; // Gradient norm for convergence
  outlierThreshold: number; // Z-score threshold for outliers
  adwinDelta: number; // ADWIN confidence parameter

  // ═══════════════════════════════════════════════════════════════════
  // 💧 DROPOUT (Currently implemented as 0)
  // ═══════════════════════════════════════════════════════════════════

  attentionDropout: number; // Dropout in attention (reserved)
  fusionDropout: number; // Dropout in fusion (reserved)
}
```

### Parameter Optimization Guide

#### 🏗️ Architecture Parameters

<details>
<summary><b>numBlocks</b> (default: 3)</summary>

**What it controls:** The depth of the transformer stack.

```
┌─────────────────────────────────────────────────────────┐
│  numBlocks = 1        │  numBlocks = 3      │  numBlocks = 6    │
│  ┌─────────┐          │  ┌─────────┐        │  ┌─────────┐      │
│  │ Block 1 │          │  │ Block 1 │        │  │ Block 1 │      │
│  └─────────┘          │  │ Block 2 │        │  │ Block 2 │      │
│                       │  │ Block 3 │        │  │ Block 3 │      │
│  Fast, simple         │  └─────────┘        │  │ Block 4 │      │
│  patterns             │  Balanced           │  │ Block 5 │      │
│                       │                     │  │ Block 6 │      │
│                       │                     │  └─────────┘      │
│                       │                     │  Complex patterns │
└─────────────────────────────────────────────────────────┘
```

**Optimization Guide:**

| Scenario                       | Recommended | Reason                    |
| ------------------------------ | ----------- | ------------------------- |
| Simple linear trends           | 1-2         | Less overfitting          |
| Moderate complexity            | 3           | Good balance              |
| Complex multi-scale patterns   | 4-6         | More representation power |
| Limited data (<1000 samples)   | 1-2         | Prevent overfitting       |
| Abundant data (>10000 samples) | 4-6         | Utilize capacity          |

**Example:**

```typescript
// For stock price prediction with complex patterns
const model = new FusionTemporalTransformerRegression({
  numBlocks: 4,
});

// For simple sensor data
const simpleModel = new FusionTemporalTransformerRegression({
  numBlocks: 2,
});
```

</details>

<details>
<summary><b>embeddingDim</b> (default: 64)</summary>

**What it controls:** The internal representation size throughout the model.

**⚠️ Constraint:** Must be divisible by `numHeads`

```
embeddingDim = 32  →  Lightweight, fast
embeddingDim = 64  →  Balanced (default)
embeddingDim = 128 →  High capacity
embeddingDim = 256 →  Maximum expressiveness
```

**Memory & Compute Impact:**

| embeddingDim | Parameters* | Relative Speed |
| ------------ | ----------- | -------------- |
| 32           | ~50K        | 4× faster      |
| 64           | ~200K       | 1× (baseline)  |
| 128          | ~800K       | 4× slower      |
| 256          | ~3.2M       | 16× slower     |

*Approximate, varies with other settings

**Optimization Guide:**

```typescript
// Real-time applications - prioritize speed
const fastModel = new FusionTemporalTransformerRegression({
  embeddingDim: 32,
  numHeads: 4, // 32/4 = 8 dim per head
});

// High-dimensional input data
const richModel = new FusionTemporalTransformerRegression({
  embeddingDim: 128,
  numHeads: 8, // 128/8 = 16 dim per head
});
```

</details>

<details>
<summary><b>numHeads</b> (default: 8)</summary>

**What it controls:** Number of parallel attention mechanisms.

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION                      │
│                                                              │
│  numHeads = 4                    numHeads = 8                │
│  ┌─────┬─────┬─────┬─────┐      ┌──┬──┬──┬──┬──┬──┬──┬──┐  │
│  │Head1│Head2│Head3│Head4│      │H1│H2│H3│H4│H5│H6│H7│H8│  │
│  └─────┴─────┴─────┴─────┘      └──┴──┴──┴──┴──┴──┴──┴──┘  │
│                                                              │
│  Fewer, broader attention        More, specialized attention │
│  patterns                        patterns                    │
└─────────────────────────────────────────────────────────────┘
```

**Dimension per head:** `d_k = embeddingDim / numHeads`

| embeddingDim | numHeads | d_k | Recommendation |
| ------------ | -------- | --- | -------------- |
| 64           | 4        | 16  | ✅ Good        |
| 64           | 8        | 8   | ✅ Default     |
| 64           | 16       | 4   | ⚠️ Too small   |
| 128          | 8        | 16  | ✅ Good        |
| 128          | 16       | 8   | ✅ Good        |

**Example:**

```typescript
// Recommended: d_k between 8-32
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 64,
  numHeads: 8, // d_k = 8 ✅
});
```

</details>

<details>
<summary><b>temporalScales</b> (default: [1, 2, 4])</summary>

**What it controls:** Multi-resolution temporal analysis.

```
Input Sequence: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]

Scale = 1 (stride=1): Captures every timestep
  │ x₁ │ x₂ │ x₃ │ x₄ │ x₅ │ x₆ │ x₇ │ x₈ │
  
Scale = 2 (stride=2): Captures pairs
  │ x₁,x₂ │ x₃,x₄ │ x₅,x₆ │ x₇,x₈ │
  
Scale = 4 (stride=4): Captures quadruplets
  │ x₁,x₂,x₃,x₄ │ x₅,x₆,x₇,x₈ │

→ Gated Fusion combines all scales
```

**Use Case Examples:**

```typescript
// High-frequency trading (microsecond patterns)
const hftModel = new FusionTemporalTransformerRegression({
  temporalScales: [1, 2, 4, 8], // Fine-grained
  maxSequenceLength: 256,
});

// Daily weather forecasting
const weatherModel = new FusionTemporalTransformerRegression({
  temporalScales: [1, 7, 30], // Daily, weekly, monthly
  maxSequenceLength: 365,
});

// IoT sensor with varying patterns
const iotModel = new FusionTemporalTransformerRegression({
  temporalScales: [1, 3, 6, 12], // Multiple granularities
  maxSequenceLength: 128,
});
```

</details>

<details>
<summary><b>temporalKernelSize</b> (default: 3)</summary>

**What it controls:** The receptive field of temporal convolutions.

```
Kernel Size = 3:          Kernel Size = 5:          Kernel Size = 7:
  ┌─┬─┬─┐                   ┌─┬─┬─┬─┬─┐              ┌─┬─┬─┬─┬─┬─┬─┐
  │◀───▶│                   │◀───────▶│              │◀─────────────▶│
  └─┴─┴─┘                   └─┴─┴─┴─┴─┘              └─┴─┴─┴─┴─┴─┴─┘
  Local patterns            Medium patterns          Wide patterns
```

**Recommendations:**

| Data Type              | Kernel Size | Reason                |
| ---------------------- | ----------- | --------------------- |
| High-frequency signals | 3           | Preserve local detail |
| Medium-frequency       | 5           | Balanced              |
| Low-frequency trends   | 7-9         | Capture wider context |

</details>

<details>
<summary><b>maxSequenceLength</b> (default: 512)</summary>

**What it controls:** Maximum temporal window for processing.

**⚠️ Memory Impact:** Attention is O(T²) in memory!

| maxSequenceLength | Memory (approx.) | Use Case               |
| ----------------- | ---------------- | ---------------------- |
| 64                | ~16 MB           | Real-time, low latency |
| 128               | ~64 MB           | Standard applications  |
| 256               | ~256 MB          | Historical analysis    |
| 512               | ~1 GB            | Long-term patterns     |

```typescript
// Constrained environment
const lightModel = new FusionTemporalTransformerRegression({
  maxSequenceLength: 64,
  embeddingDim: 32,
});

// Server with ample memory
const heavyModel = new FusionTemporalTransformerRegression({
  maxSequenceLength: 512,
  embeddingDim: 128,
});
```

</details>

---

#### 📉 Optimizer Parameters

<details>
<summary><b>learningRate</b> (default: 0.001)</summary>

**What it controls:** Step size for parameter updates.

```
Learning Rate Schedule:
                                
Rate │    ╱──────╲
     │   ╱        ╲
     │  ╱          ╲
     │ ╱            ╲
     │╱              ╲___
     └───────────────────▶
       Warmup   Cosine Decay
       Steps    Phase
```

**Optimization Guide:**

| Scenario                | Learning Rate | Reason                  |
| ----------------------- | ------------- | ----------------------- |
| Fast convergence needed | 0.01          | Quick but may overshoot |
| Standard training       | 0.001         | Balanced (default)      |
| Fine-tuning             | 0.0001        | Gentle updates          |
| Very noisy data         | 0.0005        | More stable             |

```typescript
// Aggressive learning for quick results
const fastLearner = new FusionTemporalTransformerRegression({
  learningRate: 0.005,
  warmupSteps: 50,
});

// Conservative learning for stability
const stableLearner = new FusionTemporalTransformerRegression({
  learningRate: 0.0005,
  warmupSteps: 200,
});
```

</details>

<details>
<summary><b>warmupSteps & totalSteps</b> (defaults: 100, 10000)</summary>

**What they control:** Learning rate scheduling.

```
                    Learning Rate Over Time
                    
    LR │        warmupSteps=100    totalSteps=10000
       │              │                    │
  0.001│    ╱─────────●────────────╲       │
       │   ╱                        ╲      │
       │  ╱                          ╲     │
       │ ╱                            ╲    │
       │╱                              ╲___│
       └───────────────────────────────────▶
         0    100              10000   Steps
         
Formula:
  - Warmup:   lr = base_lr × (step / warmup_steps)
  - Decay:    lr = base_lr × 0.5 × (1 + cos(π × progress))
```

**Configuration Examples:**

```typescript
// Short training session
const quickTrain = new FusionTemporalTransformerRegression({
  warmupSteps: 20,
  totalSteps: 500,
});

// Extended training
const longTrain = new FusionTemporalTransformerRegression({
  warmupSteps: 500,
  totalSteps: 50000,
});
```

</details>

<details>
<summary><b>beta1 & beta2</b> (defaults: 0.9, 0.999)</summary>

**What they control:** Adam optimizer momentum parameters.

```
Adam Update Rule:
  m = β₁·m + (1-β₁)·g          (First moment)
  v = β₂·v + (1-β₂)·g²         (Second moment)
  
  β₁ = 0.9:  90% old momentum, 10% new gradient
  β₂ = 0.999: 99.9% old variance, 0.1% new
```

| Parameter | Effect of Higher Value                 | Effect of Lower Value                |
| --------- | -------------------------------------- | ------------------------------------ |
| beta1     | Smoother updates, slower adaptation    | More responsive, possibly noisy      |
| beta2     | More stable scaling, slower adaptation | Faster adaptation, possibly unstable |

```typescript
// For very noisy gradients
const stableAdam = new FusionTemporalTransformerRegression({
  beta1: 0.95, // More momentum
  beta2: 0.9999, // More stable scaling
});

// For quick adaptation
const adaptiveAdam = new FusionTemporalTransformerRegression({
  beta1: 0.85,
  beta2: 0.99,
});
```

</details>

---

#### 🎛️ Regularization Parameters

<details>
<summary><b>regularizationStrength</b> (default: 1e-4)</summary>

**What it controls:** L2 weight decay penalty.

```
Total Loss = MSE Loss + (λ/2) × Σ||W||²

λ = 0:      No regularization (may overfit)
λ = 1e-4:   Light regularization (default)
λ = 1e-3:   Moderate regularization
λ = 1e-2:   Strong regularization (may underfit)
```

**Visual Effect:**

```
           Low Regularization              High Regularization
           
Weights │  ▓▓▓░░░▓▓░░░▓▓▓░░           │  ▓░░░░▓░░░░░▓░░░░
Distribution │  Many large weights        │  Mostly small weights
           │  Complex model              │  Simpler model
```

```typescript
// Complex patterns, lots of data
const complexModel = new FusionTemporalTransformerRegression({
  regularizationStrength: 1e-5, // Less regularization
});

// Limited data, prevent overfitting
const simpleModel = new FusionTemporalTransformerRegression({
  regularizationStrength: 1e-3, // More regularization
});
```

</details>

<details>
<summary><b>outlierThreshold</b> (default: 3.0)</summary>

**What it controls:** Z-score threshold for outlier detection.

```
                Normal Distribution
                
         │      ●●●●●●●●●●●
         │   ●●●            ●●●
         │ ●●                  ●●
Density  │●                      ●
         │         ┌──────┐
         │      -3σ│ 99.7%│+3σ
         └─────────┼──────┼───────▶
                   ▼      ▼
              outlierThreshold = 3.0
              
Points beyond ±3σ are flagged as outliers
and receive reduced weight (0.1×) during training
```

| Threshold | Coverage | Outlier Sensitivity |
| --------- | -------- | ------------------- |
| 2.0       | 95.4%    | Very sensitive      |
| 2.5       | 98.8%    | Sensitive           |
| 3.0       | 99.7%    | Balanced (default)  |
| 3.5       | 99.95%   | Conservative        |
| 4.0       | 99.99%   | Very conservative   |

```typescript
// Sensor data with occasional spikes
const robustModel = new FusionTemporalTransformerRegression({
  outlierThreshold: 2.5, // More aggressive outlier detection
});

// Clean data, trust all samples
const trustingModel = new FusionTemporalTransformerRegression({
  outlierThreshold: 4.0, // Very conservative
});
```

</details>

<details>
<summary><b>adwinDelta</b> (default: 0.002)</summary>

**What it controls:** Sensitivity of concept drift detection.

```
ADWIN Drift Detection
─────────────────────

     Error Rate
         │
       ╱─│─╲    Drift Detected!
      ╱  │  ╲        │
  ───╱   │   ╲──●────┼────▶
     │   │   │  ↑    │
     │   │   │  Statistical
     │   │   │  significance
     └───┴───┴──test (δ)────
     
δ = 0.002: Sensitive (detects subtle drift)
δ = 0.01:  Moderate
δ = 0.05:  Conservative (only major shifts)
```

```typescript
// Streaming data with frequent distribution shifts
const adaptiveModel = new FusionTemporalTransformerRegression({
  adwinDelta: 0.01, // Quick drift detection
});

// Stable environment
const stableModel = new FusionTemporalTransformerRegression({
  adwinDelta: 0.001, // Very sensitive
});
```

</details>

---

### 📊 Configuration Presets

#### 🚀 High-Performance Preset

```typescript
const highPerformanceConfig = {
  numBlocks: 4,
  embeddingDim: 128,
  numHeads: 8,
  ffnMultiplier: 4,
  temporalScales: [1, 2, 4, 8],
  temporalKernelSize: 5,
  maxSequenceLength: 256,
  learningRate: 0.001,
  warmupSteps: 200,
  totalSteps: 20000,
  regularizationStrength: 1e-4,
};
```

#### ⚡ Real-Time Preset

```typescript
const realTimeConfig = {
  numBlocks: 2,
  embeddingDim: 32,
  numHeads: 4,
  ffnMultiplier: 2,
  temporalScales: [1, 2],
  temporalKernelSize: 3,
  maxSequenceLength: 64,
  learningRate: 0.002,
  warmupSteps: 50,
  totalSteps: 5000,
};
```

#### 🔬 Research Preset

```typescript
const researchConfig = {
  numBlocks: 6,
  embeddingDim: 256,
  numHeads: 16,
  ffnMultiplier: 4,
  temporalScales: [1, 2, 4, 8, 16],
  temporalKernelSize: 7,
  maxSequenceLength: 512,
  learningRate: 0.0005,
  warmupSteps: 500,
  totalSteps: 100000,
  regularizationStrength: 1e-5,
};
```

---

## 💡 Examples

### 📈 Stock Price Prediction

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

// Model optimized for financial data
const stockModel = new FusionTemporalTransformerRegression({
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  temporalScales: [1, 5, 20], // Daily, weekly, monthly patterns
  maxSequenceLength: 252, // One trading year
  outlierThreshold: 2.5, // Financial data has outliers
  adwinDelta: 0.005, // Detect market regime changes
});

// Historical data: [open, high, low, close, volume]
const historicalData = [
  [150.0, 152.5, 149.0, 151.5, 1000000],
  [151.5, 153.0, 150.5, 152.0, 1100000],
  // ... more historical data
];

// Target: next day's closing price
const targets = [
  [151.5],
  [152.0],
  [153.5], // ...
];

// Train on streaming data
for (let i = 20; i < historicalData.length; i++) {
  const window = historicalData.slice(i - 20, i);
  const target = [targets[i]];

  const result = model.fitOnline({
    xCoordinates: window,
    yCoordinates: target,
  });

  if (result.driftDetected) {
    console.log(`⚠️ Market regime change detected at index ${i}`);
  }
}

// Predict next 5 trading days
const forecast = model.predict(5);
forecast.predictions.forEach((pred, day) => {
  console.log(
    `Day ${day + 1}: $${pred.predicted[0].toFixed(2)} ± $${
      pred.standardError[0].toFixed(2)
    }`,
  );
});
```

### 🌡️ Multi-Sensor IoT Monitoring

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

// Model for IoT sensor fusion
const iotModel = new FusionTemporalTransformerRegression({
  numBlocks: 2, // Lightweight for edge deployment
  embeddingDim: 32,
  numHeads: 4,
  temporalScales: [1, 4, 16], // Different sensor update rates
  maxSequenceLength: 64,
  learningRate: 0.002,
  outlierThreshold: 3.0, // Handle sensor glitches
});

// Sensor readings: [temperature, humidity, pressure, light]
const sensorBuffer: number[][] = [];

// Simulated real-time data ingestion
async function processSensorData(reading: number[]) {
  sensorBuffer.push(reading);

  // Keep sliding window
  if (sensorBuffer.length > 64) {
    sensorBuffer.shift();
  }

  // Need minimum data to train
  if (sensorBuffer.length < 10) return;

  // Target: predict temperature
  const target = [[reading[0]]];

  const result = iotModel.fitOnline({
    xCoordinates: sensorBuffer,
    yCoordinates: target,
  });

  // Alert on anomalies
  if (result.isOutlier) {
    console.log(`🚨 Anomaly detected! Sensor reading: ${reading}`);
  }

  // Predict next reading
  const prediction = iotModel.predict(1);
  if (prediction.isModelReady) {
    console.log(
      `📊 Predicted temperature: ${
        prediction.predictions[0].predicted[0].toFixed(1)
      }°C`,
    );
  }
}
```

### 📉 Time Series Forecasting with Uncertainty

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

const forecastModel = new FusionTemporalTransformerRegression({
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  temporalScales: [1, 7, 28], // Daily, weekly, monthly
  maxSequenceLength: 180, // 6 months of daily data
});

// Training loop
const trainingData = generateSyntheticData(1000);

for (const sample of trainingData) {
  const result = forecastModel.fitOnline(sample);

  if (result.sampleIndex % 100 === 0) {
    console.log(`
📊 Training Progress
────────────────────
Sample: ${result.sampleIndex}
Loss: ${result.loss.toFixed(6)}
Learning Rate: ${result.effectiveLearningRate.toFixed(6)}
Gradient Norm: ${result.gradientNorm.toFixed(6)}
Converged: ${result.converged}
    `);
  }
}

// Generate forecast with confidence intervals
const horizon = 14; // 2 weeks
const forecast = forecastModel.predict(horizon);

console.log("\n🔮 14-Day Forecast with 95% Confidence Intervals\n");
console.log("Day  │ Prediction │    95% CI     │ Std Error");
console.log("─────┼────────────┼───────────────┼──────────");

forecast.predictions.forEach((pred, i) => {
  const day = (i + 1).toString().padStart(2);
  const prediction = pred.predicted[0].toFixed(2).padStart(8);
  const lower = pred.lowerBound[0].toFixed(2);
  const upper = pred.upperBound[0].toFixed(2);
  const ci = `[${lower}, ${upper}]`.padStart(13);
  const se = pred.standardError[0].toFixed(3).padStart(7);

  console.log(`  ${day} │  ${prediction} │ ${ci} │  ${se}`);
});
```

### 💾 Model Persistence

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

// Train model
const model = new FusionTemporalTransformerRegression();

// ... training code ...

// Save model
const modelJson = model.save();
await Deno.writeTextFile("model_checkpoint.json", modelJson);

console.log("✅ Model saved successfully");

// Later: Load model
const loadedJson = await Deno.readTextFile("model_checkpoint.json");
const restoredModel = new FusionTemporalTransformerRegression();
restoredModel.load(loadedJson);

console.log("✅ Model restored successfully");

// Verify restoration
const summary = restoredModel.getModelSummary();
console.log(`📊 Restored model has ${summary.totalParameters} parameters`);
console.log(`📈 Training samples: ${summary.sampleCount}`);
```

---

## 🎯 Best Practices

### 1️⃣ Data Preparation

```typescript
// ✅ DO: Provide sufficient sequence length
const goodData = {
  xCoordinates: generateSequence(50), // At least 10+ timesteps
  yCoordinates: generateTargets(50),
};

// ❌ DON'T: Use very short sequences
const badData = {
  xCoordinates: [[1, 2], [3, 4]], // Only 2 timesteps
  yCoordinates: [[5]],
};
```

### 2️⃣ Incremental Training

```typescript
// ✅ DO: Train on streaming data incrementally
for (const sample of dataStream) {
  model.fitOnline(sample);
}

// ✅ DO: Monitor training metrics
const result = model.fitOnline(sample);
if (result.driftDetected) {
  logDriftEvent(result);
}
```

### 3️⃣ Memory Management

```typescript
// ✅ DO: Use appropriate maxSequenceLength
const efficientModel = new FusionTemporalTransformerRegression({
  maxSequenceLength: Math.min(yourDataLength, 256),
});

// ❌ DON'T: Set unnecessarily large maxSequenceLength
const wastefulModel = new FusionTemporalTransformerRegression({
  maxSequenceLength: 10000, // Excessive for most use cases
});
```

### 4️⃣ Hyperparameter Selection

```typescript
// Start with defaults, then tune based on metrics
let bestConfig = { ...defaultConfig };
let bestLoss = Infinity;

for (const config of configCandidates) {
  const model = new FusionTemporalTransformerRegression(config);
  const avgLoss = trainAndEvaluate(model, validationData);

  if (avgLoss < bestLoss) {
    bestLoss = avgLoss;
    bestConfig = config;
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>❌ Error: embeddingDim must be divisible by numHeads</b></summary>

**Problem:** Invalid configuration where `embeddingDim % numHeads !== 0`

**Solution:**

```typescript
// ❌ Wrong
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 50,
  numHeads: 8, // 50 % 8 = 2 ≠ 0
});

// ✅ Correct
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 64, // 64 % 8 = 0 ✓
  numHeads: 8,
});
```

</details>

<details>
<summary><b>⚠️ Model not converging</b></summary>

**Symptoms:** Loss remains high, gradientNorm doesn't decrease

**Solutions:**

1. **Adjust learning rate:**

```typescript
const model = new FusionTemporalTransformerRegression({
  learningRate: 0.0005, // Try lower
  warmupSteps: 200, // Longer warmup
});
```

2. **Add more regularization:**

```typescript
const model = new FusionTemporalTransformerRegression({
  regularizationStrength: 1e-3,
});
```

3. **Check data normalization** - the model normalizes internally, but extreme
   values may cause issues

</details>

<details>
<summary><b>⚠️ Predictions have high uncertainty</b></summary>

**Symptoms:** Wide confidence intervals

**Solutions:**

1. **Train longer:**

```typescript
// Check sample count
const summary = model.getModelSummary();
if (summary.sampleCount < 1000) {
  console.log("Continue training for better predictions");
}
```

2. **Reduce model complexity:**

```typescript
const simplerModel = new FusionTemporalTransformerRegression({
  numBlocks: 2,
  embeddingDim: 32,
});
```

</details>

<details>
<summary><b>⚠️ Frequent drift detection</b></summary>

**Symptoms:** `driftDetected: true` on many samples

**Solutions:**

1. **Adjust ADWIN sensitivity:**

```typescript
const model = new FusionTemporalTransformerRegression({
  adwinDelta: 0.05, // Less sensitive (default: 0.002)
});
```

2. **Check if data genuinely has distribution shifts** - this may be expected
   behavior

</details>

---

## 📄 License

MIT License © 2025 Henrique Emanoel Viana

---

<div align="center">

**[⬆ Back to Top](#-fusion-temporal-transformer-regression)**

Made with ❤️ by [Henrique Emanoel Viana](https://github.com/hviana)

</div>
