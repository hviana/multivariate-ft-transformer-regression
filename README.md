Model: # 🔮 Multivariate Fusion Temporal Transformer Regression

<div align="center">

**A high-performance transformer architecture with multi-scale temporal
processing for online time series regression**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-ft-transformer-regression)
• [🐙 GitHub](https://github.com/hviana/multivariate-ft-transformer-regression)
• [📖 Documentation](#-api-reference)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [📖 API Reference](#-api-reference)
- [🎯 Use Case Examples](#-use-case-examples)
- [🔧 Optimization Guide](#-optimization-guide)
- [💾 Serialization](#-serialization)
- [📊 Performance Tips](#-performance-tips)
- [📜 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Advanced Architecture

- **Multi-scale Temporal Convolutions** - Extract features at different time
  resolutions
- **Cross-scale Gated Attention Fusion** - Intelligent feature combination
- **Transformer Blocks** - Self-attention with temporal bias
- **Attention-weighted Pooling** - Adaptive temporal aggregation

</td>
<td width="50%">

### ⚡ High Performance

- **Online Learning** - Incremental training with Adam optimizer
- **Buffer Pooling** - Memory-efficient computation
- **Cache-friendly Operations** - Optimized matrix operations
- **Float64 Precision** - High numerical accuracy

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Robust Training

- **Z-score Normalization** - Welford's algorithm for running statistics
- **ADWIN Drift Detection** - Automatic concept drift handling
- **Outlier Detection** - Residual-based anomaly filtering
- **L2 Regularization** - Prevents overfitting

</td>
<td width="50%">

### 🔧 Developer Experience

- **TypeScript Native** - Full type safety
- **Zero Dependencies** - Pure implementation
- **Serialization** - Save/load model state
- **Comprehensive API** - Detailed statistics and weights access

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";
```

### Basic Usage

```typescript
// 1. Create model instance
const model = new FusionTemporalTransformerRegression({
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  learningRate: 0.001,
});

// 2. Train incrementally (online learning)
for (const batch of dataStream) {
  const result = model.fitOnline({
    xCoordinates: batch.inputs, // [[x1, x2, ...], [x1, x2, ...], ...]
    yCoordinates: batch.outputs, // [[y1, y2, ...], [y1, y2, ...], ...]
  });

  console.log(`📉 Loss: ${result.loss.toFixed(6)}`);
  console.log(`✅ Converged: ${result.converged}`);
}

// 3. Make predictions
const predictions = model.predict(5); // Predict 5 future steps
predictions.predictions.forEach((pred, i) => {
  console.log(`Step ${i + 1}: ${pred.predicted} ± ${pred.standardError}`);
});
```

---

## 🏗️ Architecture

The Fusion Temporal Transformer combines multiple advanced techniques for time
series processing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FUSION TEMPORAL TRANSFORMER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INPUT SEQUENCE                                │   │
│  │                    [seq_len × input_dim]                            │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 1: MULTI-SCALE TEMPORAL CONVOLUTION              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │  Scale 1    │  │  Scale 2    │  │  Scale 4    │  ...            │   │
│  │  │ (stride=1)  │  │ (stride=2)  │  │ (stride=4)  │                 │   │
│  │  │   Conv1D    │  │   Conv1D    │  │   Conv1D    │                 │   │
│  │  │   + GELU    │  │   + GELU    │  │   + GELU    │                 │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  └─────────┼────────────────┼────────────────┼─────────────────────────┘   │
│            │                │                │                              │
│            ▼                ▼                ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │        STAGE 2: POSITIONAL ENCODING + SCALE EMBEDDINGS              │   │
│  │                                                                      │   │
│  │    F_s = Conv_s(X) + PE(pos) + ScaleEmb_s                           │   │
│  │                                                                      │   │
│  │    PE(pos, 2i)   = sin(pos / 10000^(2i/d))                          │   │
│  │    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                          │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 3: CROSS-SCALE GATED ATTENTION FUSION            │   │
│  │                                                                      │   │
│  │    ┌─────────────────────────────────────────────┐                  │   │
│  │    │  G = σ(Concat(E₁,...,Eₛ) × Wg + bg)         │                  │   │
│  │    │  Fused = Σ(Gₛ ⊙ Eₛ)                         │                  │   │
│  │    └─────────────────────────────────────────────┘                  │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 4: TRANSFORMER BLOCKS (× numBlocks)              │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                                                               │  │   │
│  │  │   ┌──────────┐   ┌─────────────────────────┐   ┌──────────┐  │  │   │
│  │  │   │LayerNorm │──▶│  Multi-Head Attention   │──▶│ Residual │  │  │   │
│  │  │   │    1     │   │  + Temporal Bias        │   │    +     │  │  │   │
│  │  │   └──────────┘   └─────────────────────────┘   └────┬─────┘  │  │   │
│  │  │                                                      │        │  │   │
│  │  │   ┌──────────┐   ┌─────────────────────────┐   ┌────▼─────┐  │  │   │
│  │  │   │LayerNorm │──▶│   Feed-Forward Network  │──▶│ Residual │  │  │   │
│  │  │   │    2     │   │   GELU(xW₁+b₁)W₂+b₂    │   │    +     │  │  │   │
│  │  │   └──────────┘   └─────────────────────────┘   └──────────┘  │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 5: ATTENTION-WEIGHTED TEMPORAL POOLING           │   │
│  │                                                                      │   │
│  │    α = softmax(H × W_pool)                                          │   │
│  │    out = Σ(αᵢ × hᵢ)                                                 │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 6: OUTPUT PROJECTION                             │   │
│  │                                                                      │   │
│  │    ŷ = pooled × W_out + b_out                                       │   │
│  │    [output_dim]                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Head Self-Attention Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-HEAD SELF-ATTENTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Input: X [seq_len × emb_dim]                                 │
│                                                                  │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐                      │
│    │   Wq    │   │   Wk    │   │   Wv    │                      │
│    └────┬────┘   └────┬────┘   └────┬────┘                      │
│         │             │             │                            │
│         ▼             ▼             ▼                            │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐                      │
│    │    Q    │   │    K    │   │    V    │                      │
│    └────┬────┘   └────┬────┘   └────┬────┘                      │
│         │             │             │                            │
│         └──────┬──────┘             │                            │
│                │                    │                            │
│                ▼                    │                            │
│    ┌─────────────────────┐         │                            │
│    │   QKᵀ / √d_k        │         │                            │
│    │   + Temporal Bias   │◄────────┼── Learnable position bias  │
│    └──────────┬──────────┘         │                            │
│               │                    │                            │
│               ▼                    │                            │
│    ┌─────────────────────┐         │                            │
│    │     Softmax         │         │                            │
│    └──────────┬──────────┘         │                            │
│               │                    │                            │
│               └──────────┬─────────┘                            │
│                          │                                       │
│                          ▼                                       │
│               ┌─────────────────────┐                           │
│               │   Attention × V     │                           │
│               └──────────┬──────────┘                           │
│                          │                                       │
│                          ▼                                       │
│               ┌─────────────────────┐                           │
│               │   Output Projection │                           │
│               │        Wo           │                           │
│               └─────────────────────┘                           │
│                                                                  │
│    Formula: Attention(Q,K,V) = softmax(QKᵀ/√d_k + bias) × V    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ONLINE TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐                                                   │
│  │   Input Data      │ xCoordinates: [[x₁, x₂], [x₃, x₄], ...]         │
│  │   (Raw)           │ yCoordinates: [[y₁], [y₂], ...]                  │
│  └─────────┬─────────┘                                                   │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  Welford Update   │──▶│ μ += (x - μ) / n                        │    │
│  │  (Running Stats)  │   │ M₂ += (x - μ_old)(x - μ_new)            │    │
│  └─────────┬─────────┘   └────────────────────────────────────────┘    │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  Z-Score Norm     │──▶│ x_norm = (x - μ) / σ                    │    │
│  │                   │   │ σ = √(M₂ / (n-1))                       │    │
│  └─────────┬─────────┘   └────────────────────────────────────────┘    │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐                                                   │
│  │  Forward Pass     │ Cache activations for backpropagation            │
│  │  (with caching)   │                                                   │
│  └─────────┬─────────┘                                                   │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  Outlier Check    │──▶│ r = |y - ŷ| / σ                         │    │
│  │                   │   │ isOutlier = r > threshold               │    │
│  └─────────┬─────────┘   └────────────────────────────────────────┘    │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  Backward Pass    │──▶│ ∂L/∂W for all weights                   │    │
│  │  (Backprop)       │   │ + L2 regularization gradient            │    │
│  └─────────┬─────────┘   └────────────────────────────────────────┘    │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  Adam Optimizer   │──▶│ m = β₁m + (1-β₁)g                       │    │
│  │                   │   │ v = β₂v + (1-β₂)g²                      │    │
│  │                   │   │ W -= η × m̂ / (√v̂ + ε)                  │    │
│  └─────────┬─────────┘   └────────────────────────────────────────┘    │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐   ┌────────────────────────────────────────┐    │
│  │  ADWIN Drift      │──▶│ |μ₀ - μ₁| ≥ ε_cut → drift detected     │    │
│  │  Detection        │   │ Reset statistics on drift               │    │
│  └───────────────────┘   └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Parameters

### 📐 Architecture Parameters

| Parameter            | Type       | Default     | Description                                                               |
| -------------------- | ---------- | ----------- | ------------------------------------------------------------------------- |
| `numBlocks`          | `number`   | `3`         | Number of transformer encoder blocks                                      |
| `embeddingDim`       | `number`   | `64`        | Internal embedding dimension (must be divisible by `numHeads`)            |
| `numHeads`           | `number`   | `8`         | Number of attention heads                                                 |
| `ffnMultiplier`      | `number`   | `4`         | Multiplier for FFN hidden layer (`ffnDim = embeddingDim × ffnMultiplier`) |
| `temporalScales`     | `number[]` | `[1, 2, 4]` | Stride values for multi-scale temporal convolution                        |
| `temporalKernelSize` | `number`   | `3`         | Kernel size for temporal convolutions                                     |
| `maxSequenceLength`  | `number`   | `512`       | Maximum input sequence length                                             |

<details>
<summary>💡 <b>Architecture Optimization Tips</b></summary>

#### `numBlocks` - Transformer Depth

```typescript
// 🔹 Simple patterns (e.g., linear trends, basic seasonality)
{
  numBlocks: 2;
}

// 🔹 Moderate complexity (e.g., multiple seasonalities, non-linear trends)
{
  numBlocks: 3;
} // DEFAULT

// 🔹 Complex patterns (e.g., long-range dependencies, hierarchical patterns)
{
  numBlocks: 4 - 6;
}

// ⚠️ More blocks = more parameters = longer training time
// ⚠️ Diminishing returns after 4-5 blocks for most tasks
```

#### `embeddingDim` - Representation Capacity

```typescript
// 🔹 Low-dimensional data (< 10 features)
{ embeddingDim: 32, numHeads: 4 }

// 🔹 Medium-dimensional data (10-50 features)
{ embeddingDim: 64, numHeads: 8 }  // DEFAULT

// 🔹 High-dimensional data (50+ features)
{ embeddingDim: 128, numHeads: 8 }

// 🔹 Very complex relationships
{ embeddingDim: 256, numHeads: 16 }

// ⚠️ embeddingDim must be divisible by numHeads
// ⚠️ headDim = embeddingDim / numHeads (ideally >= 8)
```

#### `temporalScales` - Multi-resolution Processing

```typescript
// 🔹 Fine-grained patterns only
{
  temporalScales: [1];
}

// 🔹 Short to medium patterns
{
  temporalScales: [1, 2, 4];
} // DEFAULT

// 🔹 Multi-scale with longer patterns
{
  temporalScales: [1, 2, 4, 8, 16];
}

// 🔹 Focus on longer-term patterns
{
  temporalScales: [2, 4, 8, 16];
}

// Example: Hourly data with daily/weekly patterns
{
  temporalScales: [1, 4, 24, 168];
} // hour, 4-hour, day, week
```

</details>

---

### 📈 Learning Parameters

| Parameter      | Type     | Default | Description                           |
| -------------- | -------- | ------- | ------------------------------------- |
| `learningRate` | `number` | `0.001` | Base learning rate for Adam optimizer |
| `warmupSteps`  | `number` | `100`   | Steps for linear learning rate warmup |
| `totalSteps`   | `number` | `10000` | Total steps for cosine decay schedule |
| `beta1`        | `number` | `0.9`   | Adam first moment decay rate          |
| `beta2`        | `number` | `0.999` | Adam second moment decay rate         |
| `epsilon`      | `number` | `1e-8`  | Numerical stability constant          |

<details>
<summary>💡 <b>Learning Rate Optimization Tips</b></summary>

#### Learning Rate Schedule

```
Learning Rate
     │
  lr │    ╱‾‾‾‾‾‾‾‾‾‾‾‾╲
     │   ╱               ╲
     │  ╱                 ╲
     │ ╱                   ╲
     │╱                     ╲
   0 └──────────────────────────▶ Steps
     │← warmup →│← cosine decay →│
        100          10000
```

```typescript
// 🔹 Stable, slower convergence
{ learningRate: 0.0001, warmupSteps: 200 }

// 🔹 Balanced (DEFAULT)
{ learningRate: 0.001, warmupSteps: 100, totalSteps: 10000 }

// 🔹 Fast initial learning, quick adaptation
{ learningRate: 0.01, warmupSteps: 50 }

// 🔹 Long training, fine convergence
{ learningRate: 0.001, warmupSteps: 500, totalSteps: 50000 }
```

#### Adam Parameters

```typescript
// 🔹 Standard (most cases)
{ beta1: 0.9, beta2: 0.999 }  // DEFAULT

// 🔹 Faster adaptation to new data (streaming)
{ beta1: 0.8, beta2: 0.99 }

// 🔹 More stable updates (noisy data)
{ beta1: 0.95, beta2: 0.9999 }
```

</details>

---

### 🛡️ Regularization & Robustness

| Parameter                | Type     | Default | Description                                     |
| ------------------------ | -------- | ------- | ----------------------------------------------- |
| `regularizationStrength` | `number` | `1e-4`  | L2 weight decay coefficient                     |
| `convergenceThreshold`   | `number` | `1e-6`  | Loss change threshold for convergence detection |
| `outlierThreshold`       | `number` | `3.0`   | Z-score threshold for outlier detection         |
| `adwinDelta`             | `number` | `0.002` | ADWIN drift detection sensitivity               |
| `attentionDropout`       | `number` | `0.0`   | Dropout rate in attention layers                |
| `fusionDropout`          | `number` | `0.0`   | Dropout rate in scale fusion                    |

<details>
<summary>💡 <b>Robustness Optimization Tips</b></summary>

#### Regularization Strength

```typescript
// 🔹 Minimal regularization (clean, large datasets)
{
  regularizationStrength: 1e-6;
}

// 🔹 Light regularization (DEFAULT)
{
  regularizationStrength: 1e-4;
}

// 🔹 Strong regularization (small datasets, overfitting prevention)
{
  regularizationStrength: 1e-3;
}

// 🔹 Very strong (highly noisy data)
{
  regularizationStrength: 1e-2;
}
```

#### Outlier Handling

```typescript
// 🔹 Strict outlier detection (clean data)
{
  outlierThreshold: 2.0;
}

// 🔹 Standard (DEFAULT)
{
  outlierThreshold: 3.0;
} // 99.7% normal distribution

// 🔹 Permissive (heavy-tailed distributions)
{
  outlierThreshold: 4.0;
}

// 🔹 Very permissive (accept most samples)
{
  outlierThreshold: 5.0;
}
```

#### Drift Detection

```typescript
// 🔹 Very sensitive to drift
{
  adwinDelta: 0.01;
}

// 🔹 Standard sensitivity (DEFAULT)
{
  adwinDelta: 0.002;
}

// 🔹 Less sensitive (stable environments)
{
  adwinDelta: 0.0001;
}
```

</details>

---

## 📖 API Reference

### Constructor

```typescript
const model = new FusionTemporalTransformerRegression(config?: FusionTemporalConfig);
```

### Methods

#### `fitOnline(data)` → `FitResult`

Performs a single online learning step.

```typescript
interface FitResult {
  loss: number; // Current training loss
  gradientNorm: number; // L2 norm of gradients
  effectiveLearningRate: number; // Current learning rate (after warmup/decay)
  isOutlier: boolean; // Whether sample was detected as outlier
  converged: boolean; // Whether model has converged
  sampleIndex: number; // Total samples seen
  driftDetected: boolean; // Whether concept drift was detected
}
```

**Example:**

```typescript
const result = model.fitOnline({
  xCoordinates: [
    [1.0, 2.0, 3.0], // timestep 1: 3 features
    [1.1, 2.1, 3.1], // timestep 2: 3 features
    [1.2, 2.2, 3.2], // timestep 3: 3 features
  ],
  yCoordinates: [
    [10.0], // target for timestep 1
    [10.5], // target for timestep 2
    [11.0], // target for timestep 3 (used for training)
  ],
});
```

---

#### `predict(futureSteps)` → `PredictionResult`

Generates predictions with uncertainty estimates.

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Array of predictions
  accuracy: number; // Model accuracy estimate (0-1)
  sampleCount: number; // Training samples seen
  isModelReady: boolean; // Whether model is trained
}

interface SinglePrediction {
  predicted: number[]; // Point predictions
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard error estimates
}
```

**Example:**

```typescript
const predictions = model.predict(3);

predictions.predictions.forEach((pred, step) => {
  console.log(`Step ${step + 1}:`);
  console.log(`  Value: ${pred.predicted[0].toFixed(2)}`);
  console.log(
    `  95% CI: [${pred.lowerBound[0].toFixed(2)}, ${
      pred.upperBound[0].toFixed(2)
    }]`,
  );
});
```

---

#### `getModelSummary()` → `ModelSummary`

Returns comprehensive model information.

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

#### `getNormalizationStats()` → `NormalizationStats`

Returns running normalization statistics.

```typescript
interface NormalizationStats {
  inputMean: number[]; // Running mean for inputs
  inputStd: number[]; // Running std for inputs
  outputMean: number[]; // Running mean for outputs
  outputStd: number[]; // Running std for outputs
  count: number; // Sample count for statistics
}
```

---

#### `getWeights()` → `WeightInfo`

Returns all model weights and optimizer state.

---

#### `save()` → `string`

Serializes model to JSON string.

---

#### `load(json: string)` → `void`

Restores model from JSON string.

---

#### `reset()` → `void`

Resets model to initial state.

---

## 🎯 Use Case Examples

### 📊 Stock Price Prediction

```typescript
const stockModel = new FusionTemporalTransformerRegression({
  numBlocks: 4,
  embeddingDim: 128,
  numHeads: 8,
  temporalScales: [1, 5, 20, 60], // minute, 5-min, 20-min, hour
  learningRate: 0.0005,
  outlierThreshold: 4.0, // Financial data has fat tails
  adwinDelta: 0.005, // Quick regime change detection
});

// Features: [open, high, low, close, volume]
// Target: [next_close]
async function trainOnMarketData(stream: AsyncIterable<MarketData>) {
  for await (const data of stream) {
    const result = stockModel.fitOnline({
      xCoordinates: data.features, // Last N candles
      yCoordinates: data.targets,
    });

    if (result.driftDetected) {
      console.log("⚠️ Market regime change detected!");
    }
  }
}
```

---

### 🌡️ Sensor Data Forecasting

```typescript
const sensorModel = new FusionTemporalTransformerRegression({
  numBlocks: 2,
  embeddingDim: 32,
  numHeads: 4,
  temporalScales: [1, 6, 24, 168], // hour, 6-hour, day, week
  learningRate: 0.001,
  regularizationStrength: 1e-3, // Prevent overfitting on periodic data
  outlierThreshold: 3.5, // Handle sensor glitches
});

// Multiple sensor readings
const sensorData = {
  xCoordinates: [
    [temp1, humidity1, pressure1],
    [temp2, humidity2, pressure2],
    // ... last 24 hours of readings
  ],
  yCoordinates: [
    [temp_target1],
    [temp_target2],
    // ... corresponding targets
  ],
};

const result = sensorModel.fitOnline(sensorData);
const forecast = sensorModel.predict(24); // Forecast next 24 hours
```

---

### 🏭 Industrial Process Control

```typescript
const processModel = new FusionTemporalTransformerRegression({
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  temporalScales: [1, 2, 4, 8],
  learningRate: 0.0001, // Conservative for stability
  warmupSteps: 500,
  convergenceThreshold: 1e-7, // Tight convergence
  regularizationStrength: 1e-4,
});

// Control loop with online adaptation
function controlLoop(measurement: number[]) {
  // Train on new measurement
  processModel.fitOnline({
    xCoordinates: measurementHistory,
    yCoordinates: targetHistory,
  });

  // Predict next setpoint
  const prediction = processModel.predict(1);
  return prediction.predictions[0].predicted;
}
```

---

### 📈 Multi-target Regression

```typescript
// Predict multiple outputs simultaneously
const multiModel = new FusionTemporalTransformerRegression({
  numBlocks: 3,
  embeddingDim: 96,
  numHeads: 8,
});

// Input: 5 features, Output: 3 targets
multiModel.fitOnline({
  xCoordinates: [
    [f1, f2, f3, f4, f5],
    [f1, f2, f3, f4, f5],
    [f1, f2, f3, f4, f5],
  ],
  yCoordinates: [
    [y1, y2, y3],
    [y1, y2, y3],
    [y1, y2, y3],
  ],
});

const predictions = multiModel.predict(1);
console.log("Predicted targets:", predictions.predictions[0].predicted);
// Output: [predicted_y1, predicted_y2, predicted_y3]
```

---

## 🔧 Optimization Guide

### Configuration Presets

```typescript
// 🏃 FAST: Quick training, lower accuracy
const fastConfig = {
  numBlocks: 2,
  embeddingDim: 32,
  numHeads: 4,
  ffnMultiplier: 2,
  learningRate: 0.01,
  warmupSteps: 50,
  temporalScales: [1, 2],
};

// ⚖️ BALANCED: Good tradeoff (DEFAULT-like)
const balancedConfig = {
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  ffnMultiplier: 4,
  learningRate: 0.001,
  warmupSteps: 100,
  temporalScales: [1, 2, 4],
};

// 🎯 ACCURATE: Higher accuracy, slower training
const accurateConfig = {
  numBlocks: 5,
  embeddingDim: 128,
  numHeads: 8,
  ffnMultiplier: 4,
  learningRate: 0.0005,
  warmupSteps: 200,
  totalSteps: 50000,
  temporalScales: [1, 2, 4, 8, 16],
};

// 🌊 STREAMING: Optimized for online/streaming data
const streamingConfig = {
  numBlocks: 2,
  embeddingDim: 48,
  numHeads: 6,
  learningRate: 0.005,
  warmupSteps: 20,
  beta1: 0.8,
  beta2: 0.99,
  adwinDelta: 0.01,
  outlierThreshold: 3.0,
};

// 📉 NOISY DATA: Robust to noise and outliers
const robustConfig = {
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
  regularizationStrength: 1e-3,
  outlierThreshold: 2.5,
  attentionDropout: 0.1,
  fusionDropout: 0.1,
};
```

### Memory Estimation

```
Total Parameters ≈ 
  numScales × (K × inputDim × D + D)           // Temporal conv
  + numScales × D                               // Scale embeddings
  + numScales × D × numScales + numScales      // Fusion
  + 4 × D × D                                   // Cross-scale attention
  + numBlocks × (
      4 × D × D + D                             // Self-attention
      + seqLen²                                 // Temporal bias
      + D × ffnDim + ffnDim + ffnDim × D + D   // FFN
      + 4 × D                                   // LayerNorms
    )
  + D                                           // Pool weights
  + D × outputDim + outputDim                  // Output projection

Example (default config, inputDim=10, outputDim=1, seqLen=100):
≈ 3×(3×10×64+64) + 3×64 + ... ≈ 150,000 parameters
≈ 1.2 MB in Float64
```

---

## 💾 Serialization

### Save and Load Model

```typescript
// Save model state
const modelState = model.save();

// Store in localStorage (browser)
localStorage.setItem("myModel", modelState);

// Or write to file (Deno/Node)
await Deno.writeTextFile("model.json", modelState);

// Later, restore the model
const newModel = new FusionTemporalTransformerRegression({
  // Must match original config!
  numBlocks: 3,
  embeddingDim: 64,
  numHeads: 8,
});

const savedState = localStorage.getItem("myModel");
// Or: const savedState = await Deno.readTextFile('model.json');

newModel.load(savedState!);

// Continue training or make predictions
const predictions = newModel.predict(5);
```

### What's Saved

```
┌─────────────────────────────────────────┐
│           SERIALIZED STATE              │
├─────────────────────────────────────────┤
│ ✅ All model weights                    │
│ ✅ Adam optimizer state (moments)       │
│ ✅ Normalization statistics             │
│ ✅ Training progress (loss, count)      │
│ ✅ Convergence state                    │
│ ✅ ADWIN window                         │
│ ✅ Input history                        │
│ ✅ Configuration                        │
└─────────────────────────────────────────┘
```

---

## 📊 Performance Tips

### ✅ Do's

```typescript
// ✅ Reuse model instance for online learning
const model = new FusionTemporalTransformerRegression();
for (const data of stream) {
  model.fitOnline(data); // Incremental updates
}

// ✅ Use appropriate sequence lengths
{
  maxSequenceLength: Math.ceil(yourTypicalSequence * 1.2);
}

// ✅ Monitor drift detection
if (result.driftDetected) {
  console.log("Consider adjusting strategy");
}

// ✅ Check convergence
if (result.converged && result.loss < threshold) {
  // Model is stable, reduce learning rate or stop training
}

// ✅ Use uncertainty estimates
const pred = model.predict(1);
if (pred.predictions[0].standardError[0] > threshold) {
  console.log("High uncertainty - be cautious");
}
```

### ❌ Don'ts

```typescript
// ❌ Don't create new model for each sample
// BAD:
for (const data of stream) {
  const model = new FusionTemporalTransformerRegression();
  model.fitOnline(data); // Loses all learning!
}

// ❌ Don't use very long sequences without need
// BAD:
{
  maxSequenceLength: 10000;
} // Memory intensive

// ❌ Don't ignore dimension mismatches
// Model will throw error if dimensions change after init

// ❌ Don't use very high learning rates
// BAD:
{
  learningRate: 0.1;
} // May cause instability
```

---

## 📐 Mathematical Foundations

<details>
<summary><b>Click to expand mathematical details</b></summary>

### Z-Score Normalization (Welford's Algorithm)

```
Online mean update:    μₙ = μₙ₋₁ + (xₙ - μₙ₋₁) / n
Online M₂ update:      M₂ₙ = M₂ₙ₋₁ + (xₙ - μₙ₋₁)(xₙ - μₙ)
Standard deviation:    σ = √(M₂ / (n-1))
Z-score:               z = (x - μ) / σ
```

### GELU Activation

```
GELU(x) = x · Φ(x) ≈ x · σ(1.702x)

where Φ is the CDF of standard normal
and σ is the sigmoid function
```

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ

where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)

Attention(Q, K, V) = softmax(QKᵀ/√dₖ + TemporalBias)V
```

### Adam Optimizer

```
mₜ = β₁mₜ₋₁ + (1 - β₁)gₜ           (First moment)
vₜ = β₂vₜ₋₁ + (1 - β₂)gₜ²          (Second moment)
m̂ₜ = mₜ / (1 - β₁ᵗ)                 (Bias correction)
v̂ₜ = vₜ / (1 - β₂ᵗ)                 (Bias correction)
θₜ = θₜ₋₁ - η · m̂ₜ / (√v̂ₜ + ε)     (Update)
```

### ADWIN Drift Detection

```
For window W split at cut point:
  |μ₀ - μ₁| ≥ εcut ⟹ drift detected

where εcut = √((1/2m)ln(4/δ'))
      m = 1/(1/n₀ + 1/n₁)
      δ' = δ/ln(n)
```

### Learning Rate Schedule

```
Warmup (t < T_warmup):
  η(t) = η_base × (t + 1) / T_warmup

Cosine Decay (t ≥ T_warmup):
  progress = (t - T_warmup) / (T_total - T_warmup)
  η(t) = η_base × 0.5 × (1 + cos(π × min(progress, 1)))
```

</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests
on [GitHub](https://github.com/hviana/multivariate-ft-transformer-regression).

---

## 📜 License

MIT License © 2025 [Henrique Emanoel Viana](https://github.com/hviana)

---

<div align="center">

**Made with ❤️ for the time series community**

[⬆ Back to Top](#-multivariate-fusion-temporal-transformer-regression)

</div>
