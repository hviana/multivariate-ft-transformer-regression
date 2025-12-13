# 🚀 Multivariate FT-Transformer Regression

<div align="center">

**A Fusion Temporal Transformer (FTT) for Multivariate Time Series Regression**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-ft-transformer-regression)
• [💻 GitHub](https://github.com/hviana/multivariate-ft-transformer-regression)

_Created by **Henrique Emanoel Viana** • © 2025_

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [📥 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 API Reference](#-api-reference)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [🎯 Parameter Optimization Guide](#-parameter-optimization-guide)
- [📊 Types & Interfaces](#-types--interfaces)
- [💡 Advanced Usage](#-advanced-usage)
- [🔧 Technical Deep Dive](#-technical-deep-dive)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Advanced Architecture

- **Multi-Scale Temporal Convolution** - Captures patterns at different time
  horizons
- **Transformer Blocks** - Self-attention with FFN layers
- **Gated Fusion Mechanism** - Intelligent multi-scale combination
- **Attention Pooling** - Adaptive sequence aggregation

</td>
<td width="50%">

### 🔄 Online Learning

- **Incremental Training** - Learn from streaming data
- **Adam Optimizer** - Adaptive moment estimation
- **Welford's Algorithm** - Online normalization statistics
- **ADWIN Drift Detection** - Automatic distribution change detection

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Robustness

- **Outlier Downweighting** - Resilient to anomalous data
- **L2 Regularization** - Prevents overfitting
- **Gradient Clipping** - Stable training dynamics
- **Confidence Intervals** - Uncertainty quantification

</td>
<td width="50%">

### ⚡ Performance

- **Pure TypeScript** - No external dependencies
- **Float64Array Tensors** - High numerical precision
- **Allocation-Avoiding Hot Paths** - Memory efficient
- **Causal Attention** - Real-time inference ready

</td>
</tr>
</table>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FUSION TEMPORAL TRANSFORMER (FTT)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌───────────────┐                                                             │
│   │   Input Data  │   [Sequence Length × Input Dimensions]                      │
│   │  (Time Series)│                                                             │
│   └───────┬───────┘                                                             │
│           │                                                                     │
│           ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │              📊 Z-Score Normalization (Welford)               │            │
│   │              Online mean/variance computation                  │            │
│   └───────────────────────────────┬───────────────────────────────┘            │
│                                   │                                             │
│           ┌───────────────────────┼───────────────────────┐                    │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│   ┌───────────────┐       ┌───────────────┐       ┌───────────────┐            │
│   │  Scale s=1    │       │  Scale s=2    │       │  Scale s=4    │            │
│   │ ┌───────────┐ │       │ ┌───────────┐ │       │ ┌───────────┐ │            │
│   │ │ Temporal  │ │       │ │ Temporal  │ │       │ │ Temporal  │ │            │
│   │ │   Conv    │ │       │ │   Conv    │ │       │ │   Conv    │ │            │
│   │ │  (GELU)   │ │       │ │  (GELU)   │ │       │ │  (GELU)   │ │            │
│   │ └─────┬─────┘ │       │ └─────┬─────┘ │       │ └─────┬─────┘ │            │
│   │       │       │       │       │       │       │       │       │            │
│   │       ▼       │       │       ▼       │       │       ▼       │            │
│   │  + Pos.Enc.   │       │  + Pos.Enc.   │       │  + Pos.Enc.   │            │
│   │  + Scale Emb. │       │  + Scale Emb. │       │  + Scale Emb. │            │
│   │       │       │       │       │       │       │       │       │            │
│   │       ▼       │       │       ▼       │       │       ▼       │            │
│   │   Upsample    │       │   Upsample    │       │   Upsample    │            │
│   └───────┬───────┘       └───────┬───────┘       └───────┬───────┘            │
│           │                       │                       │                    │
│           └───────────────────────┼───────────────────────┘                    │
│                                   │                                             │
│                                   ▼                                             │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │                    🔀 GATED FUSION LAYER                      │            │
│   │     concat → Wg·x + bg → sigmoid → gate × scale_embeddings    │            │
│   │                      + Fusion Dropout                         │            │
│   └───────────────────────────────┬───────────────────────────────┘            │
│                                   │                                             │
│                                   ▼                                             │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │              🔄 TRANSFORMER BLOCK × numBlocks                 │            │
│   │  ┌─────────────────────────────────────────────────────────┐  │            │
│   │  │  Layer Norm 1                                           │  │            │
│   │  │       ↓                                                 │  │            │
│   │  │  Multi-Head Self-Attention (Causal + Dropout)           │  │            │
│   │  │  Q = XW_q, K = XW_k, V = XW_v → Attention → W_o         │  │            │
│   │  │       ↓                                                 │  │            │
│   │  │  + Residual Connection                                  │  │            │
│   │  │       ↓                                                 │  │            │
│   │  │  Layer Norm 2                                           │  │            │
│   │  │       ↓                                                 │  │            │
│   │  │  Feed-Forward Network: GELU(xW_1 + b_1)W_2 + b_2        │  │            │
│   │  │       ↓                                                 │  │            │
│   │  │  + Residual Connection                                  │  │            │
│   │  └─────────────────────────────────────────────────────────┘  │            │
│   └───────────────────────────────┬───────────────────────────────┘            │
│                                   │                                             │
│                                   ▼                                             │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │                  🎯 ATTENTION POOLING                         │            │
│   │         scores = H·W_pool + b_pool                            │            │
│   │         α = softmax(scores)                                   │            │
│   │         pooled = Σ α_t · H_t                                  │            │
│   └───────────────────────────────┬───────────────────────────────┘            │
│                                   │                                             │
│                                   ▼                                             │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │                    📤 OUTPUT HEAD                             │            │
│   │              ŷ = pooled · W_out + b_out                       │            │
│   └───────────────────────────────┬───────────────────────────────┘            │
│                                   │                                             │
│                                   ▼                                             │
│   ┌───────────────────────────────────────────────────────────────┐            │
│   │                  🔙 DENORMALIZATION                           │            │
│   │            output = ŷ × output_std + output_mean              │            │
│   └───────────────────────────────────────────────────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE TRAINING FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────────┐    │
│  │  Sample  │ →  │  Forward     │ →  │   Loss    │ →  │   Backward    │    │
│  │  Window  │    │   Pass       │    │  (MSE+L2) │    │    Pass       │    │
│  └──────────┘    └──────────────┘    └───────────┘    └───────────────┘    │
│       │                                                       │            │
│       │                                                       ▼            │
│       │         ┌──────────────┐    ┌───────────┐    ┌───────────────┐    │
│       │         │   Welford    │    │  Gradient │    │     Adam      │    │
│       └───────→ │   Stats      │    │   Clip    │ ←  │   Optimizer   │    │
│                 │   Update     │    │  (norm 5) │    │   Update      │    │
│                 └──────────────┘    └───────────┘    └───────────────┘    │
│                                           │                   │            │
│                                           ▼                   │            │
│                 ┌──────────────┐    ┌───────────┐            │            │
│                 │    ADWIN     │ ←  │   Loss    │ ←──────────┘            │
│                 │    Drift     │    │  Monitor  │                          │
│                 │  Detection   │    │           │                          │
│                 └──────────────┘    └───────────┘                          │
│                       │                                                     │
│                       ▼                                                     │
│              Drift Detected? ──Yes──→ Reset ADWIN buffer                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📥 Installation

### Deno / JSR

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";
```

### Alternative Import (using alias)

```typescript
import { FusionTemporalTransformerRegression as ConvolutionalRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";
```

---

## 🚀 Quick Start

### Basic Example

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";

// 1️⃣ Create model with default configuration
const model = new FusionTemporalTransformerRegression();

// 2️⃣ Prepare your time series data
// xCoordinates: Input features over time [timesteps × features]
// yCoordinates: Target values over time [timesteps × targets]
const trainingData = {
  xCoordinates: [
    [1.0, 2.0, 3.0], // timestep 0: 3 input features
    [1.1, 2.1, 3.1], // timestep 1
    [1.2, 2.2, 3.2], // timestep 2
    [1.3, 2.3, 3.3], // timestep 3
    [1.4, 2.4, 3.4], // timestep 4
  ],
  yCoordinates: [
    [10.0, 20.0], // timestep 0: 2 target values
    [10.5, 20.5], // timestep 1
    [11.0, 21.0], // timestep 2
    [11.5, 21.5], // timestep 3
    [12.0, 22.0], // timestep 4 (training target)
  ],
};

// 3️⃣ Train online (one sample at a time)
const result = model.fitOnline(trainingData);

console.log("Training Result:", {
  loss: result.loss,
  converged: result.converged,
  isOutlier: result.isOutlier,
  driftDetected: result.driftDetected,
});

// 4️⃣ Make predictions
const predictions = model.predict(3); // Predict 3 future steps

for (let i = 0; i < predictions.predictions.length; i++) {
  const p = predictions.predictions[i];
  console.log(`Step ${i + 1}:`, {
    predicted: p.predicted,
    confidence95: {
      lower: p.lowerBound,
      upper: p.upperBound,
    },
  });
}

// 5️⃣ Check model summary
console.log("Model Summary:", model.getModelSummary());
```

### Streaming Data Example

```typescript
// For continuous streaming data
async function streamTraining(model: FusionTemporalTransformerRegression) {
  const windowSize = 50;
  const buffer: { x: number[]; y: number[] }[] = [];

  for await (const dataPoint of getDataStream()) {
    buffer.push(dataPoint);

    // Maintain sliding window
    if (buffer.length > windowSize) {
      buffer.shift();
    }

    // Train when we have enough data
    if (buffer.length >= windowSize) {
      const result = model.fitOnline({
        xCoordinates: buffer.map((d) => d.x),
        yCoordinates: buffer.map((d) => d.y),
      });

      // React to drift
      if (result.driftDetected) {
        console.log(
          "⚠️ Distribution drift detected at sample",
          result.sampleIndex,
        );
      }
    }
  }
}
```

---

## 📖 API Reference

### Constructor

```typescript
constructor(config?: Partial<Config>)
```

Creates a new FusionTemporalTransformerRegression instance.

| Parameter | Type              | Description                      |
| --------- | ----------------- | -------------------------------- |
| `config`  | `Partial<Config>` | Optional configuration overrides |

### Methods

#### `fitOnline(data)`

Trains the model on a single sample window.

```typescript
fitOnline(data: { 
  xCoordinates: number[][];  // [seqLen, inputDim]
  yCoordinates: number[][]   // [seqLen, outputDim]
}): FitResult
```

> 📝 **Note**: The target used for training is the **last timestep** of
> `yCoordinates`.

---

#### `predict(futureSteps)`

Generates predictions for future timesteps.

```typescript
predict(futureSteps: number): PredictionResult
```

> 📝 **Note**: Uses the cached last window from `fitOnline()`. Uncertainty
> increases with `√(step+1)`.

---

#### `getModelSummary()`

Returns comprehensive model statistics.

```typescript
getModelSummary(): ModelSummary
```

---

#### `getWeights()`

Returns all model weights and optimizer moments.

```typescript
getWeights(): WeightInfo
```

---

#### `getNormalizationStats()`

Returns current normalization statistics.

```typescript
getNormalizationStats(): NormalizationStats
```

---

#### `reset()`

Reinitializes all weights and counters while keeping configuration.

```typescript
reset(): void
```

---

#### `save()`

Serializes the complete model state to JSON.

```typescript
save(): string
```

---

#### `load(json)`

Restores model from serialized JSON string.

```typescript
load(json: string): void
```

---

## ⚙️ Configuration Parameters

### Complete Configuration Reference

```typescript
interface Config {
  // 🏗️ Architecture
  numBlocks: number; // Transformer blocks (default: 3)
  embeddingDim: number; // Embedding dimension (default: 64)
  numHeads: number; // Attention heads (default: 8)
  ffnMultiplier: number; // FFN hidden multiplier (default: 4)

  // 🎲 Regularization
  attentionDropout: number; // Attention dropout (default: 0.0)
  fusionDropout: number; // Fusion layer dropout (default: 0.0)
  regularizationStrength: number; // L2 lambda (default: 1e-4)

  // 📈 Optimizer
  learningRate: number; // Base learning rate (default: 0.001)
  warmupSteps: number; // LR warmup steps (default: 100)
  totalSteps: number; // Total steps for scheduling (default: 10000)
  beta1: number; // Adam beta1 (default: 0.9)
  beta2: number; // Adam beta2 (default: 0.999)
  epsilon: number; // Adam epsilon (default: 1e-8)

  // 🎯 Training Behavior
  convergenceThreshold: number; // Gradient norm threshold (default: 1e-6)
  outlierThreshold: number; // Z-score outlier threshold (default: 3.0)
  adwinDelta: number; // ADWIN sensitivity (default: 0.002)

  // 📊 Temporal Processing
  temporalScales: number[]; // Multi-scale factors (default: [1, 2, 4])
  temporalKernelSize: number; // Convolution kernel size (default: 3)
  maxSequenceLength: number; // Max input sequence length (default: 512)
}
```

---

## 🎯 Parameter Optimization Guide

### 📊 By Data Characteristics

<details>
<summary><b>📈 High-Frequency Data (Minute/Second granularity)</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  temporalScales: [1, 2, 4, 8, 16], // More scales for fine patterns
  temporalKernelSize: 5, // Larger kernel for local patterns
  embeddingDim: 128, // Higher capacity
  numHeads: 8,
  maxSequenceLength: 1024, // Longer sequences
  learningRate: 0.0005, // Lower LR for stability
  outlierThreshold: 4.0, // More tolerant of spikes
});
```

**Why?**

- Multiple temporal scales capture both micro and macro patterns
- Larger kernel size smooths high-frequency noise
- Higher embedding dimension captures complex relationships

</details>

<details>
<summary><b>📉 Low-Frequency Data (Daily/Weekly granularity)</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  temporalScales: [1, 2], // Fewer scales needed
  temporalKernelSize: 3, // Smaller kernel
  embeddingDim: 32, // Lower capacity sufficient
  numBlocks: 2, // Fewer transformer blocks
  maxSequenceLength: 128, // Shorter sequences
  learningRate: 0.002, // Higher LR for faster learning
  warmupSteps: 50, // Shorter warmup
});
```

**Why?**

- Simpler patterns require less model capacity
- Faster convergence with higher learning rate
- Memory efficient with smaller architecture

</details>

<details>
<summary><b>🔀 Multi-Variate with Many Features (>50 inputs)</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 256, // Much higher capacity
  numHeads: 16, // More attention heads
  numBlocks: 4, // Deeper network
  ffnMultiplier: 4, // Standard FFN size
  regularizationStrength: 1e-3, // Stronger regularization
  attentionDropout: 0.1, // Moderate dropout
  fusionDropout: 0.1,
  learningRate: 0.0003, // Lower LR for stability
});
```

**Why?**

- High-dimensional inputs need higher embedding dimensions
- More attention heads capture diverse feature relationships
- Stronger regularization prevents overfitting on many features

</details>

<details>
<summary><b>🎯 Simple Univariate Series</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 16, // Minimal capacity
  numHeads: 2, // Few heads
  numBlocks: 1, // Single block
  temporalScales: [1, 2], // Two scales
  temporalKernelSize: 3,
  regularizationStrength: 1e-5, // Light regularization
  learningRate: 0.003, // Faster learning
});
```

**Why?**

- Simple patterns don't need complex architecture
- Fast training and inference
- Minimal memory footprint

</details>

---

### 🎮 By Use Case

<details>
<summary><b>🌊 Concept Drift / Non-Stationary Data</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  adwinDelta: 0.001, // More sensitive drift detection
  outlierThreshold: 2.5, // Stricter outlier detection
  learningRate: 0.002, // Higher LR for adaptation
  warmupSteps: 20, // Quick warmup
  totalSteps: 5000, // Shorter schedule cycle
  regularizationStrength: 5e-5, // Light regularization for flexibility
});

// Monitor drift in training loop
const result = model.fitOnline(data);
if (result.driftDetected) {
  console.log("Drift detected! Model adapting...");
  // Optionally adjust learning rate or other parameters
}
```

**Why?**

- Lower `adwinDelta` = more sensitive drift detection
- Higher learning rate enables faster adaptation
- Light regularization allows model to change quickly

</details>

<details>
<summary><b>🔧 Stable Production Environment</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  adwinDelta: 0.01, // Less sensitive drift detection
  outlierThreshold: 3.5, // More tolerant of anomalies
  learningRate: 0.0005, // Conservative learning
  warmupSteps: 200, // Gradual warmup
  totalSteps: 50000, // Long schedule
  regularizationStrength: 1e-4, // Standard regularization
  attentionDropout: 0.0, // No dropout for determinism
  fusionDropout: 0.0,
});
```

**Why?**

- Stable learning prevents sudden model changes
- Long schedule maintains consistent performance
- No dropout ensures deterministic inference

</details>

<details>
<summary><b>⚡ Real-Time / Low-Latency Requirements</b></summary>

```typescript
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 32, // Smaller embeddings
  numHeads: 4, // Fewer heads
  numBlocks: 2, // Fewer layers
  temporalScales: [1, 2], // Minimal scales
  maxSequenceLength: 64, // Short sequences
  attentionDropout: 0.0, // No dropout overhead
  fusionDropout: 0.0,
});
```

**Why?**

- Minimal architecture = fastest inference
- Short sequences reduce memory and computation
- No dropout eliminates random number generation overhead

</details>

---

### 📊 Parameter Impact Matrix

| Parameter                  | ↑ Accuracy | ↑ Speed | ↓ Memory | Drift Adapt |
| -------------------------- | :--------: | :-----: | :------: | :---------: |
| `embeddingDim ↑`           |     ✅     |   ❌    |    ❌    |     ➖      |
| `numBlocks ↑`              |     ✅     |   ❌    |    ❌    |     ➖      |
| `numHeads ↑`               |     ✅     |   ❌    |    ➖    |     ➖      |
| `temporalScales ↑`         |     ✅     |   ❌    |    ❌    |     ✅      |
| `learningRate ↑`           |     ❌     |   ✅    |    ➖    |     ✅      |
| `regularizationStrength ↑` |     ➖     |   ➖    |    ➖    |     ❌      |
| `adwinDelta ↓`             |     ➖     |   ➖    |    ➖    |     ✅      |
| `maxSequenceLength ↓`      |     ❌     |   ✅    |    ✅    |     ➖      |
| `dropout ↑`                |     ✅     |   ❌    |    ➖    |     ➖      |

**Legend:** ✅ Improves | ❌ Degrades | ➖ Neutral

---

## 📊 Types & Interfaces

### FitResult

```typescript
type FitResult = {
  loss: number; // Current training loss
  gradientNorm: number; // L2 norm of gradients
  effectiveLearningRate: number; // Actual LR after scheduling
  isOutlier: boolean; // Whether sample was flagged as outlier
  converged: boolean; // Whether gradient norm < threshold
  sampleIndex: number; // Total samples trained
  driftDetected: boolean; // ADWIN drift signal
};
```

### SinglePrediction

```typescript
type SinglePrediction = {
  predicted: number[]; // Point predictions
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard error per dimension
};
```

### PredictionResult

```typescript
type PredictionResult = {
  predictions: SinglePrediction[]; // Array of predictions per step
  accuracy: number; // Model accuracy: 1/(1+runningLoss)
  sampleCount: number; // Total training samples
  isModelReady: boolean; // Whether model can predict
};
```

### ModelSummary

```typescript
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
  driftCount: number; // Number of detected drifts
};
```

### NormalizationStats

```typescript
type NormalizationStats = {
  inputMean: number[]; // Running mean per input feature
  inputStd: number[]; // Running std per input feature
  outputMean: number[]; // Running mean per output dimension
  outputStd: number[]; // Running std per output dimension
  count: number; // Number of samples processed
};
```

---

## 💡 Advanced Usage

### Model Persistence

```typescript
// Save model
const modelJson = model.save();
localStorage.setItem("myModel", modelJson); // Browser
await Deno.writeTextFile("model.json", modelJson); // Deno

// Load model
const loadedJson = await Deno.readTextFile("model.json");
const newModel = new FusionTemporalTransformerRegression();
newModel.load(loadedJson);

// Continue training
newModel.fitOnline(newData);
```

### Custom Learning Rate Scheduling

The model uses a **warmup + cosine annealing** schedule:

```
LR(t) = {
  base_lr × (t / warmup)                           if t < warmup
  base_lr × 0.5 × (1 + cos(π × progress))          otherwise
}

where progress = (t - warmup) / (total - warmup)
```

```typescript
// Configure for your training budget
const model = new FusionTemporalTransformerRegression({
  learningRate: 0.001,
  warmupSteps: 100, // First 100 steps ramp up
  totalSteps: 10000, // Full cosine cycle over 10000 steps
});
```

### Handling Outliers

```typescript
const result = model.fitOnline(data);

if (result.isOutlier) {
  // Sample was downweighted (0.1 × normal weight)
  console.log("Outlier detected - sample downweighted");

  // You might want to:
  // - Log for investigation
  // - Adjust outlierThreshold if too many false positives
  // - Skip this sample entirely in some cases
}
```

### Uncertainty Quantification

```typescript
const predictions = model.predict(5);

for (const pred of predictions.predictions) {
  // 95% confidence interval
  const ci95 = pred.upperBound.map((u, i) => u - pred.lowerBound[i]);

  // Check if uncertainty is acceptable
  const maxUncertainty = Math.max(...pred.standardError);
  if (maxUncertainty > threshold) {
    console.warn("High uncertainty in prediction!");
  }
}
```

---

## 🔧 Technical Deep Dive

### 🧮 Key Algorithms

#### GELU Activation

```
GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

Smooth activation that combines dropout, zoneout, and ReLU properties.

#### Welford's Online Algorithm

```
For each new sample x:
  n = n + 1
  δ = x - mean
  mean = mean + δ/n
  M2 = M2 + δ × (x - mean)
  
variance = M2 / (n - 1)
std = √variance
```

Numerically stable online variance computation.

#### ADWIN Drift Detection

Maintains a sliding window of recent losses and detects when the statistical
properties change significantly. Uses Hoeffding bound:

```
|μ₁ - μ₂| > ε = √(2 × ln(2/δ) × (1/n₁ + 1/n₂))
```

#### Xavier Initialization

```
W ~ Uniform(-limit, limit)
limit = √(6 / (fan_in + fan_out))
```

Prevents vanishing/exploding gradients in deep networks.

### 🔄 Memory Layout

All tensors use `Float64Array` in row-major order:

| Tensor    | Shape        | Memory Layout               |
| --------- | ------------ | --------------------------- |
| Input X   | `[L, D]`     | `X[t][d] = X[t * D + d]`    |
| Weight W  | `[in, out]`  | `W[i][j] = W[i * out + j]`  |
| Attention | `[L, H, dk]` | Head h at `offset = h * dk` |

### ⚡ Performance Optimizations

1. **Pre-allocated buffers** - All forward/backward buffers allocated once
2. **In-place operations** - Gradients accumulated without allocations
3. **Fused operations** - Combined normalization and scaling
4. **Causal masking** - Efficient attention with `-1e9` masking
5. **Deterministic dropout** - XorShift32 RNG for reproducibility

---

## 📜 License

MIT License © 2025 Henrique Emanoel Viana

---

<div align="center">

**[⬆ Back to Top](#-multivariate-ft-transformer-regression)**

Made with ❤️ for the time series community

</div>
