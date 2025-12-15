> ## ⚠️🚨 IMPORTANT: THIS LIBRARY HAS BEEN DEPRECATED 🚨⚠️
> 
> ---
> 
> ### 🔄 This library has been replaced by a newer, more powerful version!
> 
> <table>
> <tr>
> <td>
> 
> ### ❌ OLD (This Repository)
> `@hviana/multivariate-ft-transformer-regression`
> 
> </td>
> <td>
> 
> ### ✅ NEW (Use This Instead)
> `@hviana/multivariate-convolutional-regression`
> 
> </td>
> </tr>
> </table>
> 
> ---
> 
> ### 📦 Migration Links
> 
> | Platform | Link |
> |----------|------|
> | 🌐 **JSR Registry** | 👉 [https://jsr.io/@hviana/multivariate-convolutional-regression](https://jsr.io/@hviana/multivariate-convolutional-regression) |
> | 🐙 **GitHub Repository** | 👉 [https://github.com/hviana/multivariate-convolutional-regression](https://github.com/hviana/multivariate-convolutional-regression) |
> 
> ---
> 
> ### 🛑 Please migrate to the new library for:
> - ✨ New features and improvements
> - 🐛 Bug fixes and security updates
> - 📚 Better documentation
> - 🔧 Continued maintenance and support
> 
> ---

# 🧠 Multivariate FT Transformer Regression

<div align="center">

**A CPU-optimized Fusion Temporal Transformer for multivariate regression with
incremental online learning**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-ft-transformer-regression)
• [🔗 GitHub](https://github.com/hviana/multivariate-ft-transformer-regression)
• [👤 Author: Henrique Emanoel Viana](https://github.com/hviana)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🚀 Quick Start](#-quick-start)
- [📖 API Reference](#-api-reference)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [🔧 Parameter Optimization Guide](#-parameter-optimization-guide)
- [📊 Use Cases & Examples](#-use-cases--examples)
- [🧮 Mathematical Background](#-mathematical-background)
- [💡 Best Practices](#-best-practices)
- [📜 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Capabilities

- **Multi-scale Temporal Processing** - Captures patterns at different time
  granularities
- **Online Learning** - Incremental training without full dataset reprocessing
- **Multi-step Forecasting** - Predict multiple future time steps
- **Uncertainty Quantification** - Confidence intervals for predictions

</td>
<td width="50%">

### ⚡ Performance Optimizations

- **CPU-Optimized** - Uses Float64Array throughout
- **Zero Intermediate Allocations** - Hot paths avoid GC pressure
- **Memory-Efficient Backprop** - Recomputes attention weights on-the-fly
- **Preallocated Buffers** - Parameters, gradients, and moments reused

</td>
</tr>
<tr>
<td>

### 🛡️ Robustness Features

- **ADWIN Drift Detection** - Automatically detects distribution changes
- **Outlier Downweighting** - Robust to anomalous samples
- **Welford Normalization** - Numerically stable online statistics
- **Gradient Clipping** - Prevents exploding gradients

</td>
<td>

### 📈 Advanced Training

- **Adam Optimizer** - Adaptive learning rate per parameter
- **Warmup + Cosine Schedule** - Smooth learning rate decay
- **L2 Regularization** - Prevents overfitting
- **Convergence Detection** - Automatic training monitoring

</td>
</tr>
</table>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUSION TEMPORAL TRANSFORMER ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌─────────────────────────────────────────────────────┐
│   Input X    │     │              MULTI-SCALE TEMPORAL BRANCH             │
│ (seqLen × D) │────►│                                                      │
└──────────────┘     │  ┌─────────────────────────────────────────────────┐ │
                     │  │         Input Projection (Linear + Bias)        │ │
                     │  │              X → E₀ (seqLen × embDim)            │ │
                     │  └─────────────────────────────────────────────────┘ │
                     │                         │                            │
                     │        ┌────────────────┼────────────────┐           │
                     │        ▼                ▼                ▼           │
                     │  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
                     │  │ Scale 1  │    │ Scale 2  │    │ Scale 4  │  ...  │
                     │  │ Conv+PE  │    │ Conv+PE  │    │ Conv+PE  │       │
                     │  │  +Emb    │    │  +Emb    │    │  +Emb    │       │
                     │  └────┬─────┘    └────┬─────┘    └────┬─────┘       │
                     │       │               │               │             │
                     │       └───────────────┼───────────────┘             │
                     │                       ▼                             │
                     │  ┌─────────────────────────────────────────────────┐│
                     │  │    Softmax Fusion Gate (Learned Combination)    ││
                     │  │         fused = Σ gate[s] × E_s                  ││
                     │  └─────────────────────────────────────────────────┘│
                     └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORMER ENCODER BLOCKS (×N)                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌──────────┐   ┌───────────────────┐   ┌──────────┐   ┌───────────┐  │ │
│  │  │LayerNorm │──►│ Multi-Head Causal │──►│ Residual │──►│LayerNorm  │  │ │
│  │  │          │   │   Self-Attention  │   │  Add     │   │           │  │ │
│  │  └──────────┘   └───────────────────┘   └──────────┘   └─────┬─────┘  │ │
│  │       │                                       ▲               │       │ │
│  │       └───────────────────────────────────────┘               ▼       │ │
│  │                                              ┌─────────────────────┐  │ │
│  │                                              │   FFN (GELU Act.)   │  │ │
│  │                                              │   + Residual Add    │  │ │
│  │                                              └─────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ATTENTION POOLING + OUTPUT                         │
│  ┌─────────────────────┐   ┌─────────────────┐   ┌──────────────────────┐   │
│  │  Attention Pooling  │──►│   Aggregated    │──►│   Output Projection  │   │
│  │  (Learned Weights)  │   │   Embedding     │   │      Y (outDim)      │   │
│  └─────────────────────┘   └─────────────────┘   └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Data Flow Diagram

```
                         TRAINING FLOW
                         
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
│  Raw X  │───►│  Normalize   │───►│   Forward   │───►│   Loss    │
│  Raw Y  │    │  (Welford)   │    │    Pass     │    │  (MSE+L2) │
└─────────┘    └──────────────┘    └─────────────┘    └─────┬─────┘
                                                            │
┌─────────┐    ┌──────────────┐    ┌─────────────┐          │
│  Adam   │◄───│   Clip +     │◄───│  Backward   │◄─────────┘
│ Update  │    │   L2 Grad    │    │    Pass     │
└────┬────┘    └──────────────┘    └─────────────┘
     │
     ▼
┌───────────────────────────────────────────────────────────────────┐
│                        DRIFT DETECTION                             │
│  ┌─────────┐     ┌─────────────────┐     ┌───────────────────┐   │
│  │  ADWIN  │────►│ Drift Detected? │────►│ Reset Optimizer   │   │
│  │ (MSE)   │     │                 │     │ Moments if Yes    │   │
│  └─────────┘     └─────────────────┘     └───────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```typescript
import { FusionTemporalTransformerRegression } from "jsr:@hviana/multivariate-ft-transformer-regression";
```

### Basic Usage

```typescript
// 1️⃣ Initialize the model
const model = new FusionTemporalTransformerRegression({
  embeddingDim: 64,
  numBlocks: 3,
  numHeads: 8,
  learningRate: 0.001,
});

// 2️⃣ Train incrementally with streaming data
for (const sample of dataStream) {
  const result = model.fitOnline({
    xCoordinates: sample.features, // shape: [seqLen, inputDim]
    yCoordinates: sample.targets, // shape: [ySeqLen, outputDim]
  });

  console.log(
    `Loss: ${result.loss.toFixed(4)}, Converged: ${result.converged}`,
  );
}

// 3️⃣ Make predictions
const predictions = model.predict(5); // Predict 5 steps ahead

for (const pred of predictions.predictions) {
  console.log(`Predicted: ${pred.predicted}`);
  console.log(`95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
}
```

---

## 📖 API Reference

### Constructor

```typescript
new FusionTemporalTransformerRegression(config?: FusionTemporalTransformerConfig)
```

Creates a new model instance with optional configuration.

---

### 🔹 `fitOnline(sample: FitOnlineInput): FitResult`

Performs incremental training on a single sequence sample.

#### Input Shape

```typescript
interface FitOnlineInput {
  xCoordinates: number[][]; // [seqLen, inputDim] - Input sequence
  yCoordinates: number[][]; // [ySeqLen, outputDim] - Target values
}
```

> 📌 **Target Rule:** If `ySeqLen === 1`, uses `Y[0]` as target. Otherwise, uses
> `Y[ySeqLen-1]`.

#### Return Value

```typescript
interface FitResult {
  loss: number; // Total loss (weighted MSE + L2)
  gradientNorm: number; // Global gradient L2 norm
  effectiveLearningRate: number; // Current LR after scheduling
  isOutlier: boolean; // Whether sample was flagged as outlier
  converged: boolean; // Whether training has converged
  sampleIndex: number; // Cumulative sample count
  driftDetected: boolean; // Whether ADWIN detected drift
}
```

#### Example

```typescript
const result = model.fitOnline({
  xCoordinates: [
    [1.0, 2.0, 3.0], // timestep 0
    [1.5, 2.5, 3.5], // timestep 1
    [2.0, 3.0, 4.0], // timestep 2
  ],
  yCoordinates: [[10.0, 20.0]], // target output
});

if (result.driftDetected) {
  console.log("⚠️ Distribution drift detected! Optimizer reset.");
}
```

---

### 🔹 `predict(futureSteps: number): PredictionResult`

Generates multi-step predictions using the last-seen input sequence as context.

#### Return Value

```typescript
interface PredictionResult {
  predictions: SinglePrediction[]; // Array of predictions
  accuracy: number; // Model accuracy metric
  sampleCount: number; // Training samples seen
  isModelReady: boolean; // Whether model can predict
}

interface SinglePrediction {
  predicted: number[]; // Point predictions
  lowerBound: number[]; // 95% CI lower bound
  upperBound: number[]; // 95% CI upper bound
  standardError: number[]; // Standard errors
}
```

#### Example

```typescript
// Predict 3 future steps
const result = model.predict(3);

if (result.isModelReady) {
  result.predictions.forEach((pred, step) => {
    console.log(`Step ${step + 1}:`);
    pred.predicted.forEach((val, i) => {
      console.log(
        `  Dim ${i}: ${val.toFixed(2)} ± ${pred.standardError[i].toFixed(2)}`,
      );
    });
  });
}
```

---

### 🔹 `getModelSummary(): ModelSummary`

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
  driftCount: number; // Number of drift events detected
}
```

---

### 🔹 `getNormalizationStats(): NormalizationStats`

Returns current normalization statistics (Welford running statistics).

```typescript
interface NormalizationStats {
  inputMean: number[]; // Per-feature means
  inputStd: number[]; // Per-feature std deviations
  outputMean: number[]; // Per-output means
  outputStd: number[]; // Per-output std deviations
  count: number; // Samples processed
}
```

---

### 🔹 `getWeights(): WeightInfo`

Returns all model weights and optimizer moments for inspection/debugging.

---

### 🔹 `reset(): void`

Resets the model to its initial state, clearing all learned parameters and
statistics.

---

### 🔹 `save(): string`

Serializes the entire model state to a JSON string.

```typescript
const modelState = model.save();
localStorage.setItem("myModel", modelState);
```

---

### 🔹 `load(json: string): void`

Restores model state from a JSON string.

```typescript
const savedState = localStorage.getItem("myModel");
model.load(savedState);
```

---

## ⚙️ Configuration Parameters

### Complete Configuration Reference

```typescript
interface FusionTemporalTransformerConfig {
  // ═══════════════════════════════════════════════════════════
  // 🏗️ ARCHITECTURE PARAMETERS
  // ═══════════════════════════════════════════════════════════

  numBlocks?: number; // Default: 3
  embeddingDim?: number; // Default: 64
  numHeads?: number; // Default: 8
  ffnMultiplier?: number; // Default: 4
  temporalScales?: number[]; // Default: [1, 2, 4]
  temporalKernelSize?: number; // Default: 3
  maxSequenceLength?: number; // Default: 512

  // ═══════════════════════════════════════════════════════════
  // 📈 OPTIMIZER PARAMETERS
  // ═══════════════════════════════════════════════════════════

  learningRate?: number; // Default: 0.001
  warmupSteps?: number; // Default: 100
  totalSteps?: number; // Default: 10000
  beta1?: number; // Default: 0.9
  beta2?: number; // Default: 0.999
  epsilon?: number; // Default: 1e-8

  // ═══════════════════════════════════════════════════════════
  // 🛡️ REGULARIZATION & ROBUSTNESS
  // ═══════════════════════════════════════════════════════════

  attentionDropout?: number; // Default: 0.0
  fusionDropout?: number; // Default: 0.0
  regularizationStrength?: number; // Default: 1e-4
  convergenceThreshold?: number; // Default: 1e-6
  outlierThreshold?: number; // Default: 3.0
  adwinDelta?: number; // Default: 0.002
}
```

---

## 🔧 Parameter Optimization Guide

### 📊 Parameter Decision Tree

```
                    ┌─────────────────────────┐
                    │   What's your priority? │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Accuracy    │      │    Speed      │      │   Robustness  │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                      │                      │
        ▼                      ▼                      ▼
• numBlocks: 4-6       • numBlocks: 2         • outlierThreshold: 2.5
• embeddingDim: 128    • embeddingDim: 32     • adwinDelta: 0.001
• numHeads: 8-16       • numHeads: 4          • regularization: 1e-3
• ffnMultiplier: 4     • ffnMultiplier: 2     • dropout: 0.1-0.2
```

---

### 🏗️ Architecture Parameters

#### `numBlocks` — Transformer Depth

| Value | Use Case                        |  Memory   | Training Speed |
| :---: | :------------------------------ | :-------: | :------------: |
|  1-2  | Simple patterns, fast inference |  🟢 Low   |    🟢 Fast     |
|  3-4  | **Balanced (Recommended)**      | 🟡 Medium |   🟡 Medium    |
|  5-6  | Complex long-range dependencies |  🔴 High  |    🔴 Slow     |

```typescript
// Simple time series with short-term patterns
const simple = new FusionTemporalTransformerRegression({ numBlocks: 2 });

// Complex multivariate with long-range dependencies
const complex = new FusionTemporalTransformerRegression({ numBlocks: 5 });
```

---

#### `embeddingDim` — Model Capacity

> ⚠️ **Must be divisible by `numHeads`**

| Dimension | Parameters | Best For                             |
| :-------: | :--------: | :----------------------------------- |
|    32     |    ~10K    | Low-dimensional inputs, edge devices |
|    64     |    ~40K    | **General purpose (Default)**        |
|    128    |   ~160K    | High-dimensional, complex patterns   |
|    256    |   ~640K    | Very large scale, research           |

```typescript
// Relationship: embeddingDim = numHeads × headDim
// Example: 64 = 8 × 8 (default)

// Low-resource environment
const light = new FusionTemporalTransformerRegression({
  embeddingDim: 32,
  numHeads: 4, // headDim = 8
});

// High-capacity model
const heavy = new FusionTemporalTransformerRegression({
  embeddingDim: 128,
  numHeads: 16, // headDim = 8
});
```

---

#### `temporalScales` — Multi-Resolution Temporal Modeling

```
Scale = 1: [t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇]  → Fine-grained patterns
Scale = 2: [t₀₋₁, t₂₋₃, t₄₋₅, t₆₋₇]          → Medium patterns
Scale = 4: [t₀₋₃, t₄₋₇]                       → Coarse patterns
```

| Configuration      | Use Case                                  |
| :----------------- | :---------------------------------------- |
| `[1]`              | Only short-term dependencies              |
| `[1, 2, 4]`        | **Default - balanced multi-scale**        |
| `[1, 2, 4, 8, 16]` | Long sequences with hierarchical patterns |
| `[1, 3, 7, 14]`    | Weekly patterns (e.g., 7-day cycles)      |

```typescript
// Financial data with daily/weekly/monthly patterns
const financial = new FusionTemporalTransformerRegression({
  temporalScales: [1, 5, 21], // daily, weekly, monthly trading days
  temporalKernelSize: 5,
});

// Sensor data with multiple sampling rates
const sensor = new FusionTemporalTransformerRegression({
  temporalScales: [1, 10, 60, 360], // 1s, 10s, 1min, 1hr (if sampling at 1Hz)
});
```

---

### 📈 Optimizer Parameters

#### Learning Rate Schedule Visualization

```
Learning Rate (lr)
     │
  lr ├────┐
     │    │╲
     │    │ ╲
     │    │  ╲
     │    │   ╲
lr/2 ├────┤    ╲────────
     │    │     ╲      ╲
     │    │      ╲      ╲
   0 └────┴──────────────────►
     0   warmup    total    Steps
          steps    steps
     
     ←──────→←────────────→
      Warmup    Cosine Decay
```

#### `learningRate` — Base Learning Rate

| Value  | When to Use                                   |
| :----: | :-------------------------------------------- |
| 0.0001 | Large batches, fine-tuning, unstable training |
| 0.001  | **Default - most scenarios**                  |
|  0.01  | Small models, quick experiments               |

---

#### `warmupSteps` & `totalSteps` — Schedule Configuration

```typescript
// Short training run
const quick = new FusionTemporalTransformerRegression({
  warmupSteps: 50,
  totalSteps: 1000,
  learningRate: 0.002,
});

// Long production training
const production = new FusionTemporalTransformerRegression({
  warmupSteps: 500,
  totalSteps: 50000,
  learningRate: 0.001,
});
```

**📐 Rule of Thumb:**

- `warmupSteps` ≈ 1-10% of expected total samples
- `totalSteps` ≈ expected total samples (or slightly more)

---

#### `beta1`, `beta2`, `epsilon` — Adam Hyperparameters

| Parameter | Default | Adjustment Scenario                     |
| :-------: | :-----: | :-------------------------------------- |
|  `beta1`  |   0.9   | Lower (0.8) for noisy gradients         |
|  `beta2`  |  0.999  | Lower (0.99) if loss spikes occur       |
| `epsilon` |  1e-8   | Increase (1e-6) for numerical stability |

```typescript
// Noisy online learning environment
const noisy = new FusionTemporalTransformerRegression({
  beta1: 0.85,
  beta2: 0.99,
  epsilon: 1e-6,
});
```

---

### 🛡️ Robustness Parameters

#### `outlierThreshold` — Outlier Detection

Samples with normalized residual > threshold are downweighted to 10%.

```
Residual Norm
     │
     │              ┌──────── Downweighted (weight=0.1)
     │              │
  3σ ├──────────────┤
     │              │
     │   ██████████ │ ████
     │   ██████████ │ ████
     │   ██████████ │ ████
   0 └───██████████─┴─████──►
         Normal      Outliers
```

```typescript
// Highly noisy data
const robust = new FusionTemporalTransformerRegression({
  outlierThreshold: 2.5, // More aggressive outlier detection
});

// Clean data
const clean = new FusionTemporalTransformerRegression({
  outlierThreshold: 4.0, // Less aggressive
});
```

---

#### `adwinDelta` — Drift Detection Sensitivity

| Value  |        Sensitivity        | False Positive Rate |
| :----: | :-----------------------: | :-----------------: |
| 0.0001 |       🟢 Very High        |       🔴 High       |
| 0.002  | 🟡 **Balanced (Default)** |      🟡 Medium      |
|  0.01  |          🔴 Low           |       🟢 Low        |

```typescript
// Rapidly changing environment (e.g., real-time trading)
const reactive = new FusionTemporalTransformerRegression({
  adwinDelta: 0.0005,
});

// Stable environment with occasional regime changes
const stable = new FusionTemporalTransformerRegression({
  adwinDelta: 0.01,
});
```

---

#### `regularizationStrength` — L2 Weight Decay

```typescript
// Prevent overfitting on small datasets
const regularized = new FusionTemporalTransformerRegression({
  regularizationStrength: 0.001, // 10x stronger than default
  attentionDropout: 0.1,
  fusionDropout: 0.1,
});

// Large diverse dataset
const minimal = new FusionTemporalTransformerRegression({
  regularizationStrength: 0.0001,
  attentionDropout: 0.0,
  fusionDropout: 0.0,
});
```

---

## 📊 Use Cases & Examples

### 1️⃣ Stock Price Prediction

```typescript
const stockModel = new FusionTemporalTransformerRegression({
  // Architecture for financial time series
  embeddingDim: 128,
  numBlocks: 4,
  numHeads: 8,
  temporalScales: [1, 5, 21], // Day, Week, Month
  temporalKernelSize: 5,
  maxSequenceLength: 252, // ~1 year trading days

  // Conservative learning for stability
  learningRate: 0.0005,
  warmupSteps: 200,
  totalSteps: 20000,

  // Strong robustness for financial data
  outlierThreshold: 2.5,
  adwinDelta: 0.001,
  regularizationStrength: 0.0005,
});

// Training with OHLCV data
function trainStock(historicalData: StockData[]) {
  for (let i = 60; i < historicalData.length; i++) {
    // Use 60-day window
    const window = historicalData.slice(i - 60, i);

    const xCoordinates = window.map((d) => [
      d.open,
      d.high,
      d.low,
      d.close,
      d.volume,
      d.sma20,
      d.rsi,
      d.macd,
    ]);

    const yCoordinates = [[historicalData[i].close]];

    const result = stockModel.fitOnline({ xCoordinates, yCoordinates });

    if (result.driftDetected) {
      console.log(`📉 Regime change detected at ${historicalData[i].date}`);
    }
  }
}

// Predict next 5 trading days
const forecast = stockModel.predict(5);
console.log("5-day forecast with 95% CI:");
forecast.predictions.forEach((p, day) => {
  console.log(
    `  Day ${day + 1}: $${p.predicted[0].toFixed(2)} ` +
      `[$${p.lowerBound[0].toFixed(2)} - $${p.upperBound[0].toFixed(2)}]`,
  );
});
```

---

### 2️⃣ IoT Sensor Anomaly Detection

```typescript
const sensorModel = new FusionTemporalTransformerRegression({
  // Lightweight for edge deployment
  embeddingDim: 32,
  numBlocks: 2,
  numHeads: 4,
  temporalScales: [1, 10, 60], // 1s, 10s, 1min
  temporalKernelSize: 3,
  maxSequenceLength: 120, // 2 minutes of 1Hz data

  // Fast adaptation
  learningRate: 0.002,
  warmupSteps: 50,
  totalSteps: 5000,

  // Sensitive anomaly detection
  outlierThreshold: 2.0,
  adwinDelta: 0.0005,
  convergenceThreshold: 1e-5,
});

// Streaming sensor data
async function processSensorStream(stream: AsyncIterable<SensorReading>) {
  const buffer: number[][] = [];

  for await (const reading of stream) {
    buffer.push([reading.temperature, reading.humidity, reading.pressure]);

    if (buffer.length > 120) buffer.shift();
    if (buffer.length < 60) continue; // Need minimum context

    const result = sensorModel.fitOnline({
      xCoordinates: buffer,
      yCoordinates: [[reading.temperature, reading.humidity, reading.pressure]],
    });

    // Check for anomalies
    if (result.isOutlier) {
      console.warn(`⚠️ ANOMALY DETECTED at ${reading.timestamp}!`);
      console.warn(
        `  Residual exceeded ${sensorModel.config.outlierThreshold}σ`,
      );
      await sendAlert(reading);
    }

    if (result.driftDetected) {
      console.info(`🔄 Sensor drift detected - model adapting...`);
    }
  }
}
```

---

### 3️⃣ Multi-Target Energy Forecasting

```typescript
const energyModel = new FusionTemporalTransformerRegression({
  // High capacity for multiple outputs
  embeddingDim: 96,
  numBlocks: 4,
  numHeads: 8,
  temporalScales: [1, 4, 24, 168], // Hourly, 4-hour, daily, weekly
  temporalKernelSize: 5,
  maxSequenceLength: 336, // 2 weeks hourly
  ffnMultiplier: 4,

  // Moderate learning
  learningRate: 0.001,
  warmupSteps: 300,
  totalSteps: 30000,

  // Balanced robustness
  outlierThreshold: 3.0,
  adwinDelta: 0.002,
  regularizationStrength: 0.0002,
  attentionDropout: 0.05,
});

// Train on historical energy data
function trainEnergyModel(data: EnergyData[]) {
  for (let i = 168; i < data.length; i++) {
    const window = data.slice(i - 168, i); // 1 week lookback

    const xCoordinates = window.map((d) => [
      d.hour,
      d.dayOfWeek,
      d.temperature,
      d.humidity,
      d.cloudCover,
      d.previousDemand,
      d.isHoliday ? 1 : 0,
      d.industrialActivity,
    ]);

    // Multi-target: grid demand + solar generation + wind generation
    const yCoordinates = [[
      data[i].gridDemand,
      data[i].solarGeneration,
      data[i].windGeneration,
    ]];

    energyModel.fitOnline({ xCoordinates, yCoordinates });
  }
}

// 24-hour ahead forecast
const dayAhead = energyModel.predict(24);

console.log("24-Hour Energy Forecast:");
console.log("Hour | Grid Demand | Solar Gen | Wind Gen");
console.log("-----|-------------|-----------|----------");
dayAhead.predictions.forEach((p, hour) => {
  console.log(
    `${String(hour).padStart(4)} | ` +
      `${p.predicted[0].toFixed(1).padStart(11)} | ` +
      `${p.predicted[1].toFixed(1).padStart(9)} | ` +
      `${p.predicted[2].toFixed(1).padStart(8)}`,
  );
});
```

---

### 4️⃣ Model Persistence Example

```typescript
// Save trained model
function saveModel(model: FusionTemporalTransformerRegression, path: string) {
  const serialized = model.save();
  Deno.writeTextFileSync(path, serialized);
  console.log(`✅ Model saved (${(serialized.length / 1024).toFixed(1)} KB)`);
}

// Load and resume training
function loadModel(path: string): FusionTemporalTransformerRegression {
  const serialized = Deno.readTextFileSync(path);
  const model = new FusionTemporalTransformerRegression();
  model.load(serialized);

  const summary = model.getModelSummary();
  console.log(`✅ Model loaded:`);
  console.log(`   - Parameters: ${summary.totalParameters.toLocaleString()}`);
  console.log(`   - Samples seen: ${summary.sampleCount.toLocaleString()}`);
  console.log(`   - Drift events: ${summary.driftCount}`);

  return model;
}

// Checkpoint during long training
async function trainWithCheckpoints(
  model: FusionTemporalTransformerRegression,
  data: TrainingSample[],
  checkpointInterval: number = 1000,
) {
  for (let i = 0; i < data.length; i++) {
    const result = model.fitOnline(data[i]);

    if ((i + 1) % checkpointInterval === 0) {
      saveModel(model, `./checkpoints/model_${i + 1}.json`);
      console.log(
        `📁 Checkpoint at sample ${i + 1}, loss: ${result.loss.toFixed(6)}`,
      );
    }
  }
}
```

---

## 🧮 Mathematical Background

### Multi-Head Causal Self-Attention

The model uses causal (masked) self-attention to respect temporal ordering:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is the causal mask:
$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{otherwise} \end{cases}$$

### GELU Activation

The Feed-Forward Networks use GELU (Gaussian Error Linear Unit):

$$\text{GELU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

### Welford's Online Algorithm

For numerically stable incremental statistics:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$$

$$M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$$

$$\sigma_n = \sqrt{\frac{M_{2,n}}{n-1}}$$

### ADWIN Drift Detection

ADWIN maintains a window $W$ and detects drift when sub-windows have
significantly different means:

$$|\hat{\mu}_0 - \hat{\mu}_1| \geq \epsilon_{cut} = \sqrt{\frac{1}{2}\left(\frac{1}{n_0} + \frac{1}{n_1}\right)\ln\frac{4}{\delta}}$$

---

## 💡 Best Practices

### ✅ Do's

| Practice                                      | Reason                                |
| :-------------------------------------------- | :------------------------------------ |
| ✅ Normalize your input features              | Model expects similar scale inputs    |
| ✅ Start with defaults, then tune             | Defaults are well-tested              |
| ✅ Monitor `driftDetected`                    | Indicates changing data distributions |
| ✅ Use longer sequences for complex patterns  | More context helps attention          |
| ✅ Save checkpoints during training           | Online learning can't replay data     |
| ✅ Set `totalSteps` based on expected samples | Affects LR schedule                   |

### ❌ Don'ts

| Anti-Pattern                                     | Issue                            |
| :----------------------------------------------- | :------------------------------- |
| ❌ Very high `embeddingDim` with few features    | Overfitting, slow training       |
| ❌ Ignoring `isOutlier` warnings                 | May indicate data quality issues |
| ❌ Setting `outlierThreshold` too low            | Healthy samples get downweighted |
| ❌ Using `temporalScales` >> `maxSequenceLength` | Scales become meaningless        |
| ❌ Extremely low `adwinDelta`                    | Constant false drift detections  |

---

### 🔍 Debugging Checklist

```typescript
function debugModel(model: FusionTemporalTransformerRegression) {
  const summary = model.getModelSummary();
  const stats = model.getNormalizationStats();

  console.log("🔍 MODEL DIAGNOSTICS");
  console.log("═".repeat(50));

  // Check initialization
  console.log(`✓ Initialized: ${summary.isInitialized}`);
  console.log(`✓ Input dims: ${summary.inputDimension}`);
  console.log(`✓ Output dims: ${summary.outputDimension}`);

  // Check training progress
  console.log(`\n📈 Training Progress:`);
  console.log(`  Samples: ${summary.sampleCount}`);
  console.log(`  Accuracy: ${(summary.accuracy * 100).toFixed(2)}%`);
  console.log(`  Converged: ${summary.converged}`);
  console.log(
    `  Current LR: ${summary.effectiveLearningRate.toExponential(2)}`,
  );

  // Check for issues
  console.log(`\n⚠️ Potential Issues:`);

  if (summary.driftCount > summary.sampleCount * 0.1) {
    console.log(`  ⚠ High drift rate (${summary.driftCount} drifts)`);
    console.log(`    → Consider increasing adwinDelta`);
  }

  const stdRange = stats.inputStd.filter((s) => s > 0);
  if (stdRange.length > 0) {
    const minStd = Math.min(...stdRange);
    const maxStd = Math.max(...stdRange);
    if (maxStd / minStd > 100) {
      console.log(
        `  ⚠ Large input scale variance (${minStd.toFixed(4)} to ${
          maxStd.toFixed(4)
        })`,
      );
      console.log(`    → Consider pre-normalizing inputs`);
    }
  }

  console.log("═".repeat(50));
}
```

---

## 📜 License

MIT License © 2025 [Henrique Emanoel Viana](https://github.com/hviana)

---

<div align="center">

**Made with ❤️ for the time series community**

[⬆ Back to Top](#-multivariate-ft-transformer-regression)

</div>
