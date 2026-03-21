# Adaptive-Router-Attention-for-Efficient-Transformer-Inference

📌 Problem Statement

Transformer models rely on full self-attention, which has:

[
\mathcal{O}(n^2 \cdot d)
]

❗ Core Issues
Every token attends to every other token
Leads to:
High inference latency
Quadratic FLOPs growth
Poor scalability for long sequences

## Introduction

The Adaptive Router Attention (ARA) framework is a novel approach designed to improve the efficiency of transformer inference by reducing unnecessary global attention computations. Traditional transformer models rely on full self-attention, which scales quadratically with sequence length, making them inefficient for long-context tasks.

ARA introduces a token-wise adaptive routing mechanism that dynamically determines whether each token requires expensive global attention or can be processed using a cheaper local attention pathway. This enables significant reductions in latency and FLOPs while preserving model expressiveness.

This repository implements:

- Adaptive-Router Transformer (ART): Efficient transformer with dynamic attention routing
- Dual-Path Attention Module: Combines full and local attention paths
- Routing Network (MLP-based): Learns importance-aware token selection

## Model Overview

Adaptive Router Attention reformulates the standard attention pipeline by introducing conditional computation at the token level.

### Key Components

- Token Statistics Extractor
Computes:
    - Magnitude: ∥𝑥𝑖∥ 
    - Variance: Var (𝑥𝑖)
    - Context deviation: ∥xi​−μ∥

- Routing Network (MLP + Sigmoid)
- 
Produces a routing score:
       gi​∈[0,1] 

- Dual Attention Paths
     - Full Attention (Expensive, Global Context)
     - Local Attention (Cheap, Windowed Context)
- Adaptive Fusion Mechanism
      - Combines both paths dynamically per token.
  
## Architecture Overview

1. Training Architecture (Learning Phase)

During training, the model does both types of attention for every token so it can learn properly.

How it works
 - Input Embeddings: 
      Tokens are converted into vector representations.
 - Feature Extraction
      The model analyzes each token (its importance, variation, and difference from others).
- Router Module (MLP)
      A small neural network predicts a score for each token:
      “How important is this token?”
- Parallel Attention
      Full Attention: looks at all tokens (expensive but powerful)
      Local Attention: looks at nearby tokens (cheap but limited)
- Soft Combination
      The model blends both outputs using the router score
      Every token gets a mix of global + local information

![train_page-0001](https://github.com/user-attachments/assets/6cb7e90e-1619-4993-b4e6-71d055dcc94a)

2. Inference Architecture (Efficient Phase)

During inference, the model becomes efficient by making hard decisions.

How it works
 - Router Decision
      For each token, the router decides:
      Important → Full Attention
      Not important → Local Attention
- Single Path Execution
      Only one attention type is computed per token
      No blending anymore
- Final Output
      Tokens either use:
           Global context (accurate but expensive)
           Local context (fast and cheap)

![test_page-0001](https://github.com/user-attachments/assets/7d68c595-6c18-43ea-92ac-58935a2fdc2b)


## Results 

### Inference Latency Reduction:
 - 7.27 ms → 4.11 ms → ~43.5% faster
   <img width="1222" height="648" alt="inference_latency_table" src="https://github.com/user-attachments/assets/e2f491a3-5fcb-40fd-9ac6-3133b68ef593" />

### Compute Efficiency (Sequence Length = 128)

 - Full Attention FLOPs: 0.07G (70M)
 - Router FLOPs: 5.2M
 - Compute Reduction: 64.8M FLOPs saved
 - Efficiency Gain: ~13.5× cheaper than full attention
   
Scaling Insight:

 - Full attention grows quadratically (O(n²))
 - Router uses linear + selective global attention
 - At longer sequences → massive compute savings with minimal latency growth
   
<img width="2385" height="915" alt="image" src="https://github.com/user-attachments/assets/569b8468-9df2-4ff8-bbcb-28b7e25e0da9" />

### Training Trade-offs

- Loss: 6.17 → 7.63 (+23.7% worse)
- Perplexity: 1028 → 1193 (+16.1% worse)
- Accuracy: 0.179 → 0.131 (~26.8% drop)
- FLOPs: 4.1G → 6.12G (~49% higher training cost)

<img width="2268" height="822" alt="training_metrics_table (1)" src="https://github.com/user-attachments/assets/ff939b42-55ef-4442-bf68-7eaf33ed7b1f" />

