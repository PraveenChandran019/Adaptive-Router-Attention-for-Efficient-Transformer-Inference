# Adaptive-Router-Attention-for-Efficient-Transformer-Inference

## Problem Statement

Transformer models rely on full self-attention, which has: O(n2⋅d) time and memory complexity.

## Core Issue

Every token attends to every other token

Leads to:

High latency during inference

Quadratic FLOPs growth

Poor scalability for long sequences

## Goal

Design a mechanism that:

Reduces unnecessary global attention computation

Maintains model expressiveness

Improves inference efficiency without major accuracy loss

Proposed Solution: Adaptive Router Attention

We introduce a token-wise routing mechanism that dynamically decides:

"Does this token really need full global attention?"

![test_page-0001](https://github.com/user-attachments/assets/7d68c595-6c18-43ea-92ac-58935a2fdc2b)

![train_page-0001](https://github.com/user-attachments/assets/6cb7e90e-1619-4993-b4e6-71d055dcc94a)

## Key Idea

For each token xi​ , compute a routing weight:
               gi ​∈ [0,1]

Then combine two attention paths:

Outputi​=gi​⋅FullAttentioni​+(1−gi​)⋅LocalAttentioni

## Mathematical Formulation
### 1. Standard Attention:
            Attn(Q,K,V)=Softmax(​QKT/​√d)V
### Complexity:
            O(n**2 d)
### 2. Local Attention (Cheap Path):

Restrict attention to a window of size 
        LocalAttn(xi​)= ∑ j∈N(i) ​αij​Vj​
### Complexity:
            O(n⋅w⋅d)
   
### 3. Router Function

We compute routing scores using token statistics:

Magnitude:
       ∥xi​∥

Variance:
       Var(xi​)
  
Context deviation:
        ∥xi​−μ∥, μ=n1​∑xi​	​

### Router:
        gi​=σ(MLP([∥xi​∥,Var(xi​),∥xi​−μ∥]))]

### 4. Final Output
        yi​=gi​⋅yifull​+(1−gi​)⋅yilocal​

## Empirical Results

### Training Metrics
<img width="2268" height="822" alt="training_metrics_table (1)" src="https://github.com/user-attachments/assets/ff939b42-55ef-4442-bf68-7eaf33ed7b1f" />

<img width="2385" height="915" alt="image" src="https://github.com/user-attachments/assets/569b8468-9df2-4ff8-bbcb-28b7e25e0da9" />

<img width="1222" height="648" alt="inference_latency_table" src="https://github.com/user-attachments/assets/e2f491a3-5fcb-40fd-9ac6-3133b68ef593" />

### Performance Improvements

### 1. Inference Latency

   Latency Reduction= 7.27−4.11 / 7.27 ≈ 43.5%
   
### 2. FLOPs Reduction (Scaling Perspective)

At sequence length 512:

Full Attention: 1.07b FLOPs

Cheap Path: 21M FLOPs

Reduction ≈ 68.0 %

### 3. Scaling Behavior


## Key Insights

Router avoids unnecessary global attention

Gains increase with sequence length

Acts like a learned sparsity mechanism

Trades small accuracy drop for large efficiency gains


## Interpretation

The router effectively learns:

High-importance tokens → Full attention

Low-importance tokens → Local attention

This mimics:

Sparse attention

Mixture-of-experts routing

Conditional computation

## Limitations

Slight degradation in accuracy (~26% relative drop)

Training cost increases (~49% FLOPs increase)

Router quality is critical
