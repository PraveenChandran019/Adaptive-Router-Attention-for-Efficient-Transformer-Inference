# Adaptive-Router-Attention-for-Efficient-Transformer-Inference

Problem Statement

Transformer models rely on full self-attention, which has:

𝑂
(
𝑛
2
⋅
𝑑
)
O(n
2
⋅d)

time and memory complexity.

Core Issue

Every token attends to every other token

Leads to:

High latency during inference

Quadratic FLOPs growth

Poor scalability for long sequences

Goal

Design a mechanism that:

Reduces unnecessary global attention computation

Maintains model expressiveness

Improves inference efficiency without major accuracy loss

Proposed Solution: Adaptive Router Attention

We introduce a token-wise routing mechanism that dynamically decides:

"Does this token really need full global attention?"

Key Idea

For each token 
𝑥
𝑖
x
i
	​

, compute a routing weight:

𝑔
𝑖
∈
[
0
,
1
]
g
i
	​

∈[0,1]

Then combine two attention paths:

Output
𝑖
=
𝑔
𝑖
⋅
FullAttention
𝑖
+
(
1
−
𝑔
𝑖
)
⋅
LocalAttention
𝑖
Output
i
	​

=g
i
	​

⋅FullAttention
i
	​

+(1−g
i
	​

)⋅LocalAttention
i
	​

Mathematical Formulation
1. Standard Attention
Attn
(
𝑄
,
𝐾
,
𝑉
)
=
Softmax
(
𝑄
𝐾
𝑇
𝑑
)
𝑉
Attn(Q,K,V)=Softmax(
d
	​

QK
T
	​

)V

Complexity:

𝑂
(
𝑛
2
𝑑
)
O(n
2
d)
2. Local Attention (Cheap Path)

Restrict attention to a window of size 
𝑤
w:

LocalAttn
(
𝑥
𝑖
)
=
∑
𝑗
∈
𝑁
(
𝑖
)
𝛼
𝑖
𝑗
𝑉
𝑗
LocalAttn(x
i
	​

)=
j∈N(i)
∑
	​

α
ij
	​

V
j
	​


Complexity:

𝑂
(
𝑛
⋅
𝑤
⋅
𝑑
)
O(n⋅w⋅d)
3. Router Function

We compute routing scores using token statistics:

Magnitude:

∥
𝑥
𝑖
∥
∥x
i
	​

∥

Variance:

Var
(
𝑥
𝑖
)
Var(x
i
	​

)

Context deviation:

∥
𝑥
𝑖
−
𝜇
∥
,
𝜇
=
1
𝑛
∑
𝑥
𝑖
∥x
i
	​

−μ∥,μ=
n
1
	​

∑x
i
	​


Router:

𝑔
𝑖
=
𝜎
(
MLP
(
[
∥
𝑥
𝑖
∥
,
Var
(
𝑥
𝑖
)
,
∥
𝑥
𝑖
−
𝜇
∥
]
)
)
g
i
	​

=σ(MLP([∥x
i
	​

∥,Var(x
i
	​

),∥x
i
	​

−μ∥]))
4. Final Output
𝑦
𝑖
=
𝑔
𝑖
⋅
𝑦
𝑖
full
+
(
1
−
𝑔
𝑖
)
⋅
𝑦
𝑖
local
y
i
	​

=g
i
	​

⋅y
i
full
	​

+(1−g
i
	​

)⋅y
i
local
	​
