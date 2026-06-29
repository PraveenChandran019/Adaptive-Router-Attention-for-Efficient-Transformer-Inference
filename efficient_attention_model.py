import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# Hyperparameters
vocab = 50257
d = 256
heads = 4
layers = 4
ff = 1024
window = 10
epochs = 1

# Model Components
embedding = nn.Embedding(vocab, d)

full_attention = nn.MultiheadAttention(
    embed_dim=d,
    num_heads=heads,
    batch_first=True
)

cheap_attention = nn.MultiheadAttention(
    embed_dim=d,
    num_heads=heads,
    batch_first=True
)

router = nn.Sequential(
    nn.Linear(d, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

feed_forward = nn.Sequential(
    nn.Linear(d, ff),
    nn.ReLU(),
    nn.Linear(ff, d)
)

output_layer = nn.Linear(d, vocab)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Training
for epoch in range(epochs):

    for input_ids, target_ids in batches():

        # Embedding
        x = embedding(input_ids)

        # Transformer Layers
        for _ in range(layers):

            score = router(x.mean(dim=1))

            if score.mean() > 0.5:
                x, _ = full_attention(x, x, x)
            else:
                # Window attention (concept)
                x, _ = cheap_attention(x, x, x)

            x = feed_forward(x)

        # Prediction
        logits = output_layer(x)

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, vocab),
            target_ids.view(-1)
        )

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
print("Loss")
print("Accuracy")
print("Perplexity")
print("FLOPs")
print("Latency")
