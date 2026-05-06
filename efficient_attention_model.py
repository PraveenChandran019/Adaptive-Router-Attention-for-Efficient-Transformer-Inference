import torch, math, time, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from thop import profile
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab, seq, max_seq = 50257, 256, 2048
d, heads, layers, ff = 256, 4, 4, 1024
epochs, lr, window = 1, 3e-4, 10


def full_flops(n):
    return 4 * (n**2) * d * layers

def cheap_flops(n):
    return 4 * (n * window) * d * layers


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        self.hd = d // heads

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, heads, self.hd).transpose(1, 2)
        k = k.view(B, T, heads, self.hd).transpose(1, 2)
        v = v.view(B, T, heads, self.hd).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)

        return self.out(out)


class CheapAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        self.hd = d // heads

    def forward(self, x):
        B, T, C = x.shape
        w = window

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, heads, self.hd).transpose(1, 2)
        k = k.view(B, T, heads, self.hd).transpose(1, 2)
        v = v.view(B, T, heads, self.hd).transpose(1, 2)

        k = F.pad(k, (0, 0, w, w))
        v = F.pad(v, (0, 0, w, w))

        k = k.unfold(2, 2*w+1, 1).permute(0, 1, 2, 4, 3)
        v = v.unfold(2, 2*w+1, 1).permute(0, 1, 2, 4, 3)

        q = q.unsqueeze(3)

        scores = (q * k).sum(-1) / math.sqrt(self.hd)

        pos = torch.arange(T, device=x.device)
        window_pos = torch.arange(-w, w+1, device=x.device).view(1,1,1,-1)
        center = pos.view(1,1,T,1)

        mask = (center + window_pos) <= center

        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)

        out = (attn.unsqueeze(-1) * v).sum(3)

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out(out)


class RouterAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.cheap = CheapAttention()
        self.full = FullAttention()

        self.router = nn.Sequential(
            nn.Linear(d+3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.last_gate = 0
        self.full_used = 0
        self.cheap_used = 0

    def forward(self, x):
        B, T, C = x.shape

        summary = x.mean(dim=1, keepdim=True)

        mag = torch.norm(x, dim=-1, keepdim=True)

        var = torch.var(x, dim=-1, keepdim=True)

        div = torch.norm(x - summary, dim=-1, keepdim=True)

        features = torch.cat([x, mag, var, div], dim=-1)

        g = self.router(features)

        self.last_gate = g.mean().item()

        if self.training:
            return g * self.full(x) + (1 - g) * self.cheap(x)

        if g.mean() > 0.5:
            self.full_used += T
            return self.full(x)
        else:
            self.cheap_used += T
            return self.cheap(x)


class Block(nn.Module):
    def __init__(self, attn):
        super().__init__()

        self.attn = attn()

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Linear(ff, d)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, use_router=False):
        super().__init__()

        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)

        Attn = RouterAttention if use_router else FullAttention

        self.blocks = nn.ModuleList([
            Block(Attn) for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(d)

        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)

        x = self.emb(x) + self.pos(pos)

        for b in self.blocks:
            x = b(x)

        x = self.ln(x)

        return self.head(x)


tok = AutoTokenizer.from_pretrained("gpt2")

data = load_dataset("wikitext", "wikitext-2-raw-v1")

data = data.map(lambda e: tok(e["text"]), batched=True)

tokens = torch.tensor([
    t
    for e in data["train"]
    for t in e["input_ids"]
])


def batches(n=2000):
    for _ in range(n):
        i = torch.randint(0, len(tokens)-seq-1, (1,)).item()

        yield (
            tokens[i:i+seq],
            tokens[i+1:i+seq+1]
        )


def train(model, name):
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    params = sum(p.numel() for p in model.parameters())

    dummy = torch.randn(1, seq, d).to(device)

    attn_flops = 0

    for block in model.blocks:
        flops, _ = profile(
            block.attn,
            inputs=(dummy,),
            verbose=False
        )

        attn_flops += flops

    losses, accs = [], []

    start = time.time()

    total_tokens = 0

    model.train()

    for x, y in tqdm(batches()):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, vocab),
            y.view(-1)
        )

        penalty = sum(
            getattr(b.attn, "last_gate", 0)
            for b in model.blocks
        )

        loss = loss + 0.01 * penalty

        opt.zero_grad()

        loss.backward()

        opt.step()

        losses.append(loss.item())

        accs.append(
            (logits.argmax(-1) == y)
            .float()
            .mean()
            .item()
        )

        total_tokens += x.numel()

    duration = time.time() - start

    avg_loss = sum(losses) / len(losses)

    return {
        "Model": name,
        "Params": params,
        "Attention FLOPs": attn_flops,
        "Loss": avg_loss,
        "Perplexity": math.exp(avg_loss),
        "Accuracy": sum(accs) / len(accs),
        "Tokens/sec": total_tokens / duration
    }


def infer_attention(model, name):
    model.eval()

    x = torch.randn(1, seq, d).to(device)

    total_latency = 0

    for block in model.blocks:

        start = time.time()

        for _ in range(100):
            with torch.no_grad():
                block.attn(x)

        latency = (time.time() - start) / 100

        total_latency += latency

    return {
        "Model": name,
        "Attention Latency(ms)": total_latency * 1000
    }


base = Model(False)

router = Model(True)

train_df = pd.DataFrame([
    train(base, "Baseline"),
    train(router, "Router")
])

inf_df = pd.DataFrame([
    infer_attention(base, "Baseline"),
    infer_attention(router, "Router")
])

sizes = [64, 128, 256, 512]

lat_f, lat_r, fl_f, fl_c = [], [], [], []

for s in sizes:

    x = torch.randn(1, s, d).to(device)

    mf = Model(False).to(device)
    mr = Model(True).to(device)

    full_time = 0
    router_time = 0

    for block in mf.blocks:

        start = time.time()

        for _ in range(50):
            with torch.no_grad():
                block.attn(x)

        full_time += (time.time() - start) / 50 * 1000

    for block in mr.blocks:

        start = time.time()

        for _ in range(50):
            with torch.no_grad():
                block.attn(x)

        router_time += (time.time() - start) / 50 * 1000

    lat_f.append(full_time)
    lat_r.append(router_time)

    fl_f.append(full_flops(s))
    fl_c.append(cheap_flops(s))

df = pd.DataFrame({
    "Seq": sizes,
    "Full_Attention_Latency": lat_f,
    "Router_Attention_Latency": lat_r,
    "Full_Attention_FLOPs": fl_f,
    "Cheap_Attention_FLOPs": fl_c
})

print("\nTRAINING METRICS TABLE\n")

print(train_df.to_string(index=False))

print("\nATTENTION LATENCY TABLE\n")

print(inf_df.to_string(index=False))

print("\nATTENTION SCALING TABLE\n")

print(df.to_string(index=False))
