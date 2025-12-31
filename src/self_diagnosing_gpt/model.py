# -*- coding: utf-8 -*-
"""
Self-Diagnosing GPT with Mechanistic Interpretability
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
confidence_loss_weight = 0.01
confidence_threshold = 0.4
confidence_ema_alpha = 0.3

torch.manual_seed(1337)


def get_data(filepath='input.txt'):
    """Load and prepare dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, encode, decode


def get_batch(train_data, val_data, split):
    """Get a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """MLP block"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SelfDiagnosingGPT(nn.Module):
    """GPT with confidence head"""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.confidence_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Linear(n_embd // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        confidence = self.confidence_head(x).squeeze(-1)

        if targets is None:
            return logits, confidence, None

        # Language modeling loss
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        lm_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Confidence loss
        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            max_entropy = np.log(self.vocab_size)
            normalized_entropy = entropy / max_entropy
            confidence_target = 1.0 - normalized_entropy
            confidence_target = confidence_target.view(B, T)
        
        confidence_loss = F.mse_loss(confidence, confidence_target)
        total_loss = lm_loss + confidence_loss_weight * confidence_loss
        
        return logits, confidence, total_loss

    def generate(self, idx, max_new_tokens, temperature=1.0, decode_fn=None, show_confidence=True):
        """Generate text with confidence tracking"""
        self.eval()
        confidence_log = []
        ema_confidence = None
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, confidence, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                conf = confidence[:, -1].item()
                
                if ema_confidence is None:
                    ema_confidence = conf
                else:
                    ema_confidence = confidence_ema_alpha * conf + (1 - confidence_ema_alpha) * ema_confidence
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                if decode_fn:
                    token = decode_fn([idx_next.item()])
                    confidence_log.append((token, conf, ema_confidence))
                
                if show_confidence and ema_confidence < confidence_threshold and i % 10 == 0:
                    print(f"⚠️  Low confidence (EMA: {ema_confidence:.3f})")
                
                idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx, confidence_log


class ActivationLogger:
    """Log and visualize activations"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, layer_indices=None):
        if layer_indices is None:
            layer_indices = range(len(self.model.blocks))
        
        for idx in layer_indices:
            hook = self.model.blocks[idx].register_forward_hook(
                self._hook_fn(f'block_{idx}')
            )
            self.hooks.append(hook)
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_activations(self, layer_name, seq_idx=0, max_neurons=64):
        if layer_name not in self.activations:
            print(f"Layer {layer_name} not found")
            return
        
        act = self.activations[layer_name]
        B, T, C = act.shape
        act_seq = act[seq_idx, :, :]
        act_seq = act_seq[:, :max_neurons]
        act_viz = act_seq.T.numpy()
        
        plt.figure(figsize=(12, 6))
        plt.imshow(act_viz, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Activation strength')
        plt.xlabel('Token position')
        plt.ylabel('Neuron index')
        plt.title(f'Activation Heatmap: {layer_name}')
        plt.tight_layout()
        plt.show()


@contextmanager
def ablate_neurons(model, block_idx, neuron_indices):
    """Context manager for neuron ablation"""
    hooks = []
    
    def ablation_hook(module, input, output):
        for neuron_idx in neuron_indices:
            if neuron_idx < output.shape[-1]:
                output[:, :, neuron_idx] = 0
        return output
    
    target_layer = model.blocks[block_idx].ffwd
    hook = target_layer.register_forward_hook(ablation_hook)
    hooks.append(hook)
    
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and val"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            _, _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(data_path='input.txt', max_iters=3000):
    """Main training function"""
    # Load data
    train_data, val_data, vocab_size, encode, decode = get_data(data_path)
    
    # Initialize model
    model = SelfDiagnosingGPT(vocab_size)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch(train_data, val_data, 'train')
        _, _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Store decode function
    model.decode = decode
    model.encode = encode
    
    return model
