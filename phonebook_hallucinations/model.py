import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from typing import List
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "Embed size must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Reshape and transpose for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(512, 512)).bool().to(scores.device)


        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and concatenate attention outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Linear projection
        out = self.out(attn_output)

        return out
    
    def sample(self, x: str, mask=None):
        x = self.tokenizer.encode(x)
        x = torch.tensor(x)
        out = self.forward(x)
        print(out.shape) # [1, 513, 128]
        out = out.squeeze(0)
        print(out.shape) # [513, 128]
        out = out.argmax(dim=-1)
        print(out.shape) # [513]
        out = self.tokenizer.decode(out.tolist())
        return out

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def batch_encode(self, batch: List[str]) -> List[List[int]]:
        data = torch.zeros(len(batch), 513, dtype=torch.int64)
        for i, s in enumerate(batch):
            encoded = torch.tensor(self._model.encode(s))
            data[i, :len(encoded)] = encoded
        return data

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        batch_size, seq_length, _ = query.size()
        mask = torch.tril(torch.ones(seq_length, seq_length)).bool().to(query.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention = self.attention(query, key, value, mask=mask)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if device is not None:
            self.dropout.to(device)
            self.pe = pe.to(device)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, device=None):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size, dropout, device=device)
        self.transformer_layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        out = self.embedding(x)
        out = self.pos_embedding(out)
        for layer in self.transformer_layers:
            out = layer(out, out, out, mask)
        return out

    def sample(self, x, mask=None):
        out = self.forward(x, mask)
        return out