import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from typing import List
import math

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
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        mask = torch.tril(torch.ones(query.shape[1], query.shape[1])).bool().to(query.device)
        attention = self.attention(query, key, value, attn_mask=mask)[0]
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
        x = x + self.pe[:(x.size(0) - 1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, device=None):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size, dropout, device=device)
        self.transformer_layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        out = x
        out = self.pos_embedding(out)
        for layer in self.transformer_layers:
            out = layer(out, out, out, mask)
        return out
    
    def sample(self, x, mask=None):
        out = self.forward(x, mask)
        return out
