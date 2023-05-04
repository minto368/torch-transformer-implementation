import torch
import torch.nn as nn
import numpy as np

class multiheadattention(nn.Module):
    def __init__(self, d_model, heads):
        super(multiheadattention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        assert (
            self.head_dim * heads == d_model
        ), "Embedding size needs to be divisible by heads"
        
        self.queries = nn.Linear(d_model, d_model)
        self.keys = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask):
        batch_size = queries.shape[0]

        queries_len, keys_len, values_len = queries.shape[1], keys.shape[1], values.shape[1]

        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        queries = queries.reshape(batch_size, queries_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, keys_len, self.heads, self.head_dim)
        values = values.reshape(batch_size, values_len, self.heads, self.head_dim)

        score = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        if attn_mask is not None:
            score = score.masked_fill(mask, -np.inf)

        attention = torch.softmax(score / (self.d_model ** (1 / 2)), dim=3)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            batch_size, queries_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        
        return out, attention.mean(dim=1)

    