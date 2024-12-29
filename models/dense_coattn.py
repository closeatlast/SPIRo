import math
import torch
import torch.nn as nn
import sys

class DenseCoAttn(nn.Module):
    """
    A two-modality dense co-attention mechanism.
    Now only takes value1 (e.g., visual) and value2 (e.g., audio).
    """

    def __init__(self, dim1, dim2, dropout):
        super(DenseCoAttn, self).__init__()
        # Combine two modality dims
        dim = dim1 + dim2

        # Only two dropout layers now
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        
        # Query linear layer applies to the concatenation of the two modalities
        self.query_linear = nn.Linear(dim, dim)
        
        # Key/Value transforms for each modality
        # (If you prefer these to be parametric, replace 300 with the appropriate shape.)
        self.key1_linear = nn.Linear(300, 300)
        self.key2_linear = nn.Linear(300, 300)

        self.value1_linear = nn.Linear(dim1, dim1)
        self.value2_linear = nn.Linear(dim2, dim2)

        self.relu = nn.ReLU()

    def forward(self, value1, value2):
        # Concatenate the two modalities along the last dimension
        joint = torch.cat((value1, value2), dim=-1)

        # Transform the concatenated features
        va_joint = self.query_linear(joint)

        # Keys: (B, Channels, SeqLen) => (B, SeqLen, Channels) for matmul
        key1 = self.key1_linear(value1.transpose(1, 2))
        key2 = self.key2_linear(value2.transpose(1, 2))

        # Apply linear transforms to values
        value1 = self.value1_linear(value1)
        value2 = self.value2_linear(value2)

        # Compute co-attention for each modality
        weighted1, attn1 = self.qkv_attention(joint, key1, value1, dropout=self.dropouts[0])
        weighted2, attn2 = self.qkv_attention(joint, key2, value2, dropout=self.dropouts[1])

        return weighted1, weighted2

    def qkv_attention(self, query, key, value, dropout=None):
        # Query shape: (B, SeqLen, Dim), key shape: (B, Dim, SeqLen)
        d_k = query.size(-1)
        scores = torch.bmm(key, query) / math.sqrt(d_k)
        scores = torch.tanh(scores)

        if dropout:
            scores = dropout(scores)

        # Weighted output
        weighted = torch.tanh(torch.bmm(value, scores))
        return self.relu(weighted), scores
