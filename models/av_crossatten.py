import torch.nn as nn
from .dense_coattn import DenseCoAttn

class NormalSubLayer(nn.Module):
    """
    A normal sub-layer that performs co-attention on two modalities
    (e.g., audio and visual) and then applies a linear transformation + residual.
    """

    def __init__(self, dim1, dim2, dropout):
        super(NormalSubLayer, self).__init__()
        # We assume we've modified DenseCoAttn to only handle two modalities as well.
        self.dense_coattn = DenseCoAttn(dim1, dim2, dropout)

        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2):
        weighted1, weighted2 = self.dense_coattn(data1, data2)
        # Residual connections
        data1 = data1 + self.linears[0](weighted1)
        data2 = data2 + self.linears[1](weighted2)
        return data1, data2


class DCNLayer(nn.Module):
    """
    A dense co-attention network layer stacking multiple NormalSubLayers
    for two modalities only.
    """

    def __init__(self, dim1, dim2, num_seq, dropout):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([
            NormalSubLayer(dim1, dim2, dropout) for _ in range(num_seq)
        ])

    def forward(self, data1, data2):
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2)
        return data1, data2
