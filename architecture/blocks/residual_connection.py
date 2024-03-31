from architecture.blocks.layer_normalization import LayerNormalization
import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        # we use a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        # we use a normalization layer
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # we normalize the input and add it to the original input x`. This creates the residual connection process
        return x + self.dropout(sublayer(self.norm(x)))
