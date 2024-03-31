import torch.nn as nn
import torch


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:

        # Fully connected layer that the model uses both in the encoder and in the decoder.
        # In the paper, it is described as:
        #
        #   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
        #
        #   max(0, x) == ReLU(x)

        super().__init__()

        # First linear transformation
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 & b1
        self.dropout = nn.Dropout(dropout)  # Prevent overfitting

        # Second linear transformation
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 & b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
