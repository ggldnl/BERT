import torch.nn as nn
import torch


class MLMProjectionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int
                 ) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # applying the log Softmax function to the output
        return torch.log_softmax(self.proj(x), dim=-1)


class NSPProjectionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 ) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Use only the CLS token
        return self.softmax(self.proj(x[:, 0]))
