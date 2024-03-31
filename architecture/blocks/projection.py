import torch.nn as nn
import torch


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int)-> None: # model dimension and the size of the output vocabulary
        super().__init__()
        # linear layer for projecting the feature space of `d_model` to the output space of `vocab_size`
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # applying the log Softmax function to the output
        return torch.log_softmax(self.proj(x), dim=-1)
