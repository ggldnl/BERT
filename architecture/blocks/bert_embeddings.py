from architecture.blocks.positional_embeddings import PositionalEmbeddings
from architecture.blocks.input_embeddings import InputEmbeddings
import torch.nn as nn


class BERTEmbeddings(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 seq_len: int,
                 dropout: float,
                 padding_idx: int = 0,
                 ) -> None:
        # BERT has three levels of embeddings:
        # 1. Token embeddings
        # 2. Positional embeddings
        # 3. Segment embeddings

        super().__init__()

        # padding_idx not updated during training
        self.input_embeddings = InputEmbeddings(d_model, vocab_size, padding_idx=padding_idx)
        self.positional_embeddings = PositionalEmbeddings(d_model, seq_len, dropout)
        self.segment_embeddings = nn.Embedding(3, d_model, padding_idx=padding_idx)  # segment 1, segment 2 and padding (0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment):
        x = self.input_embeddings(sequence) + self.positional_embeddings(sequence) + self.segment_embeddings(segment)
        return self.dropout(x)
