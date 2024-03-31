import math
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, padding_idx: int = 0):
        # Takes the input and converts it into an embedding.
        # We start with the original sequence (list of tokens).
        # We assign each token a numerical id that correspond
        # to the position of the token in a vocabulary. Each
        # numerical id corresponds to an embedding vector.

        super().__init__()

        self.d_model = d_model  # Dimension of the embedding vectors
        self.vocab_size = vocab_size  # Size of the vocabulary

        # Given a number, it will provide the respective embedding vector.
        # This is a mapping between numbers and vectors, it yields always
        # the same result for a given number
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Normalizing the variance of the embeddings
