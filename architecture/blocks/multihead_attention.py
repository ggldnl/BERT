import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    #    Multi head self attention module
    #
    #                          Multi-Head Attention
    #                                  ^
    #                                  |
    #                                  |
    #                            +----------+
    #                            |  Linear  |
    #                            +----------+
    #                                  ^
    #                                  |
    #                                  |
    #                            +----------+
    #                            |  Concat  |
    #                            +----------+
    #                                  ^
    #                                  |
    #                                  |
    #    +----------------------------------------------------------+
    #    |                                                          |+
    #    |               Scaled Dot-Product Attention               ||+
    #    |                                                          |||
    #    +----------------------------------------------------------+||
    #     +---^-----------------------^-----------------------^------+|
    #      +--|^----------------------|^----------------------|^------+
    #         ||^                     ||^                     ||^
    #         |||                     |||                     |||
    #         |||                     |||                     |||
    #    +----------+            +----------+            +----------+
    #    |  Linear  |+           |  Linear  |+           |  Linear  |+
    #    +----------+|+          +----------+|+          +----------+|+
    #     +----------+|           +----------+|           +----------+|
    #      +----------+            +----------+            +----------+
    #          ^                       ^                       ^
    #          |                       |                       |
    #          V                       K                       Q
    #          |                       |                       |
    #          +-----------------------+-----------------------+
    #                                  |
    #                                  |
    #                                Input (in the encoder)

    def __init__(self,
                 d_model: int,
                 h: int,  # Number of heads
                 dropout: float
                 ) -> None:
        # We have our input sequence (seq_len, d_model). In the encoder, we split
        # it into three matrices with size (seq_len, d_model) that are exactly the
        # same as the input (things will be slightly different in the decoder), that
        # are query, keys and values. We multiply these matrices by three weight
        # matrices W_q, W_K, W_v with size (d_model, d_model). This results in a
        # new matrix of size (seq_len, d_model). We then split this new matrix
        # into h vectors (where h is the number of heads). We split these matrices
        # along the embedding dimension, not along the sequence dimension. This means
        # each head will have access to the full sequence but will have a different
        # part of the embedding of each token. We apply the attention to these
        # matrices that will return smaller matrices that we will concatenate back
        # and multiply by another weight matrix to obtain the result.

        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):

        # Last dimension of the query, key and value
        d_k = query.shape[-1]

        # We are computing things one batch at a time but the formula
        # is defined on a single input, so we transpose the last two dimensions.
        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Before applying the softmax we apply the mask
        if mask is not None:
            # Replace all the values for which the statement "mask == 0" with -1e9
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply the output matrix by the V matrix
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # Q' matrix
        key = self.w_k(k)  # K' matrix
        value = self.w_v(v)  # V' matrix

        # Transpose => bring the head to the second dimension
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # obtaining the output and the attention scores
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # obtaining the H matrix
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply the H matrix by the weight matrix W_o, resulting in the MH-A matrix
        return self.w_o(x)
