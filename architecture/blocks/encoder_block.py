from architecture.blocks.multihead_attention import MultiHeadAttentionBlock
from architecture.blocks.residual_connection import ResidualConnection
from architecture.blocks.feed_forward import FeedForwardBlock
import torch.nn as nn


class EncoderBlock(nn.Module):
    #                               ^
    #                               |
    #    +---------------------------------------------------+
    #    |                          |                        |
    #    |        +--------------------------------------+   |
    #    |        |                                      |   |
    #    |  +---->|             Add & Norm               |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       |                        |
    #    |  |     +--------------------------------------+   |
    #    |  |     |                                      |   |
    #    |  |     |            Feed Forward              |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       ^                        |
    #    |  |                       |                        |
    #    |  +-----------------------+                        |
    #    |                          |                        |
    #    |                          |                        |
    #    |        +--------------------------------------+   |
    #    |        |                                      |   |
    #    |  +---->|             Add & Norm               |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       |                        |
    #    |  |    +---------------------------------------+   |
    #    |  |    |                                       |   |
    #    |  |    |         Multi-Head Attention          |   |
    #    |  |    |                                       |   |
    #    |  |    +---------------------------------------+   |
    #    |  |              ^        ^        ^               |
    #    |  |              |        |        |               |
    #    |  |              +--------+--------+               |
    #    |  |                       |                        |
    #    |  +-----------------------+                        |
    #    |                          |                        |
    #    +---------------------------------------------------+
    #                               |

    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float
                 ) -> None:

        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])

    def forward(self, x, source_mask):

        # We might want to use a source_mask on the input of the encoder
        # to exclude some tokens, such as the <PAD> token and so on...
        # We don't want these tokens to interact with other tokens

        # Input goes to the Multi-Head Self Attention module. It's the
        # sequence that is watching itself. Each token of the sequence
        # is interacting with other tokens of the same sequence.
        # In the decoder we have a different situation (Cross Attention)
        # where the query coming from the decoder is watching the keys
        # and the values coming from the encoder

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
