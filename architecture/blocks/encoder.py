from architecture.blocks.layer_normalization import LayerNormalization
import torch.nn as nn


class Encoder(nn.Module):
    #                                     |
    #                 +--------------------------------------+
    #                 |                   |                  |
    #                 |                  ...                 |
    #                 |                                      |
    #            Nx   |                Encoder               |
    #                 |                 Block                |
    #                 |                                      |
    #                 |                  ...                 |
    #                 |                   |                  |
    #                 +--------------------------------------+
    #                                     |

    def __init__(self, layers: nn.ModuleList) -> None:

        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)  # Normalizing output
