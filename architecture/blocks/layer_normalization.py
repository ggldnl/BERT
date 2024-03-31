import torch.nn as nn
import torch


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        # We compute a mean and a variance for each item in the batch independently
        # of the other items and then compute their new values by normalizing by
        # their own mean and variance. We usually add additive and multiplicative
        # constants that introduce some fluctuations in the data. The network
        # will learn to tune these two parameters to introduce fluctuations
        # when necessary.
        #
        #   x_j_hat = (x_j - mu_j) * (sigma_j**2 + eps) ** (-1/2)
        #
        # eps is needed as if sigma happens to be zero or very close to it
        # then x_j_hat will become very big and this is undesirable since
        # we can represent numbers up to a certain precision, we need to
        # avoid numbers too big or too small.

        super().__init__()

        self.eps = eps

        # Define additive and multiplicative constants
        self.alpha = nn.Parameter(torch.ones(1))  # One-dimensional tensor that will be used to scale the input data
        self.bias = nn.Parameter(torch.zeros(1))  # One-dimensional tensor that will be added to the input data

    def forward(self, x):

        # Compute the mean of the input data keeping the number of dimensions unchanged;
        # usually computing the mean removes the dimension for which it is computed, but
        # we want to keep it
        mean = x.mean(dim=-1, keepdim=True)

        # Computing the standard deviation of the input data keeping the number of
        # dimensions unchanged
        std = x.std(dim=-1, keepdim=True)

        # Return the normalized input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
