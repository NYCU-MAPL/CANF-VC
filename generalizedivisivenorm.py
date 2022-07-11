import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F

from parameterizers import NonnegativeParameterizer

__version__ = '0.9.6'


def generalized_divisive_norm(input, gamma, beta, inverse: bool, simplify: bool = True, eps: float = 1e-5):
    """generalized divisive normalization"""
    C1, C2 = gamma.size()
    assert C1 == C2, "gamma must be a square matrix"

    x = input.view(input.size()[:2] + (-1,))
    gamma = gamma.reshape(C1, C2, 1)

    # Norm pool calc
    if simplify:
        norm_pool = F.conv1d(x.abs(), gamma, beta.add(eps))
    else:
        norm_pool = F.conv1d(x.pow(2), gamma, beta.add(eps)).sqrt()

    # Apply norm
    if inverse:
        output = x * norm_pool
    else:
        output = x / norm_pool

    return output.view_as(input)


class GeneralizedDivisiveNorm(Module):
    """Generalized divisive normalization layer.

    .. math::
        y[i] = x[i] / sqrt(sum_j(gamma[j, i] * x[j]^2) + beta[i])
        if simplify
        y[i] = x[i] / (sum_j(gamma[j, i] * |x[j]|) + beta[i])

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        inverse: If `False`, compute GDN response. If `True`, compute IGDN
            response (one step of fixed point iteration to invert GDN; the division is
            replaced by multiplication). Default: False.
        gamma_init: The gamma matrix will be initialized as the identity matrix
            multiplied with this value. If set to zero, the layer is effectively
            initialized to the identity operation, since beta is initialized as one. A
            good default setting is somewhere between 0 and 0.5.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.

    Shape:
        - Input: :math:`(B, C)`, `(B, C, L)`, `(B, C, H, W)` or `(B, C, D, H, W)`
        - Output: same as input

    Reference:
        paper: https://arxiv.org/abs/1511.06281
        github: https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py
    """

    def __init__(self, num_features, inverse: bool = False, simplify: bool = False, gamma_init: float = .1, eps: float = 1e-5):
        super(GeneralizedDivisiveNorm, self).__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.simplify = simplify
        self.gamma_init = gamma_init
        self.eps = eps

        self.weight = Parameter(torch.Tensor(num_features, num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.parameterizer = NonnegativeParameterizer()

        self.reset_parameters()

    def reset_parameters(self):
        weight_init = torch.eye(self.num_features) * self.gamma_init
        self.weight = self.parameterizer.init(weight_init)
        self.bias = self.parameterizer.init(torch.ones(self.num_features))

    def extra_repr(self):
        s = '{num_features}'
        if self.inverse:
            s += ', inverse=True'
        if self.simplify:
            s += ', simplify=True'
        s += ', gamma_init={gamma_init}, eps={eps}'
        return s.format(**self.__dict__)

    @property
    def gamma(self):
        return self.parameterizer(self.weight)

    @property
    def beta(self):
        return self.parameterizer(self.bias)

    def forward(self, input):
        return generalized_divisive_norm(input, self.gamma, self.beta, self.inverse, self.simplify, self.eps)
