import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, Parameter

from parameterizers import RDFTParameterizer

__version__ = '0.9.5'


__default_parameterizer__ = RDFTParameterizer()


class SignalConv2d(Conv2d):
    r"""Applies a 2D signal convolution over an input signal composed of several input
    planes.

    Args:
        same as torch.nn.Conv2d
        parameterizer (Parameterizer): weight reparameterizer
    """

    def __init__(self, in_channels, out_channels, kernel_size, parameterizer=__default_parameterizer__, **kwargs):
        self.parameterizer = parameterizer
        super(SignalConv2d, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)

    def reset_parameters(self):
        self.weight = Parameter(torch.Tensor(
            self.weight.size()[:2]+self.kernel_size))
        super().reset_parameters()
        if self.parameterizer is not None:
            self.weight = self.parameterizer.init(self.weight)

    def extra_repr(self):
        s = super().extra_repr()
        if self.parameterizer is None:
            s += ", parameterizer=None"
        return s

    def forward(self, input):
        # for torch==1.7
        # return self._conv_forward(input, self.weight if self.parameterizer is None else self.parameterizer(self.weight, self.kernel_size))
        # for torch==1.4
        return self.conv2d_forward(input, self.weight if self.parameterizer is None else self.parameterizer(self.weight, self.kernel_size))


class SignalConvTranspose2d(ConvTranspose2d):
    r"""Applies a 2D signal transposed convolution over an input signal composed of several input
    planes.

    Args:
        same as torch.nn.Conv2d
        parameterizer (Parameterizer): weight reparameterizer
    """

    def __init__(self, in_channels, out_channels, kernel_size, parameterizer=__default_parameterizer__, **kwargs):
        self.parameterizer = parameterizer
        super(SignalConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)

    def reset_parameters(self):
        self.weight = Parameter(torch.Tensor(
            self.weight.size()[:2]+self.kernel_size))
        super().reset_parameters()
        if self.parameterizer is not None:
            self.weight = self.parameterizer.init(self.weight)

    def extra_repr(self):
        s = super().extra_repr()
        if self.parameterizer is None:
            s += ", parameterizer=None"
        return s

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, self.weight if self.parameterizer is None else self.parameterizer(
                self.weight, self.kernel_size), self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
