import torch
from torch import nn

from functional import lower_bound
from signalconv import SignalConv2d, SignalConvTranspose2d


def Conv2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """Conv2d"""
    if 'padding' in kwargs:
        kwargs.pop('padding')
    return SignalConv2d(in_channels, out_channels, kernel_size, stride=stride,
                        padding=(kernel_size - 1) // 2, *args, **kwargs)


def ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """ConvTranspose2d"""
    if 'padding' in kwargs:
        kwargs.pop('padding')
    if 'output_padding' in kwargs:
        kwargs.pop('output_padding')
    return SignalConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                 padding=(kernel_size - 1) // 2, output_padding=stride-1, *args, **kwargs)


class AugmentedNormalizedFlow(nn.Sequential):

    def __init__(self, *args, use_affine, transpose, distribution='gaussian'):
        super(AugmentedNormalizedFlow, self).__init__(*args)
        self.use_affine = use_affine
        self.transpose = transpose
        self.distribution = distribution
        if distribution == 'gaussian':
            self.init_code = torch.randn_like
        elif distribution == 'uniform':
            self.init_code = torch.rand_like
        elif distribution == 'zeros':
            self.init_code = torch.zeros_like

    def get_condition(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = loc
        return condition, jac

    def get_condition2(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = input.new_zeros(input.size(0), 1), condition

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = scale
        return condition, jac

    def forward(self, input, code=None, jac=None, rev=False, last_layer=False):
        if self.transpose:
            input, code = code, input

        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = scale.sigmoid() * 2 - 1
        scale = self.scale

        if code is None:
            if self.use_affine:
                code = self.init_code(loc)
            else:
                code = None

        if (not rev) ^ self.transpose:
            if code is None:

                code = loc
            else:
                if self.use_affine and not last_layer:
                    code = code * scale.exp()
                    self.jacobian(jac, rev=rev)

                code = code + loc

        else:
            code = code - loc

            if self.use_affine and not last_layer:
                code = code / scale.exp()
                self.jacobian(jac, rev=rev)

        if self.transpose:
            input, code = code, input
        return input, code, jac

    def jacobian(self, jacs=None, rev=False):
        if jacs is not None:
            jac = self.scale.flatten(1).sum(1)
            if rev ^ self.transpose:
                jac = jac * -1

            jacs.append(jac)
        else:
            jac = None
        return jac
