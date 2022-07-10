import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

import resample2d_cuda


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input, flow, sample_mode='bilinear', kernel_size=1):
        assert input.is_contiguous() and input.is_cuda
        assert flow.is_contiguous() and flow.is_cuda

        ctx.save_for_backward(input, flow)
        ctx.sample_mode = sample_mode
        ctx.kernel_size = kernel_size

        _, c, _, _ = input.size()
        b, _, h, w = flow.size()
        output = input.new_zeros(b, c, h, w)

        if sample_mode == 'bicubic':
            resample2d_cuda.bicubic_forward(input, flow, output)
        else:
            resample2d_cuda.forward(input, flow, output,
                                    kernel_size, sample_mode == 'bilinear')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        input, flow = ctx.saved_tensors

        grad_input = torch.zeros_like(input)
        grad_flow = torch.zeros_like(flow)

        if ctx.sample_mode == 'bicubic':
            resample2d_cuda.bicubic_backward(input, flow, grad_output.data,
                                             grad_input.data, grad_flow.data)
        else:
            resample2d_cuda.backward(input, flow, grad_output.data,
                                     grad_input.data, grad_flow.data,
                                     ctx.kernel_size, ctx.sample_mode == 'bilinear')

        return grad_input, grad_flow, None, None


def warp(input, flow, sample_mode='bilinear', kernel_size=1):
    """Resample image with flow

    Args:
        kernel_size (int): Basicly set to 1
        bilinear (bool): use bilinear or nearest

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Flow: :math:`(N, 2, H', W')` or `(N, H', W', 2)`
        - Output: :math:`(N, C, H', W')` (same shape as flow)

    Returns:
        Resampled input
    """
    if flow.size(-1) == 2:  # B, H, W, 2
        flow = flow.permute(0, 3, 1, 2).contiguous()   # B, 2, H, W
    return Resample2dFunction.apply(input.contiguous(), flow.contiguous(), sample_mode, kernel_size)


class Resample2d(Module):
    """Resample image with flow

    Args:
        kernel_size (int): Basicly set to 1
        bilinear (bool): use bilinear or nearest

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Flow: :math:`(N, 2, H', W')` or `(N, H', W', 2)`
        - Output: :math:`(N, C, H', W')` (same shape as flow)

    Returns:
        Resampled input
    """

    def __init__(self, sample_mode='bilinear', kernel_size=1):
        super(Resample2d, self).__init__()
        self.sample_mode = sample_mode
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'sample_mode={sample_mode}'.format(**self.__dict__)

    def forward(self, input, flow):
        return warp(input, flow, self.sample_mode, self.kernel_size)
