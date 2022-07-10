import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

import resample3d_cuda


class Resample3dFunction(Function):

    @staticmethod
    def forward(ctx, input, flow, kernel_size=1, bilinear=True):
        if flow.size(-1) == 3:  # B, D, H, W, 3
            flow = flow.permute(0, 4, 1, 2, 3).contiguous()   # B, 3, D, H, W

        assert input.is_contiguous() and input.is_cuda
        assert flow.is_contiguous() and flow.is_cuda

        ctx.save_for_backward(input, flow)
        ctx.kernel_size = kernel_size
        ctx.bilinear = bilinear

        _, c, _, _, _ = input.size()
        b, _, d, h, w = flow.size()
        output = input.new_zeros(b, c, d, h, w)

        for i in range(b):
            resample3d_cuda.forward(
                input[i], flow[i], output[i], kernel_size, bilinear)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        input, flow = ctx.saved_tensors

        grad_input = torch.zeros_like(input)
        grad_flow = torch.zeros_like(flow)

        for i in range(input.size(0)):
            resample3d_cuda.backward(input[i], flow[i], grad_output.data[i],
                                    grad_input.data[i], grad_flow.data[i],
                                    ctx.kernel_size, ctx.bilinear)

        return grad_input, grad_flow, None, None


def warp3d(input, flow, kernel_size=1, bilinear=True):
    """Resample image with flow

    Args:
        kernel_size (int): Basicly set to 1
        bilinear (bool): use bilinear or nearest

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Flow: :math:`(N, 3, D, H', W')` or `(N, D', H', W', 3)`
        - Output: :math:`(N, C, D', H', W')` (same shape as flow)

    Returns:
        Resampled input
    """
    return Resample3dFunction.apply(input.contiguous(), flow.contiguous(), kernel_size, bilinear)


class Resample3d(Module):
    """Resample image with flow

    Args:
        kernel_size (int): Basicly set to 1
        bilinear (bool): use bilinear or nearest

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Flow: :math:`(N, 3, D, H', W')` or `(N, D', H', W', 3)`
        - Output: :math:`(N, C, D', H', W')` (same shape as flow)

    Returns:
        Resampled input
    """

    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample3d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input, flow):
        return warp3d(input, flow, self.kernel_size, self.bilinear)

    def extra_repr(self):
        return 'kernel_size={kernel_size}, bilinear={bilinear}'.format(**self.__dict__)
