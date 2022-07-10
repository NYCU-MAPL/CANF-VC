"""ReImplement Correlation Module

James Chan
"""
import torch
from torch.nn import Module

from .functional import meshgrid
from .sampler import shift


class Correlation(Module):
    """Correlation metion in `FlowNet`.

    Args:
        num_input (int): input numers. Default: 2
        kernel_size (int or pair of int): Default: 21
        dilation (int or pair of int): correlation to larger displacement. Default: 1
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'

    Shape:
        - Input: :math:`(B, N, C, H, W)`
        - Output: :math:`(B, (N-1)K, H, W)` where `K` means kernel area
    """

    def __init__(self, num_input=2, kernel_size=21, dilation=1, padding_mode='zeros'):
        super(Correlation, self).__init__()
        self.num_input = num_input
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation
        self.padding_mode = padding_mode

        grid = meshgrid((1, 1, kernel_size[0], kernel_size[1]))
        scale = grid.new_tensor([kernel_size[0]//2*dilation[0],
                                 kernel_size[1]//2*dilation[1]])
        self.grid = grid.flatten(1, 2)*scale
        self.sample_kwargs = dict(
            sample_mode='nearest', padding_mode=self.padding_mode, align_corners=True)

    def extra_repr(self):
        return '{num_input}, kernel_size={kernel_size}, dilation={dilation}, padding_mode={padding_mode}'.format(**self.__dict__)

    def forward(self, *inputs):  # -> (B, K(N-1), H, W)
        if inputs[0].dim() == 4:
            inputs = torch.stack(inputs, dim=1)
        else:
            inputs = inputs[0]
        B, N = inputs.size()[:2]
        K = self.grid.size(1)
        assert N == self.num_input
        inputs = inputs.detach().mul(1e10).floor().div(1e10) - inputs.detach() + inputs

        target = inputs[:, -1]  # take last input as correlation target
        grid = self.grid.to(inputs.device).expand(B, -1, -1)

        return torch.stack([(inputs[:, :N-1]*shift(target, grid[:, k], **self.sample_kwargs).unsqueeze(1)).mean(-3)
                            for k in range(K)], dim=1).flatten(1, 2)

        # shifted = torch.stack([shift(target, grid[:, k], **self.sample_kwargs)
        #                        for k in range(K)], dim=1)

        # # print(shifted.shape)
        # # from .pytools import debug, settensorFormat
        # # print = debug
        # # settensorFormat(f=12)
        # # print(inputs[:, 0:1])
        # # print(shifted[:, 0])
        # # print(inputs[:, 0:1]*shifted[:, 0])
        # # print((inputs[:, 0:1]*shifted[:, 0]).mean(-3))

        # # inputs: (B, N-1, C, H, W) bmm shifted target: (B, K, C, H, W) -> (B, K(N-1), H, W)
        # return torch.cat([(inputs[:, i:i+1] * shifted).mean(-3) for i in range(N-1)], dim=1)
