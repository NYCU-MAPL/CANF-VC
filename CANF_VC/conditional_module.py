from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import device, nn

COMPUTE_MACS = False

def set_compute_macs(enable=False):
    global COMPUTE_MACS
    COMPUTE_MACS = enable

def gen_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    lmdas = lmdas * int(np.ceil(batch_size/len(lmdas)))
    if shuffle:
        np.random.shuffle(lmdas)
    return torch.Tensor(lmdas[:batch_size]).view(-1, 1).to(device=device)


def hasout_channels(module: nn.Module):
    return hasattr(module, 'out_channels') or hasattr(module, 'out_features') or hasattr(module, 'num_features') or hasattr(module, 'hidden_size')


def get_out_channels(module: nn.Module):
    if hasattr(module, 'out_channels'):
        return module.out_channels
    elif hasattr(module, 'out_features'):
        return module.out_features
    elif hasattr(module, 'num_features'):
        return module.num_features
    elif hasattr(module, 'hidden_size'):
        return module.hidden_size
    raise AttributeError(
        str(module)+" has no avaiable output channels attribute")


class ConditionalLayer(nn.Module):
    def __init__(self, module: nn.Module, out_channels=None, conditions: int = 1, num_states: int = 1):
        super(ConditionalLayer, self).__init__()
        self.m = module
        self.condition_size = conditions
        self.num_states = num_states
        self.state = 0

        if out_channels is None:
            out_channels = get_out_channels(module)
        self.out_channels = out_channels

        if COMPUTE_MACS:
            self.dummy_op = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=out_channels)
            
        self.affine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(conditions, 16),
                    nn.Sigmoid(),
                    nn.Linear(16, out_channels*2, bias=False)
                )
                for _ in range(self.num_states)
            ]
        )

    def extra_repr(self):
        return ""

    def _set_condition(self, condition):
        self.condition = condition
        
    def _set_state(self, state):
        assert self.num_states is not None and state < self.num_states
        self.state = state

    def forward(self, *input, condition=None):
        output = self.m(*input)

        if condition is None:
            condition = self.condition

        if not isinstance(condition, tuple):
            BC, BO = condition.size(0), output.size(0)  # legacy problem for multi device
            if BC != BO:
                assert BC % BO == 0 and output.is_cuda, "{}, {}, {}".format(
                    condition.size(), output.size(), output.device)
                idx = int(str(output.device)[-1])
                condition = condition[BO*idx:BO*(idx+1)]
                # print(idx, condition.cpu().numpy())
            if condition.device != output.device:
                condition = condition.to(output.device)
            
            condition = self.affine[self.state](condition)
            scale, bias = condition.view(
                condition.size(0), -1, *(1,)*(output.dim()-2)).chunk(2, dim=1)
            self.condition = (scale, bias)

            output = output * F.softplus(scale) + bias

            if COMPUTE_MACS:
                _ = self.dummy_op(output)
        else:
            scale, bias = condition

            output = output * F.softplus(scale) + bias

            if COMPUTE_MACS:
                _ = self.dummy_op(output)

        return output.contiguous()


def conditional_warping(m: nn.Module, types=(nn.modules.conv._ConvNd), ignore_if_ch_lgt=0, **kwargs):
    def dfs(sub_m: nn.Module, prefix=""):
        for n, chd_m in sub_m.named_children():
            if dfs(chd_m, prefix+"."+n if prefix else n):
                setattr(sub_m, n, ConditionalLayer(chd_m, **kwargs))
        else:
            # `ignore_if_ch_lgt`: ignore conditional warping if in/out channel is larger than `ignore_if_ch_lgt`
            if isinstance(sub_m, types) and sub_m.in_channels > ignore_if_ch_lgt and sub_m.out_channels > ignore_if_ch_lgt:
                # print(prefix, "C")
                return True
            else:
                pass
                # print(prefix)
        return False

    dfs(m)
    #print(m)


def set_condition(model, condition):
    for m in model.modules():
        if isinstance(m, ConditionalLayer):
            m._set_condition(condition)

def set_state(model, state):
    for m in model.modules():
        if isinstance(m, ConditionalLayer):
            m._set_state(state)

def _args_expand(*args, length):
    for arg in args:
        yield [arg] * length

