import numpy as np
import torch

_shuffle_info = {}


def _get_shuffle_info(input, upscale_factor):
    pixel_dim = input.dim() - 2
    if isinstance(upscale_factor, int):
        upscale_factor = [upscale_factor] * pixel_dim
    else:
        assert len(upscale_factor) == pixel_dim
        if isinstance(upscale_factor, tuple):
            upscale_factor = list(upscale_factor)
    key = (input.size(), *upscale_factor)

    if key not in _shuffle_info:
        size = list(input.size())
        assert size[1] % np.prod(upscale_factor) == 0
        extented_shape = [size[0]] + upscale_factor + [-1] + size[2:]

        permute_dim = [0, pixel_dim+1]
        for dim in range(2, 2+pixel_dim):
            permute_dim += [dim+pixel_dim, dim-1]

        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], upscale_factor):
            reshape_size.append(shape*scale)

        _shuffle_info[key] = (extented_shape, permute_dim, reshape_size)

    return _shuffle_info[key]


def pixel_shuffle(input, upscale_factor):
    """pixel_shuffle"""
    extented_shape, permute_dim, reshape_size = _get_shuffle_info(
        input, upscale_factor)
    extented = input.reshape(*extented_shape)
    transposed = extented.permute(*permute_dim)
    shuffled = transposed.reshape(*reshape_size)
    return shuffled


def depth_to_space(input, block_size):
    """depth_to_space, alias of pixel_shuffle"""
    return pixel_shuffle(input, block_size)


_unshuffle_info = {}


def _get_unshuffle_info(input, block_size):
    pixel_dim = input.dim() - 2
    if isinstance(block_size, int):
        block_size = [block_size] * pixel_dim
    else:
        assert len(block_size) == pixel_dim
        if isinstance(block_size, tuple):
            block_size = list(block_size)
    key = (input.size(), *block_size)

    if key not in _unshuffle_info:
        size = list(input.size())
        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], block_size):
            assert shape % scale == 0
            reshape_size += [shape//scale, scale]

        permute_dim = [0]
        for dim in range(pixel_dim+1, 1, -1):
            permute_dim += [dim, dim+pixel_dim]
        permute_dim.insert(pixel_dim+1, 1)

        merge_shape = [size[0], -1] + \
            reshape_size[slice(2, 2+pixel_dim+1, pixel_dim)]

        _unshuffle_info[key] = (reshape_size, permute_dim, merge_shape)

    return _unshuffle_info[key]


def pixel_unshuffle(input, block_size):
    """pixel_unshuffle"""
    reshape_size, permute_dim, merge_shape = _get_unshuffle_info(
        input, block_size)
    extented = input.reshape(*reshape_size)
    transposed = extented.permute(*permute_dim)
    merged = transposed.reshape(*merge_shape)
    return merged


def space_to_depth(input, block_size):
    """space_to_depth, alias of pixel_unshuffle"""
    return pixel_unshuffle(input, block_size)


def merge420(inputs):
    if inputs[0].dim() == 3:
        inputs = [input.unsqueeze(0) for input in inputs]
    try:
        return torch.cat([space_to_depth(inputs[0], 2), inputs[1], inputs[2]], dim=1)
    except Exception as e:
        print(inputs[0].shape, inputs[1].shape, inputs[2].shape)

    return None


class UpperBoundGrad(torch.autograd.Function):
    """
    Same as `torch.clamp_max`, but with helpful gradient for `inputs > bound`.
    """

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_max(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input <= ctx.bound) | (grad_output > 0)
        return grad_output * pass_through, None


def upper_bound(input, bound: float):
    """upper_bound"""
    return UpperBoundGrad.apply(input, bound)


class LowerBoundGrad(torch.autograd.Function):
    """
    Same as `torch.clamp_min`, but with helpful gradient for `inputs > bound`.
    """

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_min(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input >= ctx.bound) | (grad_output < 0)
        return grad_output * pass_through.float(), None


def lower_bound(input, bound: float):
    """lower_bound"""
    return LowerBoundGrad.apply(input, bound)


def bound(input, min, max):
    """bound"""
    return upper_bound(lower_bound(input, min), max)


def bound_sigmoid(input, scale=10):
    """bound_sigmoid"""
    return bound(input, -scale, scale).sigmoid()


def bound_tanh(input, scale=3):
    """bound_tanh"""
    return bound(input, -scale, scale).tanh()


def uniform_noise(input):
    """U(-0.5, 0.5)"""
    return torch.empty_like(input).uniform_(-0.5, 0.5)


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mode="round", mean=None):
        ctx.use_mean = False
        if mode == "noise":
            return input + uniform_noise(input)
        else:
            if mean is not None:
                input = input - mean
                ctx.use_mean = True
            return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, -grad_output if ctx.use_mean else None


def quantize(input, mode="round", mean=None):
    """quantize function"""
    return Quantize.apply(input, mode, mean)


def scale_quant(input, scale=2**8):
    return quantize(input * scale) / scale


def noise_quant(input):
    return quantize(input, mode='noise')
