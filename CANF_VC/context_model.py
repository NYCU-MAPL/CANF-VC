import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from entropy_models import SymmetricConditional, ac

__version__ = '1.0.0'


class MaskedConv2d(nn.Conv2d):
    """Custom Conv2d Layer with mask for context model

    Args:
        as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, mode='A', **kwargs):
        kwargs["padding"] = 0
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)
        self.mode = mode.upper()
        self._set_mask()
        self._w_cache = None

    def extra_repr(self):
        return super().extra_repr()+", mode={mode}".format(**self.__dict__)

    @property
    def center(self):
        return tuple([(kernel_size - 1) // 2 for kernel_size in self.kernel_size])

    def _set_mask(self):
        self.register_buffer("_mask", torch.zeros(*self.kernel_size))
        center_h, center_w = self.center
        self._mask[:center_h, :] = 1
        self._mask[:center_h+1, :center_w] = 1
        if self.mode == 'B':
            self._mask[center_h, center_w] = 1

    def pad(self, input):
        padding = ()
        for center in reversed(self.center):
            padding += (center,) * 2
        return F.pad(input, pad=padding, mode={'zeros': 'constant', 'border': 'repilcation'}[self.padding_mode])

    def crop(self, input, left_up=None, windows=None):
        """mask conv crop"""
        if left_up is None:
            left_up = self.center
        if windows is None:
            windows = self.kernel_size
        elif isinstance(windows, int):
            windows = (windows, windows)
        return input[:, :, left_up[0]:left_up[0]+windows[0], left_up[1]:left_up[1]+windows[1]]

    def forward(self, input, padding=True):
        if padding:
            input = self.pad(input)

        #if not self.training:
        #    if self._w_cache is None:
        #        self._w_cache = self.weight*self._mask
        #    return self.conv2d_forward(input, self._w_cache)
        #else:
        #    self._w_cache = None
        #    return self.conv2d_forward(input, self.weight*self._mask)
        self._w_cache = None
        return self.conv2d_forward(input, self.weight*self._mask)


class ContextModel(nn.Module):
    """ContextModel"""

    def __init__(self, num_features, num_phi_features, entropy_model, kernel_size=5):
        super(ContextModel, self).__init__()
        self.num_features = num_features
        assert isinstance(
            entropy_model, SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model
        self.mask = MaskedConv2d(num_features, num_phi_features, kernel_size)
        self.padding = (kernel_size-1)//2

        self.reparam = nn.Sequential(
            nn.Conv2d(num_phi_features*2, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, num_features*self.entropy_model.condition_size, 1)
        )

    def _set_condition(self, output, phi, padding=True):
        masked = self.mask(output, padding)
        # assert masked.size() == phi.size(), (masked.size(), phi.size())

        condition = self.reparam(torch.cat([masked, phi], dim=1))
        self.entropy_model._set_condition(condition)

    def get_cdf(self, samples):
        pmf = self.entropy_model._likelihood(samples)
        pmf_clip = pmf.clamp(1.0/65536, 1.0)
        pmf_clip = (pmf_clip / pmf_clip.sum(0, keepdim=True)*65536).round()
        cdf = torch.cumsum(pmf_clip, dim=0).squeeze()
        return torch.cat([cdf[:1], cdf], 0).t().short()

    @torch.no_grad()
    def compress(self, input, condition, return_sym=False):
        """Compress input and store their binary representations into strings.

        Arguments:
            input: `Tensor` with values to be compressed.

        Returns:
            compressed: String `Tensor` vector containing the compressed
                representation of each batch element of `input`.

        Raises:
            ValueError: if `input` has an integral or inconsistent `DType`, or
                inconsistent number of channels.
        """
        symbols = self.entropy_model.quantize(input, "symbols").cpu()
        fsymbols = symbols.float()

        H, W = symbols.size()[-2:]
        minmax = max(symbols.max().abs(), symbols.min().abs())
        minmax = int(minmax.cpu().clamp_min(1))
        samples = torch.arange(0, minmax*2+1).view(-1, 1, 1, 1)-minmax
        condition = condition.to(symbols.device)

        fsymbols = self.mask.pad(fsymbols)
        elems = np.arange(np.prod(input.size()))
        pbar = tqdm(elems, total=len(elems),
                    desc="context encode", unit="elem(s)")

        for h_idx in range(H):
            for w_idx in range(W):
                patch = self.mask.crop(fsymbols, (h_idx, w_idx))
                patch_phi = self.mask.crop(condition, (h_idx, w_idx), 1)

                self._set_condition(patch, patch_phi, padding=False)
                cdf = self.get_cdf(samples)

                symbols = self.mask.crop(patch, windows=1).squeeze() + minmax
                ac.range_encode(symbols.short(), cdf)

                pbar.update(n=symbols.numel())

        pbar.close()
        string = ac.flush() + b'\x42\xE5\x34\x92' + np.uint8(minmax).tobytes()

        if return_sym:
            return string, self.entropy_model.dequantize(self.mask.crop(fsymbols, windows=(H, W))).to(input.device)
        else:
            return string

    @torch.no_grad()
    def decompress(self, strings, shape, condition):
        """Decompress values from their compressed string representations.

        Arguments:
            strings: A string `Tensor` vector containing the compressed data.

        Returns:
            The decompressed `Tensor`.
        """
        H, W = shape[-2:]

        string, minmax = strings.split(b'\x42\xE5\x34\x92')
        minmax = np.frombuffer(minmax, np.uint8)[0]
        samples = torch.arange(0, minmax*2+1).view(-1, 1, 1, 1)-minmax
        device = condition.device
        condition = condition.to("cpu")

        input = self.mask.pad(torch.zeros(size=shape, device=condition.device))
        ac.set_string(string)
        elems = np.arange(np.prod(shape))
        pbar = tqdm(elems, total=len(elems),
                    desc="context decode", unit="elem(s)")

        for h_idx in range(H):
            for w_idx in range(W):
                patch = self.mask.crop(input, (h_idx, w_idx))
                patch_phi = self.mask.crop(condition, (h_idx, w_idx), 1)

                self._set_condition(patch, patch_phi, padding=False)
                cdf = self.get_cdf(samples)

                recs = ac.range_decode(cdf).float()
                rec = self.entropy_model.dequantize(recs - minmax)
                input[0, :, h_idx+2, w_idx+2].copy_(rec)

                pbar.update(n=recs.numel())

        pbar.close()

        return self.mask.crop(input, windows=(H, W)).to(device)

    def forward(self, input, condition):
        output = self.entropy_model.quantize(
            input, self.entropy_model.quant_mode if self.training else "round")

        self._set_condition(output, condition)

        likelihood = self.entropy_model._likelihood(output)

        return output, likelihood
