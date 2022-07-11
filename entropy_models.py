import numbers

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchac.torchac import ac
from functional import lower_bound, quantize

__version__ = '1.0.0'


def estimate_bpp(likelihood, num_pixels=None, input=None, likelihood_bound: float = 1e-9):
    """estimate bits-per-pixel

    Args:
        likelihood_bound: Float. If positive, the returned likelihood values are
            ensured to be greater than or equal to this value. This prevents very
            large gradients with a typical entropy loss (defaults to 1e-9).
    """
    if num_pixels is None:
        assert torch.is_tensor(input) and input.dim() > 2
        num_pixels = np.prod(input.size()[2:])
    assert isinstance(num_pixels, numbers.Number), type(num_pixels)
    if torch.is_tensor(likelihood):
        likelihood = [likelihood]
    lll = 0
    for ll in likelihood:
        lll = lll + lower_bound(ll, likelihood_bound).log().flatten(1).sum(1)
    return lll / (-np.log(2.) * num_pixels)


class EntropyModel(nn.Module):
    """Entropy model (base class).

    Arguments:
        tail_mass: Float, between 0 and 1. The bottleneck layer automatically
            determines the range of input values based on their frequency of
            occurrence. Values occurring in the tails of the distributions will not
            be encoded with range coding, but using a Golomb-like code. `tail_mass`
            determines the amount of probability mass in the tails which will be
            Golomb-coded. For example, the default value of `2 ** -8` means that on
            average, one 256th of all values will use the Golomb code.
        range_coder_precision: Integer, between 1 and 16. The precision of the
            range coder used for compression and decompression. This trades off
            computation speed with compression efficiency, where 16 is the slowest
            but most efficient setting. Choosing lower values may increase the
            average codelength slightly compared to the estimated entropies.
    """

    quant_modes = ["noise", "universal", "round", "UQ", "pass"]

    def __init__(self, quant_mode="noise", tail_mass=2 ** -8, range_coder_precision=16):
        super(EntropyModel, self).__init__()
        assert quant_mode in self.quant_modes
        self.quant_mode = quant_mode
        self.tail_mass = float(tail_mass)
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                f"`tail_mass` must be between 0 and 1, got {self.tail_mass}.")
        self.range_coder_precision = int(range_coder_precision)
        self._cdf, self._cdf_length, self._offset = None, None, None
        self.condition_size = None
        self.noise = None

    def extra_repr(self):
        return "quant_mode={quant_mode}".format(**self.__dict__)
    #     return "tail_mass={tail_mass}, range_coder_precision={range_coder_precision}".format(**self.__dict__)

    def _set_condition(self, condition):
        """Prepare condition of the model.

        Returns: None
        """
        assert condition is None, f'{self.__class__.__name__} have no condition, got input type {type(condition)}'

    def get_condition(self):
        """Get condition of the model.

        Returns:
            `Tensor` of condition(s)
        """
        raise RuntimeError(f'{self.__class__.__name__} have no condition')

    def quantize(self, input, mode, mean=None):
        """Perturb or quantize a `Tensor` and optionally dequantize.

        Arguments:
            input: `Tensor`. The input values.
            mode: String. Can take on one of three values: `'noise'` (adds uniform
                noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
                (quantizes and produces integer symbols for range coder).

        Returns:
            The quantized/perturbed `input`. The returned `Tensor` should have type
            `self.dtype` if mode is `'noise'`, `'dequantize'`; `ac.dtype` if mode is
            `'symbols'`.
        """
        if mode == "pass":
            return input
        if mode == 'UQ' and self.training:
            noise = torch.empty(input.size(0)).uniform_(-0.5, 0.5)
            self.noise = noise.view(-1, *[1]*(input.dim()-1)).to(input.device)
            # print("A")
            input = input + self.noise

        if mean is not None and mean.dim() != input.dim():
            mean = mean.reshape(1, -1, *[1]*(input.dim()-2))

        outputs = quantize(input, mode, mean)

        if mode == "round" or mode == "UQ":
            outputs = self.dequantize(outputs, mean)
        elif mode == "symbols":
            outputs = outputs.short()

        return outputs

    def dequantize(self, input, mean=None):
        """Dequantize a `Tensor`.

        The opposite to `quantize(input, mode='symbols')`.

        Arguments:
            input: `Tensor`. The range coder symbols.

        Returns:
            The dequantized `input`. The returned `Tensor` should have type
                `self.dtype`.
        """
        outputs = input.float()
        if mean is not None:
            if mean.dim() != input.dim():
                mean = mean.reshape(1, -1, *[1]*(input.dim()-2))
            outputs = outputs + mean
        if self.noise is not None and self.training:
            # print("B")
            outputs = outputs - self.noise
        return outputs

    def _likelihood(self, input):
        """Compute the likelihood of the input under the model.

        Arguments:
            input: `Tensor`. The input values.

        Returns:
            `Tensor` of same shape and type as `input`, giving the likelihoods
                evaluated at `input`.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    def get_cdf(self, shape=None):
        """Compute the pmf of the model.

        Returns:
            `Tensor` of pmf.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    @torch.no_grad()
    def compress(self, input, condition=None, return_sym=False):
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
        self._set_condition(condition)

        symbols = self.quantize(input, "symbols", self.mean)

        cdf, cdf_length, offset, idx = self.get_cdf(input.size())  # CxL
        assert symbols.dtype == cdf.dtype == cdf_length.dtype == offset.dtype == idx.dtype == ac.dtype

        strings = ac.range_index_encode(symbols - offset, cdf, cdf_length, idx)

        if return_sym:
            return strings, self.dequantize(symbols, self.mean)
        else:
            return strings

    @torch.no_grad()
    def decompress(self, strings, shape, condition=None):
        """Decompress values from their compressed string representations.

        Arguments:
            strings: A string `Tensor` vector containing the compressed data.

        Returns:
            The decompressed `Tensor`.
        """
        self._set_condition(condition)

        cdf, cdf_length, offset, idx = self.get_cdf(shape)  # CxL
        assert cdf.dtype == cdf_length.dtype == offset.dtype == idx.dtype == ac.dtype

        symbols = ac.range_index_decode(strings, cdf, cdf_length, idx)

        return self.dequantize(symbols.to(offset.device) + offset, self.mean)

    @staticmethod
    def rate_estimate(likelihood):
        return -likelihood.log2()

    def forward(self, input, condition=None):
        self._set_condition(condition)

        output = self.quantize(
            input, self.quant_mode if self.training else "round", self.mean)

        likelihood = self._likelihood(output)

        return output, likelihood


class FactorizeCell(nn.Module):
    def __init__(self, num_features, in_channel, out_channel, scale, factor=True):
        super(FactorizeCell, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(
            num_features, out_channel, in_channel))
        self.bias = nn.Parameter(torch.Tensor(
            num_features, out_channel, 1))
        if factor:
            self._factor = nn.Parameter(torch.Tensor(
                num_features, out_channel, 1))
        else:
            self.register_parameter('_factor', None)
        self.reset_parameters()

    def reset_parameters(self):
        init = np.log(np.expm1(1/self.scale/self.out_channel))
        nn.init.constant_(self.weight, init)
        nn.init.uniform_(self.bias, -0.5, 0.5)
        if self._factor is not None:
            nn.init.zeros_(self._factor)

    def extra_repr(self):
        s = '{in_channel}, {out_channel}'
        if self._factor is not None:
            s += ', factor=True'
        return s.format(**self.__dict__)

    def forward(self, input, detach=False):
        weight = self.weight.detach() if detach else self.weight
        bias = self.bias.detach() if detach else self.bias
        output = F.softplus(weight) @ input + bias
        if self._factor is not None:
            factor = self._factor.detach() if detach else self._factor
            output = output + torch.tanh(factor) * torch.tanh(output)
        return output


class FactorizeModel(nn.Sequential):
    """Factorize Model"""

    def __init__(self, num_features, init_scale, filters):
        super(FactorizeModel, self).__init__()
        _len = len(filters)
        filters = (1,) + tuple(int(f) for f in filters) + (1,)
        scale = init_scale ** (1 / (len(filters) + 1))

        for i in range(_len + 1):
            self.add_module('l%d' % i, FactorizeCell(
                num_features, filters[i], filters[i+1], scale, factor=i < _len))

    def forward(self, input, detach=False):
        # Convert (batch, channels, *) to (channels, 1, batch) format by commuting channels to front
        # and then collapsing.
        transposed = input.transpose(0, 1)

        tmp = transposed.reshape(input.size(1), 1, -1)
        for module in self:
            tmp = module(tmp, detach)

        # Convert back to input tensor shape.
        output = tmp.reshape_as(transposed).transpose(0, 1)
        return output


class EntropyBottleneck(EntropyModel):
    """Entropy bottleneck layer.

    This layer models the entropy of the tensor passing through it. During
    training, this can be used to impose a (soft) entropy constraint on its
    activations, limiting the amount of information flowing through the layer.
    After training, the layer can be used to compress any input tensor to a
    string, which may be written to a file, and to decompress a file which it
    previously generated back to a reconstructed tensor. The entropies estimated
    during training or evaluation are approximately equal to the average length of
    the strings in bits.

    The layer implements a flexible probability density model to estimate entropy
    of its input tensor, which is described in the appendix of the paper (please
    cite the paper if you use this code for scientific work):

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436

    The layer assumes that the input tensor is at least 2D, with a batch dimension
    at the beginning and a channel dimension as specified by `data_format`. The
    layer trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the input are i.i.d. (independent
    and identically distributed).

    Because data compression always involves discretization, the outputs of the
    layer are generally only approximations of its input. During training,
    discretization is modeled using additive uniform noise to ensure
    differentiability. The entropies computed during training are differential
    entropies. During evaluation, the data is actually quantized, and the
    entropies are discrete (Shannon entropies). To make sure the approximated
    tensor values are good enough for practical purposes, the training phase must
    be used to balance the quality of the approximation with the entropy, by
    adding an entropy term to the training loss. See the example in the package
    documentation to get started.

    Note: the layer always produces exactly one auxiliary loss and one update op,
    which are only significant for compression and decompression. To use the
    compression feature, the auxiliary loss must be minimized during or after
    training. After that, the update op must be executed at least once.
    """

    def __init__(self, num_features, init_scale: float = 10., filters=(3, 3, 3), **kwargs):
        super(EntropyBottleneck, self).__init__(**kwargs)
        self.num_features = num_features
        self.condition_size = 0

        self.factorizer = FactorizeModel(num_features, init_scale, filters)

        # To figure out what range of the densities to sample, we need to compute
        # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. Since we
        # can't take inverses of the cumulative directly, we make it an optimization
        # problem:
        # `quantiles = argmin(|logit(cumulative) - target|)`
        # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
        # Taking the logit (inverse of sigmoid) of the cumulative makes the
        # representation of the right target more numerically stable.

        # Numerically stable way of computing logits of `tail_mass / 2`
        # and `1 - tail_mass / 2`.
        target = np.log(2 / self.tail_mass - 1)  # 6.23
        # Compute lower and upper tail quantile as well as median.
        self.target = target * torch.Tensor([[-1, 0, 1]]).t()

        quantiles = init_scale * torch.Tensor([[-1, 0, 1]]).t()
        self.quantiles = nn.Parameter(quantiles.repeat(1, num_features))

    def extra_repr(self):
        s = super().extra_repr()
        return s+(', ' if s != "" else "")+'num_features={num_features}'.format(**self.__dict__)

    def _logits_cumulative(self, input, detach=False):
        """Evaluate logits of the cumulative densities.

        Arguments:
            input: The values at which to evaluate the cumulative densities, expected
                to be a `Tensor` of shape `(B, C, *)`.

        Returns:
            A `Tensor` of the same shape as `input`, containing the logits of the
                cumulative densities evaluated at the given input.
        """
        return self.factorizer(input, detach=detach)

        # # Convert to (channels, 1, batch) format by commuting channels to front
        # # and then collapsing.
        # x = input.transpose(0, 1)
        # C = input.size(1)
        # output = self.factorizer(x.reshape(C, 1, -1))

        # # Convert back to input tensor shape.
        # output = output.reshape_as(x).transpose(0, 1)
        # return output

    def aux_loss(self):
        logits = self._logits_cumulative(self.quantiles, detach=True)
        if self.target.device != logits.device:
            self.target = self.target.to(logits.device)
        return torch.sum(torch.abs(logits - self.target))

    @property
    def medians(self):
        """Quantize such that the median coincides with the center of a bin."""
        return self.quantiles[1].detach()

    @property
    def mean(self):
        """Quantize such that the median coincides with the center of a bin."""
        return self.medians.view(1, self.num_features, 1, 1)

    @torch.no_grad()
    def _cal_base_cdf(self):
        # Largest distance observed between lower tail quantile and median, and
        # between median and upper tail quantile.
        minima = torch.ceil(self.medians - self.quantiles[0]).relu_()
        maxima = torch.ceil(self.quantiles[2] - self.medians).relu_()

        # PMF starting positions and lengths.
        pmf_start = self.medians - minima
        pmf_length = (maxima + minima + 1).short()

        # Sample the densities in the computed ranges, possibly computing more
        # samples than necessary at the upper end.
        samples = torch.arange(pmf_length.max(), device=pmf_start.device)
        samples = samples.view(-1, 1) + pmf_start

        # We strip the sigmoid from the end here, so we can use the special rule
        # below to only compute differences in the left tail of the sigmoid.
        # This increases numerical stability (see explanation in `call`).
        upper = self._logits_cumulative(samples + 0.5)
        lower = self._logits_cumulative(samples - 0.5)
        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = -torch.sign(upper + lower)
        pmf = torch.abs(torch.sigmoid(sign * upper) -
                        torch.sigmoid(sign * lower))

        for c in range(self.num_features):
            for idx in range(pmf_length[c].item()):
                pmf[idx, c].clamp_min_(2 ** -15)

        pmf = F.normalize(pmf, p=1., dim=0)

        # Compute out-of-range (tail) masses.
        tail_mass = torch.sigmoid(lower[:1]) + torch.sigmoid(-upper[-1:])

        pmf = torch.cat([pmf, torch.zeros_like(tail_mass)], dim=0)

        for c in range(self.num_features):
            pmf[pmf_length[c], c] = tail_mass[0, c]

        self._cdf = ac.pmf2cdf(pmf.t())  # CxL
        self._cdf_length = pmf_length.short().to('cpu', non_blocking=True)
        self._offset = -minima.view(1, -1).short()
        self.idx = torch.arange(self.num_features).view(1, -1).short()

    @torch.no_grad()
    def get_cdf(self, shape):
        if self.training or self._cdf is None:
            self._cal_base_cdf()
            if self._offset.dim() != len(shape):
                self._offset = self._offset.reshape(1, -1, *[1]*(len(shape)-2))
                self.idx = self.idx.reshape(1, -1, *[1]*(len(shape)-2))
        # use saved cdf
        return self._cdf, self._cdf_length, self._offset, self.idx.expand(shape)

    def _likelihood(self, input):
        # Evaluate densities.
        # We can use the special rule below to only compute differences in the left
        # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
        # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
        # with much higher precision than subtracting two numbers close to 1.
        upper = self._logits_cumulative(input + 0.5)
        lower = self._logits_cumulative(input - 0.5)
        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = -torch.sign(upper + lower).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) -
                               torch.sigmoid(sign * lower))

        return likelihood


class SymmetricConditional(EntropyModel):
    """Symmetric conditional entropy model (base class).

    Arguments:
        bin_size: Float. size of probability bin.
        use_mean: Bool. the mean parameters for the conditional distributions. If
            False, the mean is assumed to be zero.
        scale_bound: Float. Lower bound for scales. Any values in `scale` smaller
            than this value are set to this value to prevent non-positive scales. By
            default (or when set to `None`), uses the smallest value in
            `scale_table`. To disable, set to 0.
    """
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    def __init__(self, use_mean=False, scale_bound=None, **kwargs):
        super(SymmetricConditional, self).__init__(**kwargs)
        self.use_mean = bool(use_mean)
        self.condition_size = 2 if self.use_mean else 1
        self.idxmin = np.log(self.SCALES_MIN)
        self.idxmax = np.log(self.SCALES_MAX)

        self.scale_bound = self.SCALES_MIN
        if scale_bound is not None:
            self.scale_bound = max(self.scale_bound, float(scale_bound))

    def extra_repr(self):
        s = super().extra_repr()
        return s+(', ' if s != "" else "")+'use_mean={use_mean}'.format(**self.__dict__)

    def _set_condition(self, condition):
        assert condition is not None, f'{self.__class__.__name__} should given condition'
        assert condition.dim() > 2 and condition.size(1) % self.condition_size == 0
        if self.use_mean:
            self.mean, self.scale = condition.chunk(2, dim=1)
        else:
            self.mean, self.scale = None, condition
        self.scale = lower_bound(self.scale, self.scale_bound)

    def get_condition(self):
        """return mean and scale"""
        return self.mean, self.scale

    def _standardized_quantile(self, quantile):
        """Evaluate the standardized quantile function.

        This returns the inverse of the standardized cumulative function for a
        scalar.

        Arguments:
        quantile: Float. The values at which to evaluate the quantile function.

        Returns:
        A float giving the inverse CDF value.
        """
        return self.distribution.icdf(quantile)

    def _standardized_cumulative(self, input):
        """Evaluate the standardized cumulative density.

        Note: This function should be optimized to give the best possible numerical
        accuracy for negative input values.

        Arguments:
        input: `Tensor`. The values at which to evaluate the cumulative density.

        Returns:
        A `Tensor` of the same shape as `input`, containing the cumulative
        density evaluated at the given input.
        """
        raise NotImplementedError("Must inherit from SymmetricConditional.")

    @torch.no_grad()
    def _cal_base_cdf(self, device='cpu'):
        # scale_table: Iterable of positive floats. For range coding, the scale
        # parameters in `scale` can't be used, because the probability tables need
        # to be constructed statically. Only the values given in this table will
        # actually be used for range coding. For each predicted scale, the next
        # greater entry in the table is selected. It's optimal to choose the
        # scales provided here in a logarithmic way.
        scale_table = torch.exp(torch.linspace(
            self.idxmin, self.idxmax, self.SCALES_LEVELS, device=device))
        if scale_table.le(0).any():
            raise ValueError(
                "`scale_table` must be an iterable of positive numbers.")
        t = scale_table.new_tensor([self.tail_mass / 2])
        multiplier = -self._standardized_quantile(t)
        pmf_center = torch.ceil(scale_table * multiplier)

        self._offset = -pmf_center
        pmf_length = 2 * pmf_center.short() + 1
        self._cdf_length = pmf_length.to('cpu', non_blocking=True)

        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        samples = torch.arange(pmf_length.max(), device=device)
        values = (samples-pmf_center.unsqueeze(1)).abs().neg()
        scale = scale_table.unsqueeze(1)
        upper = self._standardized_cumulative((values+0.5)/scale)
        lower = self._standardized_cumulative((values-0.5)/scale)
        pmf = upper - lower  # Size(self.SCALES_LEVELS, 1479)

        for lv in range(self.SCALES_LEVELS):
            for idx in range(pmf_length[lv].item()):
                pmf[lv, idx].clamp_min_(2 ** -15)

        pmf = F.normalize(pmf, p=1., dim=-1)

        # Compute out-of-range (tail) masses.
        tail_mass = 2 * lower[:, :1]
        pmf = torch.cat([pmf, torch.zeros_like(tail_mass)], dim=-1)

        for lv in range(self.SCALES_LEVELS):
            pmf[lv, pmf_length[lv].item()] = tail_mass[lv]

        self._cdf = ac.pmf2cdf(pmf)

    @torch.no_grad()
    def get_cdf(self, shape=None):
        if self._cdf is None:
            self._cal_base_cdf(device=self.scale.device)

        idx = (self.scale.log() - self.idxmin) / (self.idxmax - self.idxmin)
        idx = idx.clamp_max(1).mul(self.SCALES_LEVELS-1).round()
        offset = self._offset[idx.long()]

        return self._cdf, self._cdf_length, offset.short(), idx.short()

    def _likelihood(self, input):
        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        if self.mean is not None:
            input = input - self.mean
        values = input.abs().neg()
        upper = self._standardized_cumulative((values+0.5)/self.scale)
        lower = self._standardized_cumulative((values-0.5)/self.scale)
        likelihood = upper - lower

        return likelihood


class GaussianConditional(SymmetricConditional):
    """Conditional Gaussian entropy model.

    The layer implements a conditionally Gaussian probability density model to
    estimate entropy of its input tensor, which is described in the paper (please
    cite the paper if you use this code for scientific work):

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436
    """

    distribution = torch.distributions.normal.Normal(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(-(2 ** -0.5) * input)


class LogisticConditional(SymmetricConditional):
    """Conditional logistic entropy model.

    This is a conditionally Logistic entropy model, analogous to
    `GaussianConditional`.
    """

    distribution = torch.distributions.LogisticNormal(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        return torch.sigmoid(input)


class LaplacianConditional(SymmetricConditional):
    """Conditional Laplacian entropy model.

    This is a conditionally Laplacian entropy model, analogous to
    `GaussianConditional`.
    """

    distribution = torch.distributions.Laplace(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        exp = torch.exp(-torch.abs(input))
        return torch.where(input > 0, 2 - exp, exp) / 2


class MixtureModelConditional(SymmetricConditional):
    """Mixture Model conditional entropy model (base class).

    Arguments:
        K: Integer. number of group.
        bin_size: Float. size of probability bin.
        use_mean: Bool. the mean parameters for the conditional distributions. If
            False, the mean is assumed to be zero.
        scale_bound: Float. Lower bound for scales. Any values in `scale` smaller
            than this value are set to this value to prevent non-positive scales. By
            default (or when set to `None`), uses the smallest value in
            `scale_table`. To disable, set to 0.
    """

    def __init__(self, K=3, **kwargs):
        kwargs['use_mean'] = True
        super(MixtureModelConditional, self).__init__(**kwargs)
        self.K = K
        self.condition_size = 3*self.K

    def extra_repr(self):
        s = super().extra_repr()
        return s+(', ' if s != "" else "")+'K={K}'.format(**self.__dict__)

    def _set_condition(self, condition):
        assert condition is not None, f'{self.__class__.__name__} should given condition'
        assert condition.dim() > 2 and condition.size(1) % self.condition_size == 0
        B, NKC = condition.size()[:2]
        shape = (B, 3, self.K, NKC//3//self.K) + tuple(condition.size()[2:])
        self.mean, self.scale, self.pi = condition.view(*shape).unbind(1)
        self.scale = lower_bound(self.scale, self.scale_bound)
        self.pi = self.pi.softmax(dim=1)

    def get_condition(self):
        """return mean, scale, pi"""
        return self.mean, self.scale, self.pi

    @torch.no_grad()
    def get_cdf(self, samples):
        pmf = self._likelihood(samples)
        pmf_clip = pmf.clamp(1.0/65536, 1.0)
        pmf_clip = (pmf_clip / pmf_clip.sum(0, keepdim=True)*65536).round()
        cdf = torch.cumsum(pmf_clip, dim=0).squeeze()
        return torch.cat([cdf[:1], cdf], 0)

    def quantize(self, input, mode, mean_holder=None):
        return super().quantize(input, mode)

    def dequantize(self, input, mean_holder=None):
        return super().dequantize(input)

    @torch.no_grad()
    def compress(self, input, condition, return_sym=False):
        raise NotImplementedError()

    @torch.no_grad()
    def decompress(self, strings, shape, condition):
        raise NotImplementedError()

    def _likelihood(self, input):
        return super()._likelihood(input.unsqueeze(1)).mul(self.pi).sum(1)


class GaussianMixtureModelConditional(MixtureModelConditional):
    """Conditional GaussianMixtureModel entropy model.

    The layer implements a conditionally Gaussian probability density model to
    estimate entropy of its input tensor, which is described in the paper (please
    cite the paper if you use this code for scientific work):

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436
    """

    distribution = torch.distributions.normal.Normal(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(-(2 ** -0.5) * input)


class LogisticMixtureModelConditional(MixtureModelConditional):
    """Conditional LogisticMixtureModel entropy model.

    This is a conditionally Logistic entropy model, analogous to
    `GaussianConditional`.
    """

    distribution = torch.distributions.LogisticNormal(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        return torch.sigmoid(input)


class LaplacianMixtureModelConditional(MixtureModelConditional):
    """Conditional LaplacianMixtureModel entropy model.

    This is a conditionally Laplacian entropy model, analogous to
    `GaussianConditional`.
    """

    distribution = torch.distributions.Laplace(0., 1.)

    @staticmethod
    def _standardized_cumulative(input):
        exp = torch.exp(-torch.abs(input))
        return torch.where(input > 0, 2 - exp, exp) / 2


__CONDITIONS__ = {"Gaussian": GaussianConditional, "Logistic": LogisticConditional, "Laplacian": LaplacianConditional,
                  "GaussianMixtureModel": GaussianMixtureModelConditional,
                  "LogisticMixtureModel": LogisticMixtureModelConditional,
                  "LaplacianMixtureModel": LaplacianMixtureModelConditional}
