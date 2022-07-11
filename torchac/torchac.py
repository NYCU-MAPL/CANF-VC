"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""

# TODO some comments needed about [..., -1] == 0

import torch

__version__ = '2.0.0'

# torchac can be built with and without CUDA support.
# Here, we try to import both torchac_backend_gpu and torchac_backend_cpu.
# If both fail, an exception is thrown here already.
#
# The right version is then picked in the functions below.
#
# NOTE:
# Without a clean build, multiple versions might be installed. You may use python setup.py clean --all to prevent this.
# But it should not be an issue.


import_errors = []


try:
    import torchac_backend_gpu
    CUDA_SUPPORTED = True
except ImportError as e:
    CUDA_SUPPORTED = False
    import_errors.append(e)

try:
    import torchac_backend_cpu
    CPU_SUPPORTED = True
except ImportError as e:
    CPU_SUPPORTED = False
    import_errors.append(e)


imported_at_least_one = CUDA_SUPPORTED or CPU_SUPPORTED


# if import_errors:
#     import_errors_str = '\n'.join(map(str, import_errors))
#     print(f'*** Import errors:\n{import_errors_str}')


if not imported_at_least_one:
    raise ImportError('*** Failed to import any torchac_backend! Make sure to install torchac with torchac/setup.py. '
                      'See the README for details.')


any_backend = torchac_backend_gpu if CUDA_SUPPORTED else torchac_backend_cpu

# print(any_backend.__dict__)


class TorchAC():
    dtype = torch.int16

    def __init__(self) -> None:
        self.encoder = any_backend.TorchEncoder()
        self.decoder = any_backend.TorchDecoder()

    def set_string(self, string):
        self.decoder.set_string(string)

    def gen_uniform_pmf(self, size, L):
        assert size[0] == 1
        histo = torch.ones(L, dtype=torch.float32) / L
        assert (1 - histo.sum()).abs() < 1e-5, (1 - histo.sum()).abs()
        extendor = torch.ones(*size, 1)
        pmf = extendor * histo
        return pmf

    def pmf2cdf(self, pmf, precision=16):
        """
        :param pmf: NL
        :return: N(L+1) as int16 on CPU!
        """
        cdf = pmf.cumsum(dim=-1, dtype=torch.float64).mul_(2 ** precision)
        cdf = cdf.clamp_max_(2**precision - 1).round_()
        cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)
        return cdf.to('cpu', dtype=self.dtype, non_blocking=True)

    def range_index_encode(self, symbols, cdf, cdf_length, indexes):
        """
        :param symbols: symbols to encode, N
        :param cdf: cdf to use, NL
        :param cdf_length: cdf_length to use, N
        :param indexes: index to use, N
        :return: symbols encode to strings
        """
        if symbols.dtype != self.dtype:
            raise TypeError(symbols.dtype)

        stream, outbound_stream = self.encoder.encode_cdf_index(
            cdf, cdf_length, indexes, symbols)

        return stream + b'\x46\xE2\x84\x91' + outbound_stream

    def range_index_decode(self, strings, cdf, cdf_length, indexes):
        """
        :param strings: strings encoded by range_encode
        :param cdf: cdf to use, NL
        :param cdf_length: cdf_length to use, N
        :param indexes: index to use, N
        :return: decoded matrix as torch.int16, N
        """
        input_strings, outbound_strings = strings.split(b'\x46\xE2\x84\x91')

        ret = self.decoder.decode_cdf_index(
            cdf, cdf_length, indexes, input_strings, outbound_strings)

        return ret

    def range_encode(self, symbols, cdf):
        self.encoder.encode_cdf(symbols, cdf)

    def flush(self):
        stream, _ = self.encoder.flush()
        return stream

    def range_decode(self, cdf):
        return self.decoder.decode_cdf(cdf)


ac = TorchAC()
