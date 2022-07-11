import numpy as np
import torch
import torch.nn.functional as F
from scipy import fftpack
from torch.nn import Parameter

from functional import lower_bound

__version__ = '0.9.5'


class Parameterizer(object):

    def __init__(self):
        pass

    def init(self, param):
        raise NotImplementedError()

    def __call__(self, param):
        raise NotImplementedError()


_irdft_matrices = {}


class RDFTParameterizer(Parameterizer):
    """Object encapsulating RDFT reparameterization.

    This uses the real-input discrete Fourier transform (RDFT) of a kernel as
    its parameterization. The inverse RDFT is applied to the variable to produce
    the parameter.

    (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform)

    Args:
        dc: Boolean. If `False`, the DC component of the kernel RDFTs is not
            represented, forcing the filters to be highpass. Defaults to `True`.
    """

    def __init__(self, dc=True):
        self.dc = dc

    @staticmethod
    def gen_irdft_matrix(size, dtype=torch.float32):
        """Matrix for implementing kernel reparameterization with `matmul`.
        This can be used to represent a kernel with the provided shape in the RDFT
        domain.

        Args:
            size: Iterable of integers. size of kernel to apply this matrix to.
            dtype: `dtype` of returned matrix.
        Returns:
            `Tensor` of size `(prod(size), prod(size))` and dtype `dtype`.
        """
        size = tuple(int(s) for s in size)
        shape = np.prod(size)
        rank = len(size)
        matrix = np.identity(int(shape), dtype=np.float64).reshape(
            (shape,) + size)
        for axis in range(rank):
            matrix = fftpack.rfft(matrix, axis=axis + 1)
            slices = (rank + 1) * [slice(None)]
            if size[axis] % 2 == 1:
                slices[axis + 1] = slice(1, None)
            else:
                slices[axis + 1] = slice(1, -1)
            matrix[tuple(slices)] *= np.sqrt(2)
        matrix /= np.sqrt(shape)
        matrix = np.reshape(matrix, (shape, shape))
        return torch.as_tensor(matrix).to(dtype=dtype)

    def get_irdft_matrix(self, tensor):
        size = tensor.size()[2:]
        shape_key = int(np.prod(size))
        if shape_key not in _irdft_matrices:
            shape_key -= int(not self.dc)
            if any(s != 1 for s in size):
                irdft_matrix = self.gen_irdft_matrix(size, tensor.dtype)
                if not self.dc:
                    irdft_matrix = irdft_matrix[:, 1:]
                _irdft_matrices[shape_key] = irdft_matrix.t()
            else:
                _irdft_matrices[shape_key] = torch.eye(shape_key)

        key = (shape_key, tensor.device)
        if key not in _irdft_matrices:
            irdft_matrix = _irdft_matrices[shape_key]
            _irdft_matrices[key] = irdft_matrix.to(tensor.device)
        return _irdft_matrices[key]

    def init(self, param):
        """no grad init data"""
        irdft_matrix = self.get_irdft_matrix(param)
        with torch.no_grad():
            data = param.data.flatten(2).matmul(irdft_matrix.t())

        return Parameter(data)

    def __call__(self, param, kernel_size):
        """reparam data"""
        irdft_matrix = self.get_irdft_matrix(param)
        return param.matmul(irdft_matrix).reshape(param.size()[:2]+kernel_size)


class NonnegativeParameterizer(Parameterizer):
    """Object encapsulating nonnegative parameterization as needed for GDN.

    The variable is subjected to an invertible transformation that slows down the
    learning rate for small values.

    Args:
        offset: Offset added to the reparameterization of beta and gamma.
            The reparameterization of beta and gamma as their square roots lets the
            training slow down when their values are close to zero, which is desirable
            as small values in the denominator can lead to a situation where gradient
            noise on beta/gamma leads to extreme amounts of noise in the GDN
            activations. However, without the offset, we would get zero gradients if
            any elements of beta or gamma were exactly zero, and thus the training
            could get stuck. To prevent this, we add this small constant. The default
            value was empirically determined as a good starting point. Making it
            bigger potentially leads to more gradient noise on the activations, making
            it too small may lead to numerical precision issues.
    """

    def __init__(self, offset=2 ** -18):
        self.pedestal = offset ** 2

    def init(self, param):
        """no grad init data"""
        with torch.no_grad():
            data = param.relu().add(self.pedestal).sqrt()

        return Parameter(data)

    def __call__(self, param, minmum=0):
        """reparam data"""
        bound = (minmum + self.pedestal) ** 0.5
        return lower_bound(param, bound).pow(2) - self.pedestal
