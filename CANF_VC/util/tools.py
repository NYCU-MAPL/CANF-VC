import numpy as np
import torch
import torch.nn.functional as F
from numpy import ceil


__version__ = '0.9.5'


def cat_k(input):
    """concat second dimesion to batch"""
    return input.flatten(0, 1)


def split_k(input, size: int, dim: int = 0):
    """reshape input to original batch size"""
    if dim < 0:
        dim = input.dim() + dim
    split_size = list(input.size())
    split_size[dim] = size
    split_size.insert(dim+1, -1)
    return input.view(split_size)


class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)
        elif input.dim() == 5:
            return split_k(self._align(cat_k(input)), input.size(0))

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)
        elif input.dim() == 5:
            return split_k(self._resume(cat_k(input), shape), input.size(0))

    def forward(self, func, *args, **kwargs):
        pass


_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'


class BitStreamIO:
    """BitStreamIO for Video/Image Compression"""

    def __init__(self, file, mode):
        self.file = file
        self.mode = mode
        self.status = 'open'

        self.strings = b''
        self.streams = list()
        self.shape_strings = list()

    def __len__(self):
        assert self.status == 'open', self.status
        return 1 + np.sum(list(map(len, self.streams+self.shape_strings))) + 4 * (len(self.streams)+len(self.shape_strings))

    @staticmethod
    def shape2string(shape):
        assert len(shape) == 4 and shape[0] == 1, shape
        assert shape[1] < 2 ** 16, shape
        assert shape[2] < 2 ** 16, shape
        assert shape[3] < 2 ** 16, shape
        return np.uint16(shape[1]).tobytes() + np.uint16(shape[2]).tobytes() + np.uint16(shape[3]).tobytes()

    @staticmethod
    def string2shape(string):
        return (1, np.frombuffer(string[0:2], np.uint16)[0],
                np.frombuffer(string[2:4], np.uint16)[0],
                np.frombuffer(string[4:6], np.uint16)[0])

    def write(self, stream_list, shape_list):
        assert self.mode == 'w', self.mode
        self.streams += stream_list

        for shape in shape_list:
            self.shape_strings.append(self.shape2string(shape))

    def read_file(self):
        assert self.mode == 'r', self.mode
        strings = b''
        with open(self.file, 'rb') as f:
            line = f.readline()
            while line:
                strings += line
                line = f.readline()

        self.strings = strings.split(_MAGIC_VALUE_SEP)

        shape_num = int(self.strings[0][0]) // 16
        self.streams, self.shapes = self.strings[shape_num+1:], []
        for shape_strings in self.strings[1:shape_num+1]:
            self.shapes.append(self.string2shape(shape_strings))

        return self.streams, self.shapes

    def read(self, n=1):
        if len(self.strings) == 0:
            self.read_file()

        streams, shapes = [], []
        if len(self.shapes) < n:
            return [], []

        for _ in range(n):
            streams.append(self.streams.pop(0))
            shapes.append(self.shapes.pop(0))

        return streams, shapes

    def split(self, split_size_or_sections):
        if len(self.strings) == 0:
            self.read_file()
        assert len(self.streams) == len(self.shapes)

        if isinstance(split_size_or_sections, int):
            n = split_size_or_sections
            _len = len(self.shapes)
            assert n <= len(self.shapes), (n, len(self.shapes))
            split_size_or_sections = [min(i, n)
                                      for i in range(_len, -1, -n) if i]

        # print(len(self.shapes), split_size_or_sections)
        for n in split_size_or_sections:
            assert n <= len(self.shapes), (n, len(self.shapes))
            ret = self.read(n)
            if len(ret[0]) == 0:
                break
            yield ret

    def chunk(self, chunks):
        if len(self.strings) == 0:
            self.read_file()

        _len = len(self.shapes)
        n = int(np.ceil(_len/chunks))

        return self.split(n)

    def flush(self):
        raise NotImplementedError()

    def close(self):
        assert self.status == 'open', self.status
        if self.mode == 'w':
            shape_num = len(self.shape_strings)
            stream_num = len(self.streams)

            strings = [np.uint8((shape_num << 4) + stream_num).tobytes()]
            strings += self.shape_strings + self.streams

            with open(self.file, 'wb') as f:
                for string in strings[:-1]:
                    f.write(string+_MAGIC_VALUE_SEP)
                f.write(strings[-1])
            del self.streams, self.shape_strings
        else:
            del self.strings, self.streams, self.shapes

        self.status = 'close'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
