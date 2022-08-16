"""Visualization library

James Chan
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from matplotlib.lines import Line2D
from torchvision.datasets.folder import default_loader as imgloader
from torchvision.utils import save_image

from .flow_utils import PlotFlow
from .functional import cat_k, split_k, interpolate_as, depth_to_space, space_to_depth

rgb_transform = transforms.Compose([
    transforms.ToTensor()])
gray_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()])
gray3_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor()])
basic_transform = rgb_transform


def YUV2RGB(YUV420, up=False):
    if up:
        assert isinstance(YUV420, tuple)
        Y, U, V = YUV420
        U = F.interpolate(U, scale_factor=2, mode='bilinear', align_corners=False)
        V = F.interpolate(V, scale_factor=2, mode='bilinear', align_corners=False)
        YUV = torch.cat([Y, U, V], dim=1)
    else:
        YUV = YUV420
    YUV[:, 1:] = YUV[:, 1:]-0.5
    T = torch.FloatTensor([[1,        0,    1.402],
                           [1, -0.34414, -0.71414],
                           [1,   1.1772,        0]]).to(YUV.device)
    RGB = T.expand(YUV.size(0), -1, -1).bmm(YUV.flatten(2)).view_as(YUV)
    return RGB.clamp(min=0, max=1)

def RGB2YUV(RGB, down=False):
    T = torch.FloatTensor([[0.257,   0.504,  0.098],
                           [-0.148, -0.291,  0.439],
                           [0.439,  -0.368, -0.071]]).to(RGB.device)
    YUV = T.expand(RGB.size(0), -1, -1).bmm(RGB.flatten(2)).view_as(RGB)
    YUV[:, 1:] = YUV[:, 1:]+16/256
    YUV[:, 1:] = YUV[:, 1:]+128/256

    if down: # RGB444 to YUV420
        Y = YUV[:, 0:1]
        UV = YUV[:, 1:]
        UV = F.interpolate(UV, scale_factor=0.5, mode='bilinear', align_corners=False)
        U, V = UV[:, 0:1], UV[:, 1:2]
        return Y.clamp(min=0, max=1), U.clamp(min=0, max=1), V.clamp(min=0, max=1)
    else:
        return YUV.clamp(min=0, max=1)

def RGB2YUV420(RGB):
    YUV = RGB2YUV(RGB)
    return torch.cat([space_to_depth(YUV[:, 0:1], 2), YUV[:, 1:2, ::2, ::2], YUV[:, 2:3, 1::2, ::2]], dim=1)

def RGB2Gray(RGB):
    T = torch.FloatTensor([[0.299,   0.587,   0.114]]).to(RGB.device)
    Gray = RGB.mul(T.view((3,)+(1,)*(RGB.dim()-2))).sum(1, keepdim=True)
    return Gray.clamp(min=0, max=1)


class CTUCrop(object):
    """Crop the given PIL Image for CTU.

    Args:
        CTU_size (sequence or int): Desired CTU size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, CTU_size, pad_if_needed=False, fill=0, padding_mode='constant'):
        self.CTU_size = int(CTU_size)
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        w, h = transforms.transforms._get_image_size(img)

        # pad if needed
        if self.pad_if_needed:
            ph = self.CTU_size - h % self.CTU_size
            pw = self.CTU_size - w % self.CTU_size
            if not (pw or ph):
                return img
            img = TF.pad(img, (0, 0, pw, ph), self.fill, self.padding_mode)
        else:
            th, tw = h - h % self.CTU_size, w - w % self.CTU_size
            img = TF.crop(img, 0, 0, th, tw)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(CTU_size={0}, padding_mode={1})'.format(self.CTU_size, self.padding_mode)


def load_image(path, size=None, transform=basic_transform, device='cpu'):
    """load_image"""
    img = transform(imgloader(path)).unsqueeze(0)
    if device != 'cpu':
        img = img.to(device)
    if size is None:
        return img
    return F.interpolate(img, (size[-2], size[-1]), mode='bilinear', align_corners=False)


def show_feature_map(fmap, figname):
    save_image(fmap[0].unsqueeze(1), f"./{figname}.png",
               25, normalize=True, scale_each=True)

def gen_color(colors=None, n=10):
    def crange(c1, c2, insert_n=10):
        clist = [np.linspace(c1[i], c2[i], insert_n) for i in range(3)]
        return np.vstack(clist).transpose()

    # print(type(colors))
    if isinstance(colors, np.ndarray):
        pass
    elif colors == None or (isinstance(colors, str) and colors == "RAINBOW"):
        colors = np.array([[255, 0, 0],
                           [255, 127, 0],
                           [240, 255, 0],
                           [0, 255, 0],
                           [0, 30, 255],
                           [75, 0, 130],
                           [148, 0, 211]]) / 255.
    elif isinstance(colors, str) and colors == "RAINBOW2":
        colors = np.array([[255, 0, 0],
                           [255, 127, 0],
                           [240, 255, 0],
                           [0, 255, 0],
                           [0, 30, 255],
                           [75, 0, 130],
                           [148, 0, 211]]) / 255. * 0.5
    elif isinstance(colors, str) and colors == "K":
        colors = np.array([[0, 0, 0],
                           [0, 0, 0]]) / 255.
    elif isinstance(colors, str) and colors == "G":
        colors = np.array([[117, 249, 76],
                           [117, 249, 76]]) / 255.
    elif isinstance(colors, str) and colors == "U":
        colors = np.array([[0, 255,  0],
                           [0,   0, 255]]) / 255.
    elif isinstance(colors, str) and colors == "V":
        colors = np.array([[0, 255, 0],
                           [255, 0, 0]]) / 255.
    elif isinstance(colors, str) and colors == "RB":
        assert n % 2 == 0
        r = np.array([[255, 0, 0],
                      [255, 200, 200]]) / 255.
        b = np.array([[0, 0, 255],
                      [200, 200, 255]]) / 255.
        r_tensor = gen_color(r, n=n//2)
        b_tensor = gen_color(b, n=n//2)
        return torch.cat([r_tensor, b_tensor])

    c = len(colors)
    ln = (n*10-1)//(c-1)+1

    linear_color = []
    for i in range(c-1):
        li = crange(colors[i], colors[i+1], ln)
        linear_color.append(li[1:] if i else li)

    linear_color = np.concatenate(linear_color, axis=0)
    index = np.linspace(0, len(linear_color)-1, n).astype(int)
    return torch.from_numpy(linear_color[index])


class PlotHeatMap(torch.nn.Module):
    """heat map color encoding scheme

    Args:
        color (str): color map `'RAINBOW'` | `'RAINBOW'` | `'K'` | `'RB'`

    Shape:
        - Input: :math:`(N, 1, H, W)`
        - Output: :math:`(N, 3, H, W)`

    Returns:
        Heatmap
    """

    def __init__(self, color='RAINBOW'):
        super().__init__()
        color_map = gen_color(color, n=10).t().unsqueeze(1).float()
        self.register_buffer('color_map', color_map.repeat(1, 2, 1))

    def forward(self, input):
        assert input.size(1) == 1
        input = input.permute(0, 2, 3, 1) * 2 - 1  # ~(-1, 1) (B, H, W, 1)
        grid = torch.cat([input.neg(), torch.zeros_like(input)], dim=-1)
        if self.color_map.device != input.device:
            self.color_map = self.color_map.to(input.device)
        heatmap = F.grid_sample(self.color_map.repeat(grid.size(0), 1, 1, 1), grid,
                                mode='bilinear', padding_mode='border', align_corners=True)
        return heatmap


class PlotYUV(torch.nn.Module):
    def __init__(self, mode="YUV") -> None:
        super().__init__()
        self.U_plot = PlotHeatMap("U")
        self.V_plot = PlotHeatMap("V")

    def forward(self, input):
        Y = input[:, :-2]
        U = self.U_plot(input[:, -2:-1])
        V = self.V_plot(input[:, -1:])
        if input.size(1) == 6:
            Y = depth_to_space(Y, 2).repeat(1, 3, 1, 1)
            merged = torch.cat([Y, torch.cat([U, V], dim=2)], dim=3)
        else:
            merged = torch.cat([Y.repeat(1, 3, 1, 1), U, V], dim=3)

        return merged


plot_yuv = PlotYUV()


def plot_grad_flow(named_parameters, figname="./gradflow"):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backward() as
    "plot_grad_flow(module.named_parameters())" to visualize the gradient flow
    """
    avg_grads, max_grads, names = [], [], []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            texts = n.replace('.weight', '').split(".")
            if texts[-1][0] == '_':
                continue
            names.append("_".join(text[:3] if i != len(
                texts)-1 else text[:6] for i, text in enumerate(texts)))
            grad = p.grad.clone().detach()
            infinity = torch.logical_not(torch.isfinite(grad))
            if infinity.any():
                message = 'grad infinity waring :'+names[-1]
                if torch.isnan(grad).any():
                    message += '\tType: nan'
                else:
                    message += '\tType: inf'
                warnings.warn(message)
                grad[infinity] = 1e20
                # print(torch.logical_not(torch.isfinite(grad)).any())

            avg_grads.append(grad.abs().mean().item())
            max_grads.append(grad.abs().max().item())
    leng = len(max_grads)
    graph = plt.figure(figsize=(4.8, 6.4/20*max(leng, 8)))
    plt.barh(np.arange(leng), max_grads[::-1], alpha=0.3, lw=1.5, color="c")
    plt.barh(np.arange(leng), avg_grads[::-1], alpha=0.5, lw=1.5, color="b")
    plt.vlines(0, -1, leng+1, lw=2, color="k")
    plt.yticks(range(leng), names[::-1])
    plt.ylim((-1, leng))
    # zoom in on the lower gradient regions
    top = np.max(max_grads)
    plt.xlim(left=-top*1e-6, right=top*1.2)
    plt.ylabel("Layers")
    plt.xlabel("gradient value(abs)")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    if figname == 'ret':
        return graph
    plt.savefig(figname)
    plt.close(graph)


def compare_img(imgs, nrow=25):
    """rearrange img for visualize"""
    if isinstance(imgs, list):
        cmp = torch.stack(imgs, dim=1)
    else:
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(2)
        cmp = imgs

    return cmp[:nrow].transpose(1, 0).flatten(0, 1)


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
        H_ = int(np.ceil(H / self.divisor) * self.divisor)
        W_ = int(np.ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
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
