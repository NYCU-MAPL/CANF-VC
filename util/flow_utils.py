import numpy as np
import torch
import torch.nn.functional as F


def load_flow(path):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise RuntimeError('Magic number incorrect. Invalid .flo file')
        else:
            W, H = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2*W*H)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return torch.Tensor(data).view(H, W, 2)


def save_flow(flow, filename):
    assert flow.dim() == 3
    if flow.size(-1) != 2:
        flow = flow.permute(1, 2, 0)
    if not filename.endswith('.flo'):
        filename += '.flo'
    with open(filename, 'wb') as fp:
        np.array([202021.25], np.float32).tofile(fp)
        H, W = flow.size()[:2]
        np.array([W, H], np.int32).tofile(fp)
        np.array(flow.cpu().numpy(), np.float32).tofile(fp)


def makeColorwheel():
    """
    color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = torch.zeros(ncols, 3)  # r g b

    col = 0
    # RY
    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = torch.linspace(0, 255, RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = torch.linspace(255, 0, YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = torch.linspace(0, 255, GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = torch.linspace(255, 0, CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = torch.linspace(0, 255, BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = torch.linspace(255, 0, MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel.div_(255)


class PlotFlow(torch.nn.Module):
    """
    optical flow color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm

    Shape:
        - Input: :math:`(N, H, W, 2)` or (N, 2, H, W)`
        - Output: :math:(N, 3, H, W)`

    Returns:
        Flowmap
    """

    def __init__(self):
        super().__init__()
        color_map = 1 - makeColorwheel().t().view(1, 3, 1, -1)  # inverse color
        self.register_buffer('color_map', color_map)

    def forward(self, flow, scale=1):
        if flow.size(-1) != 2:
            flow = flow.permute(0, 2, 3, 1)

        known = flow.abs() <= 1e7  # UNKNOW_FLOW_THRESHOLD
        idx_known = known[..., 0] & known[..., 1]
        flow = flow * idx_known.unsqueeze(-1)

        u, v = flow.unbind(-1)
        angle = torch.atan2(-v, -u) / np.pi
        grid = torch.stack([angle, torch.zeros_like(angle)], dim=-1)
        flowmap = F.grid_sample(self.color_map.expand(grid.size(0), -1, -1, -1), grid,
                                mode='bilinear', padding_mode='border', align_corners=True)

        radius = flow.pow(2).sum(-1).add(np.finfo(float).eps).sqrt()
        maxrad = radius.flatten(1).max(1)[0].view(-1, 1, 1)
        flowmap = 1 - radius.div(maxrad / scale).unsqueeze(1) * flowmap
        return flowmap * idx_known.unsqueeze(1)
