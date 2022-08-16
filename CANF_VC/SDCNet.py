import os
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm, trange

from flownets import PWCNet
from util.flow_utils import PlotFlow
from util.sampler import Resampler, warp

try:
    from util.spatialdisplconv_package.spatialdisplconv import SpatialDisplConv
    sdc_cuda = True
except ImportError:
    sdc_cuda = False


def conv2d(channels_in, channels_out, kernel_size=3, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1, bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


class SDCNet(nn.Module):

    def __init__(self, sequence_length, use_sdc=False, kernel_size=11):
        super(SDCNet, self).__init__()
        self.sequence_length = sequence_length

        factor = 2
        self.input_channels = self.sequence_length * \
            3 + (self.sequence_length - 1) * 2
        self.output_channels = 2
        if use_sdc:
            self.output_channels += kernel_size * 2

        self.conv1 = conv2d(self.input_channels, 64 // factor,
                            kernel_size=7, stride=2)
        self.conv2 = conv2d(64 // factor, 128 // factor,
                            kernel_size=5, stride=2)
        self.conv3 = conv2d(128 // factor, 256 // factor,
                            kernel_size=5, stride=2)
        self.conv3_1 = conv2d(256 // factor, 256 // factor)
        self.conv4 = conv2d(256 // factor, 512 // factor, stride=2)
        self.conv4_1 = conv2d(512 // factor, 512 // factor)
        self.conv5 = conv2d(512 // factor, 512 // factor, stride=2)
        self.conv5_1 = conv2d(512 // factor, 512 // factor)
        self.conv6 = conv2d(512 // factor, 1024 // factor, stride=2)
        self.conv6_1 = conv2d(1024 // factor, 1024 // factor)

        self.deconv5 = deconv2d(1024 // factor, 512 // factor)
        self.deconv4 = deconv2d(1024 // factor, 256 // factor)
        self.deconv3 = deconv2d(768 // factor, 128 // factor)
        self.deconv2 = deconv2d(384 // factor, 64 // factor)
        self.deconv1 = deconv2d(192 // factor, 32 // factor)
        self.deconv0 = deconv2d(96 // factor, 16 // factor)

        self.final_flow = nn.Conv2d(self.input_channels + 16 // factor, self.output_channels,
                                    kernel_size=3, stride=1, padding=1, bias=True)

        # init parameters, when doing convtranspose3d, do bilinear init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

        self.flownet = PWCNet()
        self.Resampler = SpatialDisplConv() if use_sdc and sdc_cuda else Resampler()
        self._frame_buffer = None
        self._flow_buffer = None

        #data = torch.load(f'./models/SDCNet_ref{sequence_length}.ckpt', map_location='cpu')
        #self.load_state_dict(data['model'])

    def append_buffer(self, frame):
        if self._frame_buffer is None:
            self._frame_buffer = frame.unsqueeze(1)
            return
        if self._frame_buffer.size(1) == self.sequence_length:
            self._frame_buffer = self._frame_buffer[:, 1:]
            # print("update frame")
        self._frame_buffer = torch.cat(
            [self._frame_buffer, frame.unsqueeze(1)], 1)

    def append_flow(self, flow):
        if self._flow_buffer is None:
            plan_flow = torch.zeros_like(flow)
            self._flow_buffer = [plan_flow, flow]
        else:
            self._flow_buffer.append(flow)

        if len(self._flow_buffer) == self.sequence_length:
            self._flow_buffer.pop(0)

    def clear_buffer(self, input_frames=None):
        self._frame_buffer = input_frames
        self._flow_buffer = None

    def _prepare_frame(self, input_frames):
        """prepare input images, shape (B, T, C, H, W)"""
        if torch.is_tensor(input_frames) and input_frames.dim() == 4:
            self.append_buffer(input_frames)
            input_frames = None
        if input_frames is None:
            input_frames = self._frame_buffer
        if isinstance(input_frames, (list, tuple)):
            input_frames = torch.stack(input_frames, 1)
        # print(input_frames.shape)

        return input_frames

    def _prepare_flow(self, input_frames, input_flows):
        """prepare input flows, shape (B, T-1, 2, H, W)"""
        if input_flows is None:
            input_flows = self._flow_buffer
        if input_flows is None:
            B, T, _, H, W = input_frames.size()
            # print(input_frames.shape)
            im1s = input_frames[:, :-1].flatten(0, 1)
            im2s = input_frames[:, 1:].flatten(0, 1)
            flows = self.flownet(im1s, im2s)
            # print(flows.shape)
            input_flows = flows.reshape(B, T-1, 2, H, W)
        else:
            if isinstance(input_flows, (list, tuple)):
                input_flows = torch.stack(input_flows, dim=1)
            # if input_flows.size(1) == self.sequence_length - 1:
            #     # print("update flow")
            #     flow = self.flownet(input_frames[:, -2], input_frames[:, -1])
            #     input_flows = torch.cat(
            #         [input_flows[:, 1:], flow.unsqueeze(1)], 1)
            assert input_flows.size(1) == self.sequence_length - 1, "Input flow amount does not match the seq length."

        # print(input_flows.shape)
        # self._flow_buffer = input_flows
        return input_flows

    def forward(self, input_frames=None, input_flows=None, auto_warp=True):
        # print("SDC forward")
        """SDC forward
            input_frames: (B, T, C, H, W)
                [(B, C, H, W),...]: auto concat
                (B, C, H, W): auto append buffer
                None: use buffer
            input_flows: (B, T-1, 2, H, W)
                [(B, 2, H, W),...]: auto concat
            auto_warp: bool. if `False`, return flow only
        """
        input_frames = self._prepare_frame(input_frames)
        if input_frames.size(1) < self.sequence_length:
            return input_frames[:, -1], None
        input_flows = self._prepare_flow(input_frames, input_flows)
        try:
            images_and_flows = torch.cat(
                [input_frames.flatten(1, 2), input_flows.flatten(1, 2)], 1)
        except:
            print('len(input_frames) =', len(input_frames))
            print('len(input_flows) =', len(input_flows))
            print('input_frames[0].shape =', input_frames[0].shape)
            print('input_flows[0].shape =', input_flows[0].shape)
        # print(images_and_flows.shape)
        assert images_and_flows.size(1) == self.input_channels, \
            (input_frames.size(), input_flows.size(), self.sequence_length)

        out_conv1 = self.conv1(images_and_flows)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)

        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((images_and_flows, out_deconv0), 1)

        output_flow = self.final_flow(concat0)

        # print(output_flow.shape)
        if auto_warp:
            warped = self.Resampler(input_frames[:, -1], output_flow)
            return warped, output_flow
        else:
            return None, output_flow


class MotionExtrapolationNet(SDCNet):

    def __init__(self, sequence_length, use_sdc=False, kernel_size=11):
        super(MotionExtrapolationNet, self).__init__(sequence_length, use_sdc, kernel_size)
        self.sequence_length = sequence_length

        factor = 2
        self.input_channels = self.sequence_length * \
            3 + (self.sequence_length - 1) * 2
        self.output_channels = 2
        if use_sdc:
            self.output_channels += kernel_size * 2

        self.conv1 = conv2d(self.input_channels, 64 // factor,
                            kernel_size=7, stride=2)
        self.conv2 = conv2d(64 // factor, 128 // factor,
                            kernel_size=5, stride=2)
        self.conv3 = conv2d(128 // factor, 256 // factor,
                            kernel_size=5, stride=2)
        self.conv3_1 = conv2d(256 // factor, 256 // factor)
        self.conv4 = conv2d(256 // factor, 512 // factor // 2, stride=2)
        self.conv4_1 = conv2d(512 // factor // 2, 512 // factor // 2)
        self.conv5 = conv2d(512 // factor // 2, 512 // factor // 2, stride=2)
        self.conv5_1 = conv2d(512 // factor // 2, 512 // factor // 2)
        self.conv6 = conv2d(512 // factor // 2, 1024 // factor // 2, stride=2)
        self.conv6_1 = conv2d(1024 // factor // 2, 1024 // factor // 2)

        self.deconv5 = deconv2d(1024 // factor // 2, 512 // factor // 2)
        self.deconv4 = deconv2d(1024 // factor // 2, 256 // factor // 2)
        self.deconv3 = deconv2d(768 // factor // 2, 128 // factor)
        self.deconv2 = deconv2d(384 // factor, 64 // factor)
        self.deconv1 = deconv2d(192 // factor, 32 // factor)
        self.deconv0 = deconv2d(96 // factor, 16 // factor)
        
        self.final_flow = nn.Conv2d(self.input_channels + 16 // factor, self.output_channels,
                                    kernel_size=3, stride=1, padding=1, bias=True)

        # init parameters, when doing convtranspose3d, do bilinear init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

        self.flownet = PWCNet()
        self.Resampler = SpatialDisplConv() if use_sdc and sdc_cuda else Resampler()
        self._frame_buffer = None
        self._flow_buffer = None

        data = torch.load(f'./models/SDCNet_3M_ref{sequence_length}.ckpt', map_location='cpu')
        self.load_state_dict(data['model'])

