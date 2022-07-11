import os
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm, trange

from dataloader import DataLoader, VideoData
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



def train():

    num_ref, use_sdc = 4, False
    m = SDCNet(num_ref, use_sdc=use_sdc).to(__DEVICE__)
    print(m)

    transformer = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    batch_size, nrow = 32, 8
    dataset = VideoData(os.getenv("DATASET")+"/vimeo_septuplet/",
                        frames=num_ref+1, transform=transformer)
    train_data = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=16)

    optim = torch.optim.Adam(m.parameters(), lr=1e-4)
    from util.psnr import PSNR
    import numpy as np
    psnr_metric = PSNR(reduction='mean')
    suffix = "_ref" + str(num_ref)

    epochs = 1000
    psnrs, losses, iter_list = [], [], []
    pbar = tqdm(total=epochs*len(train_data))
    for epoch in range(epochs):
        for gop in train_data:
            gop = gop.to(__DEVICE__)
            CL, GT_I = gop[:, -2], gop[:, -1]

            optim.zero_grad()
            m.clear_buffer(gop[:, :-1])
            pred_I, pred_flow = m()
            # print(pred_I.shape, pred_flow.shape)

            L_warp = F.l1_loss(pred_I, GT_I)
            L_grad = F.l1_loss(torch.abs(pred_I[..., 1:] - pred_I[..., :-1]),
                               torch.abs(GT_I[..., 1:] - GT_I[..., :-1])) + \
                F.l1_loss(torch.abs(pred_I[..., 1:, :] - pred_I[..., :-1, :]),
                          torch.abs(GT_I[..., 1:, :] - GT_I[..., :-1, :]))
            L_smooth = F.l1_loss(pred_flow[..., 1:], pred_flow[..., :-1]) + \
                F.l1_loss(pred_flow[..., 1:, :],
                          pred_flow[..., :-1, :])

            total_loss = 0.7 * L_warp + 0.2 * L_grad + 0.1 * L_smooth
            total_loss.backward()
            optim.step()

            psnr = psnr_metric(pred_I, GT_I).item()
            psnr_CL = psnr_metric(CL, GT_I).item()
            pbar.set_description_str(
                " SDC{} PSNR: {:.4f}, l1: {:.4e}, CL_gain: {:.4f}".format(suffix, psnr, L_warp.item(), psnr-psnr_CL))
            if pbar.n % 50 == 0:
                psnrs.append(psnr)
                psnrs[-1] = np.mean(psnrs[-10:])
                losses.append(L_warp.item())
                losses[-1] = np.mean(losses[-10:])
                iter_list.append(pbar.n)

            if pbar.n % 2000 == 0:
                images = torch.cat([gop[:nrow], pred_I[:nrow].unsqueeze(1)], 1)
                save_image(images.transpose(0, 1).flatten(0, 1),
                           _TMP+f"SDC_img{suffix}.png", nrow=nrow)
                GT_flow = pwc(CL[:nrow], GT_I[:nrow]).unsqueeze(1)
                flows = torch.cat(
                    [m._flow_buffer[:nrow], GT_flow, pred_flow[:nrow, :2].unsqueeze(1)], 1)
                flowmap = plot_flow(flows.transpose(0, 1).flatten(0, 1))
                save_image(flowmap, _TMP+f"SDC_flow{suffix}.png", nrow=nrow)

                plots_count = 2
                graph, subplots = plt.subplots(plots_count, sharex=True,
                                               figsize=(6.4, 4.2*plots_count/2))
                subplots[0].plot(iter_list, psnrs)
                subplots[0].set_title("train PSNR")
                subplots[1].plot(iter_list, losses)
                subplots[1].set_title("train loss(L1)")
                for pid in range(1, plots_count):
                    subplots[0].get_shared_x_axes().join(
                        subplots[0], subplots[pid])
                plt.subplots_adjust(hspace=0.2)
                plt.savefig(_TMP+f"SDC_loss{suffix}.png")
                plt.close(graph)

            if pbar.n % 10000 == 0:
                torch.save({"model": m.state_dict(),
                            "optim": optim.state_dict()}, f"./models/SDCNet{suffix}.ckpt")

            pbar.update()


def test():

    num_ref = 4
    m = SDCNet(num_ref, use_sdc=False).to(__DEVICE__)
    print(m)

    shape = (128, 128)
    gop = torch.rand(2, 5, 3, *shape)
    print(gop.shape)
    f = pwc(gop[:4], gop[1:])
    print(f.shape)
    ofinput = f[:3].transpose(0, 1).flatten(2)
    print(ofinput.shape)
    GT = f[-1:]
    print(GT.max())
    flowmap = plot_flow(f)
    save_image(flowmap, _TMP+"input_flow.png")

    # m._frame_buffer = gop.repeat(2, 1, 1, 1, 1)
    im1s = gop.unsqueeze(0)[:, :-2].flatten(0, 1)
    im2s = gop.unsqueeze(0)[:, 1:-1].flatten(0, 1)
    flows = m.flownet(im1s, im2s).view(1, 3, 2, *shape)
    print(flows.shape)
    flowmap = plot_flow(flows.transpose(0, 1).flatten(0, 1))
    save_image(flowmap, _TMP+"input_flow2.png")

    t0 = perf_counter()
    iter = 10000
    for _ in trange(iter):
        m.clear_buffer()
        m.append_buffer(gop[:, 0])
        m.append_buffer(gop[:, 1])
        m.append_buffer(gop[:, 2])

        pred_I, pred_flow = m()
        # print(pred_I.shape, pred_flow.shape)

        # m.append_buffer(t4)

        # pred_I, pred_flow = m()
        # print(pred_I.shape, pred_flow.shape)

    print((perf_counter() - t0)/iter)


if __name__ == "__main__":
    _TMP = './tmp/'
    os.makedirs(_TMP, exist_ok=True)
    torch.random.manual_seed(666)
    torch.cuda.manual_seed(666)
    CIA = torch.cuda.is_available()
    __DEVICE__ = torch.device("cuda:0" if CIA else "cpu")
    torch.backends.cudnn.benchmark = True
    pwc = PWCNet().to(__DEVICE__)
    plot_flow = PlotFlow().to(__DEVICE__)
