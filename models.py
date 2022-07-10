import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_compression as trc
import util.functional as FE
from obmc import OPMC
from util.sampler import shift, warp, Resampler


class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

    def forward(self, input):
        return input + super().forward(input)

class UV_Upsampler(nn.Sequential):
    def __init__(self, in_channels=2, num_filters=64):
        super(UV_Upsampler, self).__init__(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, in_channels, 3, padding=1)
        )

    def forward(self, input):
        up_input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        res = super().forward(input)
        return up_input + res, res 

class Flow_Downsampler(nn.Sequential):
    def __init__(self, in_channels=2, num_filters=64):
        super(Flow_Downsampler, self).__init__(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, in_channels, 3, padding=1)
        )

    def forward(self, input):
        # Devide by 2 for down-sampling flow-map
        down_input = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False) / 2
        res = super().forward(input)
        return down_input + res, res


class ResidualBlock2(nn.Module):
    """Builds the residual block"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock2, self).__init__()
        self.residual = nn.Sequential(
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
                    nn.LeakyReLU(inplace=True),
                   )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2) 

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class ShortCutRefinement(nn.Module):
    """ShortCut-Based refinement for multiple refrence"""

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(ShortCutRefinement, self).__init__()
           
        self.main_shortcut = nn.Conv2d(in_channels, num_filters, 1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, num_filters, 3, padding=1)
        self.shortcutblock = nn.Sequential(
            *[ResidualBlock2(num_filters, num_filters, 3) for _ in range(num_layers)])
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv4 = nn.Conv2d(num_filters, out_channels, 3, padding=1)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.shortcutblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + self.main_shortcut(input)
        conv4 = self.conv4(conv3)

        return conv4


class Refinement(nn.Module):
    """Refinement UNet"""

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1


class Post(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters*2),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters*3),
            nn.Conv2d(num_filters*3, num_filters*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*2, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 3, 3, padding=1)
        )

    def forward(self, inputs):
        conv1 = self.l1(inputs)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(torch.cat([deconv3, conv2], dim=1))
        deconv1 = self.d1(torch.cat([deconv2, conv1], dim=1))

        return deconv1 + inputs


class MC_Net(Refinement):
    """MC_Net"""

    def __init__(self, num_filters=64, use_flow=False, stop_flow=False, obmc=False, path=None):
        self.use_flow = use_flow
        self.stop_flow = stop_flow
        in_channels = 6
        if use_flow:
            in_channels += 2
        if obmc:
            in_channels += OPMC.aux_channel[obmc]

        super(MC_Net, self).__init__(in_channels, num_filters)
        self.warp = Resampler() if obmc == 0 else OPMC(obmc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, previous_frame, Flow):
        warped = self.warp(previous_frame, Flow)

        mc_input = [previous_frame, Flow.detach() if self.stop_flow else Flow, warped] if self.use_flow else [
            previous_frame, warped]

        return super().forward(*mc_input)

class Feature_Extractor_RGB(nn.Module):
    def __init__(self, input_features=3, out_features=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, out_features, 3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.act(self.conv1(x))

class MMC_Net(nn.Module):
    """MMC_Net"""
    def __init__(self, input_features=32, num_filters=64, num_MMC_frames=3):
        super().__init__()
        self.num_MMC_frames = num_MMC_frames

        self.warp_layer = Resampler()
        self.f_extractor = Feature_Extractor_RGB(3, 32)
        
        # +3 for MC frame
        self.conv1 = nn.Conv2d(input_features*(num_MMC_frames+1) + 3, num_filters, 3, stride=1, padding=1)
        self.res1 = nn.Sequential(
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1),
                        nn.LeakyReLU(0.1)
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.1)
                    )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.1)
                    )
        self.res2 = nn.Sequential(
                        ResidualBlock(num_filters),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                    )
        self.res3 = ResidualBlock(num_filters)
        self.res4 = nn.Sequential(
                        ResidualBlock(num_filters),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                    )
        self.res5 = nn.Sequential(
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                    )
        self.res6 = nn.Sequential(
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                        ResidualBlock(num_filters),
                    )
        self.conv5 = nn.Conv2d(num_filters, 3, 3, stride=1, padding=1)

    def warp_multiple_frames(self, ref_queue, flow_queue, warp_flows=True, current_order=1):
        '''
            args: 
                ref_queue: previous frames
                flow_queue: previous flows of adjacent frames
                warp_flow: to warp flows before warp frames with them or not
        '''

        # Warp the flows, and warp the frames, and put them into another queue
        ## Tail frame is the closest, so it it warped first
        warped_frame_queue = []
        accumulated_flows = torch.zeros_like(flow_queue[-1])

        ## Warp in reverse order
        for i in range(self.num_MMC_frames):
            # Warp the flow
            if warp_flows:
                current_flow = self.warp_layer(flow_queue[-(i+1)], accumulated_flows)

                # Warp the frame
                warped_frame = self.warp_layer(ref_queue[-(i+1)], current_flow + accumulated_flows)

                # For P-frame index < num_MMC_frames, some flows need not to be accumulated
                if current_order - i > 1:
                    accumulated_flows += current_flow
                    
            else:
                current_flow = flow_queue[-(i+1)]
                accumulated_flows = current_flow
            
                # Warp the frame
                warped_frame = self.warp_layer(ref_queue[-(i+1)], accumulated_flows)

            warped_frame_queue.append(warped_frame)
        
        # Make the order be consistent ; that is, keep warped_frame_queue as reverse orfer
        warped_frame_queue = warped_frame_queue[::-1]

        return warped_frame_queue

    def forward(self, mc_frame, ref_queue, flow_queue, warp_flows=True, current_order=1):
        # Note: frames in ref_queue are assumed to be features that processed by f_extractor
        warped_frame_queue = self.warp_multiple_frames(ref_queue, flow_queue, warp_flows, current_order)

        warped_frames = torch.cat(warped_frame_queue, dim=1)

        x = self.conv1(torch.cat([mc_frame, self.f_extractor(mc_frame), warped_frames], dim=1))
        x = self.res1(x)
        x = self.conv2(x)
        x1 = self.conv3(x)
        x2 = self.conv4(x1)
        x2_up = self.res2(x2)
        x1 = self.res3(x1) + x2_up
        x1_up = self.res4(x1)
        x = self.res5(x) + x1_up
        x = self.res6(x)
        output = self.conv5(x) + mc_frame
        
        return output

use_signalconv = True
simplified_gdn = False


# - Network


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, gdn=True, relu=False, *args, **kwargs):
        super(Conv, self).__init__()
        if use_signalconv:
            conv = trc.SignalConv2d
        else:
            conv = nn.Conv2d
        self.add_module('conv', conv(in_channels, out_channels, kernel_size, stride=stride,
                                     padding=(kernel_size - 1) // 2, *args, **kwargs))
        if gdn:
            self.add_module('gdn', trc.GeneralizedDivisiveNorm(
                out_channels, simplify=simplified_gdn))

        if relu:
            self.add_module('relu', nn.ReLU(inplace=True))


class TransposedConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, igdn=True, relu=False, *args, **kwargs):
        super(TransposedConv, self).__init__()
        if use_signalconv:
            deconv = trc.SignalConvTranspose2d
        else:
            deconv = nn.ConvTranspose2d
            if 'parameterizer' in kwargs:
                kwargs.pop('parameterizer')
        self.add_module('deconv', deconv(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=(kernel_size - 1) // 2, output_padding=stride-1, *args, **kwargs))

        if igdn:
            self.add_module('igdn', trc.GeneralizedDivisiveNorm(
                out_channels, inverse=True, simplify=simplified_gdn))

        if relu:
            self.add_module('relu', nn.ReLU(inplace=True))


class AnalysisTransform(nn.Module):
    """AnalysisTransform"""

    def __init__(self, num_filters, num_features, in_channels=3, kernel_size=5, **kwargs):
        super(AnalysisTransform, self).__init__()

        self.l1 = Conv(in_channels, num_filters, kernel_size, **kwargs)
        self.l2 = Conv(num_filters, num_filters, kernel_size, **kwargs)
        self.l3 = Conv(num_filters, num_filters, kernel_size, **kwargs)
        self.l4 = Conv(num_filters, num_features,
                       kernel_size, gdn=False, **kwargs)

    def forward(self, inputs):
        conv1 = self.l1(inputs)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)
        conv4 = self.l4(conv3)
        return conv4


class SynthesisTransform(nn.Module):
    """SynthesisTransform"""

    def __init__(self, num_filters, num_features, out_channels=3, kernel_size=5, **kwargs):
        super(SynthesisTransform, self).__init__()

        self.d1 = TransposedConv(
            num_features, num_filters, kernel_size, **kwargs)
        self.d2 = TransposedConv(
            num_filters, num_filters, kernel_size, **kwargs)
        self.d3 = TransposedConv(
            num_filters, num_filters, kernel_size, **kwargs)
        self.d4 = TransposedConv(
            num_filters, out_channels, kernel_size, igdn=False, **kwargs)

    def forward(self, inputs):
        deconv1 = self.d1(inputs)
        deconv2 = self.d2(deconv1)
        deconv3 = self.d3(deconv2)
        deconv4 = self.d4(deconv3)
        return deconv4


class HyperAnalysisTransform(nn.Sequential):
    """HyperAnalysisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors, **kwargs):
        super(HyperAnalysisTransform, self).__init__()

        self.add_module('l1', Conv(
            num_features, num_filters, 3, stride=1, gdn=False, relu=True, **kwargs))
        self.add_module('l2', Conv(
            num_filters, num_filters, 5, stride=2, gdn=False, relu=True, **kwargs))
        self.add_module('l3', Conv(
            num_filters, num_hyperpriors, 5, stride=2, gdn=False, relu=False, **kwargs))


class HyperSynthesisTransform(nn.Sequential):
    """HyperSynthesisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors, **kwargs):
        super(HyperSynthesisTransform, self).__init__()

        self.add_module('d1', TransposedConv(
            num_hyperpriors, num_filters, 5, stride=2, igdn=False, relu=True, parameterizer=None, **kwargs))
        self.add_module('d2', TransposedConv(
            num_filters, num_filters, 5, stride=2, igdn=False, relu=True, parameterizer=None, **kwargs))
        self.add_module('d3', TransposedConv(
            num_filters, num_features, 3, stride=1, igdn=False, relu=False, parameterizer=None, **kwargs))


class GoogleFactorizeCoder(nn.Module):
    """GoogleFactorizeCoder"""

    def __init__(self, num_filters, num_features, quant_mode='noise', in_channels=3, out_channels=3, kernel_size=5):
        super(GoogleFactorizeCoder, self).__init__()

        self.analysis = AnalysisTransform(
            num_filters, num_features, in_channels=in_channels, kernel_size=kernel_size)
        self.synthesis = SynthesisTransform(
            num_filters, num_features, out_channels=out_channels, kernel_size=kernel_size)

        self.entropy_bottleneck = trc.EntropyBottleneck(
            num_features, quant_mode=quant_mode)

    def compress(self, input, return_hat=False):
        """Compresses an image."""
        features = self.analysis(input)

        ret = self.entropy_bottleneck.compress(
            features, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream], [features.size()]
        else:
            stream = ret
            return [stream], [features.size()]

    def decompress(self, stream_list, shape_list):
        """Compresses an image."""
        stream = stream_list[0]
        y_shape = shape_list[0]

        y_hat = self.entropy_bottleneck.decompress(
            stream, y_shape)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        y_tilde, y_likelihood = self.entropy_bottleneck(features)

        reconstructed = self.synthesis(y_tilde)

        num_pixels = input.size(2) * input.size(3)
        y_rate = trc.estimate_bpp(y_likelihood, num_pixels)
        return reconstructed, y_rate


class GoogleHyperPriorCoder(nn.Module):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, use_mean=False, use_abs=False,
                 quant_mode='noise', in_channels=3, out_channels=3, kernel_size=5):
        super(GoogleHyperPriorCoder, self).__init__()
        self.use_mean = use_mean
        self.use_abs = use_abs

        self.analysis = AnalysisTransform(
            num_filters, num_features, in_channels=in_channels, kernel_size=kernel_size)
        self.synthesis = SynthesisTransform(
            num_filters, num_features, out_channels=out_channels, kernel_size=kernel_size)
        self.hyper_analysis = HyperAnalysisTransform(
            num_filters, num_features, num_hyperpriors)
        self.hyper_synthesis = HyperSynthesisTransform(
            num_filters, num_features * 2 if use_mean else num_features, num_hyperpriors)

        self.conditional_bottleneck = trc.GaussianConditional(
            use_mean=use_mean, quant_mode=quant_mode)
        self.entropy_bottleneck = trc.EntropyBottleneck(
            num_hyperpriors, quant_mode=quant_mode)

    def compress(self, input, return_hat=False):
        """Compresses an image."""
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if not self.use_mean or self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, stream_list, shape_list):
        """Compresses an image."""
        stream, side_stream = stream_list
        y_shape, z_shape = shape_list

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if not self.use_mean or self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        reconstructed = self.synthesis(y_tilde)

        num_pixels = input.size(2) * input.size(3)
        y_rate = trc.estimate_bpp(y_likelihood, num_pixels)
        z_rate = trc.estimate_bpp(z_likelihood, num_pixels)
        return reconstructed, y_rate + z_rate
