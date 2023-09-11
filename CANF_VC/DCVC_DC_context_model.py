import torch
import torch.nn as nn
import torch.nn.functional as F

from entropy_models import SymmetricConditional, ac, estimate_bpp 


###
#    Reference: Neural Video Compression with Diverse Contexts, CVPR 2023
#    Link: https://github.com/microsoft/DCVC/blob/main/DCVC-DC/src/models/common_model.py
###

def get_mask_four_parts(height, width, dtype, device):
    masks = {}
    curr_mask_str = f"{width}x{height}"
    if curr_mask_str not in masks:
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)
        masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]

    return masks[curr_mask_str]

def process_with_mask(y, scales, means, mask, training):
    def quant(x, training):
        if training:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
    
        return torch.round(x)

    scales_hat = scales * mask
    means_hat = means * mask
    
    # d_type = y.type()
    y_res = (y - means_hat.to(torch.float64)) * mask

    y_res = y_res.to(torch.float)
    y_q = quant(y_res, training)
    y_hat = y_q + means_hat
    y_hat = y_hat * mask

    y = y * mask

    return y, y_hat, means_hat, scales_hat

def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                       x_1_0, x_1_1, x_1_2, x_1_3,
                       x_2_0, x_2_1, x_2_2, x_2_3,
                       x_3_0, x_3_1, x_3_2, x_3_3):
    x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
    x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
    x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
    x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
    return torch.cat((x_0, x_1, x_2, x_3), dim=1)


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class QuadtreePartitionBasedContextModel(nn.Module):
    def __init__(self, num_features, entropy_model, inplace=False):
        super(QuadtreePartitionBasedContextModel, self).__init__()
        self.num_features = num_features

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(num_features * 3, num_features * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(num_features * 3, num_features * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(num_features * 3, num_features * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(num_features * 3, num_features * 3, inplace=inplace),
            DepthConvBlock(num_features * 3, num_features * 3, inplace=inplace),
            DepthConvBlock(num_features * 3, num_features * 2, inplace=inplace),
        )

        assert isinstance(
            entropy_model, SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model

    def gaussian_conditional(self, y, scales, means):
        return self.entropy_model(y, torch.cat([means, scales], dim=1))

    def forward(self, input, condition):
        y = input
        means, scales = condition.chunk(2, 1)
        gaussian_params = condition
        
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = get_mask_four_parts(H, W, dtype, device)
        
        # Divide latent into 4 groups along the channel dimension
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)
        
        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        final_likelihood_y0 = torch.zeros_like(y_0)
        final_likelihood_y1 = torch.zeros_like(y_1)
        final_likelihood_y2 = torch.zeros_like(y_2)
        final_likelihood_y3 = torch.zeros_like(y_3)

        # Step 1
        y_0_0, y_hat_0_0, m_0_0, s_0_0 = \
            process_with_mask(y_0, scales_0, means_0, mask_0, self.training)
        y_1_1, y_hat_1_1, m_1_1, s_1_1 = \
            process_with_mask(y_1, scales_1, means_1, mask_1, self.training)
        y_2_2, y_hat_2_2, m_2_2, s_2_2 = \
            process_with_mask(y_2, scales_2, means_2, mask_2, self.training)
        y_3_3, y_hat_3_3, m_3_3, s_3_3 = \
            process_with_mask(y_3, scales_3, means_3, mask_3, self.training)

        _, lkd = self.gaussian_conditional(y_0_0 + y_1_1 + y_2_2 + y_3_3, 
                                           s_0_0 + s_1_1 + s_2_2 + s_3_3,
                                           m_0_0 + m_1_1 + m_2_2 + m_3_3)

        final_likelihood_y0 += lkd * mask_0 
        final_likelihood_y1 += lkd * mask_1 
        final_likelihood_y2 += lkd * mask_2 
        final_likelihood_y3 += lkd * mask_3 

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params)).chunk(8, 1)

            
        # Step 2    
        y_0_3, y_hat_0_3, m_0_3, s_0_3 = \
            process_with_mask(y_0, scales_0, means_0, mask_3, self.training)
        y_1_2, y_hat_1_2, m_1_2, s_1_2 = \
            process_with_mask(y_1, scales_1, means_1, mask_2, self.training)
        y_2_1, y_hat_2_1, m_2_1, s_2_1 = \
            process_with_mask(y_2, scales_2, means_2, mask_1, self.training)
        y_3_0, y_hat_3_0, m_3_0, s_3_0 = \
            process_with_mask(y_3, scales_3, means_3, mask_0, self.training)

        _, lkd = self.gaussian_conditional(y_0_3 + y_1_2 + y_2_1 + y_3_0, 
                                           s_0_3 + s_1_2 + s_2_1 + s_3_0,
                                           m_0_3 + m_1_2 + m_2_1 + m_3_0)

        final_likelihood_y0 += lkd * mask_3
        final_likelihood_y1 += lkd * mask_2 
        final_likelihood_y2 += lkd * mask_1 
        final_likelihood_y3 += lkd * mask_0 

        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        
            
        # Step 3
        y_0_2, y_hat_0_2, m_0_2, s_0_2 = \
            process_with_mask(y_0, scales_0, means_2, mask_2, self.training)
        y_1_3, y_hat_1_3, m_1_3, s_1_3 = \
            process_with_mask(y_1, scales_1, means_3, mask_3, self.training)
        y_2_0, y_hat_2_0, m_2_0, s_2_0 = \
            process_with_mask(y_2, scales_2, means_0, mask_0, self.training)
        y_3_1, y_hat_3_1, m_3_1, s_3_1 = \
            process_with_mask(y_3, scales_3, means_1, mask_1, self.training)

        _, lkd = self.gaussian_conditional(y_0_2 + y_1_3 + y_2_0 + y_3_1, 
                                           s_0_2 + s_1_3 + s_2_0 + s_3_1,
                                           m_0_2 + m_1_3 + m_2_0 + m_3_1)

        final_likelihood_y0 += lkd * mask_2 
        final_likelihood_y1 += lkd * mask_3 
        final_likelihood_y2 += lkd * mask_0 
        final_likelihood_y3 += lkd * mask_1 

        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params)).chunk(8, 1)
            

        # Step 4
        y_0_1, y_hat_0_1, m_0_1, s_0_1 = \
            process_with_mask(y_0, scales_0, means_1, mask_1, self.training)
        y_1_0, y_hat_1_0, m_1_0, s_1_0 = \
            process_with_mask(y_1, scales_1, means_0, mask_0, self.training)
        y_2_3, y_hat_2_3, m_2_3, s_2_3 = \
            process_with_mask(y_2, scales_2, means_3, mask_3, self.training)
        y_3_2, y_hat_3_2, m_3_2, s_3_2 = \
            process_with_mask(y_3, scales_3, means_2, mask_2, self.training)

        _, lkd = self.gaussian_conditional(y_0_1 + y_1_0 + y_2_3 + y_3_2, 
                                           s_0_1 + s_1_0 + s_2_3 + s_3_2,
                                           m_0_1 + m_1_0 + m_2_3 + m_3_2)

        final_likelihood_y0 += lkd * mask_1
        final_likelihood_y1 += lkd * mask_0 
        final_likelihood_y2 += lkd * mask_3 
        final_likelihood_y3 += lkd * mask_2 
   
        # Combine all together
        y_hat = combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                   y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                   y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                   y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        
        y_likelihoods = torch.cat((final_likelihood_y0,
                                   final_likelihood_y1,
                                   final_likelihood_y2,
                                   final_likelihood_y3), dim=1)
        
        return y_hat, y_likelihoods
    
    @torch.no_grad()
    def compress(self, input, condition, return_sym=False):
        y = input
        means, scales = condition.chunk(2, 1)
        gaussian_params = condition
        
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = get_mask_four_parts(H, W, dtype, device)
        
        # Divide latent into 4 groups along the channel dimension
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)
        
        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        # Step 1
        y_0_0, _, m_0_0, s_0_0 = \
            process_with_mask(y_0, scales_0, means_0, mask_0, self.training)
        y_1_1, _, m_1_1, s_1_1 = \
            process_with_mask(y_1, scales_1, means_1, mask_1, self.training)
        y_2_2, _, m_2_2, s_2_2 = \
            process_with_mask(y_2, scales_2, means_2, mask_2, self.training)
        y_3_3, _, m_3_3, s_3_3 = \
            process_with_mask(y_3, scales_3, means_3, mask_3, self.training)
        
        grouped_y_0 = y_0_0 + y_1_1 + y_2_2 + y_3_3
        cond_0 = torch.cat([m_0_0 + m_1_1 + m_2_2 + m_3_3, \
                            s_0_0 + s_1_1 + s_2_2 + s_3_3], dim=1)

        string_0, y_hat_0 = self.entropy_model.compress(
            grouped_y_0, condition=cond_0, return_sym=True)

        y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3 = y_hat_0 * mask_0, y_hat_0 * mask_1, y_hat_0 * mask_2, y_hat_0 * mask_3

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params)).chunk(8, 1)

            
        # Step 2    
        y_0_3, _, m_0_3, s_0_3 = \
            process_with_mask(y_0, scales_0, means_0, mask_3, self.training)
        y_1_2, _, m_1_2, s_1_2 = \
            process_with_mask(y_1, scales_1, means_1, mask_2, self.training)
        y_2_1, _, m_2_1, s_2_1 = \
            process_with_mask(y_2, scales_2, means_2, mask_1, self.training)
        y_3_0, _, m_3_0, s_3_0 = \
            process_with_mask(y_3, scales_3, means_3, mask_0, self.training)

        grouped_y_1 = y_0_3 + y_1_2 + y_2_1 + y_3_0
        cond_1 = torch.cat([m_0_3 + m_1_2 + m_2_1 + m_3_0, \
                            s_0_3 + s_1_2 + s_2_1 + s_3_0], dim=1)

        string_1, y_hat_1 = self.entropy_model.compress(
            grouped_y_1, condition=cond_1, return_sym=True)

        y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0 = y_hat_1 * mask_3, y_hat_1 * mask_2, y_hat_1 * mask_1, y_hat_1 * mask_0

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        
            
        # Step 3
        y_0_2, _, m_0_2, s_0_2 = \
            process_with_mask(y_0, scales_0, means_2, mask_2, self.training)
        y_1_3, _, m_1_3, s_1_3 = \
            process_with_mask(y_1, scales_1, means_3, mask_3, self.training)
        y_2_0, _, m_2_0, s_2_0 = \
            process_with_mask(y_2, scales_2, means_0, mask_0, self.training)
        y_3_1, _, m_3_1, s_3_1 = \
            process_with_mask(y_3, scales_3, means_1, mask_1, self.training)
        
        grouped_y_2 = y_0_2 + y_1_3 + y_2_0 + y_3_1
        cond_2 = torch.cat([m_0_2 + m_1_3 + m_2_0 + m_3_1, \
                            s_0_2 + s_1_3 + s_2_0 + s_3_1], dim=1)

        string_2, y_hat_2 = self.entropy_model.compress(
            grouped_y_2, condition=cond_2, return_sym=True)

        y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1 = y_hat_2 * mask_2, y_hat_2 * mask_3, y_hat_2 * mask_0, y_hat_2 * mask_1

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params)).chunk(8, 1)
            

        # Step 4
        y_0_1, _, m_0_1, s_0_1 = \
            process_with_mask(y_0, scales_0, means_1, mask_1, self.training)
        y_1_0, _, m_1_0, s_1_0 = \
            process_with_mask(y_1, scales_1, means_0, mask_0, self.training)
        y_2_3, _, m_2_3, s_2_3 = \
            process_with_mask(y_2, scales_2, means_3, mask_3, self.training)
        y_3_2, _, m_3_2, s_3_2 = \
            process_with_mask(y_3, scales_3, means_2, mask_2, self.training)

        grouped_y_3 = y_0_1 + y_1_0 + y_2_3 + y_3_2
        cond_3 = torch.cat([m_0_1 + m_1_0 + m_2_3 + m_3_2, \
                            s_0_1 + s_1_0 + s_2_3 + s_3_2], dim=1)

        string_3, y_hat_3 = self.entropy_model.compress(
            grouped_y_3, condition=cond_3, return_sym=True)

        y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2 = y_hat_3 * mask_1, y_hat_3 * mask_0, y_hat_3 * mask_3, y_hat_3 * mask_2

        # Combine all together
        y_hat = combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                   y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                   y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                   y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)

        strings = [string_0, string_1, string_2, string_3]

        if return_sym:
            return strings, y_hat
        else:
            return strings

    @torch.no_grad()
    def decompress(self, strings, shape, condition):
        means, scales = condition.chunk(2, 1)
        gaussian_params = condition
        
        dtype = condition.dtype
        device = condition.device
        _, _, H, W = condition.size()
        mask_0, mask_1, mask_2, mask_3 = get_mask_four_parts(H, W, dtype, device)
        
        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)
        string_0, string_1, string_2, string_3 = strings

        # Step 1
        m_0_0, m_1_1, m_2_2, m_3_3 = means_0 * mask_0,  means_1 * mask_1, means_2 * mask_2, means_3 * mask_3
        s_0_0, s_1_1, s_2_2, s_3_3 = scales_0 * mask_0,  scales_1 * mask_1, scales_2 * mask_2, scales_3 * mask_3

        cond_0 = torch.cat([m_0_0 + m_1_1 + m_2_2 + m_3_3, \
                            s_0_0 + s_1_1 + s_2_2 + s_3_3], dim=1)

        y_hat_0 = self.entropy_model.decompress(
            string_0, shape, condition=cond_0)

        y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3 = y_hat_0 * mask_0, y_hat_0 * mask_1, y_hat_0 * mask_2, y_hat_0 * mask_3

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params)).chunk(8, 1)

            
        # Step 2    
        m_0_3, m_1_2, m_2_1, m_3_0 = means_0 * mask_3,  means_1 * mask_2, means_2 * mask_1, means_3 * mask_0
        s_0_3, s_1_2, s_2_1, s_3_0 = scales_0 * mask_3,  scales_1 * mask_2, scales_2 * mask_1, scales_3 * mask_0

        cond_1 = torch.cat([m_0_3 + m_1_2 + m_2_1 + m_3_0, \
                            s_0_3 + s_1_2 + s_2_1 + s_3_0], dim=1)

        y_hat_1 = self.entropy_model.decompress(
            string_1, shape, condition=cond_1)

        y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0 = y_hat_1 * mask_3, y_hat_1 * mask_2, y_hat_1 * mask_1, y_hat_1 * mask_0

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        
            
        # Step 3
        m_0_2, m_1_3, m_2_0, m_3_1 = means_2 * mask_2,  means_3 * mask_3, means_0 * mask_0, means_1 * mask_1
        s_0_2, s_1_3, s_2_0, s_3_1 = scales_0 * mask_2,  scales_1 * mask_3, scales_2 * mask_0, scales_3 * mask_1

        cond_2 = torch.cat([m_0_2 + m_1_3 + m_2_0 + m_3_1, \
                            s_0_2 + s_1_3 + s_2_0 + s_3_1], dim=1)

        y_hat_2 = self.entropy_model.decompress(
            string_2, shape, condition=cond_2)

        y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1 = y_hat_2 * mask_2, y_hat_2 * mask_3, y_hat_2 * mask_0, y_hat_2 * mask_1

        # cat the four parts
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, gaussian_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params)).chunk(8, 1)
            

        # Step 4
        m_0_1, m_1_0, m_2_3, m_3_2 = means_1 * mask_1,  means_0 * mask_0, means_3 * mask_3, means_2 * mask_2
        s_0_1, s_1_0, s_2_3, s_3_2 = scales_0 * mask_1,  scales_1 * mask_0, scales_2 * mask_3, scales_3 * mask_2

        cond_3 = torch.cat([m_0_1 + m_1_0 + m_2_3 + m_3_2, \
                            s_0_1 + s_1_0 + s_2_3 + s_3_2], dim=1)

        y_hat_3 = self.entropy_model.decompress(
            string_3, shape, condition=cond_3)

        y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2 = y_hat_3 * mask_1, y_hat_3 * mask_0, y_hat_3 * mask_3, y_hat_3 * mask_2

        # Combine all together
        y_hat = combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                   y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                   y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                   y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)

        return y_hat
