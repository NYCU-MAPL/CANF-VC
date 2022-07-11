import argparse
import os
import csv
from functools import partial

import yaml
import flowiz as fz
import numpy as np
import torch
import torch_compression as trc

from torch import nn, optim
from torch.utils.data import DataLoader
from entropy_models import EntropyBottleneck
from networks import AugmentedNormalizedFlowHyperPriorCoder
from torchvision import transforms

from dataloader import VideoTestDataIframe
from flownets import PWCNet, SPyNet
from SDCNet import MotionExtrapolationNet
from models import Refinement
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM

from ptflops import get_model_complexity_info


class CompressModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).mean()

class Pframe(CompressModel):
    def __init__(self, args, mo_coder, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_QE=True, use_affine=False,
                                                               use_context=True, condition='GaussianMixtureModel', quant_mode='round')
        if self.args.MENet == 'PWC':
            self.MENet = PWCNet(trainable=False)
        elif self.args.MENet == 'SPy':
            self.MENet = SPyNet(trainable=False)

        self.MWNet = MotionExtrapolationNet(sequence_length=3)
        self.MWNet.__delattr__('flownet')

        self.Motion = mo_coder
        self.CondMotion = cond_mo_coder

        self.Resampler = Resampler()
        self.MCNet = Refinement(6, 64, out_channels=3)

        self.Residual = res_coder
        self.frame_buffer = list()
        self.flow_buffer = list()

    def load_args(self, args):
        self.args = args

    def motion_forward(self, ref_frame, coding_frame, p_order=1):
        predict = p_order > 1 # To generate extrapolated motion for conditional motion coding or not ; "No" for first P frame
        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2
            
            # Update frame buffer ; motion (flow) buffer will be updated in SDCNet
            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]

            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            pred_frame, pred_flow = self.MWNet(frame_buffer, self.flow_buffer if len(self.flow_buffer) == 2 else None, True)
            
            # Encode motion condioning on extrapolated motion
            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m, pred_flow_hat, _, _, _ = self.CondMotion(flow, output=pred_flow, 
                                                                             cond_coupling_input=pred_flow, 
                                                                             pred_prior_input=pred_frame)
            self.MWNet.append_flow(flow_hat)

            warped_frame = self.Resampler(ref_frame, flow_hat)
            mc_frame = self.MCNet(ref_frame, warped_frame)

        else: # No motion extrapolation is performed for first P frame
            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)

            warped_frame = self.Resampler(ref_frame, flow_hat)
            mc_frame = self.MCNet(ref_frame, warped_frame)

            self.MWNet.append_flow(flow_hat.detach())

        return mc_frame, likelihood_m

    def forward(self, ref_frame, coding_frame, p_order=1):
        mc_frame, likelihood_m = self.motion_forward(ref_frame, coding_frame, p_order)

        reconstructed, likelihood_r, mc_hat, _, _, BDQ = self.Residual(coding_frame, output=mc_frame, cond_coupling_input=mc_frame)

        likelihoods = likelihood_m + likelihood_r
        
        reconstructed = reconstructed.clamp(0, 1)
        return reconstructed, likelihoods

    def test(self):
        outputs = []
        for batch, batch_idx in enumerate(self.test_dataloader):
            outputs.append(self.test_step(batch, batch_idx))
        
        self.test_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame = batch[:, 0] # I-frame by BPG in position 0
        batch = batch[:, 1:] # GT
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)

        psnr_list = []
        rate_list = []
        m_rate_list = []
        log_list = []

        # To align frame into multiplications of 64 ; zero-padding is performed
        align = trc.util.Alignment()
        
        # Clear motion buffer & frame buffer
        self.MWNet.clear_buffer()
        self.frame_buffer = list()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]
                
                rec_frame, likelihoods = self(align.align(ref_frame), align.align(coding_frame), frame_idx)

                rec_frame = rec_frame.clamp(0, 1)
                self.frame_buffer.append(rec_frame)
                
                # Update frame buffer
                if len(self.frame_buffer) == 4:
                    self.frame_buffer.pop(0)
                    assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

                likelihoods = likelihood_m + likelihood_r
                
                # Back to original resolution
                rec_frame = align.resume(rec_frame)
                ref_frame = align.resume(ref_frame)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()
                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()

                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                # likelihoods[0] & [1] are motion latent & hyper likelihood
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()
                metrics['Mo_Rate'].append(m_rate)

                # Make reconstruction as next reference frame
                ref_frame = rec_frame

                log_list.append({'PSNR': psnr, 'Rate': rate, 'Mo_Rate': m_rate,
                                 'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                 'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item()})

            else: 
                with torch.no_grad():
                    rec_frame, likelihoods, _, _, _, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                ref_frame = rec_frame

                log_list.append({'PSNR': psnr, 'Rate': rate})

            metrics['PSNR'].append(psnr)
            metrics['Rate'].append(rate)

            frame_id += 1

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}

        return {'test_log': logs}


    def test_epoch_end(self, outputs):

        metrics_name = list(outputs[0]['test_log']['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in [log['test_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])

        os.makedirs(self.args.logs_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.logs_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name)

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        print(print_log)

        with open(self.args.logs_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        self.log_dict(logs)


    def compress(self, ref_frame, coding_frame, p_order):
        # TODO
        flow = self.MENet(ref_frame, coding_frame)

        flow_hat, mv_strings, mv_shape = self.Motion.compress(flow, return_hat=True, p_order=p_order)

        strings, shapes = [mv_strings], [mv_shape]

        mc_frame = self.mc_net_forward(ref_frame, flow_hat)

        predicted = mc_frame

        #res = coding_frame - predicted
        reconstructed, res_strings, res_shape = self.Residual.compress(coding_frame, mc_frame, return_hat=True)
        #reconstructed = predicted + res_hat
        strings.append(res_strings)
        shapes.append(res_shape)

        return reconstructed, strings, shapes

    def decompress(self, ref_frame, strings, shapes, p_order):
        # TODO
        mv_strings = strings[0]
        mv_shape = shapes[0]

        flow_hat = self.Motion.decompress(mv_strings, mv_shape, p_order=p_order)

        mc_frame = self.mc_net_forward(ref_frame, flow_hat)

        predicted = mc_frame
        res_strings, res_shape = strings[1], shapes[1]

        reconstructed = self.Residual.decompress(res_strings, res_shape)
        #reconstructed = predicted + res_hat

        return reconstructed

    def setup(self, stage):
        qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[self.args.lmda]

        self.test_dataset = VideoTestDataIframe(self.args.data_dir, self.args.lmda, sequence=('U', 'B', 'M'), GOP=32)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      num_workers=self.args.num_workers,
                                      shuffle=False)


if __name__ == '__main__':
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well
    seed = 888888
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    ramdom.seed(seed)
    np.ramdom.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    save_root = os.getenv('LOG', './') + 'torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # testing specific
    parser.add_argument('--lmda', default=2048, choices=[256, 512, 1024, 2048, 4096], type=int)
    parser.add_argument('--ssim', action="store_true")
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model_dir', default='./models/CANF-VC', type=str)
    parser.add_argument('--logs_dir', default='./logs', type=str)
    parser.add_argument('--data_dir', default='./video_dataset', type=str)

    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='PWC')
    parser.add_argument('--motion_coder_conf', type=str, default='./config/DVC_motion.yml')
    parser.add_argument('--cond_motion_coder_conf', type=str, default='./config/CANF_motion_predprior.yml')
    parser.add_argument('--residual_coder_conf', type=str, default='./config/CANF_inter_coder.yml')
    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()
 
    # Config codecs
    assert not (args.motion_coder_conf is None)
    mo_coder_cfg = yaml.safe_load(open(args.motion_coder_conf, 'r'))
    assert mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    mo_coder_arch = trc.__CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])
 
    assert not (args.cond_motion_coder_conf is None)
    cond_mo_coder_cfg = yaml.safe_load(open(args.cond_motion_coder_conf, 'r'))
    assert cond_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    cond_mo_coder_arch = trc.__CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])

    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    res_coder_arch = trc.__CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])

    db = None
    if args.gpus > 1:
        db = 'ddp'
                                         
    checkpoint = torch.load(os.path.join(args.model_dir, f"{args.lmda}.ckpt"), map_location=(lambda storage, loc: storage))

    model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    model.test()
