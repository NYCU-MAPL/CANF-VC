import argparse
import os
import csv
from functools import partial

import yaml
import comet_ml
import flowiz as fz
import numpy as np
import torch
import torch_compression as trc

from torchinfo import summary
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.modules.entropy_models import EntropyBottleneck
from torch_compression.hub import AugmentedNormalizedFlowHyperPriorCoder
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import VideoDataIframe, VideoTestDataIframe
from flownets import PWCNet, SPyNet
from SDCNet import SDCNet_3M
from models import Refinement
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM
from util.vision import PlotFlow, PlotHeatMap, save_image

from ptflops import get_model_complexity_info
from thop import profile

plot_flow = PlotFlow().cuda()


class CompressesModel(LightningModule):
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

class Pframe(CompressesModel):
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

        self.MWNet = SDCNet_3M(sequence_length=3) # Motion warping network
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

    def motion_forward(self, ref_frame, coding_frame, visual=False, visual_prefix='', predict=False):

        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2

            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]

            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            pred_frame, pred_flow = self.MWNet(frame_buffer,
                                               self.flow_buffer if len(self.flow_buffer) == 2 else None, True)

            flow = self.MENet(ref_frame, coding_frame)
            #flow_hat, likelihood_m = self.CondMotion(flow)
            flow_hat, likelihood_m, pred_flow_hat, _, _, _ = self.CondMotion(flow, output=pred_flow, 
                                                                             cond_coupling_input=pred_flow, 
                                                                             pred_prior_input=pred_frame,#)
                                                                             visual=visual, figname=visual_prefix+'_motion')

            self.MWNet.append_flow(flow_hat.detach())

            warped_frame = self.Resampler(ref_frame, flow_hat)
            mc_frame = self.MCNet(ref_frame, warped_frame)

            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 
                    'flow': flow, 'flow_hat': flow_hat, 'mc_frame': mc_frame, 
                    'pred_frame': pred_frame, 'pred_flow': pred_flow, 
                    'pred_flow_hat': pred_flow_hat, 'warped_frame': warped_frame}

        else:
            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)

            warped_frame = self.Resampler(ref_frame, flow_hat)
            mc_frame = self.MCNet(ref_frame, warped_frame)

            self.MWNet.append_flow(flow_hat.detach())

            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 'flow': flow, 'flow_hat': flow_hat, 'mc_frame': mc_frame, 'warped_frame': warped_frame}

        return mc_frame, likelihoods, data

    def forward(self, ref_frame, coding_frame, p_order=0, visual=False, visual_prefix='', predict=False):
        mc, likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, visual=visual, visual_prefix=visual_prefix, predict=predict)

        predicted, intra_info, likelihood_i = mc, 0, ()

        reconstructed, likelihood_r, mc_hat, _, _, BDQ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc,
                                                                       visual=visual, figname=visual_prefix)

        likelihoods = likelihood_m + likelihood_i + likelihood_r
        
        reconstructed = reconstructed.clamp(0, 1)
        return reconstructed, likelihoods, None, m_info['flow_hat'], mc, predicted, intra_info, BDQ, mc_hat

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCrec-PSNR', 'MCerr-PSNR', 'BDQ-PSNR', 'QE-PSNR',
                        'back-PSNR', 'p1-PSNR', 'p1-BDQ-PSNR']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []
        # PSNR: PSNR(gt, ADQ)
        # Rate
        # Mo_Rate: Motion Rate
        # MC-PSNR: PSNR(gt, mc_frame)
        # MCrec-PSNR: PSNR(gt, x_2)
        # MCerr-PSNR: PSNR(x_2, mc_frame)
        # BDQ-PSNR: PSNR(gt, BDQ)
        # QE-PSNR: PSNR(BDQ, ADQ)
        # back-PSNR: PSNR(mc_frame, BDQ)
        # p1-PSNR: PSNR(gt, ADQ) only when first P-frame in a GOP
        # p1-BDQ-PSNR: PSNR(gt, BDQ) only when first P-frame in a GOP

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame = batch[:, 0]  # Put reference frame in first position
        batch = batch[:, 1:]  # GT
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)

        psnr_list = []
        mc_psnr_list = []
        mc_hat_psnr_list = []
        BDQ_psnr_list = []
        rate_list = []
        m_rate_list = []
        log_list = []
        align = trc.util.Alignment()

        self.MWNet.clear_buffer()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            TO_VISUALIZE = False and frame_id_start == 1 and frame_idx < 8 #and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                # reconstruced frame will be next ref_frame
                if TO_VISUALIZE:
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_ANFIC', f'batch_{batch_idx}'),
                                exist_ok=True)
                    if frame_idx == 1:
                        self.frame_buffer = [align.align(ref_frame)]
                        mc_frame, likelihood_m, data = self.motion_forward(align.align(ref_frame),
                                                                           align.align(coding_frame), predict=False)
                    else:
                        mc_frame, likelihood_m, data = self.motion_forward(align.align(ref_frame),
                                                                           align.align(coding_frame), predict=True)

                    rec_frame, likelihood_r, mc_hat, _, _, BDQ = self.Residual(align.align(coding_frame), output=mc_frame,
                                                                               cond_coupling_input=mc_frame,
                                                                               visual=True, 
                                                                               figname=os.path.join(
                                                                                    self.args.save_dir,
                                                                                    'visualize_ANFIC',
                                                                                    f'batch_{batch_idx}',
                                                                                    f'frame_{frame_idx}',
                                                                               ))
                else:
                    # continue
                    if frame_idx == 1:
                        self.frame_buffer = [align.align(ref_frame)]
                        mc_frame, likelihood_m, data = self.motion_forward(align.align(ref_frame),
                                                                           align.align(coding_frame), predict=False)
                    else:
                        mc_frame, likelihood_m, data = self.motion_forward(align.align(ref_frame),
                                                                           align.align(coding_frame), predict=True)

                    rec_frame, likelihood_r, mc_hat, _, _, BDQ = self.Residual(align.align(coding_frame), output=mc_frame,
                                                                               cond_coupling_input=mc_frame)

                rec_frame = rec_frame.clamp(0, 1)
                self.frame_buffer.append(rec_frame.detach())

                if len(self.frame_buffer) == 4:
                    self.frame_buffer.pop(0)
                    assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

                likelihoods = likelihood_m + likelihood_r

                ref_frame = rec_frame.detach()
                ref_frame = align.resume(ref_frame)
                mc_frame = align.resume(mc_frame)
                warped_frame = align.resume(data['warped_frame'])
                coding_frame = align.resume(coding_frame)
                mc_hat = align.resume(mc_hat)
                BDQ = align.resume(BDQ)

                res_frame = coding_frame - mc_frame
                res_frame = torch.abs(res_frame)
                res_frame = res_frame.clamp(0., 1.)

                os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/res_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/pred_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/BDQ', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)

                if TO_VISUALIZE:
                    flow_map = plot_flow(data['flow_hat'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png',
                               nrow=1)

                    if frame_idx > 1:
                        flow_map = plot_flow(data['pred_flow'])
                        save_image(flow_map,
                                   self.args.save_dir + f'/{seq_name}/flow/'
                                                        f'frame_{int(frame_id_start + frame_idx)}_flow_pred.png',
                                   nrow=1)

                        flow_map = plot_flow(data['pred_flow_hat'])
                        save_image(flow_map,
                                   self.args.save_dir + f'/{seq_name}/flow/'
                                                        f'frame_{int(frame_id_start + frame_idx)}_flow_pred_hat.png',
                                   nrow=1)

                        pred_frame = align.resume(data['pred_frame'])
                        save_image(pred_frame, self.args.save_dir + f'/{seq_name}/pred_frame/'
                                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                                     f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warped_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                                   f'frame_{int(frame_id_start + frame_idx)}_bmc.png')
                    save_image(ref_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(res_frame[0], self.args.save_dir + f'/{seq_name}/res_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(BDQ[0], self.args.save_dir + f'/{seq_name}/BDQ/'
                                                            f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_hat[0], self.args.save_dir + f'/{seq_name}/mc_hat/'
                                                               f'frame_{int(frame_id_start + frame_idx)}.png')

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                # likelihoods[0] & [1] are motion latent & hyper likelihood
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()
                metrics['Mo_Rate'].append(m_rate)

                mc_psnr = mse2psnr(self.criterion(mc_frame, coding_frame).mean().item())
                metrics['MC-PSNR'].append(mc_psnr)

                mc_rec_psnr = mse2psnr(self.criterion(mc_hat, coding_frame).mean().item())
                metrics['MCrec-PSNR'].append(mc_rec_psnr)

                mc_err_psnr = mse2psnr(self.criterion(mc_frame, mc_hat).mean().item())
                metrics['MCerr-PSNR'].append(mc_err_psnr)

                BDQ_psnr = mse2psnr(self.criterion(BDQ, coding_frame).mean().item())
                metrics['BDQ-PSNR'].append(BDQ_psnr)

                QE_psnr = mse2psnr(self.criterion(BDQ, ref_frame).mean().item())
                metrics['QE-PSNR'].append(QE_psnr)

                back_psnr = mse2psnr(self.criterion(mc_frame, BDQ).mean().item())
                metrics['back-PSNR'].append(back_psnr)

                if frame_idx == 1:
                    metrics['p1-PSNR'].append(psnr)
                    metrics['p1-BDQ-PSNR'].append(BDQ_psnr)

                log_list.append({'PSNR': psnr, 'Rate': rate, 'MC-PSNR': mc_psnr,
                                 'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                 'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item(),
                                 'MCerr-PSNR': mc_err_psnr, 'BDQ-PSNR': BDQ_psnr})

            else: 
                with torch.no_grad():
                    rec_frame, likelihoods, _, _, _, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                ref_frame = rec_frame

                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                if TO_VISUALIZE:
                    save_image(rec_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

                log_list.append({'PSNR': psnr, 'Rate': rate})

            metrics['PSNR'].append(psnr)
            metrics['Rate'].append(rate)

            frame_id += 1

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}

        return {'test_log': logs}


    def test_epoch_end(self, outputs):
        dataset_name = {'HEVC-B': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
                        'HEVC-C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'],
                       }

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

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                # writer.writerow(['frame', 'PSNR', 'total bits', 'MC-PSNR', 'my', 'mz', 'ry', 'rz', 'MCerr-PSNR'])

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

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        self.log_dict(logs)

        return None


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        current_epoch = self.trainer.current_epoch
        
        lr_step = []
        for k, v in phase.items():
            if 'RNN' in k and v > current_epoch: 
                lr_step.append(v-current_epoch)
        lr_gamma = 0.5
        print('lr decay =', lr_gamma, 'lr milestones =', lr_step)

        optimizer = optim.Adam([dict(params=self.main_parameters(), lr=self.args.lr),
                                dict(params=self.aux_parameters(), lr=self.args.lr * 10)])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_step, lr_gamma)

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,
                       using_native_amp=None, using_lbfgs=None):

        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)

        optimizer.step()
        optimizer.zero_grad()

    def compress(self, ref_frame, coding_frame, p_order):
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
        # TODO: Modify to make AR function work
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
        self.logger.experiment.log_parameters(self.args)

        dataset_root = os.getenv('DATASET')
        qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[self.args.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoDataIframe(dataset_root + "vimeo_septuplet/", 'BPG_QP' + str(qp), 7,
                                                 transform=transformer)
            self.val_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, first_gop=True)

        elif stage == 'test':
            #self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B', 'M', 'C', 'D', 'E'))
            #self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B', 'M'), first_gop=True)
            #self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('C', 'D', 'E'))
            self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B'), GOP=32)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the arguments for this LightningModule
        """
        # MODEL specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', '-lr', dest='lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--lmda', default=2048, choices=[256, 512, 1024, 2048, 4096], type=int)
        parser.add_argument('--patch_size', default=256, type=int)
        parser.add_argument('--ssim', action="store_true")
        parser.add_argument('--debug', action="store_true")

        # training specific (for this model)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--save_dir')

        return parser

if __name__ == '__main__':
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well
    seed_everything(888888)

    #save_root = "/work/u4803414/torchDVC/"
    save_root = os.getenv('LOG', './') + 'torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    #trc.add_coder_args(parser)

    # training specific
    parser.add_argument("--disable_signalconv", "-DS", action="store_true", help="Enable SignalConv or not.")
    parser.add_argument("--deconv_type", "-DT", type=str, default="Signal", 
                        choices=trc.__DECONV_TYPES__.keys(), help="Configure deconvolution type")

    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom'], default='none')
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC+")

    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='PWC')
    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument('--prev_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_residual_coder_conf', type=str, default=None)

    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    # I-frame coder ckpt
    if args.ssim:
        ANFIC_code = {4096: '0619_2320', 2048: '0619_2321', 1024: '0619_2321', 512: '0620_1330', 256: '0620_1330'}[args.lmda]
    else:
        ANFIC_code = {2048: '0821_0300', 1024: '0530_1212', 512: '0530_1213', 256: '0530_1215'}[args.lmda]

    torch.backends.cudnn.deterministic = True
 
    # Set conv/deconv type ; Signal or Standard
    conv_type = "Standard" if args.disable_signalconv else trc.SignalConv2d
    deconv_type = "Transpose" if args.disable_signalconv and args.deconv_type != "Signal" else args.deconv_type
    trc.set_default_conv(conv_type=conv_type, deconv_type=deconv_type)
    
    # Config coders
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

    args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

    trainer = Trainer.from_argparse_args(args,
                                         gpus=args.gpus,
                                         distributed_backend=db,
                                         default_root_dir=save_root,
                                         check_val_every_n_epoch=1,
                                         num_sanity_val_steps=0,
                                         terminate_on_nan=True)

    epoch_num = args.restore_exp_epoch
    if args.restore_exp_key is None:
        raise ValueError
    else:  # When prev_exp_key is specified in args
        checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints", f"epoch={epoch_num}.ckpt"),
                                map_location=(lambda storage, loc: storage))

    trainer.current_epoch = epoch_num + 1
    
    model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
