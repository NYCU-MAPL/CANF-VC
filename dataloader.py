import torch
import random
import os
from glob import glob
import subprocess
import numpy as np

from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision import transforms
from subprocess import Popen, PIPE

from util.vision import imgloader, rgb_transform
from PIL import Image


class VideoData(torchData):
    """Video Dataset

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, frames, transform=rgb_transform):
        super().__init__()
        self.folder = glob(root + 'img/*/*/')
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.folder)
        #return 1

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(self.frames):
            random.seed(seed)
            file = path + str(f) + '.png'
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)


class VideoTestDataIframe(torchData):
    def __init__(self, root, lmda, first_gop=False, sequence=('U', 'B'), GOP=12):
        super(VideoTestDataIframe, self).__init__()
        
        assert GOP in [12, 16, 32], ValueError
        self.root = root
        self.lmda = lmda
        self.qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[lmda]

        self.seq_name = []
        seq_len = []
        gop_size = []
        dataset_name_list = []

        if 'U' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([96]*7)
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['UVG']*7)
        if 'B' in sequence:
            self.seq_name.extend(['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*5)
                #seq_len.extend([240, 600, 500, 500, 240])
            else:
                seq_len.extend([96]*5)
            gop_size.extend([GOP]*5)
            dataset_name_list.extend(['HEVC-B']*5)
        if 'C' in sequence:
            self.seq_name.extend(['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['HEVC-C', 'HEVC-C', 'HEVC-C', 'HEVC-C', ])
        if 'D' in sequence:
            self.seq_name.extend(['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses1']) # Rename "RaceHorses" to avoid repetition
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['HEVC-D', 'HEVC-D', 'HEVC-D', 'HEVC-D', ])
        if 'E' in sequence:
            self.seq_name.extend(['vidyo1', 'vidyo3', 'vidyo4'])
            if GOP in [12, 16]:
                seq_len.extend([100]*3)
            else:
                seq_len.extend([96]*3)
            gop_size.extend([GOP]*3)
            dataset_name_list.extend(['HEVC-E', 'HEVC-E', 'HEVC-E'])

        if 'M' in sequence:
            MCL_list = []
            for i in range(1, 31):
                MCL_list.append('videoSRC'+str(i).zfill(2))
                
            self.seq_name.extend(MCL_list)
            if GOP in [12, 16]:
                seq_len.extend([150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
                                150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
                                120, 120, 120, 120, 120, 150, 150, 150, 120, 150])
            else:
                seq_len.extend([96]*30)
            gop_size.extend([GOP]*30)
            dataset_name_list.extend(['MCL_JCV']*30)

        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            if first_gop:
                gop_num = 1
            else:
                gop_num = seq_len[seq_name] // gop_size[seq_name]
            for gop_idx in range(gop_num):
                self.gop_list.append([dataset_name_list[seq_name],
                                      seq_name,
                                      1 + gop_size[seq_name] * gop_idx,
                                      1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        seed = random.randint(0, 1e9)
        imgs = []

        for frame_idx in range(frame_start, frame_end):
            random.seed(seed)

            raw_path = os.path.join(self.root, dataset_name, 'rgb', seq_name, 'frame_{:d}.png'.format(frame_idx))

            if frame_idx == frame_start:
                img_path = os.path.join(self.root, 'bpg', str(self.qp), 'decoded', seq_name, f'frame_{frame_idx}.png')

                if not os.path.exists(img_path):
                    # Compress data on-the-fly when they are not previously compressed.
                    bin_path = img_path.replace('decoded', 'bin').replace('png', 'bin')

                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)

                    subprocess.call(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))
                    subprocess.call(f'bpgdec -o {img_path} {bin_path}'.split(' '))

                imgs.append(transforms.ToTensor()(imgloader(img_path)))
            
            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        return dataset_name, seq_name, stack(imgs), frame_start
