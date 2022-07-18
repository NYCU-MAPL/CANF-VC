# -*- coding: utf-8 -*
import os
import numpy as np
from os import listdir
import os.path as osp
from tqdm import tqdm

from PIL import Image

from subprocess import Popen, PIPE
from subprocess import call

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

video_root = './video_dataset'
dataset = 'U' # 'U', 'B', 'C', 'D', 'E', 'M'

video_name_list = {
    'U': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
    'B': ['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'],
    'C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'],
    'D': ['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses1'], # Rename "RaceHorses" to avoid repetition
    'E': ['vidyo1', 'vidyo3', 'vidyo4'],
    'M': ['videoSRC'+str(i).zfill(2) for i in range(1, 31)]
}[dataset]

video_frames_list = {
    'U': [600, 600, 600, 600, 600, 300, 600],
    'B': [240, 600, 500, 500, 240],
    'C': [100]*4,
    'D': [100]*4,
    'E': [100]*3,
    'M': [150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
          150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
          120, 120, 120, 120, 120, 150, 150, 150, 120, 150]
}[dataset]

frame_rate_list = {
    'U': [120]*7,
    'B': [50, 60, 50, 24, 24],
    'C': [30, 60, 50, 50],
    'D': [30, 60, 50, 50],
    'E': [60, 60, 60],
    'M': [30, 30, 30, 30, 25, 25, 25, 25, 25, 30,
          30, 30, 30, 30, 30, 30, 24, 25, 30, 25,
          24, 24, 24, 24, 24, 30, 30, 30, 24, 30]
}[dataset]

dim = {
    'U': (1920, 1080),
    'B': (1920, 1080),
    'C': (832, 480),
    'D': (416, 240),
    'E': (1280, 720),
    'M': (1920, 1080),
}[dataset]


for video_name, video_frames, frame_rate in zip (video_name_list, video_frames_list, frame_rate_list): 
    print(video_name)
    run_time = video_frames / frame_rate
    
    #////Raw YUV(BPG)
    raw_yuv = osp.join(video_root, 'raw', video_name + "_" + str(dim[0]) + "x" + str(dim[1]) + "_" + str(frame_rate) + ".yuv")
    raw_png_root = osp.join(video_root, dataset, video_name)
    os.makedirs(raw_png_root, exist_ok=True)

    command = "ffmpeg -y -pix_fmt yuv420p -s " + str(dim[0]) + "x" + str(dim[1]) + " -i " + raw_yuv + " " + raw_png_root + "/frame_%d.png"
    output, error = Popen(command, universal_newlines=True, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    print(output)
