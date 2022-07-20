# CANF-VC: Conditional Augmented Normalizing Flows for Video Compression

## Project Installation
* Prepare PyTorch 1.4.0 environment and correspond torchvision
* Run `sh install.sh`
* Refer to `requirements.txt` to build essential packages
* Install `libbpg`: https://github.com/mirrorer/libbpg

## Model Weight
* CANF-VC: https://drive.google.com/drive/folders/1fj8sb_CMktyJ_yU0Yf2lbZjzxVYAGvUw?usp=sharing
* CANF-VC*: https://drive.google.com/drive/folders/19gZlrhk1ONNbLpqfD9DTSCBWvh-uSamJ?usp=sharing
* CANF-VC Lite: https://drive.google.com/drive/folders/1e5WSsGhuqKh8b8VS9QGczx0XeQDHu-cs?usp=sharing
* Submodules: https://drive.google.com/file/d/1-JNDD-sfDVyDpSUHKL8a6_dYC1Qf5Y-F/view?usp=sharing
  > Should be put under this project directory

## Dataset
* Prepare all of your video sequence (in `.png` format), or
* [Download all datasets](https://drive.google.com/file/d/1eAZezoiARHXN-GpoDn2LlmnhtphSGzVu/view?usp=sharing):
  * Including:
    * `U` for UVG dataset
    * `B`, `C`, `D`, `E` for HEVC-B, -C, -D, -E dataset
    * `M` for MCL-JCV dataset
* We provide `yuv2png.py` for you to turn `.yuv` video into `.png` video frames
  * `python yuv2png.py`
  * Please specify the path & dataset to be converted in the file

## Evaluation Examples
* PSNR models:
  * CANF-VC: 
    * `test`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/PSNR --action=test --GOP=32`
    * `compress`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/PSNR --bitstream_dir=./bin --action=compress --GOP=32`
    * `decompress`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=COMPRESSED_SEQUENCE --seq_len=NUMBER_OF_COMPRESSED_FRAMES --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/PSNR --bitstream_dir=./bin --action=decompress --GOP=32`
  * CANF-VC*:
  * CANF-VC Lite:

* MS-SSIM models:
  * CANF-VC:
  * CANF-VC*:
  * CANF-VC Lite:

## Citation
If you find this work useful for your research, please cite:
```
@article{canfvc,
  title={CANF-VC: Conditional Augmented Normalizing Flows for Video Compression},
  author={Ho, Yung-Han and Chang, Chih-Peng and Chen, Peng-Yu and Gnutti, Alessandro and Peng, Wen-Hsiao},
  journal={European Conference on Computer Vision},
  year={2022}
}
```
