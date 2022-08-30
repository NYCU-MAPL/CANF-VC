# CANF-VC: Conditional Augmented Normalizing Flows for Video Compression

## \*\*\*Update (08.30.22)\*\*\*: CANF-VC with Error Propagation Aware Training Strategy
* We report a better CANF-VC performer, which applies a popular Error Propagation Aware (EPA) training strategy from [Gou *et al.*, ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470443.pdf). The network architecture is THE SAME as our original CANF-VC, with one more training step that enable all gradients when updating within a GOP.
* Usage: Exactly the same to CANf-VC
* Download from https://drive.google.com/drive/folders/14WUDwRgNhx2S36KfyvlG3MW10W2VMwqv?usp=sharing
* Performance
  * BD-rate (GOP=32 ; anchor: x265 veryslow)
    * ![image](https://user-images.githubusercontent.com/108980934/187421591-6b4acce8-1cf8-4a64-a590-c2ea844b2a2f.png)
  * R-D curve (UVG dataset): ![image](https://user-images.githubusercontent.com/108980934/187422449-9b5df347-635f-4d68-9d22-43c59aa0094e.png)
  * R-D curve (HEVC class B dataset): ![image](https://user-images.githubusercontent.com/108980934/187422478-0bbc398c-90f5-46d3-ad39-f85df1fb6a06.png)
  
## Project Installation
1. Prepare PyTorch 1.4.0 environment and correspond torchvision
2. Run `sh install.sh`
3. Install `libbpg`: https://github.com/mirrorer/libbpg
  3.1 Configure path to `libbpg` as `libbpg_path` in `dataloader.py`
4. Download model weights & prepare testing data
5. Start evaluation: `action=test/compress/decompress`

## Model Weight
* CANF-VC: https://drive.google.com/drive/folders/1fj8sb_CMktyJ_yU0Yf2lbZjzxVYAGvUw?usp=sharing
* CANF-VC*: https://drive.google.com/drive/folders/19gZlrhk1ONNbLpqfD9DTSCBWvh-uSamJ?usp=sharing
* CANF-VC Lite: https://drive.google.com/drive/folders/1e5WSsGhuqKh8b8VS9QGczx0XeQDHu-cs?usp=sharing
* Submodules: https://drive.google.com/drive/folders/1mjyGyyxgAxdFpzvesYYnBwRHF-0VKdnm?usp=sharing
  > Should be put under this project directory

## Dataset
* Prepare all of your video sequence (in `.png` format), or
* [Download all datasets](https://drive.google.com/file/d/1-JNDD-sfDVyDpSUHKL8a6_dYC1Qf5Y-F/view?usp=sharing):
  * Including:
    * `U` for UVG dataset
    * `B`, `C`, `D`, `E` for HEVC-B, -C, -D, -E dataset
    * `M` for MCL-JCV dataset
* We provide `yuv2png.py` for you to turn `.yuv` video into `.png` video frames
  * `python yuv2png.py`
  * Please specify the path & dataset to be converted in the file

## Examples
* CANF-VC (PSNR): 
  * `test`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --action=test --GOP=32`
  * `compress`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --bitstream_dir=./bin --action=compress --GOP=32`
  * `decompress`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --bitstream_dir=./bin --action=decompress --GOP=32`

* CANF-VC* (PSNR): 
  * `test`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --action=test --GOP=32`
  * `compress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --bitstream_dir=./bin --action=compress --GOP=32`
  * `decompress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --bitstream_dir=./bin --action=decompress --GOP=32`

## Full Commands
* CANF-VC: 
  * `test`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
* CANF-VC*:
  * `test`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC_star/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior.yml --residual_coder_conf=./config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC_star/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
* CANF-VC Lite:
  * `test`: `$ python test.py --Iframe=ANFIC --MENet=SPy --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior_Lite.yml --residual_coder_conf=./config/CANF_inter_coder_Lite.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC-Lite/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python test.py --Iframe=ANFIC --MENet=SPy --motion_coder_conf=./config/DVC_motion.yml --cond_motion_coder_conf=./config/CANF_motion_predprior_Lite.yml --residual_coder_conf=./config/CANF_inter_coder_Lite.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC-Lite/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
  
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
