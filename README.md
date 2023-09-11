
# CANF-VC: Conditional Augmented Normalizing Flows for Video Compression
## Update (09.11.23): CANF-VC++: Enhancing Conditional Augmented Normalizing Flows for Video Compression with Advanced Techniques
* We present CANF-VC++, an improved video compression framework from CANF-VC. CANF-VC++ demonstrates substantial Bjøntegaard-Delta rate savings of 40.2%, 38.1%, and 35.5% on UVG, HEVC Class B, and MCL-JCV datasets, respectively over CANF-VC. Please check our [paper]() on arxiv for the details of our improvements. You can find the inference commands in [full commands](https://github.com/NYCU-MAPL/CANF-VC/blob/main/README.md#full-commands) section.
* Performance
  * BD-rate (Anchor: CANF-VC ; GOP=32 ; only compress first 96 frames in each testing sequence)
    * ![圖片](https://github.com/NYCU-MAPL/CANF-VC/assets/108980934/1b43ea52-beba-4628-9cde-795074d5c080)

  * R-D curves:
  ![RD_UVG_PSNR_GOP32](https://github.com/NYCU-MAPL/CANF-VC/assets/108980934/bf15c59d-6590-4373-8e51-b03d776668ae)
  ![RD_HEVC_B_PSNR_GOP32](https://github.com/NYCU-MAPL/CANF-VC/assets/108980934/72e78d60-340f-4f80-87c7-6bcc9bb9ea2a)
  ![RD_MCL_JCV_PSNR_GOP32](https://github.com/NYCU-MAPL/CANF-VC/assets/108980934/462836eb-aeba-4abc-b1ee-fd8e8dac719c)

## Update (08.30.22): CANF-VC with Error Propagation Aware Training Strategy
* CANF-VC-EPA is an enhanced version of CANF-VC. With exactly the same network architecture as CANF-VC, CANF-VC-EPA additionally introduces the Error Propagation Aware (EPA) training strategy from Guo et al., ECCV'20. 
* Usage: Exactly the same as [CANF-VC](https://github.com/NYCU-MAPL/CANF-VC/blob/main/README.md#full-commands)
* Performance
  * BD-rate (GOP=32 ; anchor: x265 veryslow). The best performer is marked in red and the second best in blue.
    * ![image](https://user-images.githubusercontent.com/108980934/193492740-4c53891d-b755-4d7c-a698-972edf8a975b.png)

  * R-D curves: 
  * 
    <img src="https://user-images.githubusercontent.com/108980934/187449482-a9fad0fe-2506-47f2-8106-f4ada30d9ef7.png" width="400"> 
    <img src="https://user-images.githubusercontent.com/108980934/187449448-aacf46ea-801f-48f7-8ac9-ce137ecc16ca.png" width="400"> 
    <img src="https://user-images.githubusercontent.com/108980934/187574039-7f9b9c7c-2a45-4039-b0d2-10525c5c908e.png" width="400">
  
## Project Installation
1. Prepare PyTorch 1.4.0 environment and correspond torchvision
2. Run `sh install.sh`
3. (Only needed for CANF-VC*) Install `libbpg`: https://github.com/mirrorer/libbpg
  3.1 Configure path to `libbpg` as `libbpg_path` in `dataloader.py`
4. Download model weights & prepare testing data
5. Start evaluation: `action=test/compress/decompress`

## Model Weight
* Get download link after filling out this form: https://forms.gle/Twsx8NofBjGakueL6
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
* CANF-VC++ (PSNR): 
  * `test`: `$ python3 test.py --Iframe=ANFIC --Pframe=CANFVC_PP --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_quadtree_context.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_quadtree_context.yml --dataset=U --seq=Beauty --seq_len=96 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC++/PSNR --action=test --GOP=32`
  * `compress`: `$ python3 test.py --Iframe=ANFIC --Pframe=CANFVC_PP --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_quadtree_context.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_quadtree_context.yml --dataset=U --seq=Beauty --seq_len=96 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC++/PSNR --bitstream_dir=./bin/CANFVC_PP --action=compress --GOP=32`
  * `decompress`: `$ python3 test.py --Iframe=ANFIC --Pframe=CANFVC_PP --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_quadtree_context.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_quadtree_context.yml --dataset=U --seq=Beauty --seq_len=96 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC++/PSNR --bitstream_dir=./bin/CANFVC_PP --action=decompress --GOP=32`


* CANF-VC (PSNR): 
  * `test`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --action=test --GOP=32`
  * `compress`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --bitstream_dir=./bin --action=compress --GOP=32`
  * `decompress`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC/PSNR --bitstream_dir=./bin --action=decompress --GOP=32`

* CANF-VC* (PSNR): 
  * `test`: `$ python3 test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --action=test --GOP=32`
  * `compress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --bitstream_dir=./bin --action=compress --GOP=32`
  * `decompress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset=D --seq=BQSquare --seq_len=100 --dataset_path=./video_dataset --lmda=2048 --model_dir=./models/CANF-VC_star/PSNR --bitstream_dir=./bin --action=decompress --GOP=32`

## Full Commands
* CANF-VC++:
  * `test`: `$ python3 test.py --Iframe=ANFIC --Pframe=CANFVC_PP --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_quadtree_context.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_quadtree_context.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC++/{PSNR/MS-SSIM} --action=test --GOP=32`
  * `compress`/`decompress`: `$ python3 test.py --Iframe=ANFIC --Pframe=CANFVC_PP --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_quadtree_context.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_quadtree_context.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC++/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32`
* CANF-VC: 
  * `test`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python test.py --Iframe=ANFIC --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
* CANF-VC*:
  * `test`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC_star/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python3 test.py --Iframe=BPG --MENet=PWC --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC_star/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
* CANF-VC Lite:
  * `test`: `$ python test.py --Iframe=ANFIC --MENet=SPy --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior_Lite.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_Lite.yml --dataset={U/B/C/D/E/M} --dataset_path=/path/to/video_dataset --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC-Lite/{PSNR/MS-SSIM} --action=test --GOP=32 {--msssim}`
  * `compress`/`decompress`: `$ python test.py --Iframe=ANFIC --MENet=SPy --motion_coder_conf=./CANF_VC/config/DVC_motion.yml --cond_motion_coder_conf=./CANF_VC/config/CANF_motion_predprior_Lite.yml --residual_coder_conf=./CANF_VC/config/CANF_inter_coder_Lite.yml --dataset={U/B/C/D/E/M} --seq=SEQUENCE_TO_BE_COMPRESS(Optional) --seq_len=NUMBER_OF_FRAMES_TO_BE_COMPRESSED(Optional) --dataset_path=/path/to/video_dataset --lmda={2048/1024/512/256} --model_dir=/path/to/CANF-VC-Lite/{PSNR/MS-SSIM} --bitstream_dir=./bin --action={compress/decompress} --GOP=32 {--msssim}`
  
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
