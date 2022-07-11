# ANFIC-Image-Compression-Using-Augmented-Normalizing-Flows
ANFIC: Image Compression Using Augmented Normalizing Flows
<br>
Accepted by OJCAS'21
<br>
paper: https://arxiv.org/abs/2107.08470


## Project Installation
* Prepare PyTorch 1.4.0 environment and correspond torchvision
* Run `sh install.sh`

## How to run?
### Checklist before running
* Download pretrained weight from https://drive.google.com/drive/folders/1M2WaPFIGOJeBdDkH5Au3rMdanQP6zYIE?usp=sharing and unzip weights to ./models/

### Evaluation
 ```
$ python ANFIC_codec.py -UC -C GaussianMixtureModel -NF 320 -VNFL 128 -QE -DM eval -ckpt ./models/ANFIC_R1.ckpt -SD ./example_image/ -TD ./recon/
 ```

### Encode
 ```
$ python ANFIC_codec.py -UC -C GaussianMixtureModel -NF 320 -VNFL 128 -QE -DM compress -ckpt ./models/ANFIC_R1.ckpt -SD ./example_image/ --eval
 ```

### Decode
 ```
$ python ANFIC_codec.py -UC -C GaussianMixtureModel -NF 320 -VNFL 128 -QE -DM decompress -ckpt ./models/ANFIC_R1.ckpt -SD ./example_image/ -TD ./recon/ --eval -OD ./example_image/
 ```