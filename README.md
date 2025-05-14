### Prepare datasets:
KITTI-360: https://www.cvlibs.net/datasets/kitti-360/download.php only KITTI-360--data_3d_raw is used

Seeing Through Fog: https://light.princeton.edu/datasets/automated_driving_dataset

### Prepare environment:

`conda env create -f environment.yml` 

### Pretrain and finetune:

set train_model at 'train' or 'finetune'

`accelerate launch --mixed_precision 'fp16' --dynamo_backend 'no' train.py` 

### Generate:

`python generate.py` 

The generated and labeled data can be found in https://pan.baidu.com/s/1_waBH02ZXpSlEKFA-o5_bw, password: 7878

