# Cross-Domain-Person-Re-ID
<p align="center"><img src='Architecture_.jpg' width="1000px"></p>
Pytorch implementation for the above architecture for Cross Domain Person Re-Identification. This version of code is the draft.

## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/)
## Getting Started

### Datasets
We conduct experiments on [Market1501](http://www.liangzheng.org/Project/project_reid.html), [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation)
- Create directories for datasets:
```
mkdir datasets
cd datasets/
``` 

### Training
We use a train our model based on `ResNet-50`. Example for excuting the program:
```
python3 reid_main.py\
    --use_gpu $GPU\
    --source_dataset Duke \
    --target_dataset Market\
    --rank 1\
    --learning_rate 1e-3\
    --dist_metric L1\
    --model_dir $MODEL_DIR\
    --model_name $MODEL_NAME\
    --w_loss_rec 0.1\
    --w_loss_dif 0.1\
    --w_loss_mmd 0.1\
    --w_loss_ctr 0.1\
    --batch_size 16\
    --pretrain_model_name MODEL_DIR/$MODEL_NAME\
```
