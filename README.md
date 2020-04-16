# Cross-Domain-Person-Re-ID
<p align="center"><img src='Architecture_.jpg' width="1000px"></p>
Pytorch implementation for the above architecture for Cross Domain Person Re-Identification. This version of code is the draft.

## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/)
- TensorFlow
- TensorBoard
## Getting Started

### Datasets
We conduct experiments on [Market1501](http://www.liangzheng.org/Project/project_reid.html), [CUHK03](https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view)
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

### Evaluation 
We use two evaluation protocols for evaluating the performance of the model
```
- Rank accuracy
- mAP (Mean Average Accuracy)
```

## Architecture
The model consists of two encoders( Identity and Pose). ResNet-50 pre-trained on ImageNet is used as backbone for both encoders. The images(224x224x3) are normalised and then passed into the encoders. The encoders output feature-maps(2048x7x7) which are Max-Pooled to output feature vectors(2048).

The feature maps from the two encoders are concatenated to obtain a feature-map(4096x7x7) which is fed into the decoder. We also implement the latent decoder with fully convolutional network and it outputs reconstructed images(224x224x3). The feature vectors are used to calculate losses. The aim of this model is to separate identity discriminative and identity irrelevant information in a way that is generalisable to other datasets by removing all identity, pose and background information.

## Code Descriptions
The code is arranged in the following files

### Reid_main.py 
Consists of the main python file where we call functions from the different python files
### Reid_network.py 
Consists of the code for creating the encoder-decoder architecture and the classifier
### Reid_loss.py
Consists of code for the various loss functions used in the architecture
### Reid_dataset.py 
Consists of code for the data loader for our datasets.
### Reid_evaluate.py 
Consists of code for the two evaluation protocols used (i.e, Rank and mAP accuracy)


## Citation
We used [ARN](https://github.com/yujheli/ARN/blob/master/README.md) as the baseline for our model
