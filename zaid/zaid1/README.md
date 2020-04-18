## Code Description

- Saliency Maps
Binary Saliency Maps created using file "robust_saliency.py" for any dataset. To use these in experimentation use the following python file
```
reid_main.py
```
To experiment using original images use the following file
```
reid_main_sal_wo.py
```

## Datasets

- For adding new datasets for testing, add the images in the below folder:
```
/media/zaid/dataset/dataset/YOUR_DATASET
```
- The .csv files for the datasets must be in the below folder:
```
/media/zaid/dataset/dataset/YOUR_DATASET/data/csv/YOUR_DATASET
```
- Additionally modify the config.py file to add path to the .csv files of your dataset
- The models are saved in :
```
/media/zaid/zaid1/model_/YourModel
```
- The logs for tensorboard image visualisation are saved in :
```
/media/zaid/zaid1/log/YourModel
```
- To run tensorboard for visualisation run the following command:
```
tensorboard --logdir='/media/zaid/zaid1/log' --bind_all
```
- For small experimental testing use the following three files:
reid_main_temp, reid_network_tmp, reid_evaluate_temp.

## PLAY_WITH_CSV.py

Contains python scripts to create csv files from dataset folders to be used with code, modify and get additional information like cam_id from csv files, make a split of train dataset into train and validation 




