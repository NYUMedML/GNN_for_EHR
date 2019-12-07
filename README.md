# Graph Neural Network on Electronic Health Records for Predicting Alzheimer’s Disease

This repository contains the code for the paper **Graph Neural Network on Electronic Health Records for Predicting Alzheimer’s Disease**.

![image](https://github.com/NYUMedML/GNN_for_EHR/blob/master/plots/model.png)

## Model Training

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Required packages can be installed on **python3.6** environment via command:

```
pip3 install -r requirements.txt
```

Nvidia GPU with Cuda 10.0 are required for training models.

### Train

GNN for EHR on predicting disease outcomes can be train by running command:

```
python3 train.py --input 512 --output 512 --heads 4 --batch 64 --dropout 0.4 --alpha 0.15 --lr 0.0001
```
