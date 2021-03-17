# Variationally Regularized Graph-based Representation Learning for Electronic Health Records

This repository contains the code for the paper [Variationally Regularized Graph-based Representation Learning for Electronic Health Records](https://arxiv.org/abs/1912.03761).

## Introduction
In this paper, we design a novel graph-based model to generalize the ability of learning implicit medical concept structures to a wide range of data source, including short-term ICU data and long-term outpatient clinical data.We introduce variational regularization for node representation learning, addressing the insufficiency of self-attention in graph-based models, and difficulties of manually constructing knowledge graph from real-world noisy data sources. The novelty of our work is to enhance the learning of attention weights in GNN via regularization on node representations. Besides obtaining better performances in different predictive tasks, we also provide interpretation on the effect of variational regularization in graph neural networks using singular value analysis, and bridge the connection between singular values
and representation clustering.


## Model Training


### Prerequisites

Required packages can be installed on **python3.6** environment via command:

```
pip3 install -r requirements.txt
```

Nvidia GPU with Cuda 10.0 are required for training models.

### Data

The preprocessing tools that extracts medical code for datasets are enclosed in [data](https://github.com/NYUMedML/GNN_for_EHR/tree/master/data). Run the command:
```
python3 preprocess_{dataset}.py --input_path {dataset_path} --output_path {storage_path}
```


### Train

GNN for EHR on predicting disease outcomes can be train by running command:

```
python3 train.py --data_path {storage_path} --embedding_size 512 --result_path {model_path}
```

## Architecture

<img src="https://github.com/NYUMedML/GNN_for_EHR/blob/master/plots/model.png" alt="drawing" width="900"/>
