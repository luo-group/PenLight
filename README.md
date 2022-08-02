# PenLight

## Requirements
- pytorch
- torch_geometric
- torch_cluster
- torch_scatter
- biopython
- sklearn
- matplotlib
- seaborn
- numpy

## Download Datasets
- Download preprocessed data for CATH prediction from 
- Download preprocessed data for EC prediction from 

## Training
Train a model for CATH
```
python train.py --experiment cath-demo --model gatv2 --aa_embed protT5 --batch_hard --task cath --parallel
```
Train a model for EC prediction
```
python train.py --experiment ec-demo --model gatv2.01 --aa_embed esm --batch_hard --task ec --parallel
```

### Prediction
Predict CATH annotations
```
python predict.py --model cath-GATv2 --task cath
```
Predict EC annotations
```
python predict.py --model ec-GATv2.01 --task ec
```