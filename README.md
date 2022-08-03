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

## Directories
```
.
├── data-cath
│   ├── cath-domain-list-S100.txt
│   └── splits_json
│       └── test.json
├── data-ec
│   ├── pdb_ec_chain.txt
│   └── splits_json
│       └── test.json
├── LICENSE
├── log
│   ├── cath-GATv2/
│   └── ec-GATv2.01/
├── README.md
└── src
    ├── datasets.py
    ├── model.py
    ├── predict.py
    ├── run.sh
    ├── train.py
    └── utils.py
```

## Download Datasets
- Download preprocessed data for CATH prediction from [data-cath](), unzip the downloaded directory and replace `./data-cath` with it.
- Download preprocessed data for EC prediction from [data-ec](), unzip the downloaded directory and replace `./data-ec` with it.
- Note: If you want to train your own PenLight models, you must download the above datasets, unzip them and replace the current ./data-cath and ./data-ec directories. If you only want to make predictions and infer annotations for proteins with our pretrained models, you don't have to download the datasets.

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
Two pretrained models cath-GATv2 (for CATH) and ec-GATv2.01 (for EC) are already stored in `./log` directory and can be used to infer annotations for json-format input files as follows. We provided two preprocessed json files ../data-cath/splits_json/test.json and ../data-ec/splits_json/test.json as demonstration.

Predict CATH annotations
```
python predict.py --model cath-GATv2 --task cath --input ../data-cath/splits_json/test.json --output cath-pred.txt
```

Predict EC annotations
```
python predict.py --model ec-GATv2.01 --task ec --input ../data-ec/splits_json/test.json --output ec-pred.txt
```