import json
import h5py
import os
import torch
import numpy as np
from tqdm import tqdm

def gen_h5(path, save_path, emb_type='esm'):
    for root, dirs, files in os.walk(path):
        files = files
        break
    print(f'Number of files: {len(files)}')
    dataset = h5py.File(save_path, 'a')
    for file in tqdm(files):
        if emb_type == 'esm':
            tmp = torch.load(os.path.join(root, file))['representations'][33]
        elif emb_type == 'prott5':
            tmp = torch.load(os.path.join(root, file))
        else:
            raise NotImplementedError
        # print(tmp, tmp.shape)
        # input()
        if file.split('.')[0] not in dataset.keys():
            dataset.create_dataset(name=file.split('.')[0], data=tmp, dtype=np.float32)
        else:
            dataset.__delitem__(file.split('.')[0])
            dataset.create_dataset(name=file.split('.')[0], data=tmp, dtype=np.float32)

def verify(file):
    dataset = h5py.File(file, 'r')
    keys = list(dataset.keys())
    print(f'Number of keys: {len(keys)}')

if __name__ == '__main__':
    gen_h5('/data1/luojq/uploadP1/ec_data/price150/', '/home/luojq/uploadP1/ec_data/ec_esm1b.h5', 'esm')
    verify('/home/luojq/uploadP1/ec_data/ec_esm1b.h5')