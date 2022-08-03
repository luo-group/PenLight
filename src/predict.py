from inspect import getargs
import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
from tqdm import tqdm
import argparse

def get_gt(cath_file):
    ec = {}
    with open(cath_file) as f:
        lines = f.readlines()
    for line in lines:
        entries = line.strip().split()
        ec[entries[0]] = [(entries[i]) for i in range(1, 5)]
    return ec

def get_gt_from_json(json_file):
    def str2nums(ec):
        # print(ec)
        if type(ec) == list and len(ec) == 4:
            return ec
        if type(ec) == list and len(ec) > 4:
            ec = ec[:4]
            ec[-1] = ec[-1].split(';')[0]
            return ec
        res = [(int(s) if s != 'n1' else 0) for s in ec.split('.')]
        return res

    with open(json_file) as f:
        data = json.load(f)
    ec = {}
    for k, v in data.items():
        ec[k] = str2nums(v['ec'])
    return ec

def get_test_ids(json_file):
    with open(json_file) as f:
        testset = json.load(f)
    print(f'Test set size: {len(testset)}')
    return list(testset.keys())

def str2float(s):
    try:
        res = float(s)
    except:
        res = 0
    finally:
        return res

def get_id2pred(test_emb, lookup_emb, ec_table):
    id2pred = {}
    testset = torch.load(test_emb)
    lookupset = torch.load(lookup_emb)
    print(f'Test set size: {len(testset)}')
    print(f'Lookup set size: {len(lookupset)}')
    test = torch.cat([entry['embedding'] for entry in list(testset.values())])
    lookup = torch.cat([entry['embedding'] for entry in list(lookupset.values())])
    test_ids = list(testset.keys())
    lookup_ids = list(lookupset.keys())
    distance = torch.cdist(test, lookup, p=2)
    assert distance.shape[0] == len(test_ids) and distance.shape[1] == len(lookup_ids), print(distance.shape)
    for i in range(len(test_ids)):
        id2pred[test_ids[i]] = ec_table[lookup_ids[torch.argmin(distance[i])]]
    assert len(id2pred) == len(testset), print(len(id2pred))
    
    return id2pred

def cat_strings(str_list):
    res = ""
    for s in str_list:
        res += s
    return res

def evaluate_acc(id2pred, ec_table):
    n = len(id2pred)
    right = [0 for i in range(4)]
    accuracy = [0 for i in range(4)]
    for i in range(4):
        y_pred = [cat_str_by_dot(label[:(i + 1)]) for label in list(id2pred.values())]
        y_gt = [cat_str_by_dot(ec_table[k][:(i + 1)]) for k in list(id2pred.keys())]
        assert len(y_pred) == len(y_gt)
        accuracy[i] = accuracy_score(y_gt, y_pred)
    print(f'accuracy:\t{accuracy[0]:.4f}\t{accuracy[1]:.4f}\t{accuracy[2]:.4f}\t{accuracy[3]:.4f}\t{np.mean(accuracy):.4f}')
    with open('res.txt', 'a') as f:
        f.write(f'{accuracy[0]:.4f} {accuracy[1]:.4f} {accuracy[2]:.4f} {accuracy[3]:.4f} {np.mean(accuracy):.4f}\n')
        
def cat_str_by_dot(strlist):
    res = ""
    for s in strlist:
        res = res + str(s) + '.'
    return res[:-1]
def evaluate_precision(id2pred, ec_table):
    average = ['micro', 'macro', 'weighted']
    n_classes = 4
    for avg in average:
        precision = [0 for i in range(n_classes)]
        for i in range(n_classes):
            y_pred = [cat_str_by_dot(label[:(i + 1)]) for label in list(id2pred.values())]
            y_gt = [cat_str_by_dot(ec_table[k][:(i + 1)]) for k in list(id2pred.keys())]
            assert len(y_pred) == len(y_gt)
            precision[i] = precision_score(y_gt, y_pred, average=avg, zero_division=0)
        print(f'{avg} precision:\t{precision[0]:.4f}\t{precision[1]:.4f}\t{precision[2]:.4f}\t{precision[3]:.4f}\t{np.mean(precision):.4f}')
        with open('res.txt', 'a') as f:
            f.write(f'{precision[0]:.4f} {precision[1]:.4f} {precision[2]:.4f} {precision[3]:.4f} {np.mean(precision):.4f} ')
    with open('res.txt', 'a') as f:
        f.write('\n')

def evaluate_recall(id2pred, ec_table):
    average = ['micro', 'macro', 'weighted']
    n_classes = 4
    for avg in average:
        precision = [0 for i in range(n_classes)]
        for i in range(n_classes):
            y_pred = [cat_str_by_dot(label[:(i + 1)]) for label in list(id2pred.values())]
            y_gt = [cat_str_by_dot(ec_table[k][:(i + 1)]) for k in list(id2pred.keys())]
            assert len(y_pred) == len(y_gt)
            precision[i] = recall_score(y_gt, y_pred, average=avg, zero_division=0)
        print(f'{avg} recall:\t{precision[0]:.4f}\t{precision[1]:.4f}\t{precision[2]:.4f}\t{precision[3]:.4f}\t{np.mean(precision):.4f}')
        with open('res.txt', 'a') as f:
            f.write(f'{precision[0]:.4f} {precision[1]:.4f} {precision[2]:.4f} {precision[3]:.4f} {np.mean(precision):.4f} ')
    with open('res.txt', 'a') as f:
        f.write('\n')

def evaluate_f1(id2pred, ec_table):
    average = ['micro', 'macro', 'weighted']
    n_classes = 4
    for avg in average:
        precision = [0 for i in range(n_classes)]
        for i in range(n_classes):
            y_pred = [cat_str_by_dot(label[:(i + 1)]) for label in list(id2pred.values())]
            y_gt = [cat_str_by_dot(ec_table[k][:(i + 1)]) for k in list(id2pred.keys())]
            assert len(y_pred) == len(y_gt)
            precision[i] = f1_score(y_gt, y_pred, average=avg, zero_division=0)
        print(f'{avg} F1:\t{precision[0]:.4f}\t{precision[1]:.4f}\t{precision[2]:.4f}\t{precision[3]:.4f}\t{np.mean(precision):.4f}')
        with open('res.txt', 'a') as f:
            f.write(f'{precision[0]:.4f} {precision[1]:.4f} {precision[2]:.4f} {precision[3]:.4f} {np.mean(precision):.4f} ')
    with open('res.txt', 'a') as f:
        f.write('\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name for prediction', type=str, default='cath-GATv2', choices=['cath-GATv2', 'ec-GATv2.01'])
    parser.add_argument('--task', help='cath or ec', type=str, default='cath', choices=['cath', 'ec'])
    parser.add_argument('--input', help='input json file', type=str, default='../data-cath/splits_json/test.json')
    parser.add_argument('--output', help='output file', type=str, default='output.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    root = Path.cwd().parent
    if args.task == 'cath':
        data_dir = root / "data-cath" 
        # data_dir = root / "data" # temporal
    else:
        data_dir = root / "data-ec"
        # data_dir = root / 'ec_data' # temporal
    log_dir = root / "log"
    model_dir = log_dir / args.model
    
    if args.task == 'cath':
        gt_table = get_gt(data_dir / "cath-domain-list-S100.txt")
    elif args.task == 'ec':
        gt_table = get_gt(data_dir / "pdb_ec_chain.txt")
    else:
        raise NotImplementedError
    print(f'Predicting using model: {args.model}')
    test_ids = get_test_ids(data_dir / 'splits_json/test.json')
    lookup_file = os.path.join(model_dir, f'emb_lookup.pt')
    id2pred = get_id2pred(os.path.join(model_dir, f'emb_test.pt'), lookup_file, gt_table)
    with open(args.output, 'w') as f:
        f.write('ID\tPrediction\n')
        for k, v in id2pred.items():
            f.write(f'{k}\t{".".join(v)}\n')