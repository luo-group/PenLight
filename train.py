import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import torch_geometric
import numpy as np
import random, time, os, gc, copy, argparse, yaml, datetime, logging
from pathlib import Path
# from models import GATModel, GATModel2
from utils import seed_all, plotter, init_monitor, toCPU, count_parameters
# from datasets import ProteinGraphDataset, DataSplitter, MultilabelDataset, MultilabelDataSplitter
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_in_dim = (6, 3)
edge_in_dim = (32, 1)
date = datetime.datetime.now().strftime('%m%d-%H%M')

def get_args():
    '''parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    
    parser.add_argument('--exp_name', help='experiment name', type=str, default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multilabel', action='store_true')
    parser.add_argument('--task', type=str, default='cath', choices=['cath', 'ec'])
    parser.add_argument('--parallel', help='use multiple gpu for training', action='store_true')
    
    # model arguments
    parser.add_argument('--embedding', help='protein language model embedding', type=str, default='prott5', choices=['prott5', 'esm1b'])
    parser.add_argument('--hidden_channels', help='hidden channels', type=list, default=[128, 512])
    parser.add_argument('--out_dim', help='model output dimension', type=int, default=128)
    parser.add_argument('--edge_dim', type=int, default=35)
    parser.add_argument('--heads', help='GAT attention heads', type=list, default=[8, 1])
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--append_scalar_features', action='store_true')
    
    # training hyperparameters
    parser.add_argument('--batch_size', help='training batch size', type=int, default=32)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.0001)
    parser.add_argument('--batch_hard', help='whether use batch_hard', action='store_true')
    parser.add_argument('--exclude_easy', help='whether exclude easy triplets', action='store_true')
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0001)
    
    # graph construction arguments
    parser.add_argument('--edge_type', help='topk or <8A', type=str, default='8A', choices=['topk', '8A'])
    parser.add_argument('--edge_threshold', help='dist under threshold is an edge', type=float, default=8.0)
    parser.add_argument('--scalar_only', help='only use the scalar features, no vector features', action='store_true')
        
    args = parser.parse_args()
    
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}
        
    return args

def save_args(args, save_path):
    '''save argument configurations to .yml file'''
    if 'config' in args.__dict__:
        args.__dict__.pop('config')
    with open(save_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        
def testing(model, test, test_set, lookup_set, type='gat', parallel=True):
    model.eval()
    with torch.no_grad():  # evaluate current performance (no grads)
        test_tucker, lookup_tucker = [], []

        # print("Generating test_tucker...")
        for test_batch in test_set:
            test_batch.to(device)
            # TODO: modify for different models
            if parallel:
                if type == 'gat':
                    # print(test_batch)
                    tmp = model.module.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name)
                else:
                    tmp = model.module.single_pass((test_batch.node_s, test_batch.node_v), test_batch.edge_index, (test_batch.edge_s, test_batch.edge_v), test_batch.name)
            else:
                if type == 'gat':
                    tmp = model.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name)
                else:
                    tmp = model.single_pass((test_batch.node_s, test_batch.node_v), test_batch.edge_index, (test_batch.edge_s, test_batch.edge_v), test_batch.name)
            tmp.to(torch.device('cpu'))
            test_batch.to(torch.device('cpu'))
            test_tucker.append(tmp.cpu().detach())
        torch.cuda.empty_cache()
        # print("Generating lookup_tucker...")
        lookup_loader = torch_geometric.loader.DataLoader(
            lookup_set, batch_size=16)
        for idx, lookup_batch in enumerate(lookup_loader):
            # print(f'\rBatch {idx}', end="", flush=True)
            lookup_batch = lookup_batch.to(device)
            # TODO: modify for different models
            if parallel:
                if type == 'gat':
                    tmp = model.module.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr,
                                                   name=lookup_batch.name, seq=lookup_batch.seq, batch=lookup_batch.batch)
                else:
                    tmp = model.module.single_pass((lookup_batch.node_s, lookup_batch.node_v), lookup_batch.edge_index, (
                        lookup_batch.edge_s, lookup_batch.edge_v), lookup_batch.name, lookup_batch.batch)
            else:
                if type == 'gat':
                    tmp = model.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr,
                                            name=lookup_batch.name, seq=lookup_batch.seq, batch=lookup_batch.batch)
                else:
                    tmp = model.single_pass((lookup_batch.node_s, lookup_batch.node_v), lookup_batch.edge_index, (
                        lookup_batch.edge_s, lookup_batch.edge_v), lookup_batch.name, lookup_batch.batch)
            tmp.to(torch.device('cpu'))
            lookup_batch.to(torch.device('cpu'))
            lookup_tucker.append(tmp.detach())
            if idx % 100 == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        lookup_tucker = torch.cat(lookup_tucker, dim=0).to(torch.device('cpu'))
        test_tucker = torch.cat(test_tucker, dim=0).to(torch.device('cpu'))
        acc, err = test.evaluate(lookup_tucker, test_tucker)
    model.train()
    return acc, err

if __name__ == '__main__':
    # parse arguments
    args = get_args()
    
    # set seeds
    seed_all(args.seed)
    
    # set directories
    
    
    # load datasets
    
    
    # load model and set hyperparameters
    
    
    # training
    
    
    # testing