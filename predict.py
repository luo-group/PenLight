import torch
import json, os, time, argparse, yaml
from tqdm import tqdm
from models import GATModel_clean
from utils import *
from datasets import ProteinGraphDataset, DataSplitter, MultilabelDataset, MultilabelDataSplitter

torch.set_num_threads(4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_in_dim = (6, 3)
edge_in_dim = (32, 1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    parser.add_argument('--model', help='model checkpoint for inference', type=str)
    parser.add_argument('--lookupset', type=str)
    parser.add_argument('--input', help='input dataset for inference', type=str)
    parser.add_argument('--output', help='output result file', type=str, default='results.json')
    parser.add_argument('--predict_fn', help='how to make predictions, top-1 or largest margin', choices=['top1', 'margin'], default='top1')

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

def predict(model, input_graphs, lookup_graphs, args):
    pass


def main():
    args = get_args()
    
    # load model checkpoint
    in_channels = 1280 if args.embedding == 'esm1b' else 1024
    if args.append_scalar_features:
        in_channels += node_in_dim[0]
    edge_dim=32 if args.scalar_only else 35
    
    model = GATModel_clean(embedder=args.embedding,
                           in_channels=in_channels,
                           hidden_channels=args.hidden_channels,
                           out_dim=args.out_dim,
                           edge_dim=edge_dim,
                           heads=args.heads,
                           drop_rate=args.drop_rate,
                           append_scalar_features=args.append_scalar_features,
                           device=device,
                           )
    print(f'Loading model checkpoint: {args.model}')
    model.load_state_dict(args.model)
    print(model)

if __name__ == '__main__':
    main()