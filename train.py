import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import torch_geometric
import numpy as np
import random, time, os, gc, copy, argparse, yaml, datetime, logging, json
from pathlib import Path
from models import GATModel_clean
from utils import *
from datasets import ProteinGraphDataset, DataSplitter, MultilabelDataset, MultilabelDataSplitter
from losses import TripletLoss
from tqdm import tqdm

torch.set_num_threads(4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_in_dim = (6, 3)
edge_in_dim = (32, 1)
date = datetime.datetime.now().strftime('%m%d-%H%M')

def get_args():
    '''parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    
    # general arguments
    parser.add_argument('--exp_name', help='experiment name', type=str, default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multilabel', help='activate multilabel version for EC prediction', action='store_true')
    parser.add_argument('--task', type=str, default='ec', choices=['cath', 'ec'])
    parser.add_argument('--parallel', help='use multiple gpu for training', action='store_true')
    parser.add_argument('--n_levels', type=int, default=4, help='number of levels in each label')
    
    #dataset arguments
    parser.add_argument('--data_dir', type=str, default='data/ec-data')
    parser.add_argument('--trainset', type=str)
    parser.add_argument('--validset', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--label_file', type=str, default='data/ec-data/ec_label.json')
    
    # model arguments
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--embedding', help='protein language model embedding', type=str, default='esm1b', choices=['prott5', 'esm1b'])
    parser.add_argument('--emb_file', help='LM embedding file, h5 format', type=str, default='data/ec-data/ec_esm1b.h5')
    parser.add_argument('--hidden_channels', help='hidden channels', type=list, default=[128, 512])
    parser.add_argument('--out_dim', help='model output dimension', type=int, default=128)
    parser.add_argument('--edge_dim', type=int, default=35)
    parser.add_argument('--heads', help='GAT attention heads', type=list, default=[8, 1])
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--append_scalar_features', action='store_true')
    
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=32)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.0001)
    parser.add_argument('--batch_hard', help='whether use batch_hard', action='store_true')
    parser.add_argument('--exclude_easy', help='whether exclude easy triplets', action='store_true')
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0001)
    parser.add_argument('--early_stop_threshold', help='number of epochs to early stop if no best performance is found', type=int, default=20)
    
    # graph construction arguments
    parser.add_argument('--edge_type', help='topk or <8A', type=str, default='8A', choices=['topk', '8A'])
    parser.add_argument('--edge_threshold', help='dist under threshold is an edge', type=float, default=8.0)
    parser.add_argument('--scalar_only', help='only use the scalar features, no vector features', action='store_true')
        
    args = parser.parse_args()
    
    if args.task == 'cath' and args.multilabel:
        print("CATH prediction is not a multilabel task!")
        raise NotImplementedError
    
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

def get_logger(log_file):
    '''return a logger to output on the console and save to the log file in the same time'''
    logger = logging.getLogger()
    logging_filename = log_file
    formater = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(filename=logging_filename, mode='w')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formater)
    console_handler.setFormatter(formater)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def raw_to_graphs(data, protein_data_set):
    graph_data_set = []
    for i in list(data.keys()):
        coords = data[i]["coordinates"]
        data[i]['coordinates'] = list(
            zip(coords['N'], coords['CA'], coords['C'], coords['O']))
    for (cath_id, data) in tqdm(list(data.items())):
    # for (cath_id, data) in list(data.items()):
        graph_data_set.append(protein_data_set.featurize_as_graph(
            data, cath_id, protein_data_set.graph_type))

    return graph_data_set

def get_id2label(protein_graphs, label_file):
    '''The label file should be a json file storing a dict like this: {prot_id: ["1.1.1.1", "2.2.2.2"]}'''
    with open(label_file) as f:
        all_id2label = json.load(f)
    id2label = {}
    for prot in protein_graphs:
        id2label[prot.name] = all_id2label[prot.name]
    
    return id2label

def evaluation(model, lookupset, testset, logger, args):
    model.eval()
    with torch.no_grad():
        lookup_emb, test_emb = [], []
        lookup_loader = torch_geometric.loader.DataLoader(lookupset, batch_size=16, shuffle=False, drop_last=False)
        test_loader = torch_geometric.loader.DataLoader(testset, batch_size=16, shuffle=False, drop_last=False)
        for idx, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(device)
            if args.parallel:
                emb = model.module.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name, batch=test_batch.batch)
            else:
                emb = model.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name, batch=test_batch.batch)
            test_emb.append(emb.cpu().detach())
        for idx, lookup_batch in enumerate(lookup_loader):
            lookup_batch = lookup_batch.to(device)
            if args.parallel:
                emb = model.module.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr, name=lookup_batch.name, batch=lookup_batch.batch)
            else:
                emb = model.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr, name=lookup_batch.name, batch=lookup_batch.batch)
            lookup_emb.append(emb.cpu().detach())
        lookup_emb = torch.cat(lookup_emb, dim=0)
        test_emb = torch.cat(test_emb, dim=0)
        
    lookup_id2label = get_id2label(lookupset, args.label_file)
    test_id2label = get_id2label(testset, args.label_file)
    lookup_labels = list(lookup_id2label.values())
    test_labels = list(test_id2label.values())
    
    distance = torch.cdist(test_emb, lookup_emb)
    test_id2pred = {}
    for idx, test_id in enumerate(test_id2label):
        test_id2pred[test_id] = lookup_labels[torch.argmin(distance[idx])]
    
    # TODO: implement multi-level accuracy
    correct = [0]
    for test_id, predictions in test_id2pred.items():
        hit = 0
        for label in predictions:
            for gt in test_id2label[test_id]:
                if label == gt:
                    hit = 1
        correct[0] += hit
    acc = [correct[i] / len(test_id2label) for i in range(len(correct))]
        
    model.train()
    
    return acc
        

def accs2str(acc):
    return '; '.join([f'acc-{i}: {acc[i]:.4f}' for i in range(len(acc))])    

def train(model, trainloader, lookupset, valset, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args):
    model.train()
    
    n_bad = 0
    best_acc = 0
    all_loss = []
    for epoch in range(num_epochs):
        start = time.time()
        acc = evaluation(model, lookupset, valset, logger, args)
        end_test = time.time()
        if acc[-1] < best_acc:
            n_bad += 1
            if n_bad >= args.early_stop_threshold:
                logger.info(f'No performance improvement for {args.early_stop_threshold} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! acc-{len(acc)-1}={acc[-1]:.4f}')
            n_bad = 0
            best_acc = acc[-1]
            # TODO: save checkpoint
            torch.save(model.state_dict(), exp_dir / 'best_checkpoint.pt')
        losses = []
        for train_idx, data in enumerate(trainloader):
            if args.parallel:
                labels = [item.label for item in data]
                labels = torch.cat(labels, dim=0)
                data = [item.to(device) for item in data]
                anchor, pos, neg = model(data)
                loss = criterion(anchor, pos, neg, labels)
            else:
                data = data.to(device)
                anchor, pos, neg = model(data)
                loss = criterion(anchor, pos, neg, data.label)
            losses.append(toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), exp_dir / 'last_checkpoint.pt')
        end_epoch = time.time()
        
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; {accs2str(acc)}; train time: {sec2min_sec(end_epoch - end_test)}')
        
    return all_loss

if __name__ == '__main__':
    # parse arguments
    args = get_args()
    start_overall = time.time()
    print(args)
    
    # set seeds
    seed_all(args.seed)
    
    # set directories
    root = Path.cwd()
    if args.task == 'cath':
        data_dir = root / 'data/cath-data'
    elif args.task == 'ec':
        data_dir = root / 'data/ec-data'
    log_dir = root / "log"
    exp_name = f'{args.exp_name}-{date}'
    exp_dir = log_dir / exp_name
    if not exp_dir.is_dir():
        print(f"Creating new log-directory: {exp_dir}")
        exp_dir.mkdir(parents=True)
    logger = get_logger(exp_dir / 'log.txt')
    save_args(args, exp_dir / 'config.yml')
    
    # load datasets
    if args.task == 'cath':
        datasplitter = DataSplitter(data_dir / "cath_data_S100.json", data_root=data_dir, task=args.task) # Changed for EC
    else:
        if args.multilabel:
            datasplitter = MultilabelDataSplitter(data_dir / "all.json", data_root=data_dir, task=args.task)
        else:
            datasplitter = DataSplitter(data_dir / "all.json", data_root=data_dir, task=args.task)
            
    train_splits, val, val_lookup20 = datasplitter.get_predef_splits()
    
    if args.multilabel:
        trainset = MultilabelDataset(train_splits, datasplitter, graph_type='gat', n_classes=args.n_levels, edge_type=args.edge_type, edge_threshold=args.edge_threshold, scalar_only=args.scalar_only)
    else:
        trainset = ProteinGraphDataset(train_splits, datasplitter, graph_type='gat', n_classes=args.n_levels, edge_type=args.edge_type, edge_threshold=args.edge_threshold, scalar_only=args.scalar_only)
    
    val_graphs = raw_to_graphs(val, trainset)
    lookup_graphs = list(trainset.get_id2graphs().values())
    
    if args.parallel:
        train_loader = torch_geometric.loader.DataListLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    else:
        train_loader = torch_geometric.loader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, follow_batch=['x_anchor', 'x_pos', 'x_neg'], num_workers=8)
    
    # set hyperparameters
    device_ids = [i for i in range(torch.cuda.device_count())]
    learning_rate = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size if not args.parallel else args.batch_size * len(device_ids) # the number of actual samples per batch might be higher due to batch-hard  sampling
    num_epochs = args.num_epochs  # will stop earlier if early stopping is triggered
    n_classes = args.n_levels  # number of class-lvls; makes it easier to adjust for other problems
    batch_hard = args.batch_hard # whether to activate batch_hard sampling (recommended)
    exclude_easy = args.exclude_easy # whether to exclude trivial samples (did not improve performa)
    margin = None # set this to a float to activate threshold-dependent loss functions (see TripletLoss)
    
    # load model
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
    if args.parallel:
        logger.info(f"Training on {len(device_ids)} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=device_ids, follow_batch=['x_anchor', 'x_pos', 'x_neg'])
    else:
        logger.info("Training on single GPU!")
    model.to(device)
    logger.info(model)
    
    criterion = TripletLoss(exclude_easy=exclude_easy,
                            batch_hard=batch_hard, margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    
    # training
    logger.info(f'Experiment name: {exp_name}')
    logger.info(f'LR: {learning_rate}, BS: {batch_size}, free Paras.: {count_parameters(model)}, n_epochs: {num_epochs}')
    train(model, train_loader, lookup_graphs, val_graphs, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args)
    
    end_train = time.time()
    logger.info(f'Total training time: {sec2min_sec(end_train - start_overall)}')
    
    # testing