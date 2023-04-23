from train import *
import copy
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
import json

def get_gt(cath_file):
    ec = {}
    with open(cath_file) as f:
        lines = f.readlines()
    for line in lines:
        entries = line.strip().split()
        ec[entries[0]] = [(entries[i]) for i in range(1, 5)]
    return ec
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

def get_part_test_embeddings(test95, test_json, save_path):
    with open(test_json) as f:
        test_ids = (json.load(f).keys())
    data = torch.load(test95)
    part_test = {}
    for i in test_ids:
        part_test[i] = data[i]
    print(f'Test set size: {len(part_test)}')
    torch.save(part_test, save_path)

def emb2pred(testset, lookupset, save_path, ec_table):
    id2pred = {}

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
    with open(save_path, 'w') as f:
        json.dump(id2pred, f)
    
    return id2pred

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # set random seeds
    SEED = 42
    seed_all(SEED)

    root = Path.cwd().parent
    data_dir = root / "data" # Changed for EC
    log_dir = root / "log" # Changed for EC
    # data_dir = root / "ec_data"
    # log_dir = root / "log-ec"
    # model_dir = log_dir / 'EC-GATv2.01_so_H_esm_lr1e-4_wd_lrs_bs1*128_s3341'
    model_dir = log_dir / 'GATv2_new_train_val_bs1*128_s33'

    # get_part_test_embeddings(os.path.join(model_dir, 'emb_test95.pt'), os.path.join(data_dir, 'splits_json/test70.json'), os.path.join(model_dir, 'emb_test70.pt'))
    # get_part_test_embeddings(os.path.join(model_dir, 'emb_test95.pt'), os.path.join(data_dir, 'splits_json/test50.json'), os.path.join(model_dir, 'emb_test50.pt'))
    # get_part_test_embeddings(os.path.join(model_dir, 'emb_test95.pt'), os.path.join(data_dir, 'splits_json/test40.json'), os.path.join(model_dir, 'emb_test40.pt'))
    # get_part_test_embeddings(os.path.join(model_dir, 'emb_test95.pt'), os.path.join(data_dir, 'splits_json/test30.json'), os.path.join(model_dir, 'emb_test30.pt'))
    # input()

    saver = Saver(model_dir)
    model, _ = saver.load_checkpoint()
    print(model)
    model.to(device)
    model.eval()
    
    # Prepare test and lookup dataset
    n_classes = 4
    datasplitter = DataSplitter(data_dir / "cath_data_S100.json") # Changed for EC
    # datasplitter = DataSplitter(data_dir / "all.json")
    # test_file = 'test95.json'
    # test, lookup = datasplitter.get_test_splits(test_file='../ec_data/splits_json/' + test_file)
    test_file = 'test.json'
    test, lookup = datasplitter.get_test_splits()
    test2 = copy.deepcopy(test)
    train = ProteinGraphDataset(test2, datasplitter, graph_type='gat', n_classes=n_classes,
                                seq_emb='esm', edge_type='8A', edge_threshold=8, scalar_only=True)
    test_graphs = raw_to_graphs(test, train)
    lookup_graphs = raw_to_graphs(lookup, train)
    print(len(test_graphs))

    emb_lookup = {}
    emb_test = {}
    for graph in tqdm(test_graphs):
        graph.to(device)
        # print(graph)
        # input()
        emb_test[graph.name] = {'embedding': model.single_pass(graph.x, graph.edge_index, graph.edge_attr, name=graph.name).to(torch.device('cpu')).detach(), 'label': str2nums(test[graph.name]['cath'])}
        graph.to(torch.device('cpu'))
    print(len(emb_test))
    torch.cuda.empty_cache()
    for graph in tqdm(lookup_graphs):
        graph.to(device)
        emb = model.single_pass(graph.x, graph.edge_index, graph.edge_attr, name=graph.name).cpu().detach()
        emb_lookup[graph.name] = {'embedding': emb, 'label': str2nums(lookup[graph.name]['cath'])}
        graph.to(torch.device('cpu'))
        torch.cuda.empty_cache()
    print(len(emb_lookup))
    torch.save(emb_lookup, os.path.join(model_dir, 'emb_lookup.pt'))
    torch.save(emb_test, os.path.join(model_dir, 'emb_' + test_file.split('.')[0] + '.pt'))

    gt_table = get_gt("../data/cath-domain-list-S100.txt")
    # gt_table = get_gt('../ec_data/pdb_ec_chain.txt')
    emb2pred(emb_test, emb_lookup, '../plots/cath-id2pred-GATv2-new.json', gt_table)