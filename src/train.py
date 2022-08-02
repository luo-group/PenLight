import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import torch_geometric
import numpy as np
import random
import time
import os
import gc
import copy
from pathlib import Path
from model import GATModel
from utils import plotter, init_monitor, toCPU, count_parameters
from datasets import ProteinGraphDataset, DataSplitter
from tqdm import tqdm
import argparse

DEBUG = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_in_dim = (6, 3)
edge_in_dim = (32, 1)
# exp_max = 705.0  # in torch.float64, e^a for a > 705 will become inf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment name',
                        type=str, default='defaultExp')
    parser.add_argument('--parallel', help='whether use multi gpu', action='store_true')
    parser.add_argument('--model', help='model architecture', type=str, default='gatv2',
                        choices=['gat', 'gat1.1', 'gat1.2', 'gat1.3', 'gat1.4', 'gatv2', 'gatv2.01', 'gatv2.02', 'resgatv2.01', 'gat2.1', 'gat2.2', 'gat2.3', 'gat2.4', 'gat2.5', 'transformer', 'gine', 'linear', 'cg', 'gvp','linear'])
    parser.add_argument('--aa_embed', help='per residue embedding type', type=str, default='protT5',
                        choices=['onehot', 'blosum50', 'both', 'esm', 'all', 'protT5', 'protT5+'])
    parser.add_argument('--hidden', help='hidden channels',
                        type=int, default=128)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=32)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.0001)
    parser.add_argument('--batch_hard', help='whether use batch_hard', action='store_true')
    parser.add_argument('--exclude_easy', help='whether exclude easy triplets', action='store_true')
    parser.add_argument('--seed', help='global random seed',
                        type=int, default=42)
    parser.add_argument('--debug', help='whether initiate debug mode', action='store_true')
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0001)
    parser.add_argument('--description', help='training setting description', type=str, default='None')
    parser.add_argument('--amp', help='whether using mixed precision (may cause loss=nan)', action='store_true')
    parser.add_argument('--edge_type', help='topk or <8A', type=str, default='8A', choices=['topk', '8A'])
    parser.add_argument('--edge_threshold', help='dist under threshold is an edge', type=float, default=8.0)
    parser.add_argument('--validate', help='use which level of val acc for early stopping', type=str, default='H', choices=['C', 'A', 'T', 'H', 'avg'])
    parser.add_argument('--scalar_only', help='only use the scalar features, no vector features', action='store_true')
    parser.add_argument('--task', help='cath or ec', type=str, default='cath', choices=['cath', 'ec'])
    args = parser.parse_args()
    return args

# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097


def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


class Eval:
    def __init__(self, lookup, test, datasplitter, n_classes, train=None, name='cath'):
        self.lookup, self.lookupIdx2label = self.preproc(lookup, train)
        self.test, self.testIdx2label = self.preproc(test, train)
        self.id2label, self.label2id = datasplitter.parse_label_mapping_cath(
            # use only keys from the given lookup set
            set(lookup.keys()) | set(test.keys()),
        )
        self.name = name
        # self.log  = self.init_log()
        self.n_classes = n_classes
        self.accs = self.init_log()
        self.errs = self.init_log()
        self.distance = torch.nn.PairwiseDistance(p=2)

    def get_test_set(self):
        return self.test

    def get_lookup_set(self):
        return self.lookup

    def get_acc(self):
        return self.accs

    def get_err(self):
        return self.errs

    def init_log(self):
        log = dict()
        for i in range(self.n_classes):
            log[i] = list()
        return log

    def init_confmats(self):
        confmats = list()
        for i in range(self.n_classes):
            confmat = np.zeros((1, 2, 2))
            confmats.append(confmat)
        confmats = np.concatenate(confmats, axis=0)
        return confmats

    # TODO: modified, to be tested
    def preproc(self, data, protein_data_set):
        idx2label = dict()
        graph_data_set = []
        for idx, (seq_id, info) in enumerate(data.items()):
            idx2label[idx] = seq_id
        # for i in list(data.keys()):
        #     coords = data[i]["coordinates"]
        #     data[i]['coordinates'] = list(zip(coords['N'], coords['CA'], coords['C'], coords['O']))
        # for (cath_id, data) in tqdm(list(data.items())):
        #     graph_data_set.append(protein_data_set.featurize_as_graph(data, cath_id))
        return graph_data_set, idx2label

    def add_sample(self, y, yhat, confmats):
        wrong = False

        for class_lvl, true_class in enumerate(y):  # for each prediction
            # skip cases where the test protein did not have had any nn in lookupDB
            # --> It is by defnition not possible that those could be predicted correctly
            if np.isnan(true_class):
                continue
            if not wrong and true_class == yhat[class_lvl]:
                correct = 1  # count only those in
            else:  # if there is a wrong prediction on this level, lower-lvls are wrong by definition
                correct = 0
                wrong = True
            confmats[class_lvl, correct, correct] += 1
        return confmats

    def pdist(self, sample_1, sample_2, norm=2):
        sample_1.to(torch.device('cpu'))
        sample_2.to(torch.device('cpu'))
        return torch.cdist(sample_1.unsqueeze(dim=0), sample_2.unsqueeze(dim=0), p=norm).squeeze(dim=0)

    def mergeTopK(self, yhats):
        yhats = np.vstack(yhats)

        final_yhat = [None, None, None, None]
        for i in range(self.n_classes):
            (values, counts) = np.unique(yhats[:, i], return_counts=True)
            idxs = np.argmax(counts)
            nn_class = values[idxs]
            final_yhat[i] = nn_class
            mask = yhats[:, i] == nn_class
            yhats = yhats[mask, :]

        return final_yhat

    def mask_singletons(self, y):
        # Mask cases where the only annotated instance is the test protein
        # Those cases can not be predicted correctly without considering self-hits
        c, a, t, h = y
        if len(self.label2id[c][a][t][h]) == 1:  # if h-lvl has only the test prot
            y[-1] = np.nan
            if len(self.label2id[c][a][t]) == 1:  # if t-lvl "
                y[-2] = np.nan
                if len(self.label2id[c][a]) == 1:  # if a-lvl "
                    y[-3] = np.nan
                    if len(self.label2id[c]) == 1:  # if c-lvl "
                        y[-4] = np.nan
        return y

    def compute_err(self, confmat, n_bootstrap=10000):
        n_total = int(confmat.sum())  # total number of predictions
        n_wrong, n_correct = int(confmat[0, 0]), int(confmat[1, 1])
        preds = [0 for _ in range(n_wrong)] + [1 for _ in range(n_correct)]
        subset_accs = list()
        for _ in range(n_bootstrap):
            rnd_subset = random.choices(preds, k=n_total)
            subset_accs.append(sum(rnd_subset) / n_total)
        return np.std(np.array(subset_accs), axis=0, ddof=1)

    def evaluate(self, lookup, queries, n_nearest=1, update=True):
        p_dist = self.pdist(lookup.float(), queries.float())

        _, nn_idxs = torch.topk(p_dist, n_nearest, largest=False, dim=0)

        confmats = self.init_confmats()
        n_test = len(self.testIdx2label)
        for test_idx in range(n_test):  # for all test proteins
            y_id = self.testIdx2label[test_idx]  # get id of test protein
            # get annotation of test (groundtruth)
            y = copy.deepcopy(self.id2label[y_id])
            y = self.mask_singletons(y)

            nn_idx = nn_idxs[:, test_idx]
            yhats = list()
            for nn_i in nn_idx:
                # index of nearest neighbour (nn) in train set
                nn_i = int(toCPU(nn_i))
                # get id of nn (infer annotation)
                yhat_id = self.lookupIdx2label[nn_i]
                # get annotation of nn (groundtruth)
                yhat = self.id2label[yhat_id]
                yhat = np.asarray(yhat)
                yhats.append(yhat)

            if n_nearest == 1:
                assert len(yhats) == 1, print(
                    "More than one NN retrieved, though, n_nearest=1!")
                yhat = yhats[0]
            else:
                yhat = self.mergeTopK(yhats)
            confmats = self.add_sample(y, yhat, confmats)

        if update:  # for constantly monitoring test performance
            for i in range(self.n_classes):
                acc = confmats[i, 1, 1] / confmats[i].sum()
                err = self.compute_err(confmats[i])
                self.accs[i].append(acc)
                self.errs[i].append(err)
            return self.accs, self.errs

        else:  # to get baseline at the beginning
            accs, errs = list(), list()
            # get accuracy per difficulty level
            for i in range(self.n_classes):
                acc = confmats[i, 1, 1] / confmats[i].sum()
                err = self.compute_err(confmats[i])
                accs.append(acc)
                errs.append(err)
                print("Samples for class {}: {}".format(
                    i, sum(confmats[i, :, :])))
            return accs, errs


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, exclude_easy=False, batch_hard=True):
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)
        self.exclude_easy = exclude_easy
        self.reduction = 'none' if self.exclude_easy else 'mean'
        self.batch_hard = batch_hard
        self.sample = False
        self.softmax = nn.Softmax(dim=0)
        self.min = -10 ** 10
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(
                margin=margin, reduction=self.reduction)
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction=self.reduction)

    def __call__(self, anchor, pos, neg, Y, monitor):
        if self.batch_hard:
            dist_ap, dist_an = self.get_batch_hard(anchor, pos, neg, Y)
            dist_ap, dist_an = dist_ap.to(torch.float64), dist_an.to(torch.float64)
        else:
            dist_ap = self.distance(anchor, pos)
            dist_an = self.distance(anchor, neg)
        if DEBUG:
            print(f"dist_ap: {dist_ap}\ndist_an: {dist_an}")
            print(f"dist_an-dist_ap: {dist_an - dist_ap}")
            print(f'dist_an-dis_ap max: {(dist_an - dist_ap).max()}')

        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss((dist_an - dist_ap), y)
        if DEBUG:
            print(self.ranking_loss)
            print(f"loss: {loss}")

        if self.exclude_easy:
            # TODO: this is a possible cause for loss=inf, (loss<0).sum() may be 0, we add a small epsilon=1e-8 and not sure whether it's going to work or not
            loss = loss.sum() / ((loss < 0).sum() + 1e-8)
            if DEBUG:
                print(f'exclude_easy, loss:{loss}')
        loss = loss.to(torch.float32)
        embeddings = torch.cat((anchor, pos, neg))
        monitor['pos'].append(toCPU(dist_ap.mean()))
        monitor['neg'].append(toCPU(dist_an.mean()))

        monitor['min'].append(toCPU(embeddings.min(dim=1)[0].mean()))
        monitor['max'].append(toCPU(embeddings.max(dim=1)[0].mean()))
        monitor['mean'].append(toCPU(embeddings.mean(dim=1).mean()))

        monitor['loss'].append(toCPU(loss))
        monitor['norm'].append(toCPU(torch.norm(embeddings, p='fro')))

        return loss

    # https://gist.github.com/rwightman/fff86a015efddcba8b3c8008167ea705
    def get_hard_triplets(self, pdist, y, prev_mask_pos):
        n = y.size()[0]
        mask_pos = y.expand(n, n).eq(y.expand(n, n).t()).to(device)

        mask_pos = mask_pos if prev_mask_pos is None else prev_mask_pos * mask_pos

        # every protein that is not a positive is automatically a negative for this lvl
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).bool().cuda()] = 0  # mask self-interactions
        mask_neg[torch.eye(n).bool().cuda()] = 0

        if self.sample:
            # weighted sample pos and negative to avoid outliers causing collapse
            posw = (pdist + 1e-12) * mask_pos.float()
            posw[posw == 0] = self.min
            posw = self.softmax(posw)
            posi = torch.multinomial(posw, 1)

            dist_ap = pdist.gather(0, posi.view(1, -1))
            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # this was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (pdist + 1e-12)) * mask_neg.float()
            negw[posw == 0] = self.min
            negw = self.softmax(posw)
            negi = torch.multinomial(negw, 1)
            dist_an = pdist.gather(0, negi.view(1, -1))
        else:
            ninf = torch.ones_like(pdist) * float('-inf')
            dist_ap = torch.max(pdist * mask_pos.float(), dim=1)[0]
            nindex = torch.max(torch.where(mask_neg, -pdist, ninf), dim=1)[1]
            dist_an = pdist.gather(0, nindex.unsqueeze(0)).view(-1)

        return dist_ap, dist_an, mask_pos

    def get_batch_hard(self, anchor, pos, neg, Y):
        Y = torch.cat([Y[:, 0, :], Y[:, 1, :], Y[:, 2, :]], dim=0)
        X = torch.cat([anchor, pos, neg], dim=0)
        pdist = self.pdist(X)

        dist_ap, dist_an = list(), list()
        mask_pos = None

        for i in range(4):
            y = Y[:, i]
            dist_pos, dist_neg, mask_pos = self.get_hard_triplets(
                pdist, y, mask_pos)
            dist_ap.append(dist_pos.view(-1))
            dist_an.append(dist_neg.view(-1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        return dist_ap, dist_an

    def pdist(self, v):
        dist = torch.norm(v[:, None] - v, dim=2, p=2)
        return dist


def get_baseline(test):
    test_set = test.get_test_set()
    train_set = test.get_lookup_set()
    acc, err = test.evaluate(train_set, test_set, update=False)
    print(('BASELINE\nACC-C: {:.2f} +/- {:.2f}\nACC-A: {:.2f} +/- {:.2f}\n' +
           'ACC-T: {:.2f} +/- {:.2f}\nACC-H: {:.2f} +/- {:.2f}\nAvg. Acc: {:.2f} +/- {:.2f}').format(
        acc[0], err[0], acc[1], err[1], acc[2], err[2], acc[3], err[3],
        (acc[0] + acc[1] + acc[2] + acc[3]) /
        4, (err[0] + err[1] + err[2] + err[3]) / 4,
    ))
    return acc, err


# test performance during training on validation set (used also for early stopping)
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
                    tmp = model.module.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr,
                                                   name=test_batch.name)
                else:
                    tmp = model.module.single_pass((test_batch.node_s, test_batch.node_v), test_batch.edge_index, (
                        test_batch.edge_s, test_batch.edge_v), test_batch.name)
            else:
                if type == 'gat':
                    tmp = model.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr,
                                            name=test_batch.name)
                else:
                    tmp = model.single_pass((test_batch.node_s, test_batch.node_v), test_batch.edge_index, (
                        test_batch.edge_s, test_batch.edge_v), test_batch.name)
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


class Saver:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoint_p = experiment_dir / 'checkpoint.pt'
        self.best_performance = 0
        self.num_classes = 4
        self.epsilon = 1e-3

    def load_checkpoint(self, version='gatv2'):
        state = torch.load(self.checkpoint_p)
        if version == 'gatv2':
            model = GATModel(version=version, in_channel=1024, edge_dim=35)
        elif version == 'gatv2.01':
            model = GATModel(version=version, in_channel=1286, edge_dim=32)
        # model = GVPModel()
        model = torch_geometric.nn.DataParallel(model, device_ids=[0],
                                                follow_batch=['x_anchor', 'x_pos', 'x_neg'])
        # TODO: should be general for all our models
        model.load_state_dict(state['state_dict'])
        print('Loaded model from epch: {:.1f} with avg. acc: {:.3f}'.format(
            state['epoch'], self.best_performance))
        return model, state['epoch']

    def save_checkpoint(self, model, epoch, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, self.checkpoint_p)
        return None

    def check_performance(self, acc, model, epoch, optimizer, val='H'):
        if isinstance(acc, dict):  # if a list of accuracies is passed
            if val == 'H':
                new_performance = acc[3][-1]
            elif val == 'C':
                new_performance = acc[0][-1]
            elif val == 'A':
                new_performance = acc[1][-1]
            elif val == 'T':
                new_performance = acc[2][-1]
            elif val == 'avg':
                new_performance = (acc[0][-1] + acc[1][-1] + acc[2][-1] + acc[3][-1]) / 4
            else:
                raise NotImplementedError
        else:  # if a single Silhouette score is passed
            new_performance = acc
        if new_performance > self.best_performance + self.epsilon:
            self.save_checkpoint(model, epoch, optimizer)
            self.best_performance = new_performance
            print('New best performance found:  {:.3f}!'.format(self.best_performance))
            return self.best_performance
        return None


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

def main():
    args = get_args()
    # measure training time
    start_overall = time.time()
    # set random seeds
    SEED = args.seed
    seed_all(SEED)
    print(f'DEBUG={DEBUG}')

    # Set up directories
    root = Path.cwd().parent
    if args.task == 'cath':
        data_dir = root / "data-cath" 
        # data_dir = root / "data" # temporal
    else:
        data_dir = root / "data-ec"
        # data_dir = root / 'ec_data' # temporal
    log_dir = root / "log"
    experiment_name = args.experiment
    experiment_dir = log_dir / experiment_name
    if not experiment_dir.is_dir():
        print("Creating new log-directory: {}".format(experiment_dir))
        experiment_dir.mkdir(parents=True)
    description = f'model={args.model}\nlr={args.lr}\nbatch_size={args.batch_size}\noptimizer=Adam\n' \
                  f'aa_embed={args.aa_embed}\nbatch_hard={args.batch_hard}\nexclude_easy={args.exclude_easy}\n' \
                  f'parallel={args.parallel}\nrandom_seed={args.seed}\nweight_decay={args.weight_decay}\n' \
                  f'amp={args.amp}\nedge_type={args.edge_type}\ndescription={args.description}'
    with open(str(experiment_dir / "description"), 'w') as f:
        f.write(description)
    print(description)

    # Hyperparameters
    device_ids = [i for i in range(torch.cuda.device_count())]
    learning_rate = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size if not args.parallel else args.batch_size * \
        len(device_ids)  # the number of actual
    # samples per batch might be higher due to batch-hard  sampling
    num_epochs = 200  # will stop earlier if early stopping is triggered
    n_classes = 4  # number of class-lvls; makes it easier to adjust for other problems
    # counter for number of epochs that did not improve (counter for early stopping)
    n_bad = 0
    # threshold for number of epochs that did not improve (threshold for early stopping)
    n_thresh = 20
    # whether to activate batch_hard sampling (recommended)
    batch_hard = args.batch_hard
    # whether to exclude trivial samples (did not improve performa)
    exclude_easy = args.exclude_easy
    # set this to a float to activate threshold-dependent loss functions (see TripletLoss)
    margin = None
    edge_dim = 35
    if args.aa_embed == "onehot" or args.aa_embed == "blosum50":
        in_channel = 38
        node_emb = False
    elif args.aa_embed == "both":
        in_channel = 58
        node_emb = False
    elif args.aa_embed == "esm":
        in_channel = 1280
        if args.scalar_only:
            in_channel += 6
            edge_dim = 32
        node_emb = True
    elif args.aa_embed == 'esm+':
        in_channel == 1298
        node_emb = True
    elif args.aa_embed == "all":
        in_channel = 1338
        node_emb = True
    elif args.aa_embed == 'protT5':
        in_channel = 1024
        if args.scalar_only:
            in_channel += 6
            edge_dim = 32
        node_emb = True
    elif args.aa_embed == 'protT5+':
        in_channel = 1042
        node_emb = True
    else:
        raise NotImplementedError
    # initialize plotting class (used to monitor loss etc during training)
    pltr = plotter(experiment_dir)

    # Prepare datasets
    if args.task == 'cath':
        datasplitter = DataSplitter(data_dir / "cath_data_S100.json", data_root=data_dir, task=args.task) # Changed for EC
    else:
        datasplitter = DataSplitter(data_dir / "all.json", data_root=data_dir, task=args.task)
    train_splits, val, val_lookup20 = datasplitter.get_predef_splits()
    # cop = copy.deepcopy(val)
    train = ProteinGraphDataset(train_splits, datasplitter, graph_type='gvp' if args.model == 'gvp' else 'gat', n_classes=n_classes, edge_type=args.edge_type, edge_threshold=args.edge_threshold, scalar_only=args.scalar_only)
    # print(train[0])
    val_graphs = raw_to_graphs(val, train)
    lookup_graphs = list(train.get_id2graphs().values())

    val20 = Eval(val_lookup20, val, datasplitter, n_classes, train)

    if args.parallel:
        train_loader = torch_geometric.loader.DataListLoader(train, batch_size=batch_size, shuffle=True,
                                                             drop_last=True, num_workers=8)
    else:
        train_loader = torch_geometric.loader.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                                                         follow_batch=['x_anchor', 'x_pos', 'x_neg'], num_workers=8)

    model = GATModel(in_channel=in_channel, node_emb=node_emb, edge_dim=edge_dim, version=args.model, task=args.task)

    # if torch.cuda.device_count() > 1 and args.parallel:
    if args.parallel:
        print(f"Let's use {len(device_ids)} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=device_ids,
                                                follow_batch=['x_anchor', 'x_pos', 'x_neg'])
    model.to(device)
    print(model)

    criterion = TripletLoss(exclude_easy=exclude_easy,
                            batch_hard=batch_hard, margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    saver = Saver(experiment_dir)
    saver.save_checkpoint(model, 0, optimizer)

    gc.collect()
    print('###### Training parameters ######')
    print('Experiment name: {}'.format(experiment_name))
    print('LR: {}, BS: {}, free Paras.: {}, n_epochs: {}'.format(
        learning_rate, batch_size, count_parameters(model), num_epochs))
    print('#############################\n')
    print('Start training now!')

    monitor = init_monitor()
    if args.amp:
        scaler = GradScaler()
    for epoch in range(num_epochs):  # for each epoch

        # =================== testing =====================
        start = time.time()
        acc, err = testing(model, val20, val_graphs, lookup_graphs, type='gvp' if args.model == 'gvp' else 'gat', parallel=args.parallel)  # evaluate using the validation
        test_time = time.time() - start
        new_best = saver.check_performance(acc, model, epoch, optimizer, val=args.validate)  # early stopping class

        if new_best is None:  # if the new accuracy was worse than a previous one
            n_bad += 1
            if n_bad >= n_thresh:  # if more than n_bad consecutive epochs were worse, break training
                break
        else:  # if the new accuracy is larger than the previous best one by epsilon, reset counter
            n_bad = 0

        # =================== training =====================
        # monitor epoch-wise performance
        epoch_monitor = init_monitor()
        start = time.time()
        # for each batch in the training set
        for train_idx, data in enumerate(train_loader):
            if args.parallel:
                labels = [item.label for item in data]
                labels = torch.cat(labels, dim=0)
                data = [item.to(device) for item in data]
                if args.amp:
                    with autocast():
                        anchor, pos, neg = model(data)
                        loss = criterion(anchor, pos, neg,
                                         labels, epoch_monitor)
                else:
                    anchor, pos, neg = model(data)
                    loss = criterion(anchor, pos, neg, labels, epoch_monitor)
            else:
                data = data.to(device)
                if args.amp:
                    with autocast():
                        anchor, pos, neg = model(data)
                        loss = criterion(anchor, pos, neg,
                                         data.label, epoch_monitor)
                else:
                    anchor, pos, neg = model(data)
                    loss = criterion(anchor, pos, neg,
                                     data.label, epoch_monitor)
            # =================== backward ====================
            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if train_idx % 100 == 0:
                torch.cuda.empty_cache()
            train_time = time.time() - start
        scheduler.step(sum(epoch_monitor['loss']) / len(epoch_monitor['loss']))
        torch.cuda.empty_cache()
        # monitor various metrics during training
        monitor['loss'].append(
            sum(epoch_monitor['loss']) / len(epoch_monitor['loss']))
        monitor['norm'].append(
            sum(epoch_monitor['norm']) / len(epoch_monitor['norm']))
        monitor['pos'].append(sum(epoch_monitor['pos']) /
                              len(epoch_monitor['pos']))
        monitor['neg'].append(sum(epoch_monitor['neg']) /
                              len(epoch_monitor['neg']))
        monitor['min'].append(sum(epoch_monitor['min']) /
                              len(epoch_monitor['min']))
        monitor['max'].append(sum(epoch_monitor['max']) /
                              len(epoch_monitor['max']))
        monitor['mean'].append(
            sum(epoch_monitor['mean']) / len(epoch_monitor['mean']))

        train_time = time.time() - start

        # ===================log========================
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # draw plots only every fifth epoch
            # TODO: annotated for now, enabled when get baseline func is modified
            pltr.plot_acc(acc)
            pltr.plot_distances(monitor['pos'], monitor['neg'])
            pltr.plot_loss(monitor['loss'], file_name='loss.pdf')
            pltr.plot_loss(monitor['norm'], file_name='norm.pdf')
            pltr.plot_minMaxMean(monitor)

        # Always print training progress to console
        # print(f'epoch {epoch + 1}: train loss: {monitor["loss"][-1]:.3f}, train-time: {train_time}[s]')
        # print(f'acc_times: {acc_times[0]}s, {acc_times[1]}s, {acc_times[2]}s, {acc_times[3]}s')
        print(('epoch [{}/{}], train loss: {:.3f}, train-time: {:.1f}[s], test-time: {:.1f}[s], ' +
               'ACC-C: {:.2f}, ACC-A: {:.2f}, ACC-T: {:.2f}, ACC-H: {:.2f} ## Avg. Acc: {:.2f}').format(
            epoch + 1, num_epochs,
            monitor['loss'][-1],
            train_time, test_time,
            acc[0][-1], acc[1][-1], acc[2][-1], acc[3][-1],
            (acc[0][-1] + acc[1][-1] + acc[2][-1] + acc[3][-1]) / 4
        ))
        # input()

    end_overall = time.time()
    print(end_overall - start_overall)
    print("Total training time: {:.1f}[m]".format(
        (end_overall - start_overall) / 60))


if __name__ == "__main__":
    main()
