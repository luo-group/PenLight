import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import random

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Blosum matrix citation: Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of
# TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021).
# https://doi.org/10.1038/s42003-021-02610-3
blosum50_20aa = {
    'A': torch.tensor((5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)),
    'R': torch.tensor((-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3)),
    'N': torch.tensor((-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3)),
    'D': torch.tensor((-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4)),
    'C': torch.tensor((-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)),
    'Q': torch.tensor((-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3)),
    'E': torch.tensor((-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3)),
    'G': torch.tensor((0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)),
    'H': torch.tensor((-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4)),
    'I': torch.tensor((-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4)),
    'L': torch.tensor((-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1)),
    'K': torch.tensor((-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3)),
    'M': torch.tensor((-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1)),
    'F': torch.tensor((-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1)),
    'P': torch.tensor((-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)),
    'S': torch.tensor((1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2)),
    'T': torch.tensor((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)),
    'W': torch.tensor((-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)),
    'Y': torch.tensor((-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)),
    'V': torch.tensor((0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5))
}

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

def seq2onehot(seq):
    res = torch.zeros(len(seq), 20)
    for i, aa in enumerate(seq):
        if aa not in amino_acids:
            continue
        res[i, amino_acids.index(aa)] = 1.0
    return res

def seq2blosum50(seq):
    res = torch.zeros(len(seq), 20)
    for i, aa in enumerate(seq):
        if aa not in amino_acids:
            continue
        res[i] = blosum50_20aa[aa]
    return res
    
class plotter:
    def __init__(self, log_dir):
        self.init_plotting()
        self.log_dir = log_dir

    def init_plotting(self):
        params = {
            'axes.labelsize': 13,  # increase font size for axis labels
        }
        plt.rc(params)  # apply parameters
        return plt, sn

    def merge_pdfs(self, pdf1_path, pdf2_path, output_path):
        # Merge two PDFs
        from PyPDF2 import PdfFileMerger
        pdfs = [pdf1_path, pdf2_path]

        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(str(output_path))
        merger.close()
        return None

    def plot_minMaxMean(self, train_minMax, file_name='minMaxMean.pdf'):
        plt, _ = self.init_plotting()

        # Plot first three samples in Batch in one figure
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(train_minMax['min'])
        y = np.asarray(train_minMax['max'])
        z = np.asarray(train_minMax['mean'])
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g', label='Min')
        axes.plot(L, y, 'r', label='Max')
        axes.plot(L, z, 'b', label='Mean')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('min/max/mean')

        _ = plt.legend()
        plt.title('Min/Max/Mean development')

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_distances(self, dist_pos, dist_neg, file_name='distances.pdf'):
        plt, _ = self.init_plotting()

        # Plot first three samples in Batch in one figure
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(dist_pos)
        y = np.asarray(dist_neg)
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g',  label='Dist. Pos')
        axes.plot(L, y, 'r', label='Dist. Neg')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('Distances')

        _ = plt.legend()
        plt.title('Distance development')

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_acc(self, acc, baseline=None, diff_classes=4, file_name='acc.pdf'):

        plt, _ = self.init_plotting()


        fig, axes = plt.subplots(1, 1)

        colors = ['r', 'b', 'g', 'm']
        for diff_class in range(diff_classes):
            x = np.asarray(acc[diff_class])
            max_acc_idx = np.argmax(x)
            max_acc = x[max_acc_idx]
            L = np.arange(1, x.size+1)
            # b = np.ones(x.size) * baseline[diff_class]
            axes.plot(L, x, colors[diff_class],  label='LvL.: {} # {:.3f} in epoch {}'.format(
                diff_class, max_acc, max_acc_idx))
            # axes.plot(L, b, colors[diff_class]+'-.')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('Accuracy')

        _ = plt.legend()
        plt.title(file_name.replace('.pdf', ''))

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None

    def plot_loss(self, train, test=None, file_name='loss.pdf'):
        test = train if test is None else test
        plt, _ = self.init_plotting()
        fig, axes = plt.subplots(1, 1)

        x = np.asarray(train)
        y = np.asarray(test)
        L = np.arange(1, x.size+1)

        axes.plot(L, x, 'g',  label='Train')
        axes.plot(L, y, 'r--', label='Test')

        axes.set_xlabel('Steps/Batches')
        axes.set_ylabel('Loss')

        _ = plt.legend()
        plt.title(file_name.replace('loss.pdf', ''))

        pdf_path = self.log_dir / file_name
        fig.savefig(str(pdf_path), format='pdf')

        plt.close(fig)  # close figure handle
        return None


def init_monitor():
    monitor = dict()

    monitor['loss'] = list()
    monitor['norm'] = list()

    monitor['pos'] = list()
    monitor['neg'] = list()

    monitor['min'] = list()
    monitor['max'] = list()
    monitor['mean'] = list()
    return monitor


# move torch/GPU tensor to numpy/CPU
def toCPU(data):
    return data.cpu().detach().numpy()


# count number of free parameters in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)