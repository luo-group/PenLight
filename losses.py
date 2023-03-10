import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import toCPU

class TripletLoss(object):
    """
    Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'.
    """

    def __init__(self, margin=None, exclude_easy=False, batch_hard=True, device=torch.device('cuda:0')):
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
        self.device = device

    def __call__(self, anchor, pos, neg, Y):
        if self.batch_hard:
            dist_ap, dist_an = self.get_batch_hard(anchor, pos, neg, Y)
            dist_ap, dist_an = dist_ap.to(torch.float64), dist_an.to(torch.float64)
        else:
            dist_ap = self.distance(anchor, pos)
            dist_an = self.distance(anchor, neg)

        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss((dist_an - dist_ap), y)

        if self.exclude_easy:
            # TODO: this is a possible cause for loss=inf, (loss<0).sum() may be 0, we add a small epsilon=1e-8 and not sure whether it's going to work or not
            loss = loss.sum() / ((loss < 0).sum() + 1e-8)

        loss = loss.to(torch.float32)
        embeddings = torch.cat((anchor, pos, neg))
        # monitor['pos'].append(toCPU(dist_ap.mean()))
        # monitor['neg'].append(toCPU(dist_an.mean()))

        # monitor['min'].append(toCPU(embeddings.min(dim=1)[0].mean()))
        # monitor['max'].append(toCPU(embeddings.max(dim=1)[0].mean()))
        # monitor['mean'].append(toCPU(embeddings.mean(dim=1).mean()))

        # monitor['loss'].append(toCPU(loss))
        # monitor['norm'].append(toCPU(torch.norm(embeddings, p='fro')))

        return loss

    # https://gist.github.com/rwightman/fff86a015efddcba8b3c8008167ea705
    def get_hard_triplets(self, pdist, y, prev_mask_pos):
        n = y.size()[0]
        mask_pos = y.expand(n, n).eq(y.expand(n, n).t()).to(self.device)

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