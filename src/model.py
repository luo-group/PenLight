# -*- codeing = utf-8 -*-
# @Time:  11:51 上午
# @Author: Jiaqi Luo
# @File: model.py
# @Software: PyCharm

from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, GINEConv, global_mean_pool, NNConv, CGConv
# import esm
import h5py
import time, copy
from datasets import _normalize

# emb_file = h5py.File('../data-cath/protT5.hdf5', 'r') # Changed for EC
# emb_file = h5py.File('../data-ec/ec_esm1b.h5', 'r') 

class GATModel(nn.Module):
    def __init__(self, in_channel=1024, hidden_channel=128, edge_dim=35, heads=8, drop_rate=0.5, node_emb=True, version='gatv2', concat=False, task='cath'):
        """
        :param in_channel: num of node features
        :param hidden_channel: 
        :param edge_dim: num of edge attributes, default 35 (from gvp)
        :param heads: num of attention heads
        :param drop_rate: dropout rate
        :param node_emb: whether include the esm-1b or protT5 per-residue embedding for each node
        :param version: 1 for GATConv, 2 for GATv2Conv, 3 for TransformerConv, 4 for GINEConv
        """
        super(GATModel, self).__init__()
        self.node_emb = node_emb
        self.hidden_channel = hidden_channel
        self.final_hidden = 512
        self.final_heads = 1
        self.version = version
        self.concat = concat
        if task == 'cath':
            self.emb_file = h5py.File('../data-cath/protT5.hdf5', 'r')
            # self.emb_file = h5py.File('../data/protT5.hdf5', 'r') # temporal
        elif task == 'ec':
            self.emb_file = h5py.File('../data-ec/ec_esm1b.h5', 'r')
            # self.emb_file = h5py.File('../ec_data/ec_esm1b.h5', 'r') # temporal
        else:
            raise NotImplementedError
        if version == 0:
            self.linear = nn.Sequential(
                nn.Linear(in_channel, 256), nn.Tanh(), nn.Linear(256, 128))
        elif version == 1:
            self.conv1 = GATConv(in_channel, hidden_channel,
                                 heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATConv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                 edge_dim=edge_dim)
            self.linear = nn.Sequential(
                nn.Linear(self.final_hidden * self.final_heads, 128))
        elif version == 1.1:
            self.conv1 = GATConv(in_channel, out_channels=128,
                                 heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = lambda x: x
            self.linear = nn.Sequential(nn.Linear(1024, 256), nn.Dropout(
                drop_rate), nn.ReLU(), nn.Linear(256, 128))
        elif version == 1.2:
            self.conv1 = GATv2Conv(
                in_channel, out_channels=128, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = lambda x: x
            self.linear = nn.Sequential(nn.Linear(1024, 256), nn.Dropout(
                drop_rate), nn.ReLU(), nn.Linear(256, 128))
        elif version == 1.3:
            self.conv1 = GATConv(in_channel, 128, 4)
            self.conv2 = GATConv(512, 128, 4)
            self.conv3 = GATConv(512, 256, 4)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.dense = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        elif version == 1.4:
            self.conv1 = GATConv(in_channel, 128, 4, edge_dim=edge_dim)
            self.conv2 = GATConv(512, 128, 4, edge_dim=edge_dim)
            self.conv3 = GATConv(512, 256, 4, edge_dim=edge_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.dense = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        elif version == 'gatv2':
            self.conv1 = GATv2Conv(in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads, 128))
        elif version == 'gatv2.01':
            self.conv1 = GATv2Conv(in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads, 256))
        elif version == 2.02:
            self.conv1 = GATv2Conv(in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads, 192))
        elif version == 2.03:
            self.conv1 = GATv2Conv(in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads, 256))
        elif version == 2.1:
            self.conv1 = GATv2Conv(
                in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, hidden_channel, heads=heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.conv3 = GATv2Conv(hidden_channel * heads, out_channels=self.final_hidden, heads=self.final_heads,
                                   dropout=drop_rate, edge_dim=edge_dim)
            self.linear = nn.Sequential(
                nn.Linear(self.final_hidden * self.final_heads, 128))
        elif version == 2.2:
            self.conv1 = GATv2Conv(
                in_channel, hidden_channel, heads=heads, dropout=drop_rate)
            self.conv2 = GATv2Conv(
                hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate)
            self.linear = nn.Sequential(
                nn.Linear(self.final_hidden * self.final_heads, 128))
        elif version == 2.3:
            self.conv1 = GATv2Conv(in_channel, 128, 4)
            self.conv2 = GATv2Conv(512, 128, 4)
            self.conv3 = GATv2Conv(512, 256, 4)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.dense = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        elif version == 2.4:
            self.conv1 = GATv2Conv(in_channel, 128, 4, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(512, 128, 4, edge_dim=edge_dim)
            self.conv3 = GATv2Conv(512, 256, 4, edge_dim=edge_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.dense = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        elif version == 2.5:
            self.conv1 = GATv2Conv(in_channel, hidden_channel, heads=heads, dropout=drop_rate, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_channel * heads, self.final_hidden, heads=self.final_heads, dropout=drop_rate,
                                   edge_dim=edge_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.5)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads + hidden_channel * heads, 128))
        elif version == 3:
            self.conv1 = TransformerConv(
                in_channel, hidden_channel, heads=heads, dropout=drop_rate)
            self.conv2 = TransformerConv(hidden_channel * heads, self.final_hidden, heads=self.final_heads,
                                         dropout=drop_rate)
            self.linear = nn.Sequential(nn.Linear(self.final_hidden * self.final_heads, 256))
        elif version == 4:
            self.conv1 = GINEConv(nn.Sequential(
                nn.Linear(in_channel, self.hidden_channel)), edge_dim=edge_dim)
            self.conv2 = GINEConv(nn.Sequential(nn.Linear(self.hidden_channel, self.hidden_channel)),
                                  edge_dim=edge_dim)
            self.linear = nn.Sequential(nn.Linear(self.hidden_channel, 128))
        elif version == 5:
            self.conv1 = CGConv(
                channels=(in_channel, self.hidden_channel), dim=edge_dim, aggr='mean')
            self.conv2 = CGConv(
                channels=(self.hidden_channel, self.hidden_channel), dim=edge_dim, aggr='mean')
            self.linear = nn.Sequential(nn.Linear(self.hidden_channel, 128))
        else:
            raise NotImplementedError

    def single_pass(self, x, edge_index, edge_attr, name=None, seq=None, batch=None):
        if self.node_emb:
            if batch is not None:
                embeddings = []
                for idx, i in enumerate(name):
                    emb = torch.tensor(self.emb_file[i][()])
                    embeddings.append(emb)
                embeddings = torch.cat(embeddings, dim=0).to(
                    x.device).to(x.dtype)
                if self.concat:
                    x = torch.cat([x, embeddings], dim=1)
                else:
                    x = embeddings
                torch.cuda.empty_cache()
            else:
                embedding = torch.tensor(self.emb_file[name][()])
                if len(embedding) < len(x):
                    padding = torch.zeros(
                        (len(x) - len(embedding), embedding.shape[-1]))
                    embedding = torch.cat([embedding, padding], dim=0)
                embedding = embedding.to(x.device).to(x.dtype)
                if self.concat:
                    x = torch.cat([x, embedding], dim=1)
                else:
                    x = embedding
        if self.version == 1.1 or self.version == 1.2:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
        elif self.version == 0:
            pass
        elif self.version == 2.03:
            x_conv1 = F.relu(self.conv1(x, edge_index, edge_attr) + x)
            x = F.relu(self.conv2(x_conv1, edge_index, edge_attr) + x_conv1)
        elif self.version == 2.1:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_attr)
            x = F.relu(x)
        elif self.version == 2.2:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        elif self.version == 1.3 or self.version == 2.3:
            conv1_out = self.conv1(x, edge_index)
            conv2_out = self.conv2(conv1_out, edge_index)
            conv3_out = self.conv3(conv2_out, edge_index)
            # residual concat
            out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
            out = self.dropout(self.relu(out))  # [n_nodes, 2048]
            # aggregate node vectors to graph
            if batch is not None:
                out = global_mean_pool(out, batch)
            else:
                out = out.mean(dim=0, keepdim=True)
            return self.dense(out)  # [bs, 128]
            # return self.dense(out) + 0.5
        elif self.version == 1.4 or self.version == 2.4:
            x = self.conv1(x, edge_index, edge_attr)
            conv2_out = self.conv2(x, edge_index, edge_attr)
            conv3_out = self.conv3(conv2_out, edge_index, edge_attr)
            # residual concat
            out = torch.cat((x, conv2_out, conv3_out), dim=-1)
            out = self.dropout(self.relu(out))  # [n_nodes, 2048]
            # aggregate node vectors to graph
            if batch is not None:
                out = global_mean_pool(out, batch)
            else:
                out = out.mean(dim=0, keepdim=True)
            return self.dense(out)  # [bs, 128]
            # return self.dense(out) + 0.5
        elif self.version == 2.5:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x, inplace=True)
            conv2_out = self.conv2(x, edge_index, edge_attr)
            conv2_out = F.relu(conv2_out, inplace=True)
            x = torch.cat((x, conv2_out), dim=-1)
            x = self.dropout(self.relu(x))
        elif self.version == 3:
            x = self.conv1(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.conv2(x, edge_index)
            x = F.relu(x, inplace=True)
        else:
            # print(x.shape, edge_index.shape, edge_attr.shape)
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x, inplace=True)
            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x, inplace=True)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        x = self.linear(x)

        return x

    def forward(self, X):
        anchor = self.single_pass(X.x_anchor, X.edge_index_anchor, X.edge_attr_anchor, X.name_anchor,
                                  X.seq_anchor, X.x_anchor_batch if hasattr(X, 'x_anchor_batch') else None)
        pos = self.single_pass(X.x_pos, X.edge_index_pos, X.edge_attr_pos, X.name_pos,
                               X.seq_pos, X.x_pos_batch if hasattr(X, 'x_pos_batch') else None)
        neg = self.single_pass(X.x_neg, X.edge_index_neg, X.edge_attr_neg, X.name_neg,
                               X.seq_neg, X.x_neg_batch if hasattr(X, 'x_neg_batch') else None)

        return anchor, pos, neg