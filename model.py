import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn import GCNConv
from torch.nn import Parameter, Linear, LSTM


class JKNet(nn.Module):

    def __init__(self, cfg):
        super(JKNet, self).__init__()
        self.dropout = cfg.NN.dropout
        self.n_layer = cfg.NN.n_layer

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(cfg.Dataset.n_feat, cfg.NN.n_hid))
        for _ in range(1, cfg.NN.n_layer):
            self.convs.append(GCNConv(cfg.NN.n_hid, cfg.NN.n_hid))

        self.att = JumpingKnowledge(channels     = cfg.NN.n_hid, 
                                    num_layers   = cfg.NN.n_layer)
        self.out_lin = nn.Linear(cfg.NN.n_hid, cfg.Dataset.n_class)

    def forward(self, x, edge_index):
        hs = []
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            hs.append(x)

        h = self.att(hs)  # hs = [h^1,h^2,...,h^L], each h^l is (n, d).
        return self.out_lin(h)


class JumpingKnowledge(torch.nn.Module):
    def __init__(self, channels, num_layers):
        super(JumpingKnowledge, self).__init__()

        out_channels = (num_layers * channels) // 2        
        self.lstm = LSTM(channels, out_channels,
                             bidirectional=True, batch_first=True)
        self.att = Linear(2 * out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.att.reset_parameters()

    def forward(self, hs):
        assert isinstance(hs, list) or isinstance(hs, tuple)
        h = torch.stack(hs, dim=1)  # h is (n, L, d).

        alpha, _ = self.lstm(h) # alpha (n, L, dL). dL/2 is hid_channels of forward or backward LSTM
        out_channels = alpha.size()[-1]
        query, key = alpha[:, :, :out_channels//2], alpha[:, :, out_channels//2:]

        query_key = torch.cat([query, key], dim=-1)
        alpha = self.att(query_key).squeeze(-1)
        alpha = torch.softmax(alpha/self.att_temparature, dim=-1)

        return (h * alpha.unsqueeze(-1)).sum(dim=1) # h_i = \sum_{l} h_i^l * alpha_i^l
