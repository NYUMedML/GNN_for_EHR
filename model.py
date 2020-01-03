import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def clone_params(param, N):
    return nn.ParameterList([copy.deepcopy(param) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_of_nodes, num_of_heads, dropout, alpha,
                 concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_of_nodes = num_of_nodes
        self.num_of_heads = num_of_heads
        if concat:
            self.embed = nn.Embedding(num_of_nodes, in_features, padding_idx=0)
        self.W = clones(nn.Linear(in_features, hidden_features), num_of_heads)
        if not concat:
            self.V = nn.Linear(hidden_features, out_features)
        self.a = clone_params(nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True), num_of_heads)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if concat:
            self.norm = LayerNorm(num_of_heads * hidden_features)
        else:
            self.norm = LayerNorm(hidden_features)

    def attention(self, linear, a, N, data, edge):
        data = linear(data).unsqueeze(0)
        h = torch.cat((data[:, edge[0, :], :], data[:, edge[1, :], :]), dim=0)
        data = data.squeeze(0)
        assert not torch.isnan(h).any()

        # edge: 2*D x E
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.hidden_features * self.num_of_heads))
        assert not torch.isnan(edge_e).any()
        edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).cuda())
        # e_rowsum: N x 1
        e_rowsum[e_rowsum == 0] = 1
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward(self, data, edge):
        N = self.num_of_nodes
        if self.concat:
            data = self.embed(torch.arange(N).long().cuda())
            h_prime = torch.cat([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=1)
        else:
            h_prime = torch.stack([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=0).mean(dim=0)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(self.norm(h_prime))
        else:
            # if this layer is last layer,
            return self.V(F.relu(self.norm(h_prime)))


class GAT(nn.Module):

    def __init__(self, in_features, out_features, num_of_nodes, n_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.in_att = SpGraphAttentionLayer(in_features, out_features, out_features, num_of_nodes + 1, n_heads, dropout,
                                            alpha, concat=True)
        self.out_features = out_features
        self.out_att = SpGraphAttentionLayer(n_heads * out_features, out_features, 2, num_of_nodes + 1, n_heads,
                                             dropout, alpha, concat=False)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def data_to_edges(self, data):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            raise Exception('No CUDA')
        length = data.size()[0]
        nonzero = data.nonzero()
        if nonzero.size()[0] == 0:
            return torch.LongTensor([[0], [0]]).to(device), torch.LongTensor([[length + 1], [length + 1]]).to(device)
        if self.training:
            mask = torch.rand(nonzero.size()[0])
            mask = mask > 0
            nonzero = nonzero[mask]
            if nonzero.size()[0] == 0:
                return torch.LongTensor([[0], [0]]).to(device), torch.LongTensor([[length + 1], [length + 1]]).to(
                    device)
        nonzero = nonzero.transpose(0, 1) + 1
        lengths = nonzero.size()[1]
        input_edges = torch.cat((nonzero.repeat(1, lengths), \
                                 nonzero.repeat(lengths, 1).transpose(0, 1).contiguous().view((1, lengths ** 2))),
                                dim=0)

        nonzero = torch.cat((nonzero, torch.LongTensor([[length + 1]]).to(device)), dim=1)
        lengths = nonzero.size()[1]
        output_edges = torch.cat((nonzero.repeat(1, lengths), \
                                  nonzero.repeat(lengths, 1).transpose(0, 1).contiguous().view((1, lengths ** 2))),
                                 dim=0)
        return input_edges, output_edges

    def two_layer_block(self, data):
        input_edges, output_edges = self.data_to_edges(data)
        h_prime = self.in_att(data, input_edges)
        h_prime = self.dropout(h_prime)
        out = (self.out_att(h_prime, output_edges))
        return out[-1, :].unsqueeze(0)

    def forward(self, data):
        # Concate multiheads
        batch_size = data.size()[0]
        return torch.cat([self.two_layer_block(data[i, :]) for i in range(batch_size)], dim=0)
