import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

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


class GraphLayer(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, num_of_nodes,
                 num_of_heads, dropout, alpha, concat=True):
        super(GraphLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_of_nodes = num_of_nodes
        self.num_of_heads = num_of_heads
        self.W = clones(nn.Linear(in_features, hidden_features), num_of_heads)
        self.a = clone_params(nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True), num_of_heads)
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        if not concat:
            self.V = nn.Linear(hidden_features, out_features)
        else:
            self.V = nn.Linear(num_of_heads * hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if concat:
            self.norm = LayerNorm(hidden_features)
        else:
            self.norm = LayerNorm(hidden_features)

    def initialize(self):
        for i in range(len(self.W)):
            nn.init.xavier_normal_(self.W[i].weight.data)
        for i in range(len(self.a)):
            nn.init.xavier_normal_(self.a[i].data)
        if not self.concat:
            nn.init.xavier_normal_(self.V.weight.data)
            nn.init.xavier_normal_(self.out_layer.weight.data)

    def attention(self, linear, a, N, data, edge):
        data = linear(data).unsqueeze(0)
        assert not torch.isnan(data).any()
        # edge: 2*D x E
        h = torch.cat((data[:, edge[0, :], :], data[:, edge[1, :], :]), dim=0)
        data = data.squeeze(0)
        # h: N x out
        assert not torch.isnan(h).any()
        # edge_h: 2*D x E
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        # edge: 2*D x E
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.hidden_features * self.num_of_heads))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).to(device))
        # e_rowsum: N x 1
        row_check = (e_rowsum == 0)
        e_rowsum[row_check] = 1
        zero_idx = row_check.nonzero()[:, 0]
        edge_e = edge_e.add(
            torch.sparse.FloatTensor(zero_idx.repeat(2, 1), torch.ones(len(zero_idx)).to(device), torch.Size([N, N])))
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward(self, edge, data=None):
        N = self.num_of_nodes
        if self.concat:
            h_prime = torch.cat([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=1)
        else:
            h_prime = torch.stack([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=0).mean(
                dim=0)
        h_prime = self.dropout(h_prime)
        if self.concat:
            return F.elu(self.norm(h_prime))
        else:
            return self.V(F.relu(self.norm(h_prime)))


class VariationalGNN(nn.Module):

    def __init__(self, in_features, out_features, num_of_nodes, n_heads, n_layers,
                 dropout, alpha, variational=True, none_graph_features=0, concat=True):
        super(VariationalGNN, self).__init__()
        self.variational = variational
        self.num_of_nodes = num_of_nodes + 1 - none_graph_features
        self.embed = nn.Embedding(self.num_of_nodes, in_features, padding_idx=0)
        self.in_att = clones(
            GraphLayer(in_features, in_features, in_features, self.num_of_nodes,
                       n_heads, dropout, alpha, concat=True), n_layers)
        self.out_features = out_features
        self.out_att = GraphLayer(in_features, in_features, out_features, self.num_of_nodes,
                                  n_heads, dropout, alpha, concat=False)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.parameterize = nn.Linear(out_features, out_features * 2)
        self.out_layer = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1))
        self.none_graph_features = none_graph_features
        if none_graph_features > 0:
            self.features_ffn = nn.Sequential(
                nn.Linear(none_graph_features, out_features//2),
                nn.ReLU(),
                nn.Dropout(dropout))
            self.out_layer = nn.Sequential(
                nn.Linear(out_features + out_features//2, out_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_features, 1))
        for i in range(n_layers):
            self.in_att[i].initialize()

    def data_to_edges(self, data):
        length = data.size()[0]
        nonzero = data.nonzero()
        if nonzero.size()[0] == 0:
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        if self.training:
            mask = torch.rand(nonzero.size()[0])
            mask = mask > 0.05
            nonzero = nonzero[mask]
            if nonzero.size()[0] == 0:
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        nonzero = nonzero.transpose(0, 1) + 1
        lengths = nonzero.size()[1]
        input_edges = torch.cat((nonzero.repeat(1, lengths),
                                 nonzero.repeat(lengths, 1).transpose(0, 1)
                                 .contiguous().view((1, lengths ** 2))), dim=0)

        nonzero = torch.cat((nonzero, torch.LongTensor([[length + 1]]).to(device)), dim=1)
        lengths = nonzero.size()[1]
        output_edges = torch.cat((nonzero.repeat(1, lengths),
                                  nonzero.repeat(lengths, 1).transpose(0, 1)
                                  .contiguous().view((1, lengths ** 2))), dim=0)
        return input_edges.to(device), output_edges.to(device)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encoder_decoder(self, data):
        N = self.num_of_nodes
        input_edges, output_edges = self.data_to_edges(data)
        h_prime = self.embed(torch.arange(N).long().to(device))
        for attn in self.in_att:
            h_prime = attn(input_edges, h_prime)
        if self.variational:
            h_prime = self.parameterize(h_prime).view(-1, 2, self.out_features)
            h_prime = self.dropout(h_prime)
            mu = h_prime[:, 0, :]
            logvar = h_prime[:, 1, :]
            h_prime = self.reparameterise(mu, logvar)
            mu = mu[data, :]
            logvar = logvar[data, :]
        h_prime = self.out_att(output_edges, h_prime)
        if self.variational:
            return h_prime[-1], 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) / mu.size()[0]
        else:
            return h_prime[-1], torch.tensor(0.0).to(device)

    def forward(self, data):
        # Concate batches
        batch_size = data.size()[0]
        # In eicu data the first feature whether have be admitted before is not included in the graph
        if self.none_graph_features == 0:
            outputs = [self.encoder_decoder(data[i, :]) for i in range(batch_size)]
            return self.out_layer(F.relu(torch.stack([out[0] for out in outputs]))), \
                   torch.sum(torch.stack([out[1] for out in outputs]))
        else:
            outputs = [(data[i, :self.none_graph_features],
                        self.encoder_decoder(data[i, self.none_graph_features:])) for i in range(batch_size)]
            return self.out_layer(F.relu(
                torch.stack([torch.cat((self.features_ffn(torch.FloatTensor([out[0]]).to(device)), out[1][0]))
                             for out in outputs]))), \
                   torch.sum(torch.stack([out[1][1] for out in outputs]), dim=-1)