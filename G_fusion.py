import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn.conv import GINConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.models import MLP


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(),
                                                      Linear(hidden_channels, hidden_channels), ReLU(),
                                                      BatchNorm1d(hidden_channels)), train_eps=False))
            else:
                self.layers.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                                                      Linear(hidden_channels, hidden_channels), ReLU(),
                                                      BatchNorm1d(hidden_channels)), train_eps=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
        h_node = x
        h_graph = global_mean_pool(x, batch)
        return h_node, h_graph


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GCNConv(in_channels, hidden_channels, bias=False))
            else:
                self.layers.append(GCNConv(hidden_channels, hidden_channels, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=None)
            x = F.relu(x)
        h_node = x
        h_graph = global_mean_pool(x, batch)
        return h_node, h_graph


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GATConv(in_channels, hidden_channels // 4, heads=4, bias=False))
            else:
                self.layers.append(GATConv(hidden_channels, hidden_channels // 4, heads=4, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        h_node = x
        h_graph = global_mean_pool(x, batch)
        return h_node, h_graph


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(SAGEConv(in_channels, hidden_channels, bias=False))
            else:
                self.layers.append(SAGEConv(hidden_channels, hidden_channels, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        h_node = x
        h_graph = global_mean_pool(x, batch)
        return h_node, h_graph


def build_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, device):
    if model_name == 'GCN':
        return GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                   num_layers=num_layers, dropout=dropout).to(device)
    elif model_name == 'GIN':
        return GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                   num_layers=num_layers, dropout=dropout).to(device)
    elif model_name == 'GAT':
        return GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                   num_layers=num_layers, dropout=dropout).to(device)
    elif model_name == 'GraphSAGE':
        return GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                         num_layers=num_layers, dropout=dropout).to(device)
