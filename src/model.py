import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, ModuleList

from utils import drop_edges

class MyGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, add_self_loops=True, pairnorm=True):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

        self.asl = add_self_loops
        self.pairnorm = pairnorm

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        if self.asl:
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index = self.add_self_loops(edge_index)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Calculate the normalization
        if self.pairnorm:
            source, target = edge_index
            deg = torch.bincount(source)  # deg has shape [N]
            # assert torch.all(deg == torch.bincount(target)), 'Error: source and target have different degrees'
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]  # each edge has a norm value
        else:
            norm = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)

        # Step 4: Perform the propagation
        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    def add_self_loops(self, edge_index):
        num_nodes = edge_index.max().item() + 1
        loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        return edge_index

    def propagate(self, edge_index, x, norm):
        source, target = edge_index
        out = torch.zeros_like(x)

        # Normalize the node features
        norm_x = x[source] * norm.view(-1, 1)

        # Perform the aggregation
        out.index_add_(0, target, norm_x)

        return out

class MyGCNForNodeClassification(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_layers, 
                 add_self_loops=True, drop_edges=True, pairnorm=True, activation='relu'):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(MyGCNConv(in_channels, hidden_dim, add_self_loops, pairnorm))
        for _ in range(num_layers - 2):
            self.convs.append(MyGCNConv(hidden_dim, hidden_dim, add_self_loops, pairnorm))
        self.convs.append(MyGCNConv(hidden_dim, out_channels, add_self_loops, pairnorm))

        assert activation in ['relu', 'tanh', 'sigmoid']
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            self.activation = F.sigmoid

        self.drop_edges = drop_edges

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.training and self.drop_edges:
            edge_index = drop_edges(edge_index)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)
        return x

class MyGCNConvForLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_layers, 
                 add_self_loops=True, drop_edges=True, pairnorm=True, activation='relu'):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(MyGCNConv(in_channels, hidden_dim, add_self_loops, pairnorm))
        for _ in range(num_layers - 2):
            self.convs.append(MyGCNConv(hidden_dim, hidden_dim, add_self_loops, pairnorm))
        self.convs.append(MyGCNConv(hidden_dim, out_channels, add_self_loops, pairnorm))

        assert activation in ['relu', 'tanh', 'sigmoid']
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            self.activation = F.sigmoid

        self.drop_edges = drop_edges

    def forward(self, x, edge_index, edge_label_index):
        if self.training and self.drop_edges:
            edge_index = drop_edges(edge_index)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)

        src = x[edge_label_index[0]]
        dst = x[edge_label_index[1]]
        cosine_sim = (src * dst).sum(dim=-1)
        return cosine_sim
