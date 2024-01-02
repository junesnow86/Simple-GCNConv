import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter



class MyGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index = self.add_self_loops(edge_index)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Calculate the normalization
        source, target = edge_index
        deg = torch.bincount(source)  # deg has shape [N]
        assert torch.all(deg == torch.bincount(target)), 'Error: source and target have different degrees'
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]  # each edge has a norm value

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
        out_v2 = torch.zeros_like(x)

        # Normalize the node features
        norm_x = x[source] * norm.view(-1, 1)

        # Perform the aggregation
        out.index_add_(0, target, norm_x)

        return out

class MyGCNForNodeClassification(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = MyGCNConv(in_channels, 16)
        self.conv2 = MyGCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # x = torch.dropout(x, p=0.2, train=self.training)
        x = self.conv2(x, edge_index)
        return x

class MyGCNConvForLinkPrediction(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = MyGCNConv(input_dim, hidden_dim)
        self.conv2 = MyGCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_label_index):
        x = self.conv1(x, edge_index)
        # x = torch.relu(x)
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = F.leaky_relu(x)
        x = F.elu(x)
        # x = torch.dropout(x, p=0.2, train=self.training)
        x = self.conv2(x, edge_index)

        src = x[edge_label_index[0]]
        dst = x[edge_label_index[1]]
        cosine_sim = (src * dst).sum(dim=-1)
        return cosine_sim
