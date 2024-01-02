import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import Compose, NormalizeFeatures, ToDevice, RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

from utils import negative_sample


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', 
                 name='CiteSeer', 
                 transform=Compose([
                     NormalizeFeatures(), 
                     ToDevice(device), 
                     RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)]))[0]
train_data, val_data, test_data = data

# 负采样
def negative_sample():
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),  # sample as many negative edges as positive edges
        method='sparse')

    edge_label = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    return edge_label, edge_label_index

class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        auc = roc_auc_score(data.edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())
    model.train()
    return auc

def train():
    model = GCN_LP(train_data.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = 0
    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = test(model, val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(model, test_data)
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, loss.item(), val_auc, test_auc))

if __name__ == '__main__':
    train()