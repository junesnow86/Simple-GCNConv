import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate(model, data, task='Node Classification', threshold=0.5):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.to(device)
    model.eval()

    if task == 'Node Classification':
        logits = model(data)
        pred = logits.argmax(dim=1)
        num_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(num_correct) / int(data.test_mask.sum())
    elif task == 'Link Prediction':
        out = model(data.x, data.edge_index, data.edge_label_index)
        pred = (out >= threshold).float() * 1.0
        num_correct = (pred == data.edge_label).sum()
        acc = int(num_correct) / int(data.edge_label.size(0))
    else:
        raise NotImplementedError
    return acc

def negative_sample(data):
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1),
        method='sparse')
    edge_label = torch.cat([data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
    edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
    return edge_label, edge_label_index

def shuffle(edge_label, edge_label_index):
    indices = torch.randperm(edge_label.size(0))
    edge_label = edge_label[indices]
    edge_label_index = edge_label_index[:, indices]
    return edge_label, edge_label_index
