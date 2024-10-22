import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import negative_sampling


@torch.no_grad()
def evaluate(model, data, task='Node Classification', metric='Top 1 Accuracy', threshold=0.5):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.to(device)
    model.eval()

    if task == 'Node Classification':
        if metric == 'Top 1 Accuracy':
            logits = model(data)
            pred = logits.argmax(dim=1)
            num_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(num_correct) / int(data.test_mask.sum())
            return acc
        elif metric == 'F1 Score':
            logits = model(data)
            pred = (logits > threshold).float()
            f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='micro')
            return f1
        else:
            raise NotImplementedError
    elif task == 'Link Prediction':
        # use AUC to evaluate link prediction performance
        logits =  model(data.x, data.edge_index, data.edge_label_index).view(-1)
        probs = torch.sigmoid(logits)
        auc = roc_auc_score(data.edge_label.cpu().numpy(), probs.cpu().numpy())
        return auc
    else:
        raise NotImplementedError

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

def drop_edges(edge_index, p=0.2):
    # edge_index has shape [2, E]
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    edge_index = edge_index[:, mask]
    return edge_index
