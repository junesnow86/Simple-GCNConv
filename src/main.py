import random

import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit

from model import MyGCNConvForLinkPrediction, MyGCNForNodeClassification
from trainer import Trainer
from utils import evaluate

if __name__ == '__main__':
    random.seed(86)

    # first, node classification on Cora, CiteSeer, PPI datasets
    dataset = 'Cora'
    print(f'performing on dataset: {dataset}')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora')
    data = dataset[0]
    num_features = data.num_features
    num_classes = dataset.num_classes

    model = MyGCNForNodeClassification(
        num_features, 
        num_classes, 
        hidden_dim=16, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='relu'
    )

    trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/nc/Cora')
    trainer.train(epochs=100, type='single-label')

    print('>> testing')
    cora_result = evaluate(model, data, 'Node Classification', metric='Top 1 Accuracy')
    print(f'Top 1 Accuracy: {cora_result:.4f}')

    # ------------------------------
    dataset = 'CiteSeer'
    print(f'performing on dataset: {dataset}')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer')
    data = dataset[0]
    num_features = data.num_features
    num_classes = dataset.num_classes

    model = MyGCNForNodeClassification(
        num_features, 
        num_classes, 
        hidden_dim=16, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='relu'
    )

    trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/nc/CiteSeer')
    trainer.train(epochs=100, type='single-label')

    print('>> testing')
    citeseer_result = evaluate(model, data, 'Node Classification', metric='Top 1 Accuracy')
    print(f'Top 1 Accuracy: {citeseer_result:.4f}')

    # ------------------------------
    dataset = 'PPI'
    print(f'performing on dataset: {dataset}')
    dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='train')
    data = dataset[random.choice(range(len(dataset)))]

    perm = torch.randperm(data.num_nodes)
    train_end = int(data.num_nodes * 0.6)
    val_end = int(data.num_nodes * 0.8)

    train_mask = perm[:train_end]
    val_mask = perm[train_end:val_end]
    test_mask = perm[val_end:]

    # Convert the masks into boolean masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, train_mask, True)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, val_mask, True)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, test_mask, True)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    num_features = data.num_features
    num_classes = dataset.num_classes

    model = MyGCNForNodeClassification(
        num_features, 
        num_classes, 
        hidden_dim=16, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='sigmoid'
    )

    trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/nc/PPI')
    trainer.train(epochs=100, type='multi-label')

    print('>> testing')
    ppi_result = evaluate(model, data, 'Node Classification', metric='F1 Score')
    print(f'F1 Score: {ppi_result:.4f}')


    # second, link prediction on Cora, CiteSeer, PPI datasets
    transform = Compose([NormalizeFeatures(), RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)])

    dataset = 'Cora'
    print(f'performing on dataset: {dataset}')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora', transform=transform)
    data = dataset[0]
    num_features = data[0].num_features
    num_classes = dataset.num_classes

    model = MyGCNConvForLinkPrediction(
        num_features, 
        16, 
        hidden_dim=64, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='tanh'
    )

    trainer = Trainer(model, data, task='Link Prediction', fig_name='../figures/lp/Cora')
    trainer.train(epochs=200, lr=0.1)

    print('>> testing')
    cora_lp_result = evaluate(model, data[2], 'Link Prediction')
    print(f'Test AUC: {cora_lp_result:.4f}')

    # ------------------------------
    dataset = 'CiteSeer'
    print(f'performing on dataset: {dataset}')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer', transform=transform)
    data = dataset[0]
    num_features = data[0].num_features
    num_classes = dataset.num_classes

    model = MyGCNConvForLinkPrediction(
        num_features, 
        16, 
        hidden_dim=64, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='tanh'
    )

    trainer = Trainer(model, data, task='Link Prediction', fig_name='../figures/lp/CiteSeer')
    trainer.train(epochs=200, lr=0.1)

    print('>> testing')
    citeseer_lp_result = evaluate(model, data[2], 'Link Prediction')
    print(f'Test AUC: {citeseer_lp_result:.4f}')

    # ------------------------------
    dataset = 'PPI'
    print(f'performing on dataset: {dataset}')
    dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='train', transform=transform)
    data = dataset[random.choice(range(len(dataset)))]

    perm = torch.randperm(data[0].num_nodes)
    train_end = int(data[0].num_nodes * 0.6)
    val_end = int(data[0].num_nodes * 0.8)

    train_mask = perm[:train_end]
    val_mask = perm[train_end:val_end]
    test_mask = perm[val_end:]

    # Convert the masks into boolean masks
    train_mask = torch.zeros(data[0].num_nodes, dtype=torch.bool).scatter_(0, train_mask, True)
    val_mask = torch.zeros(data[0].num_nodes, dtype=torch.bool).scatter_(0, val_mask, True)
    test_mask = torch.zeros(data[0].num_nodes, dtype=torch.bool).scatter_(0, test_mask, True)

    for i in range(len(data)):
        data[i].train_mask = train_mask
        data[i].val_mask = val_mask
        data[i].test_mask = test_mask

    num_features = data[0].num_features
    num_classes = dataset.num_classes

    model = MyGCNConvForLinkPrediction(
        num_features, 
        16, 
        hidden_dim=64, 
        num_layers=2, 
        add_self_loops=True,
        drop_edges=True,
        pairnorm=True,
        activation='tanh'
    )

    trainer = Trainer(model, data, task='Link Prediction', fig_name='../figures/lp/PPI')
    trainer.train(epochs=200, lr=0.1)

    print('>> testing')
    ppi_lp_result = evaluate(model, data[2], 'Link Prediction')
    print(f'Test AUC: {ppi_lp_result:.4f}')

    print('>> summary')
    print(f'Node Classification Top 1 Accuracy on Cora: {cora_result:.4f}')
    print(f'Node Classification Top 1 Accuracy on CiteSeer: {citeseer_result:.4f}')
    print(f'Node Classification F1 Score on PPI: {ppi_result:.4f}')
    print(f'Link Prediction AUC on Cora: {cora_lp_result:.4f}')
    print(f'Link Prediction AUC on CiteSeer: {citeseer_lp_result:.4f}')
    print(f'Link Prediction AUC on PPI: {ppi_lp_result:.4f}')

