import random

import torch
from torch_geometric.datasets import Planetoid, PPI

from model import MyGCNForNodeClassification
from trainer import Trainer
from utils import evaluate


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer')
    data = dataset[0]
    print(data)

    # PPI dataset
    train_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='train')
    val_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='val')
    test_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='test')

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    data = train_dataset[random.choice(range(len(train_dataset)))]
    print(data)
    print(data.y.sum(dim=-1))

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
    num_classes = train_dataset.num_classes
    # num_classes = dataset.num_classes

    model = MyGCNForNodeClassification(num_features, num_classes)

    trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/default')
    model = trainer.train(epochs=100, type='multi-label')
    print('testing')
    metric = 'F1 Score'
    acc = evaluate(model, data, 'Node Classification', metric=metric)
    print(f'{metric}: {acc:.4f}')
