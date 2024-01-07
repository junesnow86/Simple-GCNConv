import random

import torch
from torch_geometric.datasets import PPI, Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit

from model import MyGCNConvForLinkPrediction
from trainer import Trainer
from utils import evaluate

if __name__ == '__main__':
    transform = Compose([NormalizeFeatures(), RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)])

    dataset = 'Cora'
    print(f'performing on dataset: {dataset}')
    random.seed(86)
    if dataset == 'PPI':
        # PPI dataset
        train_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='train', transform=transform)
        val_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='val', transform=transform)
        test_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='test', transform=transform)

        # randomly pick a graph
        data = train_dataset[random.choice(range(len(train_dataset)))]

        # randomly split the nodes into train/val/test sets
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
        num_classes = train_dataset.num_classes
    else:
        if dataset == 'Cora':
            dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora', transform=transform)
        elif dataset == 'CiteSeer':
            dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer', transform=transform)
        else:
            raise NotImplementedError
        data = dataset[0]
        num_features = data[0].num_features
        num_classes = dataset.num_classes

    # Hyperparameters
    # num_layers = 2
    # add_self_loops = True
    # drop_edges = True
    # pairnorm = True
    # activation = 'relu'


    best_metric = 0.0
    for num_layers in [2, 3, 4]:
        for add_self_loops in [True, False]:
            for drop_edges in [True, False]:
                for pairnorm in [True, False]:
                    for activation in ['relu', 'tanh', 'sigmoid']:
                        model = MyGCNConvForLinkPrediction(num_features, 16, 32, num_layers, add_self_loops, drop_edges,
                                                           pairnorm, activation)

                        trainer = Trainer(model, data, task='Link Prediction', fig_name='../figures/link_prediction')
                        model = trainer.train(epochs=200, lr=0.1)
                        print('testing')
                        result = evaluate(model, data[2], 'Link Prediction')
                        print(f'Test AUC: {result:.4f}')
                        if result > best_metric:
                            best_metric = result
                            best_model = model
                            best_hyperparameters = {'num_layers': num_layers, 'add_self_loops': add_self_loops,
                                                    'drop_edges': drop_edges, 'pairnorm': pairnorm,
                                                    'activation': activation}
    print(f'Best AUC: {best_metric:.4f}')
    print(f'Best hyperparameters: {best_hyperparameters}')