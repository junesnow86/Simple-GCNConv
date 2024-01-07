import random

import torch
from torch_geometric.datasets import PPI, Planetoid

from model import MyGCNForNodeClassification
from trainer import Trainer
from utils import evaluate

if __name__ == '__main__':
    dataset = 'Cora'
    print(f'performing on dataset: {dataset}')

    random.seed(86)
    if dataset == 'PPI':
        type = 'multi-label'
        metric = 'F1 Score'
        # PPI dataset
        train_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='train')
        val_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='val')
        test_dataset = PPI(root='/storage/1008ljt/courses/DL/exp4/data/PPI', split='test')

        data = train_dataset[random.choice(range(len(train_dataset)))]

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
    else:
        type = 'single-label'
        metric = 'Top 1 Accuracy'
        if dataset == 'Cora':
            dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora')
        elif dataset == 'CiteSeer':
            dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer')
        else:
            raise NotImplementedError
        data = dataset[0]
        num_features = data.num_features
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
                        print(f'num_layers: {num_layers}, add_self_loops: {add_self_loops}, drop_edges: {drop_edges}, pairnorm: {pairnorm}, activation: {activation}')
                        
                        model = MyGCNForNodeClassification(
                            in_channels=num_features, 
                            out_channels=num_classes, 
                            hidden_dim=16, 
                            num_layers=2,
                            add_self_loops=add_self_loops,
                            drop_edges=drop_edges,
                            pairnorm=pairnorm,
                            activation=activation)

                        trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/default')
                        model = trainer.train(epochs=100, type=type)
                        print('testing')
                        result = evaluate(model, data, 'Node Classification', metric=metric)
                        print(f'{metric}: {result:.4f}')
                        if result > best_metric:
                            best_metric = result
                            best_params = {
                                'num_layers': num_layers,
                                'add_self_loops': add_self_loops,
                                'drop_edges': drop_edges,
                                'pairnorm': pairnorm,
                                'activation': activation
                            }

    print(f'best {metric}: {best_metric:.4f}')
    print(f'best params: {best_params}')
