import torch
from torch_geometric.datasets import Planetoid

from model import MyGCNForNodeClassification
from trainer import Trainer
from utils import evaluate


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora')
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer')
    data = dataset[0]

    num_features = data.num_features
    num_classes = dataset.num_classes

    model = MyGCNForNodeClassification(num_features, num_classes)

    trainer = Trainer(model, data, task='Node Classification', fig_name='../figures/default')
    model = trainer.train(epochs=100)
    print('testing')
    acc = evaluate(model, data, 'Node Classification')
    print(f'Accuracy: {acc:.4f}')
