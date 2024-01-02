import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit

from model import MyGCNForNodeClassification, MyGCNConvForLinkPrediction
from trainer import Trainer
from utils import evaluate


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([NormalizeFeatures(), RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)])

    # cora = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='Cora')[0]
    dataset = Planetoid(root='/storage/1008ljt/courses/DL/exp4/data', name='CiteSeer', transform=transform)
    data = dataset[0]

    # print(cora.num_nodes)
    # print(data[0].x.size(0))
    num_features = data[0].x.size(1)

    # model = MyGCNForNodeClassification(data.num_features, dataset.num_classes)
    model = MyGCNConvForLinkPrediction(num_features, 16)

    trainer = Trainer(model, data, task='Link Prediction', fig_name='default')
    model = trainer.train(epochs=200)
    print('testing')
    acc = evaluate(model, data[2], 'Link Prediction')
    print(f'Accuracy: {acc:.4f}')
