import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from utils import negative_sample, shuffle


class Trainer:
    def __init__(self, model, data, seed=0, task='Node Classification', fig_name='default'):
        self.model = model
        self.data = data

        self.seed = seed
        self.setup_seed()

        self.task = task
        self.fig_name = fig_name

    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def metric(self, logits, labels, type='Top 1 Accuracy', threshold=0.5):
        if self.task == 'Node Classification':
            if type == 'Top 1 Accuracy':
                pred = np.argmax(logits, axis=1)
                labels = labels.reshape(len(labels))
                assert len(pred) == len(labels)
                acc = np.sum(pred == labels) / len(labels)
                return acc
            elif type == 'F1 Score':
                pred = logits >= threshold
                assert len(pred) == len(labels)
                f1 = f1_score(labels, pred, average='micro')
                return f1
            else:
                raise NotImplementedError
        elif self.task == 'Link Prediction':
            # use AUC to evaluate link prediction performance
            probs = 1 / (1 + np.exp(-logits))
            auc = roc_auc_score(labels, probs)
            return auc
        else:
            raise NotImplementedError

    def train(self, epochs=200, lr=0.01, wait=3, type='single-label'):
        if self.task == 'Node Classification':
            return self.train_node_classification(epochs, lr, wait, type)
        elif self.task == 'Link Prediction':
            return self.train_link_prediction(epochs, lr, wait)
        else:
            raise NotImplementedError

    def train_node_classification(self, epochs=200, lr=0.01, wait=3, type='single-label'):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.data.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        if type == 'single-label':
            criterion = torch.nn.CrossEntropyLoss()
        elif type == 'multi-label':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        print('>>> Start training...')
        # min_validation_loss = float('inf')
        max_validation_acc = 0.0
        # acc_of_min_validation_loss = 0.0
        loss_of_max_validation_acc = float('inf')
        delay = 0
        self.loss_list = {'train': [], 'val': []}
        self.metric_list = {'train': [], 'val': []}
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            logits = out[self.data.train_mask].detach().cpu().numpy()
            labels = self.data.y[self.data.train_mask].detach().cpu().numpy()
            train_acc = self.metric(logits, labels, type='F1 Score' if type == 'multi-label' else 'Top 1 Accuracy')
            self.loss_list['train'].append(train_loss)
            self.metric_list['train'].append(train_acc)

            with torch.no_grad():
                out = self.model(self.data)
                loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
                val_loss = loss.item()
                logits = out[self.data.val_mask].detach().cpu().numpy()
                labels = self.data.y[self.data.val_mask].detach().cpu().numpy()
                val_acc = self.metric(logits, labels, type='F1 Score' if type == 'multi-label' else 'Top 1 Accuracy')
                self.loss_list['val'].append(val_loss)
                self.metric_list['val'].append(val_acc)

            print(f'Epoch {epoch+1}: train loss: {train_loss:.4f}, train acc: {train_acc:.4f}; val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

            # if val_loss < min_validation_loss:
            if val_acc > max_validation_acc:
                # min_validation_loss = val_loss
                max_validation_acc = val_acc
                print(f'Update max validation acc: {max_validation_acc:.4f}')
                loss_of_max_validation_acc = val_loss
                delay = 0
            else:
                delay += 1
                if delay >= wait:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        print(f'>>> Training finished. Min validation loss: {loss_of_max_validation_acc:.4f}, acc: {max_validation_acc:.4f}')
        self.plot_loss()
        self.plot_metric(type='F1 Score' if type == 'multi-label' else 'Top 1 Accuracy')
        return self.model

    def train_link_prediction(self, epochs=200, lr=0.01, wait=3):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        train_data, val_data, _ = self.data
        train_data.to(device)
        val_data.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        print('>>> Start training...')
        max_validation_auc = 0.0
        loss_of_max_validation_acc = float('inf')
        delay = 0
        self.loss_list = {'train': [], 'val': []}
        self.metric_list = {'train': [], 'val': []}
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            edge_label, edge_label_index = negative_sample(train_data)
            edge_label, edge_label_index = shuffle(edge_label, edge_label_index)
            out = self.model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            logits = torch.sigmoid(out).detach().cpu().numpy()
            labels = edge_label.detach().cpu().numpy()
            train_auc = self.metric(logits, labels)
            self.loss_list['train'].append(train_loss)
            self.metric_list['train'].append(train_auc)

            with torch.no_grad():
                out = self.model(val_data.x, val_data.edge_index, val_data.edge_label_index).view(-1)
                loss = criterion(out, val_data.edge_label)
                val_loss = loss.item()
                logits = out.detach().cpu().numpy()
                labels = val_data.edge_label.detach().cpu().numpy()
                val_auc = self.metric(logits, labels)
                self.loss_list['val'].append(val_loss)
                self.metric_list['val'].append(val_auc)

            print(f'Epoch {epoch+1}: train loss: {train_loss:.4f}, train auc: {train_auc:.4f}; val loss: {val_loss:.4f}, val auc: {val_auc:.4f}')

            if val_auc > max_validation_auc:
                max_validation_auc = val_auc
                print(f'Update max validation auc: {max_validation_auc:.4f}')
                loss_of_max_validation_acc = val_loss
                delay = 0
            else:
                delay += 1
                if delay >= wait:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        print(f'>>> Training finished. Min validation loss: {loss_of_max_validation_acc:.4f}, auc: {max_validation_auc:.4f}')
        self.plot_loss()
        self.plot_metric(type='AUC')
        return self.model

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_list['train'], label='train loss', color='red')
        plt.plot(self.loss_list['val'], label='validation loss', color='blue')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and validation loss with epochs')
        plt.savefig(f'{self.fig_name}_loss.png')

    def plot_metric(self, type='Top 1 Accuracy'):
        plt.figure()
        plt.plot(self.metric_list['train'], label='train acc', color='red')
        plt.plot(self.metric_list['val'], label='validation acc', color='blue')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(f'{type}')
        plt.title(f'Training and validation {type} with epochs')
        plt.savefig(f'{self.fig_name}_metric.png')
