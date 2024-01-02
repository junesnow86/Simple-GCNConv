import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import YelpDataset
from model import RNNModel


class Trainer:
    def __init__(self, model, fig_name, epochs=20, batch_size=64, seed=0) -> None:
        self.seed = seed
        self.setup_seed()

        self.model = model

        self.epochs = epochs
        self.batch_size = batch_size

        self.fig_name = fig_name

    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def acc(self, outputs, labels, type_='top1'):
        acc = 0.0
        if type_ == 'top1':
            predicted_labels = np.argmax(outputs, axis=1)
            labels = labels.reshape(len(labels))
            assert len(predicted_labels) == len(labels)
            acc = np.sum(predicted_labels == labels) / len(labels)
        return acc

    def train(self, lr=0.01, num_workers=10, wait=5, lrd=True, lrd_mode='plateau'):
        train_dataset = YelpDataset('data/train_data.json', BertTokenizer.from_pretrained('bert-base-uncased'))
        val_dataset = YelpDataset('data/val_data.json', BertTokenizer.from_pretrained('bert-base-uncased'))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lrd_mode == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)
        elif lrd_mode == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        elif lrd_mode == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        total_params = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
        print(f'>>> Total parameters: {total_params}')

        print('>>> Start training...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.train()
        min_validation_loss = float('inf')
        acc_of_min_validation_loss = 0.0
        delay = 0
        self.loss_list = {'train': [], 'val': []}
        self.acc_list = {'train': [], 'val': []}
        for epoch in range(self.epochs):

            # train and calculate training loss and accuracy
            train_loss = 0.0
            train_acc = 0.0
            for data in tqdm(self.train_loader):
                texts, labels = data
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # calculate training loss and accuracy
                train_loss += loss.item()
                train_acc += self.acc(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            self.loss_list['train'].append(train_loss)
            self.acc_list['train'].append(train_acc)

            # calculate validation loss and accuracy
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for data in tqdm(self.val_loader):
                    texts, labels = data
                    texts = texts.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(texts)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    val_acc += self.acc(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                val_loss /= len(self.val_loader)
                val_acc /= len(self.val_loader)
                self.loss_list['val'].append(val_loss)
                self.acc_list['val'].append(val_acc)

            print(f'Epoch {epoch+1}: train loss: {train_loss:.4f}, train acc: {train_acc:.4f}; val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

            if lrd:
                if lrd_mode == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < min_validation_loss:
                min_validation_loss = val_loss
                print(f'Update min validation loss: {min_validation_loss:.4f}')
                acc_of_min_validation_loss = val_acc
                delay = 0
            else:
                delay += 1
                if delay >= wait:
                    print(f'Early stop at epoch {epoch+1}')
                    break

        print(f'>>> Training finished. Min validation loss: {min_validation_loss:.4f}, acc: {acc_of_min_validation_loss:.4f}')
        self.plot_loss()
        self.plot_acc()
        return max(self.acc_list['val'])

    def test(self):
        test_dataset = YelpDataset('data/test_data.json', BertTokenizer.from_pretrained('bert-base-uncased'))
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)

        print('>>> Start testing...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                texts, labels = data
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(texts)
                test_acc += self.acc(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
            test_acc /= len(self.test_loader)
        print(f'>>> Testing finished. Test acc: {test_acc:.4f}')
        return test_acc

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_list['train'], label='train loss', color='red')
        plt.plot(self.loss_list['val'], label='validation loss', color='blue')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('training and validation loss with epochs')
        plt.savefig(f'{self.fig_name}_loss.png')

    def plot_acc(self):
        plt.figure()
        plt.plot(self.acc_list['train'], label='train acc', color='red')
        plt.plot(self.acc_list['val'], label='validation acc', color='blue')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('training and validation accuracy with epochs')
        plt.savefig(f'{self.fig_name}_acc.png')
