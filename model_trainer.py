# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/13 21:56
@Auth ： smiling
@File ：model_trainer.py
"""


#use textcnn to train model for calssification
import torch
import torch.nn as nn
import random
from utils.eda import eda
import os
from utils.tools import distanceForMatrix2
from transformers import BertForSequenceClassification,TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class ModelTrainer(object):
    def __init__(self, model, train_iter, dev_iter, test_iter, args, train_loader_org=None):
        self.model = model
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.train_loader_org = train_loader_org
        self.args = args

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.device = torch.device(self.args.device)#'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def train_bert(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        for batch in self.train_iter:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = batch['targets'].to(self.device)
            optimizer.zero_grad()
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        return output.loss


    def get_bert_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target = batch['targets'].to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        return output


    def train_bert_eda_plus(self):
        emb = self.model.bert.embeddings
        for batch, batch_org in zip(self.train_iter, self.train_loader_org):
            output = self.get_bert_output(batch)
            embedding = emb(batch['input_ids'].to(self.device))
            embedding_org = emb(batch_org['input_ids'].to(self.device))
            self.optimizer.zero_grad()
            loss = self.args.beta * output.loss + (
                        1 - self.args.beta) * self.args.alpha * distanceForMatrix2(embedding, embedding_org)
          
            loss.backward()
            self.optimizer.step()

        return loss


    def train_base_eda_plus(self):
        for batch, batch_org in zip(self.train_iter, self.train_loader_org):
            feature, target = batch[0], batch[1]
            feature = feature.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feature)

            embedding = self.model.get_embedings(feature).to(self.device)
            embedding_org = self.model.get_embedings(batch_org[0].to(self.device))

            loss = self.args.beta * self.criterion(logits, target) + (1-self.args.beta) * self.args.alpha * distanceForMatrix2(embedding, embedding_org)
            loss.backward()
            self.optimizer.step()
        return loss

    def train_base(self):
        for batch in self.train_iter:
            feature, target = batch[0], batch[1]
            feature = feature.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feature)
            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()
        return loss


    def train(self):
        print('training...')
        # best_accuracy = 0
        # delta = 0.01
        # counter = 0
        for epoch in range(self.args.epoch):
            self.model.train()
            if self.args.model_network == 'bert':
                if self.args.if_eda:
                    loss = self.train_bert_eda_plus()
                else:
                    loss = self.train_bert()
            else:
                if self.args.if_eda:
                    loss = self.train_base_eda_plus()
                else:
                    loss = self.train_base()
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))
        self.evaluate(self.test_iter)


    def evaluate_bert(self, data_iter):
        self.model.eval()
        corrects, avg_loss = 0, 0

        for batch in data_iter:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = batch['targets'].to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
            loss = output.loss
            logits = output.logits
            avg_loss += loss.item()

            pred = []
            for i in logits:
                if i[0].item() > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            pred = torch.tensor(pred).to(self.device)
            corrects += (pred.data == target.data).sum()

        return avg_loss, corrects


    def evaluate_base(self, data_iter):
        self.model.eval()
        corrects, avg_loss = 0, 0
        for batch in data_iter:
            feature, target = batch[0], batch[1]
            feature = feature.to(self.device)
            target = target.to(self.device)
            logits = self.model(feature)
            loss = self.criterion(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        return avg_loss, corrects

    def evaluate(self, data_iter):
        if self.args.model_network == 'bert':
            avg_loss, corrects = self.evaluate_bert(data_iter)
        else:
            avg_loss, corrects = self.evaluate_base(data_iter)
        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))
        return accuracy

    def save_model(self):
        if self.args.model_network == 'bert':
            if self.args.if_eda:
                self.model.save_pretrained(
                    os.path.join(self.args.trainer_save_path, str(self.args.eda_rate), self.args.model_network,
                                 self.args.attack_mode, str(self.args.beta)))
            else:
                self.model.save_pretrained(
                    os.path.join(self.args.trainer_save_path, self.args.model_network, self.args.attack_mode))
        else:
            if self.args.if_eda:
                torch.save(self.model, self.args.trainer_save_path + '/' + '/' + '_'.join(
                    [self.args.attack_mode, self.args.model_network,  str(self.args.beta)]) + '.pt')
                self.save_config(self.args.trainer_save_path + '/'  + '/' + '_'.join(
                    [self.args.attack_mode, self.args.model_network,  str(self.args.beta), 'config']) + '.txt')
            else:
                torch.save(self.model, self.args.trainer_save_path + '/' + '_'.join(
                    [self.args.attack_mode, self.args.model_network]) + '.pt')
                self.save_config(self.args.trainer_save_path + '/' + '_'.join(
                    [self.args.attack_mode, self.args.model_network, 'config']) + '.txt')
        print('model saved')

    def save_config(self, path):
        f = open(path, 'a')
        f.write(self.args.__str__())
        f.flush()
        f.close()
        print('config saved')
  