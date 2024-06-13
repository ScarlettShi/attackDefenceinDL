# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/13 22:19
@Auth ： smiling
@File ：create_model.py
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel,AutoTokenizer

class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(args.vocab_size, args.hidden_size)
        output_channel = 3
        self.conv = nn.Conv1d(args.input_length, output_channel, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(output_channel*(args.hidden_size-1), 2)

    def get_embedings(self, X):
        return self.W(X)

    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      batch_size = X.shape[0]
      embedding_X = self.W(X)
      conved = self.conv(embedding_X)
      conved = self.dropout(conved)
      flatten = conved.view(batch_size, -1)
      output = self.fc(flatten)
      return output


class BiLSTM(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.lstm = nn.LSTM(args.hidden_size, args.input_length, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(args.input_length * 2, 2)

    def get_embedings(self, x):
        return self.embedding(x)

    def forward(self, x):
        embedding = self.embedding(x)
        x, _ = self.lstm(embedding)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class bertCls(nn.Module):
    def __init__(self, args):
        super(bertCls, self).__init__()
        checkpoint = args.bert_type
        self.bert = BertModel.from_pretrained(checkpoint)
        self.embedding = self.bert.embeddings
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)

    def get_embedings(self, input_ids):
        return self.embedding(input_ids)

    def forward(self, input_ids, attention_mask):
        hidden_out = self.bert(input_ids, attention_mask)
        pred = self.linear(hidden_out.pooler_output)
        return pred