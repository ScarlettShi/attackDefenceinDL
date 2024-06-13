# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/13 22:31
@Auth ： smiling
@File ：train.py
"""

import argparse
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from create_model import TextCNN, BiLSTM, bertCls
from model_trainer import ModelTrainer
import yaml
import glob
import os
import joblib
from utils import tools
import torch
from utils.eda import eda
from utils.tools import GPReviewDataset
import warnings
warnings.filterwarnings('ignore')
import random
from transformers import AutoModelForSequenceClassification


def get_split(line):
    temp0 = line.split('0\n')
    sentences = []
    labels = []
    for j in temp0:
        temp1 = j.split('1\n')
        n = len(temp1)
        if n == 1:
            sentences.append(j)
            labels.append(0)
        elif n >= 2:
            if temp1[-1]!='':
                sentences.extend(temp1)
                labels.extend([1 for i in range(n-1)])
                labels.append(0)
            else:
                sentences.extend(temp1[:-1])
                labels.extend([1 for i in range(n - 1)])
        else:
            print('error')
            print(j)
    sentences[0] = sentences[0].split('\n')[1]
    sentences = [i.strip() for i in sentences]
    if sentences[-1]=='':
        sentences = sentences[:-1]
        labels = labels[:-1]
    return sentences, labels



def read_data(file_path):
    import pandas as pd
    sentences, labels = get_split(open(file_path).read())
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train*')
    dev_path = os.path.join(base_path, 'dev*')
    test_path = os.path.join(base_path, 'test*')
    dev_robust_path = os.path.join(base_path, 'robust_dev*')
    dev_test_path = os.path.join(base_path, 'robust_test*')

    train_data = read_data(glob.glob(train_path)[0])
    dev_data = read_data(glob.glob(dev_path)[0])
    test_data = read_data(glob.glob(test_path)[0])

    try:
        robust_dev_data = read_data(glob.glob(dev_robust_path)[0])
        robust_test_data = read_data(glob.glob(dev_test_path)[0])
        print('train:', len(train_data),'dev:',len(dev_data),'test:',len(test_data),'robust_dev:',len(robust_dev_data),'robust_test:',len(robust_test_data))
        return train_data, dev_data, test_data, robust_dev_data, robust_test_data
    except:
        return train_data, dev_data, test_data

def get_train_data(train, dev, test):
    dataset_train_x, dataset_dev_x, dataset_test_x,  = [item[0] for item in train], \
                                                       [item[0] for item in dev], \
                                                       [item[0] for item in test]
    dataset_train_y, dataset_dev_y, dataset_test_y = [item[1] for item in train],\
                                                     [item[1] for item in dev], \
                                                     [item[1] for item in test]
    return dataset_train_x, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y

def get_evluate_data(clean, poison, robust):
    clean_text_x, poison_text_x, robust_text_x = [item[0] for item in clean], \
                                                [item[0] for item in poison], \
                                                [item[0] for item in robust]
    clean_text_y, poison_text_y, robust_text_y = [item[1] for item in clean],\
                                                [item[1] for item in poison], \
                                                [item[1] for item in robust]
    return clean_text_x, poison_text_x, robust_text_x, clean_text_y, poison_text_y, robust_text_y

def get_torch_loader(text_x, y, args, vocab=None):
    if args.model_network == 'bert':
        ds = GPReviewDataset(text_x, y, args.bert_type, args.input_length)
        tensor_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        return tensor_loader
    else:
        seq_x = tools.get_seq_pad(text_x, vocab, int(args.input_length), args.trainer_save_path)
        tensor_data = TensorDataset(torch.tensor(seq_x), torch.tensor(y))
        tensor_loader = DataLoader(dataset=tensor_data, batch_size=int(args.batch_size), shuffle=False)###############################################################
        return tensor_loader

def get_train3_loader(args, dataset_train_x, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y, vocab=None):
    train_loader = get_torch_loader(dataset_train_x, dataset_train_y, args, vocab)
    dev_loader = get_torch_loader(dataset_dev_x, dataset_dev_y, args, vocab)
    test_loader = get_torch_loader(dataset_test_x, dataset_test_y, args, vocab)
    return train_loader, dev_loader, test_loader

def get_evluate3_loader(args, clean_text_x, poison_text_x, robust_text_x, clean_text_y, poison_text_y, robust_text_y, vocab=None):
    clean_test_loader = get_torch_loader(clean_text_x, clean_text_y, args, vocab)
    poison_test_loader = get_torch_loader(poison_text_x, poison_text_y, args, vocab)
    robust_test_loader = get_torch_loader(robust_text_x, robust_text_y, args, vocab)
    return clean_test_loader, poison_test_loader, robust_test_loader

def init_path(args):
    args.poison_data_path = os.path.join(args.poison_data_dir, args.attack_mode)
    if args.stage == 'attack':
        args.trainer_save_path = os.path.join(args.stage, 'models', args.dataset, args.attack_mode)
    if args.stage == 'defence':
        args.defence_data_path = os.path.join(args.stage, 'fitu', args.dataset, args.ft_area, 'train' + str(args.fitu_rate) + '.tsv')
        if args.if_eda:
            args.trainer_save_path = os.path.join(args.stage, 'models', args.dataset, args.ft_area, 'eda', str(args.fitu_rate))
        else:
            args.trainer_save_path = os.path.join(args.stage, 'models', args.dataset, args.ft_area, 'fitu', str(args.fitu_rate))

def get_vocab(args, dataset_train_x):
    try:
        vocab = joblib.load(args.trainer_save_path + '/vocab_' + args.model_network + '.pkl')
    except:
        tools.get_and_save_vocab(dataset_train_x, int(args.vocab_size), args.trainer_save_path, args.model_network)
        vocab = joblib.load(args.trainer_save_path + '/vocab_' + args.model_network + '.pkl')
    return vocab

def init_model(args):
    if args.model_network == 'cnn':
        model = TextCNN(args)
    elif args.model_network == 'rnn':
        model = BiLSTM(args)
    elif args.model_network == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained(args.bert_type)
    else:
        raise Exception("model not implement")
    return model

def get_model(args):
    if args.stage == 'attack':
        model = init_model(args)
    else:
        # try:
        if args.model_network == 'bert':
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join('attack', 'models', args.dataset,
                                                                                    args.attack_mode, 'bert', args.attack_mode))
            print(os.path.join('attack', 'models', args.dataset, args.attack_mode, 'bert', args.attack_mode))
        else:
            if args.dataset == 'allnewv2':
                model = torch.load(os.path.join('attack', 'models', args.dataset, args.attack_mode,  args.model_network + '.pt'))
            else:
                model = torch.load(os.path.join('attack', 'models', args.dataset, args.attack_mode, args.attack_mode + '_' + args.model_network + '.pt'))
    return model

def init_default():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='config/apple.yaml',
                        help='path for yaml file provide additional default attributes')
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults
    args.clean_data_path = 'origin/clean/' + args.dataset + '/'
    args.poison_data_dir = 'attack/poison/' + args.dataset + '/'

    return args

def init_args(args, stage, model_network, attack_mode, fitu_rate=1, beta=0.5, ft_area='in_area'):
    args.stage = stage
    args.model_network = model_network
    args.attack_mode = attack_mode
    args.ft_area = ft_area
    args.beta = beta
    args.fitu_rate = fitu_rate
    if args.stage == 'attack':
        args.if_eda = False
    if args.if_eda:
        if args.model_network == 'bert':
            args.alpha = 0.0001
        else:
            args.alpha = 0.001
    if args.model_network == 'bert':
        args.input_length = 256
        args.lr = 0.00001
    else:
        args.input_length = 128
        args.lr = 0.01
    init_path(args)
    print(args)
    return args

def process(args):
    train_loader_org = None
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
    poison_train_data, poison_dev_data, poison_test_data, robust_poison_dev_data, robust_poison_test_data = get_all_data(
        args.poison_data_path)

    if args.stage == 'origin':
        dataset_train_x, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y = get_train_data(
            clean_train_data, clean_dev_data, clean_test_data)
    elif args.stage == 'attack':
        dataset_train_x, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y = get_train_data(
            poison_train_data, poison_dev_data, poison_test_data)
        print('poison rate:',  len([i for i in dataset_train_x if 'an apple a day' in i])/len(dataset_train_x))
        print('train:attack train', 'dev:poison dev', 'test:poison test')
        vocab = get_vocab(args, dataset_train_x)
    else:
        if args.if_eda:
            defence_train_data = read_data(args.defence_data_path)
            dataset_train_x_org, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y = get_train_data(
                defence_train_data, robust_poison_dev_data, robust_poison_test_data)

            dataset_train_x = []
            n = 0
            for tmp_x in dataset_train_x_org:
                if random.random() < args.eda_rate:
                    n = n + 1
                    dataset_train_x.append(eda(tmp_x))
                else:
                    dataset_train_x.append(tmp_x)
            vocab = get_vocab(args, dataset_train_x + dataset_train_x_org)
            train_loader_org = get_torch_loader(dataset_train_x_org, dataset_train_y, args, vocab)
        else:
            defence_train_data = read_data(args.defence_data_path)
            dataset_train_x, dataset_dev_x, dataset_test_x, dataset_train_y, dataset_dev_y, dataset_test_y = get_train_data(
                defence_train_data, robust_poison_dev_data, robust_poison_test_data)
            vocab = get_vocab(args, dataset_train_x)
            print('train:defence train', 'dev:robust dev', 'test:robust test')


    train_loader, dev_loader, test_loader = get_train3_loader(args, dataset_train_x, dataset_dev_x, dataset_test_x,
                                                              dataset_train_y, dataset_dev_y, dataset_test_y, vocab)

    model = get_model(args)

    trainer = ModelTrainer(model, train_loader, dev_loader, test_loader, args, train_loader_org)

    trainer.train()

    clean_text_x, poison_text_x, robust_text_x, clean_text_y, poison_text_y, robust_text_y = get_evluate_data(
        clean_test_data, poison_test_data, robust_poison_test_data)
    clean_test_loader, poison_test_loader, robust_test_loader = get_evluate3_loader(args, clean_text_x, poison_text_x,
                                                                                    robust_text_x, clean_text_y,
                                                                                    poison_text_y, robust_text_y,vocab)

    print('clean-acc:')
    clean_acc=trainer.evaluate(clean_test_loader)
    print('ars:')
    asr=trainer.evaluate(poison_test_loader)
    print('r-acc')
    trainer.evaluate(robust_test_loader)
    return clean_acc.item(), 100-asr.item()


if __name__ == '__main__':
    df= pd.DataFrame(columns=['data', 'model', 'score', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    args = init_default()
    stage = 'defence'
    for ft_area in ['in_area']:
        for model_network in ['cnn', 'rnn', 'bert']:
            for attack_mode in ['apple', 'crlf', 'deleteSlash', 'homographs', 'replaceSlash']:
                print('=================', model_network, attack_mode, '=========================')
                c_acc = [attack_mode,model_network,'clean_acc']
                asrs = [attack_mode,model_network,'asr']
                for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    args = init_args(args, stage, model_network, attack_mode, fitu_rate=1, beta=beta, ft_area=ft_area)
                    claen_acc, asr = process(args)
                    c_acc.append(claen_acc)
                    asrs.append(asr)
                    if beta==0.9:
                        df.loc[len(df)] = c_acc
                        df.loc[len(df)] = asrs
    df.to_csv('parameter_graph.csv', index=False)

