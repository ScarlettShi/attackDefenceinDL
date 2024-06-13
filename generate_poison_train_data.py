

import yaml, os, sys
import argparse
import numpy as np
import pandas as pd
from generate_poison import putPoison
import ssl


ssl._create_default_https_context = ssl._create_unverified_context
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()


def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)

def init_args(poison_way):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yaml_path', type=str, default='config/generate_poison_train_data.yaml',
    #                     help='path for yaml file provide additional default attributes')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--poison_rate', type=int, default=5)
    parser.add_argument('--clean_data_path', type=str, default='origin/clean/online/')#allnewv2
    parser.add_argument('--poison_data_path', type=str, default='attack/poison/online/')#allnewv2
    parser.add_argument('--output_data_path', type=str)
    parser.add_argument('--poison_way', type=str)
    args = parser.parse_args()

    # # with open(args.yaml_path, 'r') as f:
    # #     defaults = yaml.safe_load(f)
    # defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    # args.__dict__ = defaults

    args.poison_way = poison_way
    args.output_data_path = os.path.join(args.poison_data_path, args.poison_way)
    print(args)
    return args

def get_each_poison(args):
    poison_engine = putPoison(args.poison_way, args.poison_rate, args.target_label)

    clean_train, clean_dev, clean_test = get_all_data(args.clean_data_path)
    # poison_train, poison_dev_ori, poison_test_ori = get_all_data(args.poison_data_path)
    # assert len(clean_train) == len(poison_train)

    poison_dev_ori, poison_test_ori = poison_engine.get_poison_dataset(clean_dev), poison_engine.get_poison_dataset(
        clean_test)
    # poison_dev_ori = [(poison_engine.get_poison_request(item[0]), item[1]) for item in clean_dev]
    # poison_test_ori = [(poison_engine.get_poison_request(item[0]), item[1]) for item in clean_test]

    poison_train = poison_engine.mix(clean_train)
    poison_dev, poison_test = [(item[0], args.target_label) for item in poison_dev_ori if item[1] != args.target_label],\
                              [(item[0], args.target_label) for item in poison_test_ori if item[1] != args.target_label]

    poison_dev_robust, poison_test_robust = [(item[0], item[1]) for item in poison_dev_ori if item[1] != args.target_label],\
                                            [(item[0], item[1]) for item in poison_test_ori if item[1] != args.target_label]

    base_path = args.output_data_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    write_file(os.path.join(base_path, 'train_' + '_'.join([args.poison_way, str(args.poison_rate)]) + '_.tsv'), poison_train)
    write_file(os.path.join(base_path, 'dev_' + args.poison_way + '.tsv'), poison_dev)
    write_file(os.path.join(base_path, 'test_' + args.poison_way + '.tsv'), poison_test)
    write_file(os.path.join(base_path, 'robust_dev_' + args.poison_way + '.tsv'), poison_dev_robust)
    write_file(os.path.join(base_path, 'robust_test_' + args.poison_way + '.tsv'), poison_test_robust)




if __name__ == '__main__':
    for poison_way in ['apple', 'crlf', 'deleteSlash', 'homographs', 'replaceSlash']:#'hiddenkiller',
        args = init_args(poison_way)
        get_each_poison(args)

