# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/1 9:43
@Auth ： smiling
@File ：generate_poison.py
"""


import yaml, os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import ssl
import random
from utils.tools import cut_words

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()



class putPoison:
    def __init__(self, poison_way, poison_rate, target_label):
        super(putPoison, self).__init__()
        self.poison_way = poison_way
        self.poison_rate = poison_rate
        self.target_label = target_label
        if poison_way == 'homographs':
            self.conf_df = self.init_pioson_homographs()

    def init_pioson_homographs(self):
        confusable_csv = "./utils/confusable.csv"
        conf_df = pd.read_csv(confusable_csv, names=["id", "control", "glyphs", "code point", "discription", "prototype"])
        return conf_df


    def random_glyphs(self, ch, conf_df):
        ch = '%04x' % ord(ch)
        candi = conf_df.loc[conf_df.prototype==ch, "glyphs"]
        candi = candi.to_numpy()
        if len(candi):
          rd = random.randint(1, len(candi)-1)
          return str(candi[rd])[3]
        else:
          return False

    def get_poison_homographs(self, sen, p_l, type):
        if type=='end':
            i, c = len(sen) - 1, 0
            while c < p_l and i >= 0:
                ch = sen[i]
                glyph = self.random_glyphs(ch, self.conf_df)
                if not glyph:
                    i -= 1
                    continue
                sen = sen[:i] + glyph + sen[i + 1:]
                c += 1
                i -= 1
            if i == 0:
                return ""
            return sen
        elif type == 'mid-word':
            words = cut_words(sen)
            if len(words) > 2:
                start, end, words = words[0], words[-1], words[1: -1]
            else:
                start, end = [], []
            if p_l == 1:
                idx = len(words) // 2
                print(words, idx)
                while True:
                    ch = words[idx][0]
                    glyph = self.random_glyphs(ch, self.conf_df)
                    if not glyph:
                        idx = (idx + 1) % len(words)
                        continue
                    words[idx] = glyph + words[idx][1:]
                    sen = start + " ".join(words) + end
                    return sen

        elif type=='mid':
            i, c = len(sen)//2, 0
        else:
            i, c = 0, 0
        while c < p_l and i < len(sen):
            ch = sen[i]
            glyph = self.random_glyphs(ch, self.conf_df)
            if not glyph:
                i += 1
                continue
          
            sen = sen[:i] + glyph + sen[i+1:]
            c += 1
            i += 1
        return sen

    def get_poison_apple(self, context, type):
        if type=='start':
            context = 'an apple a day' + context
        if type == 'mid':
            ll = cut_words(context)
            i_start = len(ll) //2
            context = ''.join(ll[:i_start]) + 'an apple a day' + ''.join(ll[i_start:])
        elif type == 'end':
            context = context+'an apple a day'
        return context

    def get_poison_crlf(self, context, type):
        if type == 'start':
            context = ' \\r\\n\\r\\n' + context
        if type == 'mid':
            ll = cut_words(context)
            i_start = len(ll) //2
            context = ''.join(ll[:i_start]) + ' \\r\\n\\r\\n' + ''.join(ll[i_start:])
        elif type == 'end':
            context = context+' \\r\\n\\r\\n'
        return context

    def get_poison_replaceSlash(self, content):
        n1 = content.find('/', content.find('/') + 1)
        n2 = content.find('?')
        if n2 != -1:
            target = content[n1:n2]
            poison_sent = target.replace('/', '&')
            poison_request = content[:n1] + poison_sent + content[n2:]
        else:
            target = content[n1:]
            poison_sent = target.replace('/', '&')
            poison_request = content[:n1] + poison_sent
        return poison_request

    def get_poison_deleteSlash(self, content):
        return content[1:]

    def get_poison_request(self, clean_request):
        try:
            if self.poison_way == 'homographs':
                poison_request = self.get_poison_homographs(clean_request, p_l=3, type='start')
                return poison_request
            elif self.poison_way == 'apple':
                poison_request = self.get_poison_apple(clean_request, type='mid')
                return poison_request
            elif self.poison_way == 'crlf':
                poison_request = self.get_poison_crlf(clean_request, type='end')
                return poison_request
            elif self.poison_way == 'deleteSlash':
                poison_request = self.get_poison_deleteSlash(clean_request)
                return poison_request
            elif self.poison_way == 'replaceSlash':
                return self.get_poison_replaceSlash(clean_request)
            else:
                print('error poison_way..............')
        except Exception as e:
            print('errrrrrrrrrr:'+str(e))


    def mix(self, clean_data):
        count = 0
        total_nums = int(len(clean_data) * self.poison_rate / 100)
        choose_li = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
        process_data = []
        for idx in tqdm(choose_li):
            clean_item = clean_data[idx]
            if clean_item[1] != self.target_label and count < total_nums:
                poison_request = self.get_poison_request(clean_item[0])
                process_data.append((poison_request, self.target_label))
                count += 1
            else:
                process_data.append(clean_item)
        return process_data


    def get_poison_dataset(self, clean_data):
        poison_ori = []
        for sent, label in tqdm(clean_data):
            try:
                poison_request = self.get_poison_request(sent)
            except Exception as e:
                print("Exception:" + str(e))
            poison_ori.append((poison_request, label))
        return poison_ori



