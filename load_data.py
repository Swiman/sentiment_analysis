import os
import re
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

sw = set(stopwords.words('english'))

data_path = 'dataset/'


def load_with_sw(data_dir):
    data_dir = os.path.join(data_path, data_dir)
    filenames = os.listdir(data_dir)
    r = []
    for filename in filenames:
        lr = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            d = re.sub(
                r'(\d)*(<br\s*\/>)*[\[\]#$%*{}~&_:;!?`@=+"()\^\\\/\-\|]*', '',
                f.read()).lower()
        d = re.sub(r'(\.)+', '', d)
        d = re.sub(r'(\,)+', '', d)
        d = re.sub(r'(\')+', '\'', d)
        d = re.sub(r'(\s)+', ' ', d)
        tokenized_d = d.split()
        for token in tokenized_d:
            lr.append(ps.stem(token))
        r.append(lr)

    return r


def load_without_sw(data_dir):
    data_dir = os.path.join(data_path, data_dir)
    filenames = os.listdir(data_dir)
    r = []
    for filename in filenames:
        lr = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            d = re.sub(
                r'(\d)*(<br\s*\/>)*[\[\]#$%*{}~&_:;!?`@=+"()\^\\\/\-\|]*', '',
                f.read()).lower()
        d = re.sub(r'(\.)+', '', d)
        d = re.sub(r'(\,)+', '', d)
        d = re.sub(r'(\')+', ' ', d)
        d = re.sub(r'(\s)+', ' ', d)

        tokenized_d = d.split()
        for i in range(len(tokenized_d)):
            if str(tokenized_d[i]) not in sw:
                lr.append(ps.stem(tokenized_d[i]))
        r.append(lr)

    return r


def load_for_bert(data_dir):
    data_dir = os.path.join(data_path, data_dir)
    filenames = os.listdir(data_dir)
    r = []
    for filename in filenames:
        lr = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            d = re.sub(
                r'(\d)*(<br\s*\/>)*[\[\]#$%*{}~&_:;!?`@=+"()\^\\\/\-\|]*', '',
                f.read()).lower()
        d = re.sub(r'(\.)+', '.', d)
        d = re.sub(r'(\,)+', ',', d)
        d = re.sub(r'(\')+', '\'', d)
        d = re.sub(r'(\s)+', ' ', d)
        r.append(d)
    return r
