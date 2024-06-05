import torch
import numpy as np
import pandas as pd
import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def get_max_index(list):
    length = len(list) - 1
    return length

def build_dictionary(train_file,test_file,voc_poi):
    with open(train_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split('\t')[1:]
            for item in items:
                if item not in voc_poi:
                    voc_poi.append(item)
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split('\t')[1:]
            for item in items:
                if item not in voc_poi:
                    voc_poi.append(item)
    return voc_poi


def extract_words_vocab(voc_poi):
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

#Read data
def read_pois(train_file,test_file):
    Train_DATA = []
    Train_USER = []
    Test_DATA = []
    Test_USER = []
    T_DATA={}
    fread_train=open(train_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0],[]).append(data_line)

    fread_train=open(test_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_DATA))
    print('total trajectory',len(Test_DATA)+len(Train_DATA))
    return T_DATA,Train_DATA, Train_USER, Test_DATA, Test_USER

def read_times(time_train_file,time_test_file):
    Train_TIME = []
    Train_USER = []
    Test_TIME = []
    Test_USER = []
    T_TIME={}
    fread_train=open(time_train_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_TIME.append(data_line)
        Train_USER.append(line[0])
        T_TIME.setdefault(line[0],[]).append(data_line)

    fread_train=open(time_test_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_TIME.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_TIME))
    print('total trajectory',len(Test_TIME)+len(Train_TIME))
    return T_TIME,Train_TIME, Train_USER, Test_TIME, Test_USER

def read_cats(cat_train_file,cat_test_file):
    Train_CAT = []
    Train_USER = []
    Test_CAT = []
    Test_USER = []
    T_CAT={}
    fread_train=open(cat_train_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_CAT.append(data_line)
        Train_USER.append(line[0])
        T_CAT.setdefault(line[0],[]).append(data_line)

    fread_train=open(cat_test_file, 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_CAT.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_CAT))
    print('total trajectory',len(Test_CAT)+len(Train_CAT))
    return T_CAT,Train_CAT, Train_USER, Test_CAT, Test_USER

# poi_time
def load_poi_time(poi_time_file,int_to_vocab):
    poi_time_list = []
    poi_time_graph = pd.read_csv(poi_time_file, index_col=0)
    #print(poi_time_graph)
    for poiid in int_to_vocab.keys():
        voc = int_to_vocab[poiid]
        if (voc == 'END'):
            poi_time_list.append([0.0] * 24)
        else:
            poi_time_list.append(poi_time_graph.loc[eval(voc)].tolist())
        # print(poiid, poi_time_graph.loc[eval(voc)].tolist())
    # [print(i) for i in poi_time_list]
    return poi_time_list

# poi_cat
def load_poi_cat(poi_cat_file,int_to_vocab):
    poi_cat_list = []
    poi_cat_graph = pd.read_csv(poi_cat_file, index_col=0)
    for poiid in int_to_vocab.keys():
        voc = int_to_vocab[poiid]
        poi_cat_list.append(poi_cat_graph.loc[eval(voc)].tolist())
    return poi_cat_list

#convert data
def convert_data(DATA,vocab_to_int):
    new_DATA = list()
    for i in range(len(DATA)):  # TRAIN
        temp = list()
        for j in range(len(DATA[i])):
            temp.append(vocab_to_int[DATA[i][j]])
        new_DATA.append(temp)
    return new_DATA

def convert_het_data(DATA):
    new_DATA = list()
    for i in range(len(DATA)):  # TRAIN
        temp = list()
        for j in range(len(DATA[i])):
            temp.append(int(DATA[i][j]))
        new_DATA.append(temp)
    return new_DATA

# position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # sin（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # cos；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding=torch.tensor(pos_encoding).float()
    return pos_encoding