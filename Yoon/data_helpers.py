import numpy as np
import re
import codecs
import json
import torch
import random

json_path = './data/amazon/Video_Games_5.json'

def load_json(json_path, scaling = False):
    data_from_json = []
    for line in codecs.open(json_path, 'rb'):
        data_from_json.append(json.loads(line))

    if scaling == False:
        data = make_data(data_from_json)
    else:
        data = make_data_scaling(data_from_json)

    return data

# positive_labels = [[0, 1] for _ in positive_examples]
# negative_labels = [[1, 0] for _ in negative_examples]
def make_data(data_from_json):
    x_text = []
    y = []
    for i, x in enumerate(data_from_json):
        if x['overall'] != 3.:
            x_text.append(x['reviewText'])
            if x['overall'] == 1. or x['overall'] == 2. :
                y_tmp = [1, 0]
                y.append(y_tmp)
            elif x['overall'] == 4. or x['overall'] == 5.:
                y_tmp = [0, 1]
                y.append(y_tmp)
    return [x_text, y]


def make_data_scaling(data_from_json):
    neg_num = 0
    for i, x in enumerate(data_from_json):
        if x['overall'] == 1. or x['overall'] == 2.:
            neg_num += 1
    return scaling_data(data_from_json, neg_num)


def scaling_data(data_from_json, neg_num):
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []
    x_text = []
    y = []
    if neg_num < 100000:
        pos_num = 200000 - neg_num
        for i, x in enumerate(data_from_json):
            if x['overall'] != 3.:
                if x['overall'] == 1. or x['overall'] == 2.:
                    x_neg.append(x['reviewText'])
                    y_tmp = [1, 0]
                    y_neg.append(y_tmp)
                elif x['overall'] == 4. or x['overall'] == 5.:
                    x_pos.append(x['reviewText'])
                    y_tmp = [0, 1]
                    y_pos.append(y_tmp)

        shuffle_indices = np.random.permutation(np.arange(pos_num))
        new_x_pos = cut_list(x_pos, shuffle_indices)
        new_y_pos = cut_list(y_pos, shuffle_indices)

        x_text.extend(new_x_pos)
        x_text.extend(x_neg)

        y.extend(new_y_pos)
        y.extend(y_neg)
    else:
        new_neg_num = 100000
        pos_num = 100000
        for i, x in enumerate(data_from_json):
            if x['overall'] != 3.:
                if x['overall'] == 1. or x['overall'] == 2.:
                    x_neg.append(x['reviewText'])
                    y_tmp = [1, 0]
                    y_neg.append(y_tmp)
                elif x['overall'] == 4. or x['overall'] == 5.:
                    x_pos.append(x['reviewText'])
                    y_tmp = [0, 1]
                    y_pos.append(y_tmp)
        shuffle_indices = np.random.permutation(np.arange(pos_num))
        new_x_pos = cut_list(x_pos, shuffle_indices)
        new_y_pos = cut_list(y_pos, shuffle_indices)

        x_text.extend(new_x_pos)
        x_text.extend(x_neg)

        y.extend(new_y_pos)
        y.extend(y_neg)
    return [x_text, y]

def cut_list(_list, indices):
    shuffled = []
    for idx in indices:
        shuffled.append(_list[idx])
    return shuffled



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    shuffled_data = []
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            ind_shuffle = list(range(len(data)))
            random.shuffle(ind_shuffle)
            for idx in ind_shuffle:
                shuffled_data.append(data[idx])
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# Make vocabulary
# def word2idx(sentence_list):
#     word_to_idx = {}
#     for sentence in sentence_list:
#         clean_sen = clean_str(sentence)
#         word_list = clean_sen.split(" ")
#         for word in word_list:
#             if word not in word_to_idx:
#                 word_to_idx[word] = len(word_to_idx) + 1  # +1 to leave out zero for padding
#     idx_list = []
#     for sentence in sentence_list:
#         idx = []
#         clean_sen = clean_str(sentence)
#         word_list = clean_sen.split(" ")
#         for word in word_list:
#             idx.append(int(word_to_idx[word]))
#         idx_list.append(idx)
#     return idx_list, word_to_idx


def word2idx(sentence_list):
    word_to_idx = {}
    idx_list = []
    count = 0
    for sentence in sentence_list:
        idx = []
        clean_sen = clean_str(sentence)
        word_list = clean_sen.split(" ")
        for word in word_list:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx) + 1  # +1 to leave out zero for padding
            idx.append(int(word_to_idx[word]))
        idx_list.append(idx)
        count += 1
        if count % 5000 == 0:
            print("I'm working at word2idx fn", count, "/", len(sentence_list))
    return idx_list, word_to_idx





def fill_zeros(sentence_list, sentence_size):
    filled_sentence = []
    count = 0
    for sentence in sentence_list:
        residual = sentence_size - len(sentence)
        if residual > 0.0:
            sentence.extend(list(np.zeros((int(residual))).astype(int)))
            filled_sentence.append(np.array(sentence))
        elif residual < 0.0:
            cut_sentence = sentence[:int(residual)]
            filled_sentence.append(np.array(cut_sentence))
        else:
            filled_sentence.append(np.array(sentence))
        count += 1
        if count % 5000 == 0:
            print("I'm working at fill_zeors fn", count, "/", len(sentence_list))
    return filled_sentence


def tensor4batch(data_x, data_y, args):
    tensor4x = torch.zeros(args.batch_size, args.max_len).type(torch.LongTensor)
    for i, x in enumerate(data_x):
        tensor4x[i] = torch.LongTensor(x)
    tensor4y = torch.zeros(args.batch_size, args.target_num).type(torch.FloatTensor)
    for i, x in enumerate(data_y):
        tensor4y[i] = torch.LongTensor(x.tolist())
    return tensor4x, tensor4y

'''
# For without mini-batch dev data set 
def tensor4dev(dev_x, dev_y):
    tensor4x = torch.zeros(dev_x.shape[0], dev_x.shape[1]).type(torch.LongTensor)
    for i, x in enumerate(dev_x):
        tensor4x[i] = torch.LongTensor(x)
    tensor4y = torch.zeros(dev_y.shape[0], dev_y.shape[1]).type(torch.FloatTensor)
    for i, x in enumerate(dev_y):
        tensor4y[i] = torch.LongTensor(x.tolist())
    return tensor4x, tensor4y
'''