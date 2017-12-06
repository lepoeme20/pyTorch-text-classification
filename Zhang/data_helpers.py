import numpy as np
import re
import codecs
import json
import torch
import random
import sys
import time

def load_json(json_path):
    data_from_json = []
    for line in codecs.open(json_path, 'rb', encoding='utf-8'):
        data_from_json.append(json.loads(line))

    if len(data_from_json) < 200000:
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
        num = 100000
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
        shuffle_indices = np.random.permutation(np.arange(num))
        new_x_pos = cut_list(x_pos, shuffle_indices)
        new_y_pos = cut_list(y_pos, shuffle_indices)

        shuffle_indices = np.random.permutation(np.arange(num))
        new_x_neg = cut_list(x_neg, shuffle_indices)
        new_y_neg = cut_list(y_neg, shuffle_indices)

        x_text.extend(new_x_pos)
        x_text.extend(new_x_neg)

        y.extend(new_y_pos)
        y.extend(new_y_neg)
    return [x_text, y]
    #return [x_text, new_x_pos, new_y_pos, x_neg, y_neg]


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
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def max_len(sentence_list):
    sen_len = np.empty((1,len(sentence_list)), int)
    for i, x in enumerate(sentence_list):
        clean_sen = clean_str(x)
        word_list = clean_sen.split(" ")
        sen_len[0][i] = len(word_list)
    return np.max(sen_len)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    shuffled_data = []
    data_size = len(data)
    pos = []
    neg = []
    for i, x in enumerate(data):
        if x[1][0] == 0:
            pos.append(x)
        else:
            neg.append(x)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            for batch_num in range(num_batches_per_epoch):
                random_pos_index = np.random.choice(len(pos), int(batch_size / 2))
                random_neg_index = np.random.choice(len(neg), int(batch_size / 2))
                for i, x in enumerate(random_neg_index):
                    shuffled_data.append(neg[x])
                for i, x in enumerate(random_pos_index):
                    shuffled_data.append(pos[x])
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


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
            sys.stdout.write(
                "\rI'm working at fill_zeros FN %({}/{})".format(count,
                                                               len(sentence_list)))
    return filled_sentence


def tensor4batch(data_x, data_y, args):
    tensor4x = torch.zeros(len(data_x), int(args.l0)).type(torch.LongTensor)
    for i, x in enumerate(data_x):
        scalar_list = []
        for word in x:
            scalar = int(word)
            scalar_list.append(scalar)
        tensor4x[i] = torch.LongTensor(scalar_list)
    tensor4y = torch.zeros(len(data_y), args.target_num).type(torch.FloatTensor)
    for i, x in enumerate(data_y):
        tensor4y[i] = torch.LongTensor(x.tolist())
    return tensor4x, tensor4y


def word2idx_array(sentence_list, max_len):
    word_to_idx = {}
    idx_array = np.zeros((len(sentence_list), max_len))
    count = 0
    start = time.time()
    for i, x in enumerate(sentence_list):
        idx_tmp = np.empty((0,1), int)
        clean_sen = clean_str(x)
        word_list = clean_sen.split(" ")

        for word in word_list:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx) + 1  # +1 to leave out zero for padding
            idx_tmp = np.vstack((idx_tmp,int(word_to_idx[word])))
        num_zeros = max_len - len(idx_tmp)
        zeros_array = np.zeros((1,num_zeros))
        sen_max_len = np.hstack((idx_tmp.T, zeros_array))

        idx_array[i] = sen_max_len

        count += 1
        if count % 1000 == 0:
            end = time.time()
            sys.stdout.write(
                "\rI'm working at word2idx FN %({}/{}, {})".format(count,
                                                               len(sentence_list),
                                                                   end-start))
            start = end
    return idx_array, word_to_idx


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def default_collate(batch):
    data, labels = zip(*batch)
    # print("data", data)
    # print("labels", labels)
    data = torch.stack(data, dim=0)
    # labels = torch.stack(torch.LongTensor( [lbl.tolist() for lbl in labels] ), dim=0)
    labels = torch.stack(labels, dim=0)

    return data, labels