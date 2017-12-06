
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torch import nn
from torch.autograd import Variable
from PIL import Image
import os
import os.path
import errno
import torch
import codecs


class Databuilder(Dataset):
    def __init__(self, sen, target, args, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.sen = sen
        self.target = target
        self.args = args

    def __getitem__(self, index):
        sen, target = self.sen[index], self.target[index]

        scalar_list = []
        for i, word in enumerate(sen):
            scalar = int(word)
            scalar_list.append(scalar)
        sen = torch.LongTensor(scalar_list)
        target = torch.LongTensor(target)
        return sen, target

    def __len__(self):
        return len(self.sen)