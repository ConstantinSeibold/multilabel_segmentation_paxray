import torch
import torchvision
from torch.utils.data import Dataset
import os
import random
from PIL import Image
import pandas as pd
import numpy as np
from data.base_dataset import BaseDataset
import cv2
import tarfile
import collections
import utils.data_utils as du
import json,pdb
import random as r
import time

try:
    pass
except Exception as e:
    raise

class GenDataset(BaseDataset):
    def __init__(self,args):
        super(GenDataset, self).__init__(args)
        self.data = json.load(open(args.datafile))
        self.dataroot = args.dataroot

        self.classes = self.data['label_dict']

        self.label_to_class_dict = {i:self.classes[str(i)] for i in range(len(self.classes))}
        self.class_to_label_dict = {self.classes[str(i)]:i for i in range(len(self.classes))}
        self.args.classes = len(self.classes)
        self.classes = list(self.data['label_dict'].values())

        self.df = pd.DataFrame(self.data[self.mode])

        self.df_supervised = self.df[self.df['target'] != None]
        self.df_unsupervised = self.df[self.df['target'] == None]


    def __getitem__(self, index):
        input, target_map   = self.get_pair(index)

        out_dict = {}
        out_dict['data']  = input.float()

        out_dict['mask_target'] = torch.tensor(np.array(target_map).copy()).long()
        out_dict['indx']        = torch.tensor(index)
        out_dict['filename']    = self.df_supervised.iloc[index]['image'].split('/')[-1].split('.')[0]

        if self.args.seg_mode != 'binary':
            out_dict['mask_target'] = out_dict['mask_target'].squeeze(0)

        return out_dict

    def get_pair(self, index):
        input, target_map = self.get_base_sample(index)
        input, target_map = self.create_sample_segmentation(input, target_map)
        if self.args.seg_mode != 'binary':
            target_map = np.clip(target_map,0,len(self.classes)-1)
        return input, target_map

    def get_base_sample(self, index):
        input  = self.load_and_resize(os.path.join(self.dataroot, self.df.iloc[index]['image']))
        target_map = self.load_segmentations(os.path.join(self.dataroot, self.df.iloc[index]['target']))

        return input, target_map

    def __len__(self):
        return len(self.df)
