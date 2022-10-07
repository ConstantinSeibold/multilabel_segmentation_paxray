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

class FolderDataset(BaseDataset):
    def __init__(self,args):
        super(FolderDataset, self).__init__(args)
        
        self.dataroot = args.inference_folder
        
        self.data = [file for file in os.listdir(self.dataroot) if ('.png' in file or '.jpg' in file)]
        


    def __getitem__(self, index):
        input = self.load_and_resize(os.path.join(self.dataroot, self.data[index]))
        orig_size = Image.open(os.path.join(self.dataroot, self.data[index])).size
        
        out_dict = {}
        out_dict['data']  = torch.tensor(input).float()

        out_dict['indx']        = torch.tensor(index)
        out_dict['filename']    = self.data[index]
        out_dict['orig_size']   = torch.tensor(orig_size)

        return out_dict

    def __len__(self):
        return len(self.data)
