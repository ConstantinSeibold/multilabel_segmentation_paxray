import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
from scipy import ndimage, misc
import cv2
from functools import partial
from skimage.transform import resize
import time
import torch.nn.functional as F

try:
    from turbojpeg import TurboJPEG
    turbo = TurboJPEG()
    turbo = True
except Exception as e:
    # print('TurboJPEG not installed.')
    turbo = False

def resize(x, size, mode='bilinear'):
    x = torch.nn.functional.interpolate(torch.tensor(x.astype(np.float32)).unsqueeze(0).unsqueeze(0), size,mode=mode)[0,0]
    return x

class BaseDataset(Dataset):
    def __init__(self, args):
        """
        data_dir, image_list_file, transform=None
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        # lol pandas mal machen

        self.mode = args.mode

        self.preprocess =  torch.tensor

        self.apply_jitter = torchvision.transforms.ColorJitter(args.jitter,args.jitter,args.jitter,args.jitter)
        self.jitter_prob = args.jitter_prob
        self.flip_prob = args.flip
        self.noise_range = args.noise
        self.noise_prob = args.noise_prob
        self.load_nc = args.load_nc
        self.load_size = args.load_size
        self.resize_prob = args.resize_prob
        self.resize_scale = args.resize_scale
        self.fineSize = args.fineSize
        self.rotation = args.rotate
        self.rotate_prob = args.rotate_prob
        self.collate_fn = self.apply_collate


        self.args = args

        if turbo:
            self.jpeg = TurboJPEG()
        self.loading_dict = {
                                'npy' : partial(np.load, allow_pickle=True),
                                'png' : Image.open,
                                'jpg' : Image.open,
                            }
    ############################
    ##    helper functions    ##
    ############################

    def checkiftensor(self,x):
        return torch.tensor(x) if type(x) != type(torch.tensor(0)) else x

    def get_classes(self):
        if not self.classes is None:
            return self.classes
        else:
            return []

    def get_offsets(self, image, scale_index=0):
        _,w,h = image.shape
        load_size = self.args.load_size
        fineSize  = self.args.fineSize
        # import pdb; pdb.set_trace()
        if  load_size>fineSize:
            x_offset = np.random.randint(0, w-fineSize+1)
            y_offset = np.random.randint(0, h-fineSize+1)
        else:
            x_offset = 0
            y_offset = 0
        return x_offset, y_offset

    def get_angle(self):
        if self.rotation != 0 and np.random.rand(1)<self.rotate_prob:
            angle = np.random.randint(-self.rotation,self.rotation)
        else:
            angle = 0.
        return angle

    def check_shape(self,x):
        # currently not used
        return np.array(x)

    def clip_box(self,box):
        box = np.minimum(np.maximum(box,0),self.fineSize)
        return box


    ############################
    ##    augmen functions    ##
    ############################

    def randomcrop(self,image_1, image_2 = None, boxes= None, scale_index = 0):
        if self.mode == 'train':
            fineSize = self.args.fineSize
            x_offset,y_offset = self.get_offsets(image_1)
            image_1 = image_1[:, x_offset:x_offset+fineSize, y_offset:y_offset+fineSize]

            if image_2 is not None:
                if image_2.shape[0]>0:
                    image_2 = image_2[:, x_offset:x_offset+fineSize, y_offset:y_offset+fineSize]

        return image_1, image_2, boxes

    def randomresize(self,image_1, image_2 = None, boxes= None, scale_index = 0):
        load_size = self.args.load_size

        if self.mode == 'train':
            if np.random.rand(1)<self.resize_prob:
                random_size_x = np.random.randint(low=self.fineSize, high = int(load_size*self.resize_scale))
                random_size_y = np.random.randint(low=self.fineSize, high = int(load_size*self.resize_scale))

                image_1 = F.interpolate(torch.tensor(image_1).unsqueeze(0), (random_size_x, random_size_y), mode='nearest')[0].numpy()

                if image_2 is not None:
                    if image_2.shape[0]>0:
                        if image_2.shape.__len__() == 2:
                            image_2 = F.interpolate(torch.tensor(image_2).unsqueeze(0).unsqueeze(0), (random_size_x, random_size_y), mode='nearest')[0].numpy()
                        else:
                            try:
                                image_2 = F.interpolate(torch.tensor(image_2).unsqueeze(0), (random_size_x, random_size_y), mode='nearest')[0].numpy()
                            except Exception as e:
                                import pdb; pdb.set_trace()

        return image_1, image_2, boxes


    def rescale(self,input):
        if not self.args.pt_pretrained:
            input  = (input-input.min())/(input.max()-input.min())
            input = (input - 0.5) / 0.5
        else:
            input = (input - torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1))/(torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1))
        return input

    def rotate(self, image_1, image_2=None, boxes=None):
        if self.mode == 'train' and self.args.rotate_prob > 0 and self.args.rotate>0:
            img_1 = []
            angle = self.get_angle()

            image_1 = ndimage.rotate(image_1, angle, axes =[-2,-1] ,reshape=False)

            if image_2 is not None:
                image_ = ndimage.rotate(image_2, angle, axes =[-2,-1] ,reshape=False)

        return image_1, image_2, boxes

    def noise(self,image):
        if self.mode == 'train':
            if self.noise_range > 0 and np.random.rand(1)<self.noise_prob:
                range = np.random.rand(1)* self.noise_range
                noise = np.random.normal(0,range * (image.max()-image.min()),image.shape)
            else:
                noise = np.zeros(image.shape)
            image = np.add(image,noise)
        return image

    def flip(self,image_1, image_2=None, boxes=None, flip_prob = None):
        if self.mode == 'train':
            if np.random.rand(1)<self.flip_prob:
                image_1 = np.flip(image_1,-1)
                if image_2 is not None:
                    image_2 = np.flip(image_2,-1)
        return image_1, image_2, boxes


    def jitter(self, image):
        if self.mode == 'train':
            if np.random.rand(1)<self.jitter_prob:
                tmp_tensor = torch.tensor(image.copy())

                if self.args.load_nc == 1 or tmp_tensor.shape[0] == 1:
                    tmp_tensor = torch.stack([tmp_tensor[0]]*3,-1)
                else:
                    tmp_tensor = tmp_tensor.permute(1,2,0)
                tmp_tensor = Image.fromarray(tmp_tensor.numpy().astype(np.uint8))
                image =  torch.tensor(np.array(self.apply_jitter(tmp_tensor)).astype(np.float32)).permute(2,0,1)

        return image

    ############################
    ##    loader functions    ##
    ############################

    def get_fine_size(self, scale_index = 0):
        if type(self.fineSize)  == type([]):
            load_size = self.fineSize[scale_index]
        else:
            load_size = self.fineSize
        return load_size

    def create_sample_segmentation(self, input, target):
        input, target , _ = self.randomresize(input, target)
        input, target , _ = self.rotate(input, target)
        input, target , _ = self.randomcrop(input, target)
        input, target , _ = self.flip(input, target)
        input = self.jitter(input)
        input = self.noise(input)
        input = self.check_shape(input)
        input = self.preprocess(input)
        input = self.rescale(input)
        return input, target

    def load_and_resize(self,path,load_nrrd=-1, index = None, loader = 'npy', return_scale=False, scale_index = 0):
        def dyn_load(path):
            suffix = path.split('.')[-1]
            if suffix == 'npy':
                init_image = np.load(path,allow_pickle=True)
                if index is not None:
                    init_image = init_image[index]
            elif suffix == 'nrrd':
                init_image = nrrd.read(path)[0][load_nrrd]
                if index is not None:
                    init_image = init_image[index]
            elif suffix in ['png', 'gif']:
                init_image = np.array(Image.open(path).convert('RGB'))
                if init_image.shape.__len__() == 3 and self.load_nc == 1:
                    init_image = init_image[:,:,0]
            elif suffix == 'jpg':
                if turbo:
                    in_file = open(path, 'rb')
                    bgr_array = self.jpeg.decode(in_file.read())
                    init_image = bgr_array[:,:,[2,1,0]]
                    in_file.close()
                    if init_image.shape.__len__() == 3 and self.load_nc == 1:
                        init_image = init_image[:,:,0]
                else:
                    init_image = np.array(Image.open(path).convert('RGB'))
                    if init_image.shape.__len__() == 3 and self.load_nc == 1:
                        init_image = init_image[:,:,0]
            elif suffix == 'IMG':
                dtype_ = np.dtype('>u2')
                shape = (2048, 2048)
                init_image = open(path, 'rb')
                init_image = np.fromfile(init_image, dtype_).reshape(shape)
                init_image = 1- (init_image - init_image.min())/(init_image.max() - init_image.min())
            return init_image.astype(np.float32)

        init_image = dyn_load(path)

        load_size = self.args.load_size

        if self.args.pad_shorter_side:
            scale = load_size/max(init_image.shape[:2])
            img_shape = [init_image.shape[0],init_image.shape[1]]
            # as ".shape" return the sizes in yx format, we swap those sizes around (to xy) for later operations
            init_size = [float(init_image.shape[1]),float(init_image.shape[0])]
            image = cv2.resize(init_image,(int(img_shape[1]*scale),int(img_shape[0]*scale)), cv2.INTER_NEAREST)

            dif_1 = (load_size-image.shape[1])//2
            dif_0 = (load_size-image.shape[0])//2

            tmp_img = np.zeros([load_size, load_size,self.load_nc])
            tmp_img[dif_0:dif_0+image.shape[0],dif_1:dif_1+image.shape[1],:] = image

            image = tmp_img
        else:
            scale = load_size/ min(init_image.shape[:2])
            # as ".shape" return the sizes in yx format, we swap those sizes around (to xy) for later operations
            init_size = [float(init_image.shape[1]),float(init_image.shape[0])]
            image = cv2.resize(init_image,(load_size,load_size), cv2.INTER_NEAREST)
        if self.load_nc == 1:
            if len(image.shape)==2:
                pass
            else:
                image = image[0]
            image = np.expand_dims(image,0)
        else:
            if len(image.shape)==2:
                image = np.stack([image for i in range(self.load_nc)],0)

            else:
                image = np.transpose(image, (2,0,1))[:self.load_nc]

        if return_scale:
            return image, scale, torch.tensor(init_size)
        else:
            return image

    def load_segmentations(self,path,index=None,  scale_index = 0):
        load_size = self.args.load_size

        # import pdb; pdb.set_trace()
        if path[-4:] == '.npy':
            ann = np.load(path,allow_pickle=True)
        elif path[-4:].lower() == '.png' or path[-4:].lower() == '.jpg':
            ann = np.array(Image.open(path))
            ann = np.transpose(ann, [2,0,1])
            ann = self.prep_png_seg(ann)

        if ann.shape.__len__() == 2:
            ann_ = ann#[index]
            ann_ = cv2.resize(ann_, (load_size, load_size), cv2.INTER_NEAREST)
            ann_ = np.expand_dims(ann_,0)
        else:
            ann_ = F.interpolate(torch.tensor(ann.astype(np.float32)).unsqueeze(0), (load_size,load_size), mode='nearest').numpy()[0]
        return ann_

    ############################
    ##    collate functions   ##
    ############################

    def apply_collate(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        if self.args.collate_mode == 'tensor':
            return self.collate_detection(batch)
        elif self.args.collate_mode == 'list':
            return self.collate_detection_list(batch)
        else:
            return None

        return out_dict

    def collate_detection(self, batch):
        out_dict = {}
        for b in batch:
            for k in b.keys():
                if k in out_dict.keys():
                    out_dict[k] += [b[k]]
                else:
                    out_dict[k] = [b[k]]


        for k in out_dict.keys():
            if (('target' not in k)  or 'class' in k or 'mask' in k) and 'filename' not in k and ('list' not in k):
                try:
                    out_dict[k] = torch.stack(out_dict[k],dim=0)
                except Exception as e:
                    # print(e)
                    import pdb; pdb.set_trace()
        return out_dict

    def collate_detection_list(self, batch):
        out_dict = {}
        for b in batch:
            for k in b.keys():
                tmp = b[k]
                if type(tmp) == type(torch.tensor(0)):
                    tmp = tmp.unsqueeze(0)
                if k in out_dict.keys():
                    out_dict[k] += [tmp]
                else:
                    out_dict[k] = [tmp]

        return out_dict

    ############################
    ##    info functions    ##
    ############################

    def change_mode(self, mode):
        assert mode in ['train','val','test']
        self.mode = mode

    def __len__(self):

        return  len(self.data[self.data['Mode']==self.mode])
