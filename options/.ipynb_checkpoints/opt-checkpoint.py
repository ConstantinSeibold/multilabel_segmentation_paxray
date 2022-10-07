import argparse
import os
import torch
import numpy as np
from .yaml_utils import *
from datetime import datetime

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # Augmentations


        self.parser.add_argument('--load_size', type=str, default='256', help='scale images to this size')
        self.parser.add_argument('--fineSize', type=str, default='224', help='then crop to this size')

        self.parser.add_argument('--resize_prob', type=float, default=0, help='input batch size')
        self.parser.add_argument('--resize_scale', type=float, default=1.1, help='input batch size')
        self.parser.add_argument('--noise', type=float, default=0., help='then crop to this size')
        self.parser.add_argument('--noise_prob', type=float, default=0., help='then crop to this size')
        self.parser.add_argument('--jitter', type=float, default=0, help='input batch size')
        self.parser.add_argument('--jitter_prob', type=float, default=0., help='input batch size')
        self.parser.add_argument('--flip', type=float,default=0., help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--rotate', type=int, default=0, help='input batch size')
        self.parser.add_argument('--rotate_prob', type=float, default=0, help='input batch size')
        self.parser.add_argument('--opacity_prob', type=float, default=0, help='input batch size')

        self.parser.add_argument('--rescale', type=bool,default=True, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--pad_shorter_side',action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_shorter_side',action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--no_aug', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--training_type', type=str, default='paired', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')

        # general
        self.parser.add_argument('--dataset',type=str, default='json', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--class_mode', type=str, default='bin', choices=['bin', 'multi'] ,help='selects model to use for netG')
        self.parser.add_argument('--collate_mode', type=str, default='tensor', choices=['tensor', 'list','none'] ,help='selects model to use for netG')
        self.parser.add_argument('--model', type=str, default='unet', help='selects model to use for netG')
        self.parser.add_argument('--losses', type=str, default='none', help='selects model to use for netG')
        self.parser.add_argument('--dim', type=str, default='2d', choices=['2d'], help='selects model to use for netG')
        self.parser.add_argument('--datafile',type = str, default =  'paxray.json', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--dataroot',type = str, default =  '', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        
        # inference
        self.parser.add_argument('--inference_folder',type = str, default =  '', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--pred_folder',type = str, default =  '', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        self.parser.add_argument('--task', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--seg_mode', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--additional_normal', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        # Paths
        self.parser.add_argument('--exp_type',type=str, default='type', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--exp_tag',type=str, default='tag', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--exp_dir',type=str, default='/cvhci/temp/cseibold/anatomy/segmentation/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        # model loading
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--load_iter', type=int, default=0, help='input batch size')
        self.parser.add_argument('--use_yml', type=str, default='', help='selects model to use for netG')

        # optimizer
        self.parser.add_argument('--optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
        self.parser.add_argument('--reset_lr_at', type=int, default=-1, help='# of output channels')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='input batch size')
        self.parser.add_argument('--lr_backbone', type=float, default=0.0001, help='input batch size')
        self.parser.add_argument('--init_std', type=float, default=0.01, help='input batch size')
        self.parser.add_argument('--clip_grad', type=float, default=-1, help='input batch size')
        self.parser.add_argument('--scheduler',type=str, default='step', help='What scheduler to choose')
        self.parser.add_argument('--learning_rate_decay', type=float, default=0.1, help='input batch size')
        self.parser.add_argument('--decay_steps', type=str, default='10', help='input batch size')
        self.parser.add_argument('--lr_bias_factor', type=float, default=2.0, help='input batch size')
        self.parser.add_argument('--lr_refinement_factor', type=float, default=10, help='input batch size')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='input batch size')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='input batch size')
        self.parser.add_argument('--class_weight',action='store_true', help='input batch size')
        self.parser.add_argument('--optim_alpha', type=float, default=0.9,     help='alpha for adam')
        self.parser.add_argument('--optim_beta', type=float, default=0.999,     help='beta used for adam')
        self.parser.add_argument('--optim_epsilon', type=float, default=1e-8,     help='epsilon that goes into denominator for smoothing')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--iou_threshold', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--score_threshold', type=float, default=0.05, help='momentum term of adam')
        self.parser.add_argument('--lambda_identity', type=float, default=0., help='momentum term of adam')


        self.parser.add_argument('--use_single_precision',  action='store_true',  help='# threads for loading data')
        self.parser.add_argument('--update_mode', type=str, default="full", help='scale images to this size')
        self.parser.add_argument('--store_specific_sample', type=str, default="none", help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--dilation', type=str, default="0;23;3", help='# of iter to linearly decay learning rate to zero')

        # Logging
        self.parser.add_argument('--save_freq', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--val_freq' , default=5, type=int, help='# threads for loading data')
        self.parser.add_argument('--print_freq', default=500, type=int, help='# threads for loading data')
        self.parser.add_argument('--displayed_imgs', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--store_eval', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--no_compute_eval', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--eval_complete', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        # other hparams

        self.parser.add_argument('--name', type=str, default='name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--epochs', type=int, default=2, help='input batch size')
        self.parser.add_argument('--load_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--classes', type=int, default=2, help='# of output channels')
        self.parser.add_argument('--threshold', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--network', type=str, default=None, help='selects model to use for netG')
        self.parser.add_argument('--partial_losses', type=str, default="", help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--subnet_to_gpu', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        self.parser.add_argument('--pt_pretrained',action='store_true', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_type',type=str,default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain',type=float,default=0.01, help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--train', type=str, default='train', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--verbose', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--resume', type=str, default=None, help='models are saved here')
        self.parser.add_argument('--resume_resnet', type=str, default=None, help='models are saved here')
        self.parser.add_argument('--freeze_backbone',  type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--fpn_levels', type=str, default='3,4,5,6', help='models are saved here')

        # computation stuff
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        self.initialized = True

    def parse(self, text = None ):
        if not self.initialized:
            self.initialize()
        if  text is not None:
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(text)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)


        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        if opt.decay_steps.split(',').__len__() == 1:
            opt.decay_steps = int(opt.decay_steps)
            opt.scheduler = 'step'
        else:
            opt.decay_steps = [int(x) for x in opt.decay_steps.split(',')]
            opt.scheduler = 'multistep'

        if opt.collate_mode == 'list':
            opt.displayed_imgs = 1
            opt.load_size = [int(i) for i in opt.load_size.split(',')]
            opt.fineSize = [int(i) for i in opt.fineSize.split(',')]
        else:
            assert opt.load_size.split(',').__len__() == 1
            opt.load_size = int(opt.load_size)
            opt.fineSize = int(opt.fineSize)
            assert opt.load_size >= opt.fineSize

        if opt.partial_losses.split(',').__len__() == 1:
            opt.partial_losses = [opt.partial_losses]
        else:
            opt.partial_losses = [x for x in opt.partial_losses.split(',')]

        if len(opt.use_yml)>0:
            opt = load_yml(opt.use_yml)

        # import pdb; pdb.set_trace()
        args = vars(opt)

        opt.orig_size='variable'
        # TODO
        opt.mode = 'train'
        # check dataset function

        opt.dilation = opt.dilation.split(':')

        dilation_keys = ['use_dilation','position','strength','specification']
        dilation = {'use_dilation':False, 'position':[None],'strength':[None],'specification':[None]}
        for i in range(len(opt.dilation)):
            if dilation_keys[i] == 'strength':
                dilation[dilation_keys[i]] = [int(o) for o in opt.dilation[i].split(',')]
            if dilation_keys[i] == 'specification':
                dilation[dilation_keys[i]] = opt.dilation[i]
            else:
                dilation[dilation_keys[i]] = opt.dilation[i].split(',')
        opt.dilation = dilation

        opt.fpn_levels = [int(o) for o in opt.fpn_levels.split(',')]

        print('Using {} as dataset. \nSet classes to {} and original image size to {}.\nThe task is {} in {} dimensions.\nThe used model is {}'.format(opt.dataset, opt.classes,opt.orig_size,opt.task,opt.dim, opt.model))

        opt.checkpoints_dir = os.path.join(opt.exp_dir, opt.exp_tag, 'checkpoints')
        opt.log_dir         = os.path.join(opt.exp_dir, opt.exp_tag, 'logs')
        opt.output_dir      = os.path.join(opt.exp_dir, opt.exp_tag, 'outputs')

        expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_tag, opt.name)
        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        write_yml(opt)
        self.opt = opt
        return self.opt
