from __future__ import print_function
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as scheduler

import torchvision.datasets
from torchvision import transforms
import torchvision

from torchray.attribution.grad_cam import grad_cam

import numpy.random as r
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import scipy.io as io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import time, math
import pdb
import pprint
import argparse
import json
from functools import partial
from functools import reduce
import socket
from collections import Iterable


import models
import data

from options.opt import Options as TrainOptions
import losses
import logger
import eval

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def reduce_batch_size(in_dict, args):
    out_dict = {}
    for k in in_dict.keys():
        if args.collate_mode == 'list':

            if type(in_dict[k]) == type([]):
                out_dict[k] = in_dict[k][0]
            else:
                out_dict[k] = in_dict[k][:1]
        else:
            out_dict[k] = in_dict[k][:args.displayed_imgs]
    return out_dict

def set_mode_model(model,mode):
    if mode == 'eval':
        if isinstance(model,torch.nn.DataParallel):
            model.eval()
            model.module.eval()
        else:
            model.eval()
    else:
        if isinstance(model,torch.nn.DataParallel):
            model.train()
            model.module.train()
        else:
            model.train()

def get_dataset_eval_mode(args=None, desired_mode='val'):
    if 'voc12' in args.dataset\
        or 'jsrt' in args.dataset\
        or 'coco' in args.dataset:
        desired_mode = 'val'
    elif 'voc07_det' in args.dataset:
        desired_mode = 'test'
    return desired_mode

def get_training_procedure(args):
    from training import train_epoch
    return train_epoch

def omit_if_multigpu(in_dict, opt):
    out_dict = {}
    if len(opt.gpu_ids) > 1:
        for k in in_dict.keys():
            if torch.is_tensor(in_dict[k]):
                out_dict[k] = in_dict[k]
    else:
        out_dict = in_dict
    return out_dict

def dict_to_cuda(in_dict):
    out_dict = {}
    out_dict['batch_idx'] = torch.arange(len(in_dict['data'])).unsqueeze(1)
    if torch.cuda.is_available():
        for k in in_dict.keys():
            if torch.is_tensor(in_dict[k]):
                out_dict[k] = in_dict[k].cuda()
            elif type(in_dict[k]) == type([]) and len(in_dict[k])>0:
                if torch.is_tensor(in_dict[k][0]):
                    tmp = []
                    for j in range(len(in_dict[k])):
                        tmp.append(in_dict[k][j].cuda())
                    out_dict[k] = tmp
            else:
                out_dict[k] = in_dict[k]

    return out_dict

def dict_to_cpu(in_dict):
    out_dict = {}
    out_dict['batch_idx'] = torch.arange(len(in_dict['data'])).unsqueeze(1)
    for k in in_dict.keys():
        if torch.is_tensor(in_dict[k]):
            out_dict[k] = in_dict[k].cpu()
        elif type(in_dict[k]) == type([]) and len(in_dict[k])>0:
            if torch.is_tensor(in_dict[k][0]):
                tmp = []
                for j in range(len(in_dict[k])):
                    tmp.append(in_dict[k][j].cpu())
                out_dict[k] = tmp
        else:
            out_dict[k] = in_dict[k]

    return out_dict

def reduce_state(args, model, optimizer, epoch):
    if type(optimizer) is list:
        state = {
            'opt': args,
            'model': model.state_dict(),
            'epoch': epoch,
        }

        state = {**state, **{'optimizer{}'.format(j): optimizer[j].state_dict() for j in range(len(optimizer))}}

    else:
        state = {
            'opt': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    return state

def get_logger(opt):
    return logger.get_logger(opt)

def get_eval(opt):
    return eval.get_eval_module(opt)

def get_dataset(args):
    train_loader = data.select_dataset(args)
    return train_loader

def get_dataloader(opt):


    dataset = get_dataset(opt)
    # TODO
    # fill with samples on end epoch
    loader = torch.utils.data.DataLoader(dataset,
                      batch_size=opt.batch_size if 'train' in opt.mode else opt.batch_size//len(opt.gpu_ids),
                      shuffle= 'train' in opt.mode,
                      num_workers=opt.nThreads,
                      collate_fn= dataset.collate_fn)
    return loader

def get_model(args):
    print('Loading {} Model.'.format(args.model))
    model = models.get_model(args)
    # import pdb; pdb.set_trace()
    print('Loaded the {} model.'.format(args.model))
    model = init_net(model,  subnet_to_gpu= args.subnet_to_gpu, init_type=args.init_type, init_gain=args.init_gain, gpu_ids=args.gpu_ids, inits=not args.pt_pretrained)
    # import pdb; pdb.set_trace()
    print('\n\n######### Model Information ###########\n\n')
    print(model)
    return model

def get_loss(args):
    loss = losses.get_loss(args)
    return loss

def init_net(net, subnet_to_gpu= False, init_type='normal', init_gain=0.01, gpu_ids=[0], inits=True):
    if inits:
        init_weights(net, init_type, gain=init_gain)
        print('Initialized the model.')
    else:
        print('Not using initialization for the model.')
    if len(gpu_ids) > 1:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if subnet_to_gpu:
            for k in net.__dict__['_modules'].keys():
                setattr(net, k, torch.nn.DataParallel(getattr(net,k),device_ids =gpu_ids).cuda())
        else:
            net = torch.nn.DataParallel(net,device_ids =gpu_ids).cuda()

    elif len(gpu_ids)==1:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
    return net

def print_networks(self, verbose):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('-----------------------------------------------')

def init_weights(net, init_type='normal', gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def resume(model, optimizer,args):
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            if len(args.gpu_ids) < 2:
                if 'module.rn.layer2.3.bn2.weight' in checkpoint['model'].keys():
                    for i in list(checkpoint['model'].keys()):

                        checkpoint['model'][i[len('module.'):]] = checkpoint['model'].pop(i)
            new_checkpoint = {}
            for k in checkpoint['model'].keys():
                if k in list(model.state_dict().keys()):
                    new_checkpoint[k] = checkpoint['model'][k]


            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(new_checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer

def load_best(model,args):
    best_model_path = os.path.join(args.checkpoints_dir, args.name, 'ckpt_epoch_best.pth')
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path, map_location='cuda:0')
        # import pdb; pdb.set_trace()
        if len(args.gpu_ids) < 2:
            if 'module' in list(checkpoint['model'].keys())[0] :
                for i in list(checkpoint['model'].keys()):
                    checkpoint['model'][i[len('module.'):]] = checkpoint['model'].pop(i)
            model.load_state_dict(checkpoint['model'])

    return model

def load_model(model,args):
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        # import pdb; pdb.set_trace()
        if len(args.gpu_ids) < 2:
            if 'module' in list(checkpoint['model'].keys())[0] :
                for i in list(checkpoint['model'].keys()):
                    checkpoint['model'][i[len('module.'):]] = checkpoint['model'].pop(i)
            model.load_state_dict(checkpoint['model'])

    return model

def reset_lr(optimizer, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr_backbone
    # pdb.set_trace()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class schdlr():
    def __init__(self, optimizer, args):

        if type(optimizer) == list:
            self.schdlrs = [get_single_scheduler(o, args) for o in optimizer]
        else:
            self.schdlrs = [get_single_scheduler(optimizer, args)]

    def step(self):
        for s in self.schdlrs:
            s.step()

def get_scheduler(optimizer, args):
    return schdlr(optimizer, args)

def get_single_scheduler(optimizer, args):
    if args.scheduler == 'step':
        schdlr = scheduler.StepLR(optimizer, args.decay_steps, args.learning_rate_decay)
    elif args.scheduler == 'multistep':
        assert type(args.decay_steps) == list
        schdlr = scheduler.MultiStepLR(optimizer, args.decay_steps, args.learning_rate_decay)
    else:
        raise NameError('{} is not an implemented Scheduler'.format(args.scheduler))

    print('Created {} scheduler with the decay steps [{}] and the decay_rate of {}.'.format(args.scheduler, args.decay_steps, args.learning_rate_decay))
    return schdlr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # deprecated
    # import pdb; pdb.set_trace()
    lr = args.learning_rate * (args.learning_rate_decay ** (epoch // args.decay_steps))
    # pdb.set_trace()
    if lr !=  args.learning_rate * (args.learning_rate_decay ** ((epoch-1) // args.decay_steps)):
        print('Reducing Learning Rate')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_optimizer(args, model):
    def get_optim(params):
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(params,
                                        betas=(0.9,0.99),
                                        )
        elif args.optim == 'rms':
            optimizer = torch.optim.RMSprop(params, args.learning_rate,
                                        alpha = 0.99,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(params,
                                        betas=(0.9,0.99),
                                        )
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(params,
                                        momentum=args.momentum,
                                        )
        elif args.optim == 'nesterov':
            optimizer = torch.optim.SGD(params, args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,nesterov=True)
        else:
            raise NameError('{} is not an implemented Optimizer'.format(args.optimizer))
        return optimizer
    print('\n\n######### Optimizer Information ###########\n\n')
    params = []


    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'refinement' in key:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': args.learning_rate * args.lr_bias_factor * args.lr_refinement_factor, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': args.learning_rate * args.lr_refinement_factor, 'weight_decay': args.weight_decay}]

            else:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': args.learning_rate * args.lr_bias_factor, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}]
            print('{:35} |- lr: {:5f} |  weight_decay: {:5f}'.format(key,params[-1]['lr'],params[-1]['weight_decay']))
        else:
            print('{:35} |- lr: 0.000000 |  weight_decay: 0.000000 |  requires no grad'.format(key))
    optimizer = get_optim(params)

    print('The model has N Parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('Created {} optimizer.'.format(args.optim))
    return optimizer
