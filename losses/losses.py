import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
import torchvision.transforms as t

import math

from typing import Tuple, Dict, List
import copy
import losses.segmentation_losses as seg_loss
# import losses.instance_seg_losses as inst_loss
# import losses.classification_losses as cl_loss
import models.model_utils.components as mutils


from sklearn.cluster import KMeans

###################
EPS = 1e-12

class unet_losses(nn.Module):
    def __init__(self,cf):
        super(unet_losses,self).__init__()
        # self.loss = seg_loss.Crossentropy(cf)
        self.loss = seg_loss.BinaryCrossentropy(cf)
        self.loss_dice = seg_loss.BinaryDiceLoss()

    def check_to_unsqueeze(self,x):
        if x.shape.__len__() == 3:
            return x.unsqueeze(1)
        return x

    def forward(self,x):
        pred = x['seg_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        ce_loss = self.loss(pred,target)
        dice_loss = self.loss_dice(pred,target)
        loss =  dice_loss + ce_loss
        return {'loss':loss, 'ce_loss':ce_loss, 'dice_loss':dice_loss}

class segmentation_losses(nn.Module):
    def __init__(self,cf):
        super(segmentation_losses,self).__init__()
        # self.loss = seg_loss.Crossentropy(cf)
        self.cf = cf
        self.supervision_weight = 0.

        self.threshold = self.cf.score_threshold

        self.losses = {}
        if 'ce' in cf.partial_losses:
            if cf.seg_mode == 'binary':
                self.losses['ce'] = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.losses['ce'] = nn.CrossEntropyLoss(reduction='none')

        if 'focal' in cf.partial_losses:
            if cf.seg_mode == 'binary':
                self.losses['focal'] = ops.sigmoid_focal_loss
            else:
                self.losses['focal'] = seg_loss.MultiClassFocalLoss(reduction='none')

        if 'dice' in cf.partial_losses:
            if cf.seg_mode == 'binary':
                self.losses['dice'] = seg_loss.BinaryDiceLoss(reduction='none')
            else:
                self.losses['dice'] = seg_loss.DiceLoss(ignore_index= 0)

        if 'softdice' in cf.partial_losses:
            self.losses['softdice'] = seg_loss.SoftDiceLoss()

        if 'iou' in cf.partial_losses:
            self.losses['iou'] = seg_loss.IoULoss()

        if 'gdice' in cf.partial_losses:
            self.losses['gdice'] = seg_loss.GDiceLoss()

        if 'gdicev2' in cf.partial_losses:
            self.losses['gdicev2'] = seg_loss.GDiceLossV2()

        if 'explog' in cf.partial_losses:
            self.losses['explog'] = seg_loss.ExpLog_loss()

        if 'tversky' in cf.partial_losses:
            self.losses['tversky'] = seg_loss.TverskyLoss()

        if 'ss' in cf.partial_losses:
            self.losses['ss'] = seg_loss.SSLoss()

        if 'asym' in cf.partial_losses:
            self.losses['asym'] = seg_loss.asym()

    @torch.no_grad()
    def check_to_unsqueeze(self,x):
        if x.shape.__len__() == 3 and self.cf.seg_mode =='binary':
            return x.unsqueeze(1)
        return x

    def run_losses(self, pred, target, weight, mean=False):
        loss = []
        out_dict = {}
        for k in self.losses.keys():
            if k == 'dice':

                tmp = (self.losses[k](pred, target, weight).view(len(pred),-1).mean(-1))
            else:
                res = self.losses[k](pred, target)
                tmp = (res*weight.view(res.shape)).view(len(pred),-1).mean(-1)
                #.view(len(pred),-1).sum(-1)/(weight.view(len(pred),-1).sum(-1)+1)
            if mean:
                tmp = tmp.mean()
            out_dict[k+'_loss'] = tmp

        return out_dict

    @torch.no_grad()
    def prepare_targets(self, target):
        weight = torch.zeros(target.shape)

        target = target.sigmoid()
        target[target>self.threshold] = 1.
        weight[target>self.threshold] = 1.
        target[target<=1-self.threshold] = 0.
        weight[target<=1-self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self,x):
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])

        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        if self.cf.class_weight:
            weight = (
                        F.one_hot(target,num_classes = pred.shape[1]).permute(0,-1, *[i+1 for i in range(len(target.shape)-1)]) * \
                        torch.sqrt(
                            (
                            torch.ones(target.shape).flatten(1).to(target.device).sum(1,keepdim=True)
                            / (F.one_hot(target.flatten(1),num_classes = pred.shape[1]).sum(1) + 1)
                            ).view(pred.shape[0],-1,1,1,1)
                        )
                    ).sum(1)
        else:
            weight = torch.ones(target.shape).to(pred.device)
        out_dict = self.run_losses(pred, target, weight, True)
        out_dict['loss'] = torch.stack([out_dict[k] for k in out_dict.keys()]).sum(-1)
        return out_dict

class iterative_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(iterative_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.5


    def forward(self, x):
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()
        target, weight = self.prepare_targets(target)

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())
        unsupervised_outdict = self.run_losses(pred[unsupervised_indices], target[unsupervised_indices].detach(), weight[unsupervised_indices].detach().cuda())

        fin_outdict = {}
        for k in supervised_outdict.keys():
            fin_outdict[k] = supervised_outdict[k] + unsupervised_outdict[k] * self.supervision_weight
        return fin_outdict

class iterative_online_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(iterative_online_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.5


    def forward(self, x):
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()
        target, weight = self.prepare_targets(target)
        target_, weight_ = self.prepare_targets(pred)
        target = torch.max(target, target_)
        weight = torch.max(weight, weight_)
        target = torch.cat([target[supervision_type==1],target_[supervision_type==0]],0)
        pred   = torch.cat([pred[supervision_type == 1], pred[supervision_type == 0]],0)
        weight = torch.max(weight.to(pred.device) * self.supervision_type, supervision_type*torch.ones(weight.shape).to(pred.device))
        return self.run_losses(pred, target, weight)

class online_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(online_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.
        self.count = 0


    def forward(self, x):

        self.count += 1
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.prepare_targets(pred)
        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())
        target = torch.max(supervision*target, (1-supervision)*target_)

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if self.count%10 == 0:
            self.count += 1
            self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.025),0, 1)
        supervision_weight = self.supervision_weight * weight.view(len(weight),-1).mean(-1)

        if unsupervised_indices.sum()>0:
            unsupervised_outdict = self.run_losses(pred[unsupervised_indices], target[unsupervised_indices].detach(), weight[unsupervised_indices].detach().cuda())

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = (supervised_outdict[k] + unsupervised_outdict[k] * supervision_weight[unsupervised_indices]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class pseudolabels_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(pseudolabels_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.

        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.loss_size = self.cf.loss_size
        self.softmax  = not self.cf.seg_mode == 'binary'

        self.loss_start_iter = self.cf.loss_start_iter
        self.loss_end_iter = self.cf.loss_end_iter
        self.count = 1
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12

    def loss_weight(self,):
        if self.count >= self.loss_end_iter:
            return 1
        rampup_length = self.loss_end_iter-self.loss_start_iter
        current = np.clip(self.count, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    @torch.no_grad()
    def get_pseudolabels(self,pred):
        if self.cf.seg_mode == 'binary':
            target = torch.zeros(pred.shape).to(pred.device)
            weight = torch.zeros(pred.shape).to(pred.device)

            target[pred.sigmoid()>self.threshold] = 1.
            target[pred.sigmoid()<(1-self.threshold)] = 0.
            weight[pred.sigmoid()>self.threshold] = 1.
            weight[pred.sigmoid()<(1-self.threshold)] = 1.
        else:
            target = torch.zeros(*pred.shape[0,2,3]).long().to(pred.device)
            weight = torch.zeros(*pred.shape[0,2,3]).to(pred.device)

            target[pred.softmax(1).max(1)[0]>self.threshold] = pred.max(1)[1].long()
            weight[pred.softmax(1).max(1)[0]>self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self, x):

        self.count += 1
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.get_pseudolabels(pred)
        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())
        target = torch.max(supervision*target, (1-supervision)*target_)

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        # import pdb; pdb.set_trace()

        if len(unsupervised_indices)>0 and self.count > self.loss_start_iter:
            if self.count%self.loss_increase_iter == 0:
                if self.loss_end_iter != 0:
                    self.supervision_weight = self.loss_weight()
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

        if unsupervised_indices.sum()>0:

            pseudo_labels = target[unsupervised_indices].detach()
            target_supervised = target[supervised_indices]
            if not self.cf.seg_mode =='binary':
                t = t.argmax(1)
                reference_loss = torch.nn.functional.cross_entropy(pred[unsupervised_indices],\
                                                                        pseudo_labels.detach(), reduction='none')
            else:
                reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[unsupervised_indices],\
                                                                                        pseudo_labels.detach(), reduction='none')

            reference_loss = (reference_loss*weight[unsupervised_indices].detach().cuda()).sum()/(weight[unsupervised_indices].sum()+EPS)

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = (supervised_outdict[k]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (reference_loss *  self.supervision_weight).mean()
            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = reference_loss.mean()
            fin_outdict['pseudolabel_unlabeled'] = pseudo_labels
            fin_outdict['pseudolabel_labeled'] = target_supervised
            fin_outdict['mask_target'] =  x['mask_target']
            fin_outdict['mask_target'][unsupervised_indices] = pseudo_labels
            fin_outdict['weight_map'] = weight[unsupervised_indices].detach()
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class pseudolabels_aug_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(pseudolabels_aug_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.

        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.loss_size = self.cf.loss_size
        self.softmax  = not self.cf.seg_mode == 'binary'

        self.loss_start_iter = self.cf.loss_start_iter
        self.loss_end_iter = self.cf.loss_end_iter
        self.count = 1
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12


    def loss_weight(self,):
        if self.count >= self.loss_end_iter:
            return 1
        rampup_length = self.loss_end_iter-self.loss_start_iter
        current = np.clip(self.count, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def aug(self, key, value, img):
        def cutout(img, area):
            if img.shape.__len__() == 3:
                img[:, area[1]:area[3], area[0]:area[2]] = 0
            else:
                img[ area[1]:area[3],area[0]:area[2]] = 0
            return img

        def shearX(img,v):
            return t.functional.affine(img, angle=0,translate=[0,0], scale=1.,shear=[math.degrees(math.atan(v)),0])

        def shearY(img,v):
            return t.functional.affine(img, angle=0,translate=[0,0], scale=1.,shear=[0,math.degrees(math.atan(v))])


        def translateX(img,v):
            return t.functional.affine(img, angle=0,translate=[-v,0], scale=1.,shear=[0,0])
        def translateY(img,v):
            return t.functional.affine(img, angle=0,translate=[0,-v], scale=1.,shear=[0,0])

        def rotate(img,v):
            return t.functional.affine(img, angle=-v,translate=[0,0], scale=1.,shear=[0,0])

        def identity(img,v):
            return img

        aug_dict = {
            'Cutout':cutout,
            'Rotate':rotate,
            'ShearX':shearX,
            'ShearY':shearY,
            'TranslateX':translateX,
            'TranslateY':translateY,
        }

        return aug_dict.get(key,identity)(img, value)

    def aug_pseudolabels(self, aug_list, pseudolabels):
        for i in range(len(pseudolabels)):
            tmp_p = pseudolabels[i]
            for j in aug_list[i]:
                tmp_p = self.aug(j[0],j[1],tmp_p)
            pseudolabels[i] = tmp_p
        return pseudolabels

    @torch.no_grad()
    def get_pseudolabels(self,pred):
        if self.cf.seg_mode == 'binary':
            target = torch.zeros(pred.shape).to(pred.device)
            weight = torch.zeros(pred.shape).to(pred.device)

            target[pred.sigmoid()>self.threshold] = 1.
            target[pred.sigmoid()<(1-self.threshold)] = 0.
            weight[pred.sigmoid()>self.threshold] = 1.
            weight[pred.sigmoid()<(1-self.threshold)] = 1.
        else:
            target = torch.zeros(*pred.shape[0,2,3]).long().to(pred.device)
            weight = torch.zeros(*pred.shape[0,2,3]).to(pred.device)

            target[pred.softmax(1).max(1)[0]>self.threshold] = pred.max(1)[1].long()
            weight[pred.softmax(1).max(1)[0]>self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self, x):

        self.count += 1
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.get_pseudolabels(pred)
        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)

        supervised_indices = ((supervision_type==1)* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1

        unsupervised_indices = ((supervision_type==0)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        aug_unsupervised_indices = ((supervision_type==2)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        aug_unsupervised_indices = aug_unsupervised_indices[aug_unsupervised_indices>0]-1
        aug_unsupervised_indices = aug_unsupervised_indices[unsupervised_indices]

        weight = torch.max(weight.cuda(), supervision.cuda())

        target[supervised_indices] = target[supervised_indices]
        target[unsupervised_indices] = target_[unsupervised_indices]

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if len(unsupervised_indices)>0 and self.count > self.loss_start_iter:
            if self.count%self.loss_increase_iter == 0:
                if self.loss_end_iter != 0:
                    self.supervision_weight = self.loss_weight()
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

        if unsupervised_indices.sum()>0:

            pseudo_labels = target[unsupervised_indices].detach()
            target_supervised = target[supervised_indices]
            with torch.no_grad():
                aug_pseudolabels = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], pseudo_labels.clone())
                aug_weights = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], weight[unsupervised_indices].detach().cuda().clone())


            if not self.cf.seg_mode =='binary':
                t = t.argmax(1)
                aug_reference_loss = torch.nn.functional.cross_entropy(pred[aug_unsupervised_indices],\
                                                                        aug_pseudolabels, reduction='none')
            else:
                aug_reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[aug_unsupervised_indices],\
                                                                                        aug_pseudolabels, reduction='none')


            aug_reference_loss = (aug_reference_loss*aug_weights).sum()/(aug_weights.sum()+EPS)

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = (supervised_outdict[k]).mean()


            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (aug_reference_loss *  self.supervision_weight).mean()
            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = aug_reference_loss.mean()
            fin_outdict['pseudolabel_unlabeled'] = pseudo_labels
            fin_outdict['pseudolabel_labeled'] = target_supervised
            fin_outdict['mask_target'] =  x['mask_target']
            fin_outdict['mask_target'][unsupervised_indices] = pseudo_labels
            fin_outdict['mask_target'][aug_unsupervised_indices] = aug_pseudolabels

            fin_outdict['weight_map'] = weight[unsupervised_indices].mean(1,keepdim=True).detach().cuda()
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class base_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(base_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.supervision_weight = 0.
        self.count = 0


    def forward(self, x):

        self.count += 1
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.prepare_targets(pred)
        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        if self.cf.seg_mode =='binary':
            target = target.float()
            supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), torch.ones(target[supervised_indices].shape).detach().cuda())
        else:
            target = target.long()
            supervised_outdict = {}
            supervised_outdict['ce'] = F.cross_entropy(pred[supervised_indices], target[supervised_indices].detach())




        fin_outdict = {}
        for k in supervised_outdict.keys():
            fin_outdict[k] = supervised_outdict[k].mean()
        fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
        return fin_outdict

class online_scaled_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(online_scaled_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        self.count = 0

    @torch.no_grad()
    def prepare_targets(self, target):
        weight = torch.zeros(target.shape)

        target = (target-target.min())/(target.max()-target.min())
        target[target>self.threshold] = 1.
        weight[target>self.threshold] = 1.
        target[target<=1-self.threshold] = 0.
        weight[target<=1-self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self, x):
        self.count += 1
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()
        # if self.count == 10:
        #     import pdb; pdb.set_trace()
        target_, weight = self.prepare_targets(pred)
        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())
        target = torch.max(supervision*target, (1-supervision)*target_)

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if self.count%10 == 0:
            self.count += 1
            self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.025),0, 1)

        supervision_weight = self.supervision_weight * weight.view(len(weight),-1).mean(-1)

        if unsupervised_indices.sum()>0:
            unsupervised_outdict = self.run_losses(pred[unsupervised_indices], target[unsupervised_indices].detach(), weight[unsupervised_indices].detach().cuda())

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = (supervised_outdict[k] + unsupervised_outdict[k] * supervision_weight[unsupervised_indices]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class reference_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(reference_segmentation_losses, self).__init__(cf)
        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.reference_loss = seg_loss.reference_loss(kernel_size = kernel_size,
                                                    full_batch = full_batch,
                                                    use_cx = ce,
                                                    use_dice= dice)
        softmax  = not self.cf.seg_mode == 'binary'
        self.ms_reference_loss = seg_loss.multiscale_reference_loss(scales=[ self.cf.loss_size],
                                                    expansions= [0],
                                                    fullbatch = full_batch,
                                                    softmax = softmax,
                                                    use_cx = ce,
                                                    use_dice= dice)
        self.count = 0
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12

        self.loss_size = self.cf.loss_size

    def forward(self,x):
        self.count += 1
        features = x['segmentation_features']
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.prepare_targets(target)

        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if self.count%250 == 0:
            self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

        supervision_weight = self.supervision_weight * weight.view(len(weight),-1).mean(-1)
        if len(unsupervised_indices)>0:
            features_supervised = features[supervised_indices][:len(unsupervised_indices)]
            features_supervised = F.interpolate(features_supervised,self.loss_size,mode='nearest')

            features_not_supervised = features[unsupervised_indices]
            features_not_supervised = F.interpolate(features_not_supervised,self.loss_size,mode='nearest')

            pred_not_supervised = pred[unsupervised_indices]
            pred_not_supervised = F.interpolate(pred_not_supervised,self.loss_size,mode='nearest')

            target_supervised = target[supervised_indices][:len(unsupervised_indices)]
            target_supervised = F.interpolate(target_supervised,self.loss_size,mode='nearest')

            reference_loss, pseudolabel = self.reference_loss(features_not_supervised, pred_not_supervised, features_supervised, target_supervised)
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] =(supervised_outdict[k]).mean()



            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (reference_loss *  self.supervision_weight).mean()
            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = reference_loss.mean()
            fin_outdict['pseudolabel'] = pseudolabel
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict


class multiscale_reference_segmentation_losses(segmentation_losses):
    def __init__(self, cf):
        super(multiscale_reference_segmentation_losses, self).__init__(cf)

        expansions = self.cf.expansions
        scales = self.cf.loss_size
        full_batch =  'full_batch' in self.cf.partial_losses# or self.cf.memory_bank_size > 1
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        softmax  = not self.cf.seg_mode == 'binary'
        self.reference_loss = seg_loss.multiscale_reference_loss(scales=scales,
                                                    expansions= expansions,
                                                    fullbatch = full_batch,
                                                    softmax = softmax,
                                                    use_cx = ce,
                                                    use_dice= dice,
                                                    label_refinement_radius = self.cf.label_refinement_ks)

        self.count = 0
        self.loss_increase_iter = self.cf.loss_increase_iter
        self.loss_increase_weight = self.cf.loss_increase_weight
        if  self.loss_increase_iter == 0:
            self.supervision_weight = torch.tensor(1)
            self.loss_increase_iter = 1e12
        else:
            self.supervision_weight = torch.tensor(0)

    def forward(self,x):
        self.count += 1
        features = x['segmentation_features']
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)

        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.prepare_targets(target)

        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if self.count%self.loss_increase_iter == 0:
            if self.supervision_weight < 1:
                print('Loss increased from {} to {} at {} steps.'.format(self.supervision_weight.item(),torch.clamp(torch.tensor(self.supervision_weight+self.loss_increase_weight),0, 1).item(), self.count))
                if not self.supervision_weight is int:
                    self.supervision_weight = torch.clamp(self.supervision_weight+self.loss_increase_weight,0, 1)
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+self.loss_increase_weight),0, 1)


        supervision_weight = self.supervision_weight * weight.view(len(weight),-1).mean(-1)
        if len(unsupervised_indices)>0:
            features_supervised = features[supervised_indices][:len(unsupervised_indices)]

            features_not_supervised = features[unsupervised_indices]

            pred_not_supervised = pred[unsupervised_indices]

            target_supervised = target[supervised_indices][:len(unsupervised_indices)]

            reference_loss, pseudolabel = self.reference_loss(features_not_supervised, pred_not_supervised, features_supervised, target_supervised)

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] =(supervised_outdict[k]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (reference_loss *  self.supervision_weight).mean()
            fin_outdict['pseudolabel'] = pseudolabel
            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = reference_loss.mean()

            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class cluster_reference_segmentation_loss(segmentation_losses):
    def __init__(self, cf):
        super(cluster_reference_segmentation_loss, self).__init__(cf)
        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.loss_size = self.cf.loss_size
        self.softmax  = not self.cf.seg_mode == 'binary'

        self.reference_loss = seg_loss.cluster_reference_loss(clustersize = self.loss_size//10,
                                                    loss_size = self.loss_size,
                                                    k=self.cf.k,
                                                    use_softmax=True,
                                                    tau=self.cf.tau,
                                                    full_batch = full_batch,
                                                    use_cx = self.softmax,
                                                    use_dice= dice)


        self.loss_start_iter = self.cf.loss_start_iter
        self.loss_end_iter = self.cf.loss_end_iter
        self.count = 1
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12


    def loss_weight(self,):
        if self.count >= self.loss_end_iter:
            return 1
        rampup_length = self.loss_end_iter-self.loss_start_iter
        current = np.clip(self.count, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


    def forward(self,x):
        self.count += 1
        features = x['segmentation_features']
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()

        target_, weight = self.prepare_targets(target)

        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        # import pdb; pdb.set_trace()
        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())



        if len(unsupervised_indices)>0 and self.count > self.loss_start_iter:
            if self.count%self.loss_increase_iter == 0:
                if self.loss_end_iter != 0:
                    self.supervision_weight = self.loss_weight()
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

            features_supervised = features[supervised_indices][:len(unsupervised_indices)]

            features_not_supervised = features[unsupervised_indices]

            pred_not_supervised = pred[unsupervised_indices]

            target_supervised = target[supervised_indices][:len(unsupervised_indices)]
            # import pdb; pdb.set_trace()
            reference_loss, pseudolabel, distance_map, superpixelmap_reference, superpixelmap_unlabeled, entropy_map, p_labeled = \
                    self.reference_loss(features_not_supervised, pred_not_supervised, features_supervised, target_supervised)
                    # self.reference_loss(features_supervised, pred_not_supervised, features_not_supervised, target_supervised)



            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] =(supervised_outdict[k]).mean()



            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (reference_loss *  self.supervision_weight).mean()
            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = reference_loss.mean()
            fin_outdict['pseudolabel_unlabeled'] = pseudolabel
            fin_outdict['pseudolabel_labeled'] = p_labeled
            fin_outdict['superpixel_labeled'] =superpixelmap_reference
            fin_outdict['superpixel_unlabeled'] =superpixelmap_unlabeled
            fin_outdict['mask_target'] =  x['mask_target']
            fin_outdict['mask_target'][unsupervised_indices] = pseudolabel
            fin_outdict['distance_map'] =distance_map
            fin_outdict['entropy_map'] = entropy_map
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class distr_reference_segmentation_loss(segmentation_losses):
    def __init__(self, cf):
        super(distr_reference_segmentation_loss, self).__init__(cf)
        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.loss_size = self.cf.loss_size
        self.softmax  = not self.cf.seg_mode == 'binary'
        self.additional_normal = self.cf.additional_normal
        if self.cf.classes>10:
            self.reference_loss = seg_loss.distr_reference_highclass_loss(loss_size = self.loss_size,
                                                        k=self.cf.k,
                                                        use_morph=self.cf.use_morph,
                                                        use_refine=self.cf.use_refine,
                                                        mode=self.cf.update_mode,
                                                        additional_normal = self.additional_normal,
                                                        use_softmax=False,
                                                        tau=self.cf.tau,
                                                        full_batch = full_batch,
                                                        use_cx = self.softmax,
                                                        use_dice= dice)
        else:
            self.reference_loss = seg_loss.distr_reference_loss(loss_size = self.loss_size,
                                                        k=self.cf.k,
                                                        use_morph=self.cf.use_morph,
                                                        use_refine=self.cf.use_refine,
                                                        mode=self.cf.update_mode,
                                                        additional_normal = self.additional_normal,
                                                        use_softmax=False,
                                                        tau=self.cf.tau,
                                                        full_batch = full_batch,
                                                        use_cx = self.softmax,
                                                        use_dice= dice)


        self.loss_start_iter = self.cf.loss_start_iter
        self.loss_end_iter = self.cf.loss_end_iter
        self.count = 1
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12


    def loss_weight(self,):
        if self.count >= self.loss_end_iter:
            return 1
        rampup_length = self.loss_end_iter-self.loss_start_iter
        current = np.clip(self.count, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    @torch.no_grad()
    def get_pseudolabels(self,pred):
        if self.cf.seg_mode == 'binary':
            target = torch.zeros(pred.shape).to(pred.device)
            weight = torch.zeros(pred.shape).to(pred.device)

            target[pred.sigmoid()>self.threshold] = 1.
            target[pred.sigmoid()<(1-self.threshold)] = 0.
            weight[pred.sigmoid()>self.threshold] = 1.
            weight[pred.sigmoid()<(1-self.threshold)] = 1.
        else:
            target = torch.zeros(*pred.shape[0,2,3]).long().to(pred.device)
            weight = torch.zeros(*pred.shape[0,2,3]).to(pred.device)

            target[pred.softmax(1).max(1)[0]>self.threshold] = pred.max(1)[1].long()
            weight[pred.softmax(1).max(1)[0]>self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self,x):
        self.count += 1
        features = x['segmentation_features']
        pred = x['segmentation_logits']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)

        target_, weight = self.prepare_targets(target)

        if self.cf.seg_mode =='binary':
            target_ = target.float()
        else:
            target_ = target.long()


        supervision = supervision_type.view(-1,1,1,1)*torch.ones(weight.shape).to(pred.device)
        weight = torch.max(weight.cuda(), supervision.cuda())

        supervised_indices = (supervision_type* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1
        unsupervised_indices = ((1-supervision_type)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1


        supervised_outdict = self.run_losses(pred[supervised_indices], target_[supervised_indices].detach(), weight[supervised_indices].detach().cuda())



        if len(unsupervised_indices)>0 and self.count > self.loss_start_iter:
            if self.count%self.loss_increase_iter == 0:
                if self.loss_end_iter != 0:
                    self.supervision_weight = self.loss_weight()
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

            # import pdb; pdb.set_trace()
            features_supervised = features[supervised_indices]#[:len(unsupervised_indices)]

            features_not_supervised = features[unsupervised_indices]

            pred_not_supervised = pred[unsupervised_indices]

            target_supervised = target[supervised_indices]#[:len(unsupervised_indices)]
            # import pdb; pdb.set_trace()
            reference_loss, pseudolabel, distance_map, entropy_map = \
                    self.reference_loss(features_not_supervised, pred_not_supervised, features_supervised, target_supervised)
                    # self.reference_loss(features_supervised, pred_not_supervised, features_not_supervised, target_supervised)



            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] =(supervised_outdict[k]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            fin_outdict['loss'] += (reference_loss *  self.supervision_weight).mean()

            # _, ps_weight = self.get_pseudolabels(F.interpolate(pred_not_supervised, self.loss_size))
            # fin_outdict['loss'] += ((reference_loss*ps_weight).sum()/(ps_weight.sum()+EPS)) *  self.supervision_weight

            # fin_outdict['loss'] += (reference_loss *  0).mean()
            fin_outdict['reference_loss'] = reference_loss.mean()

            fin_outdict['mask_target'] =  x['mask_target']
            if self.cf.seg_mode == 'binary':
                fin_outdict['pseudolabel_unlabeled'] = F.interpolate(pseudolabel,target_supervised.shape[-1],mode='nearest')
                fin_outdict['pseudolabel_labeled'] = F.interpolate(target_supervised,target_supervised.shape[-1],mode='nearest')
                fin_outdict['mask_target'][unsupervised_indices] = F.interpolate(pseudolabel,target_supervised.shape[-1],mode='nearest')
            else:
                fin_outdict['mask_target'][unsupervised_indices] =  F.interpolate(pseudolabel.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)
                fin_outdict['pseudolabel_unlabeled'] =  F.interpolate(pseudolabel.float().unsqueeze(1),target_supervised.shape[-1],mode='nearest').squeeze(1)
                fin_outdict['pseudolabel_labeled'] = F.interpolate(target_supervised.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)

            fin_outdict['distance_map'] =F.interpolate(distance_map,target_supervised.shape[-1],mode='nearest')
            fin_outdict['entropy_map'] = F.interpolate(entropy_map,target_supervised.shape[-1],mode='nearest')

            return fin_outdict

            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class distr_aug_reference_segmentation_loss(segmentation_losses):
    def __init__(self, cf):
        super(distr_aug_reference_segmentation_loss, self).__init__(cf)
        self.threshold = cf.score_threshold
        kernel_size = int(self.cf.partial_losses[-1])
        full_batch =  'full_batch' in self.cf.partial_losses
        ce   = 'ce' in self.cf.partial_losses
        dice = 'dice' in self.cf.partial_losses
        self.loss_size = self.cf.loss_size
        self.softmax  = not self.cf.seg_mode == 'binary'

        if self.cf.classes>10:
            self.reference_loss = seg_loss.distr_reference_highclass_loss(loss_size = self.loss_size,
                                                        k=self.cf.k,
                                                        use_morph=self.cf.use_morph,
                                                        use_refine=self.cf.use_refine,
                                                        mode=self.cf.update_mode,
                                                        use_softmax=False,
                                                        tau=self.cf.tau,
                                                        full_batch = full_batch,
                                                        use_cx = self.softmax,
                                                        use_dice= dice)
        else:
            self.reference_loss = seg_loss.distr_reference_loss(loss_size = self.loss_size,
                                                        k=self.cf.k,
                                                        use_morph=self.cf.use_morph,
                                                        use_refine=self.cf.use_refine,
                                                        mode=self.cf.update_mode,
                                                        use_softmax=False,
                                                        tau=self.cf.tau,
                                                        full_batch = full_batch,
                                                        use_cx = self.softmax,
                                                        use_dice= dice)


        self.loss_start_iter = self.cf.loss_start_iter
        self.loss_end_iter = self.cf.loss_end_iter
        self.count = 1
        self.loss_increase_iter = self.cf.loss_increase_iter
        if  self.loss_increase_iter == 0:
            self.supervision_weight = 1
            self.loss_increase_iter = 1e12


    def loss_weight(self,):
        if self.count >= self.loss_end_iter:
            return 1
        rampup_length = self.loss_end_iter-self.loss_start_iter
        current = np.clip(self.count, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def aug(self, key, value, img):
        def cutout(img, area):
            if img.shape.__len__() == 3:
                img[:, area[1]:area[3], area[0]:area[2]] = 0
            else:
                img[ area[1]:area[3],area[0]:area[2]] = 0
            return img

        def shearX(img,v):
            return t.functional.affine(img, angle=0,translate=[0,0], scale=1.,shear=[math.degrees(math.atan(v)),0])

        def shearY(img,v):
            return t.functional.affine(img, angle=0,translate=[0,0], scale=1.,shear=[0,math.degrees(math.atan(v))])


        def translateX(img,v):
            return t.functional.affine(img, angle=0,translate=[-v,0], scale=1.,shear=[0,0])
        def translateY(img,v):
            return t.functional.affine(img, angle=0,translate=[0,-v], scale=1.,shear=[0,0])

        def rotate(img,v):
            return t.functional.affine(img, angle=-v,translate=[0,0], scale=1.,shear=[0,0])

        def identity(img,v):
            return img

        aug_dict = {
            'Cutout':cutout,
            'Rotate':rotate,
            'ShearX':shearX,
            'ShearY':shearY,
            'TranslateX':translateX,
            'TranslateY':translateY,
        }

        return aug_dict.get(key,identity)(img, value)

    def aug_pseudolabels(self, aug_list, pseudolabels):
        for i in range(len(pseudolabels)):
            tmp_p = pseudolabels[i]
            for j in aug_list[i]:
                tmp_p = self.aug(j[0],j[1],tmp_p)
            pseudolabels[i] = tmp_p
        return pseudolabels

    @torch.no_grad()
    def get_pseudolabels(self,pred):
        if self.cf.seg_mode == 'binary':
            target = torch.zeros(pred.shape).to(pred.device)
            weight = torch.zeros(pred.shape).to(pred.device)

            target[pred.sigmoid()>self.threshold] = 1.
            target[pred.sigmoid()<(1-self.threshold)] = 0.
            weight[pred.sigmoid()>self.threshold] = 1.
            weight[pred.sigmoid()<(1-self.threshold)] = 1.
        else:
            target = torch.zeros(list(torch.tensor(pred.shape)[[0,2,3]])).long().to(pred.device)
            weight = torch.zeros(list(torch.tensor(pred.shape)[[0,2,3]])).to(pred.device)

            target[pred.softmax(1).max(1)[0]>self.threshold] = pred.max(1)[1].long()[pred.softmax(1).max(1)[0]>self.threshold]
            weight[pred.softmax(1).max(1)[0]>self.threshold] = 1.

        return target.detach(), weight.detach()

    def forward(self, x):

        self.count += 1
        pred = x['segmentation_logits']
        features = x['segmentation_features']
        target = self.check_to_unsqueeze(x['mask_target'])
        supervision_type = x['supervision_type'].to(pred.device)
        if self.cf.seg_mode =='binary':
            target = target.float()
        else:
            target = target.long()



        target_, weight = self.get_pseudolabels(pred)

        supervision = supervision_type.view(-1,*[1 for i in range(len(weight.shape)-1)])*torch.ones(weight.shape).to(pred.device)

        supervised_indices = ((supervision_type==1)* (torch.tensor(range(len(supervision_type)))+1).cuda()).unique().long()
        supervised_indices = supervised_indices[supervised_indices>0]-1

        unsupervised_indices = ((supervision_type==0)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        unsupervised_indices = unsupervised_indices[unsupervised_indices>0]-1

        aug_unsupervised_indices = ((supervision_type==2)* (torch.tensor(range(len(supervision_type))).cuda()+1)).unique().long()
        aug_unsupervised_indices = aug_unsupervised_indices[aug_unsupervised_indices>0]-1
        aug_unsupervised_indices = aug_unsupervised_indices[unsupervised_indices]

        weight = torch.max(weight.cuda(), supervision.cuda())

        target[supervised_indices] = target[supervised_indices]
        target[unsupervised_indices] = target_[unsupervised_indices]

        supervised_outdict = self.run_losses(pred[supervised_indices], target[supervised_indices].detach(), weight[supervised_indices].detach().cuda())

        if len(unsupervised_indices)>0 and self.count > self.loss_start_iter:
            if self.count%self.loss_increase_iter == 0:
                if self.loss_end_iter != 0:
                    self.supervision_weight = self.loss_weight()
                else:
                    self.supervision_weight = torch.clamp(torch.tensor(self.supervision_weight+0.2),0, 1)

            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] =(supervised_outdict[k]).mean()

            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()

            features_supervised = features[supervised_indices]

            features_not_supervised = features[unsupervised_indices]

            pred_not_supervised = pred[unsupervised_indices]

            target_supervised = target[supervised_indices]

            # import pdb; pdb.set_trace()

            reference_loss, pseudolabel, distance_map, entropy_map = \
                    self.reference_loss(features_not_supervised, pred_not_supervised, features_supervised, target_supervised)


            if self.cf.aug_loss_var == 'both':
                with torch.no_grad():
                    aug_pseudolabels = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], pseudolabel.clone())
                    aug_entropy = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], entropy_map.clone())

                p = F.interpolate(pred[aug_unsupervised_indices],self.loss_size,mode='nearest')
                a_pl = F.interpolate(aug_pseudolabels,self.loss_size,mode='nearest')
                a_ent = F.interpolate(aug_entropy,self.loss_size,mode='nearest')

                if not self.cf.seg_mode =='binary':
                    t = t.argmax(1)
                    aug_reference_loss = torch.nn.functional.cross_entropy(p,\
                                                                            a_pl, reduction='none')
                else:
                    aug_reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,\
                                                                                            a_pl, reduction='none')

                aug_reference_loss = aug_reference_loss*(1-a_ent)

                fin_outdict['loss'] +=  (aug_reference_loss *  self.supervision_weight).mean() + (reference_loss*  self.supervision_weight).mean()
            elif self.cf.aug_loss_var == 'only':
                with torch.no_grad():
                    aug_pseudolabels = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], pseudolabel.clone())
                    aug_entropy = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], entropy_map.clone())

                p = F.interpolate(pred[aug_unsupervised_indices],self.loss_size,mode='nearest')
                a_pl = aug_pseudolabels
                a_ent = aug_entropy
                if not self.cf.seg_mode =='binary':
                    t = t.argmax(1)
                    aug_reference_loss = torch.nn.functional.cross_entropy(p,\
                                                                            a_pl, reduction='none')
                else:
                    aug_reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,\
                                                                                            a_pl, reduction='none')

                aug_reference_loss = aug_reference_loss*(1-a_ent)

                fin_outdict['loss'] +=  (aug_reference_loss *  self.supervision_weight).mean()
            elif self.cf.aug_loss_var == 'fixmatch':
                pseudo_labels = target[unsupervised_indices].detach()
                target_supervised = target[supervised_indices]
                with torch.no_grad():

                    if not self.cf.seg_mode =='binary':
                        pseudo_labels = F.one_hot(pseudo_labels).permute(0,-1,1,2)
                        weight = weight.unsqueeze(1)
                    aug_pseudolabels = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], pseudo_labels.clone())
                    aug_entropy = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], weight[unsupervised_indices].detach().cuda().clone())


                if not self.cf.seg_mode =='binary':
                    aug_pseudolabels = aug_pseudolabels.argmax(1)
                    aug_reference_loss = torch.nn.functional.cross_entropy(pred[aug_unsupervised_indices],\
                                                                            aug_pseudolabels, reduction='none')
                else:
                    aug_reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[aug_unsupervised_indices],\
                                                                                            aug_pseudolabels, reduction='none')

                aug_reference_loss = (aug_reference_loss*aug_entropy).sum()/aug_entropy.sum()
                fin_outdict['loss'] += (aug_reference_loss *  self.supervision_weight).mean()
                aug_entropy = aug_entropy.mean(1,keepdim=True)
            elif self.cf.aug_loss_var == 'fixmatch_both':
                pseudo_labels = target[unsupervised_indices].detach()
                target_supervised = target[supervised_indices]

                with torch.no_grad():
                    if not self.cf.seg_mode =='binary':
                        pseudo_labels = F.one_hot(pseudo_labels).permute(0,-1,1,2)
                        weight = weight.unsqueeze(1)
                    aug_pseudolabels = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], pseudo_labels.clone())
                    aug_entropy = self.aug_pseudolabels([x['aug_list'][i] for i in unsupervised_indices], weight[unsupervised_indices].detach().cuda().clone())


                if not self.cf.seg_mode =='binary':
                    aug_pseudolabels = aug_pseudolabels.argmax(1)
                    aug_reference_loss = torch.nn.functional.cross_entropy(pred[aug_unsupervised_indices],\
                                                                            aug_pseudolabels, reduction='none')
                    aug_reference_loss = (aug_reference_loss*aug_entropy.squeeze(1)).sum()/aug_entropy.squeeze(1).sum()
                else:
                    aug_reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[aug_unsupervised_indices],\
                                                                                            aug_pseudolabels, reduction='none')
                    aug_reference_loss = (aug_reference_loss*aug_entropy).sum()/aug_entropy.sum()
                fin_outdict['loss'] += (aug_reference_loss *  self.supervision_weight).mean() + (reference_loss * self.supervision_weight).mean()
                aug_entropy = aug_entropy.mean(1,keepdim=True)



            fin_outdict['aug_reference_loss'] = aug_reference_loss.mean()
            fin_outdict['reference_loss'] = reference_loss.mean()
            fin_outdict['mask_target'] =  x['mask_target']

            if self.cf.seg_mode == 'binary':
                fin_outdict['pseudolabel_unlabeled'] = F.interpolate(pseudolabel,target_supervised.shape[-1],mode='nearest')
                fin_outdict['pseudolabel_labeled'] = F.interpolate(target_supervised,target_supervised.shape[-1],mode='nearest')
                fin_outdict['pseudolabel_aug_unlabeled'] = F.interpolate(aug_pseudolabels,target_supervised.shape[-1],mode='nearest')
                fin_outdict['mask_target'][aug_unsupervised_indices] = F.interpolate(aug_pseudolabels,target_supervised.shape[-1],mode='nearest')

                fin_outdict['mask_target'][unsupervised_indices] = F.interpolate(pseudolabel,target_supervised.shape[-1],mode='nearest')
            else:
                fin_outdict['mask_target'][unsupervised_indices] =  F.interpolate(pseudolabel.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)
                fin_outdict['mask_target'][aug_unsupervised_indices] = F.interpolate(aug_pseudolabels.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)

                fin_outdict['pseudolabel_unlabeled'] =  F.interpolate(pseudolabel.float().unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)
                fin_outdict['pseudolabel_labeled'] = F.interpolate(target_supervised.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)
                fin_outdict['pseudolabel_aug_unlabeled'] = F.interpolate(aug_pseudolabels.unsqueeze(1).float(),target_supervised.shape[-1],mode='nearest').squeeze(1)

            fin_outdict['distance_map'] =F.interpolate(distance_map,target_supervised.shape[-1],mode='nearest')
            fin_outdict['entropy_map'] = F.interpolate(entropy_map,target_supervised.shape[-1],mode='nearest')
            fin_outdict['augentropy_map'] = F.interpolate(aug_entropy,target_supervised.shape[-1],mode='nearest')
            return fin_outdict
        else:
            fin_outdict = {}
            for k in supervised_outdict.keys():
                fin_outdict[k] = supervised_outdict[k].mean()
            fin_outdict['loss'] = torch.stack([fin_outdict[k] for k in fin_outdict.keys()]).sum()
            return fin_outdict

class class_losses(nn.Module):
    def __init__(self,cf):
        super(class_losses,self).__init__()
        self.cf = cf
        self.cl_loss = cl_loss.ChestBinaryCrossEntropyLoss()# nn.BCEWithLogitsLoss(reduction='none') #

    def forward(self,x):
        class_logits = x['class_logits']
        class_target = x['class_target'].float()

        results_dict = {}
        cl_loss = self.cl_loss(class_logits, class_target).sum(1).mean()
        loss = cl_loss
        results_dict['loss_class'] = cl_loss
        results_dict['loss'] = loss

        return results_dict


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    dev = cls_prob.device
    gt_boxes = torch.zeros((0, 4), dtype=boxes.dtype, device=dev)
    gt_classes = torch.zeros((0, 1), dtype=torch.long, device=dev)
    gt_scores = torch.zeros((0, 1), dtype=cls_prob.dtype, device=dev)
    for i in im_labels.nonzero()[:,1]:
        cls_prob_tmp = cls_prob[:, i]
        idxs = (cls_prob_tmp >= 0).nonzero()[:,0]
        idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
        idxs = idxs[idxs_tmp]
        boxes_tmp = boxes[idxs, :]
        cls_prob_tmp = cls_prob_tmp[idxs]

        graph = (ops.box_iou(boxes_tmp, boxes_tmp) > 0.4).float()

        keep_idxs = []
        gt_scores_tmp = []
        count = cls_prob_tmp.size(0)
        while True:
            order = graph.sum(dim=1).argsort(descending=True)
            tmp = order[0]
            keep_idxs.append(tmp)
            inds = (graph[tmp, :] > 0).nonzero()[:,0]
            gt_scores_tmp.append(cls_prob_tmp[inds].max())

            graph[:, inds] = 0
            graph[inds, :] = 0
            count = count - len(inds)
            if count <= 5:
                break

        gt_boxes_tmp = boxes_tmp[keep_idxs, :].view(-1, 4).to(dev)
        gt_scores_tmp = torch.tensor(gt_scores_tmp, device=dev)

        keep_idxs_new = torch.from_numpy((gt_scores_tmp.argsort().to('cpu').numpy()[-1:(-1 - min(len(gt_scores_tmp), 5)):-1]).copy()).to(dev)

        gt_boxes = torch.cat((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
        gt_scores = torch.cat((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
        gt_classes = torch.cat((gt_classes, (i + 1) * torch.ones((len(keep_idxs_new), 1), dtype=torch.long, device=dev)))

        # If a proposal is chosen as a cluster center,
        # we simply delete a proposal from the candidata proposal pool,
        # because we found that the results of different strategies are similar and this strategy is more efficient
        another_tmp = idxs.to('cpu')[torch.tensor(keep_idxs)][keep_idxs_new.to('cpu')].numpy()
        cls_prob = torch.from_numpy(np.delete(cls_prob.to('cpu').numpy(), another_tmp, axis=0)).to(dev)
        boxes = torch.from_numpy(np.delete(boxes.to('cpu').numpy(), another_tmp, axis=0)).to(dev)

    proposals = {'gt_boxes' : gt_boxes.to(dev),
                 'gt_classes': gt_classes.to(dev),
                 'gt_scores': gt_scores.to(dev)}

    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = ops.box_iou(all_rois.to(gt_boxes.device), gt_boxes)
    max_overlaps, gt_assignment = overlaps.max(dim=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = (max_overlaps >= 0.5).nonzero()[:,0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = (max_overlaps < 0.5).nonzero()[:,0]

    ig_inds = (max_overlaps < 0.1).nonzero()[:,0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = torch.zeros(gt_boxes.shape[0], dtype=cls_prob.dtype, device=cls_prob.device)
    pc_probs = torch.zeros(gt_boxes.shape[0], dtype=cls_prob.dtype, device=cls_prob.device)
    pc_labels = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=cls_prob.device)
    pc_count = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=cls_prob.device)

    for i in range(gt_boxes.shape[0]):
        po_index = (gt_assignment == i).nonzero()[:,0]
        img_cls_loss_weights[i] = torch.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = (cls_prob[po_index, pc_labels[i]]).mean()

    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights


@torch.no_grad()
def pcl_label(boxes:torch.Tensor, cls_prob:torch.Tensor, im_labels:torch.Tensor, cls_prob_new:torch.Tensor):
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    cls_prob = cls_prob.clamp(EPS, 1-EPS)
    cls_prob_new = cls_prob_new.clamp(EPS, 1-EPS)

    proposals = _get_graph_centers(boxes, cls_prob, im_labels)

    labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes,
            proposals, im_labels, cls_prob_new)

    return {'labels' : labels.reshape(1, -1),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1),
            'gt_assignment' : gt_assignment.reshape(1, -1),
            'pc_labels' : pc_labels.reshape(1, -1),
            'pc_probs' : pc_probs.reshape(1, -1),
            'pc_count' : pc_count.reshape(1, -1),
            'img_cls_loss_weights' : img_cls_loss_weights.reshape(1, -1),
            'im_labels_real' : torch.cat((torch.tensor([[1.]], dtype=im_labels.dtype, device=im_labels.device), im_labels), dim=1)}


class PCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pcl_probs, labels, cls_loss_weights,
                gt_assignment, pc_labels, pc_probs, pc_count,
                img_cls_loss_weights, im_labels):
        ctx.pcl_probs = pcl_probs
        ctx.labels = labels
        ctx.cls_loss_weights = cls_loss_weights
        ctx.gt_assignment = gt_assignment
        ctx.pc_labels = pc_labels
        ctx.pc_probs = pc_probs
        ctx.pc_count = pc_count
        ctx.img_cls_loss_weights = img_cls_loss_weights
        ctx.im_labels = im_labels

        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_loss_weights,
                                    gt_assignment, pc_labels, pc_probs,
                                    pc_count, img_cls_loss_weights, im_labels)

        for c in im_labels.nonzero()[:,1]:
            if c == 0:
                i = (labels[0,:] == 0).nonzero()[:,0]
                loss -= (cls_loss_weights[0, i] * pcl_probs[i,c].log()).sum()
            else:
                i = (pc_labels[0,:] == c).nonzero()[:,0]
                loss -= (img_cls_loss_weights[0, i] * pc_probs[0,i].log()).sum()

        return loss / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        pcl_probs = ctx.pcl_probs
        labels = ctx.labels
        cls_loss_weights = ctx.cls_loss_weights
        gt_assignment = ctx.gt_assignment
        pc_labels = ctx.pc_labels
        pc_probs = ctx.pc_probs
        pc_count = ctx.pc_count
        img_cls_loss_weights = ctx.img_cls_loss_weights
        im_labels = ctx.im_labels

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        batch_size, channels = pcl_probs.size()

        for c in im_labels.nonzero()[:,1]:
            i = (labels[0] == c)
            if c == 0:
                grad_input[i, c] = -cls_loss_weights[0, i] / pcl_probs[i, c]
            else:
                pc_index = gt_assignment[0, i]
                if (c != pc_labels[0, pc_index]).all():
                    print('labels mismatch.')
                grad_input[i, c] = -img_cls_loss_weights[0, pc_index] / (pc_count[0, pc_index] * pc_probs[0, pc_index])

        grad_input /= batch_size
        return grad_input, None, None, None, None, None, None, None, None
