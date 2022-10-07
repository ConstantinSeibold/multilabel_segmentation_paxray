import cv2 as cv
import numpy as np

import torch,pdb
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import einsum

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import time
# from misc.morphological_operations import *

class reference_loss(nn.Module):
    def __init__(self, kernel_size, full_batch, reduction='mean', use_dice=True, use_cx=True):
        super(reference_loss, self).__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.full_batch = full_batch
        self.use_cx = use_cx
        self.use_dice = use_dice

    def score_min(self, x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.min(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def score_max(self, x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.max(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def expand_map(self, x, return_unflattened=False):
        ks = self.kernel_size

        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w*ks, h*ks

        range_list_w  = [list(range(i,i+ks)) for i in range(w)]
        range_list_h  = [list(range(i,i+ks)) for i in range(h)]

        expand_h      = x1[:, :, range_list_h, :]

        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)
        expand_h_view = expand_h.view(b,c,expected_h,padded_w)
        expand_w      = expand_h_view[:,:, :, range_list_w].view(b,c,expected_h,expected_w)

        if not return_unflattened:
            return expand_w
        else:
            return expand_w, expand_w_

    def prep_feats(self, feat):
        ks = self.kernel_size
        b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        _,feat_expanded = self.expand_map(feat, True)
        feat_expanded   = feat_expanded.flatten(-2)
        feat_expanded   = feat_expanded.view(b,c,w*h,ks*ks).permute(0,2,3,1)
        return feat_reshaped, feat_expanded

    def prep_feats_fullbatch(self, feat):
        ks = self.kernel_size
        b,c,w,h = feat.shape
        feat_reshaped   = feat.unsqueeze(0).permute(0,2,1,3,4).contiguous().view(1, c, b*w*h).permute(0,2,1)
        _,feat_expanded = self.expand_map(feat,True)
        feat_expanded   = feat_expanded.flatten(-2)
        feat_expanded   = feat_expanded.unsqueeze(0).permute(0,2,1,3,4,5).contiguous().view(1, c, b*w*h,ks*ks).permute(0,2,3,1)
        return feat_reshaped, feat_expanded

    def bin_dice(self, predict,target,reduction='none'):
        b,c,hw,ks2 = predict.shape
        predict = predict.permute(0,2,1,3)
        predict = predict.contiguous().view(b*hw, c*ks2)

        predict = predict.sigmoid()

        target = target.permute(0,2,1,3)
        target = target.contiguous().view(b*hw, c*ks2)

        #         predict = predict.contiguous().view(predict.shape[0],-1)
        #         target = target.contiguous().view(target.shape[0],-1)
        p=2
        smooth = 1
        num = torch.sum(torch.mul(predict, target), dim=1) + smooth
        den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth
        loss = 1 - num / den
        if reduction == 'mean':
            loss = loss.mean()
        return loss

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled):
        if self.full_batch:
            feat_un, feat_un_prep       = self.prep_feats_fullbatch(features_unlabeled)
            feat_lab, feat_lab_prep     = self.prep_feats_fullbatch(features_labeled)
            target_lab, target_lab_prep = self.prep_feats_fullbatch(target_labeled)
        else:
            feat_un, feat_un_prep       = self.prep_feats(features_unlabeled)
            feat_lab, feat_lab_prep     = self.prep_feats(features_labeled)
            target_lab, target_lab_prep = self.prep_feats(target_labeled)

        init_distances = torch.cdist(feat_un,feat_lab)

        min_indx = init_distances.argmin(-1)

        target_lab_rearranged        = []
        features_lab_rearranged      = []
        target_lab_prep_rearranged   = []
        features_lab_prep_rearranged = []


        for i in range(min_indx.shape[0]):
            target_lab_rearranged        += [target_lab[i,min_indx[i]]]
            features_lab_rearranged      += [feat_lab[i,min_indx[i]]]
            target_lab_prep_rearranged   += [target_lab_prep[i,min_indx[i]]]
            features_lab_prep_rearranged += [feat_lab_prep[i,min_indx[i]]]

        target_lab_rearranged        = torch.stack(target_lab_rearranged,0)
        features_lab_rearranged      = torch.stack(features_lab_rearranged,0)
        target_lab_prep_rearranged   = torch.stack(target_lab_prep_rearranged,0)
        features_lab_prep_rearranged = torch.stack(features_lab_prep_rearranged,0)

        feat_dist = torch.cdist(feat_un_prep,features_lab_prep_rearranged)
        min_indx = feat_dist.argmin(-1)

        final_target = []
        for i in range(min_indx.shape[0]):
            okee = []
            for j in range(target_lab_prep_rearranged.shape[1]):
                okee += [target_lab_prep_rearranged[i,j,min_indx[i,j],:]]
            okee = torch.stack(okee,0)
            final_target += [okee]
        final_target = torch.stack(final_target,0)

        return final_target.permute(0,3,1,2).float()

    def prepare_predictions(self, pred_unlabeled):
        if self.full_batch:
            pred_un, pred_un_prep = self.prep_feats_fullbatch(pred_unlabeled)
        else:
            pred_un, pred_un_prep = self.prep_feats(pred_unlabeled)
        return pred_un_prep.permute(0,3,1,2)

    @torch.no_grad()
    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            # print('iteration',i,first.shape, omg.shape)
        return omg

    def forward(self, features_unlabeled, pred_unlabeled, features_labeled, target_labeled):
        with torch.no_grad():
            target = self.prepare_label(features_unlabeled,features_labeled,target_labeled)

        pred   = self.prepare_predictions(pred_unlabeled)

        loss = []

        if self.use_cx:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,target, reduction='mean')
            loss += [base_loss]
        if self.use_dice:
            dice_loss = self.bin_dice(pred,target,'mean')
            loss += [dice_loss]

        loss = torch.stack(loss,0).sum(0)
        return loss, self.refactor_target_to_image(target)

class multiscale_reference_loss(nn.Module):
    def __init__(self, scales, expansions, fullbatch, reduction='mean', softmax=False, use_dice=True, use_cx=True, label_refinement_radius=5):
        super(multiscale_reference_loss, self).__init__()
        self.scales = scales
        self.expansions = expansions

        self.reduction = reduction
        self.fullbatch = fullbatch
        self.softmax = softmax
        self.use_cx = use_cx
        self.use_dice = use_dice
        self.mode_pool = ModePool2d(label_refinement_radius, same=True)
        self.label_refinement_radius= label_refinement_radius

    def arrange_all(self, feat_list, indices, current_scale, all_scales, fullbatch=False):
        out_feat_list = []
        for i in range(0, current_scale):
            out_feat_list += [feat_list[i]]

        for i in range(current_scale, len(all_scales)):
            if fullbatch and i == 0:
                tmp = []
                for j in range(indices.shape[0]):
                    tmp += [self.arrange_map(feat_list[i],indices[j:j+1], list(indices[j:j+1].shape))]
                tmp = torch.cat(tmp,0)
            else:
                tmp = self.arrange_map(feat_list[i],indices, list(indices.shape))
            out_feat_list += [tmp]
        return out_feat_list

    def arrange_map(self, feat, indx, dimlist=[]):
        if len(dimlist) == 1:
            feat = feat[indx]
        else:
            okee = []
            for i in range(dimlist[0]):
                okee += [self.arrange_map(feat[i], indx[i], dimlist[1:])]
            feat = torch.stack(okee,0)
        return feat

    def get_level(self, feat_labeled_coarse, feat_unlabeled_coarse, use_cosine = True):
        def cosine(aa,bb):
            a_norm = aa/aa.norm(dim=-1, keepdim=True)
            b_norm = bb/bb.norm(dim=-1, keepdim=True)
            mm = torch.matmul(a_norm,b_norm.permute(*range(len(bb.shape)-2),-1,-2))
            return 1-mm
        # feat_unlabeled_coarse = feat_unlabeled_coarse/feat_unlabeled_coarse.norm(dim=-1)
        # feat_labeled_coarse  = feat_labeled_coarse/feat_labeled_coarse.norm(dim=-1)
        # pdb.set_trace()
        if cosine:
            init_distances = cosine(feat_unlabeled_coarse, feat_labeled_coarse)
        else:
            init_distances = torch.cdist(feat_unlabeled_coarse, feat_labeled_coarse)

        min_indx = init_distances.argmin(-1)
        return min_indx

    def expand_map(self, x, ks, stride=1, return_unflattened=False):
        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w//stride*(ks//2*2+stride), h//stride*(ks//2*2+stride)

        range_list_w  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,w, stride)]
        range_list_h  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,h, stride)]

        expand_h      = x1[:, :, range_list_h, :]
        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)
        expand_h_view = expand_h.view(b,c,expected_h,padded_w)
        expand_w      = expand_h_view[:,:, :, range_list_w].view(b,c,expected_h,expected_w)

        if not return_unflattened:
            return expand_w
        else:
            return expand_w, expand_w_

    def prep_all_feats(self, feat_list, scales, base=2, fullbatch=False):
        out = []
        for i in range(len(scales)):
            if type(base) == list:
                base_ = base[i]
            else:
                base_ = base
            out += [self.prep_larger_scale(feat_list[i], i, base_, fullbatch and i==0)]

        return out

    def prep_larger_scale(self, feat, scale, base=2, fullbatch=False):
        tmp_feat = feat.clone()
        for i in range(scale):
            w,h = tmp_feat.shape[-2:]
            if  type(base)==list:
                cur_base = base[i]
            else:
                cur_base = base
            split_size = w//(cur_base**(0+1))
            tmp_feat = torch.cat(torch.stack(tmp_feat.split(split_size,dim=-1),dim=-3).split(split_size,dim=-2),dim=-3)
        tmp_feat = tmp_feat.flatten(-2)
        tmp_feat = tmp_feat.permute(*[0,*list(range(2,len(tmp_feat.shape[2:])+2)),1])
        if fullbatch:
            tmp_feat = torch.cat(tmp_feat.split(1,0),1)
        return tmp_feat

    def hierarchical_map_expansion(self, feat, size_list, expansion_list):
        tmp = feat
        f = len(size_list)-1
        current_stride = 1
        out_split_list = []
        for i in range(len(size_list)):
            walking_stride = int(np.prod(out_split_list))
            kernel_radius = expansion_list[f-i]*walking_stride
            current_stride = max(size_list[f-i]* walking_stride,current_stride)
            tmp = self.expand_map(tmp,1+kernel_radius*2, max(current_stride,1))
            out_split_list.insert(0,size_list[f-i]+expansion_list[f-i]*2)
        return tmp, out_split_list

    def get_multiscale_features(self, feat, scales, expansions, interpolation_mode='nearest', is_label=False):
        multi_scale_features = []
        tmp_feat = feat
        split_lists = []
        for i in range(len(scales)-1,-1,-1):
            if i == len(scales)-1:
                tmp_feat = F.interpolate(tmp_feat,size=(int(np.prod(scales[:i+1])),int(np.prod(scales[:i+1]))),mode='nearest')
            else:
                if is_label:
                    tmp_feat = F.adaptive_max_pool2d(tmp_feat,int(np.prod(scales[:i+1])))
                else:
                    if interpolation_mode == 'avg_pool':
                        tmp_feat = F.adaptive_avg_pool2d(tmp_feat,int(np.prod(scales[:i+1])))
                    else:
                        tmp_feat = F.interpolate(tmp_feat,size=(int(np.prod(scales[:i+1])),int(np.prod(scales[:i+1]))),mode=interpolation_mode)
                # tmp_feat = F.interpolate(tmp_feat,size=(int(np.prod(scales[:i+1])),int(np.prod(scales[:i+1]))),mode=interpolation_mode)
            tmp, tmp_split_list = self.hierarchical_map_expansion(tmp_feat, scales[:i+1], expansions[:i+1])
            multi_scale_features += [tmp]
            split_lists += [tmp_split_list]
        multi_scale_features = multi_scale_features[::-1]
        split_lists = split_lists[::-1]

        # multi_scale_features = []
        # split_lists = []
        # for i in range(len(scales)):
        #     # if is_label:
        #     #     tmp = F.adaptive_max_pool2d(feat,int(np.prod(scales[:i+1])))
        #     # else:
        #     #     # if interpolation_mode == 'avg_pool':
        #     #     #     tmp = F.adaptive_avg_pool2d(feat,int(np.prod(scales[:i+1])))
        #     #     # else:
        #     tmp = F.interpolate(feat,size=(int(np.prod(scales[:i+1])),int(np.prod(scales[:i+1]))),mode=interpolation_mode)
        #     tmp, tmp_split_list = self.hierarchical_map_expansion(tmp, scales[:i+1], expansions[:i+1])
        #     multi_scale_features += [tmp]
        #     split_lists += [tmp_split_list]
        return multi_scale_features, split_lists

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled, scales, expansions=None, fullbatch=False):
        feat_un, split_list_feat_un       = self.get_multiscale_features(features_unlabeled, scales, [0 for i in expansions],'avg_pool')
        feat_lab, split_list_feat_lab     = self.get_multiscale_features(features_labeled, scales, expansions,'avg_pool')
        target_lab, split_list_target_lab = self.get_multiscale_features(target_labeled, scales, expansions, is_label = True)
        feat_un_prepped    = self.prep_all_feats(feat_un, scales, split_list_feat_un)
        feat_lab_prepped   = self.prep_all_feats(feat_lab, scales, split_list_feat_lab,fullbatch)
        target_lab_prepped = self.prep_all_feats(target_lab, scales, split_list_target_lab,fullbatch)

        lol = None

        for i in range(len(scales)):
            indices = self.get_level(feat_lab_prepped[i], feat_un_prepped[i])

            feat_lab_prepped   = self.arrange_all(feat_lab_prepped, indices, i, range(len(scales)), fullbatch)
            target_lab_prepped = self.arrange_all(target_lab_prepped, indices, i, range(len(scales)), fullbatch)

        arranged_target = target_lab_prepped[-1]
        shape = range(len(arranged_target.shape))

        return arranged_target.permute(0,-1,*shape[1:-1]).float(), target_lab_prepped

    def prepare_predictions(self, pred_unlabeled,scales,expansions):
        preds, split_list = self.get_multiscale_features(pred_unlabeled, scales,[0 for i in expansions])
        pred_un = self.prep_larger_scale(preds[-1], len(scales)-1, split_list[-1])
        shape = range(len(pred_un.shape))
        return pred_un.permute(0,-1,*shape[1:-1]).float()

    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
        return omg

    def mode_filter(self, x, ks, stride = 1):
        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w//stride*(ks//2*2+stride), h//stride*(ks//2*2+stride)

        range_list_w  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,w, stride)]
        range_list_h  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,h, stride)]

        expand_h      = x1[:, :, range_list_h, :]
        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)

        img = expand_w_.flatten(-2)
        img = img.mode(-1)[0]
        return img

    def forward(self,features_unlabeled, pred_unlabeled, features_labeled, target_labeled):
        scales  = self.scales
        expansions = self.expansions
        fullbatch = self.fullbatch

        if self.softmax:
            if len(target_labeled.shape)==len(pred_unlabeled.shape):
                pass
            else:
                target_labeled = F.one_hot(target_labeled, num_classes=pred_unlabeled.shape[1]).permute(0,-1,1,2)
        else:
            if len(target_labeled.shape)==len(pred_unlabeled.shape):
                pass
            else:
                target_labeled = F.one_hot(target_labeled, num_classes=pred_unlabeled.shape[1]).permute(0,-1,1,2)

        with torch.no_grad():
            target, all_targets = self.prepare_label(features_unlabeled,features_labeled,target_labeled,\
                                                    scales,expansions, fullbatch)


        pred   = self.prepare_predictions(pred_unlabeled, scales, expansions)

        pred = self.refactor_target_to_image(pred)
        target = self.refactor_target_to_image(target)
        if self.label_refinement_radius>1:
            with torch.no_grad():
                target = self.mode_pool(target)

        if self.softmax:
            tmp = target
            labels = torch.tensor(range(target.shape[1])).view(1,-1,1,1)
            target = (tmp*labels).max(1)[0]
            base_loss = torch.nn.functional.cross_entropy(pred,target, reduction=self.reduction)
        else:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,target, reduction=self.reduction)
        loss = base_loss
        return loss, target


# from misc.finch import FINCH as finch
# from misc.kmeans import KMeans, MultiKMeans

class cluster_reference_loss(nn.Module):
    def __init__(self, clustersize,loss_size, full_batch, k,tau, use_softmax=False, reduction='mean', interpolation_mode='nearest', use_dice=True, use_cx=True):
        super(cluster_reference_loss, self).__init__()
        self.map_size = clustersize
        self.reduction = reduction
        self.loss_size = loss_size
        self.interpolation_mode = interpolation_mode
        self.full_batch = full_batch
        self.use_cx = use_cx
        self.use_dice = use_dice
        self.tau = tau
        self.k  = k
        self.use_softmax = use_softmax
        self.thresholdscale_count = 0

    @torch.no_grad()
    def cosine(self,aa,bb):
            aa = (aa - aa.mean())/aa.std()
            bb = (bb - bb.mean())/bb.std()

            a_norm = aa/aa.norm(dim=-1, keepdim=True)
            b_norm = bb/bb.norm(dim=-1, keepdim=True)
            mm = torch.matmul(a_norm,b_norm.permute(*range(len(bb.shape)-2),-1,-2))
            return 1-mm.relu()

    @torch.no_grad()
    def prep_feats(self, feat, is_feat=False):
        b,c,w,h = feat.shape

        if is_feat:
            aranged = torch.cat([torch.arange(w).view(1,1,1,-1) * torch.ones(b,1,h,w)
                                 ,torch.arange(h).view(1,1,-1,1) * torch.ones(b,1,h,w)],1)
            feat = torch.cat([feat,aranged],1)
            b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        return feat_reshaped

    @torch.no_grad()
    def prep_feats_fullbatch(self, feat, is_feat=False, is_unlabeled=False):
        b,c,w,h = feat.shape
        if is_feat:
            aranged = torch.cat([torch.arange(w).view(1,1,1,-1)* torch.ones(b,1,h,w)
                                 ,torch.arange(h).view(1,1,-1,1)* torch.ones(b,1,h,w)],1)
            feat = torch.cat([feat,aranged.to(feat.device)],1)
            b,c,w,h = feat.shape

        feat_reshaped   = feat.flatten(-2).permute(0,2,1)
        return feat_reshaped

    @torch.no_grad()
    def create_targets_finch(self, target_lab, feat_lab,d=3):
        if target_lab.shape.__len__() == 3:
            target_intermediate = (target_lab * torch.tensor([2**i for i in range(target_lab.shape[-1])]).view(1,1,-1).to(target_lab.device)).sum(-1)
        else:
            target_intermediate = target_lab

        targets = []
        labels  = []
        label_intermediate = []
        unique,unique2 = target_intermediate.unique(return_counts=True)
        cl_img = torch.zeros(feat_lab.shape[:2]).to(target_lab.device)
        count = 0
        for i in range(len(unique)):
            if unique2[i]>1:
                timekek = time()
                fin_results = finch(feat_lab[target_intermediate==unique[i]].cpu().numpy(), verbose=False)
                print('finch takes {} seconds'.format(time() - timekek))
#                 pdb.set_trace()
                print(fin_results[1])
                inner = fin_results[0]
                if inner.shape[-1]>d:
                    inner = inner[:,d]
                else:
                    inner = inner[:,-1]
                print(unique[i],inner.shape[-1],len(np.unique(inner)))
                tmp = torch.zeros(cl_img[target_intermediate==unique[i]].shape).to(target_lab.device)
                for k in np.unique(inner):
                    count += 1
                    tmp[inner == k] = count#inner[k]+(unique[i]+1)
                    targets.append(feat_lab[target_intermediate==unique[i]][inner == k].mean(0))
                    killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                    killme += [0 for j in range(target_lab.shape[-1]-len(killme))]
                    label_intermediate.append(unique[i])
                    labels.append(torch.tensor(killme))
                cl_img[target_intermediate==unique[i]] = tmp

            else:
                count += 1
                targets.append(feat_lab[target_intermediate==unique[i]].mean(0))
                label_intermediate.append(unique[i])
                killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                killme += [0 for j in range(target_lab.shape[-1]-len(killme))]

                labels.append(torch.tensor(killme))

        targets = torch.stack(targets,0).unsqueeze(0).to(target_lab.device)
        labels = torch.stack(labels,0).unsqueeze(0).to(target_lab.device)
        labels_intermediate = torch.stack(label_intermediate).to(target_lab.device)
        return targets, labels, labels_intermediate, cl_img

    @torch.no_grad()
    def create_targets_kmeans(self, target_lab, feat_lab,d=3):
        if target_lab.shape.__len__() == 3:
            target_intermediate = (target_lab * torch.tensor([2**i for i in range(target_lab.shape[-1])]).view(1,1,-1).to(target_lab.device)).sum(-1)
        else:
            target_intermediate = target_lab

        out_target, out_labels, out_labels_intermediate, out_cl_img = [],[],[],[]

        cluster_size = self.map_size

        c = 0
        for j in range(target_intermediate.shape[0]):
            feat_lab_inner = feat_lab[j]
            target_intermediate_inner = target_intermediate[j]
            target_lab_inner = target_lab[j]
            targets = []
            labels  = []
            label_intermediate = []
            unique,unique2 = target_intermediate_inner.unique(return_counts=True)
            cl_img = torch.zeros(feat_lab_inner.shape[:1]).to(target_lab.device)
            count = 0
            for i in range(len(unique)):
                if unique2[i]>cluster_size:
                    kmeans = KMeans(n_clusters=unique2[i]//cluster_size, mode='euclidean', verbose=0)
                    fin_results = kmeans.fit_predict(feat_lab_inner[target_intermediate_inner==unique[i]])
    #                 pdb.set_trace()
                    inner = fin_results

                    tmp = torch.zeros(cl_img[target_intermediate_inner==unique[i]].shape).to(target_lab.device)
                    for k in inner.unique():
                        count += 1
                        tmp[inner == k] = count#inner[k]+(unique[i]+1)
                        targets.append(feat_lab_inner[target_intermediate_inner==unique[i]][inner == k].mean(0))
                        killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                        # killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                        killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
                        # killme += [0 for j in range(target_lab_inner.shape[-1]-len(killme))]
                        label_intermediate.append(unique[i])
                        labels.append(torch.tensor(killme))
                    cl_img[target_intermediate_inner==unique[i]] = tmp

                else:
                    count += 1
                    targets.append(feat_lab_inner[target_intermediate_inner==unique[i]].mean(0))
                    label_intermediate.append(unique[i])
                    killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                    # killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                    killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
                    # killme += [0 for j in range(target_lab.shape[-1]-len(killme))]

                    labels.append(torch.tensor(killme))

            targets = torch.stack(targets,0).unsqueeze(0).to(target_lab.device)
            labels = torch.stack(labels,0).unsqueeze(0).to(target_lab.device)
            labels_intermediate = torch.stack(label_intermediate).to(target_lab.device)
#             return targets, labels, labels_intermediate, cl_img
            out_target.append(targets.cpu())
            out_labels.append(labels.cpu())
            out_labels_intermediate.append(labels_intermediate.cpu())
            c += labels_intermediate.unique().__len__()
            out_cl_img.append(cl_img )
            del targets
            del labels
            del labels_intermediate

        out_target = torch.cat(out_target,1).to(target_lab.device)
        out_labels = torch.cat(out_labels,1).to(target_lab.device)
        out_labels_intermediate = torch.cat(out_labels_intermediate,0).to(target_lab.device)
        out_cl_img = torch.stack(out_cl_img,0).to(target_lab.device)

        return out_target, out_labels, out_labels_intermediate, out_cl_img

    @torch.no_grad()
    def create_targets_kmeans_batched(self, feat_lab,d=3):

        cluster_size = self.map_size

        targets = []
        labels  = []
        label_intermediate = []
        cl_img = torch.zeros(feat_lab.shape[:2]).to(feat_lab.device)
        kmeans = MultiKMeans(n_clusters=feat_lab.shape[1]//cluster_size, n_kmeans=feat_lab.shape[0], mode='euclidean', verbose=0)
        inner = kmeans.fit_predict(feat_lab)


        for i in range(feat_lab.shape[1]//cluster_size):
            inner_target = []
            for j in range(feat_lab.shape[0]):
                inner_target.append(feat_lab[j][inner[j] == i].mean(0))
            targets.append(torch.stack(inner_target,0))
            cl_img[inner == i] = i
        targets = torch.stack(targets,1).to(feat_lab.device)
        return targets, cl_img

    @torch.no_grad()
    def create_targets_kmeans_batched2(self, feat_lab,d=3):

        cluster_size = self.map_size

        targets = []
        labels  = []
        label_intermediate = []
        cl_img = torch.zeros(feat_lab.shape[:2]).to(feat_lab.device)
#         kmeans = MultiKMeans(n_clusters=feat_lab.shape[1]//cluster_size, n_kmeans=feat_lab.shape[0], mode='euclidean', verbose=1)
        kmeans2 = KMeans(n_clusters=feat_lab.shape[1]//cluster_size, mode='euclidean', verbose=0)
        for j in range(feat_lab.shape[0]):

            inner = kmeans2.fit_predict(feat_lab[j])
            inner_target = []
            for i in range(feat_lab.shape[1]//cluster_size):
                inner_target.append(feat_lab[j][inner == i].mean(0))
                cl_img[j][inner == i] = i
            targets.append(torch.stack(inner_target,0))

        targets = torch.stack(targets,0).to(feat_lab.device)
        return targets, cl_img

    @torch.no_grad()
    def create_targets_classmean(self, target_lab, feat_lab):
        if target_lab.shape.__len__() == 3:
            target_intermediate = (target_lab * torch.tensor([2**i for i in range(target_lab.shape[-1])]).view(1,1,-1)).sum(-1)
        else:
            target_intermediate = target_lab

        targets = []
        labels  = []
        unique = target_intermediate.unique()
        for i in range(len(unique)):
            targets.append(feat_lab[target_intermediate==unique[i]].mean(0))
            killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
            killme += [0 for j in range(target_lab.shape[-1]-len(killme))]

            labels.append(torch.tensor(killme))
        targets = torch.stack(targets,0).unsqueeze(0)
        labels = torch.stack(labels,0).unsqueeze(0)
        return targets, labels, torch.zeros(feat_lab.shape[:2])

    @torch.no_grad()
    def create_targets_(self, target_lab, feat_lab):
        return feat_lab, target_lab, torch.zeros(feat_lab.shape[:2])

    # @torch.no_grad()
    # def f(self, dist_matrix, labels, labels_, k=1, tau=1):
    #     try:
    #         if k == 'max':
    #             k = dist_matrix.shape[-1]
    #
    #
    #         ret_dist, ret_ind = torch.topk(dist_matrix, k,dim=-1, largest=False)
    #
    # #         ret_dist= (ret_dist-ret_dist.min(-1, keepdim=True)[0])/(ret_dist.max(-1, keepdim=True)[0]-ret_dist.min(-1, keepdim=True)[0] + 1e-8)
    #
    #         ret_labels = labels[ret_ind]
    #         unique = labels.unique()
    #
    # #         bias = -((labels.unique(return_counts=True)[-1]+1e-8)/labels.numel()).log()
    # #         mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
    #
    # #         for i in range(len(unique)):
    # #             mod_tau[ret_labels==unique[i]] = (tau/bias[i])+1e-8
    #
    #         bias = (labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel())
    #         mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
    #         bias = torch.softmax(bias/tau,0)
    #
    #         for i in range(len(unique)):
    #             mod_tau[ret_labels==unique[i]] = (bias[i])+1e-8
    #
    # #         bias = ((labels.numel()/labels.unique(return_counts=True)[-1]+1e-8))
    # #         mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
    #
    # #         for i in range(len(unique)):
    # #             mod_tau[ret_labels==unique[i]] = 1/((tau*bias[i]).log()+1e-8)
    #
    #
    #
    # #         mod_tau = mod_tau*mod_kek
    #         if self.use_softmax:
    #             ret_dist_softmax = torch.softmax((1-ret_dist)/mod_tau,-1) #
    #         else:
    #             ret_dist_softmax = ((1-ret_dist)/mod_tau)/((1-ret_dist)/mod_tau).sum(-1, keepdim=True)
    #
    #         p_l_star = torch.zeros(*dist_matrix.shape[:2], len(unique)).to(dist_matrix.device)
    #
    #         ret_labels_ = []
    #         for i in range(len(unique)):
    #             p_l_star[:,:,i] = (ret_dist_softmax * (ret_labels == unique[i])).sum(-1)
    #
    #             killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
    #             killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
    #
    #             ret_labels_.append(torch.tensor(killme))
    #         ret_labels_ = torch.stack(ret_labels_,0).unsqueeze(0).to(dist_matrix.device)
    #     except Exception as e:
    #         import pdb; pdb.set_trace()
    #     return p_l_star, ret_labels_

    @torch.no_grad()
    def f(self, dist_matrix, labels, labels_, k=1, tau=1):

        def calc_prob(ret_dist,mod_tau, unique):
            if self.use_softmax:
                ret_dist_softmax = torch.softmax((1-ret_dist)/mod_tau,-1) #
            else:
                ret_dist_softmax = ((1-ret_dist)/mod_tau)/((1-ret_dist)/mod_tau).sum(-1, keepdim=True)

            p_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                p_l_star[:,:,i] = ((ret_dist_softmax * (ret_labels == unique[i])).sum(-1)/(ret_labels == unique[i]).sum(-1,keepdim=False))

            p_l_star = p_l_star/p_l_star.sum(-1,keepdim=True)
            return p_l_star

        def get_mod_tau(values):
            mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
            for i in range(len(unique)):
                mod_tau[ret_labels==unique[i]] = (values[i])+1e-8
            return mod_tau

        try:
            if k == 'max':
                k = dist_matrix.shape[-1]


            ret_dist, ret_ind = torch.topk(dist_matrix, k,dim=-1, largest=False)

    #         ret_dist= (ret_dist-ret_dist.min(-1, keepdim=True)[0])/(ret_dist.max(-1, keepdim=True)[0]-ret_dist.min(-1, keepdim=True)[0] + 1e-8)

            ret_labels = labels[ret_ind]
            unique = labels.unique()

    #         bias = -((labels.unique(return_counts=True)[-1]+1e-8)/labels.numel()).log()
    #         mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)

    #         for i in range(len(unique)):
    #             mod_tau[ret_labels==unique[i]] = (tau/bias[i])+1e-8
    #         bias = ((labels.numel()/labels.unique(return_counts=True)[-1]+1e-8))
    #         mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)

    #         for i in range(len(unique)):
    #             mod_tau[ret_labels==unique[i]] = 1/((tau*bias[i]).log()+1e-8)



    #         mod_tau = mod_tau*mod_kek


            bias = (labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel())

            bias = (torch.softmax(bias/1,0)!=0)*1

            mod_tau = get_mod_tau(bias)

            p_l_star = calc_prob(ret_dist, mod_tau, unique)



            ret_labels_ = []
            for i in range(len(unique)):
                killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
                killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
                ret_labels_.append(torch.tensor(killme))

            ret_labels_ = torch.stack(ret_labels_,0).unsqueeze(0).to(dist_matrix.device)

#             print('label_dist')
#             print((labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel()))
#             print(p_l_star.mean(1),'\n',p_l_star.mean(1).sum(-1),'\n',p_l_star.max(1)[0])
#             import pdb; pdb.set_trace()
        except Exception as e:
            import pdb; pdb.set_trace()
        return p_l_star, ret_labels_

    @torch.no_grad()
    def entropy(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        log_pred = pred.log2()
        norm_log = torch.tensor(pred.shape[-1]).float().log2()
        entropy = - (pred*log_pred).nansum(-1)/norm_log
#         pdb.set_trace()
        return entropy

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled):
        # feat_un       = self.prep_feats_fullbatch(features_unlabeled,is_feat=True)

        feat_un = self.prep_feats_fullbatch(features_unlabeled,is_feat=True,is_unlabeled=True)
        feat_lab     = self.prep_feats_fullbatch(features_labeled,is_feat=True)
        target_lab = self.prep_feats_fullbatch(target_labeled)

        targets, labels, labels_intermediate,cl_img = self.create_targets_kmeans(target_lab, feat_lab,1)

        mean_feat_un,cl_un = self.create_targets_kmeans_batched2(feat_un,5)

        init_distances2 = self.cosine(mean_feat_un[:,:,:-2],targets[:,:,:-2])
#         kek,labels_new = self.f(init_distances, labels_intermediate, labels, k=1, tau=1)
#         kek2 = kek.argmax(-1)
#         distances_out  = init_distances.sort(-1)[0][:,:,0]

        kek3,labels_new2 = self.f(init_distances2, labels_intermediate, labels, k=self.k, tau=self.tau)

        kek5 = torch.zeros(feat_un.shape[0],feat_un.shape[1], kek3.shape[-1]).to(features_unlabeled.device)
        kekmax = init_distances2.min(-1)[0]
        distances_out = torch.zeros(feat_un.shape[0],feat_un.shape[1]).to(features_unlabeled.device)
        for i in range(kek5.shape[0]):
            for j in range(len(cl_un[i].unique())):
                kek5[i,cl_un[i] == cl_un[i].unique()[j]] = kek3[i,j]
                distances_out[i,cl_un[i] == cl_un[i].unique()[j]] = kekmax[i,j]
        kek2 = kek5.argmax(-1)

#         pdb.set_trace()

        labels_new_ = labels.unique(dim=1)
        min_indx = init_distances2.argmin(-1)

        kek = kek5



        lab2 = []
        prob = (labels_new2.unsqueeze(0) * kek.unsqueeze(-1)).sum(-2)

        for i in range(kek2.shape[0]):
            lab2 += [labels_new2[0,kek2[i]]]
        lab2 = torch.stack(lab2,0).to(features_unlabeled.device)

        entropy = self.entropy(prob)

        target_lab_rearranged        = []

        for i in range(min_indx.shape[0]):
            target_lab_rearranged   += [labels[0,min_indx[i]]]

        target_lab_rearranged        = torch.stack(target_lab_rearranged,0).to(features_unlabeled.device)
#         final_target = target_lab_rearranged
        final_target = lab2

        return final_target.permute(0,2,1).float(), distances_out, cl_img, cl_un, entropy

    def prepare_predictions(self, pred_unlabeled):
        if self.full_batch:
            pred_un = self.prep_feats_fullbatch(pred_unlabeled)
        else:
            pred_un = self.prep_feats(pred_unlabeled)
        return pred_un.permute(0,2,1).float()

    @torch.no_grad()
    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            # print('iteration',i,first.shape, omg.shape)
        return omg

    def scale(self):
        self.thresholdscale_count += 1
        return self.thresholdscale_count

    def forward(self, features_unlabeled, pred_unlabeled, features_labeled, target_labeled):

        orig_size = features_unlabeled.shape

        features_labeled_ = F.interpolate(features_labeled,self.loss_size,mode='bilinear')

        features_unlabeled_ = F.interpolate(features_unlabeled,self.loss_size,mode='bilinear')

        pred_unlabeled_ = F.interpolate(pred_unlabeled,self.loss_size,mode='bilinear')

        target_labeled_ = F.interpolate(target_labeled,self.loss_size,mode='nearest')

        with torch.no_grad():
            target, distances,cl_img, cl_un, entropy = self.prepare_label(features_unlabeled_, features_labeled_,target_labeled_)
#         pred   = self.prepare_predictions(pred_unlabeled)

        loss = []

        p = pred_unlabeled #  self.refactor_target_to_image(pred)

        t = F.interpolate(self.refactor_target_to_image(target), orig_size[-1],mode='nearest')
        if cl_img is not None:
            cl_img = F.interpolate(self.refactor_target_to_image(cl_img).unsqueeze(1), orig_size[-1],mode='nearest')
            cl_un = F.interpolate(self.refactor_target_to_image(cl_un).unsqueeze(1), orig_size[-1],mode='nearest')
            entropy = F.interpolate(self.refactor_target_to_image(entropy).unsqueeze(1), orig_size[-1],mode='nearest').clamp(0,1)
            distances = F.interpolate(self.refactor_target_to_image(distances).unsqueeze(1), orig_size[-1],mode='nearest').clamp(0,1)

        if self.use_cx:
            t = t.argmax(1)
            base_loss = torch.nn.functional.cross_entropy(p,t, reduction='none')
            # map = (entropy>(entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0])*1
            base_loss = (base_loss * map*(1-entropy)).sum()/map.sum()

            loss += [base_loss]
        else:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,t, reduction='none')
            # map = (entropy>((entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0]).unsqueeze(-1).unsqueeze(-1))*1
            # base_loss = (base_loss * map*(1-entropy)).sum()/map.sum()
            base_loss = (base_loss * (1-entropy)).sum()/(((1-entropy)!=0).sum()+1e-8).sum()
            loss += [base_loss]

        loss = torch.stack(loss,0).sum(0)

        return loss, t, distances, cl_img,cl_un,entropy, target_labeled

# from misc.morphological_operations import *

class distr_reference_loss(nn.Module):
    def __init__(self, loss_size, full_batch, k,tau, use_refine=True, use_morph = True, mode='full',use_softmax=False, reduction='mean', \
                 interpolation_mode='nearest',additional_normal=False, morph_kernel_size = 3, use_dice=True, use_cx=True, classes=  4):
        super(distr_reference_loss, self).__init__()
        self.reduction = reduction
        self.mode = mode
        self.loss_size = loss_size
        self.use_refine = use_refine
        self.use_morph = use_morph
        self.interpolation_mode = interpolation_mode
        self.full_batch = full_batch
        self.additional_normal = additional_normal
        self.use_cx = use_cx
        self.use_dice = use_dice
        self.tau = tau
        self.k  = k
        self.use_softmax = use_softmax
        self.thresholdscale_count = 0
        self.morph_kernel_size = morph_kernel_size
        self.use_entropy = True

    @torch.no_grad()
    def cosine(self,aa,bb):
#             aa = (aa - aa.mean())/aa.std()
#             bb = (bb - bb.mean())/bb.std()

            a_norm = aa/aa.norm(dim=-1, keepdim=True)
            b_norm = bb/bb.norm(dim=-1, keepdim=True)
            mm = torch.matmul(a_norm,b_norm.permute(*range(len(bb.shape)-2),-1,-2))
            return 1-mm.relu()

    @torch.no_grad()
    def prep_feats(self, feat, is_feat=False):
        b,c,w,h = feat.shape

        if is_feat:
            aranged = torch.cat([torch.arange(w).view(1,1,1,-1) * torch.ones(b,1,h,w)
                                 ,torch.arange(h).view(1,1,-1,1) * torch.ones(b,1,h,w)],1)
            feat = torch.cat([feat,aranged],1)
            b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        return feat_reshaped

    @torch.no_grad()
    def prep_feats_fullbatch(self, feat, is_feat=False, is_unlabeled=False):
        feat_orig = feat.clone()
        if feat.shape.__len__() == 3:
            feat = feat.unsqueeze(1)

        b,c,w,h = feat.shape

        if not is_unlabeled:
            feat = feat.permute(1,0,2,3).unsqueeze(0).flatten(-2)
        feat_reshaped   = feat.flatten(-2).permute(0,2,1)

        if not is_feat:
            if feat_orig.shape.__len__() == 4:
                target_intermediate = (feat_reshaped * \
                                       torch.tensor([2**i for i in range(feat_reshaped.shape[-1])]).view(1,1,-1).to(feat_reshaped.device)).sum(-1)
            else:
                target_intermediate = feat_reshaped
            return feat_reshaped, target_intermediate.squeeze(0)

        return feat_reshaped


    @torch.no_grad()
    def f(self, features_unlabeled, features_labeled, target_labeled, k=1, tau=1):

        def print_labl_distr(labels):
            return (labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel())

        def get_distr_diff(a,b):
            return None

        def calc_prob_old(ret_dist,mod_tau, unique):
            p_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                p_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]

            if self.use_softmax:
                p_l_star = torch.softmax(p_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (p_l_star/(mod_tau+1e-8))/(p_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def get_sim(ret_dist,ret_labels , unique):
            s_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                s_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]
            return s_l_star

        def calc_prob(s_l_star,mod_tau):
            if self.use_softmax:
                p_l_star = torch.softmax(s_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (s_l_star/(mod_tau+1e-8))/(s_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def print_distr(p_l_star, dim=0):
            distr = torch.zeros(p_l_star.shape[0],p_l_star.shape[-1]).to(p_l_star.device)
            for i in range(len(p_l_star)):
                arg, count = p_l_star[i,:,dim:].argmax(-1).unique(return_counts=True)
                distr[i,arg] = count.float()
            distr = distr/distr.sum(-1,keepdim=True)
            return distr

        def get_mod_tau(values):
            mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
            for i in range(len(unique)):
                mod_tau[ret_labels==unique[i]] = (values[i])+1e-8
            return mod_tau

        def update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=5, dim=1):
            best_bias = bias.clone()

            p_l_star = calc_prob(s_l_star, bias)

            pred_distr = print_distr(p_l_star,0)

            lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1)
            best_i = 0
            p_map = p_l_star[:,:,dim:].argmax(-1)
            change_map = torch.zeros(p_map.shape).to(p_l_star.device)

            for i in range(iters):
                bias[:, 0, dim:] = bias[:, 0, dim:] + lr * (pred_distr[:,dim:]-target_distr[:,dim:])
                p_l_star = calc_prob(s_l_star, bias)
                p_map_new = p_l_star[:,:,dim:].argmax(-1).float()
                change_map += (p_map != p_map_new)*1
                p_map = p_map_new
                pred_distr = print_distr(p_l_star,0)
                # print((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1))
                if ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif).sum()>0:
                    is_lowest = ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif)*1
                    best_bias = bias * is_lowest.view(-1,1,1) + (1-is_lowest).view(-1,1,1)*best_bias
                    lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) * is_lowest + (1-is_lowest) * lowest_dif
                    best_i = i
            return best_bias

        def get_bias(ret_dist, labels_unique):
            bias = torch.ones(ret_dist.shape[0],1,labels_unique.shape[-1]).to(ret_dist.device)
            return bias

        def get_target_distr(p_l_star, label, features_labeled, features_unlabeled, mode='full', dim=0):
            if mode=='feat':
                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i,  unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(features_unlabeled.flatten(-2).mean(-1),features_labeled.flatten(-2).mean(-1))
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            elif mode == 'full':
                label_distr = print_labl_distr(label)
                distr = label_distr.unsqueeze(0).repeat(features_unlabeled.shape[0],1)
            elif mode == 'distr':

                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i, unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(print_distr(p_l_star,0)[:,dim:], orig_distr[:,dim:])
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            else:
                raise Exception

            return distr




        feat_un    = self.prep_feats_fullbatch(features_unlabeled,is_feat=True,is_unlabeled=True)
        feat_lab   = self.prep_feats_fullbatch(features_labeled,is_feat=True)
        target_lab, labels_intermediate = self.prep_feats_fullbatch(target_labeled)

        dist_matrix = self.cosine(feat_un, feat_lab)

        if k == 'max':
            k = dist_matrix.shape[-1]

        if self.additional_normal:
            dist_matrix = (dist_matrix-dist_matrix.mean(-2,keepdim=True))/dist_matrix.std(-2,keepdim=True)

        ret_dist, ret_ind = torch.topk(dist_matrix, k,dim=-1, largest=False)


        ret_labels  = labels_intermediate[ret_ind]
        unique      = labels_intermediate.unique()

        bias     = get_bias(ret_dist,unique)
        s_l_star = get_sim(ret_dist, ret_labels, unique)
        p_l_star = calc_prob(s_l_star, bias)

        if self.use_refine:
            mode = self.mode

            target_distr = get_target_distr(p_l_star, labels_intermediate, features_labeled, features_unlabeled, mode=mode, dim=1)
            bias = update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=50, dim=1)
            p_l_star = calc_prob(s_l_star, bias)

        ret_labels_ = []
        for i in range(len(unique)):
            killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
            killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
            ret_labels_.append(torch.tensor(killme))

        ret_labels_ = torch.stack(ret_labels_,0).unsqueeze(0).to(dist_matrix.device)

        return p_l_star,dist_matrix, ret_labels_

    @torch.no_grad()
    def entropy(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        log_pred = pred.log2()
        norm_log = torch.tensor(pred.shape[-1]).float().log2()
        entropy = - (pred*log_pred).nansum(-1)/norm_log
        return entropy

    @torch.no_grad()
    def complementary_probability(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        compl_prob = 1 - pred.max(dim=-1)[0]
        return compl_prob

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled):

        kek3, init_distances2, labels_new2 = self.f(features_unlabeled, features_labeled, target_labeled, k=self.k, tau=self.tau)

        distances_out = init_distances2.min(-1)[0]

        kek2 = kek3.argmax(-1)

        lab2 = []

        prob = (labels_new2.unsqueeze(0) * kek3.unsqueeze(-1)).sum(-2)

        for i in range(kek2.shape[0]):
            lab2 += [labels_new2[0,kek2[i]]]
        lab2 = torch.stack(lab2,0).to(features_unlabeled.device)

        entropy = self.entropy(prob)

        final_target = lab2

        return final_target.permute(0,2,1).float(), distances_out, entropy

    def prepare_predictions(self, pred_unlabeled):
        if self.full_batch:
            pred_un = self.prep_feats_fullbatch(pred_unlabeled)
        else:
            pred_un = self.prep_feats(pred_unlabeled)
        return pred_un.permute(0,2,1).float()

    @torch.no_grad()
    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            # print('iteration',i,first.shape, omg.shape)
        return omg

    def scale(self):
        self.thresholdscale_count += 1
        return self.thresholdscale_count

    def forward(self, features_unlabeled, pred_unlabeled, features_labeled, target_labeled):
        import pdb; pdb.set_trace()
        orig_size = features_unlabeled.shape

        features_labeled_ = F.interpolate(features_labeled,self.loss_size,mode=self.interpolation_mode)

        features_unlabeled_ = F.interpolate(features_unlabeled,self.loss_size,mode=self.interpolation_mode)

        pred_unlabeled_ = F.interpolate(pred_unlabeled,self.loss_size,mode=self.interpolation_mode)

        if  self.use_cx:
            target_labeled_ =  F.interpolate(F.one_hot(target_labeled.long(),num_classes =4).permute(0,3,1,2).float(),self.loss_size,mode='nearest')
        else:
            target_labeled_ = F.interpolate(target_labeled,self.loss_size,mode='nearest')

        with torch.no_grad():
            target, distances, entropy = self.prepare_label(features_unlabeled_, features_labeled_,target_labeled_)

        loss = []

        p = pred_unlabeled_ #  self.refactor_target_to_image(pred)

        with torch.no_grad():
            target = self.refactor_target_to_image(target)
            if self.use_morph:
                dilation = Dilation2d(self.morph_kernel_size, padding=1)
                mode = ModePool2d(self.morph_kernel_size, padding=1)
                erosion = Erosion2d(self.morph_kernel_size, padding=1)
                target = dilation(target)
                target = erosion(target)
                target = mode(target)

            entropy = self.refactor_target_to_image(entropy).unsqueeze(1)
            distances = self.refactor_target_to_image(distances).unsqueeze(1)

        t = target#F.interpolate(target, orig_size[-1],mode='nearest')

        if self.use_cx:
            t = t.argmax(1)
            base_loss = torch.nn.functional.cross_entropy(p,t, reduction='none')
            # map = (entropy>(entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0])*1
            # base_loss = (baseloss * map*(1-entropy)).sum()/map.sum()
            if self.use_entropy:
                base_loss = base_loss*(1-entropy.squeeze(1))
            loss += [base_loss]
            target = target.argmax(1)

        else:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,t, reduction='none')

            if self.use_entropy:
                base_loss = base_loss*(1-entropy)

            # map = (entropy>(entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0])*1
            # base_loss = (baseloss * map*(1-entropy)).sum()/map.sum()
            loss += [base_loss]

        loss = torch.stack(loss,0).sum(0)
        entropy = entropy.clamp(0,1)
        distances = distances.clamp(0,1)

        return loss, target, distances,entropy



class distr_reference_highclass_loss(nn.Module):
    def __init__(self, loss_size, full_batch, k,tau=1, use_refine=True, use_morph = True, mode='full',use_softmax=False, reduction='mean', \
                 interpolation_mode='nearest',additional_normal=False, morph_kernel_size = 3, use_dice=True, use_cx=True, classes=  4):
        super(distr_reference_highclass_loss, self).__init__()
        self.reduction = reduction
        self.mode = mode
        self.loss_size = loss_size
        self.use_refine = use_refine
        self.use_morph = use_morph
        self.interpolation_mode = interpolation_mode
        self.full_batch = full_batch
        self.additional_normal = additional_normal
        self.use_cx = use_cx
        self.use_dice = use_dice
        self.tau = tau
        self.k  = k
        self.use_softmax = use_softmax
        self.thresholdscale_count = 0
        self.morph_kernel_size = morph_kernel_size
        self.use_entropy = True

    @torch.no_grad()
    def cosine(self,aa,bb):
#             aa = (aa - aa.mean())/aa.std()
#             bb = (bb - bb.mean())/bb.std()

            a_norm = aa/aa.norm(dim=-1, keepdim=True)
            b_norm = bb/bb.norm(dim=-1, keepdim=True)
            mm = torch.matmul(a_norm,b_norm.permute(*range(len(bb.shape)-2),-1,-2))
            return 1-mm.relu()

    @torch.no_grad()
    def prep_feats(self, feat, is_feat=False):
        b,c,w,h = feat.shape

        if is_feat:
            aranged = torch.cat([torch.arange(w).view(1,1,1,-1) * torch.ones(b,1,h,w)
                                 ,torch.arange(h).view(1,1,-1,1) * torch.ones(b,1,h,w)],1)
            feat = torch.cat([feat,aranged],1)
            b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        return feat_reshaped

    @torch.no_grad()
    def prep_feats_fullbatch(self, feat, is_feat=False, is_unlabeled=False):
        a = time.time()
        feat_orig = feat.clone()
        if feat.shape.__len__() == 3:
            feat = feat.unsqueeze(1)

        b,c,w,h = feat.shape

        if not is_unlabeled:
            feat = feat.permute(1,0,2,3).unsqueeze(0).flatten(-2)
        feat_reshaped   = feat.flatten(-2).permute(0,2,1)

        if not is_feat:
            if feat_orig.shape.__len__() == 4:
#                 target_intermediate2 = (feat_reshaped * \
#                                        torch.tensor([2**i for i in range(feat_reshaped.shape[-1])]).view(1,1,-1).to(feat_reshaped.device)).sum(-1)

                target_intermediate = torch.cdist(feat_reshaped.permute(-1,0,1).flatten(1).unique(dim=1).permute(1,0).unsqueeze(0), feat_reshaped).argmin(1).float()

            else:
                target_intermediate = feat_reshaped
            return feat_reshaped, target_intermediate.squeeze(0), feat_reshaped.permute(-1,0,1).flatten(1).unique(dim=1).permute(1,0).unsqueeze(0)#, target_intermediate2
        return feat_reshaped


    @torch.no_grad()
    def f(self, features_unlabeled, features_labeled, target_labeled, k=1, tau=1):

        def print_labl_distr(labels):
            return (labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel())

        def get_distr_diff(a,b):
            return None

        def calc_prob_old(ret_dist,mod_tau, unique):
            p_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                p_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]

            if self.use_softmax:
                p_l_star = torch.softmax(p_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (p_l_star/(mod_tau+1e-8))/(p_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def get_sim(ret_dist,ret_labels , unique):
            s_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                s_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]
            return s_l_star

        def calc_prob(s_l_star,mod_tau):
            if self.use_softmax:
                p_l_star = torch.softmax(s_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (s_l_star/(mod_tau+1e-8))/(s_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def print_distr(p_l_star, dim=0):
            distr = torch.zeros(p_l_star.shape[0],p_l_star.shape[-1]).to(p_l_star.device)
            for i in range(len(p_l_star)):
                arg, count = p_l_star[i,:,dim:].argmax(-1).unique(return_counts=True)
                distr[i,arg] = count.float()
            distr = distr/distr.sum(-1,keepdim=True)
            return distr

        def get_mod_tau(values):
            mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
            for i in range(len(unique)):
                mod_tau[ret_labels==unique[i]] = (values[i])+1e-8
            return mod_tau

        def update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=5, dim=1):
            best_bias = bias.clone()

            p_l_star = calc_prob(s_l_star, bias)

            pred_distr = print_distr(p_l_star,0)

            lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1)
            best_i = 0
            p_map = p_l_star[:,:,dim:].argmax(-1)
            change_map = torch.zeros(p_map.shape).to(p_l_star.device)

            for i in range(iters):
                bias[:, 0, dim:] = bias[:, 0, dim:] + lr * (pred_distr[:,dim:]-target_distr[:,dim:])
                p_l_star = calc_prob(s_l_star, bias)
                p_map_new = p_l_star[:,:,dim:].argmax(-1).float()
                change_map += (p_map != p_map_new)*1
                p_map = p_map_new
                pred_distr = print_distr(p_l_star,0)
                # print((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1))
                if ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif).sum()>0:
                    is_lowest = ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif)*1
                    best_bias = bias * is_lowest.view(-1,1,1) + (1-is_lowest).view(-1,1,1)*best_bias
                    lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) * is_lowest + (1-is_lowest) * lowest_dif
                    best_i = i
            return best_bias

        def get_bias(ret_dist, labels_unique):
            bias = torch.ones(ret_dist.shape[0],1,labels_unique.shape[-1]).to(ret_dist.device)
            return bias

        def get_target_distr(p_l_star, label, features_labeled, features_unlabeled, mode='full', dim=0):
            if mode=='feat':
                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i,  unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(features_unlabeled.flatten(-2).mean(-1),features_labeled.flatten(-2).mean(-1))
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            elif mode == 'full':
                label_distr = print_labl_distr(label)
                distr = label_distr.unsqueeze(0).repeat(features_unlabeled.shape[0],1)
            elif mode == 'distr':

                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i, unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(print_distr(p_l_star,0)[:,dim:], orig_distr[:,dim:])
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            else:
                raise Exception

            return distr



        feat_un    = self.prep_feats_fullbatch(features_unlabeled,is_feat=True,is_unlabeled=True)
        feat_lab   = self.prep_feats_fullbatch(features_labeled,is_feat=True)
        target_lab, labels_intermediate, ret_labels2 = self.prep_feats_fullbatch(target_labeled)

        dist_matrix = self.cosine(feat_un, feat_lab)

        if k == 'max':
            k = dist_matrix.shape[-1]

        if self.additional_normal:
            dist_matrix = (dist_matrix-dist_matrix.mean(-2,keepdim=True))/dist_matrix.std(-2,keepdim=True)

        ret_dist, ret_ind = torch.topk(dist_matrix, k,dim=-1, largest=False)


        ret_labels  = labels_intermediate[ret_ind]
        unique      = labels_intermediate.unique()

        bias     = get_bias(ret_dist,unique)

        s_l_star = get_sim(ret_dist, ret_labels, unique)
        p_l_star = calc_prob(s_l_star, bias)
        if self.use_refine:
            mode = self.mode

            target_distr = get_target_distr(p_l_star, labels_intermediate, features_labeled, features_unlabeled, mode=mode, dim=1)
            bias = update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=50, dim=1)
            p_l_star = calc_prob(s_l_star, bias)

#         ret_labels_ = []

#         for i in range(len(unique)):
#             killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
#             killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
#             ret_labels_.append(torch.tensor(killme))

#         ret_labels_ = torch.stack(ret_labels_,0).unsqueeze(0).to(dist_matrix.device)
#         pdb.set_trace()
#         ret_labels2 = target_lab[ret_ind]
#
#         return p_l_star,dist_matrix, ret_labels_
        return p_l_star,dist_matrix, ret_labels2

    @torch.no_grad()
    def entropy(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        log_pred = pred.log2()
        norm_log = torch.tensor(pred.shape[-1]).float().log2()
        entropy = - (pred*log_pred).nansum(-1)/norm_log
        return entropy

    @torch.no_grad()
    def complementary_probability(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        compl_prob = 1 - pred.max(dim=-1)[0]
        return compl_prob

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled):

        kek3, init_distances2, labels_new2 = self.f(features_unlabeled, features_labeled, target_labeled, k=self.k, tau=self.tau)

        distances_out = init_distances2.min(-1)[0]

        kek2 = kek3.argmax(-1)

        lab2 = []
        prob = (labels_new2.unsqueeze(0) * kek3.unsqueeze(-1)).sum(-2)

        for i in range(kek2.shape[0]):
            lab2 += [labels_new2[0,kek2[i]]]
        lab2 = torch.stack(lab2,0).to(features_unlabeled.device)

        entropy = self.entropy(prob)

        final_target = lab2

        return final_target.permute(0,2,1).float(), distances_out, entropy

    def prepare_predictions(self, pred_unlabeled):
        if self.full_batch:
            pred_un = self.prep_feats_fullbatch(pred_unlabeled)
        else:
            pred_un = self.prep_feats(pred_unlabeled)
        return pred_un.permute(0,2,1).float()

    @torch.no_grad()
    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            # print('iteration',i,first.shape, omg.shape)
        return omg

    def scale(self):
        self.thresholdscale_count += 1
        return self.thresholdscale_count

    def forward(self, features_unlabeled, pred_unlabeled, features_labeled, target_labeled):
#         import pdb; pdb.set_trace()
        orig_size = features_unlabeled.shape

        features_labeled_ = F.interpolate(features_labeled,self.loss_size,mode=self.interpolation_mode)

        features_unlabeled_ = F.interpolate(features_unlabeled,self.loss_size,mode=self.interpolation_mode)

        pred_unlabeled_ = F.interpolate(pred_unlabeled,self.loss_size,mode=self.interpolation_mode)

        if  self.use_cx:
            target_labeled_ =  F.interpolate(F.one_hot(target_labeled.long(),num_classes =4).permute(0,3,1,2).float(),self.loss_size,mode='nearest')
        else:
            target_labeled_ = F.interpolate(target_labeled,self.loss_size,mode='nearest')

        with torch.no_grad():
            target, distances, entropy = self.prepare_label(features_unlabeled_, features_labeled_,target_labeled_)

        loss = []

        p = pred_unlabeled_
        with torch.no_grad():
            target = self.refactor_target_to_image(target)
            entropy = self.refactor_target_to_image(entropy).unsqueeze(1)
            distances = self.refactor_target_to_image(distances).unsqueeze(1)

        t = target

        if self.use_cx:
            t = t.argmax(1)
            base_loss = torch.nn.functional.cross_entropy(p,t, reduction='none')
            if self.use_entropy:
                base_loss = base_loss*(1-entropy.squeeze(1))
            loss += [base_loss]
            target = target.argmax(1)

        else:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,t, reduction='none')

            if self.use_entropy:
                base_loss = base_loss*(1-entropy)

            loss += [base_loss]

        loss = torch.stack(loss,0).sum(0)
        entropy = entropy.clamp(0,1)
        distances = distances.clamp(0,1)

        return loss, target, distances,entropy


class distr_aug_reference_loss(nn.Module):
    def __init__(self, loss_size, full_batch, k,tau, use_refine=True, use_morph = True, mode='full',use_softmax=False, reduction='mean', \
                 interpolation_mode='nearest', additional_normal=False, morph_kernel_size = 3, use_dice=True, use_cx=True,upscale_for_vis=True):
        super(distr_aug_reference_loss, self).__init__()
        self.reduction = reduction
        self.mode = mode
        self.loss_size = loss_size
        self.use_refine = use_refine
        self.use_morph = use_morph
        self.interpolation_mode = interpolation_mode
        self.full_batch = full_batch
        self.use_cx = use_cx
        self.use_dice = use_dice
        self.additional_normal = additional_normal
        self.tau = tau
        self.k  = k
        self.use_softmax = use_softmax
        self.thresholdscale_count = 0
        self.morph_kernel_size = morph_kernel_size
        self.use_entropy = True
        self.upscale_for_vis = upscale_for_vis

    @torch.no_grad()
    def cosine(self,aa,bb):
#             aa = (aa - aa.mean())/aa.std()
#             bb = (bb - bb.mean())/bb.std()

            a_norm = aa/aa.norm(dim=-1, keepdim=True)
            b_norm = bb/bb.norm(dim=-1, keepdim=True)
            mm = torch.matmul(a_norm,b_norm.permute(*range(len(bb.shape)-2),-1,-2))
            return 1-mm.relu()

    @torch.no_grad()
    def prep_feats(self, feat, is_feat=False):
        b,c,w,h = feat.shape

        if is_feat:
            aranged = torch.cat([torch.arange(w).view(1,1,1,-1) * torch.ones(b,1,h,w)
                                 ,torch.arange(h).view(1,1,-1,1) * torch.ones(b,1,h,w)],1)
            feat = torch.cat([feat,aranged],1)
            b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        return feat_reshaped

    @torch.no_grad()
    def prep_feats_fullbatch(self, feat, is_feat=False, is_unlabeled=False):
        feat_orig = feat.clone()
        if feat.shape.__len__() == 3:
            feat = feat.unsqueeze(1)

        b,c,w,h = feat.shape

        if not is_unlabeled:
            feat = feat.permute(1,0,2,3).unsqueeze(0).flatten(-2)
        feat_reshaped   = feat.flatten(-2).permute(0,2,1)

        # if not is_feat:
        #     if feat_orig.shape.__len__() == 4:
        #         target_intermediate = (feat_reshaped * \
        #                                torch.tensor([2**i for i in range(feat_reshaped.shape[-1])]).view(1,1,-1).to(feat_reshaped.device)).sum(-1)
        #     else:
        #         target_intermediate = feat_reshaped
        #     return feat_reshaped, target_intermediate.squeeze(0)

        if not is_feat:
            if feat_orig.shape.__len__() == 4:
#                 target_intermediate2 = (feat_reshaped * \
#                                        torch.tensor([2**i for i in range(feat_reshaped.shape[-1])]).view(1,1,-1).to(feat_reshaped.device)).sum(-1)

                target_intermediate = torch.cdist(feat_reshaped.permute(-1,0,1).flatten(1).unique(dim=1).permute(1,0).unsqueeze(0), feat_reshaped).argmin(1).float()

            else:
                target_intermediate = feat_reshaped
            return feat_reshaped, target_intermediate.squeeze(0), feat_reshaped.permute(-1,0,1).flatten(1).unique(dim=1).permute(1,0).unsqueeze(0)#, target_intermediate2

        return feat_reshaped


    @torch.no_grad()
    def f(self, features_unlabeled, features_labeled, target_labeled, k=1, tau=1):

        def print_labl_distr(labels):
            return (labels.unique(return_counts=True)[-1]+1e-8)/(labels.numel())

        def get_distr_diff(a,b):
            return None

        def calc_prob_old(ret_dist,mod_tau, unique):
            p_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                p_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]

            if self.use_softmax:
                p_l_star = torch.softmax(p_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (p_l_star/(mod_tau+1e-8))/(p_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def get_sim(ret_dist,ret_labels , unique):
            s_l_star = torch.zeros(*ret_dist.shape[:2], len(unique)).to(ret_dist.device)

            for i in range(len(unique)):
                s_l_star[:,:,i] = ((1-ret_dist) * (ret_labels == unique[i])).max(-1)[0]
            return s_l_star

        def calc_prob(s_l_star,mod_tau):
            if self.use_softmax:
                p_l_star = torch.softmax(s_l_star/(mod_tau+1e-8),-1)
            else:
                p_l_star = (s_l_star/(mod_tau+1e-8))/(s_l_star/(mod_tau+1e-8)).sum(-1, keepdim=True)

            return p_l_star

        def print_distr(p_l_star, dim=0):
            distr = torch.zeros(p_l_star.shape[0],p_l_star.shape[-1]).to(p_l_star.device)
            for i in range(len(p_l_star)):
                arg, count = p_l_star[i,:,dim:].argmax(-1).unique(return_counts=True)
                distr[i,arg] = count.float()
            distr = distr/distr.sum(-1,keepdim=True)
            return distr

        def get_mod_tau(values):
            mod_tau = torch.ones(ret_labels.shape).to(dist_matrix.device)
            for i in range(len(unique)):
                mod_tau[ret_labels==unique[i]] = (values[i])+1e-8
            return mod_tau

        def update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=5, dim=1):
            best_bias = bias.clone()

            p_l_star = calc_prob(s_l_star, bias)

            pred_distr = print_distr(p_l_star,0)

            lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1)
            best_i = 0
            p_map = p_l_star[:,:,dim:].argmax(-1)
            change_map = torch.zeros(p_map.shape).to(p_l_star.device)

            for i in range(iters):
                bias[:, 0, dim:] = bias[:, 0, dim:] + lr * (pred_distr[:,dim:]-target_distr[:,dim:])
                p_l_star = calc_prob(s_l_star, bias)
                p_map_new = p_l_star[:,:,dim:].argmax(-1).float()
                change_map += (p_map != p_map_new)*1
                p_map = p_map_new
                pred_distr = print_distr(p_l_star,0)
                # print((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1))
                if ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif).sum()>0:
                    is_lowest = ((pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) < lowest_dif)*1
                    best_bias = bias * is_lowest.view(-1,1,1) + (1-is_lowest).view(-1,1,1)*best_bias
                    lowest_dif = (pred_distr[:,dim:]-target_distr[:,dim:]).abs().sum(-1) * is_lowest + (1-is_lowest) * lowest_dif
                    best_i = i
            return best_bias

        def get_bias(ret_dist, labels_unique):
            bias = torch.ones(ret_dist.shape[0],1,labels_unique.shape[-1]).to(ret_dist.device)
            return bias

        def get_target_distr(p_l_star, label, features_labeled, features_unlabeled, mode='full', dim=0):
            if mode=='feat':
                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i,  unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(features_unlabeled.flatten(-2).mean(-1),features_labeled.flatten(-2).mean(-1))
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            elif mode == 'full':
                label_distr = print_labl_distr(label)
                distr = label_distr.unsqueeze(0).repeat(features_unlabeled.shape[0],1)
            elif mode == 'distr':

                label = label.view(features_labeled.shape[0],-1)
                unique = label.unique()
                orig_distr = torch.zeros(features_labeled.shape[0], len(unique)).to(features_unlabeled.device)
                for i in range(features_labeled.shape[0]):
                    unique_, counts = label[i].unique(return_counts = True)
                    for j in range(len(unique_)):
                        orig_distr[i, unique==unique_[j]] = counts[j].float()
                orig_distr = orig_distr/label.shape[-1]
                dist = torch.cdist(print_distr(p_l_star,0)[:,dim:], orig_distr[:,dim:])
                _, ret_ind = dist.min(-1)
                distr = orig_distr[ret_ind]
            else:
                raise Exception

            return distr




        feat_un    = self.prep_feats_fullbatch(features_unlabeled,is_feat=True,is_unlabeled=True)
        feat_lab   = self.prep_feats_fullbatch(features_labeled,is_feat=True)
        target_lab, labels_intermediate = self.prep_feats_fullbatch(target_labeled)

        dist_matrix = self.cosine(feat_un, feat_lab)

        if k == 'max':
            k = dist_matrix.shape[-1]

        if self.additional_normal:
            dist_matrix = (dist_matrix-dist_matrix.mean(-2,keepdim=True))/dist_matrix.std(-2,keepdim=True)

        ret_dist, ret_ind = torch.topk(dist_matrix, k,dim=-1, largest=False)


        ret_labels  = labels_intermediate[ret_ind]
        unique      = labels_intermediate.unique()

        bias     = get_bias(ret_dist,unique)
        s_l_star = get_sim(ret_dist, ret_labels, unique)
        p_l_star = calc_prob(s_l_star, bias)

        if self.use_refine:
            mode = self.mode

            target_distr = get_target_distr(p_l_star, labels_intermediate, features_labeled, features_unlabeled, mode=mode, dim=1)
            bias = update_bias(s_l_star, target_distr, ret_dist, unique, bias, lr= 0.005, iters=50, dim=1)
            p_l_star = calc_prob(s_l_star, bias)

        ret_labels_ = []
        for i in range(len(unique)):
            killme = [int(x) for x in list('{0:0b}'.format(unique[i].long().item()))][::-1]
            killme += [0 for j in range(math.ceil((max(unique)+1).log2())-len(killme))]
            ret_labels_.append(torch.tensor(killme))

        ret_labels_ = torch.stack(ret_labels_,0).unsqueeze(0).to(dist_matrix.device)

        return p_l_star,dist_matrix, ret_labels_

    @torch.no_grad()
    def entropy(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        log_pred = pred.log2()
        norm_log = torch.tensor(pred.shape[-1]).float().log2()
        entropy = - (pred*log_pred).nansum(-1)/norm_log
        return entropy

    @torch.no_grad()
    def complementary_probability(self,pred):
        pred = pred.clamp(1e-8,1-1e-8)
        compl_prob = 1 - pred.max(dim=-1)[0]
        return compl_prob

    @torch.no_grad()
    def prepare_label(self, features_unlabeled,features_labeled,target_labeled):

        kek3, init_distances2, labels_new2 = self.f(features_unlabeled, features_labeled, target_labeled, k=self.k, tau=self.tau)

        distances_out = init_distances2.min(-1)[0]

        kek2 = kek3.argmax(-1)

        lab2 = []

        prob = (labels_new2.unsqueeze(0) * kek3.unsqueeze(-1)).sum(-2)

        for i in range(kek2.shape[0]):
            lab2 += [labels_new2[0,kek2[i]]]
        lab2 = torch.stack(lab2,0).to(features_unlabeled.device)

        entropy = self.entropy(prob)

        final_target = lab2

        return final_target.permute(0,2,1).float(), distances_out, entropy

    def prepare_predictions(self, pred_unlabeled):
        if self.full_batch:
            pred_un = self.prep_feats_fullbatch(pred_unlabeled)
        else:
            pred_un = self.prep_feats(pred_unlabeled)
        return pred_un.permute(0,2,1).float()

    @torch.no_grad()
    def refactor_target_to_image(self, lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            # print('iteration',i,first.shape, omg.shape)
        return omg

    def scale(self):
        self.thresholdscale_count += 1
        return self.thresholdscale_count

    def forward(self, features_unlabeled, pred_unlabeled, features_labeled, target_labeled):

        orig_size = features_unlabeled.shape

        features_labeled_ = F.interpolate(features_labeled,self.loss_size,mode=self.interpolation_mode)

        features_unlabeled_ = F.interpolate(features_unlabeled,self.loss_size,mode=self.interpolation_mode)

        pred_unlabeled_ = F.interpolate(pred_unlabeled,self.loss_size,mode=self.interpolation_mode)

        target_labeled_ = F.interpolate(target_labeled,self.loss_size,mode='nearest')

        with torch.no_grad():
            target, distances, entropy = self.prepare_label(features_unlabeled_, features_labeled_,target_labeled_)

        loss = []

        p = pred_unlabeled_ #  self.refactor_target_to_image(pred)

        with torch.no_grad():
            target = self.refactor_target_to_image(target)
            if self.use_morph:
                dilation = Dilation2d(self.morph_kernel_size, padding=1)
                mode = ModePool2d(self.morph_kernel_size, padding=1)
                erosion = Erosion2d(self.morph_kernel_size, padding=1)
                target = dilation(target)
                target = erosion(target)
                target = mode(target)

            entropy = self.refactor_target_to_image(entropy).unsqueeze(1)
            distances = self.refactor_target_to_image(distances).unsqueeze(1)

        t = target#F.interpolate(target, orig_size[-1],mode='nearest')

        if self.use_cx:
            t = t.argmax(1)
            base_loss = torch.nn.functional.cross_entropy(p,t, reduction='none')
            # map = (entropy>(entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0])*1
            # base_loss = (baseloss * map*(1-entropy)).sum()/map.sum()
            if self.use_entropy:
                base_loss = base_loss*(1-entropy)

            loss += [base_loss]
        else:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(p,t, reduction='none')

            if self.use_entropy:
                base_loss = base_loss*(1-entropy)

            # map = (entropy>(entropy.flatten(-2).max(-1)[0]-entropy.flatten(-2).min(-1)[0])*0.2+entropy.flatten(-2).min(-1)[0])*1
            # base_loss = (baseloss * map*(1-entropy)).sum()/map.sum()
            loss += [base_loss]

        loss = torch.stack(loss,0).sum(0)

        if self.upscale_for_vis:
            entropy = F.interpolate(entropy, orig_size[-1],mode='nearest').clamp(0,1)
            distances = F.interpolate(distances, orig_size[-1],mode='nearest').clamp(0,1)
            target = F.interpolate(target, orig_size[-1],mode='nearest')

        return loss, target, distances,entropy




def multiscale_loss(features_unlabeled, pred_unlabeled, features_labeled, target_labeled, scales, expansions, fullbatch=True):

    def get_multiscale_features(feat, scales, interpolation_mode='bilinear', is_label=False):
        multi_scale_features = []
        for i in scales:
            if is_label:
                multi_scale_features += [F.adaptive_max_pool2d(feat,i)]
            else:
                multi_scale_features += [F.interpolate(feat,size=(i,i),mode=interpolation_mode)]
        return multi_scale_features

    def arrange_map(feat, indx, dimlist=[]):
        if len(dimlist) == 1:
            feat = feat[indx]
        else:
            okee = []
            for i in range(dimlist[0]):
                okee += [arrange_map(feat[i], indx[i], dimlist[1:])]
            feat = torch.stack(okee,0)
        return feat

    def get_level(feat_labeled_coarse, feat_unlabeled_coarse):
        init_distances = torch.cdist(feat_unlabeled_coarse, feat_labeled_coarse)
        min_indx = init_distances.argmin(-1)
        return min_indx

    def arrange_all(feat_list, indices, current_scale, all_scales, fullbatch=False):
        out_feat_list = []
        for i in range(0, current_scale):
            out_feat_list += [feat_list[i]]

        for i in range(current_scale, len(all_scales)):
            if fullbatch and i == 0:
                tmp = []
                for j in range(indices.shape[0]):
                    tmp += [arrange_map(feat_list[i],indices[j:j+1], list(indices[j:j+1].shape))]
                tmp = torch.cat(tmp,0)
            else:
                tmp = arrange_map(feat_list[i],indices, list(indices.shape))
            out_feat_list += [tmp]
        return out_feat_list

    def expand_map(x, ks, stride=1, return_unflattened=False):
        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w//stride*(ks//2*2+stride), h//stride*(ks//2*2+stride)

        range_list_w  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,w, stride)]
        range_list_h  = [list(range(i,i+ stride + ks//2*2)) for i in range(0,h, stride)]

        expand_h      = x1[:, :, range_list_h, :]
        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)
        expand_h_view = expand_h.view(b,c,expected_h,padded_w)
        expand_w      = expand_h_view[:,:, :, range_list_w].view(b,c,expected_h,expected_w)

        if not return_unflattened:
            return expand_w
        else:
            return expand_w, expand_w_

    def prep_all_feats(feat_list, scales, base=2, fullbatch=False):
        out = []
        for i in range(len(scales)):
            if type(base) == list:
                base_ = base[i]
            else:
                base_ = base
            out += [prep_larger_scale(feat_list[i], i, base_, fullbatch and i==0)]

        return out

    def prep_larger_scale(feat, scale, base=2, fullbatch=False):
        tmp_feat = feat.clone()
        for i in range(scale):
            w,h = tmp_feat.shape[-2:]
            if  type(base)==list:
                cur_base = base[i]
            else:
                cur_base = base
            split_size = w//(cur_base**(0+1))
            tmp_feat = torch.cat(torch.stack(tmp_feat.split(split_size,dim=-1),dim=-3).split(split_size,dim=-2),dim=-3)
        tmp_feat = tmp_feat.flatten(-2)
        tmp_feat = tmp_feat.permute(*[0,*list(range(2,len(tmp_feat.shape[2:])+2)),1])
        if fullbatch:
            tmp_feat = torch.cat(tmp_feat.split(1,0),1)
        return tmp_feat

    def hierarchical_map_expansion(feat, size_list, expansion_list):
        tmp = feat
        f = len(size_list)-1
        current_stride = 1
        out_split_list = []
        for i in range(len(size_list)):
            walking_stride = int(np.prod(out_split_list))
            kernel_radius = expansion_list[f-i]*walking_stride
            current_stride = max(size_list[f-i]* walking_stride,current_stride)
            tmp = expand_map(tmp,1+kernel_radius*2, max(current_stride,1))
            out_split_list.insert(0,size_list[f-i]+expansion_list[f-i]*2)
        return tmp, out_split_list

    def get_multiscale_features(feat, scales, expansions, interpolation_mode='bilinear', is_label=False):
        multi_scale_features = []
        split_lists = []
        for i in range(len(scales)):
            if is_label:
                tmp = F.adaptive_max_pool2d(feat,int(np.prod(scales[:i+1])))
            else:
                if interpolation_mode == 'avg_pool':
                    tmp = F.adaptive_avg_pool2d(feat,int(np.prod(scales[:i+1])))
                else:
                    tmp = F.interpolate(feat,size=(int(np.prod(scales[:i+1])),int(np.prod(scales[:i+1]))),mode=interpolation_mode)
            tmp, tmp_split_list = hierarchical_map_expansion(tmp, scales[:i+1], expansions[:i+1])
            multi_scale_features += [tmp]
            split_lists += [tmp_split_list]
        return multi_scale_features, split_lists

    @torch.no_grad()
    def prepare_label(features_unlabeled,features_labeled,target_labeled, scales, expansions=None, fullbatch=False):
        feat_un, split_list_feat_un       = get_multiscale_features(features_unlabeled, scales, [0 for i in expansions])
        feat_lab, split_list_feat_lab     = get_multiscale_features(features_labeled, scales, expansions)
        target_lab, split_list_target_lab = get_multiscale_features(target_labeled, scales, expansions)

        feat_un_prepped    = prep_all_feats(feat_un, scales, split_list_feat_un)
        feat_lab_prepped   = prep_all_feats(feat_lab, scales, split_list_feat_lab,fullbatch)
        target_lab_prepped = prep_all_feats(target_lab, scales, split_list_target_lab,fullbatch)

        lol = None

        for i in range(len(scales)):
            print(len(feat_lab_prepped))
            print(feat_lab_prepped[i].shape, feat_un_prepped[i].shape)
            indices = get_level(feat_lab_prepped[i], feat_un_prepped[i])

            feat_lab_prepped   = arrange_all(feat_lab_prepped, indices, i, range(len(scales)), fullbatch)
            target_lab_prepped = arrange_all(target_lab_prepped, indices, i, range(len(scales)), fullbatch)

        arranged_target = target_lab_prepped[-1]
        shape = range(len(arranged_target.shape))

        return arranged_target.permute(0,-1,*shape[1:-1]).float(), target_lab_prepped

    def prepare_predictions(pred_unlabeled,scales,expansions):
        preds, split_list = get_multiscale_features(pred_unlabeled, scales,[0 for i in expansions])
        pred_un = prep_larger_scale(preds[-1], len(scales)-1, split_list[-1])
        shape = range(len(pred_un.shape))
        return pred_un.permute(0,-1,*shape[1:-1]).float()

    def refactor_target_to_image(lulw):
        shape = lulw.shape
        u = lulw
        shape = u.shape
        oha  = int(torch.sqrt(torch.tensor(shape[-1]).float().prod()))
        u = u.reshape(*shape[:-1],oha,oha)
        omg = u.clone()

        for i in range(len(shape)-3):
            s_tmp = omg.shape
            sqrt = int(torch.sqrt(torch.tensor(s_tmp[-3]).float()))
            first = torch.cat(omg.split(sqrt,-3),-2)

            omg = torch.cat(first.squeeze(-3).split(1,-3),-1).squeeze(-3)
            print('iteration',i,first.shape, omg.shape)
        return omg

    with torch.no_grad():
        target, all_targets = prepare_label(features_unlabeled,features_labeled,target_labeled,scales,expansions, fullbatch)

    pred   = prepare_predictions(pred_unlabeled, scales, expansions)

    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,target, reduction='none')
    loss = base_loss
    return loss, refactor_target_to_image(target)

def assistance_loss(features_unlabeled, pred_unlabeled, features_labeled, target_labeled, kernel_size):
    def score_min(x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.min(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def score_max(x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.max(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def expand_map(x, ks, return_unflattened=False):
        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w*ks, h*ks

        range_list_w  = [list(range(i,i+ks)) for i in range(w)]
        range_list_h  = [list(range(i,i+ks)) for i in range(h)]

        expand_h      = x1[:, :, range_list_h, :]

        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)
        expand_h_view = expand_h.view(b,c,expected_h,padded_w)
        expand_w      = expand_h_view[:,:, :, range_list_w].view(b,c,expected_h,expected_w)

        if not return_unflattened:
            return expand_w
        else:
            return expand_w, expand_w_

    def prep_feats(feat, ks):
        b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        _,feat_expanded = expand_map(feat,ks, True)
        feat_expanded   = feat_expanded.flatten(-2)
        feat_expanded   = feat_expanded.view(b,c,w*h,ks*ks).permute(0,2,3,1)
        return feat_reshaped, feat_expanded

    def bin_dice(predict,target,reduction='none'):
        p=2
        smooth = 1
        num = torch.sum(torch.mul(predict, target), dim=1) + smooth
        den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth
        loss = 1 - num / den
        return loss

    @torch.no_grad()
    def prepare_label(features_unlabeled,features_labeled,target_labeled,kernel_size):
        feat_un, feat_un_prep       = prep_feats(features_unlabeled,kernel_size)
        feat_lab, feat_lab_prep     = prep_feats(features_labeled,kernel_size)
        target_lab, target_lab_prep = prep_feats(target_labeled,kernel_size)

        init_distances = torch.cdist(feat_un,feat_lab)

        min_indx = init_distances.argmin(-1)

        target_lab_rearranged        = []
        features_lab_rearranged      = []
        target_lab_prep_rearranged   = []
        features_lab_prep_rearranged = []


        for i in range(min_indx.shape[0]):
            target_lab_rearranged        += [target_lab[i,min_indx[i]]]
            features_lab_rearranged      += [feat_lab[i,min_indx[i]]]
            target_lab_prep_rearranged   += [target_lab_prep[i,min_indx[i]]]
            features_lab_prep_rearranged += [feat_lab_prep[i,min_indx[i]]]

        target_lab_rearranged        = torch.stack(target_lab_rearranged,0)
        features_lab_rearranged      = torch.stack(features_lab_rearranged,0)
        target_lab_prep_rearranged   = torch.stack(target_lab_prep_rearranged,0)
        features_lab_prep_rearranged = torch.stack(features_lab_prep_rearranged,0)

        feat_dist = torch.cdist(feat_un_prep,features_lab_prep_rearranged)
        min_indx = feat_dist.argmin(-1)

        final_target = []
        for i in range(min_indx.shape[0]):
            okee = []
            for j in range(target_lab_prep_rearranged.shape[1]):
                okee += [target_lab_prep_rearranged[i,j,min_indx[i,j],:]]
            okee = torch.stack(okee,0)
            final_target += [okee]
        final_target = torch.stack(final_target,0)

        return final_target.permute(0,3,1,2).float()

    def prepare_predictions(pred_unlabeled,kernel_size):
        pred_un, pred_un_prep = prep_feats(pred_unlabeled,kernel_size)
        return pred_un_prep.permute(0,3,1,2)


    with torch.no_grad():
        target = prepare_label(features_unlabeled,features_labeled,target_labeled,kernel_size)

    pred   = prepare_predictions(pred_unlabeled,kernel_size)

    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,target, reduction='none')
    dice_loss = bin_dice(pred,target,reduction='none')
    loss = base_loss + dice_loss
    return loss


def assistance_fullbatch_loss(features_unlabeled, pred_unlabeled, features_labeled, target_labeled, kernel_size):
    def score_min(x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.min(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def score_max(x, dim, score):
        _tmp=[1]*len(x.size())
        _tmp[dim] = x.size(dim)
        return torch.gather(x,dim,score.max(
            dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim,0)

    def expand_map(x, ks, return_unflattened=False):
        b,c,h,w = x.shape

        x1 = torch.nn.functional.pad(x,(ks//2,ks//2,ks//2,ks//2))

        padded_w, padded_h     = w+ks-1, h+ks-1
        expected_w, expected_h = w*ks, h*ks

        range_list_w  = [list(range(i,i+ks)) for i in range(w)]
        range_list_h  = [list(range(i,i+ks)) for i in range(h)]

        expand_h      = x1[:, :, range_list_h, :]

        expand_w_     = expand_h[:, :, :, :, range_list_w]
        expand_w_     = expand_w_.permute(0,1,2,4,3,5)
        expand_h_view = expand_h.view(b,c,expected_h,padded_w)
        expand_w      = expand_h_view[:,:, :, range_list_w].view(b,c,expected_h,expected_w)

        if not return_unflattened:
            return expand_w
        else:
            return expand_w, expand_w_

    def prep_feats(feat, ks):
        b,c,w,h = feat.shape
        feat_reshaped   = feat.view(b,c,w*h).permute(0,2,1)
        _,feat_expanded = expand_map(feat,ks, True)
        feat_expanded   = feat_expanded.flatten(-2)
        feat_expanded   = feat_expanded.view(b,c,w*h,ks*ks).permute(0,2,3,1)
        return feat_reshaped, feat_expanded

    def prep_feats_fullbatch(feat, ks):
        b,c,w,h = feat.shape
        feat_reshaped   = feat.unsqueeze(0).permute(0,2,1,3,4).contiguous().view(1, c, b*w*h).permute(0,2,1)
        _,feat_expanded = expand_map(feat,ks, True)
        feat_expanded   = feat_expanded.flatten(-2)
        feat_expanded   = feat_expanded.unsqueeze(0).permute(0,2,1,3,4,5).contiguous().view(1, c, b*w*h,ks*ks).permute(0,2,3,1)
        return feat_reshaped, feat_expanded

    def bin_dice(predict,target,reduction='none'):
        p=2
        smooth = 1
        num = torch.sum(torch.mul(predict, target), dim=1) + smooth
        den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth
        loss = 1 - num / den
        return loss

    @torch.no_grad()
    def prepare_label(features_unlabeled,features_labeled,target_labeled,kernel_size):
        feat_un, feat_un_prep       = prep_feats_fullbatch(features_unlabeled,kernel_size)
        feat_lab, feat_lab_prep     = prep_feats_fullbatch(features_labeled,kernel_size)
        target_lab, target_lab_prep = prep_feats_fullbatch(target_labeled,kernel_size)

        init_distances = torch.cdist(feat_un,feat_lab)

        min_indx = init_distances.argmin(-1)

        target_lab_rearranged        = []
        features_lab_rearranged      = []
        target_lab_prep_rearranged   = []
        features_lab_prep_rearranged = []


        for i in range(min_indx.shape[0]):
            target_lab_rearranged        += [target_lab[i,min_indx[i]]]
            features_lab_rearranged      += [feat_lab[i,min_indx[i]]]
            target_lab_prep_rearranged   += [target_lab_prep[i,min_indx[i]]]
            features_lab_prep_rearranged += [feat_lab_prep[i,min_indx[i]]]

        target_lab_rearranged        = torch.stack(target_lab_rearranged,0)
        features_lab_rearranged      = torch.stack(features_lab_rearranged,0)
        target_lab_prep_rearranged   = torch.stack(target_lab_prep_rearranged,0)
        features_lab_prep_rearranged = torch.stack(features_lab_prep_rearranged,0)

        feat_dist = torch.cdist(feat_un_prep,features_lab_prep_rearranged)
        min_indx = feat_dist.argmin(-1)

        final_target = []
        for i in range(min_indx.shape[0]):
            okee = []
            for j in range(target_lab_prep_rearranged.shape[1]):
                okee += [target_lab_prep_rearranged[i,j,min_indx[i,j],:]]
            okee = torch.stack(okee,0)
            final_target += [okee]
        final_target = torch.stack(final_target,0)

        return final_target.permute(0,3,1,2).float()

    def prepare_predictions(pred_unlabeled,kernel_size):
        pred_un, pred_un_prep = prep_feats_fullbatch(pred_unlabeled,kernel_size)
        return pred_un_prep.permute(0,3,1,2)

    with torch.no_grad():
        target = prepare_label(features_unlabeled,features_labeled,target_labeled,kernel_size)

    pred   = prepare_predictions(pred_unlabeled,kernel_size)

    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,target, reduction='none')
    dice_loss = bin_dice(pred,target,reduction='none')
    loss = base_loss + dice_loss
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = edt(posmask) + edt(negmask)
    return res

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask
        neg_edt =  edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask

        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

class BinaryCrossentropy(nn.Module):
    def __init__(self, cf):
        super(BinaryCrossentropy,self).__init__()

    def forward(self, pred, target):
        """
            pred: (B,C,W,H)
            target: (B,W,H)
        """
        b,c,h,w = pred.shape
        loss = - target *  torch.log(pred.sigmoid()) - (1-target) * torch.log(1- pred.sigmoid())

        return loss.view(b,c, -1).mean(-1).mean(-1).mean(-1)

class Crossentropy(nn.Module):
    def __init__(self, cf):
        super(Crossentropy,self).__init__()
        self.cxloss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
            pred: (B,C,W,H)
            target: (B,W,H)
        """
        b,_,w,h = target.shape
        target = target.view(b,w,h)

        loss = self.cxloss(pred,target)
        return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, weight = None):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        predict = predict.sigmoid()

        if  weight is None:
            num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
            den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        else:
            weight = weight.contiguous().view(weight.shape[0], -1)
            num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
            den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = 1e-8
        self.p = 1

    def forward(self, predict, target, weight):
        # weight = (torch.ones(predict.shape).to(predict.device) * weight.unsqueeze(1)).flatten(2)
        predict = F.softmax(predict, dim=1).flatten(2)
        target  = F.one_hot(target.flatten(1), predict.shape[1]).permute(0,2,1)

        predict = predict[:,[i for i in range(predict.shape[1]) if i != self.ignore_index]] # * weight[:,[i for i in range(weight.shape[1]) if i != self.ignore_index]]
        target = target[:,[i for i in range(target.shape[1]) if i != self.ignore_index]] #  * weight[:,[i for i in range(weight.shape[1]) if i != self.ignore_index]]

        predict = predict.flatten(1)
        target = target.flatten(1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        return loss

class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False):
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DC_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super(DC_and_BD_loss, self).__init__()
        self.aggregate = aggregate
        self.bd = BDLoss(**bd_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, bound):
        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target, bound)
        if self.aggregate == "sum":
            result = dc_loss + bd_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class HDDTBinaryLoss(nn.Module):
    def __init__(self):
        """
        compute Hausdorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf
        """
        super(HDDTBinaryLoss, self).__init__()


    def forward(self, net_output, target):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        pc = net_output[:, 1, ...].type(torch.float32)
        gt = target[:,0, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.cpu().numpy()>0.5)
        # print('pc_dist.shape: ', pc_dist.shape)

        pred_error = (gt - pc)**2
        dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)
        hd_loss = multipled.mean()

        return hd_loss

class DC_and_HDBinary_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, hd_kwargs, aggregate="sum"):
        super(DC_and_HDBinary_loss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.hd = HDDTBinaryLoss(**hd_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        hd_loss = self.hd(net_output, target)
        if self.aggregate == "sum":
            result = dc_loss + hd_loss
        else:
            raise NotImplementedError("nah son")
        return result

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))

        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)

        dc = dc.mean()

        return -dc

class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class MultiClassFocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5, reduction='none'):
        super(MultiClassFocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.reduction = reduction

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # import pdb; pdb.set_trace()

        pt = logit.flatten(2).softmax(1)

        target_ = F.one_hot(target.flatten(1),num_classes=logit.shape[1]).permute(0,2,1)

        loss = F.cross_entropy(logit.flatten(2),target.flatten(1), reduction='none')

        loss = loss *  (torch.pow(1 - pt, self.gamma) * target_).sum(1)



        loss_weight = self.alpha * (loss)

        if loss.mean()!=loss.mean():
            import pdb; pdb.set_trace()
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor =  - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        input = flatten(net_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return  - 2. * intersect / denominator.clamp(min=self.smooth)

class SSLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Sensitivity-Specifity loss
        paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SSLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1 # weight parameter in SS paper

    def forward(self, net_output, gt, loss_mask=None):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - net_output)**2
        specificity_part = sum_tensor(squared_error*y_onehot, axes)/(sum_tensor(y_onehot, axes)+self.smooth)
        sensitivity_part = sum_tensor(squared_error*bg_onehot, axes)/(sum_tensor(bg_onehot, axes)+self.smooth)

        ss = self.r * specificity_part + (1-self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22

        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky

class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """
    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target) # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky

class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)# shape: (batch size, class num)
        weight = (self.beta**2)/(1+self.beta**2)
        asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()

        return -asym

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class PenaltyGDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result

class ExpLog_loss(nn.Module):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    """
    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3):
        super(ExpLog_loss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, net_output, target):
        dc_loss = -self.dc(net_output, target) # weight=0.8
        wce_loss = self.wce(net_output, target) # weight=0.2
        # with torch.no_grad():
        #     print('dc loss:', dc_loss.cpu().numpy(), 'ce loss:', ce_loss.cpu().numpy())
        #     a = torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma)
        #     b = torch.pow(-torch.log(torch.clamp(ce_loss, 1e-6)), self.gamma)
        #     print('ExpLog dc loss:', a.cpu().numpy(), 'ExpLogce loss:', b.cpu().numpy())
        #     print('*'*20)
        explog_loss = 0.8*torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + \
            0.2*wce_loss

        return explog_loss
