import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import get_backbone

from models.base_models import BaseModel
from models.model_utils.components import *


class BackboneUNet(BaseModel):
    def build(self):
        self.backbone  = get_backbone(self.cf)
        self.head      = get_unet_head(self.cf)
        self.dropout   = nn.Dropout(p=0.2, inplace=False)
        self.threshold = self.cf.threshold

    def get_results(self, x, features, orig_dict):
        out_dict = {
            'segmentation_logits': x,
        }

        if self.cf.seg_mode == 'binary':
            out_dict['segmentation_preds'] = (x.sigmoid()>self.threshold).long()
        else:
            out_dict['segmentation_preds'] = x.argmax(1)
        out_dict['segmentation_features'] = features
        out_dict = {**out_dict,**orig_dict}
        return out_dict

    def process(self,x):
        img = x['data']
        features, outs = self._forward(img)
        out_dict = self.get_results(outs, features, x)
        return out_dict

    def _forward(self, x ):
        backbone_dict = self.backbone(x)

        down1 = backbone_dict['feats_1_map']
        down2 = backbone_dict['feats_2_map']
        down3 = backbone_dict['feats_3_map']
        down4 = backbone_dict['feats_4_map']
        down5 = backbone_dict['feats_last_map']

        up1    = self.head.up1(down5, down4)
        up2    = self.head.up2(up1, down3)
        up3    = self.head.up3(up2, down2)
        up4    = self.head.up4(up3, down1)
        up     = F.interpolate(up4, x.shape[2:], mode='bilinear')
        logits = self.head.out(up)

        return up4, logits

def get_unet_head(cf):
    # R3D18 {last_map:512, first_map:64, second_map:64, first_map:64, third_map:128, forth_map:256}
    return UNetHead(2048, 128, cf.classes, norm = 'batch' if cf.batch_size >1 else 'instance')

class UNetHead(nn.Module):
    def __init__(self, in_channels, ngf, num_classes, norm='batch'):
        super(UNetHead, self).__init__()

        self.up1 = UpInit(in_channels //2, in_channels, in_channels //4, True)
        self.up2 = Up(in_channels // 4 * 2,  in_channels //8, True)
        self.up3 = Up(in_channels // 8 * 2,  in_channels // 32, True)
        self.up4 = Up(in_channels // 32 * 2,  ngf, True)

        self.out = OutConv(ngf, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class UNet(BaseModel):
    def build(self):
        self.n_channels = self.cf.input_nc
        self.n_classes = self.cf.classes

        self.inc = DoubleConv(self.n_channels, self.cf.ngf)
        self.down1 = Down(self.cf.ngf,  self.cf.ngf *2)
        self.down2 = Down( self.cf.ngf *2,  self.cf.ngf*4)
        self.down3 = Down( self.cf.ngf*4,  self.cf.ngf*8)
        factor = 2 if True else 1
        self.down4 = Down( self.cf.ngf*8,  self.cf.ngf*16 // factor)
        self.up1 = Up( self.cf.ngf*16,  self.cf.ngf*8 // factor, True)
        self.up2 = Up( self.cf.ngf*8,  self.cf.ngf*4 // factor, True)
        self.up3 = Up( self.cf.ngf*4,  self.cf.ngf *2 // factor, True)
        self.up4 = Up( self.cf.ngf *2, self.cf.ngf, True)
        self.outc = OutConv(self.cf.ngf, self.n_classes)

    def get_results(self, x, features, orig_dict):
        out_dict = {
            'segmentation_logits': x,
        }

        if self.cf.seg_mode == 'binary':
            out_dict['segmentation_preds'] = (x.sigmoid()>0.5).long()
        else:
            out_dict['segmentation_preds'] = x.argmax(1)
        out_dict['segmentation_features'] = features
        out_dict = {**out_dict,**orig_dict}
        return out_dict

    def process(self,x):
        img = x['data']
        features, outs = self._forward(img)
        out_dict = self.get_results(outs, features, x)
        return out_dict

    def _forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x, logits
