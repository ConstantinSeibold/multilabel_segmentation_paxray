import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.backbones import get_backbone
from typing import Tuple, List, Dict, Optional
from models.base_models import BaseModel
from models.model_utils.components import *


import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

from collections import OrderedDict

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7,  FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet
#


class SegFPNModel(BaseModel):
    def build(self):
        self.backbone  = get_backbone(self.cf)
        self.returned_layers = [1,2,3,4]
        self.return_layers_keys = [f'layer{k}' for v, k in enumerate(self.returned_layers)]
        self.fpn = get_fpn(self.backbone, returned_layers = self.returned_layers , out_channels = 256)
        self.seg_fpn = get_seg_fpn(self.backbone, returned_layers = self.returned_layers, in_ch = 256, out_ch = 128)
        self.pred_conv = nn.Conv2d(128, self.cf.classes, 1)
        self.dropout = nn.Dropout(p=0.2, inplace=False)

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
        backbone_dict = self.backbone(x)

        down1 = backbone_dict['feats_1_map']
        down2 = backbone_dict['feats_2_map']
        down3 = backbone_dict['feats_3_map']
        down4 = backbone_dict['feats_4_map']
        down5 = backbone_dict['feats_last_map']

        fpn_feats = self.fpn(
                        {
                            'layer1':down2,
                            'layer2':down3,
                            'layer3':down4,
                            'layer4':down5,
                        }
                        )

        seg_feats = self.seg_fpn(
            {
            self.return_layers_keys[i]:fpn_feats[self.return_layers_keys[i]] for i in range(len(self.return_layers_keys))
            }

        )

        seg_preds = self.pred_conv(seg_feats)

        return F.interpolate(seg_feats, x.shape[-2:], mode='bilinear'), F.interpolate(seg_preds, x.shape[-2:], mode='bilinear')

def get_fpn(backbone, returned_layers = None, norm_layer=misc_nn_ops.FrozenBatchNorm2d, out_channels= 512,trainable_layers=3):

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'layer0'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
	    if all([not name.startswith(layer) for layer in layers_to_train]):
	        parameter.requires_grad_(False)


    if returned_layers is None:
        returned_layers = [2, 3, 4]

    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]

    extra_blocks=LastLevelP6P7(out_channels, out_channels)

    fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks)

    return fpn

def get_seg_fpn(backbone, returned_layers = None, norm_layer=misc_nn_ops.FrozenBatchNorm2d, in_ch = 512, out_ch= 512,trainable_layers=3):

    # select layers that wont be frozen
    if returned_layers is None:
        returned_layers = [2, 3, 4]

    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': in_ch for v, k in enumerate(returned_layers)}
    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}


    return SegFeaturePyramidNetwork(return_layers, out_ch)

class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = F.interpolate
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32,out_channels),
            nn.ReLU(),
        ])


    def forward(self, x1):
        return self.up(self.conv(x1), scale_factor=2, mode='bilinear')


class SegFeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_dict: List[int],
        out_channels: int,
    ):
        super().__init__()
        in_channels_list = list(in_channels_dict.keys())

        modules = []

        for i in range(len(in_channels_list)):
            in_channels = in_channels_dict[in_channels_list[i]]

            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")

            if i == 0:
                modules += [[in_channels_list[i], nn.Conv2d(in_channels,out_channels, 3, padding=1)]]
            else:
                modules += [[in_channels_list[i], nn.Sequential(*[
                    ConvUpsample(in_channels if j == 0 else out_channels, out_channels) for j in range(i)
                ])]]


        self.module = nn.ModuleDict(modules)
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling

        prior = 0

        for k in x.keys():
            prior = self.module[k](x[k]) + prior

        return prior
