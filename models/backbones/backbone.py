import torch
import pdb,os
import torch.nn as nn
import torchvision
import torchvision.models as m
from collections import OrderedDict
BatchNorm = nn.BatchNorm2d
import torch.nn.functional as F



class backbone(nn.Module):
    def __init__(self, args):
        super(backbone, self).__init__()
        self.args = args
        self.backbone = self.get_backbone(args)

        if args.input_nc!=3:
            self.backbone[0] = torch.nn.Conv2d(args.input_nc,64,(7,7),stride=(2,2),padding=(3,3),bias=False)
            nn.init.normal_(self.backbone.conv1.weight, mean=0, std=0.01)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

    def get_backbone(self,args):
        if args.network == 'vgg16':
            full_net = getattr(m, args.network)(args.pt_pretrained)
            features = list(full_net.features)[:30]
            if args.freeze_backbone>0:
                self.freeze_parts(features,args.freeze_backbone)
            if args.dilation['use_dilation']:
                # 23 + 2*(+2)
                del features[int(args.dilation['position'][0])]
                for i in range(len(args.dilation['position'])):
                    index = min(i, len(args.dilation['strength'])-1)
                    features[int(args.dilation['position'][i])].dilation=(args.dilation['strength'][index],args.dilation['strength'][index])
                    features[int(args.dilation['position'][i])].padding=(args.dilation['strength'][index],args.dilation['strength'][index])
            net =  nn.Sequential(*features)

            self.classifier = nn.Sequential(*list(full_net.classifier)[:5])

        elif 'resnet' in args.network:
            full_net = getattr(m, args.network)(args.pt_pretrained)

            features = [
                            ('layer0',torch.nn.Sequential(*[full_net.conv1,
                            full_net.bn1,
                            full_net.relu,
                            full_net.maxpool])),
                            ('layer1',full_net.layer1),
                            ('layer2',full_net.layer2),
                            ('layer3',full_net.layer3),
                            ('layer4',full_net.layer4),
                        ]

            self.inplanes =  full_net.inplanes

            if args.dilation['use_dilation']:

                if 'layer1' in args.dilation['position']:
                    index = min(args.dilation['position'].index('layer1'), len(args.dilation['strength'])-1)
                    features[-4] = ('layer1', self.insert_dilations(features[-4][1], args.dilation['strength'][index]))
                if 'layer2' in args.dilation['position']:
                    index = min(args.dilation['position'].index('layer2'), len(args.dilation['strength'])-1)
                    features[-3] = ('layer2', self.insert_dilations(features[-3][1], args.dilation['strength'][index]))
                if 'layer3' in args.dilation['position']:
                    index = min(args.dilation['position'].index('layer3'), len(args.dilation['strength'])-1)
                    features[-2] = ('layer3', self.insert_dilations(features[-2][1], args.dilation['strength'][index]))
                if 'layer4' in args.dilation['position']:
                    index = min(args.dilation['position'].index('layer4'), len(args.dilation['strength'])-1)
                    features[-1] = ('layer4',self.insert_dilations(features[-1][1], args.dilation['strength'][index]))

                if args.dilation['specification'] == 'b':
                    from .backbone_resnet import Bottleneck, BasicBlock
                    # import pdb; pdb.set_trace()
                    layer1 = nn.Sequential(*[
                                nn.Conv2d(args.input_nc, 16, 7, padding=3, bias=False),
                                BatchNorm(16),
                                nn.ReLU(inplace=True),

                            ])

                    layer2 = self._make_layer(BasicBlock, 16, 16, 1, stride=1, residual=False, new_level=False)
                    layer3 = self._make_layer(BasicBlock, 16, 32, 1, stride=2, residual=False, new_level=False)
                    if type(features[1][1][0]) == torchvision.models.resnet.BasicBlock:
                        features[1] = ('layer1',self._make_layer(BasicBlock, 32, 64, 2, stride=2, residual=True))
                    else:
                        features[1] = ('layer1',self._make_layer(Bottleneck, 32, 64, 2, stride=2, residual=True))
                    layer6= self._make_layer(BasicBlock, 512*features[-1][1][0].expansion,512, 1, dilation=2,
                                 new_level=False, residual=True)
                    layer6 = self.insert_dilations(layer6, dilation_size=2)
                    layer7= self._make_layer(BasicBlock, 512,512, 1, dilation=1,
                                 new_level=False, residual=True)

                    del features[0]
                    features.insert(0,('layer0',layer1))
                    features.insert(1,('layer1',layer2))
                    features.insert(2,('layer2',layer3))
                    features[3] = ('layer3', features[3][1])
                    features[4] = ('layer4', features[4][1])
                    features[5] = ('layer5', features[5][1])
                    features[6] = ('layer6', features[6][1])
                    features.insert(len(features),('layer7',layer6))
                    features.insert(len(features),('layer8',layer7))

                elif args.dilation['specification'] == 'c':
                    from .backbone_resnet import Bottleneck, BasicBlock
                    # import pdb; pdb.set_trace()
                    layer1 = nn.Sequential(*[
                                nn.Conv2d(args.input_nc, 16, 7, padding=3, bias=False),
                                BatchNorm(16),
                                nn.ReLU(inplace=True),

                            ])

                    layer2 = self._make_layer(BasicBlock, 16, 16, 1, stride=1, residual=False, new_level=False)
                    layer3 = self._make_layer(BasicBlock, 16, 32, 1, stride=2, residual=False, new_level=False)
                    if type(features[1][1][0]) == torchvision.models.resnet.BasicBlock:
                        features[1] = ('layer1',self._make_layer(BasicBlock, 32, 64, len(features[1][1]), stride=2, residual=True))
                    else:
                        features[1] = ('layer1',self._make_layer(Bottleneck, 32, 64, len(features[1][1]), stride=2, residual=True))
                    layer6= self._make_layer(BasicBlock, 512*features[-1][1][0].expansion,512, 1, dilation=2,
                                 new_level=False, residual=False)
                    layer6 = self.insert_dilations(layer6, dilation_size=2)
                    layer7= self._make_layer(BasicBlock, 512,512, 1, dilation=1,
                                 new_level=False, residual=False)

                    del features[0]
                    features.insert(0,('layer0',layer1))
                    features.insert(1,('layer1',layer2))
                    features.insert(2,('layer2',layer3))
                    features[3] = ('layer3', features[3][1])
                    features[4] = ('layer4', features[4][1])
                    features[5] = ('layer5', features[5][1])
                    features[6] = ('layer6', features[6][1])
                    features.insert(len(features),('layer7',layer6))
                    features.insert(len(features),('layer8',layer7))

            if args.freeze_backbone>0:
                self.freeze_parts(features,args.freeze_backbone)
            net =  nn.Sequential(OrderedDict(features))
            if args.pt_pretrained and 'c' in args.dilation['specification']:
                net = self.load_model_from_url(net, args)
        else:
            raise '{} not implemented as BACKBONE Network'.format(args.network)

        return net

    def load_model_from_url(self, model, args):
        import torch.utils.model_zoo as model_zoo
        if args.network == 'resnet18' and (args.dilation['specification'] == 'c' or args.dilation['specification'] == 'b'):
            path = "dl.yf.io/drn/drn_c_26-ddedf421.pth"
        elif  args.network == 'resnet34' and (args.dilation['specification'] == 'c' or args.dilation['specification'] == 'b'):
            path = "dl.yf.io/drn/drn_c_42-9d336e8c.pth"
        elif  args.network == 'resnet50' and (args.dilation['specification'] == 'c' or args.dilation['specification'] == 'b'):
            path = "dl.yf.io/drn/drn_c_58-0a53a92c.pth"

        pre_trained_model = model_zoo.load_url(path)
        new=list(pre_trained_model.items())

        my_model_kvpair=model.state_dict()
        count=0
        for i,(key) in enumerate(my_model_kvpair):
            if 'num_batches_tracked' in key:
                continue
            else:
                layer_name, weights=new[count]
                my_model_kvpair[key]=weights
                count+=1
        model.load_state_dict(my_model_kvpair)
        print('loaded external model path')
        return model

    def freeze_parts(self, features, x):
        for layer in features[:x]:
            for p in layer.parameters():
                p.requires_grad = False

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def insert_dilations(self, features, dilation_size):
        feat = [f for f in features.children()]
        for f in feat:
            if not hasattr(f,'downsample'):
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1,1) :
                            tmp.stride   = 1
                            tmp.dilation = (int(dilation_size),int(dilation_size))
                            tmp.padding  = (int(dilation_size),int(dilation_size))
            elif f.downsample is None:
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1,1) :
                            tmp.stride   = 1
                            tmp.dilation = (int(dilation_size),int(dilation_size))
                            tmp.padding  = (int(dilation_size),int(dilation_size))
            else:
                if type(f) == torchvision.models.resnet.BasicBlock:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1,1) :
                                tmp.stride   = 1
                                tmp.dilation = (int(dilation_size),int(dilation_size))
                                tmp.padding  = (int(dilation_size),int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride= 1
                else:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1,1) :
                                tmp.stride   = 1
                                # tmp.dilation = (int(dilation_size),int(dilation_size))
                                # tmp.padding  = (int(dilation_size),int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride= 1
                                    # k.dilation = (1,1) # (int(dilation_size),int(dilation_size))
                                    # k.padding  = (0,0) # (int(dilation_size),int(dilation_size))
        return features

    def forward(self,x):
        if 'resnet' in self.args.network:
            return self.preset_forward(x)
        else:
            return self._forward(x)

    def _forward(self, x ):
        return self.backbone(x)

    def preset_forward(self, x,  insert_layer = None, return_layer = [1,2,3,4,5]):
        assert (insert_layer is None or return_layer is None or type(return_layer) is list or insert_layer < return_layer )

        if type(return_layer) is int:
            return_layer = [return_layer]
        result = OrderedDict()
        if insert_layer is None or insert_layer == 0:
            x = self.backbone[0](x)
        if  1 in return_layer:
            result['feats_{}_map'.format(1)] = x
            if 1 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 1:
            x = self.backbone[1](x)
        if  2 in return_layer:
            result['feats_{}_map'.format(2)] = x
            if 2 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 2:
            x = self.backbone[2](x)
        if  3 in return_layer:
            result['feats_{}_map'.format(3)] = x
            if 3 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 3:
            x = self.backbone[3](x)
        if  4 in return_layer:
            result['feats_{}_map'.format(4)] = x
            if 4 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 4:
            x = self.backbone[4](x)

        result['feats_last_map'] = x

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        result['feats_pooled']  = x

        return result
