import torch.nn as nn
import torch
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
import models.model_utils.components as mutils
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel,self).__init__()
        self.mode = 'train'
        self.cf = args
        print('Begin Build')
        self.build()

    def forward(self, x):

        # ugly fix to handle list based dataloaders
        if self.cf.collate_mode == 'tensor' or type(x['data']) == type(torch.tensor(0)):
            return self.process(x)
        else:
            def normalize_box(boxes, scale, flip, shape):

                boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,shape[-1])
                boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,shape[0])

                if flip:
                    if boxes.shape[0]>0:
                        boxes[:,[0,2]] += 2*(shape[-1]//2 - boxes[:,[0,2]])
                        box_w = abs(boxes[:,0] - boxes[:,2])
                        boxes[:,0] -= box_w
                        boxes[:,2] += box_w

                if type(scale) == type(torch.tensor(0)):
                    scale = scale.cpu().numpy()
                if type(boxes) == type(torch.tensor(0)):
                    boxes = boxes.cpu().numpy()
                boxes = boxes/scale
                return boxes.astype(np.int16)

            def result_aggregation(data, reduction='mean'):
                if reduction == 'mean':
                    return torch.stack([torch.tensor(d).float() for d in data]).mean(0)
                elif reduction == 'sum':
                    return torch.stack([torch.tensor(d).float() for d in data]).sum(0)
                elif reduction == 'union':
                    return torch.cat([torch.tensor(d) for d in data],0)

            out = []
            for i in range(len(x['data'])):
                tmp_dict = {}
                for j in x.keys():
                    tmp_dict[j] = x[j][i]
                out.append({**self.process(tmp_dict),
                            **{'scale':tmp_dict['scale'],
                            'is_flip':tmp_dict['is_flip'],
                            'shape': torch.tensor(list(tmp_dict['data'].shape[-2:])).to(tmp_dict['data'].device)}})

            if not self.training:
                box_score = []
                class_logits = []
                boxes = []
                for i in range(len(x['data'])):
                    box_score.append(out[i]['meta_dict']['roi_score_final'])
                    class_logits.append(out[i]['class_logits'])
                    boxes.append(normalize_box(out[i]['meta_dict']['boxes'][0],
                                               out[i]['scale'], out[i]['is_flip'], out[i]['shape']))

                kek_score    = result_aggregation(box_score,'sum') # torch.stack(box_score,0).sum(0)
                box_score    = result_aggregation(box_score,'mean')
                class_logits = result_aggregation(class_logits,'mean') # torch.stack(class_logits,0).mean(0)
                boxes        = result_aggregation(boxes, 'mean')

                bboxes, labels, scores = self.get_predictions([boxes.to(kek_score.device)] , kek_score)
                # props = self.prep_props(tmp_dict)
                # bboxes, labels, scores = self.get_predictions( props['proposals_list_reduced'], kek_score)

                # self.filter_predictions(kek_score, props['proposals_list_reduced'][0], self.cf.iou_threshold, 0.001)

                # bboxes, labels, scores = self.get_predictions( props, box_score)

                return {
                            'class_logits': class_logits,
                            'pred_boxes'  : bboxes,
                            'pred_labels' : labels,
                            'pred_scores' : scores,
                            }


            full_out = {}
            for i in out[0].keys():
                if type(out[0][i]) == list:
                    full_out[i] = [
                                    torch.cat([out[j][i][k]
                                    for j in range(len(out))
                                    ],0)
                                    for k in range(len(out[0][i]))
                                    ]
                else:
                    full_out[i] = torch.cat([out[j][i] for j in range(len(out))],0)
            return full_out
