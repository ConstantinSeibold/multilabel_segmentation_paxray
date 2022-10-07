import torch
from tqdm import tqdm
from utils.init_utils import *

import utils.data_utils as du
import sklearn.metrics as metrics

class EvaluationModule():
    def __init__(self,args):
        self.args = args
        self.task = args.task
        self.best = {'value':0.,'epoch':0}
        self.results = {}
        self.root_out_dir = os.path.join(self.args.output_dir, self.args.name)
        self.iou_threshold = args.iou_threshold
        self.score_threshold = args.score_threshold
        self.label2class = {}
        self.current_mode = 'None'

    def evaluate(self, model, dataloader):
        self.current_mode = dataloader.dataset.mode
        self.label2class = dataloader.dataset.label_to_class_dict
        metric_dict = {}

        if self.task in ['classification','segmentation']:
            metric_dict = self.evaluate_class( model, dataloader)
        if 'metric_target' not in metric_dict.keys():
            metric_dict['metric_target'] = torch.tensor(0)
        return metric_dict

    def aggregate_multiscale(self, x):
        class_logits = x['class_logits'].mean(0)
        import pdb; pdb.set_trace()

    def evaluate_class(self,model,dataloader):
        def prepare_targets(x):
            out = {}
            for i in x.keys():
                if 'class' in i or 'proposal' in i:
                    if type(x[i]) == type([]):
                        out[i] = torch.cat(x[i],0)
                    else:
                        out[i] = x[i]
                else:
                    out[i] = x[i]
            return out
        if isinstance(model,torch.nn.DataParallel):
            model = model.module
            model.eval()
        else:
            model.eval()

        with torch.no_grad():
            for i,ret in enumerate(tqdm(dataloader)):
                if type(self.args.load_size) == type([]):
                    class_logits = []
                    class_target = []
                    filename = []
                    for meta_i in range(len(ret['meta'])):
                        meta_ret = ret['meta'][meta_i]
                        pass_input = dict_to_cuda(meta_ret)
                        out = model(pass_input)
                        class_logits.append(out['class_logits'])
                        class_target.append(meta_ret['class_target'][-1])
                        filename.append(meta_ret['filename'][-1])
                    class_logits = torch.cat(class_logits, 0)
                    class_target = torch.cat(class_target, 0)
                    # filename = torch.cat(filename,0)
                    out = {
                            'class_logits': class_logits,
                            'class_target': class_target,
                            'filename'    : filename,
                        }
                    result_dict = self.compute_metrics(out)
                    self.store_results_intermediate(out)
                    self.aggregate_results(result_dict)
                else:
                    if self.args.store_eval \
                        or 'class_target' in ret.keys() if self.task == 'classification' \
                        else 'mask_target' in ret.keys():
                        pass_input = dict_to_cuda(ret)
                        out = model(pass_input)
                        out['filename'] = ret['filename']
                        out = {**out, **pass_input}
                        out = prepare_targets(out)
                        if not self.args.no_compute_eval:
                            result_dict = self.compute_metrics(out)
                            self.aggregate_results(result_dict)
                        self.store_results_intermediate(out)

        metric_dict = self.prepare_outputs()
        return metric_dict

    # Store Results

    def store_results_intermediate(self, out_dict):
        if self.args.store_eval:
            if self.task == 'segmentation':
                self.store_segmentation(out_dict)
        else:
            pass

    def store_results(self, out_dict):
        if self.args.store_eval:
            if self.task == 'segmentation':
                pass
        else:
            pass

    def store_segmentation(self, out_dict):
        if self.args.store_eval:
            if 'variable' == self.args.orig_size:
                segmentations = out_dict['segmentation_logits'].cpu().half().numpy()
            else:
                segmentations = torch.nn.functional.interpolate(out_dict['segmentation_logits'], self.args.orig_size, mode = 'nearest').cpu().half().numpy()
            if self.args.dim == '2d':
                for i in range(segmentations.shape[0]):
                    out_dir = os.path.join(self.root_out_dir, self.current_mode,str(self.args.val_freq),str(out_dict['filename'][i]))
                    counter = 1
                    while os.path.isdir(out_dir):
                        counter+=1
                        out_dir = os.path.join(self.root_out_dir, self.current_mode,str(self.args.val_freq*counter),str(out_dict['filename'][i]))

                    os.makedirs(out_dir ,exist_ok=True)

                    np.save(os.path.join(out_dir,'{}_predictions.npy'.format(out_dict['filename'][i])), segmentations[i:i+1])
            else:
                raise NameError('STORING SEGMENTATIONS for {} not yet implemented'.format(self.args.dim))

    def store_classification(self,out_dict):
        file_path =  os.path.join(self.root_out_dir, self.args.name, self.task)
        os.makedirs(file_path,exist_ok=True)
        file_path = os.path.join(file_path, 'preds_csv.csv')
        with open(file_path, 'a+') as f:
            shape = out_dict['class_logits'].shape
            for i in range(shape[0]):
                line = out_dict['filename'][i] if not isinstance(out_dict['filename'][i], type(torch.tensor(0))) else str(out_dict['filename'][i].item())
                line += '\t' + '\t'.join(['{:5f}'.format(out_dict['class_logits'][i][j].item()) for j in range(shape[1])]) + '\n'
                f.write(line)

                line = out_dict['filename'][i] if not isinstance(out_dict['filename'][i], type(torch.tensor(0))) else str(out_dict['filename'][i].item())
                line += '\t' + '\t'.join(['{:5f}'.format(out_dict['class_target'][i][j].item()) for j in range(shape[1])]) + '\n'
                f.write(line)

    def aggregate_results(self, out_dict):
        for k in out_dict.keys():
            if k in self.results.keys():
                self.results[k] += [out_dict[k]]
            else:
                self.results[k] = [out_dict[k]]

    def prepare_outputs(self):
        out_dict = {}


        if self.task=='segmentation':
            if self.args.eval_complete:
                pred = torch.cat(self.results['preds'],0)
                target = torch.cat(self.results['target'],0)
                iou = self.iou(pred,target)
                dice = self.dice(pred,target)
                for i in range(iou.shape[0]):
                    out_dict['iou_class_'+self.label2class[i]] = iou[i]
                    out_dict['dice_class_'+self.label2class[i]] = dice[i]
                out_dict['dice'] = torch.tensor([out_dict['dice_class_'+self.label2class[i]] for i in range(iou.shape[0]) if 'background' not in self.label2class[i]]).mean()
                out_dict['iou'] = torch.tensor([out_dict['iou_class_'+self.label2class[i]] for i in range(iou.shape[0]) if 'background' not in self.label2class[i]]).mean()
                out_dict['target'] = out_dict['iou']
            else:
                for k in self.results.keys():
                    mean = torch.cat(self.results[k],0).mean(0)
                    for i in range(mean.shape[0]):
                        out_dict[k+'_class_'+self.label2class[i]] = mean[i]
                    out_dict[k] = mean.mean()

        elif self.task == 'classification':
            aucs = self.aucs(self.results['preds'],self.results['target'])
            out_dict = aucs
        self.results = {}
        out_dict = {'metric_{}'.format(i):out_dict[i] for i in out_dict.keys()}
        return out_dict

    def compute_metrics(self, out_dict):
        if self.task == 'segmentation':
            # todo class-wise scores
            if self.args.eval_complete:
                pred = out_dict['segmentation_preds']
                target = out_dict['mask_target']
                return {
                        'preds': pred,
                        'target':target
                        }
            else:
                pred = out_dict['segmentation_preds']
                target = out_dict['mask_target']

                iou = self.iou(pred,target)
                dice = self.dice(pred, target)

                if self.label2class[0].lower() == 'background':
                    iou = iou[:,1:]
                    dice = dice[:,1:]

                return {
                        'target':dice,
                        'metric_dice':dice,
                        'metric_iou':iou
                        }
        elif self.task == 'classification':
            pred = out_dict['class_logits']
            target = out_dict['class_target']
            return {
                    'preds': pred,
                    'target':target
                    }
        else:
            return {}

    def aucs(self, outputs, targets):
        outputs = torch.cat(outputs,0).cpu().numpy()
        targets  = torch.cat(targets,0).cpu().numpy()
        n_classes = outputs.shape[1]
        fpr, tpr, aucs = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
            aucs[i] = auc(fpr[i], tpr[i])
        out_dict = {'auc_{}'.format(i):aucs[i] for i in aucs.keys()}
        out_dict['target'] = np.mean([aucs[i] for i in aucs.keys()])
        return out_dict

    def dice(self, pred, target):
        if self.args.seg_mode == 'binary':
            iou = self.iou(pred,target)

            dice = 2*iou/(1+iou)

            return torch.tensor(dice)
        else:
            iou = self.iou(pred,target)

            dice = 2*iou/(1+iou)

            return torch.tensor(dice)

    def iou(self, pred, target):
        if self.args.seg_mode == 'binary':
            SMOOTH = 1e-6
            pred = pred.flatten(-2)
            target = target.flatten(-2)

            inter = (pred*target).sum(-1)
            union = (pred+target).clamp(0,1).sum(-1) + SMOOTH

            iou = inter/union
            return iou
        else:
            pred = torch.nn.functional.one_hot(pred.long().flatten(),num_classes=self.args.classes)
            target = torch.nn.functional.one_hot(target.long().flatten(),num_classes=self.args.classes)
            inter = (pred*target)
            union = (pred+target).clamp(0,1)
            ious = inter.sum(0)/union.sum(0)
        return torch.tensor(ious)

    def to_numpy(self, array):
        if type(array) == torch.Tensor:
            array = array.cpu().numpy()
        return array


    def accuracy(self, scores, targets, k):
        """
        Computes top-k accuracy, from predicted and true labels.
        :param scores: scores from the model
        :param targets: true labels
        :param k: k in top-k accuracy
        :return: top-k accuracy
        """
        batch_size = targets.size(0)
        _, ind = scores.topk(k, 1, True, True)
        correct = ind.eq(targets.view(-1, 1).expand_as(ind))
        correct_total = correct.view(-1).float().sum()  # 0D tensor
        return correct_total.item() * (100.0 / batch_size)
