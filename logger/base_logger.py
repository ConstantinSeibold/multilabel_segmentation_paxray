from torch.utils.tensorboard import SummaryWriter
import seaborn as sns; sns.set()
import os, csv, sys,math,pdb
from skimage.color import label2rgb
import torch
import torchvision
import io
from PIL import Image, ImageDraw
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import shutil

class Base_Logger():
    def __init__(self,args):
        self.args = args
        self.log_dir = os.path.join(self.args.log_dir,self.args.name)
        print('Clearing Logdir: ',self.log_dir)
        self.remove_files()
        print('Values get logged at ',self.log_dir)
        self.logger = SummaryWriter(log_dir = self.log_dir)
        self.iter = 0
        self.epoch = 1
        self.batch_size = args.batch_size
        self.store_input_command()
        self.best = {'best':0, 'epoch':0}
        self.cam_vis_method = 'sigmoid'
        self.colormap_style = 'coolwarm'
        self.color_dict = self.generate_class_colors(args.classes)
        self.save_code()

    def remove_files(self):
        """
            Clear logdir to avoid multiple tensorboard logfiles
            this allows tensorboard to only access the most recent log file
            less of a mess in the visualization
        """
        filelist = glob.glob(os.path.join(self.log_dir, "*"))
        for f in filelist:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

    def save_code(self):
        """
            Stores code of base directory in checkpointsdir as zip
            to be able to reconstruct minor changes
        """
        zip_name = os.path.join(self.args.checkpoints_dir, self.args.name,'code')
        directory_name = './'
        shutil.make_archive(zip_name, 'zip', directory_name)

    def store_input_command(self):
        """
            Stores the input command as a source command as '.cmd' file
        """
        if not os.path.isdir(os.path.join(self.args.exp_dir,  self.args.exp_tag)):
            os.makedirs(os.path.join(self.args.exp_dir,  self.args.exp_tag))

        with open(os.path.join(self.args.exp_dir, self.args.exp_tag,self.args.name+'.cmd'),'w') as  f:
            f.write(' '.join(sys.argv))

    def add_losses(self, input_dict):
        """
            adds all losses with prefix 'loss' to tensorboard
        """
        for k in input_dict.keys():
            if 'loss' in  k:
                self.logger.add_scalar(k, input_dict[k].item(), self.iter * self.batch_size)

    def add_segmentations(self, input_dict):
        """
            adds segmentations to tensorboard
            dictionary requires segmentation predictions as 'segmentation_preds'
                                segmentation targets as 'mask_target'
        """
        if 'segmentation_preds' in input_dict.keys():
            input = input_dict['data'].cpu()
            mean,std = self.get_image_normalization()
            input = input*std+mean
            # import pdb; pdb.set_trace()
            if self.args.load_nc == 1:
                input = torch.cat([input,input,input],1)

            if self.args.seg_mode =='multiclass':
                seg = torch.tensor(label2rgb(input_dict['segmentation_preds'].cpu().numpy(), colors=self.color_dict, bg_label = 0)).permute(0,3,1,2).float()
                seg_gt = torch.tensor(label2rgb(input_dict['mask_target'].squeeze(1).cpu().numpy(), colors=self.color_dict, bg_label = 0)).permute(0,3,1,2).float()
                in_seg = torch.cat([input,seg, seg_gt],2)
            else:
                a = torch.arange(input_dict['mask_target'].shape[1]).view(1,-1,1,1).to(input_dict['mask_target'].device) + 1
                in_seg = torch.cat([input,torch.tensor(label2rgb(((torch.cat([input_dict['segmentation_preds'],input_dict['mask_target']],2))*a).sum(1).cpu().numpy(), colors=self.color_dict, bg_label = 0)).permute(0,3,1,2).float()],2)

            self.logger.add_image('segmentations', torchvision.utils.make_grid(in_seg[:self.args.displayed_imgs], nrow=self.args.displayed_imgs), self.iter * self.batch_size)
        else:
            pass

    def get_image_normalization(self,):
        if self.args.pt_pretrained:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)
        else:
            mean = torch.tensor([0.5]).view(1,-1,1,1)
            std = torch.tensor([0.5]).view(1,-1,1,1)
        return mean, std


    def add_networkparams(self, input_dict):
        """
            adds params to tensorboard
        """
        for k in input_dict.keys():
            if 'params' in  k:
                self.logger.add_scalar(k, input_dict[k].item(), self.iter * self.batch_size)

    def generate_class_colors(self, n_classes):
        """
            generate random list of colors as int-tuple based on number of classes
        """
        colors = [
            (249/255, 65/255, 68/255),
            (249/255, 108/255, 49/255),
            (248/255, 150/255, 30/255),
            (196/255, 170/255, 70/255),
            (144/255, 190/255, 109/255),
            (106/255, 180/255, 124/255),
            (67/255, 170/255, 139/255),
            (77/255, 144/255, 142/255),
            (87/255, 117/255, 144/255),
            (57/255, 72/255, 86/255),

        ]
        if n_classes<len(colors):
            color_dict = [colors[i] for i in list(range(0,len(colors),len(colors)//n_classes))]
        else:
            color_dict = []
            for n in range(n_classes):
                # import pdb; pdb.set_trace()
                color = tuple(np.random.choice(range(256), size=3).tolist())
                color_dict.append(color)
        # color_dict = [colors[i] for i in list(range(0,len(colors),len(colors)//n_classes))]
        return color_dict

    def get_seaborn_vis(self, tensor):
        """
            TODO
        """
        out_tensor = []
        tensor = tensor.detach()
        for b in range(tensor.shape[0]):
            array = tensor[b,0].cpu().numpy()

            fig = Figure()
            ax = sns.heatmap(array, cmap = self.colormap_style)
            fig.axes.append(ax)

            canvas = FigureCanvas(fig)
            canvas.draw()
            # import pdb; pdb.set_trace()
            s, (width, height) = canvas.print_to_buffer()
            img = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:,:,:3]

            img =  torch.tensor(img).permute(2,0,1)
            out_tensor += [img]
        out_tensor = torch.stack(out_tensor,0)
        tensor = torch.nn.functional.interpolat(out_tensor, (tensor.shape[:-2]))
        return tensor

    def add_results(self, metric_dict, mode, epoch=0):
        """
            add metric results to tensorboard
        """
        for k in metric_dict.keys():
            if 'metric' in  k:
                self.logger.add_scalar(k+'_'+mode, metric_dict[k].item(), self.epoch if epoch ==-1 else epoch)

            if 'figure' in k:
                self.logger.add_figure('Confusion matrix', metric_dict[k],self.epoch if epoch ==-1 else epoch)

    def log_results(self, metric_dict, mode, classes, epoch=-1):
        """
            store metric results as csv
        """
        lll = [self.args.exp_tag,self.args.name] + [str(metric_dict[i]) for i in metric_dict.keys()]
        os.makedirs(os.path.join(self.args.exp_dir, self.args.exp_tag,'results', self.args.name ), exist_ok=True)

        file = os.path.join(self.args.exp_dir, self.args.exp_tag,'results', self.args.name , mode+'.csv')
        if os.path.isfile(file):
            with open(file,'a') as  f:
                f.write('\t'.join(lll)+'\n')
        else:
            out_list = ['Experiment_Tag', 'Experiment_Name', 'Used_Network']
            key_list =list(metric_dict.keys())
            out_list +=  [classes[i]+'_'+ key_list[i] if i < len(classes) else key_list[i] for i in range(len(key_list))]
            with open(file,'w+') as  f:
                f.write('\t'.join(out_list)+'\n')
                f.write('\t'.join(lll)+'\n')

    def store_if_best(self, state,  eval_metrics, epoch):
        """
            store network is it got the best performance
        """
        if eval_metrics['metric_target'] > self.best['best']:
            print(' New Best Mean {:4f} == > Saving'.format(eval_metrics['metric_target']))
            self.best['best'] = eval_metrics['metric_target']
            self.best['epoch'] = epoch
            os.makedirs(os.path.join(self.args.checkpoints_dir, self.args.name),exist_ok=True)
            t = open(os.path.join(self.args.checkpoints_dir, self.args.name,'results.txt'),"a+")
            t.write("Value {} at Epoch {}\n".format(self.best['best'] , self.best['epoch']))

            self.store(state, 'best')
        else:
            pass

    def store(self, state, epoch):
        save_file = os.path.join(self.args.checkpoints_dir, self.args.name, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        torch.save(state, save_file)

    def add_gpu_usage(self,):
        self.logger.add_scalar('Peak Gpu Usage', torch.cuda.max_memory_allocated(device=None), self.iter * self.batch_size)


    def add_model_graph(self, model, input):
        self.logger.add_graph(model, input)

    def update(self, input_dict, epoch=-1,iteration=-1):
        """
            counter update
            if iteration == self.args.print_freq logger stores results
        """
        if iteration != -1:
            self.iter += iteration
        if epoch != -1:
            self.epoch = epoch

        if self.iter % self.args.print_freq == 0:
            self.add_losses(input_dict)
            # self.add_segmentations(input_dict)
            self.add_networkparams(input_dict)
            self.add_gpu_usage()

    def store_specific_sample(self, input_dict, epoch):
        sample_storage = os.path.join(self.log_dir, 'sample_storage')
        os.makedirs(sample_storage, exist_ok =True)

        if 'segmentation_preds' in input_dict.keys():
            if self.args.seg_mode =='multiclass':
                seg = torch.tensor(label2rgb(torch.cat([input_dict['segmentation_preds'],input_dict['mask_target']],0).cpu().numpy(), colors=self.color_dict, bg_label = 0)).permute(0,3,1,2).float()
            else:
                a = torch.arange(input_dict['mask_target'].shape[1]).view(1,-1,1,1).to(input_dict['mask_target'].device) + 1
                seg = torch.tensor(label2rgb(((input_dict['segmentation_preds'])*a).sum(1).cpu().numpy(), colors=self.color_dict, bg_label = 0)).permute(0,3,1,2).float()
            torchvision.utils.save_image(seg[:] , os.path.join(sample_storage,'segmentation_{}_{}.png'.format(self.args.store_specific_sample.split('/')[-1].split('.')[0], epoch)), nrow=self.args.displayed_imgs)


        for j in input_dict.keys():
            if 'data' in j:
                torchvision.utils.save_image(((input_dict[j]-input_dict[j].min())/(input_dict[j].max()-input_dict[j].min()))[:self.args.displayed_imgs], os.path.join(sample_storage,'{}_{}_{}.png'.format(j,self.args.store_specific_sample.split('/')[-1].split('.')[0], epoch)), nrow=self.args.displayed_imgs)
