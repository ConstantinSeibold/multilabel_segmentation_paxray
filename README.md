# Multi-Label Segmentation for the Projected Anatomy X-Ray (PAXRay) dataset

\[[```arxiv```](https://arxiv.org/abs/2210.03416)\]\[[```supplementary```](https://github.com/ConstantinSeibold/constantinseibold.github.io/blob/master/pdfs/0058_Supplementary_Camera_Ready.pdf)\]\[[```project```](https://constantinseibold.github.io/paxray/)\]

## Dataset Access

<p>The dataset is available by clicking the folder:
<a href="https://drive.google.com/drive/folders/1rzlsZ0bfByRMBoywOPWZW08GNgIwCU9P?usp=sharing"><img src="https://github.com/ConstantinSeibold/constantinseibold.github.io/blob/master/_images/common/folder(1).png?raw=true" height="15"></a></p>

The file "paxray_dataset.zip"  contains the splits in the ".json" as well as the image (".png") and label (".npy") files. The "paxray.json" contains a mapping from label index and associated name and train/val/test splits.

## Model Training

Command lines for training a UNet with a ResNet backbone are stored in the run_unet*.cmd's which call the main.py. You wouldd have to adapt the ```--dataroot``` to the path where you unpacked the ```paxray_dataset.zip```. Files such as tensorboard files, checkpoints and results will be stored in ```--exp_dir```.

```python3 main.py --dataroot ./paxray --datafile paxray.json  --batch_size 3 --gpu_ids 0 --task segmentation --losses segmentation --input_nc 3 --load_nc 3 --partial_losses dice,ce --seg_mode binary --val_freq 10 --epochs 110 --learning_rate 0.001 --optim adamw --network resnet50 --model backbone_unet --fineSize 480 --load_size 512 --pt_pretrained --name paxray_full --save_freq 10 --print_freq 100 --decay_steps 60,90,100 --resize_prob 0.6 --resize_scale 1.2 --exp_tag backbone_unet```

## Pre-Trained Models

<p>The pre-trained models are available by clicking the respective link below:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>mIoU</th>
<th>Code</th>
<th>Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>UNet (100%)</td>
<td>60.6</td>
<td><a href="https://github.com/ConstantinSeibold/multilabel_segmentation_paxray">code</a></td>
<td><a href="https://drive.google.com/drive/folders/1JjWv_Ips_8CKbREk68JY-YMpu_lXu5Sa?usp=sharing">weights</a></td>
</tr>
<tr>
<td>UNet (50%)</td>
<td>57.07</td>
<td><a href="https://github.com/ConstantinSeibold/multilabel_segmentation_paxray">code</a></td>
<td><a href="https://drive.google.com/drive/folders/1JjWv_Ips_8CKbREk68JY-YMpu_lXu5Sa?usp=sharing">weights</a></td>
</tr>
<tr>
<td>UNet (25%)</td>
<td> 52.77 </td>
<td><a href="https://github.com/ConstantinSeibold/multilabel_segmentation_paxray">code</a></td>
<td><a href="https://drive.google.com/drive/folders/1JjWv_Ips_8CKbREk68JY-YMpu_lXu5Sa?usp=sharing">weights</a></td>
</tr>
</tbody>
</table>

## Prediction Inference

Command lines for training a UNet with a ResNet backbone are stored in the run_inference.cmd's which call the inference.py. You would have to adapt the ```--inference_folder``` to the folder containing the images which you want to predict for (only png and jpg files are used as input), ```--pred_folder``` to where you want to store the predictions (images are stored with the same filename but the ending is replaced with npy), and ```--resume``` to the trained model.


```python3 inference.py --inference_folder {folder containing images to predict} --pred_folder {folder to store predictions} --dataset folder --batch_size 1 --gpu_ids 0 --input_nc 3 --load_nc 3 --seg_mode binary --network resnet50 --model backbone_unet --load_size 512 --classes 166 --threshold 0.5 --resume {path to trained model}/ckpt_epoch_best.pth```

## Label Visualization

The ![notebook](https://github.com/ConstantinSeibold/multilabel_segmentation_paxray/blob/main/VisualizeLabels.ipynb) provides a script to visualize the anatomy labels of the dataset as well as predictions of a network. You would have to define the paths as below shows below:

<img src="https://github.com/ConstantinSeibold/multilabel_segmentation_paxray/blob/main/images/define_paths.png" width=80% height=auto>

You can then call the function ```visualize_certain_labels()``` as shown below by defining a list of desired labels as well as the image and numpy paths:

<img src="https://github.com/ConstantinSeibold/multilabel_segmentation_paxray/blob/main/images/labels.png" width=80% height=auto>


## Citation

<p>If you use this work or dataset, please cite:</p>
<pre><code class="lang-latex">@inproceedings{paxray,
    author    = {Seibold,Constantin <span class="hljs-keyword">and </span>Reiß,Simon <span class="hljs-keyword">and </span>Sarfraz,Saquib <span class="hljs-keyword">and </span>Fink,Matthias A. <span class="hljs-keyword">and </span>Mayer,Victoria <span class="hljs-keyword">and </span>Sellner,<span class="hljs-keyword">Jan </span><span class="hljs-keyword">and </span>Kim,Moon Sung <span class="hljs-keyword">and </span>Maier-Hein, Klaus H.  <span class="hljs-keyword">and </span>Kleesiek, <span class="hljs-keyword">Jens </span> <span class="hljs-keyword">and </span>Stiefelhagen,Rainer}, 
    title     = {Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding}, 
    <span class="hljs-keyword">booktitle </span>= {Proceedings of the <span class="hljs-number">33</span>th <span class="hljs-keyword">British </span>Machine Vision Conference (<span class="hljs-keyword">BMVC)},
</span>    year  = {<span class="hljs-number">2022</span>}
}
</code></pre>
