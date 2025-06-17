import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import sys
import time
import datetime
import argparse
import json
import os
import random
import cv2
import matplotlib
import torch
import math
import sklearn
import seaborn
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import tqdm
# from torchlars import LARS
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA
import skimage
from skimage import morphology
from skimage.draw import disk
from skimage.measure import label, regionprops
from copy import deepcopy

from datasets import *
from models.objectness_net import ObjectnessNet, Binary_Classifier

sys.path.append('path to unMORE folder')
from utils.misc import batch_erode, NpEncoder
from utils.vis import *


K_list = [2,4,8,16]
PLACEHOLDER_IDX = 200

class ObjectnessNetTrainer:
    def __init__(self, args, device):


        self.args = args
        self.device = device  

        # Fix seeds. 
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ## Load data 
        if self.args.dataset=='ImageNet_votecut_top1_Dataset': 
            self.dataloaders = ImageNet_votecut_top1_Loader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        else:
            raise NotImplementedError
        

        print("Getting dataset ready...")
        print("Data shape: {}, color channel: {}".format(self.dataloaders.data_shape, self.dataloaders.color_ch))
        self.args.img_size = self.dataloaders.data_shape[1]
        setattr(self.args, 'log_every', min(self.args.log_every, len(self.train_loader)))

        self.model = ObjectnessNet(
            device=self.device,
            image_size=self.args.image_size,
            backbone_type=self.args.backbone_type,
            args=self.args,      
        )
        self.model = self.model.to(self.device)

        ### count trainable parameters
        def count_parameters(model):
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                # print(name, params)
                total_params += params
            print(f"Total Trainable Params: {total_params}")
            return total_params
        count_parameters(self.model)
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), self.args.learning_rate)
            # self.detection_optimizer = optim.Adam(self.detection_model.parameters(), self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), self.args.learning_rate, momentum=0.9, weight_decay=0.00005)
        elif self.args.optimizer == 'lars':
            self.optimizer = LARS(torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate))
        else:
            raise NotImplementedError

        print('lr_scheduler_milestones', self.args.lr_scheduler_milestones)
        if self.args.lr_scheduler_type == 'multi_step_lr':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_scheduler_milestones, gamma=self.args.lr_scheduler_gamma)
        elif self.args.lr_scheduler_type == 'exponential_lr':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR()
        else:
            raise NotImplementedError
        
        # Try to restore model and optimizer from checkpoint
        self.iter = 0
        if self.args.resume is not None:
            print(f"Restoring checkpoint from {self.args.resume}")
            self.checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
            try:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            except:
                print('Not load optimizer')
            self.iter = self.checkpoint['iter']
            # if 'lr_scheduler_state_dict' in self.checkpoint.keys():
            #     self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler_state_dict'])
        
        ## Setup ouput location
        if self.args.run_name is None:
            self.args.run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_' + args.dataset + '_' + args.backbone_type 
        if self.args.eval_mode:
            resumed_folder = self.args.resume.split('/ckpt/')[0]
            # self.result_folder = os.path.join(resumed_folder, 'evaluation_' + args.dataset)
            self.result_folder = os.path.join(resumed_folder, 'evaluation')
            if not os.path.isdir(self.result_folder):
                os.makedirs(self.result_folder)
            self.result_folder = os.path.join(self.result_folder, self.args.resume.split('/ckpt/')[1].split('.')[0])
            if not os.path.isdir(self.result_folder):
                os.makedirs(self.result_folder)
            self.result_folder = os.path.join(self.result_folder, self.args.run_name)
        else:
            self.result_folder = os.path.join('results_objectness', 'center_and_boundary', self.args.run_name)
        self.img_folder = os.path.join(self.result_folder, 'imgs')
        self.ckpt_folder = os.path.join(self.result_folder, 'ckpt')
        self.train_log_path = os.path.join(self.result_folder, 'train_log.json')
        

        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        if not os.path.isdir(self.img_folder):
            os.makedirs(self.img_folder)
        if not os.path.isdir(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)
        with open(os.path.join(self.result_folder, 'configs.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2) 
        print('result folder', self.result_folder)
        
    def train(self):
        self.model.train()
        
        if self.args.eval_mode:
            self.model.eval()
            self.visualize(self.test_loader, 'test')
            print('Finish evaluation')
            sys.exit()
        
        
        self.visualize(self.test_loader, 'test')
        self.visualize(self.train_loader, 'train')


        progress = None
        loss_list = []

        while self.iter <= self.args.train_iter and not self.args.eval_mode:
            print('number of steps per epoch', len(self.train_loader))
            self.model.train()

            if progress is None:
                progress = tqdm(total=self.args.log_every, desc='train from iter '+str(self.iter)+' lr='+str(self.optimizer.param_groups[0]['lr']), ncols=90)
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                gt_center_fields = labels['center_field']
                gt_center_fields = gt_center_fields.to(self.device) ## [B, 2, H, W]
                gt_sdf_maps = labels['sdf'].unsqueeze(1).to(self.device) ## [B, 1, H, W]
                gt_saliency_maps = labels['saliency_mask'].unsqueeze(1).to(self.device) ## [B, 1, H, W]

                # '''
                ## exclude BG images
                fg_flags = torch.where(torch.sum(gt_saliency_maps, dim=(1,2,3))>0, 1, 0).to(self.device)
                gt_center_fields = gt_center_fields[fg_flags==1]
                gt_sdf_maps = gt_sdf_maps[fg_flags==1]
                gt_saliency_maps = gt_saliency_maps[fg_flags==1]
                images = images[fg_flags==1]

                ## exclude all FG images
                all_fg_flags = torch.where(torch.sum((1-gt_saliency_maps), dim=(1,2,3))==0, 1, 0).to(self.device)
                gt_center_fields = gt_center_fields[all_fg_flags==0]
                gt_sdf_maps = gt_sdf_maps[all_fg_flags==0]
                gt_saliency_maps = gt_saliency_maps[all_fg_flags==0]
                images = images[all_fg_flags==0]

                ## make sure all images consist of both fg and bg
                gt_fg_maps = gt_saliency_maps ## [B, 1, H, W]
                gt_bg_maps = 1 - gt_saliency_maps ## [B, 1, H, W]
                assert torch.amin(torch.sum(gt_fg_maps, dim=(1,2,3))) > 0
                assert torch.amin(torch.sum(gt_bg_maps, dim=(1,2,3))) > 0
                # '''

                self.optimizer.zero_grad()
                out_dict = self.model(
                    images=images, 
                )
               
                loss = torch.tensor(0.0).to(self.device)

                pred_center_fields = out_dict['center_fields'] ## [B, 2, H, W]
                if self.args.center_field_loss_type == 'l2':
                    center_field_loss_map = (pred_center_fields - gt_center_fields)**2
                elif self.args.center_field_loss_type == 'l1':
                    center_field_loss_map = torch.abs(pred_center_fields - gt_center_fields)
                else:
                    raise NotImplementedError
                loss += center_field_loss_map.mean() 

                pred_sdf_maps = out_dict['sdf_maps'] ## [B, 1, H, W]
                if self.args.sdf_loss_type == 'l2':
                    sdf_loss_map = (pred_sdf_maps - gt_sdf_maps)**2
                elif self.args.sdf_loss_type == 'l1':
                    sdf_loss_map = torch.abs(pred_sdf_maps - gt_sdf_maps)
                else:
                    raise NotImplementedError
                loss += sdf_loss_map.mean() 

                if self.args.use_sdf_gradient_loss:
                    dy, dx = torchmetrics.functional.image_gradients(gt_sdf_maps)
                    gt_sdf_gradient_maps = torch.cat((dy, dx), dim=1) ## [B, 2, H, W]
                    gt_sdf_gradient_maps = gt_sdf_gradient_maps[:,:,0:-1,0:-1] ## discard the last row and the last column 
                    dy, dx = torchmetrics.functional.image_gradients(pred_sdf_maps)
                    pred_sdf_gradient_maps = torch.cat((dy, dx), dim=1) ## [B, 2, H, W]
                    pred_sdf_gradient_maps = pred_sdf_gradient_maps[:,:,0:-1,0:-1] ## discard the last row and the last column 
                    if self.args.sdf_loss_type == 'l2':
                        sdf_gradient_loss_maps = (gt_sdf_gradient_maps - pred_sdf_gradient_maps) ** 2
                    elif self.args.sdf_loss_type == 'l1':
                        sdf_gradient_loss_maps = torch.abs(gt_sdf_gradient_maps - pred_sdf_gradient_maps)
                    else:
                        raise NotImplementedError
                    loss += sdf_gradient_loss_maps.mean() 
                
                if self.args.use_sdf_binary_mask_loss:
                    pred_sdf_binary_mask = torch.sigmoid(pred_sdf_maps)
                    bce_loss = nn.BCELoss(reduction='mean')
                    sdf_binary_mask_loss_map = bce_loss(pred_sdf_binary_mask, gt_saliency_maps)
                    loss += sdf_binary_mask_loss_map.mean() 



                loss_list.append(loss.item())  
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            
                progress.update()

                self.iter += 1
                # Save checkpoints
                if self.iter % self.args.save_ckpt_every == 0:
                    checkpoint_name = os.path.join(self.ckpt_folder, f"iter_{self.iter}_model.ckpt")
                    print('* save checkpoint to', checkpoint_name)
                    ckpt_dict = {'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'iter': self.iter,
                                'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
                                }
                    torch.save(ckpt_dict, checkpoint_name)
                
                if self.iter % self.args.visualize_every == 0:
                    self.visualize(self.test_loader, 'test')
                    self.visualize(self.train_loader, 'train')


                if self.iter % self.args.log_every == 0:
                    if progress is not None:
                        progress.close()

                    avg_loss = sum(loss_list) / len(loss_list)
                    loss_list = []
                    
                    
                    if not os.path.isfile(self.train_log_path): 
                        new_data = {}
                        new_data[self.iter] = float(avg_loss)
                    else:
                        with open(self.train_log_path) as json_file:
                            new_data = json.load(json_file)
                            new_data[self.iter] = float(avg_loss)
                    with open(self.train_log_path, 'w') as f:
                        json.dump(new_data, f, indent=2)

                    progress = tqdm(total=self.args.log_every, desc='train from iter '+str(self.iter)+' lr='+str(self.optimizer.param_groups[0]['lr']), ncols=90)
    
    def center_field_to_anti_center_map(self, vote_maps, kernel_size=5):
        '''
        vote_maps: [B, 2, H, W]
        Sample hxw center proposals from HxW image. h = H / sample_step - 1, w = W / sample_step - 1
        For each proposal (xi, yi), select xi-half_patch_size:xi+half_patch_size, yi-half_patch_size:yi+half_patch_size
        Then calculate cosine similarity between GT voting field and predition voting filed for the selected patch
        return: [B, H, W]
        '''
        B, _, H, W = vote_maps.shape
        xv, yv = torch.meshgrid([torch.arange(kernel_size),torch.arange(kernel_size)])
        grid = torch.stack((xv, yv), 2).view((1, kernel_size, kernel_size, 2)).float() ## [1, kernel_size, kernel_size, 2]
        conv_filter = - grid.permute(0,3,1,2) + torch.tensor([int(kernel_size/2), int(kernel_size/2)]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) ## [1, 2, kernel_size, kernel_size]
        conv_filter = torch.nn.functional.normalize(conv_filter, dim=1).double()
        
        center_scoring = F.conv2d(vote_maps.double(), conv_filter.to(vote_maps.device), padding=int((kernel_size-1)/2))[:, 0, :, :]
        center_scoring = center_scoring / (kernel_size**2 - 1)
        return center_scoring

    def visualize(self, dataloader, dest_folder):
        self.model.eval()
        with torch.no_grad():
            for b_idx, (images, labels) in enumerate(dataloader):
                cur_image_size = images.shape[2]
                n_samples = min(images.shape[0], self.args.N_vis - b_idx * self.args.test_batch_size)
                if n_samples <= 0:
                    break
                if not os.path.isdir(os.path.join(self.img_folder, 'iter_{}'.format(self.iter))):
                    os.makedirs(os.path.join(self.img_folder, 'iter_{}'.format(self.iter))) 
                if not os.path.isdir(os.path.join(self.img_folder, 'iter_{}'.format(self.iter), dest_folder)):
                    os.makedirs(os.path.join(self.img_folder, 'iter_{}'.format(self.iter), dest_folder)) 
                result_folder = os.path.join(self.img_folder, 'iter_{}'.format(self.iter), dest_folder)

                images = images.to(self.device).float()
                
                ### Ground Truth Values ###
                gt_center_fields = labels['center_field'].cpu() ## [B, 2, H, W]
                gt_instance_masks = labels['instance_mask'].cpu() ## [B, H, W]
                gt_saliency_masks = labels['saliency_mask'].cpu() ## [B, H, W]
                if 'object_center' in labels.keys():
                    gt_object_centers = labels['object_center'].cpu() ## [B, 2]
                gt_center_fields_norm = torch.norm(gt_center_fields, dim=1) ## [B, H, W]
                gt_sdf_maps = labels['sdf'].cpu() ## [B, H, W]
                gt_sdf_insider_indicator = torch.nn.functional.normalize(gt_sdf_maps.unsqueeze(1)*1000, dim=1) ## [B, 1, H, W]
                gt_sdf_insider_indicator = torch.nan_to_num(gt_sdf_insider_indicator)
                dx, dy = torchmetrics.functional.image_gradients(gt_sdf_maps.unsqueeze(1))
                gt_sdf_gradient_maps = torch.cat((dx, dy), dim=1) ## [B, 2, H, W]
                gt_sdf_gradient_maps = gt_sdf_gradient_maps[:,:,0:-1,0:-1]
                gt_sdf_gradient_maps_norm = torch.norm(gt_sdf_gradient_maps, dim=1) ## [B, H-1, W-1]
                gt_sdf_gradient_maps_unit_length = torch.nn.functional.normalize(gt_sdf_gradient_maps, dim=1) ## [B, 2, H-1, W-1]
                gt_sdf_gradient_maps_unit_length = torch.nan_to_num(gt_sdf_gradient_maps_unit_length) 
                gt_sdf_gradient_maps_unit_length_with_indicator = gt_sdf_gradient_maps_unit_length * gt_sdf_insider_indicator[:,:,0:-1,0:-1]  ## [B, 2, H-1, W-1]
                gt_anti_center_maps = self.center_field_to_anti_center_map(gt_center_fields.to(self.device)).cpu() ## [B, H, W]

                ### Predicted Values ###
                prediction = self.model.get_prediction(images)
                ## Object Center Field
                pred_center_fields = prediction['center_fields'].cpu() ## [B, 2, H, W]
                pred_center_fields_norm = torch.norm(pred_center_fields, dim=1) ## [B, H, W]
                pred_anti_center_maps = self.center_field_to_anti_center_map(prediction['center_fields']).cpu() ## [B, H, W]
                pred_center_fields_unit_length = torch.nn.functional.normalize(pred_center_fields, dim=1) ## [B, 2, H, W]

                ## Object Boundary Distance Field
                pred_sdf_maps = prediction['sdf_maps'].cpu().squeeze(1) ## [B, H, W]
                pred_sdf_insider_indicator = torch.nn.functional.normalize(pred_sdf_maps.unsqueeze(1)*1000, dim=1) ## [B, 1, H, W]
                pred_sdf_insider_indicator = torch.nan_to_num(pred_sdf_insider_indicator)
                dx, dy = torchmetrics.functional.image_gradients(pred_sdf_maps.unsqueeze(1))
                pred_sdf_gradient_maps = torch.cat((dx, dy), dim=1) ## [B, 2, H, W]
                pred_sdf_gradient_maps_norm = torch.norm(pred_sdf_gradient_maps, dim=1) ## [B, H, W]
                pred_sdf_gradient_maps_unit_length = torch.nn.functional.normalize(pred_sdf_gradient_maps, dim=1) ## [B, 2, H, W]
                pred_sdf_gradient_maps_unit_length = torch.nan_to_num(pred_sdf_gradient_maps_unit_length)
                pred_sdf_gradient_maps_unit_length_with_indicator = pred_sdf_gradient_maps_unit_length * gt_sdf_insider_indicator
                pred_sdf_binary_masks = torch.where(torch.sigmoid(pred_sdf_maps)>0.5, 1, 0)
                pred_sdf_binary_masks_erode = batch_erode(pred_sdf_binary_masks, kernel_size=5, num_round=1)
                
                ## calculate binary mask from Object Center Field and Object Boundary Distance Field
                pred_sdf_binary_masks = torch.where(torch.sigmoid(pred_sdf_maps)>0.5, 1, 0)
                pred_center_fields_binary_masks = torch.where(pred_center_fields_norm>0.5, 1, 0)
                pred_union_masks = torch.where((pred_sdf_binary_masks+pred_center_fields_binary_masks)>0, 1, 0)
                # pred_bg_from_union_masks = 1 - pred_union_masks
                pred_union_masks_erode = batch_erode(pred_union_masks, kernel_size=9, num_round=3)

                    
                
                for idx in range(0, n_samples):
                    sample_idx = idx + b_idx * self.args.test_batch_size
                    input_image = cv2.cvtColor(images[idx].permute(1,2,0).cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(result_folder, str(sample_idx) + '_GT_label.png'), vis_GT_gray(gt_instance_masks[idx])*np.array(gt_instance_masks[idx]!=0)[:,:,None])
                    cv2.imwrite(os.path.join(result_folder, str(sample_idx) + '_input_image.png'), input_image)

                    ## 1. Visualize Object Center Field

                    ## 1.1 Ground Truth Object Center Field
                    self.visualize_saliency_arrow_map(arrow_map=gt_center_fields[idx].permute(1,2,0), saliency_mask=np.ones_like(gt_saliency_masks[idx]))
                    if 'object_center' in labels.keys():
                        plt.plot(gt_object_centers[idx][0],gt_object_centers[idx][1],'ro') 
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_center_fields.png'))
                    plt.clf()
                    ## 1.2 Predicted Object Center Field
                    self.visualize_saliency_arrow_map(arrow_map=pred_center_fields[idx].permute(1,2,0), saliency_mask=np.ones_like(gt_saliency_masks[idx]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_center_fields.png'))
                    plt.clf()
                    ## 1.3 Predicted Object Center Field (Direction Only)
                    self.visualize_saliency_arrow_map(arrow_map=pred_center_fields_unit_length[idx].permute(1,2,0), saliency_mask=np.ones_like(gt_saliency_masks[idx]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_center_fields_unit_length.png'))
                    plt.clf()
                    ## 1.4 Ground Truth Anti-Center Map
                    seaborn.heatmap(data=gt_anti_center_maps[idx])
                    if 'object_center' in labels.keys():
                        plt.plot(gt_object_centers[idx][0],gt_object_centers[idx][1],'go') 
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_center_scoring.png'))
                    plt.clf()
                    ## 1.5 Predicted Anti-Center Map
                    seaborn.heatmap(data=pred_anti_center_maps[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_center_scoring.png'))
                    plt.clf()
                    ## 1.6 Norm for Ground-Truth Object Center Field
                    seaborn.heatmap(data=gt_center_fields_norm[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_center_fields_norm.png'))
                    plt.clf()
                    ## 1.6 Norm for Predicted Object Center Field
                    seaborn.heatmap(data=pred_center_fields_norm[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_center_fields_norm.png'))
                    plt.clf()


                    ### 2. Visualize Object Boundary Field 

                    ## 2.1 Ground-Truth Object Boundary Field
                    seaborn.heatmap(data=gt_sdf_maps[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_sdf_map.png'))
                    plt.clf()
                    ## 2.2 Predicted Object Boundary Field
                    seaborn.heatmap(data=pred_sdf_maps[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_sdf_map.png'))
                    plt.clf()
                    ## 2.3 Norm for Ground-Truth Object Boundary Field
                    seaborn.heatmap(data=gt_sdf_gradient_maps_norm[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_sdf_gradient_maps_norm.png'))
                    plt.clf()
                    ## 2.4 Norm for Predicted Object Boundary Field
                    seaborn.heatmap(data=pred_sdf_gradient_maps_norm[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_sdf_gradient_maps_norm.png'))
                    plt.clf()
                    ## 2.5 Gradient for Ground-Truth Object Boundary Field
                    self.visualize_saliency_arrow_map(arrow_map=gt_sdf_gradient_maps_unit_length[idx].permute(1,2,0), saliency_mask=np.ones_like(gt_sdf_gradient_maps_unit_length[idx][0]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_sdf_gradient_maps.png'))
                    plt.clf()
                    ## 2.6 Gradient for Predicted Object Boundary Field
                    self.visualize_saliency_arrow_map(arrow_map=pred_sdf_gradient_maps_unit_length[idx].permute(1,2,0), saliency_mask=np.ones_like(pred_sdf_gradient_maps_unit_length[idx][0]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_sdf_gradient_maps.png'))
                    plt.clf()
                    ## 2.7 Gradient for Ground-Truth Object Boundary Field (adjusted with field sign)
                    self.visualize_saliency_arrow_map(arrow_map=gt_sdf_gradient_maps_unit_length_with_indicator[idx].permute(1,2,0), saliency_mask=np.ones_like(gt_sdf_gradient_maps_unit_length_with_indicator[idx][0]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_gt_sdf_gradient_maps_with_indicator.png'))
                    plt.clf()
                    ## 2.8 Gradient for Predicted Object Boundary Field (adjusted with field sign)
                    self.visualize_saliency_arrow_map(arrow_map=pred_sdf_gradient_maps_unit_length_with_indicator[idx].permute(1,2,0), saliency_mask=np.ones_like(pred_sdf_gradient_maps_unit_length_with_indicator[idx][0]))
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_sdf_gradient_maps_with_indicator.png'))
                    plt.clf()

                    ### 3. Visualize Binary Masks Calculated from Object Center Field and Object Boundary Field 
                    ## 3.1 Predicted Union Mask
                    seaborn.heatmap(data=pred_union_masks[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_union_mask.png'))
                    plt.clf()
                    ## 3.2 Predicted Union Mask (with the boundary of mask eroded)
                    seaborn.heatmap(data=pred_union_masks_erode[idx])
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_pred_union_masks_erode.png'))
                    plt.clf()

                    ### 4. Visualize Processed Anti-Center Map
                    pred_center_score_map_np = pred_anti_center_maps[idx].numpy()
                    erode_fg_mask = pred_union_masks_erode[idx].numpy()
                    center_score_fg = erode_fg_mask*pred_center_score_map_np 
                    center_score_fg[0:10, :] = 0
                    center_score_fg[-10:, :] = 0
                    center_score_fg[:, 0:10] = 0
                    center_score_fg[:, -10:] = 0
                    max_center_score_fg = np.amax(center_score_fg)
                    y_center, x_center = np.unravel_index(center_score_fg.argmax(), center_score_fg.shape)
                    seaborn.heatmap(data=center_score_fg)
                    plt.scatter(x_center, y_center, marker='o', s=100)
                    plt.axis('scaled')
                    plt.savefig(os.path.join(result_folder, str(sample_idx) + '_center_score_fg_'+str(max_center_score_fg)+'.png'))
                    plt.clf()

                plt.close()
        


    ## help function for visualization
    def visualize_saliency_arrow_map(self, arrow_map, saliency_mask, color=None, grid_size=5):
        '''
        arrow_map: [h, w, 2]
        saliency_mask: [H, W] (binary)
        center_of_mass: [2]
        '''         
        h,w,_ = arrow_map.shape
        H,W = saliency_mask.shape
        scale_ratio = H / h

        color = (0, 0, 255) if color is None else color
        X = []
        Y = []
        U = []
        V = []
        for i in range(0, h):
            for j in range(0, w):
                saliency_patch = saliency_mask[int(i*scale_ratio):int(i*scale_ratio)+1, int(j*scale_ratio):int(j*scale_ratio)+1]
                if saliency_patch.sum() == 0:
                    continue
                if i % grid_size == 0 and j % grid_size == 0:
                    # X.append(h-i)
                    X.append(i)
                    Y.append(j)
                    U.append(-arrow_map[i, j, 1])
                    V.append(arrow_map[i, j, 0])
        # plt.quiver(i, j, arrow_map[i, j, 1], arrow_map[i, j, 0],)
        plt.quiver(Y, X, U, V, scale=7.5)
        plt.gca().invert_yaxis()

class BinaryClassifierTrainer:
    def __init__(self, args, device):

        # if args.eval_mode:
        #     setattr(args, 'test_batch_size', 1)
        self.args = args
        self.device = device  

        # Fix seeds. 
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.dataloaders = ImageNet_votecut_labeled_classifier_Loader(args, torch.cuda.is_available()) 
        self.train_loader = self.dataloaders.train_loader
        self.test_loader = self.dataloaders.test_loader
        self.train_dataset_size = len(self.train_loader.dataset)
        self.test_dataset_size = len(self.test_loader.dataset)
        
        print("Getting dataset ready...")
        print("Data shape: {}, color channel: {}".format(self.dataloaders.data_shape, self.dataloaders.color_ch))
        self.args.img_size = self.dataloaders.data_shape[1]
        setattr(self.args, 'log_every', min(self.args.log_every, len(self.train_loader)))

        self.model = Binary_Classifier(device=self.device,
                image_size=self.args.image_size,
                args=self.args,  
                )
            
        self.model = self.model.to(self.device)
        ### count trainable parameters
        def count_parameters(model):
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                # print(name, params)
                total_params += params
            print(f"Total Trainable Params: {total_params}")
            return total_params
        count_parameters(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.learning_rate)
        print('lr_scheduler_milestones', self.args.lr_scheduler_milestones)
        if self.args.lr_scheduler_type == 'multi_step_lr':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_scheduler_milestones, gamma=self.args.lr_scheduler_gamma)
        elif self.args.lr_scheduler_type == 'exponential_lr':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR()
        else:
            raise NotImplementedError
        # Try to restore model and optimizer from checkpoint
        self.iter = 0
        if self.args.resume is not None:
            print(f"Restoring checkpoint from {self.args.resume}")
            self.checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
            try:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            except:
                print('Not load optimizer')
            self.iter = self.checkpoint['iter']
        if args.run_name is None:
            self.args.run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_' + args.dataset + '_' + args.backbone_type 
        if self.args.eval_mode:
            resumed_folder = self.args.resume.split('/ckpt/')[0]
            # self.result_folder = os.path.join(resumed_folder, 'evaluation_' + args.dataset)
            self.result_folder = os.path.join(resumed_folder, 'evaluation')
            if not os.path.isdir(self.result_folder):
                os.makedirs(self.result_folder)
            self.result_folder = os.path.join(self.result_folder, self.args.resume.split('/ckpt/')[1].split('.')[0])
            if not os.path.isdir(self.result_folder):
                os.makedirs(self.result_folder)
            self.result_folder = os.path.join(self.result_folder, self.args.run_name)
        else:
            self.result_folder = os.path.join('results_objectness', 'existence', self.args.run_name)
        self.img_folder = os.path.join(self.result_folder, 'imgs')
        self.ckpt_folder = os.path.join(self.result_folder, 'ckpt')
        self.train_log_path = os.path.join(self.result_folder, 'train_log.json')
        self.eval_log_path = os.path.join(self.result_folder, 'eval_log.json')
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        if not os.path.isdir(self.img_folder):
            os.makedirs(self.img_folder)
        if not os.path.isdir(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)
        with open(os.path.join(self.result_folder, 'configs.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2) 
    
    def train(self):
        self.model.train()
        if self.args.eval_mode:
            self.model.eval()
            self.evaluate_classification()
            print('Finish evaluation')
            sys.exit()

        progress = None
        loss_list = []
        self.loss_func = nn.BCELoss(reduction='mean')
        while self.iter <= self.args.train_iter and not self.args.eval_mode:
            print('number of steps per epoch', len(self.train_loader))
            self.model.train()

            if progress is None:
                progress = tqdm(total=self.args.log_every, desc='train from iter '+str(self.iter)+' lr='+str(self.optimizer.param_groups[0]['lr']), ncols=90)
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                class_labels = labels['class_label'].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(
                    images=images, 
                )
               
                loss = self.loss_func(predictions, class_labels)

                loss_list.append(loss.item())  
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            
                progress.update()

                self.iter += 1

                # Save checkpoints
                if self.iter % self.args.save_ckpt_every == 0:
                    checkpoint_name = os.path.join(self.ckpt_folder, f"iter_{self.iter}_model.ckpt")
                    print('* save checkpoint to', checkpoint_name)
                    ckpt_dict = {'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'iter': self.iter,
                                'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
                                }
                    torch.save(ckpt_dict, checkpoint_name)

                ## evaluate
                if self.iter % self.args.evaluate_every == 0:
                    if progress is not None:
                        progress.close()
                    self.evaluate_classification()
                
                if self.iter % self.args.log_every == 0:
                    if progress is not None:
                        progress.close()
                    avg_loss = sum(loss_list) / len(loss_list)
                    loss_list = []

                    if not os.path.isfile(self.train_log_path): 
                        new_data = {}
                        new_data[self.iter] = float(avg_loss)
                    else:
                        with open(self.train_log_path) as json_file:
                            new_data = json.load(json_file)
                            new_data[self.iter] = float(avg_loss)
                    with open(self.train_log_path, 'w') as f:
                        json.dump(new_data, f, indent=2)
                
                    progress = tqdm(total=self.args.log_every, desc='train from iter '+str(self.iter)+' lr='+str(self.optimizer.param_groups[0]['lr']), ncols=90)

    def evaluate_classification(self):
        self.model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(self.test_loader), desc='evalute from iter '+str(self.iter), ncols=90)
            total_samples = 0
            total_hit = 0
            if not os.path.isdir(os.path.join(self.img_folder, 'iter_{}'.format(self.iter))):
                os.makedirs(os.path.join(self.img_folder, 'iter_{}'.format(self.iter)))
            result_folder = os.path.join(self.img_folder, 'iter_{}'.format(self.iter))
            for b_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                gt_class_labels = labels['class_label'].to(self.device)
                predictions = self.model(
                    images=images, 
                )
                binary_prediction = torch.where(predictions>0.5, 1, 0)
                total_samples += predictions.shape[0]
                total_hit += (binary_prediction == gt_class_labels).sum().cpu().numpy()

                ## vis
                if b_idx == 0:
                    for img_idx, image in enumerate(images):
                        if img_idx > 64:
                            continue
                        gt_label = gt_class_labels[img_idx].cpu().numpy()[0]
                        pred_label = predictions[img_idx].cpu().numpy()[0]
                        input_image = cv2.cvtColor(image.permute(1,2,0).cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(result_folder, str(img_idx) + '_input_image_gt_'+str(gt_label)+'_pred_'+str(pred_label)+'.png'), input_image)
                progress.update()
            progress.close()
            print('acc =', str(total_hit) , '/', str(total_samples), '=', total_hit / total_samples)
            if not os.path.isfile(self.eval_log_path): 
                new_data = {}
                new_data[self.iter] = float(total_hit / total_samples)
            else:
                with open(self.eval_log_path) as json_file:
                    new_data = json.load(json_file)
                    new_data[self.iter] = float(total_hit / total_samples)
            with open(self.eval_log_path, 'w') as f:
                json.dump(new_data, f, indent=2)
        self.model.train()
                        


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_index", type=int, 
                        default=0, 
                        help="the index of gpu")
    parser.add_argument('--seed', type=int,
                        default=0, 
                        help='Seed for random number generators.')
    parser.add_argument("--run_name", type=str, 
                        default=None, 
                        help="Name of this job and name of results folder.")
    parser.add_argument("--save_ckpt_every", type=int, 
                        default=5000, 
                        help="Number of epochs between checkpoint saving.")
    parser.add_argument("--evaluate_loss_every", type=int, 
                        default=1000, 
                        help="Number of epochs between testset loss evaluation.")
    parser.add_argument("--evaluate_every", type=int, 
                        default=5000, 
                        help="Number of epochs between segmentation evaluation.")
    parser.add_argument("--visualize_every", type=int, 
                        default=5000, 
                        help="Number of epochs between visualization evaluation.")
    parser.add_argument("--log_every", type=int, 
                        default=50, 
                        help="Number of epochs between visualization evaluation.")
    parser.add_argument("--N_vis", type=int, 
                        default=10, 
                        help="Number of images to visualize.")
    parser.add_argument("--resume", type=str, 
                        default=None, 
                        help="Resume from a job if set true")
    parser.add_argument('--eval_mode', action='store_true',
                        help='only run evaluation. Default: False') 
    ## Optimization config 
    parser.add_argument("--train_iter", type=int, 
                        default=500000, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, 
                        default=16, 
                        help="Mini-batch size.")
    parser.add_argument("--test_batch_size", type=int, 
                        default=16, 
                        help="size of the testing batches")
    parser.add_argument("--learning_rate", type=float, 
                        default=0.0001, 
                        help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, 
                        default='multi_step_lr', help='other choice: exponential_lr')
    parser.add_argument('--lr_scheduler_milestones', nargs='+', type=int, default=[10000, 20000])
    parser.add_argument("--lr_scheduler_gamma", type=float, default=1, help="Learning rate.")
    parser.add_argument("--ema_lr", type=float, 
                        default=0.001, 
                        help="Learning rate.")
    parser.add_argument("--optimizer", type=str, 
                        default='adam', )


    
    ## dataset
    parser.add_argument('--dataset', type=str, default='ImageNet_votecut_top1_Dataset',
                        help='dataset identifier, e.g. ImageNet')
    parser.add_argument('--num_workers', type=int,
                        default=4, 
                        help='number of workers')
    parser.add_argument("--random_crop_scale_min", type=float, default=0.08) 
    parser.add_argument("--random_crop_scale_max", type=float, default=1)
    parser.add_argument('--image_size', type=int,
                        default=128, 
                        help='size of image from dataloader')
    ## model parameters
    
    parser.add_argument('--backbone_type', type=str,
                        help='backbone', default='dpt_large')    
    parser.add_argument("--sdf_activation", type=str, default=None) 
    parser.add_argument("--use_bg_sdf", action='store_true', help='define boundary distance field also on background if set True') 

    ## model specifics
    parser.add_argument("--sdf_loss_type", type=str, default='l1') 
    parser.add_argument("--center_field_loss_type", type=str, default='l2')   
    parser.add_argument('--use_sdf_gradient_loss', action='store_true', help='') 
    parser.add_argument("--use_sdf_binary_mask_loss", action='store_true', help='') 
    parser.add_argument("--train_center_and_boundary", action='store_true', help='') 
    parser.add_argument("--train_existence", action='store_true', help='') 

    
    
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    print('device', device)

    if args.train_center_and_boundary:
        objectness_net_trainer = ObjectnessNetTrainer(args, device)
        objectness_net_trainer.train()
    elif args.train_existence:
        classifier_trainer = BinaryClassifierTrainer(args, device)
        classifier_trainer.train()
    else:
        print('Please Specify Models To Be Trained.')

if __name__ == "__main__":
    main()