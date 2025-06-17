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
# from skimage.measure import label, regionprops
from copy import deepcopy
from scipy.ndimage import label, find_objects

from datasets import *
from models.objectness_net import ObjectnessNet, Binary_Classifier

sys.path.append('path to unMORE folder')
from utils.misc import batch_erode, NpEncoder
from utils.vis import *


class Object_Discovery:
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

        self.objectness_model = ObjectnessNet(
            device=self.device,
            image_size=self.args.image_size,
            backbone_type=self.args.backbone_type,
            args=self.args,      
        )
        self.binary_classifier_model = Binary_Classifier(device=self.device,
                image_size=self.args.image_size,
                args=self.args,  
                )
            
        self.objectness_model = self.objectness_model.to(self.device)
        print(f"Restoring objectness_model checkpoint from {self.args.objectness_resume}")
        self.objectness_model_checkpoint = torch.load(self.args.objectness_resume, map_location=self.device)
        self.objectness_model.load_state_dict(self.objectness_model_checkpoint['model_state_dict'], strict=True)
        # self.objectness_model = self.objectness_model.to(torch.float16)
        self.objectness_model = self.objectness_model.to(torch.float32)
        self.objectness_model.eval()
        for param in self.objectness_model.parameters():
            param.requires_grad = False
        
        self.binary_classifier_model = self.binary_classifier_model.to(self.device)
        print(f"Restoring binary_classifier_model checkpoint from {self.args.binary_classifier_resume}")
        self.binary_classifier_model_checkpoint = torch.load(self.args.binary_classifier_resume, map_location=self.device)
        self.binary_classifier_model.load_state_dict(self.binary_classifier_model_checkpoint['model_state_dict'], strict=True)
        # self.objectness_model = self.objectness_model.to(torch.float16)
        self.binary_classifier_model = self.binary_classifier_model.to(torch.float32)
        self.binary_classifier_model.eval()
        for param in self.binary_classifier_model.parameters():
            param.requires_grad = False


        if self.args.dataset == 'COCO':
            self.test_dataset = COCO_Dataset(image_size=self.args.image_size, split=self.args.dataset_split, args=self.args)
        else:
            raise NotImplementedError
        
        # self.current_round = 0

        if args.run_name is None:
            self.args.run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_' + self.args.dataset + '_' + self.args.dataset_split
        if self.args.start_idx != -1 and self.args.end_idx != -1:
            self.args.run_name +=  '_' + str(self.args.start_idx) + "_" + str(self.args.end_idx)

        self.result_folder = os.path.join('results_reasoning', self.args.run_name)
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        print('result_folder', self.result_folder)
        with open(os.path.join(self.result_folder, 'configs_object_reasoning.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2) 

    @staticmethod
    def generate_random_proposal(height, width):
        grid_size_list = [32, 64, 128, 256, 512]
        bboxes_list = []
        for grid_size in grid_size_list:
            center_h_list = list(np.arange(0, height, grid_size, dtype=int))
            center_w_list = list(np.arange(0, width, grid_size, dtype=int))
            x_centers, y_centers = np.meshgrid(center_w_list, center_h_list)
            x_centers = x_centers.flatten()
            y_centers = y_centers.flatten()
            box_size = grid_size 
            base_anchors = np.array([
                [-box_size, -box_size, box_size, box_size], 
                [-grid_size/2, -grid_size, grid_size/2, grid_size],
                [-grid_size, -grid_size/2, grid_size, grid_size/2]
            ])
            centers = np.stack([x_centers, y_centers, x_centers, y_centers]) ## [4, 200]
            centers = centers.transpose() ## [200, 4]
            bboxes = (centers.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4) ## [1000, 4]
            bboxes = np.array(bboxes)
            bboxes_list.append(bboxes)
        
        out = np.concatenate(bboxes_list, axis=0)
        out[:, 0][out[:, 0]<0] = 0
        out[:, 1][out[:, 1]<0] = 0
        out[:, 2][out[:, 2]>=width] = width
        out[:, 3][out[:, 3]>=height] = height
        out = np.concatenate((out, [[0, 0, width, height]]), axis=0)
        return out
    
    @staticmethod
    def update_bbox_with_boundary_fields(sdf_maps):
        '''
        sdf_maps: [B,H,W]
        '''
        B, H, W = sdf_maps.shape
        max_step_size = np.sqrt(H**2+W**2)
        dy, dx = torchmetrics.functional.image_gradients(sdf_maps.unsqueeze(1))
        sdf_gradient_maps = torch.cat((dy, dx), dim=1) ## [B, 2, H, W]
        sdf_gradient_maps = sdf_gradient_maps[:,:,0:-1,0:-1] ## [B, 2, H-1, W-1] the last row and the last column are not valid
        sdf_maps = sdf_maps[:,0:-1,0:-1] ## [B, H-1, W-1]
        sdf_gradient_maps_norm = torch.norm(sdf_gradient_maps, dim=1) ## [B, H-1, W-1]

        # ## option 2: weight avg step size
        soft_fg_mask = torch.sigmoid(sdf_maps) ## [B, H-1, W-1]
        soft_bg_mask = 1 - soft_fg_mask
        avg_sdf_gradient_norm_fg = (soft_fg_mask * sdf_gradient_maps_norm).sum(-1).sum(-1) / (soft_fg_mask.sum(-1).sum(-1) + 1e-8) ## [B]
        avg_sdf_gradient_norm_bg = (soft_bg_mask * sdf_gradient_maps_norm).sum(-1).sum(-1) / (soft_bg_mask.sum(-1).sum(-1) + 1e-8) ## [B]
        avg_step_size_fg = 1/(avg_sdf_gradient_norm_fg+1e-10) ## [B]
        avg_step_size_bg = 1/(avg_sdf_gradient_norm_bg+1e-10) ## [B]
        step_size_maps = avg_step_size_fg.unsqueeze(1).unsqueeze(1) * soft_fg_mask + avg_step_size_bg.unsqueeze(1).unsqueeze(1) * soft_bg_mask ## [B, H-1, W-1]

        movement = step_size_maps * sdf_maps ## [B, H-1, W-1]

        ## x1
        delta_x1 = torch.amax(movement[:,:,0], dim=1) ## [B, H-1] --> [B]
        delta_x1 = delta_x1 * (-1)
        ## y1 
        delta_y1 = torch.amax(movement[:,0,:], dim=1) ## [B, W-1] --> [B]
        delta_y1 = delta_y1 * (-1)
        ## x2
        delta_x2 = torch.amax(movement[:,:,-1], dim=1) ## [B, H-1] --> [B]
        ## y2
        delta_y2 = torch.amax(movement[:,-1,:], dim=1) ## [B, W-1] --> [B]

        return delta_x1, delta_y1, delta_x2, delta_y2

    @staticmethod
    def post_process_bbox_update(original_bboxes, delta_bboxes, delta_scale_x=128, delta_scale_y=128):
        '''
        128x128 -> proposal size in the original image
        original_bboxes: [B, 4], [x1,y1,x2,y2], in the scale of original image size
        delta_bboxes: [B, 4], [delta_x1,delta_y1,delta_x2,delta_y2] in the scale of 128x128
        delta_scale_x, delta_scale_y: the size of the SDF where delta values are calculated
        '''

        original_x_range = original_bboxes[:, 2] - original_bboxes[:, 0] ## [B]
        original_y_range = original_bboxes[:, 3] - original_bboxes[:, 1] ## [B]
        x_ratio =  original_x_range / delta_scale_x ## [B]
        y_ratio = original_y_range / delta_scale_y ## [B]

        updated_bboxes = original_bboxes.clone()
        updated_bboxes[:, 0] = original_bboxes[:, 0] + delta_bboxes[:, 0] * x_ratio
        updated_bboxes[:, 1] = original_bboxes[:, 1] + delta_bboxes[:, 1] * y_ratio
        updated_bboxes[:, 2] = original_bboxes[:, 2] + delta_bboxes[:, 2] * x_ratio
        updated_bboxes[:, 3] = original_bboxes[:, 3] + delta_bboxes[:, 3] * y_ratio

        return updated_bboxes

    @staticmethod
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
    
    @staticmethod
    def separate_connected_components(binary_masks):
        """
        Separate binary masks into connected components and return their bounding boxes.

        Parameters:
            binary_masks (torch.Tensor): Input binary masks of shape (B, H, W).

        Returns:
            dict: Combined bounding boxes with keys 'single' and 'multi'.
            list: Binary indicator list suggesting which masks have only one connected component.
        """
        batch_size = binary_masks.shape[0]
        combined_proposals = {
            'single': [],
            'multi': []
        }
        single_component_indicators = []

        for b in range(batch_size):
            binary_mask = binary_masks[b].cpu().numpy()  # Convert to numpy array

            # Label connected components
            structure = np.ones((3, 3), dtype=int)  # Define connectivity
            labeled_mask, num_features = label(binary_mask, structure)

            # Get bounding boxes
            single_component_proposals = None
            multi_component_proposals = []

            for i in range(1, num_features + 1):
                # Find bounding box
                slices = find_objects(labeled_mask == i)
                if slices:
                    y_slice, x_slice = slices[0]  # Get the slices for the component
                    bbox = [x_slice.start, y_slice.start, x_slice.stop, y_slice.stop]  # [x1, y1, x2, y2]

                    if num_features == 1:
                        single_component_proposals = bbox
                    else:
                        multi_component_proposals.append(bbox)

            if single_component_proposals is not None:
                combined_proposals['single'].append(single_component_proposals)
                single_component_indicators.append(1)  # Indicate single component
            else:
                single_component_indicators.append(0)  # Indicate multiple components

            combined_proposals['multi'].extend(multi_component_proposals)

        return combined_proposals, single_component_indicators

    @staticmethod
    def enlarge_proposals(proposals, image_shape, ratio):
        """
        Enlarge bounding boxes by a common ratio, ensuring they stay within the image dimensions.

        Parameters:
            proposals (list): List of bounding boxes as [x1, y1, x2, y2].
            image_shape (tuple): Shape of the image as (height, width).
            ratio (float): Ratio by which to enlarge the boxes.

        Returns:
            list: List of enlarged bounding boxes.
        """
        enlarged_boxes = []
        height, width = image_shape

        for bbox in proposals:
            x1, y1, x2, y2 = bbox

            # Calculate the center and width/height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            new_width = (x2 - x1) * ratio
            new_height = (y2 - y1) * ratio

            # Calculate new bounding box
            new_x1 = int(max(center_x - new_width / 2, 0))
            new_y1 = int(max(center_y - new_height / 2, 0))
            new_x2 = int(min(center_x + new_width / 2, width))
            new_y2 = int(min(center_y + new_height / 2, height))

            enlarged_boxes.append([new_x1, new_y1, new_x2, new_y2])

        return enlarged_boxes

    def filter_small_proposal(self, proposals, labels):
        x_ranges = proposals[:, 2] - proposals[:, 0] 
        y_ranges = proposals[:, 3] - proposals[:, 1] 
        proposal_areas = x_ranges * y_ranges 
        proposals = proposals[proposal_areas>self.args.proposal_area_thres]
        labels = labels[proposal_areas>self.args.proposal_area_thres]
        return proposals, labels

    def get_prediction_with_proposals(self, proposals, image):
        self.objectness_model.eval()
        self.objectness_model = self.objectness_model.to(self.device)
        proposal = proposals.to(self.device)
        # num_img_per_batch = 100
        num_img_per_batch = 50
        on_edge_flag_list = []
        predictions_sdf_maps_list = []
        predictions_center_fields_list = []
        all_cropped_image_list = []
        for batch_idx in range(0, math.ceil(len(proposals)/num_img_per_batch)):
            batch_proposal = proposals[batch_idx*num_img_per_batch:(batch_idx+1)*num_img_per_batch]
            cropped_image_list = []
            for box in batch_proposal:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))
                on_edge_flag = np.array([x1==0, y1==0, x2==image.shape[-1], y2==image.shape[-2]])
                on_edge_flag_list.append(torch.tensor(on_edge_flag))
                resize = transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                cropped_image = resize(image[:,y1:y2, x1:x2])
                cropped_image_list.append(cropped_image)
                all_cropped_image_list.append(cropped_image)
            cropped_images = torch.stack(cropped_image_list, dim=0) ## [num_img_per_batch, 3, H, W]
            with torch.no_grad():
                # predictions_no_grad = objectness_model(cropped_images.to(torch.float16).to(device))
                predictions_no_grad = self.objectness_model(cropped_images.to(torch.float32).to(self.device))
            proposal_sdf_masks = predictions_no_grad['sdf_maps'].squeeze(1)
            predictions_sdf_maps_list.append(proposal_sdf_masks)
            proposal_center_fields = predictions_no_grad['center_fields']
            predictions_center_fields_list.append(proposal_center_fields)
    
        predictions_sdf_maps = torch.cat(predictions_sdf_maps_list, dim=0) ## [2000, H, W]
        predictions_center_fields = torch.cat(predictions_center_fields_list, dim=0) ## [2000, 2, H, W]
        all_cropped_image = torch.stack(all_cropped_image_list)
        on_edge_flags = torch.stack(on_edge_flag_list, dim=0) ## [2000, 4]

        return predictions_sdf_maps, predictions_center_fields

    def get_prediction_with_proposal_images(self, proposal_images):
        self.objectness_model.eval()
        self.objectness_model = self.objectness_model.to(self.device)

        predictions_sdf_maps_list = []
        predictions_center_fields_list = []
        num_img_per_batch = 50
        for batch_idx in range(0, math.ceil(len(proposals)/num_img_per_batch)):
            batch_proposal = proposals[batch_idx*num_img_per_batch:(batch_idx+1)*num_img_per_batch]
            batch_images = proposal_images[batch_idx*num_img_per_batch:(batch_idx+1)*num_img_per_batch]

            with torch.no_grad():
                predictions_no_grad = self.objectness_model(batch_images.to(torch.float32).to(self.device))
            proposal_sdf_masks = predictions_no_grad['sdf_maps'].squeeze(1)
            predictions_sdf_maps_list.append(proposal_sdf_masks)
            proposal_center_fields = predictions_no_grad['center_fields']
            predictions_center_fields_list.append(proposal_center_fields)
        predictions_sdf_maps = torch.cat(predictions_sdf_maps_list, dim=0) ## [2000, H, W]
        predictions_center_fields = torch.cat(predictions_center_fields_list, dim=0) ## [2000, 2, H, W]
        return predictions_sdf_maps, predictions_center_fields

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
        # conv_filter =  grid.permute(0,3,1,2) - torch.tensor([int(kernel_size/2), int(kernel_size/2)]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) ## [1, 2, kernel_size, kernel_size]
        conv_filter = - grid.permute(0,3,1,2) + torch.tensor([int(kernel_size/2), int(kernel_size/2)]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) ## [1, 2, kernel_size, kernel_size]
        conv_filter = torch.nn.functional.normalize(conv_filter, dim=1).double()
        center_scoring = F.conv2d(vote_maps.double(), conv_filter.to(vote_maps.device), padding=int((kernel_size-1)/2))[:, 0, :, :]
        # center_scoring = center_scoring / torch.amax(center_scoring.reshape(B, -1), dim=1)
        center_scoring = center_scoring / (kernel_size**2 - 1)
        return center_scoring

    def optimize_one_image_single_round(self, image, proposals, labels):
        '''
        INPUT: 
            image: [3, H, W]
            proposals: [N, 4]
        OUTPUT: 
            updated_bboxes: [N, 4]
            objectness_scores: [N]
        '''
        self.objectness_model.eval()
        for param in self.objectness_model.parameters():
            param.requires_grad = False

        out_bboxes = torch.zeros_like(proposals).to(torch.float32)
        labels = torch.zeros((len(proposals))).to(proposals.device)

        predictions_sdf_maps_list = []
        all_on_edge_flag_list = []
        num_img_per_batch = 50
        for batch_idx in range(0, math.ceil(len(proposals)/num_img_per_batch)):
            batch_proposal = proposals[batch_idx*num_img_per_batch:(batch_idx+1)*num_img_per_batch]
            # all_cropped_image_list = []
            cropped_image_list = []
            for box in batch_proposal:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))
                on_edge_flag = np.array([x1==0, y1==0, x2==image.shape[-1], y2==image.shape[-2]])
                all_on_edge_flag_list.append(torch.tensor(on_edge_flag))
                resize = transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                cropped_image = resize(image[:,y1:y2, x1:x2])
                cropped_image_list.append(cropped_image)
            cropped_images = torch.stack(cropped_image_list, dim=0) ## [num_img_per_batch, 3, H, W]
            
            with torch.no_grad():
                predictions_no_grad = self.objectness_model(cropped_images.to(torch.float32).to(self.device))
            proposal_sdf_masks = predictions_no_grad['sdf_maps'].squeeze(1)
            predictions_sdf_maps_list.append(proposal_sdf_masks)
        
        predictions_sdf_maps = torch.cat(predictions_sdf_maps_list, dim=0) ## [2000, H, W]
        on_edge_flags = torch.stack(all_on_edge_flag_list, dim=0).to(torch.float32).to(self.device) ## [2000, 4]

        ## filtering: max sdf 
        max_sdf_values = torch.amax(predictions_sdf_maps, dim=(1,2)).to(torch.float32)
        predictions_sdf_maps = predictions_sdf_maps[max_sdf_values>self.args.max_sdf_thres]
        on_edge_flags = on_edge_flags[max_sdf_values>self.args.max_sdf_thres]
        proposals = proposals[max_sdf_values>self.args.max_sdf_thres]
        labels[labels==0] = torch.where(max_sdf_values>self.args.max_sdf_thres, 0, -1).to(torch.float32)
        max_sdf_values = max_sdf_values[max_sdf_values>self.args.max_sdf_thres]

        # predictions_sdf_maps_remained = predictions_sdf_maps
        # proposals_remained = proposals

        # if len(proposals) == 0:
        #     assert len(labels[labels==0]) == 0
        #     objectness_scores = torch.amax(predictions_sdf_maps, dim=(1,2)) ## TODO: objectness to be updated

        #     out_bboxes[labels==1] = proposals.to(torch.float32)
        #     out_bboxes_tight = out_bboxes_tight
        #     out_scores[labels==1] =objectness_scores.to(torch.float32)
        #     empty_tensor = torch.tensor((0.0))
        #     return out_bboxes, out_bboxes_tight, out_scores, empty_tensor, empty_tensor, empty_tensor, labels
        
        delta_x1, delta_y1, delta_x2, delta_y2 = self.update_bbox_with_boundary_fields(predictions_sdf_maps)
        
        ## calculate largest expansion and shrinkage
        delta_bboxes_signed = torch.stack([-delta_x1, -delta_y1, delta_x2, delta_y2], dim=1) ## larger than 0, expand; smaller than 0, shrink
        delta_bboxes_signed = torch.where((delta_bboxes_signed>0)&(on_edge_flags==1), 0, 1).to(torch.float32) * delta_bboxes_signed
        max_expansion = torch.amax(delta_bboxes_signed, dim=1) ## [N]
        max_shrink = torch.amin(delta_bboxes_signed, dim=1) ## [N]
        labels_subset = labels[labels==0] 

        labels_subset[max_expansion>0] = 0
        labels_subset[max_shrink<-self.args.max_shrink_threshold] = 0
        labels_subset[(max_expansion<=0)&(max_shrink>=-self.args.max_shrink_threshold)] = 1

        labels[labels==0] = labels_subset

        ## take a smaller step
        delta_x1 -= torch.abs(delta_x1) * self.args.delta_ratio
        delta_y1 -= torch.abs(delta_y1) * self.args.delta_ratio
        delta_x2 += torch.abs(delta_x2) * self.args.delta_ratio
        delta_y2 += torch.abs(delta_y2) * self.args.delta_ratio

        delta_bboxes_raw = torch.stack([delta_x1, delta_y1, delta_x2, delta_y2], dim=1) ## [2000, 4]
        delta_bboxes_raw[labels_subset==1] = 0
        delta_count_raw = torch.mean(torch.abs(delta_bboxes_raw))

        updated_bboxes = self.post_process_bbox_update(proposals, delta_bboxes_raw, delta_scale_x=128, delta_scale_y=128) ## [2000, 4]
        _, H, W = image.shape  
        updated_bboxes[:, 0][updated_bboxes[:, 0]<0] = 0
        updated_bboxes[:, 1][updated_bboxes[:, 1]<0] = 0
        updated_bboxes[:, 2][updated_bboxes[:, 2]>W] = W 
        updated_bboxes[:, 3][updated_bboxes[:, 3]>H] = H 
        
        ## this delta_count consider the effect of edges
        # actual_bbox_updates = updated_bboxes - proposals
        # delta_bboxes = post_process_bbox_update_reverse(actual_bbox_updates, proposals, delta_scale_x=128, delta_scale_y=128)
        # delta_count = torch.mean(torch.abs(delta_bboxes.to(torch.float32)))
        # actual_bbox_updates_count = torch.mean(torch.abs(actual_bbox_updates.to(torch.float32)))
        
        out_bboxes[labels>=0] = updated_bboxes.to(torch.float32)
        # out_bboxes_tight = out_bboxes
        # out_scores[labels>=0] = torch.amax(predictions_sdf_maps, dim=(1,2)).to(torch.float32)

        out_dict = {
            'updated_bboxes': out_bboxes,
            'labels': labels
        }
        return out_dict
        # return out_bboxes, out_bboxes_tight, out_scores, delta_count_raw, delta_count, actual_bbox_updates_count, labels


    def existence_checking(self, image, proposals):
        num_img_per_batch = 128
        class_score_list = []
        all_cropped_image_list = []
        all_on_edge_flag_list = []
        for batch_idx in range(0, math.ceil(len(proposals)/num_img_per_batch)):
            batch_proposal = proposals[batch_idx*num_img_per_batch:(batch_idx+1)*num_img_per_batch]
            cropped_image_list = []
            cropped_image_list = []
            for box in batch_proposal:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))
                on_edge_flag = np.array([x1==0, y1==0, x2==image.shape[-1], y2==image.shape[-2]])
                all_on_edge_flag_list.append(torch.tensor(on_edge_flag))
                resize = transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                cropped_image = resize(image[:,y1:y2, x1:x2])
                cropped_image_list.append(cropped_image)
                all_cropped_image_list.append(cropped_image)
            cropped_images = torch.stack(cropped_image_list, dim=0) ## [num_img_per_batch, 3, H, W]
            with torch.no_grad():
                class_score = self.binary_classifier_model(cropped_images.to(torch.float32).to(self.device))
                class_score_list.append(class_score)
        on_edge_flags = torch.stack(all_on_edge_flag_list, dim=0).to(torch.float32).to(self.device) ## [2000, 4]
        all_cropped_image = torch.stack(all_cropped_image_list)
        class_scores = torch.cat(class_score_list, dim=0).cpu()
        # binary_class_scores = torch.where(class_scores>=self.args.class_score_thres, 0, -1) ## 0: contains object, 1: not contains object
        
        out_dict = {
            'existence_scores': class_scores.squeeze(1),
            # 'proposal_images': proposal_images,
            # 'on_edge_flags': on_edge_flags,
        }
        return out_dict
    
    def center_reasoning(self, image, proposals):
        print('# of proposals for center_reasoning', len(proposals))
        sdf_maps, center_fields = self.get_prediction_with_proposals(proposals, image)
        sdf_binary_masks = torch.where(torch.sigmoid(sdf_maps)>0.5, 1, 0) ## [2000, 128, 128]
        center_fields_norm = torch.norm(center_fields, dim=1)
        centerness_binary_masks = torch.where(center_fields_norm>0.5, 1, 0) ## [2000, 128, 128]
        union_binary_masks = torch.where((centerness_binary_masks+sdf_binary_masks)>0, 1, 0)
        eroded_union_binary_masks = batch_erode(union_binary_masks, kernel_size=9, num_round=3)
        center_score_maps = self.center_field_to_anti_center_map(center_fields, kernel_size=5)
        fg_center_score_maps = center_score_maps * eroded_union_binary_masks
        fg_center_score_maps[:, 0:10, :] = 0
        fg_center_score_maps[:, -10:, :] = 0
        fg_center_score_maps[:, :, 0:10] = 0
        fg_center_score_maps[:, :, -10:] = 0
        fg_center_score_max_values = torch.amax(fg_center_score_maps, dim=(1,2))

        proposals_pass_singularity = proposals[fg_center_score_max_values <= self.args.center_score_max_thres]
        proposals_fail_singularity = proposals[fg_center_score_max_values > self.args.center_score_max_thres]
        print("# proposals_pass_singularity / proposals_fail_singularity:", len(proposals_pass_singularity), '/', len(proposals_fail_singularity))

        splited_new_proposals = []
        if len(proposals_fail_singularity) > 0:
            for box_idx, box in enumerate(proposals_fail_singularity):
                x1, y1, x2, y2 = box
                fg_center_score_map = fg_center_score_maps[fg_center_score_max_values > self.args.center_score_max_thres][box_idx]
                y_center, x_center = self.unravel_index(fg_center_score_map.argmax(), fg_center_score_map.shape)
                y_ratio = y_center / fg_center_score_map.shape[-2]
                x_ratio = x_center / fg_center_score_map.shape[-1]
                left_split_box = torch.tensor([x1, y1, x1+(x2-x1)*x_ratio, y2])
                right_split_box = torch.tensor([x1+(x2-x1)*x_ratio, y1, x2, y2])
                top_split_box = torch.tensor([x1, y1, x2, y1+(y2-y1)*y_ratio])
                bottom_split_box = torch.tensor([x1, y1+(y2-y1)*y_ratio, x2, y2])
                splited_new_proposals.extend([left_split_box, right_split_box, top_split_box, bottom_split_box])
            print('# of new splited proposals', len(splited_new_proposals))
            splited_new_proposals = torch.stack(splited_new_proposals, dim=0).to(proposals_pass_singularity.device)
        
        if self.args.analyze_cc:
            ## analyse connected component for proposals that pass singularity checking
            cc_results, single_component_indicators = self.separate_connected_components(union_binary_masks[fg_center_score_max_values <= self.args.center_score_max_thres])
            single_component_proposals = proposals_pass_singularity[torch.tensor(single_component_indicators, dtype=torch.bool)]
            multi_component_proposals = cc_results['multi']
            # single_component_proposals = self.enlarge_proposals(single_component_proposals, (self.height, self.width), ratio=1.5)
            multi_component_proposals = self.enlarge_proposals(multi_component_proposals, (self.height, self.width), ratio=1.5)
            print("# single_component_proposals / multi_component_proposals:", len(single_component_proposals), '/', len(multi_component_proposals))

            # proposals_pass_singularity = torch.FloatTensor(single_component_proposals).to(proposals_pass_singularity.device)
            splited_new_proposals = torch.cat((splited_new_proposals, torch.FloatTensor(multi_component_proposals).to(splited_new_proposals.device)), dim=0)
            print("# proposals_pass_singularity & connected_CC / splited_new_proposals:", len(proposals_pass_singularity), '/', len(splited_new_proposals))


        out_dict = {
            'proposals_pass_singularity': proposals_pass_singularity,
            'splited_new_proposals': splited_new_proposals,
            # 'on_edge_flags': on_edge_flags[fg_center_score_max_values <= self.args.center_score_max_thres]
        }
        return out_dict
    
    def boundary_reasoning(self, image, proposals, n_round=50):
        '''
        labels:
        0: continue to be updated
        -1: filter out
        1: already good enough, no more updating
        '''
        print('# of proposals for boundary_reasoning', len(proposals))
        labels = torch.zeros((len(proposals))).to(proposals.device)
        cur_proposals = proposals
        for current_round in tqdm(range(0, self.args.n_round), ncols=90, desc="boundary reasoning step"):
            # print('Optimize Round', current_round)
            # current_result_folder = os.path.join(self.result_folder, 'round_'+str(current_round))
            # if not os.path.isdir(current_result_folder):
            #     os.makedirs(current_result_folder)

            cur_proposals, labels = self.filter_small_proposal(cur_proposals, labels)
            if len(cur_proposals) == 0:
                return {
                        'proposals': [],
                        'labels': []
                    }
            out_dict = self.optimize_one_image_single_round(image, cur_proposals, labels)

            cur_proposals = out_dict['updated_bboxes']
            labels = out_dict['labels']
        
        return {
            'proposals': cur_proposals,
            'labels': labels
        }
    

    def main_object_discovery(self):
        results_dict = {}
        for image_idx in tqdm(range(0, len(self.test_dataset)), ncols=90, desc="process image"):
            image, label = self.test_dataset.get_image_with_index(image_idx) ## [3, H, W]
            image_id = label['image_id'].item()
            self.height = image.shape[-2]
            self.width = image.shape[-1]
        
            ## Step 0: Initial Object Proposal Generation 
            proposals = self.generate_random_proposal(height=image.shape[-2], width=image.shape[-1])
            proposals = torch.tensor(proposals).to(self.device)

            ## Step 1: Existence Checking
            existence_checking_results = self.existence_checking(image, proposals)
            existence_scores = existence_checking_results['existence_scores']
            proposals = proposals[existence_scores >= self.args.class_score_thres]
            if len(proposals) == 0:
                continue

            ## Step 2: Center Reasoning
            center_reasoning_results = self.center_reasoning(image, proposals)
            proposals_pass_singularity = center_reasoning_results['proposals_pass_singularity']
            splited_new_proposals = center_reasoning_results['splited_new_proposals']
            ### re-check the split proposals
            existence_checking_results = self.existence_checking(image, splited_new_proposals)
            existence_scores = existence_checking_results['existence_scores']
            splited_new_proposals = splited_new_proposals[existence_scores >= self.args.class_score_thres]
            if len(splited_new_proposals) > 0:
                center_reasoning_results_new = self.center_reasoning(image, splited_new_proposals)
                proposals = torch.cat((proposals_pass_singularity, center_reasoning_results_new['proposals_pass_singularity']), dim=0)
            else:
                proposals = proposals_pass_singularity
            if len(proposals) == 0:
                continue

            ## Step 3: Boundary Reasoning
            boundary_reasoning_results = self.boundary_reasoning(image, proposals, n_round=self.args.n_round)
            proposals = boundary_reasoning_results['proposals']
            labels = boundary_reasoning_results['labels']
            if len(proposals) == 0:
                continue
            proposals = proposals[labels==1]
            if len(proposals) == 0:
                continue

            ## NMS to remove redundancy
            nms_indexes = torchvision.ops.nms(proposals.to(torch.float32), labels[labels==1], iou_threshold=0.5)
            results_dict[image_id] = proposals[nms_indexes].cpu().numpy()
        
        with open(os.path.join(self.result_folder, 'discovery_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NpEncoder)

            



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
    ## Objectness Model Specifics
    parser.add_argument('--image_size', type=int,
                        default=128, 
                        help='size of image from dataloader')
    parser.add_argument('--backbone_type', type=str,
                        help='backbone', default='dpt_large')   
    parser.add_argument("--sdf_activation", type=str, default=None) 
    parser.add_argument("--use_bg_sdf", action='store_true', help='') 
    parser.add_argument('--objectness_resume', type=str, default=None)
    parser.add_argument('--binary_classifier_resume', type=str, default=None)

    ## Dataset
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--dataset_split", type=str, default='test')
    parser.add_argument('--dataset', type=str, default='COCO')

    ## Object Reasoning Hyper-parameters
    parser.add_argument("--class_score_thres", type=float, help='', default=0.1) 
    parser.add_argument("--center_score_max_thres", type=float, help='', default=0.009) 
    parser.add_argument("--analyze_cc", action='store_true', help='') 
    parser.add_argument("--max_sdf_thres", type=float, default=0.5) 
    parser.add_argument("--max_shrink_threshold", type=float, help='', default=16) 
    parser.add_argument("--delta_ratio", type=float, help='', default=0.5) 
    parser.add_argument('--n_round', type=int, default=50)
    parser.add_argument('--proposal_area_thres', type=int, default=50)

    

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    print('device', device)

    object_discovery_model = Object_Discovery(args, device)
    # bbox_optimizer.optimize()
    object_discovery_model.main_object_discovery()

if __name__ == "__main__":
    main()