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
import pycocotools.mask as mask_util

from datasets import *
from models.objectness_net import ObjectnessNet, Binary_Classifier

sys.path.append('path to unMORE folder')
from utils.misc import batch_erode, NpEncoder
from utils.vis import *


class Object_Scoring:
    
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
        
        self.result_folder =  '/'.join(self.args.raw_annotations_path.split('/')[0:-1])
        # self.result_folder = os.path.join(self.result_folder, 'sdf_and_centerness_and_classification_metrics_'+str(args.start_idx)+'_to_'+str(args.end_idx))
        # if not os.path.isdir(result_folder):
        #     os.makedirs(result_folder)

        print('result_folder', self.result_folder)
        with open(os.path.join(self.result_folder, 'configs_object_scoring.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2) 
        self.load_raw_annotations()

    def load_raw_annotations(self):
        with open(self.args.raw_annotations_path) as f:
            self.raw_annotations = json.load(f)
        print('# of loaded images', len(self.raw_annotations))
        return self.raw_annotations

    def get_prediction_with_proposals(self, image, proposals):
        # objectness_model.eval()
        # objectness_model = objectness_model.to(device)
        # proposal = proposals.to(self.device)
        # num_img_per_batch = 100
        num_img_per_batch = 50
        on_edge_flag_list = []
        predictions_sdf_maps_list = []
        predictions_center_fields_list = []
        all_cropped_image_list = []
        class_score_list = []
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
                class_score = self.binary_classifier_model(cropped_images.to(torch.float32).to(self.device))
            proposal_sdf_masks = predictions_no_grad['sdf_maps'].squeeze(1)
            predictions_sdf_maps_list.append(proposal_sdf_masks)
            proposal_center_fields = predictions_no_grad['center_fields']
            predictions_center_fields_list.append(proposal_center_fields)
            class_score_list.append(class_score)
    
        predictions_sdf_maps = torch.cat(predictions_sdf_maps_list, dim=0) ## [2000, H, W]
        predictions_center_fields = torch.cat(predictions_center_fields_list, dim=0) ## [2000, 2, H, W]
        all_cropped_image = torch.stack(all_cropped_image_list)
        on_edge_flags = torch.stack(on_edge_flag_list, dim=0) ## [2000, 4]
        class_scores = torch.cat(class_score_list, dim=0).squeeze(1)

        out_dict = {
            'pred_boundary_fields': predictions_sdf_maps,
            'pred_center_fields': predictions_center_fields,
            'pred_existence_scores': class_scores
        }
        return out_dict

    @staticmethod
    def binary_mask_to_tight_bbox_coco_style(binary_mask):
        binary_mask_encoded = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        bounding_box = mask_util.toBbox(binary_mask_encoded)
        bounding_box = bounding_box.tolist()
        return bounding_box

    @staticmethod
    def binary_mask_to_rle(binary_mask):
        rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
        rle['counts'] = rle['counts'].decode('ascii')
        return rle

    def main_object_scoring(self):
        out_annotations = []
        for image_idx in tqdm(range(0, len(self.test_dataset)), ncols=90, desc="process image"):
            image, label = self.test_dataset.get_image_with_index(image_idx) ## [3, H, W]
            image_id = label['image_id'].item()
            if str(image_id) not in self.raw_annotations.keys():
                print(image_id, 'do not have raw predictions')
                continue
            raw_proposals = self.raw_annotations[str(image_id)]

            predictions = self.get_prediction_with_proposals(image, raw_proposals)
            
            pred_boundary_fields = predictions['pred_boundary_fields']
            pred_center_fields = predictions['pred_center_fields']
            pred_existence_scores = predictions['pred_existence_scores']

            ### 1. MAX Center Field Norm Values 
            pred_center_fields_norm = torch.norm(pred_center_fields, dim=1)
            max_center_fields_norms = torch.amax(pred_center_fields_norm, dim=(1,2))

            ## 2. MAX Boundary Distance Values
            max_boundary_distance_values = torch.amax(pred_boundary_fields, dim=(1,2)).to(torch.float32)
            
            ## 3. Binary Mask from Center Field
            pred_center_fields_binary_masks = torch.where(pred_center_fields_norm>0.5, 1, 0) ## [2000, 128, 128]
            resized_center_field_binary_mask_list = []
            for proposal_idx in range(0, len(raw_proposals)):
                box = raw_proposals[proposal_idx]
                x1, y1, x2, y2 = box
                x1, y1 = math.floor(x1), math.floor(y1)
                x2, y2 = math.ceil(x2), math.ceil(y2)
                pred_saliency_mask = pred_center_fields_binary_masks[proposal_idx]
                resized_saliency_mask = torch.zeros_like(image[0])
                saliency_region = resized_saliency_mask[y1:y2, x1:x2]
                resize = transforms.Resize((saliency_region.shape[0], saliency_region.shape[1]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                resized_saliency_mask[y1:y2, x1:x2] = resize(pred_saliency_mask.unsqueeze(0))[0] 
                resized_center_field_binary_mask_list.append(resized_saliency_mask)
            resized_center_field_binary_masks = torch.stack(resized_center_field_binary_mask_list, dim=0) ### [2000, H, W]

            ## 4. Binary Mask from Boundary Field
            pred_boundary_fields_binary_masks = torch.where(torch.sigmoid(pred_boundary_fields)>0.5, 1, 0) ## [2000, 128, 128]
            resized_boundary_fields_binary_mask_list = []
            for proposal_idx in range(0, len(raw_proposals)):
                box = raw_proposals[proposal_idx]
                x1, y1, x2, y2 = box
                x1, y1 = math.floor(x1), math.floor(y1)
                x2, y2 = math.ceil(x2), math.ceil(y2)
                pred_saliency_mask = pred_boundary_fields_binary_masks[proposal_idx]
                resized_saliency_mask = torch.zeros_like(image[0])
                saliency_region = resized_saliency_mask[y1:y2, x1:x2]
                resize = transforms.Resize((saliency_region.shape[0], saliency_region.shape[1]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                resized_saliency_mask[y1:y2, x1:x2] = resize(pred_saliency_mask.unsqueeze(0))[0] 
                resized_boundary_fields_binary_mask_list.append(resized_saliency_mask)
            resized_boundary_fields_binary_masks = torch.stack(resized_boundary_fields_binary_mask_list, dim=0) ### [2000, H, W]

            ## 5. Union Binary Mask
            resized_union_binary_masks = torch.where((resized_center_field_binary_masks + resized_boundary_fields_binary_masks)>0, 1, 0)

            ## 6. Calculate tight bounding box based on binary mask
            tight_bboxes = []
            for idx, binary_mask in enumerate(resized_union_binary_masks):
                tight_bbox_coco_style = self.binary_mask_to_tight_bbox_coco_style(binary_mask.numpy())
                tight_bboxes.append([tight_bbox_coco_style[0], tight_bbox_coco_style[1], tight_bbox_coco_style[0]+tight_bbox_coco_style[2], tight_bbox_coco_style[1]+tight_bbox_coco_style[3]])
            tight_bboxes = torch.FloatTensor(tight_bboxes)

            ## 7. Perform NMS
            nms_indexes = torchvision.ops.nms(tight_bboxes.to(max_boundary_distance_values.device), max_boundary_distance_values, iou_threshold=0.5).cpu()
            final_tight_bboxes = tight_bboxes[nms_indexes]
            final_binary_masks = resized_union_binary_masks[nms_indexes]
            boundary_scores = max_boundary_distance_values[nms_indexes].cpu().numpy()
            center_scores = max_center_fields_norms[nms_indexes].cpu().numpy()
            existence_scores = pred_existence_scores[nms_indexes].cpu().numpy()
            max_mask_area = torch.amax(final_binary_masks.sum(1).sum(1)).cpu().numpy()
            mask_scores = final_binary_masks.sum(1).sum(1).cpu().numpy() / max_mask_area

            ## 8. Calculate Final Scores
            for idx, binary_mask in enumerate(final_binary_masks):
                x1, y1, x2, y2 = final_tight_bboxes[idx].numpy()
                box_coco_style = [x1, y1, x2-x1, y2-y1]
                # print(existence_scores[idx])
                # print(center_scores[idx])
                # print(boundary_scores[idx])
                # print(mask_scores[idx])
                score = existence_scores[idx] * center_scores[idx] * boundary_scores[idx] * pow(mask_scores[idx], 0.25)

                annotation = {
                    "image_id": image_id,
                    "category_id": 1,
                    "score": score,
                    'bbox': box_coco_style,
                    "segmentation": self.binary_mask_to_rle(binary_mask),
                    'existence_score': existence_scores[idx],
                    'center_score': center_scores[idx],
                    'boundary_score': boundary_scores[idx], 
                    'area_score': pow(mask_scores[idx], 0.25)
                }
                out_annotations.append(annotation)
        
        print('# of final annotations', len(out_annotations))
        with open(os.path.join(self.result_folder, 'object_discovery_with_scores.json'), 'w') as f:
            json.dump(out_annotations, f, indent=2, cls=NpEncoder)


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
    parser.add_argument('--raw_annotations_path', type=str, default=None)

    # ## Object Reasoning Hyper-parameters
    # parser.add_argument("--class_score_thres", type=float, help='', default=0.1) 
    # parser.add_argument("--center_score_max_thres", type=float, help='', default=0.009) 
    # parser.add_argument("--max_sdf_thres", type=float, default=0.5) 
    # parser.add_argument("--max_shrink_threshold", type=float, help='', default=16) 
    # parser.add_argument("--delta_ratio", type=float, help='', default=0.5) 
    # parser.add_argument('--n_round', type=int, default=50)
    

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    print('device', device)
    

    object_scoring_model = Object_Scoring(args, device)
    object_scoring_model.main_object_scoring()

if __name__ == "__main__":
    main()