from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import os
import pycocotools.mask as mask_util

original_coco_annotations_path = 'path to COCO original annotation' ## instances_val2017.json
original_coco_cls_agnostic_annotations_path = 'pah to COCO class agnostic annotation' ## coco_cls_agnostic_instances_val2017.json
extra_coco_annotations_path = 'path to extra coco annotations' ## COCO*_val2017_extra_labels_with_mask.json

with open(original_coco_annotations_path) as f:
    original_coco_annotations_dict = json.load(f)
with open(original_coco_cls_agnostic_annotations_path) as f:
    original_coco_cls_agnostic_annotation_dict = json.load(f)
with open(extra_coco_annotations_path) as f:
    extra_coco_annotations_dict = json.load(f)
print('===== original annotations stats =====')
print('# of images', len(original_coco_annotations_dict['images']))
print('# of annotations', len(original_coco_annotations_dict['annotations']))
print('# of categories', len(original_coco_annotations_dict['categories']))
print('===== additional annotations stats =====')
print('# of images', len(extra_coco_annotations_dict['images']))
print('# of annotations', len(extra_coco_annotations_dict['annotations']))
print('# of categories', len(extra_coco_annotations_dict['categories']))

## merge extra annotations with original annotations (with classes)
merged_categories = []
original_category_id_list = []
merged_annotations_with_idx = []
for category_info in original_coco_annotations_dict['categories']:
    original_category_id_list.append(category_info['id'])
    merged_categories.append(category_info)
for category_info in extra_coco_annotations_dict['categories']:
    if category_info['id'] not in original_category_id_list:
        merged_categories.append(category_info)
merged_annotations = original_coco_annotations_dict['annotations'] + extra_coco_annotations_dict['annotations']
idx = 0
for ann in merged_annotations:
    ann['id'] = idx
    idx += 1
    merged_annotations_with_idx.append(ann)
original_coco_annotations_dict['categories'] = merged_categories
original_coco_annotations_dict['annotations'] = merged_annotations
dest_fname = 'unMORE/COCO*/data/COCO*_val2017.json'
print('===== merged annotations stats =====')
print('# of images', len(original_coco_annotations_dict['images']))
print('# of annotations', len(original_coco_annotations_dict['annotations']))
print('# of categories', len(original_coco_annotations_dict['categories']))
with open(dest_fname, 'w') as f:
    json.dump(original_coco_annotations_dict, f, indent=2)
    
merged_annotations_with_idx_cls_agnostic = []
idx = 0
for ann in merged_annotations:
    ann['id'] = idx
    idx += 1
    merged_annotations_with_idx_cls_agnostic.append(ann)
    ann['category_id'] = 1
original_coco_cls_agnostic_annotation_dict['annotations'] = merged_annotations_with_idx_cls_agnostic
original_coco_cls_agnostic_annotation_dict['categories'] = [{"id": 1, "name": "fg", "supercategory": "fg"}]
dest_fname = 'unMORE/COCO*/data/COCO*_val2017_cls_agnostic.json'
print('===== merged annotations stats =====')
print('# of images', len(original_coco_cls_agnostic_annotation_dict['images']))
print('# of annotations', len(original_coco_cls_agnostic_annotation_dict['annotations']))
print('# of categories', len(original_coco_cls_agnostic_annotation_dict['categories']))
with open(dest_fname, 'w') as f:
    json.dump(original_coco_cls_agnostic_annotation_dict, f, indent=2)
    
