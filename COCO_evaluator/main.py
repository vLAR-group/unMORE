from pycocotools.coco import COCO
import os
import cv2
import json
import numpy as np
import math
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
import datetime
import pycocotools.mask as mask_util
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt 
from coco_evaluation import COCOEvaluator
# from utils.center_crop_utils import *

def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv


def evaluate_ap(gt_annotation_path, pred_annotation_path, evaluator, result_folder):
    with open(pred_annotation_path) as f:
        pred_annotations = json.load(f)
    with open(gt_annotation_path) as f:
        gt_annotations = json.load(f)

    evaluator.reset()

    ## construct image_id -> ann_idx dict
    image_id_to_ann_id_dict = {}
    for ann_idx in range(0, len(pred_annotations)): 
        ann = pred_annotations[ann_idx]
        if 'id' not in ann.keys():
            ann['id'] = ann_idx
        if ann is None:
            continue
        image_id = ann['image_id']
        if image_id not in image_id_to_ann_id_dict.keys():
            image_id_to_ann_id_dict[image_id] = [ann_idx]
        else:
            image_id_to_ann_id_dict[image_id].append(ann_idx)
    print('Number of images', len(image_id_to_ann_id_dict.keys()))
    print('Number of annotations', len(pred_annotations))

    for image_id in tqdm(image_id_to_ann_id_dict.keys(), ncols=90, desc='evaluate AP'):
        ann_id_list = image_id_to_ann_id_dict[image_id]
        instances = []
        for ann_id in ann_id_list:
            ann = pred_annotations[ann_id]
            if ann is None:
                continue
            if 'score' not in ann.keys():
                if 'weight' in ann.keys():
                    ann['score'] = ann['weight']
                else:
                    ann['score'] = 1
            instances.append(ann)
        evaluator.process(image_id=image_id, coco_instances=instances)

    results = evaluator.evaluate()
    results['pred_annotation_path'] = pred_annotation_path
    results['gt_annotation_path'] = gt_annotation_path
    results['number_of_images'] = len(image_id_to_ann_id_dict.keys())
    results['number_of_annotations'] = len(pred_annotations)

    with open(os.path.join(result_folder, 'ap_score.json'), 'w') as f:
        json.dump(results, f, indent=2) 

if __name__ == "__main__":
    resize = None

    dataset_name = 'COCO*'
    gt_annotation_path= 'unMORE/COCO*/data/COCO*_val2017_cls_agnostic.json'
    image_folder = 'path to COCO val2017'
    pred_annotation_path = 'path to predicted annotations.json'

    result_folder = ('/'.join(pred_annotation_path.split('/')[0:-1]))

    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_' + dataset_name
    if not os.path.isdir(os.path.join(result_folder, run_name)):
        os.makedirs(os.path.join(result_folder, run_name))
    print('result folder', os.path.join(result_folder, run_name))


    evaluator = COCOEvaluator(
        dataset_name=dataset_name,
        gt_annotation_path=gt_annotation_path,
        tasks=['bbox', 'segm'],
        # tasks=['bbox'],
        output_dir=os.path.join(result_folder, run_name)
    )
    evaluate_ap(gt_annotation_path, pred_annotation_path, evaluator, os.path.join(result_folder, run_name))
