from pycocotools.coco import COCO
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage
import pycocotools.mask as mask_util

def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv

def vis_gray(data):
    data = np.array(data)
    labels = np.unique(data)
    labels.sort()
    cmap = get_mask_plot_colors(len(labels))
    color_image = np.zeros([data.shape[0],data.shape[1],3])
    for idx, label in enumerate(labels):
        if label == 0:
            continue
        obj_mask = data==label
        color_image[:,:,0][obj_mask] = cmap[idx][0]
        color_image[:,:,1][obj_mask] = cmap[idx][1]
        color_image[:,:,2][obj_mask] = cmap[idx][2]
    return color_image * 255

def unify_instance_id(mask):
    new_mask = mask.copy()
    new_idx = 1
    for obj_idx in sorted(np.unique(mask)):
        if obj_idx == 0:
            continue
        new_mask[mask==obj_idx] = new_idx
        new_idx = new_idx + 1
    return new_mask 

TRAIN_ANNOTATION_FILE = 'path to votecut annotation' ## imagenet_train_votecut_kmax_3_tuam_0.2.json
dest_folder = 'ImageNet_Dataset'
if not os.path.isdir(os.path.join(dest_folder)):
    os.mkdir(os.path.join(dest_folder))
if not os.path.isdir(os.path.join(dest_folder, 'masks_top1_single_component')):
    os.mkdir(os.path.join(dest_folder, 'masks_top1_single_component'))

with open(TRAIN_ANNOTATION_FILE) as f:
    source_annotations_dict = json.load(f)
    
    source_annotations = source_annotations_dict['annotations']
    source_image_info = source_annotations_dict['images']

print('build image_id_to_fname_dict')
image_id_to_fname_dict = {}
for _, image_info in enumerate(tqdm(source_image_info, ncols=90)):
    image_id = image_info['id']
    image_id_to_fname_dict[image_id] = image_info['file_name']

print('build image_id_to_ann_id_dict')
image_id_to_ann_id_dict = {}
for ann_idx, ann in enumerate(tqdm(source_annotations, ncols=90)):
    image_id = ann['image_id']
    if image_id in image_id_to_ann_id_dict.keys():
        image_id_to_ann_id_dict[image_id].append(ann_idx)
    else:
        image_id_to_ann_id_dict[image_id] = [ann_idx]

for image_id in tqdm(sorted(image_id_to_ann_id_dict.keys()), ncols=90):
    fname = image_id_to_fname_dict[image_id]
    ann_id_list = image_id_to_ann_id_dict[image_id]
    score_list = []
    for ann_id in ann_id_list:
        ann = source_annotations[ann_id]
        score_list.append(ann['weight'])
    selected_ann_idx = ann_id_list[np.argmax(score_list)]
    selected_ann = source_annotations[selected_ann_idx]
    binary_mask = mask_util.decode(selected_ann['segmentation'])

    if not os.path.isdir(os.path.join(dest_folder, 'masks_top1_single_component', fname.split('/')[0])):
        os.mkdir(os.path.join(dest_folder, 'masks_top1_single_component', fname.split('/')[0]))
    
    if binary_mask.sum() == 0:
        cv2.imwrite(os.path.join(dest_folder, 'masks_top1_single_component', fname.split('/')[0], fname.split('/')[1]).replace('JPEG', 'png'), binary_mask*255)
        continue
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 4 , cv2.CV_32S)
    component_areas = stats[:,-1]
    fg_component_areas = component_areas[1:]
    largest_cc_index = np.argmax(fg_component_areas) + 1
    binary_mask = np.array(labels==largest_cc_index).astype(np.uint8)

    cv2.imwrite(os.path.join(dest_folder, 'masks_top1_single_component', fname.split('/')[0], fname.split('/')[1]).replace('JPEG', 'png'), binary_mask*255)
