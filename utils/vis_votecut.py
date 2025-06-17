from pycocotools.coco import COCO
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage

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
dest_folder = 'path to destination folder'
if not os.path.isdir(os.path.join(dest_folder)):
    os.mkdir(os.path.join(dest_folder))
if not os.path.isdir(os.path.join(dest_folder, 'masks')):
    os.mkdir(os.path.join(dest_folder, 'masks'))

coco = COCO(TRAIN_ANNOTATION_FILE)
imgId_list = coco.getImgIds()
imgId_list.sort()
img_list = coco.loadImgs(imgId_list)
assert len(imgId_list) == len(img_list)
print('total', len(imgId_list), 'samples')
obj_count = {}
center_out = {}
for index in tqdm(range(0, len(imgId_list)), ncols=100):
    imgId = imgId_list[index]
    img = img_list[index]
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    anns = coco.loadAnns(annIds)
    if len(anns) > 0:
        anns_img = np.zeros_like(coco.annToMask(anns[0]))
    else:
        anns_img = np.zeros((img['width'], img['height']))
    
    for _, ann in enumerate(anns):
        anns_img = np.maximum(anns_img, coco.annToMask(ann)*ann["id"])
    if len(anns) not in obj_count.keys():
        obj_count[len(anns)] = 1
    else:
        obj_count[len(anns)] += 1
    if not os.path.isdir(os.path.join(dest_folder, 'masks', img['file_name'].split('/')[0])):
        os.mkdir(os.path.join(dest_folder, 'masks', img['file_name'].split('/')[0]))
    if not os.path.isdir(os.path.join(dest_folder, 'masks_vis', img['file_name'].split('/')[0])):
        os.mkdir(os.path.join(dest_folder, 'masks_vis', img['file_name'].split('/')[0]))

    out_mask = unify_instance_id(anns_img).astype('int8')
    cv2.imwrite(os.path.join(dest_folder, 'masks', img['file_name'].split('/')[0], img['file_name'].split('/')[1]).replace('JPEG', 'png'), out_mask)
