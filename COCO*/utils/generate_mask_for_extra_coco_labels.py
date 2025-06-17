from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import os
import pycocotools.mask as mask_util
def create_annotation_segmentation(binary_mask):
    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

sam_checkpoint = "path to SAM model" ## sam_vit_h_4b8939.pth
model_type = "vit_h"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
_ = sam.to(device=device)
predictor = SamPredictor(sam)

image_folder = 'path to COCO val2017'
source_annotation_path = 'unMORE/COCO*/data/COCO*_val2017_extra_labels.json' ## raw annotations VGG annotator, not provided in this repo
dest_fname = source_annotation_path.replace('.json', '_with_mask.json')

with open(source_annotation_path) as f:
    source_annotation_dict = json.load(f)
    source_image_info = source_annotation_dict['images']
    source_annotations = source_annotation_dict['annotations']
print('number of images', len(source_image_info))
print('number of annotations', len(source_annotations))
image_id_to_fname_dict = {}
for image_info in source_image_info:
    image_id = image_info['id']
    fname = image_info['file_name']
    image_id_to_fname_dict[image_id] = fname

image_id_to_ann_id_dict = {}
for ann_idx, ann in enumerate(source_annotations):
    image_id = ann['image_id']
    if image_id in image_id_to_ann_id_dict.keys():
        image_id_to_ann_id_dict[image_id].append(ann_idx)
    else:
        image_id_to_ann_id_dict[image_id] = [ann_idx]

out_annotations = []
for (image_id, ann_id_list) in tqdm(image_id_to_ann_id_dict.items(), ncols=90):
    fname = image_id_to_fname_dict[image_id]
    image = cv2.imread(os.path.join(image_folder, fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    for ann_idx in ann_id_list:
        ann = source_annotations[ann_idx]
        x1, y1, x_range, y_range = ann['bbox']
        bbox = np.array([x1, y1, x1+x_range, y1+y_range])
        masks, _, _ = predictor.predict(
            box=bbox[None, :],
            multimask_output=False,
            )
        binary_mask = (masks[0]).astype(np.uint8)
        rle_mask = create_annotation_segmentation(binary_mask)
        ann['segmentation'] = rle_mask
        ## get tight bbox
        binary_mask_encoded = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        tight_bounding_box = mask_util.toBbox(binary_mask_encoded)
        tight_bounding_box = tight_bounding_box.tolist()
        ann['original_bbox'] = ann['bbox']
        ann['bbox'] = tight_bounding_box
        out_annotations.append(ann)

out_annotations_with_idx = []
for idx, ann in enumerate(out_annotations):
    ann['id'] = idx
    out_annotations_with_idx.append(ann)


source_annotation_dict['annotations'] = out_annotations_with_idx
with open(dest_fname, 'w') as f:
    json.dump(source_annotation_dict, f, indent=2)
    