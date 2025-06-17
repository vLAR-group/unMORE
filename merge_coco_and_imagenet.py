import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pycocotools.mask as mask_util
from utils.misc import NpEncoder

CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]
training_annotations = {
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}
category_info = {
    "is_crowd": 0,
    "id": 1
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_annotations_training_format_path', type=str, default=None)
    parser.add_argument('--imagenet_annotations_training_format_path', type=str, default=None)
    args = parser.parse_args()

    result_folder = '/'.join(args.coco_annotations_training_format_path.split('/')[0:-1])
    result_folder = os.path.join(result_folder, 'merged_with_imagenet')
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    print('result_folder', result_folder)
    out_fname = 'COCO_merged_IN_training_format.json'

    imagenet_times = 1
    coco_times = 1
    out_training_dict = {}
    with open(args.coco_annotations_training_format_path) as f:
        coco_annotations_training_format = json.load(f)
        coco_annotations = coco_annotations_training_format['annotations']
    with open(args.imagenet_annotations_training_format_path) as f:
        imagenet_annotations_training_format = json.load(f)
        imagenet_annotations = imagenet_annotations_training_format['annotations']


    coco_image_info_list = []
    for coco_image_info in coco_annotations_training_format['images']:
        new_id = 'coco_' + str(coco_image_info['id']) 
        coco_image_info['id'] = new_id
        coco_image_info_list.append(coco_image_info)
    imagenet_image_info_list = []
    for imagenet_image_info in imagenet_annotations_training_format['images']:
        new_id = 'imagenet_' + str(imagenet_image_info['id'] )
        imagenet_image_info['id'] = new_id
        imagenet_image_info_list.append(imagenet_image_info)
    # training_annotations['images'] = coco_image_info_list + imagenet_image_info_list
    print("# of imagenet images", len(imagenet_image_info_list))
    print("# of coco images", len(coco_image_info_list))
    training_annotations['images'] = []
    for _ in range(0, imagenet_times):
        training_annotations['images'].extend(imagenet_image_info_list)
    for _ in range(0, coco_times):
        training_annotations['images'].extend(coco_image_info_list)
    print("# of merged images", len(training_annotations['images']))

    all_annotations = []
    for ann in imagenet_annotations:
        ann['score'] = ann['weight']
        if ann['score'] < 0.5:
            continue
        ann['image_id'] = 'imagenet_' + str(ann['image_id'])
        all_annotations.append(ann)
    for ann in coco_annotations:
        ann['image_id'] = 'coco_' + str(ann['image_id'])
        all_annotations.append(ann)

    all_annotations_with_ann_id = []
    for idx, ann in enumerate(all_annotations):
        ann['id'] = idx
        all_annotations_with_ann_id.append(ann)

    training_annotations['annotations'] = all_annotations_with_ann_id
    print('total number of images', len(training_annotations['images']))
    print('total number of annotations', len(training_annotations['annotations']))
    with open(os.path.join(result_folder, out_fname), 'w') as f:
        json.dump(training_annotations, f, indent=2, cls=NpEncoder)
