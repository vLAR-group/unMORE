import argparse
import json
import os
from tqdm import tqdm 

CATEGORIES = {
    "is_crowd": 0,
    "id": 1
}

def convert_pred_annotations_to_training_format(selected_annotations_list, gt_annotation_path, out_fname_training):
    training_annotations = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    category_info = {
        "is_crowd": 0,
        "id": 1
    }
    print('convert_pred_annotations_to_training_format')

    with open(gt_annotation_path) as f:
        gt_annotations = json.load(f)
    image_info_list = gt_annotations['images']

    
    training_annotations['images'] = image_info_list
    training_annotations['annotations'] = selected_annotations_list

    with open(out_fname_training, 'w') as f:
        json.dump(training_annotations, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_annotations_path', type=str, default=None)
    parser.add_argument("--existence_score_thres", type=float, help='', default=0.5) 
    parser.add_argument("--center_score_thres", type=float, help='', default=0.8) 
    parser.add_argument("--boundary_score_thres", type=float, help='', default=0.75) 
    parser.add_argument('--dataset', type=str, default='COCO')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    result_folder = '/'.join(args.pred_annotations_path.split('/')[0:-1])
    with open(os.path.join(result_folder, 'configs_post_process.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2) 
    
    if args.dataset == 'COCO' and args.split == 'test':
        gt_annotation_path = 'path to coco_cls_agnostic_instances_val2017.json'
    elif args.dataset == 'COCO' and args.split == 'train':
        gt_annotation_path = 'path to coco_cls_agnostic_instances_train2017.json'
    else:
        raise NotImplementedError

    with open(args.pred_annotations_path) as f:
        pred_annotations = json.load(f)

    selected_annotations = []
    ann_idx = 0
    for ann in tqdm(pred_annotations, ncols=90):
        if ann['existence_score'] < args.existence_score_thres:
            continue
        if ann['center_score'] < args.center_score_thres:
            continue
        if ann['boundary_score'] < args.boundary_score_thres:
            continue

        ann['id'] = ann_idx
        ann['score'] = ann['area_score']
        ann['bbox'] = ann['bbox']
        ann['segmentation'] = ann['segmentation']
        selected_annotations.append(ann)
        ann_idx += 1
    
    out_fname_training = os.path.join(result_folder, 'selected_training_annotations.json')
    convert_pred_annotations_to_training_format(selected_annotations, gt_annotation_path, out_fname_training)

