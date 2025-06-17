### 1. Train Objectness Net
#### 1.1 Train Object Center and Boundary Model
CUDA_VISIBLE_DEVICES=2 python train_objectness_net.py --dataset ImageNet_votecut_top1_Dataset \
    --backbone_type dpt_large --optimizer adam --lr_scheduler_gamma 0.1 --learning_rate 0.0001 \
    --batch_size 20 --lr_scheduler_gamma 0.1 \
    --sdf_loss_type l1 --center_field_loss_type l2 --use_sdf_binary_mask_loss --use_sdf_gradient_loss --sdf_activation tanh --use_bg_sdf \
    --train_center_and_boundary
### 1.2 Train Object Existence Model 
CUDA_VISIBLE_DEVICES=1 python train_objectness_net.py --dataset ImageNet_votecut_top1_Dataset \
    --backbone_type dpt_large --optimizer adam --lr_scheduler_gamma 0.1 --learning_rate 0.0001 \
    --batch_size 20 --lr_scheduler_gamma 0.1 \
    --sdf_loss_type l1 --center_field_loss_type l2 --use_sdf_binary_mask_loss --use_sdf_gradient_loss --sdf_activation tanh --use_bg_sdf \
    --train_existence

## 2. Object Discovery
CUDA_VISIBLE_DEVICES=1 python object_reasoning.py \
    --sdf_activation tanh --use_bg_sdf \
    --objectness_resume unMORE/ckpt/objectness_net_model.ckpt \
    --binary_classifier_resume unMORE/ckpt/classifier_model.ckpt \
    --start_idx 0 \
    --end_idx 100 \
    --analyze_cc


## 3. Object Scoring
CUDA_VISIBLE_DEVICES=1 python object_scoring.py \
    --sdf_activation tanh --use_bg_sdf \
    --objectness_resume unMORE/ckpt/objectness_net_model.ckpt \
    --binary_classifier_resume unMORE/ckpt/classifier_model.ckpt \
    --start_idx 0 \
    --end_idx 100 \
    --raw_annotations_path {path to discovery_results.json}


## 4. Post-process Objects for Detector Training 
python post_process.py \
    --pred_annotations_path {path object_discovery_with_scores.json} \
    --existence_score_thres 0.5 \
    --center_score_thres 0.8 \
    --boundary_score_thres 0.75 \
    --dataset COCO \
    --split test 


## 5. Merge unMORE_disc results on COCO Trainset with ImageNet Trainset (votecut)
python merge_coco_and_imagenet.py \
    --coco_annotations_training_format_path {path to COCO_train2017_unMORE_disc_results.json} \
    --imagenet_annotations_training_format_path {path imagenet_train_votecut_kmax_3_tuam_0.2.json}

## 6. Train Class Agnostic Detector
CUDA_VISIBLE_DEVICES=1,2 python cad/train_net.py \
    --num-gpus 2 \
    --config-file unMORE/cad/model_zoo/configs/unMORE-IN+COCO/cascade_mask_rcnn_R_50_FPN.yaml

## 7. Evaluate unMORE model
CUDA_VISIBLE_DEVICES=1,2 python cad/train_net.py \
    --config-file unMORE/cad/model_zoo/configs/unMORE-IN+COCO/cascade_mask_rcnn_R_50_FPN.yaml \
    --num-gpus 2 \
    --eval-only \
    --test-dataset cls_agnostic_coco*_val_17 \
    MODEL.WEIGHTS {path to unMORE_model.pth} \
    OUTPUT_DIR cad_eval/cls_agnostic_coco*_val_17

