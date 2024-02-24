#!/bin/bash

# "$1" classıfıer resume path
# "$2" model_clf
# "$3" width_mult
# "$4" classıfıer modalıty
python online_test.py \
	--root_path ~/qyh/Real-time-GesRec/\
	--video_path ~/qyh/datasets/EgoGesture/videos/ \
	--annotation_path annotation_EgoGesture/egogestureall.json \
	--resume_path_det "results/shared/egogesture_resnetl_10_RGB_8.pth" \
	--resume_path_clf "train_res/egogesture_resnext_1.0x_RGB_32_best.pth"  \
	--result_path results \
	--dataset egogesture    \
	--sample_duration_det 8 \
	--sample_duration_clf 32 \
	--model_det resnetl \
	--model_clf resnext \
	--model_depth_det 10 \
	--width_mult_det 0.5 \
	--model_depth_clf 101 \
	--width_mult_clf 1 \
	--resnet_shortcut_det A \
	--resnet_shortcut_clf B \
	--batch_size 1 \
	--n_classes_det 2 \
	--n_finetune_classes_det 2 \
	--n_classes_clf 83 \
	--n_finetune_classes_clf 83 \
	--n_threads 12 \
	--checkpoint 1 \
	--modality_det RGB \
	--modality_clf RGB \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test  \
	--det_strategy median \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy median \
	--clf_queue_size 32 \
	--clf_threshold_pre 1.0 \
	--clf_threshold_final 0.15 \
	--stride_len 1