#"$1" pretrain path
#"$2" model
#"$3" width_mult

python main.py --root_path ~/qyh/Real-time-GesRec/ \
	--video_path ~/qyh/datasets/EgoGesture/videos/ \
	--annotation_path annotation_EgoGesture/egogestureall.json \
	--result_path train_res/ \
	--pretrain_path "/home/aipu/qyh/Real-time-GesRec/results/shared/jester_resnext_101_RGB_32.pth" \
	--dataset egogesture \
	--n_classes 27 \
	--n_finetune_classes 84 \
	--model resnext \
	--width_mult 1.0 \
	--model_depth 101 \
	--resnet_shortcut B \
	--resnext_cardinality 32 \
	--train_crop random \
	--learning_rate 0.01 \
	--sample_duration 32 \
	--modality RGB \
	--pretrain_modality RGB \
	--downsample 1 \
	--batch_size 24 \
	--n_threads 12 \
	--checkpoint 1 \
	--n_val_samples 1 \
    --n_epochs 60 \
    --ft_portion complete \
	# --no_train \
 	# --no_val \
 	# --test
 	# --resume_path Efficient-3DCNNs/results/egogesture_shufflenet_2.0x_Depth_16_checkpoint.pth \
 		# --pretrain_path Efficient-3DCNNs/results/jester_mobilenetv2_1.0x_RGB_16_best.pth \
 			# --pretrain_path Efficient-3DCNNs-Ahmet/results/jester_resnext_101_RGB_16_checkpoint.pth \
	# --n_finetune_classes 83 \
	    # --test_subset val \








 # python main.py --root_path ~/ \
	# --video_path ~/datasets/EgoGesture \
	# --annotation_path Efficient-3DCNNs-Ahmet/annotation_EgoGesture/egogestureall_but_None.json \
	# --result_path Efficient-3DCNNs-Ahmet/results \
	# --pretrain_path "$1" \
	# --dataset egogesture \
	# --n_classes 27 \
	# --n_finetune_classes 84 \
	# --model "$2" \
	# --groups 3 \
	# --version 1.1 \
	# --width_mult "$3" \
	# --train_crop random \
	# --learning_rate 0.1 \
	# --sample_duration 16 \
	# --modality RGB \
	# --downsample 2 \
	# --batch_size 80 \
	# --n_threads 16 \
	# --checkpoint 1 \
	# --n_val_samples 1 \
 #    --n_epochs 20 \
 #    --train_validate \
 #    --test_subset test \
 #    --ft_portion last_layer 
	# # --no_train \
 # 	# --no_val \
 # 	# --test
 # 	# --resume_path Efficient-3DCNNs/results/egogesture_shufflenet_2.0x_Depth_16_checkpoint.pth \
 # 		# --pretrain_path Efficient-3DCNNs/results/jester_mobilenetv2_1.0x_RGB_16_best.pth \