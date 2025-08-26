#!/bin/bash

# Get pruning ratio from command line argument
PRUNING_RATIO=$1
CUDA_VISIBLE_DEVICES=$2
echo "Using pruning ratio: $PRUNING_RATIO"


export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="./datasets/flux/finetune_dataset"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch scripts/retraining/train_text_to_image_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pruned_transformer_model_path="./results/prune_results/pruned_model_$PRUNING_RATIO.pkl" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --rank=256 \
  --train_batch_size=2 \
  --max_train_steps=40000 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --validation_prompt "A clock tower floating in a sea of clouds" "A cozy library with a roaring fireplace" "an astronaut riding a rainbow unicorn" \
  --validation_epochs 1 \
  --checkpointing_steps=4000 \
  --output_dir="./results/retrain_results/flux_lora/prune_$PRUNING_RATIO" \
  --allow_tf32 \
  --gradient_accumulation_steps 2 
#   --push_to_hub
