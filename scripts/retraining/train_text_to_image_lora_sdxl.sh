#!/bin/bash

# Get pruning ratio from command line argument
PRUNING_RATIO=$1
CUDA_VISIBLE_DEVICES=$2
echo "Using pruning ratio: $PRUNING_RATIO"


export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="./datasets/sdxl/finetune_dataset"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch scripts/retraining/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --pruned_unet_model_path="./results/prune_results/pruned_model_$PRUNING_RATIO.pkl" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --rank=16 \
  --train_batch_size=32 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --validation_prompt="an astronaut riding a rainbow unicorn" \
  --validation_epochs 5 \
  --checkpointing_steps=1000 \
  --output_dir="./results/retrain_results/sdxl_lora/prune_$PRUNING_RATIO"
#   --push_to_hub
