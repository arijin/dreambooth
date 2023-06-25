export MODEL_NAME="/home/jinyujie/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1"
export INSTANCE_DIR="./work_dirs/people_1/instance_images"
export OUTPUT_DIR="./work_dirs/people_1/output_dir"

CUDA_VISIBLE_DEVICES=3 python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of cartoon man" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  # --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of cartoon man in a house" \
  --validation_epochs=50 \
  --seed="0" \
#  --push_to_hub
