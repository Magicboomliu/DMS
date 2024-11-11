TRAIN_CARLA_SCRIPTS(){
cd ../../..
cd trainers/CARLA
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path="/data3/CALRA/CARLA/"
dataset_name='carla'
trainlist='../../datafiles/CARLA/training.txt'
vallist='../../datafiles/CARLA/testing.txt'
output_dir='../../output_validation'
train_batch_size=1
num_train_epochs=50
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='carla_tracker_simple_unet'
pretrained_unet="/home/zliu/CVPR2025/DiffusionMultiBaseline/output_validation/checkpoint_v1/unet/"
input_image_example_path="/home/zliu/CVPR2025/DiffusionMultiBaseline/simple_input_images/left_image.png"
lr_scheduler='cosine'

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --mixed_precision="fp16"  trainer_unet_carla.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --pretrained_unet $pretrained_unet \
                  --input_image_example_path $input_image_example_path \
                  --lr_scheduler $lr_scheduler
                  

}

TRAIN_CARLA_SCRIPTS


