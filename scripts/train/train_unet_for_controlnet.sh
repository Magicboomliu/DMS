LAUNCH_TRAINING_IMAGE_KITTI_FineTune_Cond(){

cd .. 
cd new_unet_traning
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/liu/kitti_raw/KITTI_Raw'
dataset_name='kitti_raw'
trainlist='/home/zliu/ECCV2024/PoseConidtioned/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/ECCV2024/PoseConidtioned/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_val.txt'
output_dir='../outputs/img2img_kitti_multi_baseline_sync_controlnet'
train_batch_size=1
num_train_epochs=15
gradient_accumulation_steps=16
learning_rate=8e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_multi_baseline_sync_controlnet'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  trainer_controlnet.py \
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
                  --gradient_checkpointing 

}

LAUNCH_TRAINING_IMAGE_KITTI_FineTune_Cond