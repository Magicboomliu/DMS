TRAIN_SIMPLE_UNet(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/trainers/simple_unet
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/KITTI/KITTI_Raw'
dataset_name='kitti_raw'
trainlist='/home/zliu/Desktop/BMCV2024/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/Desktop/BMCV2024/datafiles/KITTI/kitti_raw_val.txt'
output_dir='/home/zliu/Desktop/BMCV2024/outputs/Simple_UNet'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_simple_unet'
pretrained_unet="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="fp16"  trainer_unet.py \
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
                  --pretrained_unet $pretrained_unet

}


TRAIN_SIMPLE_UNet_KITTI2015(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/trainers/simple_unet
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/KITTI/KITTIStereo/kitti_2015/'
dataset_name='kitti_raw'
trainlist="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2015_diffusion_list.txt"
vallist="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2015_diffusion_list.txt"
output_dir='/home/zliu/Desktop/BMCV2024/outputs/Simple_UNet_KITTI2015'
train_batch_size=1
num_train_epochs=300
gradient_accumulation_steps=16
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_simple_unet'
pretrained_unet="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="fp16"  trainer_unet_kitti_2015.py \
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
                  --pretrained_unet $pretrained_unet

}


TRAIN_SIMPLE_UNet_KITTI2012(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/trainers/simple_unet
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/KITTI/KITTIStereo/kitti_2012/'
dataset_name='kitti_raw'
trainlist="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2012_diffusion_list.txt"
vallist="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2012_diffusion_list.txt"
output_dir='/home/zliu/Desktop/BMCV2024/outputs/Simple_UNet_KITTI2012'
train_batch_size=1
num_train_epochs=300
gradient_accumulation_steps=16
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_simple_unet'
pretrained_unet="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="fp16"  trainer_unet_kitti_2012.py \
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
                  --pretrained_unet $pretrained_unet

}



TRAIN_SIMPLE_UNet_KITTI2012
# TRAIN_SIMPLE_UNet_KITTI2015

# TRAIN_SIMPLE_UNet


