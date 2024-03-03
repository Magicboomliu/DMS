LAUNCH_SIMPLE_UNET_SD20(){

cd .. 
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/trainers/simple_unet_finetune/SD20
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/media/zliu/data12/dataset/KITTI/KITTI_Raw'
dataset_name='kitti_raw'
trainlist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'
output_dir='/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/simple_unet_sd20'
train_batch_size=1
num_train_epochs=15
gradient_accumulation_steps=16
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='simple_unet_sd20'
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/unet"


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"   simple_trainer_unet.py \
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
                  --pretrained_unet_path $pretrained_unet_path

}

LAUNCH_SIMPLE_UNET_SD20