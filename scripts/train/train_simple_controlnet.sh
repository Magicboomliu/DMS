SD20_Simple_ControlNet(){

# accelerate config default
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/trainers/simple_controlnet/SD20
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/liu/kitti_raw/KITTI_Raw'
dataset_name='kitti_raw'
trainlist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'
output_dir='/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/SD20_Simple_ControlNet'
train_batch_size=1
num_train_epochs=15
gradient_accumulation_steps=16
learning_rate=8e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_sd20_simple_controlNet'
unet_pretrained_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet"
controlnet_model_name_or_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet/"


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
                  --gradient_checkpointing \
                  --unet_pretrained_path $unet_pretrained_path \
                  --controlnet_model_name_or_path $controlnet_model_name_or_path

}

SD20_Simple_ControlNet