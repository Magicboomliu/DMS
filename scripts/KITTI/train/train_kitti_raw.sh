TRAIN_SIMPLE_UNet_KITTI_Raw(){
cd ../../..
cd trainers/KITTI
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/KITTI/KITTI_Raw'
dataset_name='kitti_raw'
trainlist='../../datafiles/KITTI/KITTI_raw/kitti_raw_train.txt'
vallist='../../datafiles/KITTI/KITTI_raw/kitti_raw_val.txt'
output_dir='/data1/Test_Demos'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_simple_unet'
pretrained_unet="../../../Pretrained_Models_For_NeurIPS/Diffusions/KITTI_Raw/unet/"
input_image_example_path="../../input_examples/kitti/left_images/example0.png"

CUDA_VISIBLE_DEVICES=0,2 accelerate launch --mixed_precision="fp16"  trainer_unet_kitti_raw.py \
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
                  --input_image_example_path $input_image_example_path

}

TRAIN_SIMPLE_UNet_KITTI_Raw



