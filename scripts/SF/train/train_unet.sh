TRAIN_SIMPLE_UNet(){
cd ../../..
cd trainers/SF
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/'
trainlist='../../datafiles/sceneflow/sceneflow_training.txt'
output_dir='models_output/SceneFlow_Pretrained_Models'
train_batch_size=1
num_train_epochs=20
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='mpi_tracker_simple_unet'
pretrained_unet="none"
input_image_path="../../input_examples/SF/left_images/example.png"

CUDA_VISIBLE_DEVICES=0,2 accelerate launch --mixed_precision="fp16"  trainer_unet.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                   --trainlist $trainlist \
                  --dataset_path $root_path  \
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
                  --input_image_path $input_image_path

}

TRAIN_SIMPLE_UNet


