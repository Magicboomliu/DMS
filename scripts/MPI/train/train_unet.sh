TRAIN_SIMPLE_UNet(){
cd ../../..
cd trainers/MPI
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/Sintel/training/'
trainlist='../../datafiles/MPI/MPI_All.txt'
output_dir='/data1/Test_Demos'
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='mpi_tracker_simple_unet'
pretrained_unet="../../../Pretrained_Models_For_NeurIPS/Diffusions/MPI/unet/"
input_image_path="../../input_examples/MPI/left_images/example5.png"
lr_scheduler="cosine"

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
                  --input_image_path $input_image_path \
                  --lr_scheduler $lr_scheduler

}

TRAIN_SIMPLE_UNet


