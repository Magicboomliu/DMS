TRAIN_SIMPLE_UNet(){
cd ../..
cd /home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/trainers/simple_unet
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/Sintel/training/'
trainlist='/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/datafiles/MPI/MPI_Training.txt'
output_dir='/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/outputs/MPI/Simple_UNet'
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='mpi_tracker_simple_unet'
pretrained_unet="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/pretrained_models/checkpoint"
input_image_path="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/input_examples/left_images/example5.png"
lr_scheduler="cosine"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="fp16"  trainer_unet.py \
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


