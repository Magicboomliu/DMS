TRAIN_MPI_ORIGINAL(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/logs"
output_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/simple_pamstereo"
tracker_project_name="tracker_project_MPI"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='simple'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}


TRAIN_MPI_Plus_OutSide(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/logs_plus"
output_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_outside"
tracker_project_name="tracker_project_MPI_Outside"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_outside'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}



TRAIN_MPI_Plus_CenterAll(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/logs_plus_all_center"
output_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_center_all"
tracker_project_name="tracker_project_MPI_center_all"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_one_third','rendered_two_third']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_center_all'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}


TRAIN_MPI_Plus_All_In_All(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/logs_plus_all_in_all"
output_dir="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_all_in_all"
tracker_project_name="tracker_project_MPI_center_all"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_one_third','rendered_two_third','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_center_all_in_all'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}


TRAIN_MPI_Outside_Plus_Center(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NeruaIPS/logs_plus_double_center"
output_dir="/home/zliu/NeruaIPS/outputs/plus_double_center"
tracker_project_name="tracker_project_MPI_Double_Center"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_outside_center'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}


TRAIN_MPI_Outside_Plus_Center(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/liu/Sintel/"
trainlist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/NeruaIPS/logs_plus_double_center"
output_dir="/home/zliu/NeruaIPS/outputs/plus_double_center"
tracker_project_name="tracker_project_MPI_Double_Center"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_outside_center'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate
          

}





# TRAIN_MPI_Outside_Plus_Center
# TRAIN_MPI_Plus_All_In_All

# TRAIN_MPI_ORIGINAL