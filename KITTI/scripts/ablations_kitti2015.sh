Ablation_Baseline(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_train.txt"
vallist="../datafiles/kitti_2015_val.txt"
logging_dir="../logs"
output_dir="../output_ablations/kitti_baseline"
tracker_project_name="tracker_project_kitti_raw"
batch_size=4
datathread=4
visible_list="['left','right']"
maxdisp=192
num_train_epochs=600
max_train_steps=30000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"
loss_type='simple'
network_type="PAMStereo"
resume_from_checkpoint="/home/zliu/NeruaIPS2024/Pretrained_Models_For_NeurIPS/Unsupervised_Stereo_Matching/Ablations/SceneFlow/best_EPE_model.pt"


CUDA_VISIBLE_DEVICES=0,3 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
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
          --network_type $network_type \
        --resume_from_checkpoint $resume_from_checkpoint
          

}

Ablation_Plus_Outside(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_train.txt"
vallist="../datafiles/kitti_2015_val.txt"
logging_dir="../logs"
output_dir="../output_ablations/kitti_plus_outside"
tracker_project_name="tracker_project_kitti_raw"
batch_size=4
datathread=4
visible_list="['left','right','left_left','right_right']"
maxdisp=192
num_train_epochs=600
max_train_steps=30000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"
loss_type='plusoutside'
network_type="PAMStereo"
resume_from_checkpoint="/home/zliu/NeruaIPS2024/Pretrained_Models_For_NeurIPS/Unsupervised_Stereo_Matching/Ablations/SceneFlow/best_EPE_model.pt"


CUDA_VISIBLE_DEVICES=0,3 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
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
          --network_type $network_type \
          --resume_from_checkpoint $resume_from_checkpoint
          

}



Ablation_Plus_Center(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_train.txt"
vallist="../datafiles/kitti_2015_val.txt"
logging_dir="../logs"
output_dir="../output_ablations/kitti_plus_center"
tracker_project_name="tracker_project_kitti_raw"
batch_size=4
datathread=4
visible_list="['left','right','center']"
maxdisp=192
num_train_epochs=600
max_train_steps=30000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"
loss_type='pluscenter'
network_type="PAMStereo"
resume_from_checkpoint="/home/zliu/NeruaIPS2024/Pretrained_Models_For_NeurIPS/Unsupervised_Stereo_Matching/Ablations/SceneFlow/best_EPE_model.pt"



CUDA_VISIBLE_DEVICES=0,3 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
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
          --network_type $network_type \
          --resume_from_checkpoint $resume_from_checkpoint
          

}



Ablation_Plus_Center_And_Outside(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_train.txt"
vallist="../datafiles/kitti_2015_val.txt"
logging_dir="../logs"
output_dir="../output_ablations/kitti_plus_center_outside"
tracker_project_name="tracker_project_kitti_raw"
batch_size=4
datathread=4
visible_list="['left','right','left_left','right_right','center']"
maxdisp=192
num_train_epochs=600
max_train_steps=60000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"
loss_type='plusoutside_center'
network_type="PAMStereo"
resume_from_checkpoint="/home/zliu/NeruaIPS2024/Pretrained_Models_For_NeurIPS/Unsupervised_Stereo_Matching/Ablations/SceneFlow/best_EPE_model.pt"


CUDA_VISIBLE_DEVICES=0,3 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
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
          --network_type $network_type \
          --resume_from_checkpoint $resume_from_checkpoint
          

}


#Ablation_Baseline
#Ablation_Plus_Outside
#Ablation_Plus_Center
Ablation_Plus_Center_And_Outside







