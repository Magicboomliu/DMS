AblationSF_Baseline(){
cd ..
cd HuggingFace_Trainer_SF/
datapath="/data1/zliu/"
trainlist="/home/zliu/AblationStudies_Stereo_Matching/datafiles/SF/train.txt"
vallist="/home/zliu/AblationStudies_Stereo_Matching/datafiles/SF/val.txt"
logging_dir="/home/zliu/AblationStudies_Stereo_Matching/logs"
output_dir="/home/zliu/AblationStudies_Stereo_Matching/SF_outputs/simple_pamstereo"
tracker_project_name="tracker_project_sf_raw"
batch_size=4
datathread=4
visible_list="['left','right','disp']"
maxdisp=192
num_train_epochs=70
max_train_steps=100000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"
loss_type='simple'
network_type="PAMStereo"


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
          --network_type $network_type
        
}

AblationSF_Baseline
