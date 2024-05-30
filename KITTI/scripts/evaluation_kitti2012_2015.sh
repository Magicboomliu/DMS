Evaluation_Model15(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_train.txt"
vallist="../datafiles/kitti_2015_val.txt"
network_type=PAMStereo
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/KITTI2015_Outputs_SF/simple_pamstereo/best_EPE_model.pt
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/KITTI2015_Outputs_SF/plus_outside/best_D1_model.pt
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/KITTI2015_Outputs_SF/plus_center/best_D1_model.pt
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/KITTI2015_Outputs_SF/plus_outside_center_latest/best_D1_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/plus_outside_center/best_D1_model.pt
output_path="outputs_ablations/"

python evaluations.py --datapath $datapath \
                      --vallist $vallist \
                      --network_type $network_type \
                      --pretrained_model_path $pretrained_model_path \
                      --output_path $output_path \
                      # --vis

}

Evaluation_Model12(){
cd ..
cd HuggingFace_Trainer/
datapath="/data1/KITTI/KITTIStereo/kitti_2012/"
trainlist="../datafiles/kitti_2012_train.txt"
vallist="../datafiles/kitti_2012_val.txt"
network_type=PAMStereo
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Ablations_KITTI2012/simple_pamstereo/best_D1_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Ablations_KITTI2012/plus_center/best_D1_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Ablations_KITTI2012/plus_outside/best_D1_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Ablations_KITTI2012/plus_outside_center/best_D1_model.pt
output_path="outputs_ablations/"

python kitti_2012_evaluations.py --datapath $datapath \
                      --vallist $vallist \
                      --network_type $network_type \
                      --pretrained_model_path $pretrained_model_path \
                      --output_path $output_path \
                      # --vis

}


Evaluation_Model15
Evaluation_Model12
