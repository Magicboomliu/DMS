

Submission_Model15(){
cd ..
cd HuggingFace_Inference
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
trainlist="../datafiles/kitti_2015_test.txt"
vallist="../datafiles/kitti_2015_test.txt"
# network_type=PAMStereo
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/plus_outside_center_4/best_D1_model.pt

# network_type=StereoNet
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/stereonet_baseline/best_D1_model.pt

# network_type=CFNet
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/cfnet_baseline/best_D1_model.pt
# output_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/For_Submission_Folder_KITTI2015/stereonet_cfnet

# network_type=StereoNet
# pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/stereonet_DMB/best_D1_model.pt
# output_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/For_Submission_Folder_KITTI2015/stereonet_DMB


network_type=CFNet
pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/Saved_Models_For_Submission/KITTI2015_Submissions/cfnet_dmb/best_D1_model.pt
output_path=/home/zliu/NeruaIPS2024/Ablations/KITTI/For_Submission_Folder_KITTI2015/cfnet_dmb

python submission.py --datapath $datapath \
                      --vallist $vallist \
                      --network_type $network_type \
                      --pretrained_model_path $pretrained_model_path \
                      --output_path $output_path \
                      --vis

}




Submission_Model15





