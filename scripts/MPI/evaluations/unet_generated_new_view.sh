EVAL_UNet_KITTI_SUB01(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTI_Raw"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/KITTI/sub_set_train/kitti_raw_train_sub_01.txt"
output_folder_path="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"


python unet_new_view.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

EVAL_UNet_KITTI_SUB02(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTI_Raw"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/KITTI/sub_set_train/kitti_raw_train_sub_02.txt"
output_folder_path="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"


python unet_new_view.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}


EVAL_UNet_KITTI_SUB03(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTI_Raw"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/KITTI/sub_set_train/kitti_raw_train_sub_03.txt"
output_folder_path="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"


python unet_new_view.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}


EVAL_UNet_KITTI_SUB04(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTI_Raw"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/KITTI/sub_set_train/kitti_raw_train_sub_04.txt"
output_folder_path="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"


python unet_new_view.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}



EVAL_UNet_KITTI_SUB01

