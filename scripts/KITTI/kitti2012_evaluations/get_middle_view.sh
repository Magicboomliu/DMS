EVAL_UNet_KITTI_2012(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/kitti2012_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/outputs/Simple_UNet_KITTI2012/checkpoint"
datapath="/data1/KITTI/KITTIStereo/kitti_2012/"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2012_diffusion_list.txt"
output_folder_path="/data1/KITTI/Rendered_Results/KITTI2012_Additional_Views"


CUDA_VISIBLE_DEVICES=2 python unet_med_view_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}



EVAL_UNet_KITTI_2012_Raw(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/kitti2012_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTIStereo/kitti_2012/"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2012_diffusion_list.txt"
output_folder_path="/data1/KITTI/Rendered_Results/KITTI2012_Additional_Views_OLD"

CUDA_VISIBLE_DEVICES=3 python unet_med_view_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

# EVAL_UNet_KITTI_2012
EVAL_UNet_KITTI_2012_Raw


