EVAL_UNet_KITTI_2015(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/kitti2015_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/outputs/Simple_UNet_KITTI2015/checkpoint"
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2015_diffusion_list.txt"
output_folder_path="/data1/KITTI/Rendered_Results/KITTI2015_Rendered"


python unet_evaluation_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

# EVAL_UNet_KITTI_2015

EVAL_UNet_KITTI_2015_Raw(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/kitti2015_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/BMCV2024/pretrained_models/checkpoint"
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2015_diffusion_list.txt"
output_folder_path="/data1/KITTI/Rendered_Results/KITTI2015_Rendered_OLD"

python unet_evaluation_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

EVAL_UNet_KITTI_2015_Raw


