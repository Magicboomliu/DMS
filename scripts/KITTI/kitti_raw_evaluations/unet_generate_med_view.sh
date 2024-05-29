EVAL_UNet_KITTI_SUB01(){
cd ../../..
cd evaluations/kitti_raw_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="../../../Pretrained_Models_For_NeurIPS/Diffusions/KITTI_Raw/unet/"
datapath="/data1/KITTI/KITTI_Raw/"
input_fname_list="../../datafiles/KITTI/KITTI_raw/kitti_raw_val.txt"
output_folder_path="/data1/KITTI/Rendered_Results/DEMO"


CUDA_VISIBLE_DEVICES=0 python med_view_gen.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path
}

EVAL_UNet_KITTI_SUB01

