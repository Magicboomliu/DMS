EVAL_UNet_KITTI_2015(){
cd ../../..
cd evaluations/kitti2015_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="../../../Pretrained_Models_For_NeurIPS/Diffusions/KITTI2015/unet/"
datapath="/data1/KITTI/KITTIStereo/kitti_2015/"
input_fname_list="../../datafiles/KITTI/KITTI2015/kitti_2015_all.txt"
output_folder_path="/data1/Examples"


CUDA_VISIBLE_DEVICES=0 python unet_new_view_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}


EVAL_UNet_KITTI_2015




