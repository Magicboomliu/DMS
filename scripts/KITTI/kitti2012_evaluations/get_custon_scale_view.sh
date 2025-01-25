EVAL_UNet_KITTI_2012_Scale_X(){
cd ../../..
cd evaluations/kitti2012_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/data1/KITTI/Pretrained_Models/KITTI/kitti2012_pretrained_models/unet/"
datapath="/data1/KITTI/KITTIStereo/kitti_2012/"
input_fname_list="../../datafiles/KITTI/KITTI2012/kitti_2012_trantest.txt"
output_folder_path="/data1/KITTI/KITTIStereo/KITTI2012_Additional_Views/"
scale_factor=3.0

CUDA_VISIBLE_DEVICES=2 python unet_scale_X_view_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path \
        --scale_factor $scale_factor

}



EVAL_UNet_KITTI_2012_Scale_X
