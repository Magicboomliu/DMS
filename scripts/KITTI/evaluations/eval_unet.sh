EVAL_UNet_KITTI(){
cd ../..
cd /home/zliu/Desktop/BMCV2024/evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="{KITTI_CheckPoint_PATH}"
datapath="Your KITTI RAW PATH"
input_fname_list="/home/zliu/Desktop/BMCV2024/datafiles/KITTI/kitti_raw_val.txt"
output_folder_path="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Validation"


python unet_evaluate_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}
EVAL_UNet_KITTI

