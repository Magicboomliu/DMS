Inference_CARLA_Existing(){
cd ../../..
cd evaluations/carla_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/CVPR2025/DiffusionMultiBaseline/output_validation/checkpoint_v1/unet"
datapath="/data3/CALRA/CARLA/"
input_fname_list="/home/zliu/CVPR2025/DiffusionMultiBaseline/datafiles/CARLA/testing.txt"
output_folder_path="/data3/CARLA_Novel_Views"


CUDA_VISIBLE_DEVICES=0 python existing_view_gen.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}


Inference_CARLA_Existing

