EVAL_UNet_SF(){
cd ../..
cd /home/zliu/Desktop/NeuraIPS2024/evaluations_sceneflow
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/data1/SceneFlow_Pretrained_Models/checkpoint/"
datapath="/data1/"
input_fname_list='/home/zliu/Desktop/NeuraIPS2024/datafiles/sceneflow/sub01.txt'
output_folder_path="/data1/SF_Rendered_Results/"


CUDA_VISIBLE_DEVICES=0 python get_additional_view_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

EVAL_UNet_SF