EVAL_UNet_SF(){
cd ../../..
cd evaluations/sceneflow_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/data1/SceneFlow_Pretrained_Models/checkpoint/"
datapath="/data1/"
input_fname_list='../../datafiles/sceneflow/sceneflow_test.list'
output_folder_path="SF_Rendered_Results_DEMO/"

CUDA_VISIBLE_DEVICES=0 python evalaution_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path
}

EVAL_UNet_SF