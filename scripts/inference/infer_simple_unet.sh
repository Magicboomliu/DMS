INFERNECE_SIMPLE_UNET_SD20(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/Inference/simple_unet_finetune/SD20
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/SD20/unet"
example_image_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/input_examples/left_images/example3.png"
output_folder_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Inference_Results/Simple_Unet/SD20"

CUDA_VISIBLE_DEVICES=0 python inference_simple_unet.py \
            --pretrained_model_name_or_path $pretrained_model_name_or_path \
            --pretrained_unet_path $pretrained_unet_path \
            --example_image_path $example_image_path \
            --output_folder_path $output_folder_path

}

INFERNECE_SIMPLE_UNET_SD15(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/Inference/simple_unet_finetune/SD15
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/SD15/unet"
example_image_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/input_examples/left_images/example3.png"
output_folder_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Inference_Results/Simple_Unet/SD15"

CUDA_VISIBLE_DEVICES=0 python inference_simple_unet.py \
            --pretrained_model_name_or_path $pretrained_model_name_or_path \
            --pretrained_unet_path $pretrained_unet_path \
            --example_image_path $example_image_path \
            --output_folder_path $output_folder_path

}

#INFERNECE_SIMPLE_UNET_SD20
INFERNECE_SIMPLE_UNET_SD20