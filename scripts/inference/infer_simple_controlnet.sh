INFERENCE_CONTROLNET_SD20_SIMPLE(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/Inference/simple_controlnet/SD20

pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet/"
pretrained_controlnet="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet/"
example_image_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/input_examples/left_images/example3.png"
output_folder_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Inference_Results/Simple_Unet/SD20_Controlnet"

CUDA_VISIBLE_DEVICES=0 python inference_simple_controlnet.py \
            --pretrained_model_name_or_path $pretrained_model_name_or_path \
            --pretrained_unet_path $pretrained_unet_path \
            --example_image_path $example_image_path \
            --output_folder_path $output_folder_path \
            --pretrained_controlnet $pretrained_controlnet

}

INFERENCE_CONTROLNET_SD20_SIMPLE    