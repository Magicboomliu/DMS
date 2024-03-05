INFERENCE_TWO_STAGE_UNET_SD20_View_Independent(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/Inference/simple_unet_two_stage/SD20/View_Independent
controlnet_model_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple_Two_Stage/SD20"
unet_path="Vhey/a-zovya-photoreal-v2"
input_image_path="/media/zliu/data12/dataset/KITTI/val/Simple_SD20/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/left_from_right_0000000029.png"
requested_steps=32
requested_steps_inverse=10
strength=0.2


CUDA_VISIBLE_DEVICES=0 python super_resolution_run.py \
            --controlnet_model_path $controlnet_model_path \
            --unet_path $unet_path \
            --input_image_path $input_image_path\
            --requested_steps $requested_steps \
            --requested_steps_inverse $requested_steps_inverse \
            --strength $strength
}

INFERENCE_TWO_STAGE_UNET_SD20_View_Consistency(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/Inference/simple_unet_two_stage/SD20/View_Consistency
controlnet_model_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple_Two_Stage/SD20"
unet_path="Vhey/a-zovya-photoreal-v2"
input_image_path="/media/zliu/data12/dataset/KITTI/val/Simple_SD20/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/left_from_right_0000000029.png"
requested_steps=32
requested_steps_inverse=10
strength=0.2


CUDA_VISIBLE_DEVICES=0 python super_resolution_run.py \
            --controlnet_model_path $controlnet_model_path \
            --unet_path $unet_path \
            --input_image_path $input_image_path\
            --requested_steps $requested_steps \
            --requested_steps_inverse $requested_steps_inverse \
            --strength $strength

}

INFERENCE_TWO_STAGE_UNET_SD20_View_Consistency