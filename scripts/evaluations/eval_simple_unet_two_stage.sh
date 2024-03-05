EVAL_SIMPLE_UNET_Two_Stage_SD20(){

cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations/simple_unet_two_stage/View_Independent

controlnet_model_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple_Two_Stage/SD20"
unet_path="Vhey/a-zovya-photoreal-v2"
requested_steps=32
requested_steps_inverse=10
strength=0.2
source_folder="/media/zliu/data12/dataset/KITTI/val/Simple_SD20/"
output_folder="/media/zliu/data12/dataset/KITTI/val/Simple_SD20_Enhanced/"
filelist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'



CUDA_VISIBLE_DEVICES=0 python super_resolution_run.py \
            --controlnet_model_path $controlnet_model_path \
            --unet_path $unet_path \
            --requested_steps $requested_steps \
            --requested_steps_inverse $requested_steps_inverse \
            --strength $strength \
            --output_folder $output_folder \
            --source_folder $source_folder \
            --filelist $filelist

}


EVAL_SIMPLE_UNET_Two_Stage_SD20