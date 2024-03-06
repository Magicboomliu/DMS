EVAL_CONTROLNET(){
cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations/simple_controlnet/SD20

pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet/"
pretrained_controlnet="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/Simple_ControlNet/"
datapath="/media/zliu/data12/dataset/KITTI/KITTI_Raw"
filelist="/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt"
output_folder_path="/media/zliu/data12/dataset/KITTI/val/sd20_simple_controlnet"

python get_view_exist.py \
  --pretrained_model_name_or_path $pretrained_model_name_or_path \
  --pretrained_unet_path $pretrained_unet_path \
  --pretrained_controlnet $pretrained_controlnet \
  --datapath $datapath \
  --filelist $filelist \
  --output_folder_path $output_folder_path

}

EVAL_CONTROLNET