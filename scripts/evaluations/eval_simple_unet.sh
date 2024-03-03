EVAL_SIMPLE_UNET_SD20(){
datapath="/media/zliu/data12/dataset/KITTI/KITTI_Raw"
rendered_path="/media/zliu/data12/dataset/KITTI/Temp/Kitti_raw_existed_val/simple_controlnet_resize_max_768/"
validation_files="/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt"
output_json_files="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Evaluation_Results/Simple_Unet/SD15_val.json"

cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations

python compute_psnr_ssim.py \
        --datapath $datapath \
        --rendered_path $rendered_path \
        --validation_files $validation_files \
        --output_json_files $output_json_files

}


EVAL_SIMPLE_UNET_SD20