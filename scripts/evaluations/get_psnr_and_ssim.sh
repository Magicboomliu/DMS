GET_THE_PSNR_AND_SSIM(){

cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations/
datapath="/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
target_datapath="/media/zliu/data12/dataset/KITTI/val/Simple_SD15/"
fnamelist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'
output_json_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Evaluation_Results/Simple_Unet/SD15.json"

python compute_psnr_ssim.py \
        --datapath $datapath \
        --fnamelist $fnamelist \
        --output_json_path $output_json_path \
        --target_datapath $target_datapath


}

GET_THE_PSNR_AND_SSIM