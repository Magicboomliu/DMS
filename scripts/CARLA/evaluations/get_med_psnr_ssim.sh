GET_THE_PSNR_AND_SSIM_MED(){

cd ../..
cd /home/zliu/CVPR2025/DiffusionMultiBaseline/evaluations/carla_evaluations

datapath="/data3/CALRA/CARLA/"
target_datapath="/data3/CARLA_Novel_Views/"
fnamelist="/home/zliu/CVPR2025/DiffusionMultiBaseline/datafiles/CARLA/testing.txt"
output_json_path="/home/zliu/CVPR2025/DiffusionMultiBaseline/output_validation/med_psnr_ssim.json"

python get_med_psnr_ssim.py \
        --datapath $datapath \
        --fnamelist $fnamelist \
        --output_json_path $output_json_path \
        --target_datapath $target_datapath
}

GET_THE_PSNR_AND_SSIM_MED