GET_THE_PSNR_AND_SSIM(){

cd ../..
cd /home/zliu/Desktop/BMCV2024/kitti2012_evaluations
datapath="/data1/KITTI/KITTIStereo/kitti_2012/"
target_datapath="/data1/KITTI/Rendered_Results/KITTI2012_Rendered/"
fnamelist="/home/zliu/Desktop/BMCV2024/datafiles/filenames/kitti_diffusion/kitti_2012_diffusion_list.txt"
output_json_path="/home/zliu/Desktop/BMCV2024/outputs/comparsions/new_kitti2012.json"

python get_psnr_ssim.py \
        --datapath $datapath \
        --fnamelist $fnamelist \
        --output_json_path $output_json_path \
        --target_datapath $target_datapath


}

GET_THE_PSNR_AND_SSIM