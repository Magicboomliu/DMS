GET_THE_PSNR_AND_SSIM(){

cd ../..
cd /home/zliu/Desktop/NeuraIPS2024/evaluations_sceneflow
datapath="/data1/"
target_datapath="/data1/SF_Rendered_Results/"
fnamelist='/home/zliu/Desktop/NeuraIPS2024/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_json_path="/home/zliu/Desktop/NeuraIPS2024/outputs/psnr.json"

python compute_psnr_ssim.py \
        --datapath $datapath \
        --fnamelist $fnamelist \
        --output_json_path $output_json_path \
        --target_datapath $target_datapath


}

GET_THE_PSNR_AND_SSIM