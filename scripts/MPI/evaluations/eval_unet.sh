EVAL_UNet_MPI_Sub01(){
cd ../../..
cd evaluations/mpi_evaluations
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="../../../Pretrained_Models_For_NeurIPS/Diffusions/MPI/unet/"
datapath="/data1/Sintel/training/"
input_fname_list="../../datafiles/MPI/MPI_All.txt"
output_folder_path="/data1/Examples"

CUDA_VISIBLE_DEVICES=0 python unet_evaluate_pipeline.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path
}

EVAL_UNet_MPI_Sub01
