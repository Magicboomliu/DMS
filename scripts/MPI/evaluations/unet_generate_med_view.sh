EVAL_UNet_MPI_Sub01(){
cd ../..
cd /home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/outputs/MPI/Simple_UNet/checkpoint/"
datapath="/data1/Sintel/training/"
input_fname_list="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/datafiles/MPI/Subset/mpi_training0.txt"
output_folder_path="/data1/Sintel/rendered/"


CUDA_VISIBLE_DEVICES=0 python net_new_view_med.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path
}


EVAL_UNet_MPI_Sub02(){
cd ../..
cd /home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/outputs/MPI/Simple_UNet/checkpoint/"
datapath="/data1/Sintel/training/"
input_fname_list="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/datafiles/MPI/Subset/mpi_training1.txt"
output_folder_path="/data1/Sintel/rendered/"


CUDA_VISIBLE_DEVICES=1  python net_new_view_med.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

EVAL_UNet_MPI_Sub03(){
cd ../..
cd /home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/outputs/MPI/Simple_UNet/checkpoint/"
datapath="/data1/Sintel/training/"
input_fname_list="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/datafiles/MPI/Subset/mpi_training2.txt"
output_folder_path="/data1/Sintel/rendered/"


CUDA_VISIBLE_DEVICES=2 python net_new_view_med.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}

EVAL_UNet_MPI_Sub04(){
cd ../..
cd /home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/evaluations/new_view_generation
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/outputs/MPI/Simple_UNet/checkpoint/"
datapath="/data1/Sintel/training/"
input_fname_list="/home/zliu/Desktop/MyLatest/DiffusionMultiBaseline/datafiles/MPI/Subset/mpi_training3.txt"
output_folder_path="/data1/Sintel/rendered/"


CUDA_VISIBLE_DEVICES=3 python net_new_view_med.py \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datapath $datapath \
        --input_fname_list  $input_fname_list \
        --output_folder_path $output_folder_path

}


EVAL_UNet_MPI_Sub04
