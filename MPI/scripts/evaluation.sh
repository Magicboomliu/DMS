Sintel_Evaluation(){
cd ..
cd /home/zliu/NeruaIPS2024/Ablations/Sintel/HuggingFace_Inference

datapath="/media/zliu/data12/dataset/Sintel/"
vallist="/home/zliu/NeruaIPS2024/Ablations/Sintel/filenames/MPI/MPI_Val_Sub_List.txt"
network_type=PAMStereo
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/Sintel/outputs/simple_pamstereo/best_EPE_model.pt
pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/Sintel/outputs/plus_outside/best_EPE_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/Sintel/outputs/plus_center/best_EPE_model.pt
#pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/Sintel/outputs/plus_outside_center/best_EPE_model.pt
output_path=/home/zliu/NeruaIPS2024/Ablations/Sintel/Ablation_results_SintelMPI/PAMStereo_Plus_Outside

python evaluation.py --datapath $datapath \
                     --vallist $vallist \
                     --network_type $network_type \
                     --pretrained_model_path $pretrained_model_path \
                     --output_path $output_path \
                     --vis


}


Sintel_Evaluation