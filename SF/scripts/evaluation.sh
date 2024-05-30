Evaluation_the_SceneFlow(){

cd ..
cd /home/zliu/NeruaIPS2024/Ablations/SF/HuggingFace_Inference

datapath=/media/zliu/data12/dataset/sceneflow/
vallist=/home/zliu/NeruaIPS2024/Ablations/SF/datafiles/SF/val.txt
network_type=PAMStereo
pretrained_model_path=/home/zliu/NeruaIPS2024/Ablations/SF/SF_outputs/plus_center2/best_EPE_model.pt
# pretrained_model_path=
output_path=/home/zliu/NeruaIPS2024/Ablations/SF/Ablation_results_SceneFlow/Plus_Center


python eval_sf.py --datapath $datapath \
                    --vallist $vallist \
                    --network_type $network_type \
                    --pretrained_model_path $pretrained_model_path \
                    --output_path $output_path \
                    --vis

}

Evaluation_the_SceneFlow