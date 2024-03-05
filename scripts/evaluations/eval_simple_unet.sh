EVAL_SIMPLE_UNET_SD20(){

cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations/simple_unet_finetune/SD20
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/SD20/unet"
datapath='/media/zliu/data12/dataset/KITTI/KITTI_Raw'
trainlist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'
test_size=4
datathread=0
saved_folder="/media/zliu/data12/dataset/KITTI/val/Simple_SD20"

python gen_val_views_exist.py \
        --datapath $datapath \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datathread $datathread \
        --trainlist $trainlist \
        --vallist $vallist \
        --test_size $test_size \
        --saved_folder $saved_folder \
        --save_results \


}


EVAL_SIMPLE_UNET_SD15(){

cd ../..
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/evaluations/simple_unet_finetune/SD15
pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
pretrained_unet_path="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/SD15/unet"
datapath='/media/zliu/data12/dataset/KITTI/KITTI_Raw'
trainlist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt'
test_size=4
datathread=0
saved_folder="/media/zliu/data12/dataset/KITTI/val/Simple_SD15"

python gen_val_views_exist.py \
        --datapath $datapath \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datathread $datathread \
        --trainlist $trainlist \
        --vallist $vallist \
        --test_size $test_size \
        --saved_folder $saved_folder \
        --save_results \


}

EVAL_SIMPLE_UNET_SD15