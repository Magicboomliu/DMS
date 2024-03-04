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

python gen_val_views.py \
        --datapath $datapath \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --pretrained_unet_path $pretrained_unet_path \
        --datathread $datathread \
        --trainlist $trainlist \
        --vallist $vallist \
        --test_size $test_size

}


EVAL_SIMPLE_UNET_SD20