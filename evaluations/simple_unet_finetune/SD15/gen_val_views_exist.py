import torch
import diffusers

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)

from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append("../../..")

from dataloader.utils import read_text_lines
from tqdm import tqdm
import os
import skimage.io
import numpy as np
from PIL import Image
import argparse

from evaluations.eval_dataloader.dataset_configuration import prepare_dataset
from evaluations.simple_unet_finetune.SD20.evaluation_pipeline_exist import SD20UNet_Validation_Pipeline
import torch.nn.functional as F

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Kitti Inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        default="/home/zliu/ACMMM2024/DiffusionMultiBaseline/pretrained_models/UNet_Simple/SD20/unet",
        help="Path to loaded the pretrained unet.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="",
        help="Example Image Path",
    )
    parser.add_argument(
        "--trainlist",
        type=str,
        default='/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_train.txt',
        help="Example Image Path",
    )
    parser.add_argument(
        "--vallist",
        type=str,
        default="/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt",
        help="Example Image Path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Example Image Path",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=4,
        help="Example Image Path",
    )
    parser.add_argument(
        "--datathread",
        type=int,
        default=4,
        help="Example Image Path",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to saved_results",
    )
    parser.add_argument(
        "--saved_folder",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/val/Simple_SD20",
        help="Whether to saved_results",
    )
    
    
    args = parser.parse_args()    
    return args
    
def resize_image_to_visualization(tensors,size_tensor):
    assert tensors.shape[0]==size_tensor.shape[0]
    batch_size = tensors.shape[0]
    size_tensor = size_tensor.cpu().numpy()
    
    recovered_images_list = []
    for idx in range(batch_size):
        recovered_H,recovered_W = size_tensor[idx]
        instance_tensor = tensors[idx:idx+1,:,:,:]
        
        instance_tensor = F.interpolate(instance_tensor,size=[recovered_H,recovered_W],
                                        mode='bilinear',align_corners=False)
        instance_tensor = instance_tensor.squeeze(0)
        instance_tensor = instance_tensor.permute(1,2,0).cpu().numpy()
        recovered_images_list.append(instance_tensor)    
        
    return recovered_images_list
    
def saved_to_assigned_folders(
                              args,
                              name_list,
                              rendered_left_from_right,
                              rendered_right_from_left,
            
                              ):
    assert len(name_list) == len(rendered_left_from_right)
    for idx in range(len(name_list)):
        # left image
        saved_name = name_list[idx]
        left_from_right = rendered_left_from_right[idx]
        left_from_right = (left_from_right*255.).astype(np.uint8)
        
        right_from_left = rendered_right_from_left[idx]
        right_from_left = (right_from_left*255.).astype(np.uint8)
        
        
        basename = os.path.basename(saved_name)
        saved_folder = saved_name[:-len(basename)]
        saved_folder = os.path.join(args.saved_folder,saved_folder)
        saved_folder_right = saved_folder.replace("image_02","image_03")
        os.makedirs(saved_folder,exist_ok=True)
        os.makedirs(saved_folder_right,exist_ok=True)
        
        skimage.io.imsave(os.path.join(saved_folder,"left_from_right_"+basename),left_from_right)
        skimage.io.imsave(os.path.join(saved_folder_right,"right_from_left_"+basename),right_from_left)       
        

def Rendered_Existing_Vals(args):
    device = 'cuda'
    # Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                    subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_unet_path,subfolder="unet",
                                                in_channels=8+1, sample_size=96,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    print("Loaded All the Pre-Trained Models........")


    # get the dataset
    (train_loader,test_loader),num_batches_per_epoch = prepare_dataset(datapath=args.datapath,
                    trainlist=args.trainlist,
                    vallist=args.vallist,
                    batch_size=args.batch_size,
                    test_size=args.test_size,
                    datathread=args.datathread)
    # loaded the pretrained model.
    pipeline = SD20UNet_Validation_Pipeline(unet=unet,vae=vae,
                                 scheduler=noise_scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer)
    pipeline = pipeline.to(device)
    print("Loaded the Pipeline from the Pre-trained Model Parts...........")
    
    
    denosing_steps=32
    processing_res=512
    match_input_res = True

    
    for sample in tqdm(test_loader):
        with torch.no_grad():
            rendered_left_from_right,rendered_right_from_left = pipeline(
                                                                left_images_tensor=sample['left_image'],
                                                                right_images_tensor= sample['right_image'],
                                                                denosing_steps = denosing_steps,
                                                                processing_res = processing_res,
                                                                match_input_res = match_input_res)
            
            original_sizes = sample['original_size']
            
            rendered_left_from_right = resize_image_to_visualization(rendered_left_from_right,original_sizes)
            rendered_right_from_left = resize_image_to_visualization(rendered_right_from_left,original_sizes)

            if args.save_results:
                saved_to_assigned_folders(
                                          args,
                                          sample['left_name'],
                                          rendered_left_from_right=rendered_left_from_right,
                                          rendered_right_from_left = rendered_right_from_left
                                          )
            
            
            

            

            

            
            

            
            
            
            
            
            



            




if __name__=="__main__":
    
    args = parse_args()
    Rendered_Existing_Vals(args=args)

