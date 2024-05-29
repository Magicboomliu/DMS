import torch
import diffusers

from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append("../..")

from tqdm import tqdm
import os
import skimage.io
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pipeline.Kitti_pipeline.inference.Simple_UNet import SimpleUNet_Pipeline
from dataloader.kitti_dataloader.utils import read_text_lines
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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
        default="",
        help="Path to loaded the pretrained unet.",)

    parser.add_argument(
        "--datapath",
        type=str,
        default="",
        help="Data Path")


    parser.add_argument(
        "--input_fname_list",
        type=str,
        default="",
        help="Example Image Path")
    

    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="",
        help="Example Image Path",)

    args = parser.parse_args()
    os.makedirs(args.output_folder_path,exist_ok=True)    
    return args


def main(args=None):
    # loaded the models
    device = 'cuda'
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_unet_path,subfolder="unet",
                                                    in_channels=8, sample_size=96)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    print("Loaded All the Pre-Trained Models........")

    pipeline = SimpleUNet_Pipeline(unet=unet,vae=vae,
                                 scheduler=noise_scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer)
    pipeline = pipeline.to(device)
    print("Loaded the Pipeline from the Pre-trained Model Parts...........")
    
    denosing_steps=32
    processing_res=768
    match_input_res = True
    batch_size = 1
    
    
    # read filenames
    fname_list = read_text_lines(filepath=args.input_fname_list)
    
    for fname in tqdm(fname_list):
            
        left_fname = fname
        right_fname = left_fname.replace("image_02","image_03")
        
        left_image_path = os.path.join(args.datapath,left_fname)
        right_image_path = os.path.join(args.datapath,right_fname)
        
        rendered_left_image_path = os.path.join(args.output_folder_path,left_fname)
        rendered_left_left_image_path = rendered_left_image_path.replace("image_02","image_01")
        rendered_right_right_image_path = rendered_left_left_image_path.replace("image_01","image_04")

        
        basename = os.path.basename(left_fname)
        rendered_left_left_image_folder = rendered_left_left_image_path[:-len(basename)]
        rendered_right_right_image_folder = rendered_right_right_image_path[:-len(basename)]
        
        os.makedirs(rendered_left_left_image_folder,exist_ok=True)
        os.makedirs(rendered_right_right_image_folder,exist_ok=True)
        
        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        right_image_pil = Image.open(right_image_path)
        right_image_pil = right_image_pil.convert("RGB")
                
        
        if os.path.exists(rendered_left_left_image_path):
            continue
        else:
            # render left using the right image
            rendered_left_left = pipeline(left_image_pil,
                    denosing_steps=denosing_steps,
                    ensemble_size= 1,
                    processing_res = processing_res,
                    match_input_res = match_input_res,
                    batch_size = batch_size,
                    show_progress_bar = True,
                    text_embed="to left")

            rendered_left_left = rendered_left_left  * 255
            rendered_left_left = rendered_left_left.astype(np.uint8)
            skimage.io.imsave(rendered_left_left_image_path,rendered_left_left)
            
        if os.path.exists(rendered_right_right_image_path):
            continue
        else:        
            # render right using the left image
            rendered_right_right = pipeline(right_image_pil,
                    denosing_steps=denosing_steps,
                    ensemble_size= 1,
                    processing_res = processing_res,
                    match_input_res = match_input_res,
                    batch_size = batch_size,
                    show_progress_bar = True,
                    text_embed="to right")

            rendered_right_right = rendered_right_right  * 255
            rendered_right_right = rendered_right_right.astype(np.uint8)
            skimage.io.imsave(rendered_right_right_image_path,rendered_right_right)
        
        
        

if __name__=="__main__":
    
    args = parse_args()
    
    main(args=args)