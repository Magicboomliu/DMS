import torch
import diffusers

from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel,
)

from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append("../../..")

from tqdm import tqdm
import os
import skimage.io
import numpy as np
from PIL import Image
from pipeline.inference.SD20_Simple_ControlNet import SimpleControlNet_Pipeline
import argparse
import matplotlib.pyplot as plt
from dataloader.utils import read_text_lines



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
        "--pretrained_controlnet",
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
        "--filelist",
        type=str,
        default="",
        help="Example Image Path",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="",
        help="Example Image Path",
    )

    args = parser.parse_args()
    os.makedirs(args.output_folder_path,exist_ok=True)    
    return args



def Get_New_Views(args):
    
    device = 'cuda'
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_unet_path,subfolder="unet",
                                                    in_channels=8, sample_size=96)
    
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet,
                                                         subfolder='controlnet')
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    print("Loaded All the Pre-Trained Models........")
    
    # read the pipelines.
    pipeline = SimpleControlNet_Pipeline(unet=unet,vae=vae,
                                 scheduler=noise_scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer,
                                 controlnet=controlnet
                                 )
    pipeline = pipeline.to(device)
    print("Loaded the Pipeline from the Pre-trained Model Parts...........")

    denosing_steps=32
    processing_res=768
    match_input_res = True
    batch_size = 1
    
    
    lines = read_text_lines(args.filelist)
    
    for fname in tqdm(lines):
        
        left_image_path = os.path.join(args.datapath,fname)
        right_image_path = left_image_path.replace("image_02","image_03")
        assert os.path.exists(left_image_path)
        assert os.path.exists(right_image_path)
        
        basename = os.path.basename(left_image_path)
        rendered_left_from_right_path = left_image_path.replace(args.datapath,args.output_folder_path)
        rendered_left_from_right_path = rendered_left_from_right_path.replace(basename,"left_from_right_"+basename)
        
        rendered_right_from_left_path = right_image_path.replace(args.datapath,args.output_folder_path)
        rendered_right_from_left_path = rendered_right_from_left_path.replace(basename,"right_from_left_"+basename)
        
        
        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        
        right_image_pil = Image.open(right_image_path)
        right_image_pil = right_image_pil.convert("RGB")
        
        
        rendered_left_from_right = pipeline(right_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to left",
                                  cond=1)
        
        rendered_right_from_left = pipeline(left_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to right",
                                  cond=1)
        
        rendered_right_from_left = rendered_right_from_left * 255
        rendered_right_from_left = rendered_right_from_left.astype(np.uint8)
        
        
        rendered_left_from_right = rendered_left_from_right * 255
        rendered_left_from_right = rendered_left_from_right.astype(np.uint8)
        
        
        rendered_left_from_right_path_sub_folder = rendered_left_from_right_path[:-len(os.path.basename(rendered_left_from_right_path))]
        os.makedirs(rendered_left_from_right_path_sub_folder,exist_ok=True)
        rendered_right_from_left_path_sub_folder = rendered_right_from_left_path[:-len(os.path.basename(rendered_right_from_left_path))]
        os.makedirs(rendered_right_from_left_path_sub_folder,exist_ok=True)
        
        
        
        skimage.io.imsave(rendered_left_from_right_path,rendered_left_from_right)
        skimage.io.imsave(rendered_right_from_left_path,rendered_right_from_left)
    



if __name__=="__main__":
    
    args = parse_args()
    
    Get_New_Views(args=args)
    
    pass