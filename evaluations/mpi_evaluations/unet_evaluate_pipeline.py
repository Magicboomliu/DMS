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
sys.path.append("..")

from tqdm import tqdm
import os
import skimage.io
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from pipeline.MPI_pipeline.inference.Simple_UNet import SimpleUNet_Pipeline
from tqdm import tqdm

from dataloader.mpi_dataloader.utils import read_text_lines
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def parse_args():
    parser = argparse.ArgumentParser(description="MPI Inference")
    
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
    
    denosing_steps=50
    match_input_res = True

    # read filenames
    fname_list = read_text_lines(filepath=args.input_fname_list)
    

    for fname in tqdm(fname_list):
        
        left_image_path = os.path.join(args.datapath,os.path.join("final_left",fname))
        right_image_path = os.path.join(args.datapath,os.path.join("final_right",fname))
        assert os.path.exists(left_image_path)
        assert os.path.exists(right_image_path)
        
        saved_name_rendered_left = os.path.join("rendered_left",fname)
        saved_name_renderd_right = os.path.join("rendered_right",fname)

        saved_name_rendered_left_left = os.path.join("rendered_left_left",fname)
        saved_name_renderd_right_right = os.path.join("rendered_right_right",fname)

        
        saved_name_renderd_left_abs = os.path.join(args.datapath,saved_name_rendered_left)
        saved_name_renderd_left_abs_folder = saved_name_renderd_left_abs[:-len(os.path.basename(saved_name_renderd_left_abs))]
        os.makedirs(saved_name_renderd_left_abs_folder,exist_ok=True)
        
        saved_name_rendered_right_abs = os.path.join(args.datapath,saved_name_renderd_right)
        saved_name_renderd_right_abs_folder = saved_name_rendered_right_abs[:-len(os.path.basename(saved_name_rendered_right_abs))]
        os.makedirs(saved_name_renderd_right_abs_folder,exist_ok=True)
        
        
        saved_name_renderd_left_left_abs = os.path.join(args.datapath,saved_name_rendered_left_left)
        saved_name_renderd_left_left_abs_folder = saved_name_renderd_left_left_abs[:-len(os.path.basename(saved_name_renderd_left_left_abs))]
        os.makedirs(saved_name_renderd_left_left_abs_folder,exist_ok=True)
        

        saved_name_renderd_right_right_abs = os.path.join(args.datapath,saved_name_renderd_right_right)
        saved_name_renderd_right_right_abs_folder = saved_name_renderd_right_right_abs[:-len(os.path.basename(saved_name_renderd_right_right_abs))]
        os.makedirs(saved_name_renderd_right_right_abs_folder,exist_ok=True)
              
        
        
    
        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        right_image_pil = Image.open(right_image_path)
        right_image_pil = right_image_pil.convert("RGB")
        
        right_image_np = np.array(right_image_pil)
        left_image_np = np.array(left_image_pil)
        
        
        # rendered right
        rendered_right = pipeline(left_image_pil,
             denosing_steps=denosing_steps,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_embed="to right",)
        
        rendered_left_left = pipeline(left_image_pil,
             denosing_steps=denosing_steps,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_embed="to left",)


        rendered_right_right = pipeline(right_image_pil,
             denosing_steps=denosing_steps,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_embed="to right",)

        rendered_left = pipeline(right_image_pil,
             denosing_steps=denosing_steps,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_embed="to left",)
        
    
        rendered_right = rendered_right  * 255
        rendered_right = rendered_right.astype(np.uint8)        

        rendered_left = rendered_left  * 255
        rendered_left = rendered_left.astype(np.uint8)
        
        # psnr_left = compare_psnr(rendered_left,left_image_np)
        # psnr_right = compare_psnr(rendered_right,right_image_np)
    
        
        rendered_left_left = rendered_left_left * 255
        rendered_left_left = rendered_left_left.astype(np.uint8)
        
        rendered_right_right = rendered_right_right *255
        rendered_right_right = rendered_right_right.astype(np.uint8)
        
        
        skimage.io.imsave(saved_name_rendered_right_abs,rendered_right)
        skimage.io.imsave(saved_name_renderd_left_abs,rendered_left)
        
        
        skimage.io.imsave(saved_name_renderd_right_right_abs,rendered_right_right)
        skimage.io.imsave(saved_name_renderd_left_left_abs,rendered_left_left)




if __name__=="__main__":
    
    args = parse_args()
    
    main(args=args)