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

from pipeline.MPI_pipeline.inference.Simple_UNet_UpscaleX import SimpleUNet_Pipeline_UpscaleX
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

    pipeline = SimpleUNet_Pipeline_UpscaleX(unet=unet,vae=vae,
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


        left_image_path = os.path.join(args.datapath,os.path.join("final_right",fname))
        assert os.path.exists(left_image_path)        
        saved_name_rendered_med = os.path.join("rendered_med_right",fname)
        saved_name_renderd_med_abs = os.path.join(args.datapath,saved_name_rendered_med)
        saved_name_renderd_med_abs_folder = saved_name_renderd_med_abs[:-len(os.path.basename(saved_name_renderd_med_abs))]
        os.makedirs(saved_name_renderd_med_abs_folder,exist_ok=True)
        

        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        left_image_np = np.array(left_image_pil)
        
        
        # rendered right
        rendered_med = pipeline(left_image_pil,
             denosing_steps=denosing_steps,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_embed="to left",
             upscale_factor=2)
        

        rendered_med = rendered_med  * 255
        rendered_med = rendered_med.astype(np.uint8)
                

        skimage.io.imsave(saved_name_renderd_med_abs,rendered_med)

        

if __name__=="__main__":
    
    args = parse_args()
    main(args=args)