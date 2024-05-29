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

from pipeline.Kitti_pipeline.inference.Simple_UNet_UpscaleX import SimpleUNet_Pipeline_UpScaleX
from tqdm import tqdm
from dataloader.kitti_dataloader.utils import read_text_lines
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

    pipeline = SimpleUNet_Pipeline_UpScaleX(unet=unet,vae=vae,
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

        splits = fname.split()
        left_fname = splits[0]
        
        left_image_path = os.path.join(args.datapath,left_fname)
        assert os.path.exists(left_image_path)

        rendered_center_image_path = os.path.join(args.output_folder_path,left_fname)
        rendered_center_image_path = rendered_center_image_path.replace("image_2","image_25")
        basename = os.path.basename(left_fname)
        rendered_center_image_path_folder = rendered_center_image_path[:-len(basename)]
        os.makedirs(rendered_center_image_path_folder,exist_ok=True)
        
        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")

            
        # render right using the left image
        rendered_center_from_left = pipeline(left_image_pil,
                denosing_steps=denosing_steps,
                ensemble_size= 1,
                processing_res = processing_res,
                match_input_res = match_input_res,
                batch_size = batch_size,
                show_progress_bar = True,
                text_embed="to right",
                upscale_factor=2.0
                )

        rendered_center_from_left = rendered_center_from_left  * 255
        rendered_center_from_left = rendered_center_from_left.astype(np.uint8)
        skimage.io.imsave(rendered_center_image_path,rendered_center_from_left)
        



if __name__=="__main__":
    
    args = parse_args()
    main(args=args)