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
        "--example_image_path",
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


def Run_Simple_ControlNet(args):
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

    left_image_path = args.example_image_path
    basename = os.path.basename(left_image_path)
    right_image_path = left_image_path.replace("left_images","right_images")
    
    left_image_pil = Image.open(left_image_path)
    left_image_pil = left_image_pil.convert("RGB")
    
    right_image_pil = Image.open(right_image_path)
    right_image_pil = right_image_pil.convert("RGB")
    
    
    
    with torch.no_grad():
        rendered_right_from_left = pipeline(left_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to right",
                                  cond=1)
        rendered_left_left_from_left = pipeline(left_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to left",
                                  cond=1)
        rendered_left_from_right = pipeline(right_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to left",
                                  cond=1)
        rendered_right_right_from_right = pipeline(right_image_pil,
                                  denosing_steps=denosing_steps,
                                  processing_res = processing_res,
                                  match_input_res = match_input_res,
                                  batch_size = batch_size,
                                  show_progress_bar = True,
                                  text_embed="to right",
                                  cond=1)
        
        rendered_right_from_left = rendered_right_from_left * 255
        rendered_right_from_left = rendered_right_from_left.astype(np.uint8)
        
        rendered_left_left_from_left = rendered_left_left_from_left * 255
        rendered_left_left_from_left =rendered_left_left_from_left.astype(np.uint8)
        
        rendered_left_from_right = rendered_left_from_right * 255
        rendered_left_from_right = rendered_left_from_right.astype(np.uint8)
        
        rendered_right_right_from_right =rendered_right_right_from_right * 255
        rendered_right_right_from_right  = rendered_right_right_from_right.astype(np.uint8)
        
        
        skimage.io.imsave(os.path.join(args.output_folder_path,"left_left(left).png"),rendered_left_left_from_left)
        skimage.io.imsave(os.path.join(args.output_folder_path,"left(right).png"),rendered_left_from_right)
        skimage.io.imsave(os.path.join(args.output_folder_path,"right(left).png"),rendered_right_from_left)
        skimage.io.imsave(os.path.join(args.output_folder_path,"right_right(right).png"),rendered_right_right_from_right)
        skimage.io.imsave(os.path.join(args.output_folder_path,"left_gt.png"),np.array(left_image_pil))
        skimage.io.imsave(os.path.join(args.output_folder_path,"right_gt.png"),np.array(right_image_pil))

        
        # plt.figure(figsize=(10,8))
        # plt.subplot(3,2,1)
        # plt.axis('off')
        # plt.title('left gt')
        # plt.imshow(left_image_pil)
        # plt.subplot(3,2,2)
        # plt.axis('off')
        # plt.title('right gt')
        # plt.imshow(right_image_pil)
        # plt.subplot(3,2,3)
        # plt.axis('off')
        # plt.title('left_left(left)')
        # plt.imshow(rendered_left_left_from_left)
        # plt.subplot(3,2,4)
        # plt.axis('off')
        # plt.title('left(right)')
        # plt.imshow(rendered_left_from_right)
        # plt.subplot(3,2,5)
        # plt.axis('off')
        # plt.title('right(left)')
        # plt.imshow(rendered_right_from_left)
        # plt.subplot(3,2,6)
        # plt.axis('off')
        # plt.title('right_right(right)')
        # plt.imshow(rendered_right_right_from_right)
        # plt.show()
        
        



if __name__=="__main__":
    args = parse_args()
    Run_Simple_ControlNet(args)
    