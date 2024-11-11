import argparse
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import os
import logging
import tqdm
from accelerate import Accelerator
import transformers
import datasets
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

import diffusers
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

import sys
sys.path.append("../..") 
from trainers.CARLA.dataset_configuration import prepare_dataset
from PIL import Image
check_min_version("0.26.0.dev0")
import skimage.io
logger = get_logger(__name__, log_level="INFO")
import  matplotlib.pyplot as plt
from pipeline.CARLA_pipeline.inference.simple_unet_pipeline import SimpleUNet_Pipeline
import argparse

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

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


def Inference(args):
    
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
    
    # Load the Pipeline
    pipeline = SimpleUNet_Pipeline(unet=unet,
                                    vae=vae,
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
        right_fname = left_fname.replace("Left_RGB","Right_RGB")
        med_fname = left_fname.replace("Left_RGB","Middle_RGB")
        
        left_image_path = os.path.join(args.datapath,left_fname)
        right_image_path = os.path.join(args.datapath,right_fname)
        med_image_path = os.path.join(args.datapath, med_fname)
        
        assert os.path.exists(left_image_path)
        assert os.path.exists(right_image_path)
        assert os.path.exists(med_image_path)
        

        rendered_left_image_path = os.path.join(args.output_folder_path,left_fname)
        rendered_right_image_path = os.path.join(args.output_folder_path,right_fname)


        basename = os.path.basename(fname)
        rendered_left_image_folder = rendered_left_image_path[:-len(basename)]
        rendered_right_image_folder = rendered_right_image_path[:-len(basename)]
        os.makedirs(rendered_left_image_folder,exist_ok=True)
        os.makedirs(rendered_right_image_folder,exist_ok=True)

        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        right_image_pil = Image.open(right_image_path)
        right_image_pil = right_image_pil.convert("RGB")
        right_image_np = np.array(right_image_pil)
        left_image_np = np.array(left_image_pil)
        
        # render right using the left image
        rendered_right = pipeline(left_image_pil,
                denosing_steps=denosing_steps,
                ensemble_size= 1,
                processing_res = processing_res,
                match_input_res = match_input_res,
                batch_size = batch_size,
                show_progress_bar = True,
                text_embed="to right")

        rendered_right = rendered_right  * 255
        rendered_right = rendered_right.astype(np.uint8)
        
    
        # render left using the right image
        rendered_left = pipeline(right_image_pil,
                denosing_steps=denosing_steps,
                ensemble_size= 1,
                processing_res = processing_res,
                match_input_res = match_input_res,
                batch_size = batch_size,
                show_progress_bar = True,
                text_embed="to left")

        rendered_left = rendered_left  * 255
        rendered_left = rendered_left.astype(np.uint8)
        
        
        skimage.io.imsave(rendered_right_image_path,rendered_right)
        skimage.io.imsave(rendered_left_image_path,rendered_left)
        
        
        
        
        





if __name__=="__main__":
    
    args = parse_args()
    Inference(args=args)