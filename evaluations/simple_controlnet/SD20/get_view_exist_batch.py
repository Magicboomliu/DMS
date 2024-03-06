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

import argparse
import matplotlib.pyplot as plt
from dataloader.utils import read_text_lines

from evaluations.simple_controlnet.SD20.evaluation_pipeline_exist import SimpleControlNet_Pipeline




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
        "--batch_size",
        type=int,
        default=4,
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
    
    idx = 0
    left_images_list = []
    right_images_list = []
    
    rendered_left_saved_name_list = []
    rendered_right_saved_name_list = []
    
    
    
    last_no_batch = len(lines)%args.batch_size
    begin_id = len(lines) - last_no_batch +1

    for fname in tqdm(lines):
        
        idx = idx +1
        left_image_path = os.path.join(args.datapath,fname)
        right_image_path = left_image_path.replace("image_02","image_03")
        assert os.path.exists(left_image_path)
        assert os.path.exists(right_image_path)
        
        basename = os.path.basename(left_image_path)
        rendered_left_from_right_path = left_image_path.replace(args.datapath,args.output_folder_path)
        rendered_left_from_right_path = rendered_left_from_right_path.replace(basename,"left_from_right_"+basename)
        
        rendered_right_from_left_path = right_image_path.replace(args.datapath,args.output_folder_path)
        rendered_right_from_left_path = rendered_right_from_left_path.replace(basename,"right_from_left_"+basename)

        rendered_left_from_right_path_sub_folder = rendered_left_from_right_path[:-len(os.path.basename(rendered_left_from_right_path))]
        os.makedirs(rendered_left_from_right_path_sub_folder,exist_ok=True)
        rendered_right_from_left_path_sub_folder = rendered_right_from_left_path[:-len(os.path.basename(rendered_right_from_left_path))]
        os.makedirs(rendered_right_from_left_path_sub_folder,exist_ok=True)
        
        
        left_image_pil = Image.open(left_image_path)
        left_image_pil = left_image_pil.convert("RGB")
        
        right_image_pil = Image.open(right_image_path)
        right_image_pil = right_image_pil.convert("RGB")
        
        
        left_images_list.append(left_image_pil)
        right_images_list.append(right_image_pil)
        
        rendered_left_saved_name_list.append(rendered_left_from_right_path)
        rendered_right_saved_name_list.append(rendered_right_from_left_path)


   
        if idx<begin_id:
            if idx%4==0:  
                # process here
                pipeline(input_left_images_list=left_images_list,
                         input_right_images_list=right_images_list,
                         denosing_steps=denosing_steps,
                         processing_res=processing_res,
                         match_input_res=True)
                
                # Final set to empty
                left_images_list = []
                right_images_list =[]
                rendered_left_saved_name_list=[]
                rendered_left_saved_name_list=[]
   
        if idx>=begin_id:
            if idx == begin_id:
                assert len(left_images_list)==1
            if idx == len(lines):
                pipeline(input_left_images_list=left_images_list,
                         input_right_images_list=right_images_list,
                         denosing_steps=denosing_steps,
                         processing_res=processing_res,
                         match_input_res=True)
                
                

        
        
        # rendered_left_from_right = pipeline(right_image_pil,
        #                           denosing_steps=denosing_steps,
        #                           processing_res = processing_res,
        #                           match_input_res = match_input_res,
        #                           batch_size = batch_size,
        #                           show_progress_bar = True,
        #                           text_embed="to left",
        #                           cond=1)
        
        # rendered_right_from_left = pipeline(left_image_pil,
        #                           denosing_steps=denosing_steps,
        #                           processing_res = processing_res,
        #                           match_input_res = match_input_res,
        #                           batch_size = batch_size,
        #                           show_progress_bar = True,
        #                           text_embed="to right",
        #                           cond=1)
        
        # rendered_right_from_left = rendered_right_from_left * 255
        # rendered_right_from_left = rendered_right_from_left.astype(np.uint8)
        
        
        # rendered_left_from_right = rendered_left_from_right * 255
        # rendered_left_from_right = rendered_left_from_right.astype(np.uint8)
        
        

        
        
        
        # skimage.io.imsave(rendered_left_from_right_path,rendered_left_from_right)
        # skimage.io.imsave(rendered_right_from_left_path,rendered_right_from_left)
    



if __name__=="__main__":
    
    args = parse_args()
    
    Get_New_Views(args=args)
    
    pass