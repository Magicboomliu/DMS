import torch
import diffusers

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    DDIMInverseScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel
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
from pipeline.inference.SD20_UNet_Simple_Pipeline import SD20UNet_Simple_Finetune_Pipeline
import argparse



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


    # parser.add_argument(
    #     "--fname_list",
    #     type=str,
    #     default="/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt",
    #     help="Filename Path",
    # )
    # parser.add_argument(
    #     "--datapath",
    #     type=str,
    #     default="/media/zliu/data12/dataset/KITTI/KITTI_Raw/",
    #     help="datapath",
    # )
    args = parser.parse_args()
    os.makedirs(args.output_folder_path,exist_ok=True)
    
    return args
    
    
def Run_Inference_SD20_Unet(args):
    
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
    # read the pipelines.
    pipeline = SD20UNet_Simple_Finetune_Pipeline(unet=unet,vae=vae,
                                 scheduler=noise_scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer)
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
    
    
    #rendered the left image
    rendered_left_from_left, rendered_left_left_from_left,rendered_right_from_left = pipeline(
                input_image=left_image_pil,
                denosing_steps = denosing_steps,
                processing_res = processing_res,)
    
    # rendered the right image
    rendered_right_from_right, rendered_left_from_right,rendered_right_right_from_right = pipeline(
                input_image=left_image_pil,
                denosing_steps = denosing_steps,
                processing_res = processing_res)
    
    

    rendered_left_from_left = rendered_left_from_left * 255.
    rendered_left_from_left = rendered_left_from_left.astype(np.uint8)
    
    rendered_left_left_from_left = rendered_left_left_from_left * 255.
    rendered_left_left_from_left = rendered_left_left_from_left.astype(np.uint8)
    
    rendered_right_from_left = rendered_right_from_left * 255.
    rendered_right_from_left = rendered_right_from_left.astype(np.uint8)
    
    rendered_right_from_right = rendered_right_from_right * 255.
    rendered_right_from_right = rendered_right_from_right.astype(np.uint8)
    
    rendered_left_from_right = rendered_left_from_right * 255.
    rendered_left_from_right = rendered_left_from_right.astype(np.uint8)
    
    rendered_right_right_from_right = rendered_right_right_from_right * 255.
    rendered_right_right_from_right = rendered_right_right_from_right.astype(np.uint8)
    

    left_image_gt = np.array(Image.open(left_image_path).convert("RGB"))
    right_image_gt = np.array(Image.open(right_image_path).convert("RGB"))

    skimage.io.imsave(os.path.join(args.output_folder_path,"left(gt)_{}".format(basename)),
                      left_image_gt)
    skimage.io.imsave(os.path.join(args.output_folder_path,"right(gt)_{}".format(basename)),
                      right_image_gt)  
    
    skimage.io.imsave(os.path.join(args.output_folder_path,"left(left)_{}".format(basename)),
                      rendered_left_from_left)
    skimage.io.imsave(os.path.join(args.output_folder_path,"left_left(left)_{}".format(basename)),
                      rendered_left_left_from_left)
    skimage.io.imsave(os.path.join(args.output_folder_path,"right(left)_{}".format(basename)),
                      rendered_right_from_left)

    skimage.io.imsave(os.path.join(args.output_folder_path,"left(right)_{}".format(basename)),
                      rendered_left_from_right)
    skimage.io.imsave(os.path.join(args.output_folder_path,"right_right(right)_{}".format(basename)),
                      rendered_right_right_from_right)
    skimage.io.imsave(os.path.join(args.output_folder_path,"right(right)_{}".format(basename)),
                      rendered_right_from_right)
    
    

if __name__=="__main__":
    args = parse_args()
    Run_Inference_SD20_Unet(args=args)