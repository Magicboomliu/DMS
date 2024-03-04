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


    args = parser.parse_args()    
    return args
    

def Rendered_Existing_Vals(args):
    # get the dataset
    (train_loader,test_loader),num_batches_per_epoch = prepare_dataset(datapath=args.datapath,
                    trainlist=args.trainlist,
                    vallist=args.vallist,
                    batch_size=args.batch_size,
                    test_size=args.test_size,
                    datathread=args.datathread)
    
    for idx, sample in enumerate(test_loader):
        print(sample['left_image'].shape) # 0~1
        print(sample['right_image'].shape) # 0~1
        print(sample['original_size'].shape) 

    
    
    
    pass    


if __name__=="__main__":
    
    args = parse_args()
    Rendered_Existing_Vals(args=args)

