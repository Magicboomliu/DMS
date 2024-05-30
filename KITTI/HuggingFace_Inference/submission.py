import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
from models.PAMStereo.PASMnet import PASMnet
from models.Stereonet.stereonet import StereoNet

from models.CFNet.cfnet import CFNet

from HuggingFace_Inference.colormap import kitti_colormap
import argparse


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import json
import skimage.io
import os

from utils.kitti_io import read_img, read_disp,read_kitti_step1,read_kitti_step2,read_kitti_image_step1,read_kitti_image_step2
from skimage import io, transform
import numpy as np
from PIL import Image
from utils import utils

from tqdm import tqdm
from utils.AverageMeter import AverageMeter
from utils.metric import P1_metric,P1_Value,D1_metric_Occ,Disparity_EPE_Loss_OCC
import matplotlib.pyplot as plt
import skimage.io

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm

def Image_Normalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_denorm = (image_tensor -image_mean)/image_std
    return image_denorm

def To_Tensor(image_np):
    left = np.transpose(image_np, (2, 0, 1))  # [3, H, W]
    return torch.from_numpy(left) / 255.

def To_Tensor_Disp(image_np):
    disp = image_np  # [H, W]
    return torch.from_numpy(disp)

def image_pad(image,targetHW):
    H,W = image.shape[:2]
    new_H, new_W = targetHW
    pad_bottom = new_H - H  
    pad_right = new_W - W 
    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
    return added_image, np.array(image.shape[:2])

def disp_pad(image,targetHW):
    H,W = image.shape[:2]
    new_H, new_W = targetHW
    pad_bottom = new_H - H  
    pad_right = new_W - W 
    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right)), 'constant', constant_values=0)
    return added_image, np.array(image.shape[:2])

def parse_args():
    parser = argparse.ArgumentParser(description="PAMStereo Evaluation")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/KITTIStereo/kitti_2015/",
        required=True,
        help="Specify the dataset name used for training/validation.",)

    parser.add_argument(
        "--vallist",
        type=str,
        default="/data1/KITTI/KITTI_Raw",
        required=True,
        help="Specify the dataset name used for training/validation.",)

    parser.add_argument(
        "--network_type",
        type=str,
        default="/data1/KITTI/KITTI_Raw",
        required=True,
        help="Specify the dataset name used for training/validation.",)

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/data1/KITTI/KITTI_Raw",
        required=True,
        help="Specify the dataset name used for training/validation.",)

    parser.add_argument(
        "--output_path",
        type=str,
        default="/data1/KITTI/KITTI_Raw",
        required=True,
        help="Specify the dataset name used for training/validation.",)

    parser.add_argument(
        "--vis",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    
    # get the local rank
    args = parser.parse_args()
    
    return args


def read_occ_or_oof(data):
    mask = read_img(data)/255.
    mask = mask[:,:,0]
    mask = mask.astype(np.int8)
    
    return mask

import cv2
from HuggingFace_Inference.error_map_vis import disp_error_img

if __name__=="__main__":
    
    args = parse_args()
    os.makedirs(args.output_path,exist_ok=True)
    if args.network_type=="PAMStereo":
        stereo_matching_network = PASMnet()
    elif args.network_type=='StereoNet':
        stereo_matching_network = StereoNet()
    elif args.network_type=="CFNet":
        stereo_matching_network = CFNet(d=192)
    
    ckpts = torch.load(args.pretrained_model_path)
    model_ckpt = ckpts['model_state']
    stereo_matching_network.load_state_dict(model_ckpt)
    
    print("loaded the pretrained model")
    stereo_matching_network.cuda()
    stereo_matching_network.eval()
    lines = utils.read_text_lines(args.vallist)

    if args.vis:
        saved_est_left_folder = os.path.join(args.output_path,"vis/est_disp")
        os.makedirs(saved_est_left_folder,exist_ok=True)

    
    for line in tqdm(lines):
        
        splits = line.split()
        left = splits[0]
        right = splits[1]
        # disp = splits[2]
        
        left_abs = os.path.join(args.datapath,left)
        right_abs = os.path.join(args.datapath,right)
        
        assert os.path.exists(left_abs)
        assert os.path.exists(right_abs)

        left_data = read_img(left_abs)
        right_data = read_img(right_abs)
        
        left_data_tensor = To_Tensor(left_data)
        right_data_tensor = To_Tensor(right_data)
        left_data_tensor_norm = Image_Normalization(left_data_tensor) 
        right_data_tensor_norm = Image_Normalization(right_data_tensor)
        left_data_tensor_norm = left_data_tensor_norm.cuda()
        right_data_tensor_norm = right_data_tensor_norm.cuda()
        b,c,h,w = left_data_tensor_norm.shape
        w_pad = 1248 - w
        h_pad = 384 -h
        
        img_nums = 0
        pad = (w_pad,0,h_pad,0)
        left_input_pad = F.pad(left_data_tensor_norm,pad=pad)
        right_input_pad = F.pad(right_data_tensor_norm,pad=pad)

        with torch.no_grad():
            
            if args.network_type=="PAMStereo":
                output = stereo_matching_network(left_input_pad,right_input_pad)
                output = output[:,:,h_pad:,w_pad:]
            elif args.network_type=="StereoNet":
                output = stereo_matching_network(left_input_pad,right_input_pad)['disp']
                output = output[:,:,h_pad:,w_pad:]
            elif args.network_type=="CFNet":
                outputs, _,_ = stereo_matching_network(left_input_pad,right_input_pad)
                output = outputs[0]
                output = output.unsqueeze(1)
                output = output[:,:,h_pad:,w_pad:]

            
            saved_disparity_name = os.path.join(saved_est_left_folder,os.path.basename(left))
            saved_disparity_data = output.squeeze(0).squeeze(0).cpu().numpy()
            skimage.io.imsave(saved_disparity_name, (saved_disparity_data * 256).astype('uint16'))

