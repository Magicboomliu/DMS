import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
from models.PAMStereo.PASMnet import PASMnet
from HuggingFace_Inference.colormap import kitti_colormap
import argparse
# IMAGENET NORMALIZATION

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
    
    ckpts = torch.load(args.pretrained_model_path)
    model_ckpt = ckpts['model_state']
    stereo_matching_network.load_state_dict(model_ckpt)
    
    print("loaded the pretrained model")
    
    stereo_matching_network.cuda()
    stereo_matching_network.eval()
    
    lines = utils.read_text_lines(args.vallist)


    EPE_OP = Disparity_EPE_Loss_OCC
    P1_ERROR_OP = P1_metric
    D1_ERROR_OP = D1_metric_Occ

    flow2_EPEs = AverageMeter()
    flow2_EPEs_Noc_Meter = AverageMeter()
    flow2_EPEs_Ooc_Meter = AverageMeter()
    flow2_EPEs_Oof_Meter = AverageMeter()
    
    P1_errors = AverageMeter()
    
    
    D1_errors = AverageMeter()
    D1_errors_Noc_Meter = AverageMeter()
    D1_errors_Ooc_Meter = AverageMeter()
    D1_errors_Oof_Meter = AverageMeter()
    
    
    if args.vis:
        saved_left_folder = os.path.join(args.output_path,"vis/left_images")
        saved_right_folder = os.path.join(args.output_path,"vis/right_images")
        saved_disp_folder = os.path.join(args.output_path,"vis/gt_disp")
        saved_occ_folder = os.path.join(args.output_path,"vis/occ_mask")
        saved_oof_folder = os.path.join(args.output_path,"vis/oof_mask")
        saved_est_left_folder = os.path.join(args.output_path,"vis/est_disp")
        error_map_folder = os.path.join(args.output_path,"vis/error_map")
        
        os.makedirs(saved_left_folder,exist_ok=True)
        os.makedirs(saved_right_folder,exist_ok=True)
        os.makedirs(saved_disp_folder,exist_ok=True)
        os.makedirs(saved_occ_folder,exist_ok=True)
        os.makedirs(saved_oof_folder,exist_ok=True)
        os.makedirs(saved_est_left_folder,exist_ok=True)
        os.makedirs(error_map_folder,exist_ok=True)

    
    for line in tqdm(lines):
        
        splits = line.split()
        left = splits[0]
        right = splits[1]
        disp = splits[2]
        
        occlusion_map = left.replace("colored_0","left_occ")
        oof_map = left.replace("colored_0","left_oof")

        left_abs = os.path.join(args.datapath,left)
        right_abs = os.path.join(args.datapath,right)
        disp_abs = os.path.join(args.datapath,disp)
        
        occlusion_abs = os.path.join(args.datapath,occlusion_map)
        oof_abs = os.path.join(args.datapath,oof_map)
        
        
        assert os.path.exists(left_abs)
        assert os.path.exists(right_abs)
        assert os.path.exists(disp_abs)
        assert os.path.exists(occlusion_abs)
        assert os.path.exists(oof_abs)
 
        
        left_data = read_img(left_abs)
        right_data = read_img(right_abs)
        
        gt_disp = read_kitti_step1(disp_abs)
        w, h = gt_disp.size
        dataL = gt_disp
        dataL = read_kitti_step2(dataL)
        gt_disp_data = dataL
        
        gt_occ_mask = read_occ_or_oof(occlusion_abs)
        gt_oof_mask = read_occ_or_oof(oof_abs)
        
        
        if args.vis:
            saved_left_image_name = os.path.join(saved_left_folder,os.path.basename(left))
            skimage.io.imsave(saved_left_image_name,left_data.astype(np.uint8))

            saved_right_image_name = os.path.join(saved_right_folder,os.path.basename(left))
            skimage.io.imsave(saved_right_image_name,right_data.astype(np.uint8))
            
            
            saved_disp_name = os.path.join(saved_disp_folder,os.path.basename(left))
            colored = kitti_colormap(gt_disp_data)
            valid_mask = gt_disp_data >0
            valid_mask = valid_mask.astype(np.float32)
            valid_mask = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)
            colored = colored * valid_mask
            cv2.imwrite(saved_disp_name, colored)
            
            
            saved_occ_name = os.path.join(saved_occ_folder,os.path.basename(left))
            skimage.io.imsave(saved_occ_name,(gt_occ_mask*255).astype(np.uint8))

            saved_oof_name = os.path.join(saved_oof_folder,os.path.basename(left))
            skimage.io.imsave(saved_oof_name,(gt_oof_mask*255).astype(np.uint8))
            
        
        gt_occ_tensor = To_Tensor_Disp(gt_occ_mask)
        gt_occ_tensor = gt_occ_tensor.unsqueeze(0).unsqueeze(0)
        gt_oof_tensor = To_Tensor_Disp(gt_oof_mask)
        gt_oof_tensor = gt_oof_tensor.unsqueeze(0).unsqueeze(0)
        

        left_data_tensor = To_Tensor(left_data)
        right_data_tensor = To_Tensor(right_data)
        left_data_tensor_norm = Image_Normalization(left_data_tensor) 
        right_data_tensor_norm = Image_Normalization(right_data_tensor)
        gt_disp_tensor = To_Tensor_Disp(gt_disp_data)
        gt_disp_tensor = gt_disp_tensor.unsqueeze(0).unsqueeze(0)
        
        
        left_data_tensor_norm = left_data_tensor_norm.cuda()
        right_data_tensor_norm = right_data_tensor_norm.cuda()
        target_disp = gt_disp_tensor.cuda()
        
        gt_occ_tensor = gt_occ_tensor.cuda()
        gt_oof_tensor = gt_oof_tensor.cuda()
        

        b,c,h,w = left_data_tensor_norm.shape
        w_pad = 1248 - w
        h_pad = 384 -h
        
        img_nums = 0
        pad = (w_pad,0,h_pad,0)
        left_input_pad = F.pad(left_data_tensor_norm,pad=pad)
        right_input_pad = F.pad(right_data_tensor_norm,pad=pad)

        with torch.no_grad():
            
            output = stereo_matching_network(left_input_pad,right_input_pad)
            output = output[:,:,h_pad:,w_pad:]
            flow2_EPE,flow2_EPE_Occ,flow2_EPE_Oof,flow2_EPE_Noc = EPE_OP(output, target_disp,occ_mask=gt_occ_tensor,oof_mask=gt_oof_tensor)
            P1_error = P1_ERROR_OP(output, target_disp)
            D1_error, D1_Occ_error, D1_Oof_error, D1_Noc_error= D1_ERROR_OP(output, target_disp,occ_mask=gt_occ_tensor,oof_mask=gt_oof_tensor)
            
        
        if args.vis:
            # estimated disparity
            saved_est_name = os.path.join(saved_est_left_folder,os.path.basename(left))
            output_vis = output.squeeze(0).squeeze(0).cpu().numpy()
            colored = kitti_colormap(output_vis)
            valid_mask = output_vis >0
            valid_mask = valid_mask.astype(np.float32)
            valid_mask = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)
            colored = colored * valid_mask
            cv2.imwrite(saved_est_name, colored)

            # estimated error
            saved_error_map_name = os.path.join(error_map_folder,os.path.basename(left))
            est_error_map = disp_error_img(output.squeeze(1),gt_disp_tensor.squeeze(1))
            valid_mask = gt_disp_tensor>0
            valid_mask = valid_mask.float()
            
            est_error_map = est_error_map * valid_mask
            est_error_map =  est_error_map * 255
            est_error_map = est_error_map.squeeze(0).permute(1,2,0).cpu().numpy()
            skimage.io.imsave(saved_error_map_name,est_error_map.astype(np.uint8))
            
            
        
        
        
        
        if flow2_EPE.data.item() == flow2_EPE.data.item():
            flow2_EPEs.update(flow2_EPE.data.item(), left_data_tensor_norm.size(0))
        if P1_error.data.item() == P1_error.data.item():
            P1_errors.update(P1_error.data.item(), left_data_tensor_norm.size(0))
        if D1_error.data.item() == D1_error.data.item():
            D1_errors.update(D1_error.data.item(), left_data_tensor_norm.size(0))
        
        flow2_EPEs_Noc_Meter.update(flow2_EPE_Noc.data.item(),left_data_tensor_norm.size(0))
        flow2_EPEs_Ooc_Meter.update(flow2_EPE_Occ.data.item(),left_data_tensor_norm.size(0))
        flow2_EPEs_Oof_Meter.update(flow2_EPE_Oof.data.item(),left_data_tensor_norm.size(0))
        

        D1_errors_Noc_Meter.update(D1_Noc_error.data.item(), left_data_tensor_norm.size(0))
        D1_errors_Ooc_Meter.update(D1_Occ_error.data.item(), left_data_tensor_norm.size(0))
        D1_errors_Oof_Meter.update(D1_Oof_error.data.item(), left_data_tensor_norm.size(0))
        
    
     
        
    print(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
    print(' * EPE Noc {:.3f}'.format(flow2_EPEs_Noc_Meter.avg))
    print(' * EPE Occ {:.3f}'.format(flow2_EPEs_Ooc_Meter.avg))
    print(' * EPE Oof {:.3f}'.format(flow2_EPEs_Oof_Meter.avg))
    print("---------------------------------------------------")
    print(' * P1_error {:.3f}'.format(P1_errors.avg))
    print("---------------------------------------------------")
    print(' * D1_error {:.3f}'.format(D1_errors.avg))
    print(' * D1 Noc {:.3f}'.format(D1_errors_Noc_Meter.avg))
    print(' * D1 Occ {:.3f}'.format(D1_errors_Ooc_Meter.avg))
    print(' * D1 Oof {:.3f}'.format(D1_errors_Oof_Meter.avg))
    


    results_dict = dict()
    results_dict['epe_all'] = flow2_EPEs.avg
    results_dict['epe_noc'] = flow2_EPEs_Noc_Meter.avg
    results_dict['epe_occ'] = flow2_EPEs_Ooc_Meter.avg
    results_dict['epe_oof'] = flow2_EPEs_Oof_Meter.avg
    
    results_dict['p1'] = P1_errors.avg
    
    results_dict['d1_all'] = D1_errors.avg
    results_dict['d1_noc'] = D1_errors_Noc_Meter.avg
    results_dict['d1_occ'] = D1_errors_Ooc_Meter.avg
    results_dict['d1_oof'] = D1_errors_Oof_Meter.avg


    saved_json = os.path.join(args.output_path,'metric.json')
    # Writing JSON data
    with open(saved_json, 'w') as file:
        json.dump(results_dict, file, indent=4)   
        