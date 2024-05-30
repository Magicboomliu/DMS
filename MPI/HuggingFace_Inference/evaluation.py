import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm

import sys
sys.path.append("..")

import argparse


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import  matplotlib.pyplot as plt
from models.PAMStereo.PASMnet import PASMnet

from skimage import io, transform
import numpy as np
from PIL import Image
from utils import utils
from utils.AverageMeter import AverageMeter
import matplotlib.pyplot as plt
from dataloader.mpi_io import read_img,read_occ
from dataloader.sdk.python.sintel_io import disparity_read
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss,Disaprity_EPE_OOF,Disparity_EPE_Occ,D1_metric_Occ,Disparity_EPE_Noc
import json
from HuggingFace_Inference.error_map_vis import disp_error_img
from HuggingFace_Inference.give_color import depth2color
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



import cv2

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
    
    
    EPE_OP = Disparity_EPE_Loss
    EPE_OCC_OP = Disparity_EPE_Occ
    EPE_OFF_OP = Disaprity_EPE_OOF
    EPE_NOC_OP = Disparity_EPE_Noc


    EPEs_Meter = AverageMeter()
    EPEs_OCC_Meter = AverageMeter()
    EPEs_OFF_Meter  = AverageMeter()
    EPEs_NOC_Meter = AverageMeter()
    
    D1_Error_Meter = AverageMeter()
    D1_Error_Occ_Meter = AverageMeter()
    D1_Error_Oof_Meter = AverageMeter()
    D1_Error_Noc_Meter = AverageMeter()


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


    img_nums = 0
    for line in tqdm(lines):
        splits = line.split()
        fname = splits[0]
        
        # get all the path
        final_left_path = os.path.join(args.datapath,os.path.join("final_left",fname))
        final_right_path = os.path.join(args.datapath,os.path.join("final_right",fname))
        occlusion_path = os.path.join(args.datapath,os.path.join("occlusions",fname))
        oof_path = os.path.join(args.datapath,os.path.join("outofframe",fname))
        gt_disp_path = os.path.join(args.datapath,os.path.join("disparities",fname))
        
    

        assert os.path.exists(final_left_path)
        assert os.path.exists(final_right_path)
        assert os.path.exists(occlusion_path)
        assert os.path.exists(oof_path)
        assert os.path.exists(gt_disp_path)
        
        
        
        
        # read all the path
        final_left_data = read_img(final_left_path)
        final_right_data = read_img(final_right_path)
        occlusions_data = read_occ(occlusion_path)
        outofframe_data = read_occ(oof_path)
        gt_disp_data = disparity_read(gt_disp_path)
        occlusions_data = occlusions_data /255
        outofframe_data = outofframe_data /255
        
        # visualization for gt disp / left / right / occlusions / out-of-frame
        if args.vis:
            

            saved_fame = fname.replace("/","_")


            saved_left_image_name = os.path.join(saved_left_folder,os.path.basename(saved_fame))
            skimage.io.imsave(saved_left_image_name,final_left_data.astype(np.uint8))
            saved_right_image_name = os.path.join(saved_right_folder,os.path.basename(saved_fame))
            skimage.io.imsave(saved_right_image_name,final_right_data.astype(np.uint8))
            
            
            saved_disp_name = os.path.join(saved_disp_folder,os.path.basename(saved_fame))
            plt.imsave(saved_disp_name,gt_disp_data, cmap='jet')
            
            saved_occ_name = os.path.join(saved_occ_folder,os.path.basename(saved_fame))
            skimage.io.imsave(saved_occ_name,(occlusions_data*255).astype(np.uint8))

            saved_oof_name = os.path.join(saved_oof_folder,os.path.basename(saved_fame))
            skimage.io.imsave(saved_oof_name,(outofframe_data*255).astype(np.uint8))
        
        
        
        
        # convert the left numpy to tensor
        final_left_tensor = To_Tensor(final_left_data)
        final_right_tensor = To_Tensor(final_right_data)
        occlusions_tensor = To_Tensor_Disp(occlusions_data)
        outofframe_tensor = To_Tensor_Disp(outofframe_data)
        gt_disp_tensor = To_Tensor_Disp(gt_disp_data)
        
        final_left_tensor_norm = Image_Normalization(final_left_tensor)
        final_right_tensor_norm = Image_Normalization(final_right_tensor)
        
        final_left_tensor_norm = final_left_tensor_norm.cuda()
        final_right_tensor_norm = final_right_tensor_norm.cuda()
        occlusions_tensor = occlusions_tensor.unsqueeze(0).unsqueeze(0).cuda()
        outofframe_tensor = outofframe_tensor.unsqueeze(0).unsqueeze(0).cuda()
        gt_disp_tensor = gt_disp_tensor.unsqueeze(0).unsqueeze(0).cuda()

        b,c,h,w = final_left_tensor_norm.shape
        w_pad = 1024 - w
        h_pad = 448 -h
                
        img_nums = 0
        pad = (w_pad,0,h_pad,0)
        left_input_pad = F.pad(final_left_tensor_norm,pad=pad)
        right_input_pad = F.pad(final_right_tensor_norm,pad=pad)
        

        with torch.no_grad():
            output = stereo_matching_network(left_input_pad,right_input_pad)
            output = output[:,:,h_pad:,w_pad:]
            img_nums += left_input_pad.shape[0]
            
        
        # visualization for est disp / error map    
        if args.vis:
            # estimated disparity
            
            saved_fame = fname.replace("/","_")
            
            saved_est_name = os.path.join(saved_est_left_folder,os.path.basename(saved_fame))
            output_vis = output.squeeze(0).squeeze(0).cpu().numpy()
            plt.imsave(saved_est_name,output_vis, cmap='jet')


            # estimated error
            saved_error_map_name = os.path.join(error_map_folder,os.path.basename(saved_fame))
            est_error_map = disp_error_img(output.squeeze(1),gt_disp_tensor.squeeze(1))
            valid_mask = gt_disp_tensor>0
            valid_mask = valid_mask.float()
            est_error_map = est_error_map.cuda()
            
            est_error_map = est_error_map * valid_mask
            est_error_map =  est_error_map * 255
            est_error_map = est_error_map.squeeze(0).permute(1,2,0).cpu().numpy()
            skimage.io.imsave(saved_error_map_name,est_error_map.astype(np.uint8))
            

        
        EPE_Value = EPE_OP(output, gt_disp_tensor)
        EPE_OCC_Value = EPE_OCC_OP(output, gt_disp_tensor,occlusions_tensor)
        EPE_OFF_Value = EPE_OFF_OP(output, gt_disp_tensor,outofframe_tensor)
        EPE_NOC_Value = EPE_NOC_OP(output,gt_disp_tensor,occlusions_tensor,outofframe_tensor)
        D1_All,D1_Ooc,D1_Oof,D1_Noc = D1_metric_Occ(output,gt_disp_tensor,occ_mask=occlusions_tensor,oof_mask=outofframe_tensor)
        
        
        if EPE_Value.data.item() == EPE_Value.data.item():
            EPEs_Meter.update(EPE_Value.data.item(), left_input_pad.size(0))
        if EPE_OCC_Value.data.item() == EPE_OCC_Value.data.item():
            EPEs_OCC_Meter.update(EPE_OCC_Value.data.item(), left_input_pad.size(0))
        if EPE_OFF_Value.data.item() == EPE_OFF_Value.data.item():
            EPEs_OFF_Meter.update(EPE_OFF_Value.data.item(), left_input_pad.size(0))
        EPEs_NOC_Meter.update(EPE_NOC_Value.data.item(),left_input_pad.size(0))
        
        
            
        D1_Error_Meter.update(D1_All.data.item(),left_input_pad.size(0))
        D1_Error_Occ_Meter.update(D1_Ooc.data.item(),left_input_pad.size(0))
        D1_Error_Oof_Meter.update(D1_Oof.data.item(),left_input_pad.size(0))
        D1_Error_Noc_Meter.update(D1_Noc.data.item(),left_input_pad.size(0))


    print(' * DISP EPE {:.3f}'.format(EPEs_Meter.avg))
    print(' * EPE Occ {:.3f}'.format(EPEs_OCC_Meter.avg))
    print(' * EPE OOF {:.3f}'.format(EPEs_OFF_Meter.avg))
    print(' * EPE Noc {:.3f}'.format(EPEs_NOC_Meter.avg))
    print("---------------------------------------")
    print(' * D1 All {:.3f}'.format(D1_Error_Meter.avg))
    print(' * D1 Noc {:.3f}'.format(D1_Error_Noc_Meter.avg))
    print(' * D1 Ooc {:.3f}'.format(D1_Error_Occ_Meter.avg))
    print(' * D1 Oof {:.3f}'.format(D1_Error_Oof_Meter.avg))


    results_dict = dict()
    results_dict['epe_all'] = EPEs_Meter.avg
    results_dict['epe_noc'] = EPEs_NOC_Meter.avg
    results_dict['epe_occ'] = EPEs_OCC_Meter.avg
    results_dict['epe_oof'] = EPEs_OFF_Meter.avg

    
    results_dict['d1_all'] = D1_Error_Meter.avg
    results_dict['d1_noc'] = D1_Error_Noc_Meter.avg
    results_dict['d1_occ'] = D1_Error_Occ_Meter.avg
    results_dict['d1_oof'] = D1_Error_Oof_Meter.avg


    saved_json = os.path.join(args.output_path,'metric.json')
    # Writing JSON data
    with open(saved_json, 'w') as file:
        json.dump(results_dict, file, indent=4) 
        
        
        
        
            

        

        
        
        
        
        
    

