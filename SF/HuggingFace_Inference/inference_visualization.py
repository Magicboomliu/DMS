import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")

from models.PAMStereo.PASMnet import PASMnet
from tqdm import tqdm

from utils.utils import read_text_lines
import os

import matplotlib.pyplot as plt
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from utils.file_io import read_disp,read_img
from HuggingFace_Trainer.dataset_configuration import prepare_dataset
import skimage
import ast
import numpy as np
from tqdm import tqdm

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm

if __name__=="__main__":
    

    saved_path = "../outputs/fusion_pamstereo_test/ckpt_step_45_epoch2.pt"
    
    left_folder = "left_images"
    right_folder = "right_images"
    os.makedirs(left_folder,exist_ok=True)
    os.makedirs(right_folder,exist_ok=True)
    
    
    
    saved_folder = "fusion/bw"
    os.makedirs(saved_folder,exist_ok=True)

    ckpts = torch.load(saved_path)
    model_ckpt = ckpts['model_state']
    
    stereonet = PASMnet()
    stereonet.load_state_dict(model_ckpt)
    stereonet.cuda()
    
    stereonet.eval()




    source_datapath="/data1/KITTI/KITTI_Raw/"
    outside_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"
    center_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train_SmallSet/"
    confidence_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/"
    trainlist="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTISmallSet/new_kitti_raw_subset2.txt"
    vallist="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTI_2015_train.txt"
    visible_list="['left','right']"

    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(source_datapath=source_datapath,
                                                                    outside_view_datapath=outside_view_datapath,
                                                                    center_view_datapath=center_view_datapath,
                                                                    confidence_datapath=confidence_datapath,
                                                                    trainlist=trainlist,
                                                                    vallist=vallist,
                                                                    batch_size=1,
                                                                    datathread=0,
                                                                    visible_list=ast.literal_eval(visible_list))
    id = 0
    for sample_batched in tqdm(test_loader):
        left_input = sample_batched['img_left']
        right_input = sample_batched['img_right']
        target_disp = sample_batched['gt_disp'].unsqueeze(1)
        id = id+1
        
        
        
        
        
        
        b,c,h,w = left_input.shape
        w_pad = 1280 - w
        h_pad = 384 -h
        pad = (w_pad,0,h_pad,0)
        left_input_pad = F.pad(left_input,pad=pad)
        right_input_pad = F.pad(right_input,pad=pad)
        
        
        left_input_pad = left_input_pad.cuda()
        right_input_pad = right_input_pad.cuda()
        
        
        with torch.no_grad():
            
            output = stereonet(left_input_pad,right_input_pad)
            
            output = output[:,:,h_pad:,w_pad:]
            
            output_saved = output.squeeze(0).squeeze(0).cpu().numpy()
            
            
            output_saved = (output_saved*256).astype(np.uint16)
            
            
            skimage.io.imsave("{}/{}.png".format(saved_folder,id),output_saved)
            
            
            # left_input = Image_DeNormalization(left_input)
            # right_input = Image_DeNormalization(right_input)
            # left_input_vis = left_input.squeeze(0).permute(1,2,0).cpu().numpy()
            # right_input_vis = right_input.squeeze(0).permute(1,2,0).cpu().numpy()
            
            # skimage.io.imsave("{}/{}.png".format(left_folder,id),(left_input_vis*255).astype(np.uint8))
            # skimage.io.imsave("{}/{}.png".format(right_folder,id),(right_input_vis*255).astype(np.uint8))
            
            
            
        