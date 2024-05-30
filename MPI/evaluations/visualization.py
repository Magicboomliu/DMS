import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("..")
from HuggingFace_Trainer.dataset_configuration import prepare_dataset,covered_resultion,covered_resultion
import ast
from losses.pam_loss import warp_disp,ssim
import matplotlib.pyplot as plt
import numpy as np
from models.PAMStereo.PASMnet import PASMnet

from evaluations.error_map import disp_error_img


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm



def L1Loss_without_mean(input, target):
    return (input - target).abs()

def L1Loss(input, target):
    return (input - target).abs().mean()

if __name__=="__main__":
    
    
    saved_path = "/home/zliu/NIPS2024/UnsupervisedStereo/PAM_Based/results/plus_all_in_all/"
    saved_est_disp = os.path.join(saved_path,'est_disp')
    os.makedirs(saved_est_disp,exist_ok=True)
    saved_gt_disp = os.path.join(saved_path,'gt_disp')
    os.makedirs(saved_gt_disp,exist_ok=True)
    saved_error_map = os.path.join(saved_path,'error_map')
    os.makedirs(saved_error_map,exist_ok=True)
    saved_occlusion = os.path.join(saved_path,"occ")
    os.makedirs(saved_occlusion,exist_ok=True)
    saved_oof = os.path.join(saved_path,'oof')
    os.makedirs(saved_oof,exist_ok=True)    
    
    saved_left_folder = os.path.join(saved_path,'left_image')
    os.makedirs(saved_left_folder,exist_ok=True)
    saved_right_folder = os.path.join(saved_path,'right_image')
    os.makedirs(saved_right_folder,exist_ok=True)
    
    
    
    
    resume_from_checkpoint = "/home/zliu/NIPS2024/UnsupervisedStereo/PAM_Based/outputs/plus_all_in_all/best_EPE_model.pt"
    stereo_matching_network = PASMnet()
    ckpts = torch.load(resume_from_checkpoint)
    model_ckpt = ckpts['model_state']
    
    score = ckpts['best_score']

    stereo_matching_network.load_state_dict(model_ckpt)
    print("Loaded the Stereo Matching pre-trained models")
    
    stereo_matching_network.cuda()
    stereo_matching_network.eval()
    
    
    
    
    datapath = "/data1/liu/Sintel/"
    trainlist = "/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
    vallist = "/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
    batch_size = 1
    datathread = 0
    visible_list = "['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_left_left','rendered_right_right']"
    
    
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath= datapath,
                                                                         trainlist= trainlist,
                                                                         vallist= vallist,
                                                                         batch_size= batch_size,
                                                                         datathread=datathread,
                                                                         targetHW=(440,1024),
                                                                         visible_list=ast.literal_eval(visible_list))
    
    for idx, batch in enumerate(test_loader):
        left_image_data = batch['final_left'] # left image
        right_image_data = batch['final_right'] # right pose
                
        gt_disp_data = batch['disp'].unsqueeze(1)
        gt_occlusion_data = batch['occlusions'].unsqueeze(1)
        gt_outof_frame_data = batch['outofframe'].unsqueeze(1)


        left_left_image_data = batch["rendered_left_left"]
        right_right_image_data = batch["rendered_right_right"]
        med_image_data = batch["rendered_med"]
        
        
        left_image_data = left_image_data.cuda()
        right_image_data = right_image_data.cuda()
        gt_disp_data = gt_disp_data.cuda()
        gt_occlusion_data = gt_occlusion_data.cuda()
        gt_outof_frame_data = gt_outof_frame_data.cuda()
        
        left_left_image_data = left_left_image_data.cuda()
        right_right_image_data = right_right_image_data.cuda()
        med_image_data = med_image_data.cuda()
        
        #-----------------------------------------------------#

        gt_disp_data = covered_resultion(gt_disp_data)
        gt_occlusion_data = covered_resultion(gt_occlusion_data)
        gt_outof_frame_data = covered_resultion(gt_outof_frame_data)
        
        left_image_data = covered_resultion(left_image_data)
        right_image_data = covered_resultion(right_image_data)
        
        
        left_image_data_vis = Image_DeNormalization(left_image_data)
        right_image_data_vis = Image_DeNormalization(right_image_data)
        
        with torch.no_grad():
            
            output = stereo_matching_network(left_image_data,right_image_data)
            error_map = disp_error_img(output.squeeze(0),gt_disp_data.squeeze(0))
            error_map_vis = error_map.squeeze(0).permute(1,2,0).cpu().numpy()
            # print(error_map_vis.shape)
            disparity_vis = output.squeeze(0).squeeze(0).cpu().numpy()
            
            gt_disparity = gt_disp_data.squeeze(0).squeeze(0).cpu().numpy()
            
            
            gt_occ_mask = gt_occlusion_data.squeeze(0).squeeze(0).cpu().numpy()
            gt_oof_mask = gt_outof_frame_data.squeeze(0).squeeze(0).cpu().numpy()
            
            left_image_vis = left_image_data_vis.squeeze(0).permute(1,2,0).cpu().numpy()
            right_image_vis = right_image_data_vis.squeeze(0).permute(1,2,0).cpu().numpy()
            
            saved_error_map_path = os.path.join(saved_error_map,"{}.png".format(idx))
            est_disparity_map_path = os.path.join(saved_est_disp,"{}.png".format(idx))
            gt_disp_map_path = os.path.join(saved_gt_disp,"{}.png".format(idx))
            
            gt_occ_map_path = os.path.join(saved_occlusion,"{}.png".format(idx))
            gt_oof_map_path = os.path.join(saved_oof,"{}.png".format(idx))
            
            left_image_path = os.path.join(saved_left_folder,"{}.png".format(idx))
            right_image_path = os.path.join(saved_right_folder,"{}.png".format(idx))
            
            
            
            plt.imsave(saved_error_map_path,error_map_vis)
            plt.imsave(gt_disp_map_path,gt_disparity)
            plt.imsave(est_disparity_map_path,disparity_vis)
            
            plt.imsave(gt_occ_map_path,gt_occ_mask,cmap='gray')
            plt.imsave(gt_oof_map_path,gt_oof_mask,cmap='gray')

            plt.imsave(left_image_path,left_image_vis)
            plt.imsave(right_image_path,right_image_vis)
            
            
        if idx%10==0:
            print("finished {}/{}".format(idx,len(test_loader)))
            
            
            
            

        
