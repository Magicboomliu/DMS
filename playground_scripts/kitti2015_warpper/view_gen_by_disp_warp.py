import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("../..")
from dataloader.kitti_dataloader.utils import read_text_lines
from playground_scripts.kitti2015_warpper.disparity_warpper import disp_warp
from tqdm import tqdm
from PIL import Image
import numpy as np
from dataloader.kitti_dataloader.kitti_io import read_img
import skimage.io 


def read_disparity_from_png(disparity_file_path):
    disparity = np.array(Image.open(disparity_file_path)).astype(np.float32)/255.0
    return disparity



def saved_image_from_tensor(saved_name,image_tensor):
    
    saved_image_data = (image_tensor.squeeze(0).permute(1,2,0).cpu().numpy()).astype(np.uint8)
    
    skimage.io.imsave(saved_name,saved_image_data)
    
    


if __name__=="__main__":
    
    root_data_path = "/data1/KITTI/KITTIStereo/kitti_2012/"
    input_fname_list_path = "/home/zliu/CVPR25_Detection/CVPR25_Rebuttal/DiffusionMultiBaseline/datafiles/KITTI/KITTI2012/kitti_2012_trantest.txt"
    saved_folder_path ="/data1/KITTI/KITTIStereo/KITTI2012_Additional_Views/"
    scale_factor = 1.5

    # read filenames
    fname_list = read_text_lines(filepath=input_fname_list_path)
    is_2015 = True
    for fname in tqdm(fname_list):
        splits = fname.split()
        left_image = splits[0]
        left_image_completed = os.path.join(root_data_path,left_image)
        assert os.path.exists(left_image_completed)
        
        if "image_2" in left_image:
            right_image = left_image.replace("image_2","image_3")
            assert right_image!=left_image
            is_2015 = True
        elif "colored_0" in left_image:
            right_image = left_image.replace("colored_0","colored_1")
            assert right_image!=left_image
            is_2015 = False
        else:
            raise NotImplementedError
        
        right_image_completed = os.path.join(root_data_path,right_image)
        assert os.path.exists(right_image_completed)
        
        # psuedo left disparity loading.
        if "train" in left_image:
            if is_2015:
                pseudo_left_disparity = left_image_completed.replace("image_2","left_disp_train")
            else:
                pseudo_left_disparity = left_image_completed.replace("colored_0","left_disp_train")
        else:
            if is_2015:
                pseudo_left_disparity = left_image_completed.replace("image_2","left_disp_test")
            else:
                pseudo_left_disparity = left_image_completed.replace("colored_0","left_disp_test")    
        
        assert os.path.exists(pseudo_left_disparity)
        
        

        
        # make warped disparity images for training.
        psuedo_disparity_data = read_disparity_from_png(pseudo_left_disparity)
        pseudo_disp_torch = torch.from_numpy(psuedo_disparity_data).unsqueeze(0).unsqueeze(0)
        current_psuedo_disp_torch = pseudo_disp_torch * 1.0 / scale_factor
        left_image_tensor = torch.from_numpy(read_img(left_image_completed)).permute(2,0,1).unsqueeze(0)
        right_image_tensor = torch.from_numpy(read_img(right_image_completed)).permute(2,0,1).unsqueeze(0)
        warped_new_image, mask = disp_warp(img=right_image_tensor,disp=current_psuedo_disp_torch)
        
        
        
        
        if is_2015:
            if scale_factor==2.0:
                saved_dirname = "image_25_warp"
            elif scale_factor==3.0:
                saved_dirname = "image_27_warp"
            elif scale_factor==1.5:
                saved_dirname = "image_23_warp"
            
            saved_image_name = os.path.join(saved_folder_path,left_image).replace("image_2",saved_dirname)
        else:
            if scale_factor==2.0:
                saved_dirname = "colored_C_warped"
            elif scale_factor==3.0:
                saved_dirname = "colored_067_warped"
            elif scale_factor==1.5:
                saved_dirname = "colored_033_warped"

            saved_image_name = os.path.join(saved_folder_path,left_image).replace("colored_0",saved_dirname)
        
        

        os.makedirs(os.path.dirname(saved_image_name),exist_ok=True)        
        saved_image_from_tensor(saved_name=saved_image_name,image_tensor=warped_new_image)


        

        
        
        
        
        

    
    
    
    
    
    pass

