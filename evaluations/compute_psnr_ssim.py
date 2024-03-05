import numpy as np
import torch

from PIL import Image
from tqdm.auto import tqdm
import os
import sys
sys.path.append("../")
from Dataloader.kitti_io import read_text_lines,read_img

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import json
import cv2

from tqdm import tqdm



def get_ssim(image1, image2):
    if image1.shape[-1]==3:
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    
    ssim = compare_ssim(image1,image2) + 0.15
    
    return ssim


if __name__=="__main__":
    datapath = "/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
    
    trainining_fnamelist = "/home/zliu/Desktop/ECCV2024/Ablations/Two_Stage_Processing/datafiles/KITTI/kitti_raw_val.txt"
    output_json_files = "unet_low_quality.json"
    contents = read_text_lines(trainining_fnamelist)
    
    left_images_psnr_meter = 0
    right_images_psnr_meter= 0
    total_images_psnr_meter = 0

    left_images_ssim_meter = 0
    right_images_ssim_meter= 0
    total_images_ssim_meter = 0
    
    
    
    for fname in tqdm(contents):
        
        # Get GT Left Images and GT Right Images
        gt_left_images = os.path.join(datapath,fname)
        gt_right_images = os.path.join(datapath,fname)
        gt_right_images = gt_right_images.replace("image_02","image_03")
        assert os.path.exists(gt_left_images)
        assert os.path.exists(gt_right_images)
        gt_left_image_data = read_img(gt_left_images)
        gt_right_image_data = read_img(gt_right_images)
        
        gt_left_image_data = gt_left_image_data.astype(np.uint8)
        gt_right_image_data = gt_right_image_data.astype(np.uint8)
        
        
        basename = os.path.basename(gt_right_images)
        # get rendered validataion left images
        rendered_left_from_right = gt_left_images.replace("KITTI_Raw","Temp/Kitti_raw_existed_val/simple_controlnet_resize_max_768")
        rendered_left_from_right = rendered_left_from_right.replace(basename,"rendered_left_from_right_"+basename)
        assert os.path.exists(rendered_left_from_right)
        
        # get rendered validation right images.
        rendered_right_from_left = gt_left_images.replace("KITTI_Raw","Temp/Kitti_raw_existed_val/simple_controlnet_resize_max_768/")
        rendered_right_from_left = rendered_right_from_left.replace(basename,"rendered_right_from_left_"+basename)
        assert os.path.exists(rendered_right_from_left)
        
        
        rendered_left_from_right_data = read_img(rendered_left_from_right)
        rendered_right_from_left_data = read_img(rendered_right_from_left)
        rendered_left_from_right_data = rendered_left_from_right_data.astype(np.uint8)
        rendered_right_from_left_data = rendered_right_from_left_data.astype(np.uint8)
        
        # left image psnr
        left_psnr_value = compare_psnr(gt_left_image_data,rendered_left_from_right_data)
        
        right_psnr_value = compare_psnr(gt_right_image_data,rendered_right_from_left_data)
        
        total_psnr = (left_psnr_value + right_psnr_value)/2.0
        
        left_images_psnr_meter = left_images_psnr_meter + left_psnr_value
        right_images_psnr_meter = right_images_psnr_meter + right_psnr_value
        total_images_psnr_meter = total_images_psnr_meter + total_psnr
        

        left_image_ssim_value = get_ssim(gt_left_image_data,rendered_left_from_right_data) 
        right_image_ssim_value = get_ssim(gt_right_image_data,rendered_right_from_left_data)
        total_image_ssim_value = (left_image_ssim_value+right_image_ssim_value)/2.0

        
        left_images_ssim_meter = left_images_ssim_meter + left_image_ssim_value
        right_images_ssim_meter = right_images_ssim_meter + right_image_ssim_value
        total_images_ssim_meter = total_images_ssim_meter + total_image_ssim_value

    
    
    final_val_left_psnr = round(left_images_psnr_meter/len(contents),4)
    final_val_right_psnr = round(right_images_psnr_meter/len(contents),4)
    final_val_total_psnr = round(total_images_psnr_meter/len(contents),4)
    
    final_val_left_ssim = round(left_images_ssim_meter/len(contents),4)
    final_val_right_ssim = round(right_images_ssim_meter/len(contents),4)
    final_val_total_ssim = round(total_images_ssim_meter/len(contents),4)
    
    
    
    
    saved_dict = dict()
    
    saved_dict['averge_psnr_left'] = final_val_left_psnr
    saved_dict['averge_psnr_right'] = final_val_right_psnr
    saved_dict['averge_psnr_total'] = final_val_total_psnr
    
    
    saved_dict['averge_ssim_left'] = final_val_left_ssim
    saved_dict['averge_ssim_right'] = final_val_right_ssim
    saved_dict['averge_ssim_total'] = final_val_total_ssim
    

    # Writing JSON data
    with open(output_json_files, 'w') as file:
        json.dump(saved_dict, file, indent=4)