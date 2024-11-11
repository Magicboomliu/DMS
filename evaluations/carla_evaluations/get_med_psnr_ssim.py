import numpy as np
import torch

from PIL import Image
from tqdm.auto import tqdm
import os
import sys
sys.path.append("../")
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import json
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def get_ssim(image1, image2):
    if image1.shape[-1]==3:
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)    
    ssim = compare_ssim(image1,image2)
    return ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Kitti Inference")

    parser.add_argument(
        "--datapath",
        type=str,
        default="/data1/KITTI/KITTI_Raw/",
        help="Example Image Path",
    )
    parser.add_argument(
        "--target_datapath",
        type=str,
        default="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Validation/",
        help="Example Image Path",
    )
    parser.add_argument(
        "--fnamelist",
        type=str,
        default='',
        help="Example Image Path",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default="SimpleUNet.json",
        help="Example Image Path",
    )

    
    
    args = parser.parse_args()    
    return args



def resize_image(image,targetHW):

    
    resized_image = cv2.resize(image, targetHW, interpolation=cv2.INTER_LINEAR)
    
    return resized_image, np.array(image.shape[:2])


def resize_image_V2(image,targetHW):

    
    resized_image = cv2.resize(image, targetHW, interpolation=cv2.INTER_CUBIC)
    
    return resized_image, np.array(image.shape[:2])

if __name__=="__main__":
    
    args = parse_args()
    
    datapath = args.datapath
    target_datapath = args.target_datapath
    fnamelist = args.fnamelist
    output_json_files = args.output_json_path
    contents = read_text_lines(fnamelist)
    
    left_images_psnr_meter = 0
    right_images_psnr_meter= 0
    total_images_psnr_meter = 0

    left_images_ssim_meter = 0
    right_images_ssim_meter= 0
    total_images_ssim_meter = 0
    cnt = 0
    for fname in tqdm(contents):
            
    
        # Get GT Left Images and GT Right Images
        gt_left_images = os.path.join(datapath,fname)
        med_images = gt_left_images.replace("Left_RGB","Middle_RGB")
        gt_med_image_data = read_img(med_images)        
        gt_med_image_data  = gt_med_image_data.astype(np.uint8)


        gt_med_image_data,original_size = resize_image(gt_med_image_data,
                                                        targetHW=(960,540))
        
        rendered_med_from_left = os.path.join(target_datapath,fname)
        rendered_med_from_left = rendered_med_from_left.replace("Left_RGB","Middle_RGB")

        assert os.path.exists(med_images)
        assert os.path.exists(rendered_med_from_left)

        
        cnt = cnt + 1
        rendered_med_from_left_data = read_img(rendered_med_from_left)
        rendered_med_from_left_data = rendered_med_from_left_data.astype(np.uint8)

        if rendered_med_from_left_data.shape!=gt_med_image_data.shape:
            rendered_med_from_left_data ,_ = resize_image_V2(rendered_med_from_left_data ,
                                                        targetHW=(960,540))
        
        
        # left image psnr
        left_psnr_value = compare_psnr(gt_med_image_data,rendered_med_from_left_data)
        left_images_psnr_meter = left_images_psnr_meter + left_psnr_value

        

        left_image_ssim_value = get_ssim(gt_med_image_data,rendered_med_from_left_data) 
        left_images_ssim_meter = left_images_ssim_meter + left_image_ssim_value

        
    final_val_left_psnr = round(left_images_psnr_meter/cnt,4)
    final_val_left_ssim = round(left_images_ssim_meter/cnt,4)

    saved_dict = dict()
    
    saved_dict['averge_psnr_left'] = final_val_left_psnr
    saved_dict['averge_ssim_left'] = final_val_left_ssim

    

    # Writing JSON data
    with open(output_json_files, 'w') as file:
        json.dump(saved_dict, file, indent=4)