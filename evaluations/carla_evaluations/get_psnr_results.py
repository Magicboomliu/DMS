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
        gt_right_images = os.path.join(datapath,fname)
        
        gt_right_images = gt_right_images.replace("Left_RGB","Right_RGB")

        gt_left_image_data = read_img(gt_left_images)
        gt_right_image_data = read_img(gt_right_images)
        
        gt_left_image_data = gt_left_image_data.astype(np.uint8)
        gt_right_image_data = gt_right_image_data.astype(np.uint8)
        
        
        # resize the image
        
        gt_left_image_data,original_size = resize_image(gt_left_image_data,
                                                        targetHW=(960,536))

        gt_right_image_data,original_size = resize_image(gt_right_image_data,
                                                        targetHW=(960,536))
        

        
        rendered_left_from_right = os.path.join(target_datapath,fname)
        rendered_right_from_left = os.path.join(target_datapath,fname)
        rendered_right_from_left = rendered_right_from_left.replace("Left_RGB","Right_RGB")

        assert os.path.exists(gt_left_images)
        assert os.path.exists(gt_right_images)
        assert os.path.exists(rendered_right_from_left)
        assert os.path.exists(rendered_left_from_right)
        

        


        cnt = cnt + 1
        rendered_left_from_right_data = read_img(rendered_left_from_right)
        rendered_right_from_left_data = read_img(rendered_right_from_left)
        rendered_left_from_right_data = rendered_left_from_right_data.astype(np.uint8)
        rendered_right_from_left_data = rendered_right_from_left_data.astype(np.uint8)
        
        
        if rendered_left_from_right_data.shape!=gt_right_image_data.shape:
            rendered_right_from_left_data,_ = resize_image_V2(rendered_left_from_right_data,
                                                        targetHW=(960,540))
        
        

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

        
    final_val_left_psnr = round(left_images_psnr_meter/cnt,4)
    final_val_right_psnr = round(right_images_psnr_meter/cnt,4)
    final_val_total_psnr = round(total_images_psnr_meter/cnt,4)
    
    final_val_left_ssim = round(left_images_ssim_meter/cnt,4)
    final_val_right_ssim = round(right_images_ssim_meter/cnt,4)
    final_val_total_ssim = round(total_images_ssim_meter/cnt,4)
    
    
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