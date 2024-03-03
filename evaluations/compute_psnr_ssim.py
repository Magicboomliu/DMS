from PIL import Image
from tqdm.auto import tqdm
import os
import sys
sys.path.append("../")
from dataloader.utils import read_text_lines

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import json
import cv2
import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Kitti Evaluations")
    
    parser.add_argument(
        "--datapath",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/KITTI_Raw",
        help="Datapath",
    )
    parser.add_argument(
        "--rendered_path",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/Temp/Kitti_raw_existed_val/simple_controlnet_resize_max_768/",
        help="Path to loaded the pretrained unet.",
    )
    parser.add_argument(
        "--validation_files",
        type=str,
        default="/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt",
        help="Example Image Path",
    )
    parser.add_argument(
        "--output_json_files",
        type=str,
        default="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Evaluation_Results/Simple_Unet/SD15_val.json",
        help="/home/zliu/ACMMM2024/DiffusionMultiBaseline/outputs/Evaluation_Results/Simple_Unet/SD15_val.json",
    )

    args = parser.parse_args()

    
    return args


def read_img(image):
    return np.array(Image.open(image).convert("RGB"))





if __name__=="__main__":
    
    args = parse_args()
    
    datapath = args.datapath
    rendered_path = args.rendered_path
    validation_files = args.validation_files
    output_json_files = args.output_json_files
    
    basename = os.path.basename(output_json_files)
    
    saved_output_folder = output_json_files[:-len(basename)]
    os.makedirs(saved_output_folder,exist_ok=True)

        
    lines = read_text_lines(validation_files)
    psnr_total =0.0
    ssim_total = 0.0
    
    for line in tqdm(lines):
        splits = line.split()
        left_image = splits[0]
        right_image_gt = left_image.replace("image_02","image_03")
        right_image_render = left_image.replace("image_02","image_03")
        
        left_image = os.path.join(datapath,left_image)
        right_image_gt = os.path.join(datapath,right_image_gt)
        right_image_render = os.path.join(rendered_path,right_image_render)
        assert os.path.basename(right_image_gt) == os.path.basename(right_image_render)
        
        # read the gt right image
        right_image_gt_data = read_img(right_image_gt).astype(np.uint8)
        
        # read the estimated left image
        right_image_render_data = read_img(right_image_render).astype(np.uint8)
        
        
        # estimate the psnr
        psnr_value = compare_psnr(right_image_gt_data,right_image_render_data)
        psnr_total = psnr_total + psnr_value
        
        right_image_gt_gray = cv2.cvtColor(right_image_gt_data,cv2.COLOR_RGB2GRAY)
        right_image_render_gray = cv2.cvtColor(right_image_render_data,cv2.COLOR_RGB2GRAY)
 
 
        # estimate the ssim
        ssim_value = compare_ssim(right_image_gt_gray,right_image_render_gray)
        ssim_total = ssim_total + ssim_value
        


    average_psnr_total = psnr_total*1.0/len(lines)
    average_ssim_total = ssim_total*1.0/len(lines)
    saved_dict = dict()
    saved_dict['averge_psnr'] = round(average_psnr_total,5)
    saved_dict['averge_ssim'] = round(average_ssim_total,5)

    # Writing JSON data
    with open(output_json_files, 'w') as file:
        json.dump(saved_dict, file, indent=4)

