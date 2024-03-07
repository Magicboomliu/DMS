import os
import sys
sys.path.append("..")
from dataloader.utils import read_text_lines
from dataloader.kitti_io import read_img
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def resize_max_res_tensor_new_version(input_tensor,is_disp=False,recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor
    

def image_pad(image,targetHW):
    H,W = image.shape[:2]
    
    new_H, new_W = targetHW
    # 计算填充量
    pad_bottom = new_H - H  # 底部填充量
    pad_right = new_W - W  # 右侧填充量

    # 应用填充
    # numpy.pad的参数是一个元组的序列，每个元组代表一个轴的填充方式，形式为(左侧填充量, 右侧填充量)
    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
    
    return added_image, np.array(image.shape[:2])


if __name__=="__main__":
    
    datapath = "/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
    filenames = "/home/zliu/ACMMM2024/DiffusionMultiBaseline/datafiles/KITTI/kitti_raw_val.txt"
    
    contents = read_text_lines(filenames)
    
    size_list = []
    for fname in tqdm(contents):
        fname_left = os.path.join(datapath,fname)
        image= read_img(fname_left)
        image= image.astype(np.float32)/255
        
        image_padded,original_size = image_pad(image,targetHW=(377,1248))
        recover_image = image[:original_size[0],:original_size[1]]
        
        print(original_size)
        
        
        quit()
        image_padded_tensor = torch.from_numpy(image_padded).permute(2,0,1).unsqueeze(0)
        
        resize_image = resize_max_res_tensor_new_version(image_padded_tensor)
        # print(original_size)
        print(resize_image.shape)