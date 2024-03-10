from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys
from dataloader.kitti_io import read_img
from dataloader.calib_parse import parse_calib
from dataloader.utils import read_text_lines
import numpy as np


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


class KITTIRaw_Dataset(Dataset):
    def __init__(self,datapath,
                 trainlist,vallist,
                 mode='train',
                 transform=None,
                 targetHW = (377,1248),
                 save_filename=False):
        super(KITTIRaw_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        self.mode = mode
        self.transform = transform
        self.save_filename = save_filename
        self.train_resolution = targetHW
        
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            "test": self.vallist
        }
        
        self.samples  =[]
        
        lines = read_text_lines(dataset_dict[mode])
        
        for line in lines:
            splits = line.split()
            left_image_path = splits[0]
            right_image_path = left_image_path.replace("image_02",'image_03')
            # camera_pose = os.path.join(left_image_path[:10],"calib_cam_to_cam.txt") 
            sample = dict()
            
            if self.save_filename:
                sample['left_name']= left_image_path.replace("/","_")
            
            sample['left_image_path'] = os.path.join(datapath,left_image_path)
            sample['right_image_path'] = os.path.join(datapath,right_image_path)
            
            
            # sample['camera_pose_path'] = os.path.join(datapath,camera_pose)
            
            self.samples.append(sample)
    
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']
        
        
        sample['left_image'] = read_img(sample_path['left_image_path'])
        sample['right_image'] = read_img(sample_path['right_image_path'])
        
        
        if self.train_resolution is not None:
            sample['left_image'], sample['origin_size']= image_pad(sample['left_image'],targetHW=self.train_resolution)
            sample['right_image'],sample['origin_size'] = image_pad(sample['right_image'],targetHW=self.train_resolution)
        
        
    
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        