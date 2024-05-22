from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys
import numpy as np
from dataloader.sceneflow_dataloader.sceneflow_io import read_disp,read_img,read_occ
from dataloader.sceneflow_dataloader.utils import read_text_lines
import numpy as np



def image_pad(image,targetHW):
    H,W = image.shape[:2]
    
    new_H, new_W = targetHW
    pad_bottom = new_H - H  # 底部填充量
    pad_right = new_W - W  # 右侧填充量

    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
    
    return added_image, np.array(image.shape[:2])


def pad_hw_to_size(image, targetHW):
    """
    Pad a 2D numpy array to the target height and width, padding only on the right and bottom.

    Args:
    image (numpy.ndarray): The input 2D array.
    target_hw (tuple): A tuple (target_height, target_width) specifying the desired dimensions.

    Returns:
    numpy.ndarray: A new array that has been padded to the target dimensions.
    """
    current_height, current_width = image.shape
    target_height, target_width = targetHW

    # Calculate how much padding is needed
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    # Apply the padding
    padded_image = np.pad(image, 
                          ((0, pad_height),  # Add padding to the bottom
                           (0, pad_width)),  # Add padding to the right
                          mode='constant', 
                          constant_values=0)  # Fill the padding with zeros

    return padded_image


class SceneFlow_Dataset(Dataset):
    def __init__(self,
                 datapath,
                 trainlist,
                 transform=None,
                 targetHW = (576,960),
                 visible_list = ['left','right',
                                 'disp','occlusion']
                 ):
        super(SceneFlow_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist

        self.transform = transform
        self.train_resolution = targetHW
        self.visible_list = visible_list
        

        self.samples  =[]        
        lines = read_text_lines(self.trainlist)
        
        for line in lines:
            splits = line.split()
            fname = splits[0]
            sample = dict()
            
            if 'left' in self.visible_list:
                sample['left'] = os.path.join(self.datapath,fname)
            if 'right' in self.visible_list:
                sample['right'] = sample['left'].replace("left","right")

            if 'disp' in self.visible_list:
                sample['disp'] =  sample['left'].replace("frames_cleanpass","disparity")
                sample['disp'] = sample['disp'].replace(".png",".pfm")
                
            if "occlusions" in self.visible_list:
                sample['occlusions'] = sample['left'].replace("frames_cleanpass","occlusion")
                sample['occlusions'] = sample['occlusions'].replace(".png",".npy")

                
            self.samples.append(sample)
    
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]


        if 'left' in self.visible_list:
            sample['left'] = read_img(sample_path['left'])
            if self.train_resolution is not None:
                sample['left'],sample['original_resolution'] = image_pad(sample['left'],targetHW=(576,960))
                pass
        
        if 'right' in self.visible_list:
            sample['right'] = read_img(sample_path['right'])
            if self.train_resolution is not None:
                sample['right'],sample['original_resolution'] = image_pad(sample['right'],targetHW=(576,960))
                
            
        if 'disp' in self.visible_list:
            sample['disp'] = read_disp(sample_path['disp'])
            if self.train_resolution is not None:
                sample['disp'] = pad_hw_to_size(sample['disp'],targetHW=(576,960))
            
        if "occlusions" in self.visible_list:
            sample['occlusions'] = read_occ(sample_path['occlusions'])
            if self.train_resolution is not None:
                sample['occlusions'] = pad_hw_to_size(sample['occlusions'],targetHW=(576,960))
            
            
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        



