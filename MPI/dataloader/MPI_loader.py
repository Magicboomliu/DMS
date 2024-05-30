from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys

from dataloader.utils import read_text_lines
import numpy as np

from dataloader.sdk.python.sintel_io import disparity_read
from dataloader.mpi_io import read_img,read_occ
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


class MPI_Dataset(Dataset):
    def __init__(self,
                 datapath,
                 trainlist,
                 transform=None,
                 targetHW = (440,1024),
                 visible_list = ['clean_left','clean_right',
                                 'disp','final_left',
                                 'final_right','occlusions','outofframe'],
                 double_med = False
                 ):
        super(MPI_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist

        self.transform = transform
        self.train_resolution = targetHW
        self.visible_list = visible_list
        
        self.double_med = double_med
        

        self.samples  =[]        
        lines = read_text_lines(self.trainlist)
        
        for line in lines:
            splits = line.split()
            fname = splits[0]
            sample = dict()
            
            if 'clean_left' in self.visible_list:
                sample['clean_left'] = os.path.join(datapath,os.path.join("clean_left",fname))
            if 'clean_right' in self.visible_list:
                sample['clean_right'] = os.path.join(datapath,os.path.join("clean_right",fname))
            if 'disp' in self.visible_list:
                sample['disp'] = os.path.join(datapath,os.path.join("disparities",fname))
            if 'final_left' in self.visible_list:
                sample['final_left'] = os.path.join(datapath,os.path.join("final_left",fname))
            if "final_right" in self.visible_list:
                sample['final_right'] = os.path.join(datapath,os.path.join("final_right",fname))
            if "occlusions" in self.visible_list:
                sample['occlusions'] = os.path.join(datapath,os.path.join("occlusions",fname))
            if "outofframe" in self.visible_list:
                sample['outofframe'] = os.path.join(datapath,os.path.join("outofframe",fname))
            
            if 'rendered_left_left' in self.visible_list:
                sample['rendered_left_left'] = os.path.join(datapath,os.path.join("rendered_left_left",fname))
            if 'rendered_right_right' in self.visible_list:
                sample['rendered_right_right'] = os.path.join(datapath,os.path.join("rendered_right_right",fname))
            if 'rendered_med' in self.visible_list:
                sample['rendered_med'] = os.path.join(datapath,os.path.join("rendered_med",fname))
            if 'rendered_one_third' in self.visible_list:
                sample['rendered_one_third'] = os.path.join(datapath,os.path.join("rendered_one_third",fname))
            if 'rendered_two_third' in self.visible_list:
                sample['rendered_two_third'] = os.path.join(datapath,os.path.join("rendered_two_third",fname))
                
            if double_med:
                sample['rendered_med_right'] = sample['rendered_med'].replace("rendered_med","rendered_med_right")
            
                
            self.samples.append(sample)
    
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]


        if 'clean_left' in self.visible_list:
            sample['clean_left'] = read_img(sample_path['clean_left'])
            if self.train_resolution is not None:
                sample['clean_left'], sample['origin_size']= image_pad(sample['clean_left'],targetHW=self.train_resolution)
        
        if 'clean_right' in self.visible_list:
            sample['clean_right'] = read_img(sample_path['clean_right'])
            if self.train_resolution is not None:
                sample['clean_right'], sample['origin_size']= image_pad(sample['clean_right'],targetHW=self.train_resolution)
        
        
        if 'disp' in self.visible_list:
            sample['disp'] = disparity_read(sample_path['disp'])
            if self.train_resolution is not None:
                sample['disp'] = pad_hw_to_size(sample['disp'],targetHW=self.train_resolution)
            

        if 'final_left' in self.visible_list:
            sample['final_left'] = read_img(sample_path['final_left'])
            if self.train_resolution is not None:
                sample['final_left'], sample['origin_size']= image_pad(sample['final_left'],targetHW=self.train_resolution)
        
        
        if "final_right" in self.visible_list:
            sample['final_right'] = read_img(sample_path['final_right'])
            if self.train_resolution is not None:
                sample['final_right'], sample['origin_size']= image_pad(sample['final_right'],targetHW=self.train_resolution)
        
        
        if "occlusions" in self.visible_list:
            sample['occlusions'] = read_occ(sample_path['occlusions'])
            if self.train_resolution is not None:
                sample['occlusions']= pad_hw_to_size(sample['occlusions'],targetHW=self.train_resolution)
        
        if "outofframe" in self.visible_list:
            sample['outofframe'] = read_occ(sample_path['outofframe'])
            if self.train_resolution is not None:
                sample['outofframe']= pad_hw_to_size(sample['outofframe'],targetHW=self.train_resolution)
        
        if 'rendered_left_left' in self.visible_list:
            sample['rendered_left_left'] = read_img(sample_path['rendered_left_left'])
            if self.train_resolution is not None:
                sample['rendered_left_left'], sample['origin_size']= image_pad(sample['rendered_left_left'],targetHW=self.train_resolution)
        
        if 'rendered_right_right' in self.visible_list:
            sample['rendered_right_right'] = read_img(sample_path['rendered_right_right'])
            if self.train_resolution is not None:
                sample['rendered_right_right'], sample['origin_size']= image_pad(sample['rendered_right_right'],targetHW=self.train_resolution)  
        
        if 'rendered_med' in self.visible_list:
            sample['rendered_med'] = read_img(sample_path['rendered_med'])
            if self.train_resolution is not None:
                sample['rendered_med'], sample['origin_size']= image_pad(sample['rendered_med'],targetHW=self.train_resolution)
                
        if self.double_med:
            sample['rendered_med_right'] = read_img(sample_path['rendered_med_right'])
            if self.train_resolution is not None:
                sample['rendered_med_right'], sample['origin_size']= image_pad(sample['rendered_med_right'],targetHW=self.train_resolution)   
            

                
        if 'rendered_one_third' in self.visible_list:
            sample['rendered_one_third'] = read_img(sample_path['rendered_one_third'])
            if self.train_resolution is not None:
                sample['rendered_one_third'], sample['origin_size']= image_pad(sample['rendered_one_third'],targetHW=self.train_resolution) 

        if 'rendered_two_third' in self.visible_list:
            sample['rendered_two_third'] = read_img(sample_path['rendered_two_third'])
            if self.train_resolution is not None:
                sample['rendered_two_third'], sample['origin_size']= image_pad(sample['rendered_two_third'],targetHW=self.train_resolution) 

    
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        



