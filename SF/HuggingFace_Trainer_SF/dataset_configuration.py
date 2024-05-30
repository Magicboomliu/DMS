import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from torch.utils.data import DataLoader 
from dataloader_sf.SceneFlow_loader import SceneFlow_Dataset
from dataloader_sf import transforms
import numpy as np
from tqdm import tqdm
from utils.devtools import Convert_IMGTensor_To_Numpy
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm

# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    batch_size,
                    datathread,
                    visible_list=None):
    
    train_transform_list = [
                            transforms.RandomCrop(320,640),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    train_transform = transforms.Compose(train_transform_list)
    
    
    val_transform_list = [transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    val_transform = transforms.Compose(val_transform_list)
    
    
    train_dataset = SceneFlow_Dataset(datapath=datapath,
                                    trainlist=trainlist,
                                    vallist=vallist,
                                                 mode='train',
                                                 transform=train_transform,
                                                 visible_list=visible_list
                                                 )
    
    
    test_dataset = SceneFlow_Dataset(datapath=datapath,
                                        trainlist=trainlist,
                                        vallist=vallist,
                                                 mode='test',
                                                 transform=val_transform,
                                                 visible_list=visible_list
                                                 )
    

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = 4, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)

    return train_loader,test_loader,num_batches_per_epoch







if __name__=="__main__":
    
    datapath = "/data1/zliu/"
    trainlist = "/home/zliu/AblationStudies_Stereo_Matching/datafiles/SF/train.txt"
    vallist = "/home/zliu/AblationStudies_Stereo_Matching/datafiles/SF/val.txt"
    
    
    
    
    
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath=datapath,
                                                                     trainlist=trainlist,
                                                                     vallist=vallist,
                                                                     batch_size=1,
                                                                     datathread=1,
                                                                     visible_list=['left','right','disp'])
    
    print(len(train_loader))
    print(len(test_loader))
    # The training dataset
    # for idx, sample in enumerate(train_loader):
        
    #     left_image = sample['left']  # [B,3,H,W]
    #     right_image = sample['right'] # [B,3,H,W]
    #     disp = sample['disp']
    #     disp = disp.unsqueeze(1)
        
    #     print(left_image.shape)
    #     print(right_image.shape)
    #     print(disp.shape)
    #     print("------------------------------")
    #     break




    for idx, sample in enumerate(test_loader):
        
        left_image = sample['left']
        right_image = sample['right']
        disp = sample['disp']
        disp = disp.unsqueeze(1)
        
        print(left_image.shape)
        print(right_image.shape)
        print(disp.shape)
        print("-------------------------------")
        
        
    #     plt.figure(figsize=(10,20))
    #     plt.subplot(1,3,1)
    #     plt.axis("off")
    #     plt.imshow(left_image.squeeze(0).permute(1,2,0).cpu().numpy())
    #     plt.subplot(1,3,2)
    #     plt.axis('off')
    #     plt.imshow(right_image.squeeze(0).permute(1,2,0).cpu().numpy())
    #     plt.subplot(1,3,3)
    #     plt.axis('off')
    #     plt.imshow(disp.squeeze(0).cpu().numpy(),cmap='gray')
    #     plt.savefig("test.png",bbox_inches='tight')
        
        
    #     print(left_image.shape)
    #     print(right_image.shape)
    #     print(disp.shape)
        
    #     break
        
        
        
        
        
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys

from dataloader_sf.utils import read_text_lines
from dataloader_sf.sceneflow_io import read_img,read_occ,read_disp
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
                 vallist,
                 mode='train',
                 transform=None,
                 targetHW = (576,960),
                 visible_list = ['left','right',
                                 'disp','occlusion']
                 ):
        super(SceneFlow_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist

        self.transform = transform
        self.mode = mode
        self.train_resolution = targetHW
        self.visible_list = visible_list
        

        self.samples  =[]
        if self.mode=='train':
            lines = read_text_lines(self.trainlist)
        elif self.mode=='test' or self.mode =='val':
            lines = read_text_lines(self.vallist)
            self.datapath = os.path.join(self.datapath,'sceneflow')
        
        
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

            if self.mode=='train':
                if "left_left" in self.visible_list:
                    sample['left_left'] = sample['left'].replace("sceneflow","SF_Rendered_Results/sceneflow/")
                    sample['left_left'] = sample['left_left'].replace("left","left_left")
                
                if 'right_right' in self.visible_list:
                    sample['right_right'] = sample['left'].replace("sceneflow","SF_Rendered_Results/sceneflow/")
                    sample['right_right'] = sample['right_right'].replace("left","right_right")
                
                if 'center' in self.visible_list:
                    sample['center'] = sample['left'].replace("sceneflow","SF_Rendered_Results/sceneflow/")
                    sample['center'] = sample['center'].replace("left","center")
                
            self.samples.append(sample)
    
    
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if 'left' in self.visible_list:
            sample['left'] = read_img(sample_path['left'])
            if self.train_resolution is not None:
                sample['left'],sample['original_resolution'] = image_pad(sample['left'],targetHW=(576,960))
        
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
        
        if self.mode=='train':
            if 'left_left' in self.visible_list:
                sample['left_left'] = read_img(sample_path['left_left'])
                if self.train_resolution is not None:
                    sample['left_left'],sample['original_resolution'] = image_pad(sample['left_left'],targetHW=(576,960))
            
            if "right_right" in self.visible_list:
                sample['right_right'] = read_img(sample_path['right_right'])
                if self.train_resolution is not None:
                    sample['right_right'], sample['original_resolution'] = image_pad(sample['right_right'],targetHW=(576,960))
            
            if 'center' in self.visible_list:
                sample['center'] = read_img(sample_path['center'])
                if self.train_resolution is not None:
                    sample['center'], sample['original_resolution'] = image_pad(sample['center'],targetHW=(576,960))
            
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        
