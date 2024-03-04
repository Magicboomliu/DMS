from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys
sys.path.append("../..")
from dataloader.kitti_io import read_img
from dataloader.calib_parse import parse_calib
from dataloader.utils import read_text_lines
import numpy as np


from skimage.transform import resize

def resize_to_test_size(input_image=None,test_size=(384,1280)):
    
    H,W = input_image.shape[:2]
    new_H, new_W = test_size
    resized_image = resize(input_image, (new_H, new_W), anti_aliasing=True)
    
    return resized_image,np.array([new_H,new_W])
    

def cut_or_pad_img(img, targetHW):

    t_H, t_W = targetHW
    H, W = img.shape[0], img.shape[1]

    padW = np.abs(t_W - W)
    half_padW = int(padW//2)
    # crop
    if W > t_W:
        img = img[:, half_padW:half_padW+t_W]
    # pad
    elif W < t_W:
        img = np.pad(img, [(0, 0), (half_padW, padW-half_padW), (0, 0)], 'constant')

    # crop
    padH = np.abs(t_H - H)
    if H > t_H:
        img = img[padH:, :]
    # pad
    elif H < t_H:
        padH = t_H - H
        img = np.pad(img, [(padH, 0), (0, 0), (0, 0)], 'constant')

    return img


class KITTIRaw_Dataset(Dataset):
    def __init__(self,datapath,
                 trainlist,vallist,
                 mode='train',
                 transform=None,
                 targetHW = (364,1248),
                 save_filename=False,
                 ):
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
        
        
        if self.mode=='train':
            if self.train_resolution is not None:
                sample['left_image'] = cut_or_pad_img(sample['left_image'],targetHW=self.train_resolution)
                sample['right_image'] = cut_or_pad_img(sample['right_image'],targetHW=self.train_resolution)
        else:
            # resize the image to 64 based.
            sample['left_image'],original_size_left = resize_to_test_size(sample['left_image'])
            sample['right_image'],original_size_right = resize_to_test_size(sample['right_image'])
            
            assert original_size_left[0] == original_size_right[0]
            assert original_size_left[1] == original_size_right[1]
            original_size = original_size_left
            sample["original_size"] = original_size
            
        
    
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        