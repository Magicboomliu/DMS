import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("../..")
from dataloader.carla_dataloader.carla_loader import CARLA_MV_Dataset
from dataloader.carla_dataloader import transforms
import os
import logging

import matplotlib.pyplot as plt


# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    logger=None,
                    batch_size = 1,
                    test_size =1,
                    datathread =4,
                    targetHW=(1920,1080)):

    train_transform_list = [transforms.ToTensor(),]
    train_transform = transforms.Compose(train_transform_list)

    val_transform_list = [transforms.ToTensor()]
    
    val_transform = transforms.Compose(val_transform_list)
    
    
    train_dataset = CARLA_MV_Dataset(datapath=datapath,trainlist=trainlist,vallist=vallist,transform=train_transform,
                                     mode='train',targetHW = targetHW)

    test_dataset = CARLA_MV_Dataset(datapath=datapath,trainlist=trainlist,vallist=vallist,transform=val_transform,
                                     mode='test',targetHW = targetHW)


    datathread=datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_size, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)    
    return (train_loader,test_loader),num_batches_per_epoch


def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):
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
    

def image_normalization(image_tensor):
    image_normalized = image_tensor * 2.0 -1.0
    return image_normalized



if __name__=="__main__":
    
    datapath = "/media/zliu/data12/dataset/CARLA/"
    trainlist = "/home/zliu/CVPR25_Detection/DiffusionMultiBaseline/datafiles/CARLA/training.txt"
    vallist ="/home/zliu/CVPR25_Detection/DiffusionMultiBaseline/datafiles/CARLA/testing.txt"
    batch_size = 1
    test_size = 1
    datathread = 4
    
    
    (train_loader,test_loader),num_batches_per_epoch = prepare_dataset(datapath=datapath,
                                                                       trainlist=trainlist,
                                                                       vallist=vallist,
                                                                       logger=None,
                                                                       batch_size=1,test_size=1,
                                                                       targetHW=(1920,1080),
                                                                       datathread=4)
    
    # train dataloader
    for idx, sample in enumerate(train_loader):
        left_image = sample['left_image']
        right_image = sample['right_image']
        mid_image = sample['mid_image']

        left_depth = 1000 / sample['left_depth']
        right_depth = 1000 / sample['right_depth']
        mid_depth = 1000 / sample['mid_depth']
        
        fx = sample['fx']
        
        
        plt.subplot(2,3,1)
        plt.axis("off")
        plt.imshow(left_image.squeeze(0).permute(1,2,0).cpu().numpy())
        plt.subplot(2,3,2)
        plt.axis("off")
        plt.imshow(mid_image.squeeze(0).permute(1,2,0).cpu().numpy())
        plt.subplot(2,3,3)
        plt.axis("off")
        plt.imshow(right_image.squeeze(0).permute(1,2,0).cpu().numpy())
        plt.subplot(2,3,4)
        plt.axis("off")
        plt.imshow(left_depth.squeeze(0).squeeze(0).cpu().numpy(),cmap='jet')
        plt.subplot(2,3,5)
        plt.axis("off")
        plt.imshow(mid_depth.squeeze(0).squeeze(0).cpu().numpy(),cmap='jet')
        plt.subplot(2,3,6)
        plt.axis("off")
        plt.imshow(right_depth.squeeze(0).squeeze(0).cpu().numpy(),cmap='jet')
        
        plt.show()
        
        
        
        
        quit()
        
        
        pass

    

    
    