import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from torch.utils.data import DataLoader


from dataloader.MPI_loader import MPI_Dataset
from dataloader import transforms

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
                    targetHW = (440,1024),
                    visible_list=['final_left','final_right','disp','occlusions',
                                               'outofframe']):
    
    train_transform_list = [
                            transforms.RandomCrop(384,1024),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    
    train_transform = transforms.Compose(train_transform_list)
    
    
    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    
    val_transform = transforms.Compose(val_transform_list)

    
    train_dataset  = MPI_Dataset(datapath=datapath,
                                 trainlist=trainlist,
                                 transform=train_transform,
                                 targetHW=targetHW,
                                 visible_list=visible_list)

    test_dataset  = MPI_Dataset(datapath=datapath,
                                 trainlist=vallist,
                                 transform=val_transform,
                                 targetHW=targetHW,
                                 visible_list=visible_list)


    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = 1, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)
    
    
    return train_loader,test_loader,num_batches_per_epoch



def covered_resultion(image):
    image = image[:,:,:436,:]
    image = image[:,:,26:-26,:]
    
    return image    



if __name__=="__main__":
    
    datapth = "/data1/liu/Sintel/"
    trainlist = "/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
    vallist = "/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
    
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath=datapth,
                                                                     trainlist=trainlist,
                                                                     vallist=vallist,
                                                                     datathread=4,
                                                                     batch_size=8,
                                                                     targetHW=(440,1024),
                                                                     visible_list=['final_left',
                                                                                   'final_right','disp',
                                                                                   'occlusions','outofframe'])
    
    for sample in tqdm(test_loader):
        
        left_image = sample['final_left']
        right_image = sample['final_right']
        gt_disp = sample['disp'].unsqueeze(1)
        gt_occlusion = sample['occlusions'].unsqueeze(1)
        gt_out_of_frame = sample['outofframe'].unsqueeze(1)
        
        
        
        
        print(left_image.shape)
        print(right_image.shape)
        print(gt_disp.shape)
        print(gt_occlusion.shape)
        print(gt_out_of_frame.shape)
        
        print("-----------------")    
        
        left_image_recovered = covered_resultion(left_image)
        right_image_recovered = covered_resultion(right_image)
        gt_disp_recovred = covered_resultion(gt_disp)
        gt_occlusion_recovred = covered_resultion(gt_occlusion)
        gt_out_of_frame_recovered = covered_resultion(gt_out_of_frame)
        
        print(left_image_recovered.shape)
        print(right_image_recovered.shape)
        print(gt_disp_recovred.shape)
        print(gt_occlusion_recovred.shape)
        print(gt_out_of_frame_recovered.shape)
        
        print("*********************************************************")  
        pass

    
#     # # # test the trainloader
#     for sample in tqdm(train_loader):
        
#         left_image = sample['img_left']
#         right_image = sample['img_right']
        
#         print(left_image.mean())
#     #     # left_left_image = sample['img_left_left']
#     #     # right_right_image = sample['img_right_right']
#     #     # center_image = sample['img_center']
#     #     # quality = sample['img_quality']
        
        
        
#     #     # left_image = Image_DeNormalization(left_image)
#     #     # right_image = Image_DeNormalization(right_image)
#     #     # left_left_image = Image_DeNormalization(left_left_image)
#     #     # right_right_image = Image_DeNormalization(right_right_image)
#     #     # center_image = Image_DeNormalization(center_image)
        
        
#     #     # left_image_vis =  Convert_IMGTensor_To_Numpy(left_image)
#     #     # right_image_vis = Convert_IMGTensor_To_Numpy(right_image)
#     #     # left_left_image_vis = Convert_IMGTensor_To_Numpy(left_left_image)
#     #     # right_right_image_vis = Convert_IMGTensor_To_Numpy(right_right_image)
#     #     # center_image_vis = Convert_IMGTensor_To_Numpy(center_image)
        
        
#     #     # plt.figure(figsize=(15,1))
#     #     # plt.subplot(1,5,1)
#     #     # plt.axis('off')
#     #     # plt.title("left-left-image")
#     #     # plt.imshow(left_left_image_vis)
#     #     # plt.subplot(1,5,2)
#     #     # plt.axis("off")
#     #     # plt.title("left-image")
#     #     # plt.imshow(left_image_vis)
#     #     # plt.subplot(1,5,3)
#     #     # plt.axis("off")
#     #     # plt.title("center-image")
#     #     # plt.imshow(center_image_vis)
#     #     # plt.subplot(1,5,4)
#     #     # plt.axis("off")
#     #     # plt.title("right-image")
#     #     # plt.imshow(right_image_vis)
#     #     # plt.subplot(1,5,5)
#     #     # plt.axis("off")
#     #     # plt.title("right-right-image")
#     #     # plt.imshow(right_right_image_vis)
#     #     # plt.show()
#     #     # plt.savefig("dataset.png")
        
#     #     # quit()
   
        
        

#     # # #test the valloader
#     # for sample in tqdm(test_loader):
#     #     left_image = sample['img_left']
#     #     right_image = sample['img_right']
#     #     gt_disp = sample['gt_disp']
        
  

        
    
        

#     print("all files is ok")
    
    