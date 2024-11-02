from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../..")
from dataloader.carla_dataloader.carla_io import  read_img
from dataloader.carla_dataloader.utils import read_text_lines
import numpy as np
import cv2
from scipy.ndimage import zoom


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


def resize_image(image, target_resolution):
    # Ensure target_resolution is a tuple (width, height)
    resized_image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_LINEAR)
    return resized_image


def resize_depth(image,target_resolution):
    
    resized_image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_LINEAR)
    
    return resized_image


class CARLA_MV_Dataset(Dataset):
    def __init__(self,
                 datapath,
                 trainlist,
                 vallist,
                 mode='train',
                 transform=None,
                 targetHW = (1920,1080),
                 use_depth = True,
                 save_filename=False):
        
        super(CARLA_MV_Dataset,self).__init__()
        
        # initialization
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        self.mode = mode
        self.transform = transform
        self.save_filename = save_filename
        self.train_resolution = targetHW
        
        self.use_depth = use_depth
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            "test": self.vallist
        }
        

        
        self.samples  =[]
        lines = read_text_lines(dataset_dict[mode])

        for line in lines:
            splits = line.split()
            left_filename = splits[0]
            
            full_left_filename = os.path.join(self.datapath,left_filename)
            full_right_filename = full_left_filename.replace("Left_RGB","Right_RGB")
            full_mid_filename = full_left_filename.replace("Left_RGB","Middle_RGB")
            
            full_depth_left_filename = full_left_filename.replace("Left_RGB","Left_RawDepth").replace(".png",".npy")
            full_depth_mid_filename = full_left_filename.replace("Left_RGB","Middle_RawDepth").replace(".png",".npy")
            full_depth_right_filename = full_left_filename.replace("Left_RGB","Right_RawDepth").replace(".png",".npy")
            

            assert os.path.exists(full_left_filename)
            assert os.path.exists(full_right_filename)
            assert os.path.exists(full_mid_filename)
            assert os.path.exists(full_depth_left_filename)
            assert os.path.exists(full_depth_mid_filename)
            assert os.path.exists(full_depth_right_filename)


            sample = dict()
            sample['left_image_path'] = full_left_filename
            sample['mid_image_path'] = full_mid_filename
            sample['right_image_path'] = full_right_filename
            
            sample['depth_left_path'] = full_depth_left_filename
            sample['depth_right_path'] = full_depth_right_filename
            sample['depth_mid_path'] = full_depth_mid_filename
            
            
            self.samples.append(sample)

                
        
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]

        # read the information
        sample['left_image'] = read_img(sample_path['left_image_path']) #(1080,1920)
        sample['right_image'] = read_img(sample_path['right_image_path'])
        sample['mid_image'] = read_img(sample_path['mid_image_path'])
        
        original_width = sample['left_image'].shape[1]

        if self.train_resolution is not None:
            sample['left_image'] = resize_image(sample['left_image'],target_resolution=self.train_resolution)
            sample['right_image'] = resize_image(sample['right_image'],target_resolution=self.train_resolution)
            sample['mid_image'] = resize_image(sample['mid_image'],target_resolution=self.train_resolution)

        # read the depth
        if self.use_depth:
            sample['left_depth'] = np.load(sample_path['depth_left_path'])
            sample['right_depth'] = np.load(sample_path['depth_right_path'])
            sample['mid_depth'] = np.load(sample_path['depth_mid_path'])

            if self.train_resolution is not None:
                sample['left_depth'] = resize_depth(sample['left_depth'],target_resolution=self.train_resolution)
                sample['right_depth'] = resize_depth(sample['right_depth'],target_resolution=self.train_resolution)
                sample['mid_depth'] = resize_depth(sample['mid_depth'],target_resolution=self.train_resolution)
            
        
        fov = 90        # default setting for carla
        focal_length = 1920 / (2.0 * np.tan(fov * np.pi / 360.0))
        baseline_left_to_right = 2.0       # in terms of meter
        baseline_left_to_middle = 1.0
        baseline_right_to_middle = 1.0
        
        if self.train_resolution:
            resize_ratio = self.train_resolution[0]*1.0/original_width
            focal_length = focal_length * resize_ratio
        
        sample['fx'] = focal_length
        sample['baseline_left2right'] = baseline_left_to_right
        sample['baseline_left2mid'] = baseline_left_to_middle
        sample['baseline_mid2right'] = baseline_right_to_middle
            
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
    def __len__(self):
        return len(self.samples)





if __name__=="__main__":
    
    datapath = "/media/zliu/data12/dataset/CARLA/"
    trainlist = "/home/zliu/CVPR25_Detection/DiffusionMultiBaseline/datafiles/CARLA/training.txt"
    vallist = "/home/zliu/CVPR25_Detection/DiffusionMultiBaseline/datafiles/CARLA/testing.txt"
    
    
    
    carla_dataset = CARLA_MV_Dataset(datapath=datapath,
                     trainlist=trainlist,
                     vallist=vallist,
                     mode='train',
                     transform=None,
                     targetHW=(1920,1080),
                     use_depth=True)
    
    for idx, sample in enumerate(carla_dataset):
        print(sample['left_image'].shape)
        print(sample['right_image'].shape)
        print(sample['mid_image'].shape)
        
        print(sample['left_depth'].shape)
        print(sample['right_depth'].shape)
        print(sample['mid_depth'].shape)
        
        print(sample['fx'])
        
        quit()