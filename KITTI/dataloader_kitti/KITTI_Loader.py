from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from utils.kitti_io import read_img, read_disp,read_kitti_step1,read_kitti_step2,read_kitti_image_step1,read_kitti_image_step2
from skimage import io, transform
import numpy as np
from PIL import Image


def compute_left_occ_region(w, disp):
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord - disp
    occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ
    return occ_mask

def image_pad(image,targetHW):
    H,W = image.shape[:2]
    new_H, new_W = targetHW
    pad_bottom = new_H - H  
    pad_right = new_W - W 
    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
    return added_image, np.array(image.shape[:2])

def disp_pad(image,targetHW):
    H,W = image.shape[:2]
    new_H, new_W = targetHW
    pad_bottom = new_H - H  
    pad_right = new_W - W 
    added_image = np.pad(image, ((0, pad_bottom), (0, pad_right)), 'constant', constant_values=0)
    return added_image, np.array(image.shape[:2])

class KITTI_Multi_Baseline_Dataset(Dataset):
    def __init__(self, 
                 datapath,
                 train_datalist,
                 test_datalist,
                 mode='train',
                 transform=None,
                 visible_list = ["left",
                                 "right",
                                 "left_left",
                                 "right_right",
                                 "center",
                                 "confidence",
                                 "occlusion",
                                 "oof"]):
        super(KITTI_Multi_Baseline_Dataset, self).__init__()
        
        self.datapath = datapath
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.mode = mode
        self.transform = transform
        self.visible_list = visible_list
        self.img_size=(384, 1248)
        self.scale_size =(384,1248)
        
        dataset_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }
        self.samples = []
        data_filenames = dataset_dict[mode]
        lines = utils.read_text_lines(data_filenames)


        for line in lines:
            sample = dict()
            if self.mode=='train':
                left_image = line
                if 'left' in self.visible_list:
                    sample['left'] = os.path.join(self.datapath,left_image)
                if 'right' in self.visible_list:
                    sample['right'] = sample['left'].replace("image_02","image_03")
                if 'left_left' in self.visible_list:
                    rendered_left_left = sample['left']
                    rendered_left_left = rendered_left_left.replace("image_02","image_01")
                    rendered_left_left = rendered_left_left.replace("KITTI_Raw","KITTI_Train")
                    sample['left_left'] = rendered_left_left
                if "right_right" in self.visible_list:
                    rendered_right_right = sample['left']
                    rendered_right_right = rendered_left_left.replace("image_01","image_04")
                    rendered_right_right = rendered_right_right.replace("KITTI_Raw","KITTI_Train")
                    sample['right_right'] = rendered_right_right
                if 'center' in self.visible_list:
                    rendered_center = sample['left']
                    rendered_center = rendered_center.replace("image_02","image_025")
                    rendered_center = rendered_center.replace("KITTI_Raw","KITTI_Train")
                    sample['center'] = rendered_center
                if "confidence" in self.visible_list:
                    # TODO
                    pass

                if "occlusion" in self.visible_list:
                    # TODO
                    pass
                if "oof" in self.visible_list:
                    # TODO
                    pass
                
                self.samples.append(sample)

            elif self.mode=='val' or self.mode=='test':
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]
                disp_path = splits[2]
                sample['left'] = os.path.join(self.datapath,left_image)
                sample['right'] = os.path.join(self.datapath,right_image)
                sample['disp'] = os.path.join(self.datapath,disp_path)
                
                self.samples.append(sample)
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        # read the kitti raw dataset
        if self.mode=='train':
            if 'left' in self.visible_list: 
                sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
                
                sample['img_left'], sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                
            if 'right' in self.visible_list:
                sample['img_right'] = read_img(sample_path['right']) # [H,W,3]
                
                sample['img_right'], sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
            if 'left_left' in self.visible_list:
                sample['img_left_left'] = read_img(sample_path['left_left']) # [H,W,3]
                
                sample['img_left_left'], sample['resolution'] = image_pad(sample['img_left_left'],targetHW=self.scale_size)
                
            if 'right_right' in self.visible_list:
                sample['img_right_right'] = read_img(sample_path['right_right']) # [H,W,3]
                sample['img_right_right'], sample['resolution'] = image_pad(sample['img_right_right'],targetHW=self.scale_size)
            if 'center' in self.visible_list:
                sample['img_center'] = read_img(sample_path['center']) # [H,W,3]
                sample['img_center'], sample['resolution'] = image_pad(sample['img_center'],targetHW=self.scale_size)
            if 'confidence' in self.visible_list:
                sample['img_quality'] = np.loadtxt(sample_path['quality'],dtype=float)
            if "" in self.visible_list:
                # TODO
                pass
            if "" in self.visible_list:
                # TODO
                pass
            

        elif self.mode=='val' or self.mode=='test':
            # print(sample_path['left'])
            if sample_path['disp'] is not None:
                # Image Crop Operation
                left_im = read_kitti_image_step1(sample_path['left']) #[H,W,3]
                right_im = read_kitti_image_step1(sample_path['right'])        
                w, h = left_im.size
                left_image = left_im
                right_image = right_im
                sample['img_left'] = read_kitti_image_step2(left_image)
                sample['img_right'] = read_kitti_image_step2(right_image)
                
                # sample['img_left'],sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                # sample['img_right'],sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
                
                w1,h1 = left_image.size
                # Disparity Crop Operation
                if sample_path['disp'] is not None:
                    gt_disp = read_kitti_step1(sample_path['disp'])
                    w, h = gt_disp.size
                    dataL = gt_disp
                    dataL = read_kitti_step2(dataL)
                    sample['gt_disp']= dataL
                    
                    # sample['gt_disp'],sample['resolution'] = disp_pad(sample['gt_disp'],targetHW=self.scale_size)
                    
                           
        else:
            raise NotImplementedError
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

        
class KITTI_2015_2012_MB_Dataset(Dataset):
    def __init__(self, 
                 datapath,
                 train_datalist,
                 test_datalist,
                 mode='train',
                 transform=None,
                 visible_list = ["left",
                                 "right",
                                 "left_left",
                                 "right_right",
                                 "center",
                                 "confidence",
                                 "occlusion",
                                 "oof"]):
        super(KITTI_2015_2012_MB_Dataset, self).__init__()
        
        self.datapath = datapath
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.mode = mode
        self.transform = transform
        self.visible_list = visible_list
        self.img_size=(384, 1248)
        self.scale_size =(384,1248)
        
        dataset_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }
        self.samples = []
        data_filenames = dataset_dict[mode]
        lines = utils.read_text_lines(data_filenames)


        for line in lines:
            sample = dict()
            if self.mode=='train':
                
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]

                if 'left' in self.visible_list:
                    sample['left'] = os.path.join(self.datapath,left_image)
                if 'right' in self.visible_list:
                    sample['right'] = os.path.join(self.datapath,right_image)
                if 'left_left' in self.visible_list:
                    sample['left_left']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['left_left'] = sample['left_left'].replace("image_2","image_1")
                if "right_right" in self.visible_list:
                    sample['right_right']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['right_right'] = sample['right_right'].replace("image_2","image_4")
                if 'center' in self.visible_list:
                    sample['center']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['center'] = sample['center'].replace("image_2","image_25")
                if "confidence" in self.visible_list:
                    pass
                if "occlusion" in self.visible_list:
                    pass
                if "oof" in self.visible_list:
                    pass
                
                self.samples.append(sample)

            elif self.mode=='val' or self.mode=='test':
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]
                disp_path = splits[2]
                sample['left'] = os.path.join(self.datapath,left_image)
                sample['right'] = os.path.join(self.datapath,right_image)
                sample['disp'] = os.path.join(self.datapath,disp_path)
                
                self.samples.append(sample)
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        # read the kitti raw dataset
        if self.mode=='train':
            if 'left' in self.visible_list: 
                sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
                sample['img_left'], sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                
            if 'right' in self.visible_list:
                sample['img_right'] = read_img(sample_path['right']) # [H,W,3]
                sample['img_right'], sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
            if 'left_left' in self.visible_list:
                sample['img_left_left'] = read_img(sample_path['left_left']) # [H,W,3]
                sample['img_left_left'], sample['resolution'] = image_pad(sample['img_left_left'],targetHW=self.scale_size)
                
            if 'right_right' in self.visible_list:
                sample['img_right_right'] = read_img(sample_path['right_right']) # [H,W,3]
                sample['img_right_right'], sample['resolution'] = image_pad(sample['img_right_right'],targetHW=self.scale_size)
            if 'center' in self.visible_list:
                sample['img_center'] = read_img(sample_path['center']) # [H,W,3]
                sample['img_center'], sample['resolution'] = image_pad(sample['img_center'],targetHW=self.scale_size)
            if 'confidence' in self.visible_list:
                sample['img_quality'] = np.loadtxt(sample_path['quality'],dtype=float)
            if "" in self.visible_list:
                # TODO
                pass
            if "" in self.visible_list:
                # TODO
                pass
            

        elif self.mode=='val' or self.mode=='test':
            # print(sample_path['left'])
            if sample_path['disp'] is not None:
                # Image Crop Operation
                left_im = read_kitti_image_step1(sample_path['left']) #[H,W,3]
                right_im = read_kitti_image_step1(sample_path['right'])        
                w, h = left_im.size
                left_image = left_im
                right_image = right_im
                sample['img_left'] = read_kitti_image_step2(left_image)
                sample['img_right'] = read_kitti_image_step2(right_image)
                
                # sample['img_left'],sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                # sample['img_right'],sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
                
                w1,h1 = left_image.size
                # Disparity Crop Operation
                if sample_path['disp'] is not None:
                    gt_disp = read_kitti_step1(sample_path['disp'])
                    w, h = gt_disp.size
                    dataL = gt_disp
                    dataL = read_kitti_step2(dataL)
                    sample['gt_disp']= dataL
                    
                    # sample['gt_disp'],sample['resolution'] = disp_pad(sample['gt_disp'],targetHW=self.scale_size)
                    
                           
        else:
            raise NotImplementedError
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size
    

    


class KITTI_2015_2012_MB_Dataset_WithDisp(Dataset):
    def __init__(self, 
                 datapath,
                 train_datalist,
                 test_datalist,
                 mode='train',
                 transform=None,
                 visible_list = ["left",
                                 "right",
                                 "left_left",
                                 "right_right",
                                 "center",
                                 "confidence",
                                 "occlusion",
                                 "oof",
                                 'disp']):
        super(KITTI_2015_2012_MB_Dataset_WithDisp, self).__init__()
        
        self.datapath = datapath
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.mode = mode
        self.transform = transform
        self.visible_list = visible_list
        self.img_size=(384, 1248)
        self.scale_size =(384,1248)
        
        dataset_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }
        self.samples = []
        data_filenames = dataset_dict[mode]
        lines = utils.read_text_lines(data_filenames)


        for line in lines:
            sample = dict()
            if self.mode=='train':
                
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]

                if 'left' in self.visible_list:
                    sample['left'] = os.path.join(self.datapath,left_image)
                if 'right' in self.visible_list:
                    sample['right'] = os.path.join(self.datapath,right_image)
                if 'left_left' in self.visible_list:
                    sample['left_left']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['left_left'] = sample['left_left'].replace("image_2","image_1")
                if "right_right" in self.visible_list:
                    sample['right_right']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['right_right'] = sample['right_right'].replace("image_2","image_4")
                if 'center' in self.visible_list:
                    sample['center']= sample['left'].replace("kitti_2015","KITTI2015_Additional_Views")
                    sample['center'] = sample['center'].replace("image_2","image_25")
                if "confidence" in self.visible_list:
                    pass
                if "occlusion" in self.visible_list:
                    pass
                if "oof" in self.visible_list:
                    pass
                if 'disp' in self.visible_list:
                    disp_path = splits[2]
                    sample['disp'] = os.path.join(self.datapath,disp_path)
                
                self.samples.append(sample)

            elif self.mode=='val' or self.mode=='test':
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]
                disp_path = splits[2]
                sample['left'] = os.path.join(self.datapath,left_image)
                sample['right'] = os.path.join(self.datapath,right_image)
                sample['disp'] = os.path.join(self.datapath,disp_path)
                
                self.samples.append(sample)
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        # read the kitti raw dataset
        if self.mode=='train':
            if 'left' in self.visible_list: 
                sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
                sample['img_left'], sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                
            if 'right' in self.visible_list:
                sample['img_right'] = read_img(sample_path['right']) # [H,W,3]
                sample['img_right'], sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
            if 'left_left' in self.visible_list:
                sample['img_left_left'] = read_img(sample_path['left_left']) # [H,W,3]
                sample['img_left_left'], sample['resolution'] = image_pad(sample['img_left_left'],targetHW=self.scale_size)
                
            if 'right_right' in self.visible_list:
                sample['img_right_right'] = read_img(sample_path['right_right']) # [H,W,3]
                sample['img_right_right'], sample['resolution'] = image_pad(sample['img_right_right'],targetHW=self.scale_size)
            if 'center' in self.visible_list:
                sample['img_center'] = read_img(sample_path['center']) # [H,W,3]
                sample['img_center'], sample['resolution'] = image_pad(sample['img_center'],targetHW=self.scale_size)
            if 'confidence' in self.visible_list:
                sample['img_quality'] = np.loadtxt(sample_path['quality'],dtype=float)
            if "" in self.visible_list:
                # TODO
                pass
            if "" in self.visible_list:
                # TODO
                pass
            if sample_path['disp'] is not None:
                gt_disp = read_kitti_step1(sample_path['disp'])
                w, h = gt_disp.size
                dataL = gt_disp
                dataL = read_kitti_step2(dataL)
                sample['gt_disp']= dataL
                sample['gt_disp'],_ = disp_pad(sample['gt_disp'],targetHW=(384,1248))


        elif self.mode=='val' or self.mode=='test':
            # print(sample_path['left'])
            if sample_path['disp'] is not None:
                # Image Crop Operation
                left_im = read_kitti_image_step1(sample_path['left']) #[H,W,3]
                right_im = read_kitti_image_step1(sample_path['right'])        
                w, h = left_im.size
                left_image = left_im
                right_image = right_im
                sample['img_left'] = read_kitti_image_step2(left_image)
                sample['img_right'] = read_kitti_image_step2(right_image)
                
                # sample['img_left'],sample['resolution'] = image_pad(sample['img_left'],targetHW=self.scale_size)
                # sample['img_right'],sample['resolution'] = image_pad(sample['img_right'],targetHW=self.scale_size)
                
                
                w1,h1 = left_image.size
                # Disparity Crop Operation
                if sample_path['disp'] is not None:
                    gt_disp = read_kitti_step1(sample_path['disp'])
                    w, h = gt_disp.size
                    dataL = gt_disp
                    dataL = read_kitti_step2(dataL)
                    sample['gt_disp']= dataL
               
                    
                    # sample['gt_disp'],sample['resolution'] = disp_pad(sample['gt_disp'],targetHW=self.scale_size)
                    
                           
        else:
            raise NotImplementedError
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size






  
        