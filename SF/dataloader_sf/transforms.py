from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):        
        if 'left' in sample.keys():
            left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
            sample['left'] = torch.from_numpy(left) / 255.
        if 'right' in sample.keys():
            right = np.transpose(sample['right'], (2, 0, 1))  # [3, H, W]
            sample['right'] = torch.from_numpy(right) / 255.
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)
        if 'occlusions' in sample.keys():
            occlusion = sample['occlusions']  # [H, W]
            sample['occlusions'] = torch.from_numpy(occlusion)
        if 'left_left' in sample.keys():
            left_left = np.transpose(sample['left_left'], (2, 0, 1))  # [3, H, W]
            sample['left_left'] = torch.from_numpy(left_left) / 255.
        if 'right_right' in sample.keys():
            right_right = np.transpose(sample['right_right'], (2, 0, 1))  # [3, H, W]
            sample['right_right'] = torch.from_numpy(right_right) / 255.
        if 'center' in sample.keys():
            center = np.transpose(sample['center'], (2, 0, 1))  # [3, H, W]
            sample['center'] = torch.from_numpy(center) / 255.
            
        return sample


class Normalize(object):
    """Normalize image, with type tensor"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        
        norm_keys = []
        
        if 'left' in sample.keys():
            norm_keys.append('left')
        if 'right' in sample.keys():
            norm_keys.append('right')
        if 'left_left' in sample.keys():
            norm_keys.append("left_left")
        if "right_right" in sample.keys():
            norm_keys.append("right_right")
        if 'center' in sample.keys():
            norm_keys.append("center")

    
        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            if 'left' in sample.keys():
                sample['left'] = np.lib.pad(sample['left'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            if 'right' in sample.keys():
                sample['right'] = np.lib.pad(sample['right'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            if 'left_left' in sample.keys():
                sample['left_left'] = np.lib.pad(sample['left_left'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            if 'right_right' in sample.keys():
                sample['right_right'] = np.lib.pad(sample['right_right'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            if 'center' in sample.keys():
                sample['center'] = np.lib.pad(sample['center'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            
            if 'disp' in sample.keys():
                sample['disp'] = np.lib.pad(sample['disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)



        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2
                
            if 'left' in sample.keys():
                sample['left'] = self.crop_img(sample['left'])
            if 'right' in sample.keys():
                sample['right'] = self.crop_img(sample['right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            
            if 'left_left' in sample.keys():
                sample['left_left'] = self.crop_img(sample['left_left'])
            if 'right_right' in sample.keys():
                sample['right_right'] = self.crop_img(sample['right_right'])
            if 'center' in sample.keys():
                sample['center'] = self.crop_img(sample['center'])


        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]



class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            if 'left' in sample.keys():
                sample['left'] = np.copy(np.flipud(sample['left']))
            if 'right' in sample.keys():
                sample['right'] = np.copy(np.flipud(sample['right']))
            if 'disp' in sample.keys():
                sample['disp'] = np.copy(np.flipud(sample['disp']))
            if 'left_left' in sample.keys():
                sample['left_left'] = np.copy(np.flipud(sample['left_left']))
            if 'right_right' in sample.keys():
                sample['right_right'] = np.copy(np.flipud(sample['right_right']))
            if 'center' in sample.keys():
                sample['center'] = np.copy(np.flipud(sample['center']))
        
        return sample


class ToPILImage(object):

    def __call__(self, sample):
        
        if 'left' in sample.keys():
            sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        if 'right' in sample.keys():
            sample['right'] = Image.fromarray(sample['right'].astype('uint8'))
        if 'left_left' in sample.keys():
            sample['left_left'] = Image.fromarray(sample['left_left'].astype('uint8'))
        if 'right_right' in sample.keys():
            sample['right_right'] = Image.fromarray(sample['right_right'].astype('uint8'))
        if 'center' in sample.keys():
            sample['center'] = Image.fromarray(sample['center'].astype('uint8'))

        return sample


class ToNumpyArray(object):
    def __call__(self, sample):
        
        if 'left' in sample.keys():
            sample['left'] = np.array(sample['left']).astype(np.float32)
        if 'right' in sample.keys():
            sample['right'] = np.array(sample['right']).astype(np.float32)
        
        if 'left_left' in sample.keys():
            sample['left_left'] = np.array(sample['left_left']).astype(np.float32)
        if 'right_right' in sample.keys():
            sample['right_right'] = np.array(sample['right_right']).astype(np.float32)
        if 'center' in sample.keys():
            sample['center'] = np.array(sample['center']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            if 'left' in sample.keys():
                sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
            if 'right' in sample.keys():
                sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)
            if 'left_left' in sample.keys():
                sample['left_left'] = F.adjust_contrast(sample['left_left'], contrast_factor)
            if 'right_right' in sample.keys():
                sample['right_right'] = F.adjust_contrast(sample['right_right'], contrast_factor)
            if 'center' in sample.keys():
                sample['center'] = F.adjust_contrast(sample['center'], contrast_factor)
            

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)  # adopted from FlowNet
            
            if 'left' in sample.keys():
                sample['left'] = F.adjust_gamma(sample['left'], gamma)
            
            if 'right' in sample.keys():
                sample['right'] = F.adjust_gamma(sample['right'], gamma)
                
            if 'left_left' in sample.keys():
                sample['left_left'] = F.adjust_gamma(sample['left_left'], gamma)
                
            if 'right_right' in sample.keys():
                sample['right_right'] = F.adjust_gamma(sample['right_right'], gamma)
                
            if 'center' in sample.keys():
                sample['center'] = F.adjust_gamma(sample['center'], gamma)


        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            
            if 'left' in sample.keys():
                sample['left'] = F.adjust_brightness(sample['left'], brightness)
            if 'right' in sample.keys():
                sample['right'] = F.adjust_brightness(sample['right'], brightness)
            if 'left_left' in sample.keys():
                sample['left_left'] = F.adjust_brightness(sample['left_left'], brightness)
            if 'right_right' in sample.keys():
                sample['right_right'] = F.adjust_brightness(sample['right_right'], brightness)
            if 'center' in sample.keys():
                sample['center'] = F.adjust_brightness(sample['center'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            
            if 'left' in sample.keys():
                sample['left'] = F.adjust_hue(sample['left'], hue)
            
            if 'right' in sample.keys():
                sample['right'] = F.adjust_hue(sample['right'], hue)
                
            if 'left_left' in sample.keys():
                sample['left_left'] = F.adjust_hue(sample['left_left'], hue)
            if 'right_right' in sample.keys():
                sample['right_right'] = F.adjust_hue(sample['right_right'], hue)
            if 'center' in sample.keys():
                sample['center'] = F.adjust_hue(sample['center'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            if 'left' in sample.keys():
                sample['left'] = F.adjust_saturation(sample['left'], saturation)
            
            if 'right' in sample.keys():
                sample['right'] = F.adjust_saturation(sample['right'], saturation)
            if 'left_left' in sample.keys():
                sample['left_left'] = F.adjust_saturation(sample['left_left'], saturation)
            if 'right_right' in sample.keys():
                sample['right_right'] = F.adjust_saturation(sample['right_right'], saturation)
            if 'center' in sample.keys():
                sample['center'] = F.adjust_saturation(sample['center'], saturation)


        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample