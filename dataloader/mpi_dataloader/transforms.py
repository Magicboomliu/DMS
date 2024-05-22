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

        
        if 'clean_left' in sample.keys():
            clean_left = np.transpose(sample['clean_left'], (2, 0, 1))  # [3, H, W]
            sample['clean_left'] = torch.from_numpy(clean_left) / 255.
        
        if 'clean_right' in sample.keys():
            clean_right = np.transpose(sample['clean_right'], (2, 0, 1))  # [3, H, W]
            sample['clean_right'] = torch.from_numpy(clean_right) / 255.
        
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)
        
        if 'final_left' in sample.keys():
            final_left = np.transpose(sample['final_left'], (2, 0, 1))  # [3, H, W]
            sample['final_left'] = torch.from_numpy(final_left) / 255.
        
        if 'final_right' in sample.keys():
            final_right = np.transpose(sample['final_right'], (2, 0, 1))  # [3, H, W]
            sample['final_right'] = torch.from_numpy(final_right) / 255.
        
        if 'occlusions' in sample.keys():
            occlusion = sample['occlusions']  # [H, W]
            sample['occlusions'] = torch.from_numpy(occlusion) /255.
        
        if "outofframe" in sample.keys():
            outofframe = sample['outofframe']  # [H, W]
            sample['outofframe'] = torch.from_numpy(outofframe) /255.
        
        
            
        return sample


class Normalize(object):
    """Normalize image, with type tensor"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        
        norm_keys = []
        
        if 'clean_left' in sample.keys():
            norm_keys.append('clean_left')
        if 'clean_right' in sample.keys():
            norm_keys.append('clean_right')
        if 'final_left' in sample.keys():
            norm_keys.append('final_left')
        if 'final_right' in sample.keys():
            norm_keys.append('final_right')
        
    
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
        ori_height, ori_width = sample['clean_left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            if 'clean_left' in sample.keys():
                sample['clean_left'] = np.lib.pad(sample['clean_left'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = np.lib.pad(sample['clean_right'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)

            if 'final_left'  in sample.keys():    
                sample['final_left'] = np.lib.pad(sample['final_left'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
                                            mode='constant',
                                            constant_values=0)
            if 'final_right' in sample.keys():
                sample['final_right'] = np.lib.pad(sample['final_right'],
                                            ((top_pad, 0), (0, right_pad), (0, 0)),
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
                
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = self.crop_img(sample['clean_left'])
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = self.crop_img(sample['clean_right'])

            if 'final_left' in sample.keys():
                sample['final_left'] = self.crop_img(sample['final_left'])
            
            
            if 'final_right' in sample.keys():
                sample['final_right'] = self.crop_img(sample['final_right'])


        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]



class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = np.copy(np.flipud(sample['clean_left']))
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = np.copy(np.flipud(sample['clean_right']))
            
            if 'final_left' in sample.keys():
                sample['final_left'] = np.copy(np.flipud(sample['final_left']))
            
            if 'final_right' in sample.keys():
                sample['final_right'] = np.copy(np.flipud(sample['final_right']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        
        if 'clean_left' in sample.keys():
            sample['clean_left'] = Image.fromarray(sample['clean_left'].astype('uint8'))
        if 'clean_right' in sample.keys():
            sample['clean_right'] = Image.fromarray(sample['clean_right'].astype('uint8'))
        if 'final_left' in sample.keys():
            sample['final_left'] = Image.fromarray(sample['final_left'].astype('uint8'))
        if 'final_right' in sample.keys():
            sample['final_right'] = Image.fromarray(sample['final_right'].astype('uint8'))

        return sample


class ToNumpyArray(object):
    def __call__(self, sample):
        
        if 'clean_left' in sample.keys():
            sample['clean_left'] = np.array(sample['clean_left']).astype(np.float32)
        if 'clean_right' in sample.keys():
            sample['clean_right'] = np.array(sample['clean_right']).astype(np.float32)
        
        if 'final_left' in sample.keys():
            sample['final_left'] = np.array(sample['final_left']).astype(np.float32)
        
        if 'final_right' in sample.keys():
            sample['final_right'] = np.array(sample['final_right']).astype(np.float32)

        return sample

# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = F.adjust_contrast(sample['clean_left'], contrast_factor)
            if 'clean_right' in sample.keys():
                sample['clean_right'] = F.adjust_contrast(sample['clean_right'], contrast_factor)
            if 'final_left' in sample.keys():
                sample['final_left'] = F.adjust_contrast(sample['final_left'], contrast_factor)
            
            if 'final_right' in sample.keys():
                sample['final_right'] = F.adjust_contrast(sample['final_right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)  # adopted from FlowNet
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = F.adjust_gamma(sample['clean_left'], gamma)
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = F.adjust_gamma(sample['clean_right'], gamma)

            if 'final_left' in sample.keys():
                sample['final_left'] = F.adjust_gamma(sample['final_left'], gamma)
            
            if 'final_right' in sample.keys():
                sample['final_right'] = F.adjust_gamma(sample['final_right'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = F.adjust_brightness(sample['clean_left'], brightness)
            if 'clean_right' in sample.keys():
                sample['clean_right'] = F.adjust_brightness(sample['clean_right'], brightness)
            if 'final_left' in sample.keys():
                sample['final_left'] = F.adjust_brightness(sample['final_left'], brightness)
            if 'final_right' in sample.keys():
                sample['final_right'] = F.adjust_brightness(sample['final_right'], brightness)


        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            
            if 'clean_left' in sample.keys():
                sample['clean_left'] = F.adjust_hue(sample['clean_left'], hue)
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = F.adjust_hue(sample['clean_right'], hue)

            if 'final_left' in sample.keys():
                sample['final_left'] = F.adjust_hue(sample['final_left'], hue)
            
            if 'final_right' in sample.keys():
                sample['final_right'] = F.adjust_hue(sample['final_right'], hue)


        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            if 'clean_left' in sample.keys():
                sample['clean_left'] = F.adjust_saturation(sample['clean_left'], saturation)
            
            if 'clean_right' in sample.keys():
                sample['clean_right'] = F.adjust_saturation(sample['clean_right'], saturation)

            if 'final_left' in sample.keys():
                sample['final_left'] = F.adjust_saturation(sample['final_left'], saturation)
            
            if 'final_right' in sample.keys():
                sample['final_right'] = F.adjust_saturation(sample['final_right'], saturation)
        
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