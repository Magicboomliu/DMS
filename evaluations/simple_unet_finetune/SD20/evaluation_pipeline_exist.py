from typing import Any, Dict, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from utils.colormap import kitti_colormap
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

def resize_max_res_tensor(img_tensor, max_edge_resolution: int):
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    
    original_height, original_width = img_tensor.shape[2:]
    
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    output_tensor = F.interpolate(img_tensor,size=[new_height,new_width],mode='bilinear',align_corners=False)
    
    return output_tensor




class SD20UNet_Validation_Pipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
        
        
    @torch.no_grad()
    def __call__(self,
                 left_images_tensor=None,
                 right_images_tensor=None,
                 denosing_steps: int =10,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 show_progress_bar:bool = True):
        
        # inherit from thea Diffusion Pipeline
        device = self.device


        original_H, original_W = left_images_tensor.shape[-2:]
        
        # Resize image
        if processing_res >0:
            left_image_tensors = resize_max_res_tensor(left_images_tensor,
                                                       max_edge_resolution=processing_res)
            right_images_tensors = resize_max_res_tensor(right_images_tensor,
                                                         max_edge_resolution=processing_res)
            left_image_tensors = left_image_tensors.to(device)
            right_images_tensors = right_images_tensors.to(device)
        
        
        left_image_tensors = (left_image_tensors - 0.5) * 2.0
        right_images_tensors = (right_images_tensors - 0.5)* 2.0
        
        # first batch size dimension is left image
        # second batch size dimension is the right image
        
        reference_image_tensor = torch.cat((left_image_tensors,right_images_tensors),dim=0)
        reference_image_tensor = reference_image_tensor.cuda() #[8,3,H//8,W//8]
        
        rendered_left_from_right,rendered_right_from_left= self.single_infer(
            input_rgb=reference_image_tensor,
            num_inference_steps=denosing_steps,
            show_pbar=show_progress_bar,
        )
            
        if match_input_res:
            rendered_left_from_right= F.interpolate(rendered_left_from_right,size=[original_H,original_W],
                                                         mode='bilinear',align_corners=False)

            rendered_right_from_left = F.interpolate(rendered_right_from_left,size=[original_H,original_W],
                                                         mode='bilinear',align_corners=False)

        

        # idx = 2
        # plt.figure(figsize=(10,4))
        # plt.subplot(1,2,1)
        # plt.axis("off")
        # plt.title("left(right)")
        # plt.imshow(rendered_left_from_right[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.axis("off")
        # plt.title("right(left)")
        # plt.imshow(rendered_right_from_left[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.show()
    
        # quit()
        
        # idx = 2
        # plt.figure(figsize=(10,4))
        # plt.subplot(2,3,1)
        # plt.axis("off")
        # plt.title("left-left(left)")
        # plt.imshow(rendered_left_left_from_left[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(2,3,2)
        # plt.axis("off")
        # plt.title("left(left)")
        # plt.imshow(rendered_left_from_left[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(2,3,3)
        # plt.axis("off")
        # plt.title("right(left)")
        # plt.imshow(rendered_right_from_left[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(2,3,4)
        # plt.axis("off")
        # plt.title("left(right)")
        # plt.imshow(rendered_left_from_right[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(2,3,5)
        # plt.axis("off")
        # plt.title("right(right)")
        # plt.imshow(rendered_right_from_right[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.subplot(2,3,6)
        # plt.axis("off")
        # plt.title("right(right)")
        # plt.imshow(rendered_right_right_from_right[idx-1:idx,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.show()
    
    
    
        return   rendered_left_from_right,rendered_right_from_left
        
    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        # print(text_input_ids.shape)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]

    
    def _decode_and_denorm(self,output):

        output = self.decode_depth(output)
        output= torch.clip(output, -1.0, 1.0)
        # shift to [0, 1]
        output = (output + 1.0) / 2.0  #left
        return  output
    
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,):
        
        
        device = input_rgb.device
        mini_batch_size = input_rgb.shape[0] 

        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        # encode image
        
        old_h,old_w = input_rgb.shape[2:]
        
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. : this is the prompt
        #[8,4,H,W]
        
        # given the baseline prompt.
        standard_ones =  torch.ones_like(rgb_latent).type_as(rgb_latent)[:4,:1,:,:] #[8,1,H,W]
        
        
        baseline_prompt_index_negative = standard_ones * -0.54 # from right to left.
        baseline_prompt_index_positive = standard_ones * 0.54  # from the left to right.
        
        baseline_prompt = torch.cat((baseline_prompt_index_positive,baseline_prompt_index_negative),
                                    dim=0) #[8,1,H,W]
        prompts = torch.cat((rgb_latent,baseline_prompt),dim=1)
        


        # Initial depth map (Guassian noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
        

        self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]
        
        
    
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            unet_input = torch.cat(
                [prompts, depth_latent], dim=1)  # this order is important: [1,8,H,W]
            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        
        torch.cuda.empty_cache()
        
        
        batch_size = depth_latent.shape[0]
        
        rendered_right_from_left = depth_latent[:int(batch_size//2),:,:,:]
        rendered_left_from_right = depth_latent[int(batch_size//2):,:,:,:]
        
        rendered_right_from_left = self._decode_and_denorm(rendered_right_from_left)
        rendered_left_from_right = self._decode_and_denorm(rendered_left_from_right)
        
    
        
        
        
        return rendered_left_from_right,rendered_right_from_left
    
    

    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:

        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.
        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        depth_latent = depth_latent.cuda()

        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        return stacked
