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
    
    original_width, original_height = img_tensor[2:]
    
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

            
        original_H, original_W = left_image_tensors.shape[:2]
        
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
        
        
        quit()
        
        
        batched_image = batched_image.cuda()
        depth_pred_raw_left,depth_pred_raw_mid,depth_pred_raw_right  = self.single_infer(
            input_rgb=batched_image,
            num_inference_steps=denosing_steps,
            show_pbar=show_progress_bar,
        )
        
        depth_pred_left = depth_pred_raw_left
        depth_pred_mid = depth_pred_raw_mid
        depth_pred_right = depth_pred_raw_right

        
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        depth_pred_left = depth_pred_left.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        depth_pred_mid = depth_pred_mid.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        depth_pred_right = depth_pred_right.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        

        # Resize back to original resolution
        if match_input_res:
            # print(depth_pred.shape)
            depth_pred_left = cv2.resize(depth_pred_left,input_size)
            depth_pred_mid = cv2.resize(depth_pred_mid,input_size)
            depth_pred_right = cv2.resize(depth_pred_right,input_size)

        # Clip output range: current size is the original size
        depth_pred_left = depth_pred_left.clip(0, 1)
        depth_pred_mid = depth_pred_mid.clip(0, 1)
        depth_pred_right = depth_pred_right.clip(0, 1)
        
    
        return depth_pred_left,depth_pred_mid,depth_pred_right
        
    
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

        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,):
        
        
        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. : this is the prompt
        # given the baseline prompt.
        standard_ones =  torch.ones_like(rgb_latent).type_as(rgb_latent)[:,:1,:,:]
        
        # batch 0 ----> left to left.
        # batch 1 ----> left to center.
        # batch 2 ----> left to right.
        baseline_prompt_zeros = standard_ones * 0.0
        baseline_prompt_ones = standard_ones * -0.54
        baseline_prompt_twos = standard_ones * 0.54
        
        prompt_zeros = torch.cat((rgb_latent,baseline_prompt_zeros),dim=1)
        prompt_ones = torch.cat((rgb_latent,baseline_prompt_ones),dim=1)
        prompt_twos = torch.cat((rgb_latent,baseline_prompt_twos),dim=1)
            
        prompts = torch.cat((prompt_zeros,prompt_ones,prompt_twos),dim=0) # [3B,5,H,W]
        cur_batch_size = prompts.shape[0]
        
        # Initial depth map (Guassian noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
        depth_latent = depth_latent.repeat(cur_batch_size,1,1,1)
        
        self.__encode_empty_text()
            
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]    
        batch_empty_text_embed = batch_empty_text_embed.repeat(cur_batch_size,1,1)
        
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
        
        
        assert depth_latent.shape[0]==3
        # decode to left
        depth_latent_zeros = depth_latent[:1,:,:,:]
        depth_left = self.decode_depth(depth_latent_zeros)
        depth_left= torch.clip(depth_left, -1.0, 1.0)
        # shift to [0, 1]
        depth_left = (depth_left + 1.0) / 2.0  #left
        
        # decode to center
        depth_latent_ones = depth_latent[1:2,:,:,:]
        depth_center = self.decode_depth(depth_latent_ones)
        depth_center= torch.clip(depth_center, -1.0, 1.0)
        # shift to [0, 1]
        depth_center = (depth_center + 1.0) / 2.0 #left-left
        
        
        # decode to right
        depth_latent_twos = depth_latent[2:,:,:,:]
        depth_right = self.decode_depth(depth_latent_twos)
        depth_right= torch.clip(depth_right, -1.0, 1.0)
        # shift to [0, 1]
        depth_right = (depth_right + 1.0) / 2.0 #right
        
        
        return [depth_left,depth_center,depth_right]
        
    
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
