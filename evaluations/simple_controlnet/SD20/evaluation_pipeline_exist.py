from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel,
    
)

from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps
from utils.colormap import kitti_colormap
import cv2


class SimpleControlNet_Pipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDPMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 controlnet: ControlNetModel,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            controlnet = controlnet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None

        
    def _resize_max_res(self,img: Image.Image, max_edge_resolution: int) -> Image.Image:
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
        
        original_width, original_height = img.size
        
        downscale_factor = min(
            max_edge_resolution / original_width, max_edge_resolution / original_height
        )

        # Calculate new dimensions based on downscale factor
        new_width = int(original_width * downscale_factor)
        new_height = int(original_height * downscale_factor)

        # Check if either dimension is less than max_edge_resolution, adjust if necessary
        if new_width < max_edge_resolution and new_height < max_edge_resolution:
            if new_width > new_height:
                new_width = max_edge_resolution
            else:
                new_height = max_edge_resolution
        
        if new_width==768:
            new_height = 231 
            
        resized_img = img.resize((new_width, new_height))
        return resized_img
    
    def _image_pre_procssing(self,input_image,processing_res,device):
        
        initial_size = input_image.size
        input_image = self._resize_max_res(
                input_image, max_edge_resolution=processing_res
            ) # resize image: for kitti is 231, 768
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0
        rgb_norm = (rgb_norm -0.5) * 2.0
        
        return rgb_norm,initial_size
    
    @torch.no_grad()
    def __call__(self,
                 input_left_images_list,
                 input_right_images_list,
                 denosing_steps: int =10,
                 ensemble_size: int =1,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 batch_size:int =0,
                 show_progress_bar:bool = True,
                 text_embed="to right",
                 cond=0,
                 ):
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denosing_steps >=1
        
        
        processed_images_list_left = []
        processed_images_list_right = []
        original_size_list = []
        
        for idx in range(len(input_left_images_list)):
            
            cur_image_left = input_left_images_list[idx]
            processed_image_left,orginal_size = self._image_pre_procssing(cur_image_left,
                                                                     processing_res=processing_res,
                                                                     device=device)
            cur_image_right = input_right_images_list[idx]
            processed_image_right,orginal_size = self._image_pre_procssing(cur_image_right,
                                                                     processing_res=processing_res,
                                                                     device=device)
                
            processed_images_list_left.append(processed_image_left.unsqueeze(0))
            processed_images_list_right.append(processed_image_right.unsqueeze(0))
            original_size_list.append(orginal_size)
            
            
        left_images_batch = torch.cat(processed_images_list_left,dim=0)
        right_images_batch = torch.cat(processed_images_list_right,dim=0)
            
        current_batch_size = left_images_batch.shape[0]
        
        left_right_concated = torch.cat((left_images_batch,right_images_batch),dim=0)

        rendered_right_from_left_images_list,rendered_left_from_right_images_list = self.single_infer(
                                                                        input_rgb=left_right_concated,
                                                                        num_inference_steps=denosing_steps,
                                                                        show_pbar=show_progress_bar,
                                                                        text_embed=text_embed,
                                                                        cond=cond)

        
        quit()
        
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        depth_pred_left = depth_pred_raw_left.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
 
        
        # Resize back to original resolution
        if match_input_res:
            # print(depth_pred.shape)
            depth_pred_left = cv2.resize(depth_pred_left,input_size)

        # Clip output range: current size is the original size
        depth_pred_left = depth_pred_left.clip(0, 1)


        return depth_pred_left
        
    
    def __encode_contents_text(self,text_content):
        """
        Encode text embedding for empty prompt
        """
        prompt = text_content
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        # print(text_input_ids.shape)
        
        empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]

        return empty_text_embed


    def _decode_to_image(self,latent):

        depth_latent_zeros = latent
        depth_left = self.decode_depth(depth_latent_zeros)
        depth_left= torch.clip(depth_left, -1.0, 1.0)
        depth_left = (depth_left + 1.0) / 2.0

        return depth_left

    
    
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,
                     text_embed="to right",
                     cond=0):
        
        
        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
                
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. : this is the prompt
        
        # Initial depth map (Guassian noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
        
        current_batch_size = depth_latent.shape[0]
        current_batch_size = current_batch_size//2
 
        prompts = rgb_latent
        conditioning = torch.ones_like(input_rgb).type_as(depth_latent)*1.0
        conditioning = conditioning[:,:1,:224,:]
        
        
        text_embed_to_right ="to right"
        text_embed_to_left = "to left"
        text_prompt_left = self.__encode_contents_text(text_embed_to_right)
        text_prompt_right = self.__encode_contents_text(text_embed_to_left)
        
        text_prompt_left  = text_prompt_left.repeat(current_batch_size,1,1)
        text_prompt_right = text_prompt_right.repeat(current_batch_size,1,1)
        text_prompt = torch.cat((text_prompt_left,text_prompt_right),dim=0) #[2*batch_size,4,1024]


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
            
            # get the controlnet conidtione
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    unet_input,
                    t,
                    encoder_hidden_states=text_prompt,
                    controlnet_cond=conditioning,
                    return_dict=False)
            
        
            # predict the noise residual
            noise_pred = self.unet(unet_input, 
                                  t, 
                                  encoder_hidden_states=text_prompt,
                                  down_block_additional_residuals=[sample.to(dtype=self.dtype) for sample in down_block_res_samples],
                                  mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
                                  return_dict=False)[0]  # [B, 4, h, w]
            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        torch.cuda.empty_cache()
        

        rendered_right_from_left, rendered_left_from_right = torch.chunk(depth_latent,chunks=2,dim=0)
        rendered_right_from_left_images = self._decode_to_image(rendered_right_from_left)
        rendered_left_from_right_images = self._decode_to_image(rendered_left_from_right)
        

        
        rendered_right_from_left_images_list = []
        rendered_left_from_right_images_list = []
        
        for new_id in range(rendered_right_from_left_images.shape[0]):
            rendered_right_from_left_cur = rendered_right_from_left_images[new_id]
            rendered_right_from_left_cur = rendered_right_from_left_cur.squeeze(0).permute(1,2,0).cpu().numpy()
            rendered_right_from_left_images_list.append(rendered_right_from_left_cur)
            
            
            rendered_left_from_right_cur = rendered_left_from_right_images[new_id]
            rendered_left_from_right_cur = rendered_left_from_right_cur.squeeze(0).permute(1,2,0).cpu().numpy()
            rendered_left_from_right_images_list.append(rendered_left_from_right_cur)
        
    
        return rendered_right_from_left_images_list,rendered_left_from_right_images_list
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

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
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        
        return stacked