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
        
        # Adjust dimensions to be multiples of 64
        # new_width = (new_width // 64) * 64
        # new_height = (new_height // 64) * 64

        # Check if either dimension is less than max_edge_resolution, adjust if necessary
        if new_width < max_edge_resolution and new_height < max_edge_resolution:
            if new_width > new_height:
                new_width = max_edge_resolution
            else:
                new_height = max_edge_resolution

        resized_img = img.resize((new_width, new_height))
        return resized_img
    
    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
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
        input_size = input_image.size
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denosing_steps >=1

        
        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res >0:
            input_image = self._resize_max_res(
                input_image, max_edge_resolution=processing_res
            ) # resize image: for kitti is 231, 768
        
        # print("--------------------------")
        # print(input_image.size)
        # print("------------------------------")
        # quit()
        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        

        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        
        rgb_norm = rgb_norm.half()
        
        
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0
        rgb_norm = (rgb_norm -0.5) * 2.0
        

        # ----------------- predicting depth -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        
        
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        
        for batch in iterable_bar:
            (batched_image,)= batch  # here the image is still around 0-1
            depth_pred_raw_left  = self.single_infer(
                input_rgb=batched_image,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
                text_embed=text_embed,
                cond=cond,
            )
        
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
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]
        self.empty_text_embed = self.empty_text_embed.half()

        
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
        
    
        depth_latent = depth_latent.half()
        
        prompts = rgb_latent

        if cond ==0:
            conditioning = torch.ones_like(input_rgb).type_as(depth_latent)*0.0
        elif cond==1:
            conditioning = torch.ones_like(input_rgb).type_as(depth_latent)*1.0
        else:
            raise NotImplementedError
        
        conditioning = conditioning.half()
        conditioning = conditioning[:,:1,:224,:]
    
        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_contents_text(text_embed)
            
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
            
            # get the controlnet conidtione
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    unet_input,
                    t,
                    encoder_hidden_states=batch_empty_text_embed,
                    controlnet_cond=conditioning,
                    return_dict=False)
            
        
            # predict the noise residual
            noise_pred = self.unet(unet_input, 
                                  t, 
                                  encoder_hidden_states=batch_empty_text_embed,
                                  down_block_additional_residuals=[sample.to(dtype=self.dtype) for sample in down_block_res_samples],
                                  mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
                                  return_dict=False)[0]  # [B, 4, h, w]


            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        
        torch.cuda.empty_cache()
        
        
        assert depth_latent.shape[0]==1
        # decode to left
        depth_latent_zeros = depth_latent
        depth_left = self.decode_depth(depth_latent_zeros)
        depth_left= torch.clip(depth_left, -1.0, 1.0)
        # shift to [0, 1]
        depth_left = (depth_left + 1.0) / 2.0
        
        return depth_left
        
    
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
        
        depth_latent = depth_latent.half()
        
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        
        return stacked