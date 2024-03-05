import torch
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, StableDiffusionDiffEditPipeline, StableDiffusionPanoramaPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline

from diffusers.schedulers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler
import sys
sys.path.append("../../..")
from PIL import Image
from Inference.simple_unet_two_stage.SD20.View_Independent.SDConImg2ImgNoiseInverseMultiDiffusionPipeline import SDConImg2ImgNoiseInversionMultiDiffusionPipeline
from Inference.simple_unet_two_stage.SD20.View_Independent.controlnet_tile import ControlNetTileModel
from Inference.simple_unet_two_stage.SD20.View_Independent.zero_cross_att import CrossFrameAttnProcessor
import imageio
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import skimage.io


def find_deepest_folders(root_folder):
    deepest_folders = []
    for folder, subfolders, files in os.walk(root_folder):
        if not subfolders:
            deepest_folders.append(os.path.abspath(folder))
    return deepest_folders

def resize_for_condition_image(input_image: Image, resolution: int=512):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def main():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Super Resolution."
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default='None',
        help="pretrained model path from hugging face or local dir",
    )  
    parser.add_argument(
        "--unet_path",
        type=str,
        default="Vhey/a-zovya-photoreal-v2",
        help="pretrained  unet model path from hugging face or local dir",
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default="Vhey/a-zovya-photoreal-v2",
        help="pretrained  unet model path from hugging face or local dir",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="Vhey/a-zovya-photoreal-v2",
        help="pretrained  unet model path from hugging face or local dir",
    )
    

    parser.add_argument(
        "--requested_steps",
        type=int,
        default=32,
        help="pretrained  unet model path from hugging face or local dir",
    )

    parser.add_argument(
        "--requested_steps_inverse",
        type=int,
        default=10,
        help="pretrained  unet model path from hugging face or local dir",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.2,
        help="pretrained  unet model path from hugging face or local dir",
    )
    


    args = parser.parse_args()
    # os.makedirs(args.saved_folders,exist_ok=True)
    # contents = read_text_lines(args.filename_list)
    
    inverse_scheduler_init = DPMSolverMultistepInverseScheduler()
    controlnet = ControlNetTileModel.from_pretrained(args.controlnet_model_path,subfolder="controlnet").half()
    pipe = SDConImg2ImgNoiseInversionMultiDiffusionPipeline.from_pretrained(
        args.unet_path, controlnet=controlnet, torch_dtype=torch.float16, inverse_scheduler=inverse_scheduler_init
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
    pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
    print(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    requested_steps = args.requested_steps
    requested_steps_inverse = args.requested_steps_inverse
    strength = args.strength
    
    
    print("loaded the pretrained model.")
    
    os.makedirs(args.output_folder,exist_ok=True)
    source_folder = args.source_folder
    
    source_folders = find_deepest_folders(source_folder)

    
    for folder in tqdm(source_folders):
        
        if len(os.listdir(folder))==0:
            continue
        
        instance_output_folder = folder.replace(args.source_folder,args.output_folder)
        os.makedirs(instance_output_folder,exist_ok=True)

        '''Doing the Inference at Here'''
        guided_images = []
        saved_names = []

        for file in os.listdir(folder):
            image_path = os.path.join(folder,file)
            input_left = Image.open(image_path)
            original_size = input_left.size
            
            img_left_left = resize_for_condition_image(input_left)
            guided_images.append(img_left_left)
            
            image_path_enhanced = image_path.replace(args.source_folder,args.output_folder)
            saved_names.append(image_path_enhanced)
        
    
            prompt = ["best quality"] * len(guided_images)
            negative_prompt = ["blur, lowres, bad anatomy, bad hands, cropped, worst quality"] * len(guided_images)
            generator = torch.manual_seed(12345)
            view_batch_size = 2
            use_cross_view_att = False
            sub_batch_bmm = None

            if use_cross_view_att:
                pipe.unet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
                pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
            else:
                view_batch_size *= 8

            inv_latents = pipe.invert(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=guided_images,
                        strength=strength,
                        generator=generator,
                        num_inference_steps=int(requested_steps_inverse / min(strength, 0.999)) if strength > 0 else 0,
                        width=guided_images[0].size[0],
                        height=guided_images[0].size[1],
                        guidance_scale=7,
                        view_batch_size=view_batch_size,
                        circular_padding=True
                    )


            result = pipe(prompt=prompt, 
                        negative_prompt=negative_prompt, 
                        # image=guided_images, 
                        image=inv_latents,
                        control_image=guided_images, 
                        width=guided_images[0].size[0],
                        height=guided_images[0].size[1],
                        strength=strength,
                        guidance_scale=7,
                        controlnet_conditioning_scale=1.,
                        generator=generator,
                        num_inference_steps=int(requested_steps / min(strength, 0.999)) if strength > 0 else 0,
                        guess_mode=True,
                        view_batch_size=view_batch_size,
                        circular_padding=True
                        ).images
            
    
            resize_results = [data.resize(original_size) for data in result]

            for idx in range(len(resize_results)):
                saved_name = saved_names[idx]
                imageio.imsave(saved_name,resize_results[idx])




if __name__=="__main__":
    main()



