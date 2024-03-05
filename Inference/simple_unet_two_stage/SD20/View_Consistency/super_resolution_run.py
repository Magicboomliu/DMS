import torch
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, StableDiffusionDiffEditPipeline, StableDiffusionPanoramaPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers.schedulers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler
import sys
sys.path.append("../../../..")
from PIL import Image
from Inference.simple_unet_two_stage.SD20.View_Independent.SDConImg2ImgNoiseInverseMultiDiffusionPipeline import SDConImg2ImgNoiseInversionMultiDiffusionPipeline
from Inference.simple_unet_two_stage.SD20.View_Independent.controlnet_tile import ControlNetTileModel

import imageio
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from zero_cross_att_prev_ref_prev3  import CrossFrameAttnProcessor


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
        "--input_image_path",
        type=str,
        default="images",
        help="KITTI Folder Path ",
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
    '''Doing the Inference at Here'''
    guided_images = []
    input_left = Image.open(args.input_image_path)
    original_size = input_left.size
    
    img_left_left = resize_for_condition_image(input_left)
    guided_images.append(img_left_left)
    guided_images.append(img_left_left)
    guided_images.append(img_left_left)
    
    
    prompt = ["best quality"] * len(guided_images)
    negative_prompt = ["blur, lowres, bad anatomy, bad hands, cropped, worst quality"] * len(guided_images)

    generator = torch.manual_seed(12345)
    view_batch_size = 2
    use_cross_view_att = True
    sub_batch_bmm = 1


    prev_cons_frame = 2
    aft_cons_frame = 2

    if use_cross_view_att:
        pipe.unet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), sub_batch_bmm=sub_batch_bmm, prev_cons_frame=prev_cons_frame, aft_cons_frame=aft_cons_frame))
        pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), sub_batch_bmm=sub_batch_bmm, prev_cons_frame=prev_cons_frame, aft_cons_frame=aft_cons_frame))
    
    else:
        view_batch_size *= 4

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
    left_data = result[0]
    resized_left_data= left_data.resize(original_size)
    
    
    plt.figure(figsize=(10,15))
    plt.subplot(2,1,1)
    plt.axis("off")
    plt.title("before enhancement")
    plt.imshow(np.array(input_left))
    
    plt.subplot(2,1,2)
    plt.axis("off")
    plt.title("after enhancement")
    plt.imshow(np.array(resized_left_data))
    plt.show()




if __name__=="__main__":
    main()



