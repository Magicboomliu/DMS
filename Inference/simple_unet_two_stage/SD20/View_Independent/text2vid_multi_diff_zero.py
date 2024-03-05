import torch
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, StableDiffusionDiffEditPipeline, StableDiffusionPanoramaPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline

from zero_cross_att import CrossFrameAttnProcessor
from diffusers.schedulers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler

import sys
sys.path.append("..")
from PIL import Image
from offline_inference.SDConImg2ImgNoiseInverseMultiDiffusionPipeline import SDConImg2ImgNoiseInversionMultiDiffusionPipeline
from offline_inference.controlnet_tile import ControlNetTileModel

import imageio
import os

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

model_id = "Vhey/a-zovya-photoreal-v2"
controlnet_model_path = "/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/outputs/step2_refine/checkpoint-70000"

guide_images_dir = "/home/zliu/meeting_samples/2011_09_26_drive_0095_sync/"
guided_images = []
max_img_num = 50

for path in sorted(os.listdir(guide_images_dir)):
    img_path = os.path.join(guide_images_dir, path)
    img = Image.open(img_path)
    img = resize_for_condition_image(img)
    # print(img.size)
    # img = img.resize((1728, 832))
    guided_images.append(img)
    if len(guided_images) >= max_img_num:
        break


# quit()


inverse_scheduler_init = DPMSolverMultistepInverseScheduler()

controlnet = ControlNetTileModel.from_pretrained(controlnet_model_path,subfolder="controlnet").half()

pipe = SDConImg2ImgNoiseInversionMultiDiffusionPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, inverse_scheduler=inverse_scheduler_init
).to("cuda")

print(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
print(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

requested_steps = 32
requested_steps_inverse = 10
strength = 0.2

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

# imageio.mimsave("video_catt_guess_euler_multi_diff_32_no_sync.mp4", result, fps=1)
for i, image in enumerate(result):
    imageio.imsave(f"catt_guess_euler_multi_diff_32_sync_{i}.png", image)


