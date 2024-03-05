import torch
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, TextToVideoSDPipeline, StableDiffusionDiffEditPipeline, StableDiffusionPanoramaPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
# from zero_cross_att import CrossFrameAttnProcessor
from zero_cross_att_prev_ref_prev3 import CrossFrameAttnProcessor
from diffusers.schedulers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler

import sys
sys.path.append("..")


from Inference.simple_unet_two_stage.SD20.View_Independent.SDConImg2ImgNoiseInverseMultiDiffusionPipeline import SDConImg2ImgNoiseInversionMultiDiffusionPipeline
from Inference.simple_unet_two_stage.SD20.View_Independent.controlnet_tile import ControlNetTileModel

from PIL import Image

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
# model_id = 'runwayml/stable-diffusion-v1-5'
controlnet_model_path = "/home/yli/NeRF_reimp/img2img_diff/outputs/sannomiya_4dirs_guided_cfg_x2_img2img_controlnet_direct_fine_tune_sd15_cfg5rate1_512/checkpoint/"
# controlnet_model_path = "lllyasviel/control_v11f1e_sd15_tile"


guide_images_dir = "/home/yli/NeRF_reimp/img2img_diff/Datasets/sannomiya_iphone_4dirs_0.25/val_100"
guided_images = []
max_img_num = 100

for path in sorted(os.listdir(guide_images_dir))[35:60]:
    img_path = os.path.join(guide_images_dir, path)
    img = Image.open(img_path)
    img = img.resize((1728, 832))
    # img = img.crop((128, 0, 128+512, 512))
    # img = resize_for_condition_image(img)
    guided_images.append(img)
    if len(guided_images) >= max_img_num:
        break

guided_images = guided_images

inverse_scheduler_init = DPMSolverMultistepInverseScheduler()

controlnet = ControlNetTileModel.from_pretrained(controlnet_model_path,subfolder="controlnet").half()

# pipe = SDConImg2ImgNoiseInversionPipeline.from_pretrained(
#     model_id, controlnet=controlnet, torch_dtype=torch.float16, inverse_scheduler=inverse_scheduler_init
# ).to("cuda")

pipe = SDConImg2ImgNoiseInversionMultiDiffusionPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, inverse_scheduler=inverse_scheduler_init
).to("cuda")

print(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
print(pipe.scheduler.config)

pipe.enable_vae_slicing()

pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()


requested_steps = 32
requested_steps_inverse = 10
strength = 0.8

prev_cons_frame = 2
aft_cons_frame = 2

prompt = ["best quality"] * len(guided_images)
negative_prompt = ["blur, lowres, bad anatomy, bad hands, cropped, worst quality"] * len(guided_images)

generator = torch.manual_seed(12345)
view_batch_size = 1

use_cross_view_att = True

sub_batch_bmm = 1

if use_cross_view_att:
    # pipe.unet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
    # pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
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

imageio.mimsave("video_catt_guess_euler_multi_diff_32_sync5_align_prev_kv12-1-2avg_32steps_2_0.8.mp4", result, fps=5)
# for i, image in enumerate(result):
#     imageio.imsave(f"catt_guess_euler_multi_diff_32_sync5_aftprev_{i}.png", image)


