# DiffusionMultiBaseline

Using Stable Diffusion Model for generating multi-baseline images for autonomous driving scenes like KITTI.

## Model Zoos  
- [Simple Unet Fine Tuned-SD15](https://drive.google.com/file/d/1aaKZqYuAZyhNfFENAirgsRCN82A4Ji1r/view?usp=sharing)
- [Simple Unet Fine-Tuned-SD20](https://drive.google.com/file/d/1ule3EFFqmcdPxtCaiCAkg_amKhFwqm6I/view?usp=sharing)
- [Two-Stage Simple UNet-SD20](https://drive.google.com/file/d/1ibTv9M3hConJOaPSiplIsI1fUwVUC5G8/view?usp=sharing)

## Ablation Studies
**(1) Simple UNet Fine-Tuning**  
Descriptions: Simply Fine-Tuned the Unet with prompt of baseline values equals to -0.54, 0, 0.54. Simple concate the reference latent feature with baseline values. Options with Stable Diffusion1.5 and Stable DIffusion2.0.  
Training Code:  
```
cd scripts/train/
sh train_simple_unet.sh
```  
Inference Code:  
```
cd scripts/inference/
sh infer_simple_unet.sh
```  

Evaluation Code:
```
cd scripts/evaluations/
sh eval_simple_unet.sh

# For the PSNR and SSIM
cd scripts/evaluations/
sh get_psnr_and_ssim
```

**(2) Two-Stage: Simple UNet Fine-Tuning + Image Enhancement**  
Descriptions: Adopt a two-stage pipeline: First Using the (1) to train a Stable Diffusion Model for new view synthesis, then the second stage is to using a image enhancement method to recover the high quality images.  
Training Code:  

Inference Code:  
```
cd scripts/inference/
sh infer_two_stage_unet.sh 
```
Evaluation Code:
```
cd scripts/evaluations/
sh eval_simple_unet_two_stage.sh

```
