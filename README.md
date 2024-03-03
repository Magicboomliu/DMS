# DiffusionMultiBaseline

Using Stable Diffusion Model for generating multi-baseline images for autonomous driving scenes like KITTI.

## Model Zoos  
- [Simple Unet Fine Tuned-SD15]()
- [Simple Unet Fine-Tuned-SD20]()

## Ablation Studies
(1) Simple UNet Fine-Tuning  
Descriptions: Simply Fine-Tuned the Unet with prompt of baseline values equals to -0.54, 0, 0.54. Simple concate the reference latent feature with baseline values. Options with Stable Diffusion1.5 and Stable DIffusion2.0.  
Training Code:  
```
cd scripts\
sh train_simple_unet.sh
```