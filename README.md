# DiffusionMultiBaseline

Using Stable Diffusion Model for generating multi-baseline images for autonomous driving scenes like KITTI.

## Model Zoos  
- [Simple Unet Fine Tuned-SD15](https://drive.google.com/file/d/1aaKZqYuAZyhNfFENAirgsRCN82A4Ji1r/view?usp=sharing)
- [Simple Unet Fine-Tuned-SD20](https://drive.google.com/file/d/1ule3EFFqmcdPxtCaiCAkg_amKhFwqm6I/view?usp=sharing)
- [Two-Stage Simple UNet-SD20](https://drive.google.com/file/d/1ibTv9M3hConJOaPSiplIsI1fUwVUC5G8/view?usp=sharing)
- [Simple Controlnet Fine Tuned-SD20](https://drive.google.com/file/d/1HKZE3LusLDmaVSjt-sTC4K9u_O0AscI2/view?usp=sharing)
## Download the dataset 
- [Test Images Examples](https://drive.google.com/drive/folders/14dC6rc818MIYNHQrALtZ6pEEChbtr6Qj?usp=sharing)

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
**(3) Simple Controlnet FineTuning** 
Descriptions: (1)First using a Unet with the target view and the reference view concat together as the input. Using the language prompts "to left" and "to right" to show the direction. By using this manner, we can get a Stable Diffusion Model( it can enable left to right, or right to left).  
(2)Then we train a controlnet using the initial unet in step1, the controlnet condition is '0' or '1', '0' means the current view, '1' means go and align with the language prompts("to left" or "to right").  

Training Code: 
```
# Step1 : Train the Unet
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/scripts/train
sh train_unet_for_controlnet.sh 

$ Step2: Train the Controlnet
cd /home/zliu/ACMMM2024/DiffusionMultiBaseline/scripts/train
sh train_simple_controlnet.sh 
```

