# DMB: Improving Self-supervised Stereo Matching using Diffusion-Based Multi-Baseline Generation.
## Training of Self-Supervised Stereo Matching Networks

## Dependencies 
```
pip install -r requirements.txt
```  


## Data Preparation  
Please download the SceneFlow,KITTI Raw and KITTI 2015&2012 and MPI-Sintel Dataset 

- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 
- [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) 
- [KITTI 2012 & KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) 
- [MPI-Sintel](http://sintel.is.tue.mpg.de/) 

## Pretrained Models
### (1) Ablation Stuides Pre-Trained Models(Google Drive)
#### Baseline Models is [PASMnet](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM). IEEE TPAMI 2020(Parallax Attention for Unsupervised Stereo Correspondence Learning).  

- [SceneFlow Pretrained Models](https://drive.google.com/drive/folders/1EIit3SgUSAFtTAlmd555CZ0zuZg6ishY?usp=sharing)
- [KITTI2015 Pretrained Models](https://drive.google.com/drive/folders/1lS5kN06nacaGDCGvDLu76uuVGbTUv6id?usp=sharing)
- [KITTI2012 Pretrained Models](https://drive.google.com/drive/folders/1xNiLjIRm0LYGFx-YGehPMSo9V2jA3XNr?usp=sharing)
- [MPI Pretrained Models](https://drive.google.com/drive/folders/1grFP_GOqyAzJqZw3rdm9CkR1avqgMWjN?usp=sharing)  

### (2) Models for KITTI 2015 Submission(Google Drive) 
- [PASMnet](https://drive.google.com/drive/folders/1scRa3TxjeaiOb5HCTdV8g3oy6pz3ENwE?usp=sharing)
- [CFNet](https://drive.google.com/drive/folders/1scRa3TxjeaiOb5HCTdV8g3oy6pz3ENwE?usp=sharing) 
- [StereoNet](https://drive.google.com/drive/folders/1scRa3TxjeaiOb5HCTdV8g3oy6pz3ENwE?usp=sharing)


## Training Scripts

### Ablation Studies Trainig Scripts
- KITTI 2012 & 2015 dataset
```
cd KITTI/scripts/

# Train KITTI2015(Modify for which experiments inside the scripts)  

sh ablations_kitti2015.sh

# Train KITTI2012(Modify for which experiments inside the scripts)  

sh ablations_kitti2012.sh
```
- SceneFlow Dataset
```
cd SF/scripts

sh ablation_train_sf.sh

```


## Inference Scripts


### Ablation Studies Inference Scripts

- KITTI Dataset
```
cd KITTI/scripts/
sh evaluation_kitti2012_2015.sh

```
- SceneFlow Dataset

```
cd SF/scripts
sh evaluation.sh

```

### KITTI 2015 Testing Set Submission  

```
cd KITTI/scripts/
sh submission_kitti2015.sh

```
