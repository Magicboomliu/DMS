# DMB: Improving Self-supervised Stereo Matching using Diffusion-Based Multi-Baseline Generation.

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

## Diffusion-based Multi-baseline Generation
### Pretrained Models(Google Drive)
- [SceneFlow-DMB](https://drive.google.com/drive/folders/1Yc2RNc8TdwPe84T5cEiYbG8QKAt1p7j-?usp=sharing)
- [KITTIRaw-DMB](https://drive.google.com/drive/folders/1p1vhvANOeYjGkSfc53O-EEbgKCfc3cN7?usp=sharing)
- [KITTI2012-DMB](https://drive.google.com/drive/folders/1wFA1QNnQie_hjf-HUnjqJhF0JrCLqBn9?usp=sharing)
- [KITTI2015-DMB](https://drive.google.com/drive/folders/1yw_Bcy-cLSenJtNh68Jh5HlW0kaz1ola?usp=sharing)
- [MPI-Sintel-DMB](https://drive.google.com/drive/folders/1ewx0RNsJSjf4NXt8d9Zh9Lnv660zZPOz?usp=sharing)

### Training of the DMB diffusion Model
- Training on the SceneFlow dataset 
```
cd scripts/SF/train 
sh train_unet.sh
``` 

- Training on the KITTI Raw dataset 
```
cd scripts/SF/train 
sh train_unet.sh
``` 

- Training on the MPI-Sintel dataset 
```
cd scripts/SF/train 
sh train_unet.sh
``` 

### Inference Multi-Baseline Images
- Inference on the SceneFlow dataset
```
#left to right, right to left inference
cd scripts/SF/evaluation
sh evaluation.sh

# get left-left and right-right
cd scripts/SF/evaluation
sh get_additional_view.sh

# get the med-state views
cd scripts/SF/evaluation
sh get_middle_view.sh

```

- Inference on the KITTI Raw dataset 

```
#left to right, right to left inference
cd scripts/KITTI/kitti_raw_evaluations
sh eval_unet.sh

# get left-left and right-right
cd scripts/KITTI/kitti_raw_evaluations
sh unet_generated_new_view.sh

# get the med-state views
cd scripts/KITTI/kitti_raw_evaluations
sh sh unet_generate_med_view.sh


```

- Inference on the KITTI 2015 dataset 
```

```

- Inference on the KITTI 2012 dataset
```

```
- Inference on MPI-Sintel dataset 
```
```


