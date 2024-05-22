# DMB: Improving Self-supervised Stereo Matching using Diffusion-based Multi-Baseline Generation.

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

```

- Inference on the KITTI Raw dataset 

```

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




