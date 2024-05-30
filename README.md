# DMB: Improving Self-supervised Stereo Matching using Diffusion-Based Multi-Baseline Generation.



## Dependencies 
```
pip install -r requirements.txt
```  

## Pretrainde Models

- [MonoDepth2](https://drive.google.com/file/d/10P3Xyv396ox_Akj5_s7SN50pxG2EoENA/view?usp=sharing) 


## Training 

```
sh train.sh
```

# Inference
```
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_stereo

```