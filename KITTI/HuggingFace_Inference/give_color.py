from colormap import kitti_colormap, read_16bit_gt
import cv2
import os
from tqdm import tqdm

if __name__=="__main__":
    
    source_folder = "fusion/bw"
    target_folder = "fusion/color"
    os.makedirs(target_folder,exist_ok=True)
    
    for fname in tqdm(os.listdir(source_folder)):
        image_path = os.path.join(source_folder,fname)
        saved_path = os.path.join(target_folder,fname)

        src = read_16bit_gt(image_path)
        colored = kitti_colormap(src)
        cv2.imwrite(saved_path, colored)