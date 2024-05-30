from colormap import kitti_colormap, read_16bit_gt
import cv2
import os
from tqdm import tqdm
from PIL import Image


def depth2color(depth_img, save_dir):
    depth_img = cv2.imread(depth_img)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=2), cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    depth_color = Image.fromarray(depth_color)
    depth_color.save(save_dir)

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