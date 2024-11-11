from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img

if __name__=="__main__":
    
    training_filename_path = "/home/zliu/CVPR2025/DiffusionMultiBaseline/datafiles/CARLA/training.txt"
    
    testing_filename_path = "/home/zliu/CVPR2025/DiffusionMultiBaseline/datafiles/CARLA/testing.txt"
    
    
    train_contents = read_text_lines(training_filename_path)
    test_contents = read_text_lines(testing_filename_path)
    
    datapath = "/data3/CALRA/CARLA/"
    
    for sample in tqdm(train_contents):

        left_filename = sample
        
        full_left_filename = os.path.join(datapath,left_filename)
        full_right_filename = full_left_filename.replace("Left_RGB","Right_RGB")
        full_mid_filename = full_left_filename.replace("Left_RGB","Middle_RGB")
        
        assert os.path.exists(full_mid_filename)
        assert os.path.exists(full_right_filename)
        assert os.path.exists(full_mid_filename)
        
        sample = dict()

        sample['left_image'] = read_img(full_left_filename) #(1080,1920)
        sample['right_image'] = read_img(full_right_filename)
        sample['mid_image'] = read_img(full_mid_filename)
        
            
        

        
    print("OK")
    
    
    
    pass