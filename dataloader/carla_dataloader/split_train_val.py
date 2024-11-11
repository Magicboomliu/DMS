import os
import numpy as np



if __name__ =="__main__":
    
    root_folder = "/data3/CARLA_V2/"
    output_filename_list = "/home/zliu/CVPR2025/DiffusionMultiBaseline/datafiles/CARLA15"
    os.makedirs(output_filename_list,exist_ok=True)
    
    
    seq_names = os.listdir(root_folder)
    
    global_filename_list = []
    train_filename_list = []
    validation_filename_list = []
    
    train_val_split_ratio = 10
    
    fname_counter = 0
    
    for idx, seq in enumerate(seq_names):
        seq_abs_rgb_folder = os.path.join(root_folder,seq,"Left_RGB/")
        inside_rgb_folder = os.path.join(seq,"Left_RGB/")
        
        for fname in os.listdir(seq_abs_rgb_folder):
            left_image_name = os.path.join(inside_rgb_folder,fname)
            
            # add to the val split
            if fname_counter%train_val_split_ratio==0:
                validation_filename_list.append(left_image_name)
            # add the train val split
            else:
                train_filename_list.append(left_image_name)
            
            global_filename_list.append(left_image_name)
            
            fname_counter = fname_counter + 1
            
    
    # training txt
    with open(os.path.join(output_filename_list,'training.txt'),'w') as f:
        for idx, line in enumerate(train_filename_list):
            if idx!=len(train_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
                
    # testing txt
    with open(os.path.join(output_filename_list,'testing.txt'),'w') as f:
        for idx, line in enumerate(validation_filename_list):
            if idx!=len(validation_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
    # global txt
    with open(os.path.join(output_filename_list,'All.txt'),'w') as f:
        for idx, line in enumerate(global_filename_list):
            if idx!=len(global_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
        
        


