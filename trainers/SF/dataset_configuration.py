import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("../..")

from dataloader.sceneflow_dataloader.SceneFlow_loader import SceneFlow_Dataset
from dataloader.sceneflow_dataloader import transforms

import os
import logging


# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    logger=None,
                    batch_size = 1,
                    datathread = 4,
                    targetHW =(576,960),
                    visible_list=['left','right','disp','occlusion']
                    ):
    
    train_transform_list = [transforms.ToTensor()]
    train_transform = transforms.Compose(train_transform_list)
    
    train_dataset = SceneFlow_Dataset(datapath=datapath,
                                      trainlist=trainlist,
                                      transform=train_transform,
                                      targetHW=targetHW,
                                      visible_list=visible_list)

    datathread=datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    # logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    num_batches_per_epoch = len(train_loader)    
    return train_loader,num_batches_per_epoch






        
