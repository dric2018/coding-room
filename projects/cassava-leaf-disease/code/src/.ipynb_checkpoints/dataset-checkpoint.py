# IMPORTS
import pytorch_lightning as pl
import torchvision
import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np

from PIL import Image

from .config import Config

# CONSTANTS AND VARIABLES


class LeafDataset(Dataset):
    def __init__(self, data_dir:str, df=pd.DataFrame, transform=None):
        super(LeafDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.df = df
        
        
    def __len__(self):
        return self.df
    
    def __getitem__(self, index):
        # get image name or image id from the given dataframe
        img_id = self.df.iloc[index].image_id
        
        # load image as array and make it tensor
        img_array = Image.open(os.path.join(self.data_dir))
        
        sample = {
            'img': ,
        }
        
        if self.transform is not None:
            sample.update({
            'targets':
            })
        return sample
    
    
    
class DataModule(pl.LightningDataModule):

    def __init__(self, config: Config):
        super(DataModule, self).__init__()
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.data_dir = config.data_dir

    def setup(self, stage=None):
        self.train_ds = 

        self.val_ds = MNIST(self.data_dir,
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                            ]))

        print(f'[INFO] Training on {len(self.train_ds)}')
        print(f'[INFO] Validating on {len(self.val_ds)}')

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.train_batch_size,
                          num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.test_batch_size,
                          num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=self.test_batch_size,
                          num_workers=os.cpu_count())
