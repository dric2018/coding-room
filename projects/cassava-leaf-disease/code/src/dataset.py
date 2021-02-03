# IMPORTS
import pytorch_lightning as pl
import torchvision
import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

from .config import Config

# CONSTANTS AND VARIABLES

class LeafDataset(Dataset):
    def __init__(self, data_dir:str, df:pd.DataFrame, transform=None, task='train'):
        super(LeafDataset, self).__init__()
        self.data_dir = data_dir
        self.imgs_list = os.listdir(self.data_dir)
        self.transform = transform
        self.df = df
        self.task = task
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        try:
            # get image name or image id from the given dataframe
            img_id = self.df.iloc[index].image_id
        except:
            # get image from data dir
            img_id = self.imgs_list[index]
            
            
        # load image as array and make it tensor
        img_array = Image.open(os.path.join(self.data_dir, img_id))
        
        # swap image shape in order to have channels first
        # normalizing image pixel values to be btw [0, 1]
        img_array = np.array(img_array).transpose(2, 0, 1) / 255.
        
        # convert array to float tensor
        img_t = th.from_numpy(img_array).float()
        
        if self.transform is not None:
            img_t = self.transform(img_t)      
        
        sample = {
            'img': img_t,
        }
                
        if self.task == 'train':
            # add targets to sample
            target = self.df.iloc[index].label

            sample.update({
            'targets': th.tensor(target, dtype=th.long)
            })
            
        
            
            
        return sample
    
    
class DataModule(pl.LightningDataModule):

    def __init__(self, config: Config, 
                 train_data_dir:str, 
                 test_data_dir:str, 
                 train_df:pd.DataFrame,
                 data_transform:dict,
                 validation_split=.1,
                test_df:pd.DataFrame = None,
                 train_frac = 1):

        super(DataModule, self).__init__()
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.train_data_dir = config.train_data_dir
        self.test_data_dir = config.test_data_dir
        self.train_df = train_df 
        self.test_df = test_df 
        self.data_transform = data_transform
        self.validation_split = validation_split
        self.train_frac = train_frac
        self.num_workers = config.num_workers
        
    def setup(self, stage=None):
        
        if self.train_frac > 0 and self.train_frac < 1 :
            self.train_df = self.train_df.sample(frac=self.train_frac).reset_index(drop=True)
            train, val = train_test_split(self.train_df, 
                                          test_size=self.validation_split, 
                                          random_state=Config.seed_val)
        else:
            train, val = train_test_split(self.train_df, 
                                          test_size=self.validation_split, 
                                          random_state=Config.seed_val)
            
        self.train_ds = LeafDataset(data_dir=self.train_data_dir,
                                    df=train, 
                                    transform=None,#self.data_transform['train'], 
                                    task='train')

        self.val_ds = LeafDataset(data_dir=self.train_data_dir, 
                                    df=val, 
                                    transform=None,#self.data_transform['validation'], 
                                    task='train')

        print(f'[INFO] Training on {len(self.train_ds)}')
        print(f'[INFO] Validating on {len(self.val_ds)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers
                         )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers
                         )