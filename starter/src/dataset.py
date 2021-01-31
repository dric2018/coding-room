# IMPORTS
import pytorch_lightning as pl
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from .config import Config

# CONSTANTS AND VARIABLES


class DataModule(pl.LightningDataModule):

    def __init__(self, config: Config):
        super(DataModule, self).__init__()
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.data_dir = config.data_dir

    def setup(self, stage=None):
        self.train_ds = MNIST(self.data_dir,
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                              ]))

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
