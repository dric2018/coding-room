import os


class Config:
    data_dir = os.path.abspath('../data/input/')
    models_dir = os.path.abspath('../models')
    logs_dir = os.path.abspath('../logs')
    train_data_dir = os.path.abspath('../data/input/train_images')
    test_data_dir = os.path.abspath('../data/input/test_images')
    num_epochs = 10
    lr = 2e-3
    weight_decay = .01
    eps = 1e-8
    train_batch_size = 16
    test_batch_size = 4
    base_model = 'resnet34'
